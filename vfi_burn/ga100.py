"""Verified GA100 (CMP 170HX 8 GB) fuse-controller protocol and Linux BAR0/PMBus adapter.

Register addresses, commands, and Goldilocks arming/programming sequences are
cross-verified across:
  - Kate Land's GA100VFI writeup (blog.kate.land)
  - sayhellotojungle/cmp170hx-ffcalc (live 17-record CMP 170HX burn)
  - kprinssu/ga100-ff-tools (live 3-card CMP 170HX arm + burn)
  - darkmaniac7/cmp170hx-8gb-vfi-tools (aligned 32-bit BAR0 store + PMBus safety)

Every MMIO access uses AlignedMMIO (single 32-bit memoryview.cast("I") loads/stores,
NEVER struct.pack_into). DEBUGCTRL (0x00820060) is never written.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import mmap
import os
from pathlib import Path
import re
import stat
import struct
import time

from .backend import Health, Host
from .ffrecord import (
    DEVICE_ID_DENIED_BLOCKS,
    DIE_ID_ROWS,
    FF_EMPTY_WORD,
    FF_ROW_MAX,
    FF_ROW_MIN,
    TOTAL_FUSE_ROWS,
    TOTAL_GPC,
    compute_sm_count,
    decode_ff,
)
from .mmio import AlignedMMIO, REG_DEBUGCTRL, REG_FUSECTRL, REG_PLM_F8
from .model import BurnError, PreTriggerRefusal, SKU_CMP170HX_8GB, is_poison

BAR0_MAP_SIZE = 0x01000000  # 16 MiB BAR0 window

# Core fuse-controller registers (verified across all GA100 references).
REG_PMC_BOOT_0 = 0x00000000
REG_GFW_BOOT = 0x00118234
REG_FUSEADDR = 0x00820004
REG_FUSEDATA = 0x00820008        # READ-ONLY data port (CMD_READ output)
REG_FUSEWDATA = 0x0082000C       # WRITE data port (CMD_PROGRAM input)
REG_EN_SW_OVERRIDE = 0x00820040  # write 1 while DSOV=0 to enable shadow writes
REG_IFF_RECORD = 0x00820050      # must be 0 before CMD_SENSE_CTRL
REG_CRC_STATUS = 0x00820070      # 0x55 = healthy CRC
REG_EFP_STATUS = 0x00820078      # OPT_ENABLE_FUSE_PROGRAM (0 -> 1 after SENSE_CTRL)
REG_DFP_STATUS = 0x0082007C      # OPT_DISABLE_FUSE_PROGRAM (must be 0)
REG_DSOV_STATUS = 0x00820084     # OPT_DISABLE_SW_OVERRIDE (0 in Goldilocks window)
REG_PLM_F4 = 0x008200F4          # FUSEWDATA PLM (opens to 0xFFFFFFFF via WDATA_WR_SECURE=0)
REG_PLM_FC = 0x008200FC          # FUSEOPT shadow PLM (0xFFFFFFFF in Goldilocks)
REG_WDATA_WR_SECURE = 0x0082048C # write 0 -> PLM_F4 auto-opens to 0xFFFFFFFF
REG_CTRL_WR_SECURE = 0x008205B4  # write 0 -> PLM_F8 opens to 0xFFFFFFFF
REG_RESENSE_DISABLE = 0x00821058 # 0 before SENSE_CTRL; becomes 1 after (1 sense/window)
REG_SBFE_SHADOW = 0x00821108     # OPT_SECURED_BURN_FEATURE_ENABLE (write 0 after SENSE_CTRL)

# FUSECTRL commands & state field.
CMD_IDLE = 0x0
CMD_READ = 0x1
CMD_PROGRAM = 0x2                # transient state 3 (0xE00C0000) -> IDLE (4) in ~1 ms
CMD_SENSE_CTRL = 0x3             # restores EFP=1 while keeping DSOV=0
FUSECTRL_STATE_SHIFT = 16
FUSECTRL_STATE_MASK = 0x1F
FUSECTRL_IDLE_STATE = 4
FUSECTRL_CMD_MASK = 0x1F
DEBUGCTRL_RWL_BIT = 1 << 1       # 0 = main array; 1 = RIR macro (NEVER set)


@dataclass(frozen=True)
class ExperimentalRegister:
    """Read-only experimental GA100 register for cross-variant inspection (never burned)."""

    name: str
    address: int
    category: str
    cmp170hx_ref: str
    a100_ref: str
    description: str
    aliases: tuple[str, ...] = ()


# Experimental read-only inspection catalog (from GA100 cross-variant MMIO reference table).
# These registers are NOT mapped in FF burn plans and are strictly read-only in vfi_burn.
EXPERIMENTAL_INSPECT_REGS: tuple[ExperimentalRegister, ...] = (
    # --- ECC & feature control OTP fuses ---
    ExperimentalRegister(
        "FUSE_ECC_EN",
        0x00820228,
        "ECC & Feature Fuses",
        "0x0",
        "0x1",
        "OPT_ECC_EN; ECC enable OTP fuse (bit 0; 0=CMP/consumer, 1=A100/Pro)",
        aliases=("OPT_ECC_EN",),
    ),
    ExperimentalRegister(
        "FUSE_FEAT_OVR_DIS",
        0x008203F0,
        "ECC & Feature Fuses",
        "0x0",
        "0x0",
        "OPT_FEATURE_FUSES_OVERRIDE_DISABLE; 0=HS feature overrides available",
        aliases=("OPT_FEATURE_FUSES_OVERRIDE_DISABLE",),
    ),
    ExperimentalRegister(
        "FUSE_HALF_FBPA_EN",
        0x0082049C,
        "ECC & Feature Fuses",
        "0x0",
        "0x0",
        "OPT_HALF_FBPA_ENABLE; per-FBPA half-capacity fuse mask (24-bit)",
        aliases=("OPT_HALF_FBPA_ENABLE",),
    ),
    ExperimentalRegister(
        "FUSE_FB_CONFIG",
        0x00820328,
        "ECC & Feature Fuses",
        "0x0",
        "0x0",
        "OPT_FB_CONFIG; 4-bit memory topology selector",
        aliases=("OPT_FB_CONFIG",),
    ),
    ExperimentalRegister(
        "FUSE_SKU_ID",
        0x00821060,
        "ECC & Feature Fuses",
        "0x68",
        "0x54/0x5b/0x7b",
        "OPT_SKU_ID; 8-bit SKU identifier (170HX 10G=0x68, A100=0x54/0x5b/0x7b)",
        aliases=("OPT_SKU_ID",),
    ),
    # --- SM speed select OTP fuses (FMA / Tensor Core throttle) ---
    ExperimentalRegister(
        "FUSE_SS_DP",
        0x00820224,
        "SM Speed Select OTP Fuses",
        "0x1",
        "0x0",
        "OPT_SM_SPEED_SELECT_DP; FP64 throttle (1-bit: 0=full, 1=reduced)",
        aliases=("OPT_SM_SPEED_SELECT_DP",),
    ),
    ExperimentalRegister(
        "FUSE_SS_FFMA",
        0x0082059C,
        "SM Speed Select OTP Fuses",
        "0x5",
        "0x0",
        "OPT_SM_SPEED_SELECT_FFMA; FP32 FMA throttle (3-bit: 0=full..5=max)",
        aliases=("OPT_SM_SPEED_SELECT_FFMA",),
    ),
    ExperimentalRegister(
        "FUSE_SS_FMLA16",
        0x008207D4,
        "SM Speed Select OTP Fuses",
        "0x5",
        "0x0",
        "OPT_SM_SPEED_SELECT_FMLA16; FP16 MLA/Tensor throttle (3-bit)",
        aliases=("OPT_SM_SPEED_SELECT_FMLA16",),
    ),
    ExperimentalRegister(
        "FUSE_SS_FMLA32",
        0x008207D8,
        "SM Speed Select OTP Fuses",
        "0x5",
        "0x0",
        "OPT_SM_SPEED_SELECT_FMLA32; FP32 MLA/Tensor throttle (3-bit)",
        aliases=("OPT_SM_SPEED_SELECT_FMLA32",),
    ),
    ExperimentalRegister(
        "FUSE_SS_IMLA0",
        0x008207DC,
        "SM Speed Select OTP Fuses",
        "0x5",
        "0x0",
        "OPT_SM_SPEED_SELECT_IMLA0; INT MLA0 / DP4A throttle (3-bit)",
        aliases=("OPT_SM_SPEED_SELECT_IMLA0",),
    ),
    ExperimentalRegister(
        "FUSE_SS_IMLA1",
        0x008207E0,
        "SM Speed Select OTP Fuses",
        "0x5",
        "0x0",
        "OPT_SM_SPEED_SELECT_IMLA1; INT MLA1 throttle (3-bit)",
        aliases=("OPT_SM_SPEED_SELECT_IMLA1",),
    ),
    ExperimentalRegister(
        "FUSE_SS_IMLA2",
        0x008207E4,
        "SM Speed Select OTP Fuses",
        "0x5",
        "0x0",
        "OPT_SM_SPEED_SELECT_IMLA2; INT MLA2 throttle (3-bit)",
        aliases=("OPT_SM_SPEED_SELECT_IMLA2",),
    ),
    ExperimentalRegister(
        "FUSE_SS_IMLA3",
        0x008207E8,
        "SM Speed Select OTP Fuses",
        "0x5",
        "0x0",
        "OPT_SM_SPEED_SELECT_IMLA3; INT MLA3 throttle (3-bit)",
        aliases=("OPT_SM_SPEED_SELECT_IMLA3",),
    ),
    ExperimentalRegister(
        "FUSE_SS_IMLA4",
        0x008207EC,
        "SM Speed Select OTP Fuses",
        "0x5",
        "0x0",
        "OPT_SM_SPEED_SELECT_IMLA4; INT MLA4 throttle (3-bit)",
        aliases=("OPT_SM_SPEED_SELECT_IMLA4",),
    ),
    # --- Feature override & readout chain (L3/HS-gated runtime state) ---
    ExperimentalRegister(
        "FEAT_OVR_ECC_PLM",
        0x00823800,
        "Feature Override & Readout Chain",
        "0xffffff8f",
        "0xffffff8f",
        "FEATURE_OVERRIDE_ECC PLM (0xffffff8f = L3/HS writable)",
    ),
    ExperimentalRegister(
        "FEAT_OVR_PLM",
        0x00823804,
        "Feature Override & Readout Chain",
        "0xffffff8f",
        "0xffffff8f",
        "FEATURE_OVERRIDE PLM (self-sealing, L3/HS writable)",
    ),
    ExperimentalRegister(
        "FEAT_OVR_ECC",
        0x0082380C,
        "Feature Override & Readout Chain",
        "0x00888888",
        "0x00001111/0x00110111",
        "Per-unit ECC override (SM_LRF/L1/LTC/DRAM/CBU)",
    ),
    ExperimentalRegister(
        "FEAT_OVR_ECC_1",
        0x00823810,
        "Feature Override & Readout Chain",
        "0x002aaaaa",
        "0x00104015/0x00104104",
        "Per-unit ECC override 1 (icache/FECS/GPCCS/PMU/HUBMMU)",
    ),
    ExperimentalRegister(
        "FEAT_OVR_ECC_2",
        0x0082382C,
        "Feature Override & Readout Chain",
        "0x0000000a",
        "0x00000000/0x00000001",
        "Per-unit ECC override 2 (LTC_CBC/SM_URF)",
    ),
    ExperimentalRegister(
        "FEAT_READOUT_0",
        0x00823814,
        "Feature Override & Readout Chain",
        "0x00000233",
        "0xef8ff100",
        "Effective Quadro[5:0] + ECC[31:12] status readout (RO)",
    ),
    ExperimentalRegister(
        "FEAT_READOUT_1",
        0x00823818,
        "Feature Override & Readout Chain",
        "0x016db6ed",
        "0x00000000",
        "Effective SM speed-select throttle readout across all 9 units (RO)",
    ),
    ExperimentalRegister(
        "FEAT_READOUT_2",
        0x00823828,
        "Feature Override & Readout Chain",
        "0x00000000",
        "0x00000007",
        "Effective row-remapper + ECC LTC_CBC/SM_URF readout (RO)",
    ),
    ExperimentalRegister(
        "FBPA_ECC_CTRL",
        0x009A0470,
        "Feature Override & Readout Chain",
        "0x00000000",
        "0x00000041",
        "Broadcast FBPA DRAM ECC control register",
    ),
)

EXPERIMENTAL_REGS_BY_NAME: dict[str, int] = {}
for _exp in EXPERIMENTAL_INSPECT_REGS:
    EXPERIMENTAL_REGS_BY_NAME[_exp.name] = _exp.address
    for _alias in _exp.aliases:
        EXPERIMENTAL_REGS_BY_NAME[_alias] = _exp.address


def decode_experimental_register(reg: ExperimentalRegister, value: int) -> str:
    """Return a concise human-readable interpretation of a read-only experimental register."""
    if value == 0xFFFFFFFF:
        return "LINK/BAR0 DOWN (0xffffffff)"
    if is_poison(value):
        return f"POISON/UNREACHABLE (0x{value:08x})"
    name = reg.name
    if name == "FUSE_ECC_EN":
        return "ENABLED (A100/Pro)" if (value & 1) else "DISABLED (CMP/consumer)"
    if name == "FUSE_FEAT_OVR_DIS":
        return "LOCKED (overrides disabled)" if (value & 1) else "OPEN (HS override path available)"
    if name == "FUSE_HALF_FBPA_EN":
        return "NONE (full FBPA capacity)" if value == 0 else f"HALF-CAPACITY MASK 0x{value:x}"
    if name == "FUSE_FB_CONFIG":
        return f"topology selector {value & 0xF}"
    if name == "FUSE_SKU_ID":
        known_skus = {
            0x68: "CMP 170HX (10GB ref)",
            0x54: "A100 PCIe 40GB",
            0x5B: "A100 SXM4 40GB",
            0x7B: "A100 PCIe 80GB",
            0x8E: "Drive A100 32GB",
        }
        sku_note = known_skus.get(value & 0xFF, "other/unlisted SKU")
        return f"SKU 0x{value & 0xFF:02x} ({sku_note})"
    if name == "FUSE_SS_DP":
        return "THROTTLED (reduced FP64)" if (value & 1) else "FULL SPEED (unthrottled)"
    if name.startswith("FUSE_SS_"):
        v = value & 0x7
        if v == 0:
            return "FULL SPEED (0)"
        if v == 5:
            return "MAX THROTTLE (5; CMP)"
        return f"THROTTLED (level {v})"
    if name == "FEAT_READOUT_0":
        ecc_bits = (value >> 12) & 0xFFFFF
        quadro_bits = value & 0x3F
        ecc_state = "ECC ACTIVE" if ecc_bits else "ECC OFF"
        return f"{ecc_state} (ECC[31:12]=0x{ecc_bits:05x}, Quadro[5:0]=0x{quadro_bits:02x})"
    if name == "FEAT_READOUT_1":
        return "NO ACTIVE THROTTLE (0x0)" if value == 0 else f"THROTTLE ACTIVE (0x{value:08x})"
    if name == "FEAT_READOUT_2":
        return "ECC/REMAP OFF (0x0)" if value == 0 else f"ACTIVE FLAGS 0x{value:x}"
    if name == "FBPA_ECC_CTRL":
        return "DRAM ECC OFF" if value == 0 else f"DRAM ECC ACTIVE (0x{value:x})"
    if name in ("FEAT_OVR_ECC_PLM", "FEAT_OVR_PLM"):
        return "L3/HS writable" if value == 0xFFFFFF8F else f"PLM 0x{value:08x}"
    return reg.description


# Option / defect registers (fused registers used as ground truth; OEM FW can mask status regs).
OPTION_REGS: dict[str, int] = {
    "GPC_DISABLE": 0x00820350,
    "GPC_DEFECTIVE": 0x008205C4,
    "GPC_STATUS": 0x00820C1C,
    "FBP_DISABLE": 0x00820364,
    "FBP_DEFECTIVE": 0x008205CC,
    "FBP_STATUS": 0x00820D38,
    "FBIO_DISABLE": 0x0082036C,
    "FBIO_LAYOUT": 0x0082036C,
    "FBIO_DEFECTIVE": 0x008205D4,
    "FBPA_DISABLE": 0x00820368,
    "FBPA_LAYOUT": 0x00820368,
    "FBPA_DEFECTIVE": 0x008205D0,
    "ROP_L2_DISABLE": 0x008202C4,
    "ROP_L2_LAYOUT": 0x008202C4,
    "ROP_L2_DEFECTIVE": 0x008205E8,
    "PES_DISABLE": 0x00820690,
    "PES_DEFECTIVE": 0x00820698,
    "NVLINK_DISABLE": 0x00820684,
    "NVLINK_DEFECTIVE": 0x0082068C,
    "SYS_PIPE_DISABLE": 0x008203A4,
    "SYS_PIPE_DEFECTIVE": 0x008203AC,
    "DISABLE_GEN3_SPEED": 0x00820250,
    "OPT_DISABLE_GEN3_SPEED": 0x00820250,
    "MAGIC_D": 0x00820520,
    "OPT_PCIE_MAGIC_BITS_D": 0x00820520,
    "BOOT_GEN23_DIS": 0x0082057C,
    "OPT_PCIE_BOOT_GEN23_DIS": 0x0082057C,
    "BOOT_GEN3_DIS": 0x00820580,
    "OPT_PCIE_BOOT_GEN3_DIS": 0x00820580,
    "PCIE_DEVIDA": 0x008204D8,
    "PCIE_DEVIDB": 0x0082056C,
    **{f"TPC_GPC{i}_DISABLE": 0x00820760 + 4 * i for i in range(TOTAL_GPC)},
    **{f"TPC_GPC{i}_STATUS": 0x00820C38 + 4 * i for i in range(TOTAL_GPC)},
    **EXPERIMENTAL_REGS_BY_NAME,
}


TPC_DEFECTIVE_REGS: tuple[int, ...] = (
    0x008205C8,
    0x00820508,
    0x0082050C,
    0x00820514,
    0x00820564,
    0x008205AC,
    0x00820B84,
    0x00820B80,
)
for _i, _addr in enumerate(TPC_DEFECTIVE_REGS):
    OPTION_REGS[f"TPC_GPC{_i}_DEFECTIVE"] = _addr

# MP2888B PMBus constants (from cmp170hx-8gb-vfi-tools/tools/vfi_common.py).
VRM_I2C_ADDR = 0x25
PMBUS_REG_STATUS_WORD = 0x79
PMBUS_REG_READ_VOUT = 0x8B
PMBUS_REG_CTRL_MODE = 0xE1
E1_PMBUS_HELD = 0xC2FA
HEALTHY_RAIL_MIN_MV = 500.0
PROFILE_REGS = (0xBA, 0xB9, 0x30, 0x31, 0x32, 0x45)
PROFILE_FAST = {
    0xBA: 0x0001,
    0xB9: 0x0001,
    0x30: 0xCE04,
    0x31: 0x0002,
    0x32: 0x0F81,
    0x45: 0x7218,
}

BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")


def normalize_bdf(bdf: str) -> str:
    s = bdf.strip().lower()
    if re.match(r"^[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$", s):
        s = "0000:" + s
    if not BDF_RE.match(s):
        raise BurnError(f"invalid PCI BDF {bdf!r}; expected dddd:bb:dd.f (e.g. 0000:05:00.0)")
    return s


def is_fusectrl_idle(ctrl: int) -> bool:
    return (
        not is_poison(ctrl)
        and ((ctrl >> FUSECTRL_STATE_SHIFT) & FUSECTRL_STATE_MASK) == FUSECTRL_IDLE_STATE
        and (ctrl & FUSECTRL_CMD_MASK) == 0
    )


def parse_pcie_lnkcap2(cfg: bytes) -> int:
    """Walk PCIe capability list in config space (256 B) and read LnkCap2 (cap + 0x2C)."""
    if len(cfg) < 0x40:
        raise BurnError("PCI config space shorter than 64 bytes")
    if struct.unpack_from("<I", cfg, 0)[0] == 0xFFFFFFFF:
        raise BurnError("PCI config space reads 0xFFFFFFFF (link down)")
    ptr = cfg[0x34] & 0xFC
    seen: set[int] = set()
    while ptr and ptr not in seen and ptr + 2 <= len(cfg):
        seen.add(ptr)
        if cfg[ptr] == 0x10:  # PCI Express Capability ID
            if ptr + 0x30 > len(cfg):
                raise BurnError("PCIe capability truncated before LnkCap2")
            return struct.unpack_from("<I", cfg, ptr + 0x2C)[0]
        ptr = cfg[ptr + 1] & 0xFC
    raise BurnError("PCI Express capability (ID 0x10) not found in config space")


class GA100FuseController:
    """Low-level GA100 fuse-controller protocol over AlignedMMIO.

    Works identically over real BAR0 mmap or an in-memory bytearray/mock for unit tests.
    When ``read_only=True``, writes are strictly allowlisted to CMD_READ (and optionally
    CMD_SENSE_CTRL + volatile pre-sense latch); CMD_PROGRAM / FUSEWDATA(nonzero) /
    SBFE_SHADOW are physically blocked inside ``_wr``.
    """

    def __init__(
        self,
        mmio: AlignedMMIO,
        *,
        read_only: bool = True,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.mmio = mmio
        self.read_only = read_only
        self._sleep = sleep
        self._clock = clock

    def rd(self, offset: int) -> int:
        return self.mmio.read32(offset)

    def _wr(self, offset: int, value: int) -> None:
        if offset == REG_DEBUGCTRL:
            raise BurnError("REFUSED write to DEBUGCTRL (0x00820060)")
        if offset == REG_FUSEDATA:
            raise BurnError("REFUSED write to read-only FUSEDATA (0x00820008); use FUSEWDATA")
        if self.read_only:
            allowed_ro = (
                offset == REG_FUSEADDR
                or (offset == REG_FUSECTRL and value in (CMD_IDLE, CMD_READ, CMD_SENSE_CTRL))
                or (offset == REG_FUSEWDATA and value == 0)
                or (offset == REG_IFF_RECORD and value == 0)
                or (offset == REG_EN_SW_OVERRIDE and value in (0, 1))
                or (offset == REG_WDATA_WR_SECURE and value == 0)
            )
            if not allowed_ro:
                raise BurnError(
                    f"read-only controller refused write 0x{offset:08x}=0x{value:08x}"
                )
        self.mmio.write32(offset, value, allow_debugctrl=False)

    def _check_fusectrl_first(self) -> int:
        """Read FUSECTRL first and halt immediately on poison without touching any other register."""
        ctrl = self.rd(REG_FUSECTRL)
        if ctrl == 0xFFFFFFFF:
            raise BurnError("FUSECTRL reads 0xFFFFFFFF (BAR0 dead / memory decode off / link lost)")
        if is_poison(ctrl):
            raise BurnError(
                f"FUSECTRL poisoned ({ctrl:#010x}): STOP all BAR0 reads and cold-boot the card"
            )
        return ctrl

    def _check_bar0_alive(self) -> None:
        self._check_fusectrl_first()
        pmc = self.rd(REG_PMC_BOOT_0)
        if pmc in (0, 0xFFFFFFFF) or is_poison(pmc):
            raise BurnError(f"PMC_BOOT_0 reads {pmc:#010x}: BAR0 inaccessible / GPU off")

    def poll_idle(self, timeout_s: float = 0.2) -> int:
        deadline = self._clock() + timeout_s
        while True:
            ctrl = self._check_fusectrl_first()
            if is_fusectrl_idle(ctrl):
                return ctrl
            if self._clock() >= deadline:
                raise BurnError(
                    f"FUSECTRL did not return to IDLE within {timeout_s:.2f}s (reads {ctrl:#010x})"
                )
            self._sleep(0.001)

    def read_health_registers(self, *, status_word: int = 0, vout_mv: float = 787.5) -> Health:
        """Read health registers in BURN_SPEC order: FUSECTRL FIRST, stopping on poison."""
        ctrl = self._check_fusectrl_first()
        dsov = self.rd(REG_DSOV_STATUS)
        if is_poison(dsov):
            return Health(ctrl, dsov, 0, 0, 0, status_word, vout_mv, 0)
        crc = self.rd(REG_CRC_STATUS)
        plm_fc = self.rd(REG_PLM_FC)
        plm_f8 = self.rd(REG_PLM_F8)
        plm_f4 = self.rd(REG_PLM_F4)
        en_sw = self.rd(REG_EN_SW_OVERRIDE)
        dbg = self.rd(REG_DEBUGCTRL)
        efp = self.rd(REG_EFP_STATUS)
        dfp = self.rd(REG_DFP_STATUS)
        sbfe = self.rd(REG_SBFE_SHADOW)
        wsec = self.rd(REG_WDATA_WR_SECURE)
        resense_dis = self.rd(REG_RESENSE_DISABLE)
        return Health(
            fusectrl=ctrl,
            dsov=dsov,
            crc=crc,
            plm=plm_fc,
            plm_f8=plm_f8,
            status_word=status_word,
            vout_mv=vout_mv,
            en_sw=en_sw,
            debugctrl=dbg,
            efp=efp,
            dfp=dfp,
            plm_f4=plm_f4,
            sbfe=sbfe,
            wdata_wr_secure=wsec,
            resense_disable=resense_dis,
        )

    def read_row(self, row: int, *, timeout_s: float = 0.2) -> int:
        """Read one physical fuse row (0..511) via CMD_READ; restores FUSEADDR."""
        if isinstance(row, bool) or not (0 <= row < TOTAL_FUSE_ROWS):
            raise BurnError(f"fuse row {row!r} out of range 0..{TOTAL_FUSE_ROWS - 1}")
        self._check_bar0_alive()
        self.poll_idle(timeout_s)
        dbg = self.rd(REG_DEBUGCTRL)
        if dbg != 0:
            raise BurnError(f"DEBUGCTRL=0x{dbg:08x} nonzero before CMD_READ row {row}; aborting")
        orig_addr = self.rd(REG_FUSEADDR)
        try:
            self._wr(REG_FUSECTRL, CMD_IDLE)
            self._wr(REG_FUSEADDR, row)
            if self.rd(REG_FUSEADDR) != row:
                raise BurnError(f"FUSEADDR did not latch row {row}")
            self._wr(REG_FUSECTRL, CMD_READ)
            self.poll_idle(timeout_s)
            val = self.rd(REG_FUSEDATA)
            if is_poison(val):
                raise BurnError(f"FUSEDATA poisoned ({val:#010x}) reading row {row}")
            return val
        finally:
            try:
                ctrl = self.rd(REG_FUSECTRL)
                if is_fusectrl_idle(ctrl):
                    self._wr(REG_FUSEADDR, orig_addr)
            except Exception:
                pass

    def sense_ctrl(self) -> None:
        """Issue Goldilocks SENSE_CTRL (FUSECTRL=3) to restore EFP 0->1 while preserving DSOV=0.

        Idempotent if EFP==1 and DSOV==0 already (only one SENSE_CTRL is permitted per
        glitch window because RESENSE_DISABLE becomes 1 afterwards).
        Uses the kprinssu/ga100-ff-tools no-burn safety latch (FUSEWDATA=0 verified before cmd 3).
        """
        self._check_bar0_alive()
        self.poll_idle(0.2)
        dbg = self.rd(REG_DEBUGCTRL)
        if dbg != 0:
            raise BurnError(f"DEBUGCTRL=0x{dbg:08x} nonzero before SENSE_CTRL")
        dsov = self.rd(REG_DSOV_STATUS)
        crc = self.rd(REG_CRC_STATUS)
        efp = self.rd(REG_EFP_STATUS)
        if dsov != 0:
            raise BurnError(f"DSOV={dsov} (want 0): not inside Goldilocks window")
        if crc != 0x55:
            raise BurnError(f"CRC_STATUS=0x{crc:x} (want 0x55) before SENSE_CTRL")
        if efp == 1:
            # Already sensed in this window; never re-issue CMD_SENSE_CTRL (RESENSE_DISABLE=1).
            if (self.rd(REG_PLM_F8) & 1) == 0:
                raise BurnError("PLM_f8 bit 0 closed (frame too deep)")
            return

        if self.rd(REG_RESENSE_DISABLE) != 0:
            raise BurnError(
                "RESENSE_DISABLE=1 while EFP=0: cannot re-sense in this window; cold-boot and re-glitch"
            )
        if (self.rd(REG_PLM_F8) & 1) == 0 or self.rd(REG_PLM_F8) != 0xFFFFFFFF:
            raise BurnError(
                f"PLM_f8=0x{self.rd(REG_PLM_F8):08x} not open before SENSE_CTRL (frame too deep)"
            )

        # Volatile enable + open FUSEWDATA PLM so the FUSEWDATA=0 safety latch can be verified.
        self._wr(REG_EN_SW_OVERRIDE, 1)
        if (self.rd(REG_EN_SW_OVERRIDE) & 1) != 1:
            raise BurnError("EN_SW_OVERRIDE=1 did not stick (DSOV=0 window may be invalid)")
        self._wr(REG_WDATA_WR_SECURE, 0)
        if self.rd(REG_PLM_F4) != 0xFFFFFFFF:
            raise BurnError(
                f"FUSEWDATA_PLM (0x8200F4) stayed 0x{self.rd(REG_PLM_F4):08x} after WDATA_WR_SECURE=0"
            )
        if self.rd(REG_IFF_RECORD) != 0:
            self._wr(REG_IFF_RECORD, 0)
            if self.rd(REG_IFF_RECORD) != 0:
                raise BurnError("IFF_RECORD (0x820050) did not clear to 0 before SENSE_CTRL")

        # No-burn safety latch: OTP is 0->1 only, so FUSEWDATA=0 can never blow any bit.
        self._wr(REG_FUSEWDATA, 0)
        if self.rd(REG_FUSEWDATA) != 0:
            raise BurnError(
                f"FUSEWDATA did not read back 0 (got 0x{self.rd(REG_FUSEWDATA):08x}); aborting SENSE_CTRL"
            )

        self._wr(REG_FUSECTRL, CMD_IDLE)
        self._wr(REG_FUSECTRL, CMD_SENSE_CTRL)
        self._sleep(0.2)
        self.poll_idle(2.0)

        # Verify post-SENSE_CTRL invariants.
        if (self.rd(REG_PLM_F8) & 1) == 0:
            raise BurnError("PLM_f8 bit 0 collapsed after SENSE_CTRL: frame too deep; re-hunt shallower")
        if self.rd(REG_DSOV_STATUS) != 0:
            raise BurnError("WINDOW LOST: DSOV returned to 1 after SENSE_CTRL; cold-boot and re-glitch")
        if self.rd(REG_CRC_STATUS) != 0x55:
            raise BurnError(f"CRC_STATUS=0x{self.rd(REG_CRC_STATUS):x} after SENSE_CTRL; STOP")
        if self.rd(REG_EFP_STATUS) != 1:
            raise BurnError("EFP_STATUS remained 0 after SENSE_CTRL; cannot arm fuse programming")
        if self.rd(REG_DFP_STATUS) != 0:
            raise BurnError(f"DISABLE_FUSE_PROGRAM=0x{self.rd(REG_DFP_STATUS):x} after SENSE_CTRL")
        if self.rd(REG_DEBUGCTRL) != 0:
            raise BurnError(f"DEBUGCTRL=0x{self.rd(REG_DEBUGCTRL):08x} nonzero after SENSE_CTRL")

    def armed_problems(self) -> list[str]:
        """Return [] when all GA100 FF programming prerequisites hold; else list every failure."""
        ctrl = self.rd(REG_FUSECTRL)
        if is_poison(ctrl):
            return [f"FUSECTRL poisoned ({ctrl:#010x}): STOP reads and cold-boot"]
        problems: list[str] = []
        if not is_fusectrl_idle(ctrl):
            problems.append(f"FUSECTRL non-idle (0x{ctrl:08x})")
        dbg = self.rd(REG_DEBUGCTRL)
        if dbg != 0 or (dbg & DEBUGCTRL_RWL_BIT):
            problems.append(f"DEBUGCTRL=0x{dbg:08x} (want 0, RWL=0)")
        crc = self.rd(REG_CRC_STATUS)
        if crc != 0x55:
            problems.append(f"CRC_STATUS=0x{crc:x} (want 0x55)")
        dsov = self.rd(REG_DSOV_STATUS)
        if dsov != 0:
            problems.append(f"DSOV={dsov} (want 0)")
        efp = self.rd(REG_EFP_STATUS)
        if efp != 1:
            problems.append(f"EFP_STATUS={efp} (want 1; run SENSE_CTRL first)")
        dfp = self.rd(REG_DFP_STATUS)
        if dfp != 0:
            problems.append(f"DISABLE_FUSE_PROGRAM=0x{dfp:x} (want 0)")
        en_sw = self.rd(REG_EN_SW_OVERRIDE)
        if (en_sw & 1) != 1:
            problems.append(f"EN_SW_OVERRIDE=0x{en_sw:x} (want 1)")
        sbfe = self.rd(REG_SBFE_SHADOW)
        if sbfe != 0:
            problems.append(f"SBFE_SHADOW=0x{sbfe:x} (want 0)")
        wsec = self.rd(REG_WDATA_WR_SECURE)
        if wsec != 0:
            problems.append(f"WDATA_WR_SECURE=0x{wsec:x} (want 0)")
        plm_f4 = self.rd(REG_PLM_F4)
        if plm_f4 != 0xFFFFFFFF:
            problems.append(f"PLM_f4 (FUSEWDATA)=0x{plm_f4:08x} (want 0xffffffff)")
        plm_f8 = self.rd(REG_PLM_F8)
        if plm_f8 != 0xFFFFFFFF:
            problems.append(f"PLM_f8 (FUSECTRL)=0x{plm_f8:08x} (want 0xffffffff)")
        plm_fc = self.rd(REG_PLM_FC)
        if plm_fc != 0xFFFFFFFF:
            problems.append(f"PLM_fc (FUSEOPT)=0x{plm_fc:08x} (want 0xffffffff)")
        return problems

    def arm(self) -> None:
        """Open volatile shadow gates (EN_SW_OVERRIDE=1, SBFE=0, WDATA_WR_SECURE=0) after SENSE_CTRL."""
        if self.read_only:
            raise BurnError("cannot arm fuse writes on a read-only controller")
        self._check_bar0_alive()
        self.poll_idle(0.2)
        if self.rd(REG_DSOV_STATUS) != 0:
            raise BurnError("cannot arm: DSOV != 0 (outside Goldilocks window)")
        if self.rd(REG_EFP_STATUS) != 1:
            raise BurnError("cannot arm: EFP_STATUS != 1 (run SENSE_CTRL first)")
        if self.rd(REG_DEBUGCTRL) != 0:
            raise BurnError(f"cannot arm: DEBUGCTRL=0x{self.rd(REG_DEBUGCTRL):08x} nonzero")

        self._wr(REG_EN_SW_OVERRIDE, 1)
        if (self.rd(REG_EN_SW_OVERRIDE) & 1) != 1:
            raise BurnError("EN_SW_OVERRIDE=1 did not stick")
        self._wr(REG_SBFE_SHADOW, 0)
        if self.rd(REG_SBFE_SHADOW) != 0:
            raise BurnError(f"SBFE_SHADOW did not clear (reads 0x{self.rd(REG_SBFE_SHADOW):x})")
        self._wr(REG_WDATA_WR_SECURE, 0)
        if self.rd(REG_WDATA_WR_SECURE) != 0:
            raise BurnError("WDATA_WR_SECURE did not clear to 0")
        # BURN_SPEC §3/§6: a closed/partial PLM_f8 means the frame is too deep -> abort.
        # Do NOT re-open it via CTRL_WR_SECURE; that would mask the diagnostic.
        plm_f8 = self.rd(REG_PLM_F8)
        if plm_f8 != 0xFFFFFFFF:
            raise BurnError(
                f"PLM_f8=0x{plm_f8:08x} not fully open at arm time: frame too deep / host access "
                "revoked; refusing to re-open it in software"
            )

        problems = self.armed_problems()
        if problems:
            raise BurnError("controller not armed: " + "; ".join(problems))

    def _relatch_no_burn(self, orig_addr: int | None) -> None:
        """Best effort, only when FUSECTRL is idle (never after poison): restore FUSEADDR
        and put the FUSEWDATA=0 no-burn latch back, so a stray CMD_PROGRAM (other tool,
        operator typo, firmware) cannot re-burn the staged word into whatever row
        FUSEADDR now selects."""
        try:
            if not is_fusectrl_idle(self.rd(REG_FUSECTRL)):
                return
            if orig_addr is not None:
                self._wr(REG_FUSEADDR, orig_addr)
            self._wr(REG_FUSEWDATA, 0)
        except Exception:
            pass

    def program_row(
        self,
        row: int,
        word: int,
        *,
        timeout_s: float = 10.0,
        pre_trigger_check=None,
    ) -> int:
        """Burn ONE 32-bit FF record into blank row [222..486] with full hardware assertions.

        Every failure before CMD_PROGRAM is raised as ``PreTriggerRefusal`` (nothing burned).
        Failures after CMD_PROGRAM are plain ``BurnError`` (row may be partially programmed).
        """
        orig_addr: int | None = None
        staged = False
        try:
            if self.read_only:
                raise BurnError("cannot program fuse on a read-only controller")
            if isinstance(row, bool) or not (FF_ROW_MIN <= row <= FF_ROW_MAX):
                raise BurnError(
                    f"REFUSED: row {row} is outside the FF region {FF_ROW_MIN}..{FF_ROW_MAX} "
                    "(rows 0..221 are regular/control/FPF; 487..511 are IFF)"
                )
            ff = decode_ff(word)  # Enforces nonzero, 32-bit, and bit 5 (IFF) == 0.
            if (ff.chain, ff.block) in DEVICE_ID_DENIED_BLOCKS:
                raise BurnError(
                    f"REFUSED: FF word 0x{word:08x} targets read-only PCIe device-ID block "
                    f"(chain {ff.chain} block {ff.block})"
                )
            if is_poison(word):
                raise BurnError(f"REFUSED: payload 0x{word:08x} matches poison prefix")

            self.arm()

            current = self.read_row(row)
            if current != FF_EMPTY_WORD:
                raise BurnError(
                    f"REFUSED: target FF row {row} is not blank (reads 0x{current:08x}, want 0x00000000)"
                )

            orig_addr = self.rd(REG_FUSEADDR)
            staged = True
            self._wr(REG_FUSECTRL, CMD_IDLE)
            self._wr(REG_FUSEADDR, row)
            if self.rd(REG_FUSEADDR) != row:
                raise BurnError(f"FUSEADDR did not latch row {row} before CMD_PROGRAM")
            self._wr(REG_FUSEWDATA, word)
            rb_wdata = self.rd(REG_FUSEWDATA)
            if rb_wdata != word:
                raise BurnError(
                    f"FUSEWDATA readback mismatch before CMD_PROGRAM: "
                    f"wrote 0x{word:08x}, read 0x{rb_wdata:08x} (PLM_f4 blocked?)"
                )

            # Final pre-trigger gate: closes TOCTOU between arm/staging and CMD_PROGRAM.
            final_problems = self.armed_problems()
            if final_problems:
                raise BurnError(
                    "final pre-trigger controller gate failed: " + "; ".join(final_problems)
                )
            if self.rd(REG_FUSEADDR) != row:
                raise BurnError(
                    f"FUSEADDR drifted before CMD_PROGRAM (reads 0x{self.rd(REG_FUSEADDR):08x}, want {row})"
                )
            if self.rd(REG_FUSEWDATA) != word:
                raise BurnError(
                    f"FUSEWDATA drifted before CMD_PROGRAM (reads 0x{self.rd(REG_FUSEWDATA):08x}, want 0x{word:08x})"
                )
            if pre_trigger_check is not None:
                pre_trigger_check()
        except PreTriggerRefusal:
            if staged:
                self._relatch_no_burn(orig_addr)
            raise
        except BurnError as exc:
            if staged:
                self._relatch_no_burn(orig_addr)
            raise PreTriggerRefusal(str(exc)) from exc
        except BaseException:
            if staged:
                self._relatch_no_burn(orig_addr)
            raise

        # ---- point of no return: CMD_PROGRAM ----
        try:
            self._wr(REG_FUSECTRL, CMD_PROGRAM)
            self.poll_idle(timeout_s)
        finally:
            self._relatch_no_burn(orig_addr)

        got = self.read_row(row)
        if got != word:
            raise BurnError(
                f"row {row} program readback mismatch: wrote 0x{word:08x}, got 0x{got:08x}"
            )
        return got


def _default_bdf_lock_dir() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        rt_path = Path(runtime)
        if rt_path.is_dir() and not rt_path.is_symlink() and os.access(rt_path, os.W_OK):
            return rt_path / "vfi_burn"
    return Path.home() / ".local" / "state" / "vfi_burn"


class BDFLock:
    """Exclusive non-blocking per-PCI-BDF lock preventing concurrent sessions on the same GPU."""

    def __init__(self, bdf: str, *, lock_dir: str | Path | None = None) -> None:
        self.bdf = normalize_bdf(bdf)
        base = Path(lock_dir) if lock_dir is not None else _default_bdf_lock_dir()
        if base.is_symlink():
            raise BurnError(f"refusing symlink BDF lock directory: {base}")
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        if base.stat().st_mode & 0o077:
            raise BurnError(
                f"BDF lock directory not private (must be mode 0700): {base} — run: chmod 700 {base}"
            )
        safe_name = self.bdf.replace(":", "_")
        self.path = base / f"bdf_{safe_name}.lock"
        if self.path.is_symlink():
            raise BurnError(f"refusing symlink BDF lock file: {self.path}")
        self.fd: int | None = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            if stat.S_IMODE(os.fstat(self.fd).st_mode) != 0o600:
                raise BurnError(f"BDF lock file is not mode 0600: {self.path}")
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BurnError(
                    f"PCI device {self.bdf} is already locked by another vfi_burn process "
                    f"({self.path}); never run concurrent sessions on the same GPU"
                ) from exc
            try:
                os.ftruncate(self.fd, 0)
                os.write(self.fd, f"pid={os.getpid()} bdf={self.bdf}\n".encode("ascii"))
            except OSError:
                pass
        except Exception:
            os.close(self.fd)
            self.fd = None
            raise

    def release(self) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def __enter__(self) -> "BDFLock":
        return self

    def __exit__(self, *args) -> None:
        self.release()


class GA100LiveAdapter:
    """Linux live hardware adapter for CMP 170HX 8 GB (10de:20c2) via sysfs BAR0 + optional Pico PMBus.

    Read-only by default (``writable=False``). When ``writable=True`` is requested by
    ``burn --execute`` or ``manual write-record --execute``, sets ``write_enabled = True``
    (and legacy alias ``writer_verified = True``) to arm ``write_one_record`` through
    ``GA100FuseController.program_row``. This reflects operator write permission, NOT
    independent hardware bench certification.
    """

    simulation = False

    def __init__(
        self,
        bdf: str,
        *,
        pico_port: str | None = None,
        writable: bool = False,
        assume_vout_mv: float | None = None,
        assume_hold_active: bool = False,
        lock_dir: str | Path | None = None,
    ) -> None:
        self.bdf = normalize_bdf(bdf)
        self.pico_port = pico_port
        self.writable = bool(writable)
        self.write_enabled = bool(writable)
        self.writer_verified = bool(writable)  # legacy alias; means write mode enabled, not bench-certified
        self.assume_vout_mv = assume_vout_mv
        self.assume_hold_active = assume_hold_active
        self._bdf_lock: BDFLock | None = None
        self._card = None
        self._fd: int | None = None
        self._mm = None

        self._sysfs = Path("/sys/bus/pci/devices") / self.bdf
        if not self._sysfs.exists():
            raise BurnError(f"PCI device not found in sysfs: {self._sysfs}")

        self._bdf_lock = BDFLock(self.bdf, lock_dir=lock_dir)
        try:
            if self.pico_port:
                self._card = self._open_pico(self.pico_port)
                # Check rail over I2C BEFORE opening BAR0 or reading PCI config space.
                _, vout_mv, _ = self._read_pmbus_rail()
                if vout_mv < HEALTHY_RAIL_MIN_MV:
                    raise BurnError(
                        f"rail at {vout_mv:.1f} mV (brown-out/latch): refusing to touch PCI config or BAR0"
                    )

            # Verify PCI config memory decode is on before mapping BAR0.
            cfg = self._read_pci_config()
            if struct.unpack_from("<I", cfg, 0)[0] == 0xFFFFFFFF:
                raise BurnError("PCI config space reads 0xFFFFFFFF (link down); park/power-cycle card")
            cmd = struct.unpack_from("<H", cfg, 4)[0]
            if (cmd & 0x0002) == 0:
                raise BurnError(
                    f"PCI memory decode is off (COMMAND=0x{cmd:04x}); "
                    f"enable with: sudo setpci -s {self.bdf} COMMAND=0x0002:0x0002"
                )

            res0 = self._sysfs / "resource0"
            self._fd = os.open(res0, os.O_RDWR | os.O_SYNC | getattr(os, "O_CLOEXEC", 0))
            self._mm = mmap.mmap(
                self._fd, BAR0_MAP_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
            )
            self._mmio = AlignedMMIO(self._mm)
            self.controller = GA100FuseController(self._mmio, read_only=not self.writable)
        except Exception:
            self.close()
            raise

    @property
    def rail_telemetry_live(self) -> bool:
        return self._card is not None

    @staticmethod
    def _open_pico(port: str):
        if not os.path.exists(port):
            raise BurnError(f"Pico serial port not found: {port}")
        try:
            from ga100vfi.glitch import Card
        except ImportError as exc:
            raise BurnError(
                "cannot import ga100vfi.glitch.Card for --port; install pico-ga100vfi/host "
                "or omit --port and pass --assume-vout-mv if using an external rail monitor"
            ) from exc
        return Card(port)

    def _close_pico(self) -> None:
        if getattr(self, "_card", None) is not None:
            try:
                self._card.close()
            except Exception:
                pass
            self._card = None

    def close(self) -> None:
        if getattr(self, "_mm", None) is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        if getattr(self, "_fd", None) is not None:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
        self._close_pico()
        if getattr(self, "_bdf_lock", None) is not None:
            self._bdf_lock.release()
            self._bdf_lock = None

    def __enter__(self) -> "GA100LiveAdapter":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _vrm16(self, reg: int) -> int:
        raw = self._card.read(VRM_I2C_ADDR, reg, 2)
        return raw[0] | (raw[1] << 8)

    def _read_pmbus_rail(self) -> tuple[int, float, int]:
        """Return (e1_ctrl_mode, vout_mv, status_word) over Pico I2C."""
        _, e1 = self._card.lock_status()
        vout_mv = float(self._vrm16(PMBUS_REG_READ_VOUT))
        status_word = self._vrm16(PMBUS_REG_STATUS_WORD)
        return e1, vout_mv, status_word

    def _vrm_profile_tainted(self) -> bool:
        if self._card is None:
            return False
        for reg in PROFILE_REGS:
            try:
                if self._vrm16(reg) == PROFILE_FAST[reg]:
                    return True
            except Exception:
                return True
        return False

    def _read_pci_config(self) -> bytes:
        with open(self._sysfs / "config", "rb") as f:
            return f.read(256)

    def pci_id(self) -> str:
        vendor = (self._sysfs / "vendor").read_text(encoding="utf-8").strip().lower()
        device = (self._sysfs / "device").read_text(encoding="utf-8").strip().lower()
        return f"{vendor.removeprefix('0x')}:{device.removeprefix('0x')}"

    def host(self) -> Host:
        drv_link = self._sysfs / "driver"
        bound_driver = drv_link.resolve().name if drv_link.exists() else None
        nvidia_loaded = Path("/sys/module/nvidia").exists()
        nouveau_loaded = Path("/sys/module/nouveau").exists()
        lockdown_path = Path("/sys/kernel/security/lockdown")
        lockdown = lockdown_path.read_text(encoding="utf-8").strip() if lockdown_path.exists() else "[none]"
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        vrm_tainted = self._vrm_profile_tainted()
        if self._card is not None:
            e1, vout_mv, _ = self._read_pmbus_rail()
            hold_active = (e1 == E1_PMBUS_HELD) and (vout_mv >= HEALTHY_RAIL_MIN_MV)
        else:
            # Without Pico PMBus telemetry, treat DSOV==0 + CRC==0x55 or explicit flag as held indicator.
            hold_active = self.assume_hold_active or (
                not is_poison(self.controller.rd(REG_FUSECTRL))
                and self.controller.rd(REG_DSOV_STATUS) == 0
            )
        return Host(
            bound_driver=bound_driver,
            nvidia_loaded=nvidia_loaded,
            nouveau_loaded=nouveau_loaded,
            lockdown=lockdown,
            vrm_tainted=vrm_tainted,
            hold_active=hold_active,
            boot_id=boot_id,
        )

    def health(self) -> Health:
        if self._card is not None:
            _, vout_mv, status_word = self._read_pmbus_rail()
        elif self.assume_vout_mv is not None:
            vout_mv, status_word = float(self.assume_vout_mv), 0
        else:
            raise BurnError(
                "no live PMBus rail telemetry: pass --port /dev/ttyACM0 (Pico) "
                "or explicitly supply --assume-vout-mv <mV> if rail is externally verified"
            )
        return self.controller.read_health_registers(status_word=status_word, vout_mv=vout_mv)

    def die_row(self, index: int) -> int:
        if index not in DIE_ID_ROWS:
            raise BurnError(f"die_row index {index} outside 8..15")
        return self.controller.read_row(index)

    def fuse_row(self, index: int) -> int:
        return self.controller.read_row(index)

    def register(self, address: int) -> int:
        return self.controller.rd(address)

    def field(self, name: str) -> int:
        if name == "LnkCap2":
            return parse_pcie_lnkcap2(self._read_pci_config())
        if name == "SM_COUNT":
            gpc = self.controller.rd(OPTION_REGS["GPC_DISABLE"])
            tpcs = [self.controller.rd(OPTION_REGS[f"TPC_GPC{i}_DISABLE"]) for i in range(TOTAL_GPC)]
            return compute_sm_count(gpc, tpcs)
        key = name.strip().upper()
        if key in OPTION_REGS:
            return self.controller.rd(OPTION_REGS[key])
        raise BurnError(f"unknown GA100 field {name!r}")

    def active_sense_readonly(self) -> None:
        self.controller.sense_ctrl()

    def _pre_trigger_rail_check(self) -> None:
        if self._card is None:
            return
        e1, vout_mv, status_word = self._read_pmbus_rail()
        if e1 != E1_PMBUS_HELD:
            raise BurnError(
                f"live PMBus hold lost before CMD_PROGRAM (E1=0x{e1:04x}, expected 0x{E1_PMBUS_HELD:04x})"
            )
        if vout_mv < HEALTHY_RAIL_MIN_MV:
            raise BurnError(
                f"live rail collapse before CMD_PROGRAM ({vout_mv:.1f} mV < {HEALTHY_RAIL_MIN_MV} mV)"
            )
        if status_word == 0x2042 or (status_word & 0x8840):
            raise BurnError(
                f"live PMBus critical fault before CMD_PROGRAM (STATUS_WORD=0x{status_word:04x})"
            )

    def write_one_record(self, row: int, word: int) -> None:
        if not self.writable:
            raise PreTriggerRefusal("GA100LiveAdapter opened read-only; pass --execute to enable writes")
        check_cb = self._pre_trigger_rail_check if self._card is not None else None
        self.controller.program_row(row, word, pre_trigger_check=check_cb)

    def capture_cold_snapshot(
        self,
        *,
        previous_boot_id: str,
        read_rows: bool = True,
    ) -> dict:
        """Capture a cold-boot snapshot dict compatible with `vfi_burn verify-cold`."""
        host = self.host()
        pci = self.pci_id()
        if pci != SKU_CMP170HX_8GB:
            raise BurnError(f"unexpected PCI ID {pci} (expected {SKU_CMP170HX_8GB})")
        die_rows = {str(r): f"0x{self.die_row(r):08x}" for r in DIE_ID_ROWS}
        fpf_rows: dict[str, str] = {}
        if read_rows:
            pass_a = {r: self.fuse_row(r) for r in range(FF_ROW_MIN, FF_ROW_MAX + 1)}
            pass_b = {r: self.fuse_row(r) for r in range(FF_ROW_MIN, FF_ROW_MAX + 1)}
            if pass_a != pass_b:
                raise BurnError("FF rows differed between two CMD_READ passes during snapshot")
            fpf_rows = {str(r): f"0x{v:08x}" for r, v in pass_a.items() if v != 0}
        fields: dict[str, str | int] = {}
        for fname in (
            "GPC_DISABLE",
            "FBP_DISABLE",
            "FBIO_LAYOUT",
            "ROP_L2_LAYOUT",
            "FBPA_LAYOUT",
            "MAGIC_D",
            "BOOT_GEN3_DIS",
            "DISABLE_GEN3_SPEED",
            "LnkCap2",
            "SM_COUNT",
        ):
            val = self.field(fname)
            fields[fname] = val if fname == "SM_COUNT" else f"0x{val:08x}"
        experimental_registers: dict[str, str] = {}
        controller_state: dict[str, str] = {}
        ctrl_now = self.controller.rd(REG_FUSECTRL)
        if not is_poison(ctrl_now):
            controller_state = {
                "fusectrl": f"0x{ctrl_now:08x}",
                "dsov": f"0x{self.controller.rd(REG_DSOV_STATUS):08x}",
                "en_sw": f"0x{self.controller.rd(REG_EN_SW_OVERRIDE):08x}",
                "plm_f8": f"0x{self.controller.rd(REG_PLM_F8):08x}",
            }
            for exp in EXPERIMENTAL_INSPECT_REGS:
                try:
                    experimental_registers[exp.name] = f"0x{self.controller.rd(exp.address):08x}"
                except Exception as exc:
                    experimental_registers[exp.name] = f"ERR:{exc}"
        return {
            "pci_id": pci,
            "cold_boot": True,
            "boot_id": host.boot_id,
            "previous_boot_id": previous_boot_id,
            "die_rows": die_rows,
            "fpf_rows": fpf_rows,
            "fields": fields,
            "controller_state": controller_state,
            "experimental_registers": experimental_registers,
        }
