"""Unified GA100 (CMP 170HX 8 GB) fuse & register definition catalog.

Single source of truth for:
  - What fuses and fuse-related BAR0 registers we know about
  - Human-readable descriptions and cross-variant reference values (170HX vs A100)
  - Where each fuse lives (FF chain/bit/width and/or BAR0 MMIO fused/defect/status offsets)
  - Fuse type (FF option, OTP-only fuse, L3/HS override latch, or status readout)
  - Whether FF override is possible (`ff_overridable`), burnable (`burnable`), or opt-in (`opt_in`)

This module depends only on ``model.py`` so both ``mmio.py``, ``ffrecord.py``, and ``ga100.py``
can consume it without circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .model import is_poison

TOTAL_GPC = 8
TPC_PER_GPC = 8
SM_PER_TPC = 2
PES_PER_GPC = 3
TOTAL_FBP = 12
TOTAL_FUSE_ROWS = 512
DIE_ID_ROWS: tuple[int, ...] = tuple(range(8, 16))
FF_EMPTY_WORD = 0x00000000

# Chain 2 blocks 37 and 38 carry PCIe device-ID fuses (DEVIDA/DEVIDB); denied by default
# unless explicitly validated as an intentional A100 identity conversion (requires A100 VBIOS
# + 128 GB BAR1 confirmation).
DEVICE_ID_DENIED_BLOCKS: frozenset[tuple[int, int]] = frozenset({(2, 37), (2, 38)})

# Verified GA100 target PCI Device IDs for intentional DEVIDA (c2/b37) + DEVIDB (c2/b38)
# conversion or revert (encoded in FF as ``device_id << 1`` with bit 0 == 0).
A100_TARGET_DEVICE_IDS: dict[int, str] = {
    0x20F1: "A100 PCIe 40 GB (10de:20f1)",
    0x20B5: "A100 PCIe 80 GB (10de:20b5)",
    0x20B0: "A100 SXM4 40 GB (10de:20b0)",
    0x20B2: "A100 SXM4 80 GB (10de:20b2)",
}
# SXM4 IDs stay decodable (so existing plans/cards can be read), but the planner and
# engine refuse to burn them on a CMP 170HX: it is a PCIe board with no SXM4 VBIOS.
A100_SXM4_DEVICE_IDS: frozenset[int] = frozenset({0x20B0, 0x20B2})
CMP170HX_TARGET_DEVICE_IDS: dict[int, str] = {
    0x20C2: "CMP 170HX 8 GB (10de:20c2)",
    0x2082: "CMP 170HX 10 GB (10de:2082)",
}
BURNABLE_TARGET_DEVICE_IDS: dict[int, str] = {
    **A100_TARGET_DEVICE_IDS,
    **CMP170HX_TARGET_DEVICE_IDS,
}

# Core fuse-controller BAR0 registers (verified across GA100 references).
REG_PMC_BOOT_0 = 0x00000000
REG_GFW_BOOT = 0x00118234
REG_FUSECTRL = 0x00820000
REG_FUSEADDR = 0x00820004
REG_FUSEDATA = 0x00820008        # READ-ONLY data port (CMD_READ output)
REG_FUSEWDATA = 0x0082000C       # WRITE-ONLY data port (CMD_PROGRAM input); reads do not reflect it
REG_EN_SW_OVERRIDE = 0x00820040  # write 1 while DSOV=0 to enable shadow writes
REG_IFF_RECORD = 0x00820050      # IFF window: start=v&0x1FF, last=(v>>16)&0x1FF (factory 0x01FB01E8 = rows 488..507); never written
REG_DEBUGCTRL = 0x00820060       # RWL bit 1 (0 = main array, 1 = RIR macro; NEVER write)
REG_CRC_STATUS = 0x00820070      # 0x55 = healthy CRC
REG_EFP_STATUS = 0x00820078      # OPT_ENABLE_FUSE_PROGRAM (0 -> 1 after SENSE_CTRL)
REG_DFP_STATUS = 0x0082007C      # OPT_DISABLE_FUSE_PROGRAM (must be 0)
REG_DSOV_STATUS = 0x00820084     # OPT_DISABLE_SW_OVERRIDE (0 in Goldilocks window)
REG_PLM_F4 = 0x008200F4          # FUSEWDATA PLM (opens to 0xFFFFFFFF via WDATA_WR_SECURE=0)
REG_PLM_F8 = 0x008200F8          # FUSECTRL PLM (0xFFFFFFFF in Goldilocks)
REG_PLM_FC = 0x008200FC          # FUSEOPT shadow PLM (0xFFFFFFFF in Goldilocks)
REG_WDATA_WR_SECURE = 0x0082048C # write 0 -> PLM_F4 auto-opens to 0xFFFFFFFF
REG_CTRL_WR_SECURE = 0x008205B4  # write 0 -> PLM_F8 opens to 0xFFFFFFFF
REG_RESENSE_DISABLE = 0x00821058 # 0 before SENSE_CTRL; becomes 1 after (1 sense/window)
REG_SBFE_SHADOW = 0x00821108     # OPT_SECURED_BURN_FEATURE_ENABLE (write 0 after SENSE_CTRL)

CONTROLLER_REGISTERS: dict[str, int] = {
    "PMC_BOOT_0": REG_PMC_BOOT_0,
    "GFW_BOOT": REG_GFW_BOOT,
    "FUSECTRL": REG_FUSECTRL,
    "FUSEADDR": REG_FUSEADDR,
    "FUSEDATA": REG_FUSEDATA,
    "FUSEWDATA": REG_FUSEWDATA,
    "EN_SW_OVERRIDE": REG_EN_SW_OVERRIDE,
    "IFF_RECORD": REG_IFF_RECORD,
    "DEBUGCTRL": REG_DEBUGCTRL,
    "CRC_STATUS": REG_CRC_STATUS,
    "EFP_STATUS": REG_EFP_STATUS,
    "DFP_STATUS": REG_DFP_STATUS,
    "DSOV_STATUS": REG_DSOV_STATUS,
    "PLM_F4": REG_PLM_F4,
    "PLM_F8": REG_PLM_F8,
    "PLM_FC": REG_PLM_FC,
    "WDATA_WR_SECURE": REG_WDATA_WR_SECURE,
    "CTRL_WR_SECURE": REG_CTRL_WR_SECURE,
    "RESENSE_DISABLE": REG_RESENSE_DISABLE,
    "SBFE_SHADOW": REG_SBFE_SHADOW,
}

CONTROLLER_REG_NAMES_BY_ADDR: dict[int, str] = {addr: name for name, addr in CONTROLLER_REGISTERS.items()}

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


class FuseType(Enum):
    """Classification of a GA100 fuse or fuse-unit register."""

    FF_OPTION = "ff_option"            # Mapped in GA100 Fuseless Fuse (FF) chains
    OTP_ONLY = "otp_only"              # OTP fuse visible via BAR0; NOT mapped in FF burn chains
    HS_OVERRIDE = "hs_override"        # L3/HS-gated runtime feature override / PLM register
    STATUS_READOUT = "status_readout"  # Read-only effective feature status / ECC control readout


@dataclass(frozen=True)
class FFLocation:
    """Bit location of a fuse field inside the GA100 FF chain/block space."""

    chain: int
    bit: int
    width: int

    @property
    def blocks(self) -> tuple[int, ...]:
        return tuple(sorted({(self.bit + i) // 16 for i in range(self.width)}))


@dataclass(frozen=True)
class FuseDefinition:
    """Authoritative metadata for a single GA100 fuse or read-only fuse-unit register."""

    key: str
    name: str
    fuse_type: FuseType
    category: str
    functional_group: str
    description: str
    ff_location: FFLocation | None = None
    ff_overridable: bool = False
    burnable: bool = False
    opt_in: bool = False
    canonical_field: str | None = None
    defect_key: str = ""
    fused_addr: int | None = None
    defect_addr: int | None = None
    status_addr: int | None = None
    aliases: tuple[str, ...] = ()
    cmp170hx_ref: str = ""
    a100_ref: str = ""

    @property
    def blocks(self) -> tuple[int, ...]:
        return self.ff_location.blocks if self.ff_location is not None else ()


@dataclass(frozen=True)
class OptionField:
    """Projection of an FF-mapped FuseDefinition for FF block codec/replay callers."""

    key: str
    name: str
    chain: int
    bit: int
    width: int
    category: str
    defect_key: str = ""
    note: str = ""
    burnable: bool = True
    opt_in: bool = False

    @property
    def blocks(self) -> tuple[int, ...]:
        return tuple(sorted({(self.bit + i) // 16 for i in range(self.width)}))


@dataclass(frozen=True)
class ExperimentalRegister:
    """Projection of a read-only non-FF FuseDefinition for cross-variant inspection."""

    name: str
    address: int
    category: str
    cmp170hx_ref: str
    a100_ref: str
    description: str
    aliases: tuple[str, ...] = ()


def _tpc_fuse_definitions() -> list[FuseDefinition]:
    return [
        FuseDefinition(
            key=f"tpc_gpc{i}_disable",
            name=f"NV_FUSE_OPT_TPC_GPC{i}_DISABLE",
            fuse_type=FuseType.FF_OPTION,
            category="floorsweep",
            functional_group="compute",
            description=f"TPC disable mask for GPC{i} (8 bits, 1 bit per TPC; 1=disabled)",
            ff_location=FFLocation(chain=1, bit=187 + 8 * i, width=8),
            ff_overridable=True,
            burnable=True,
            canonical_field=f"TPC_GPC{i}_DISABLE",
            defect_key=f"tpc_gpc{i}_defective",
            fused_addr=0x00820760 + 4 * i,
            defect_addr=TPC_DEFECTIVE_REGS[i],
            status_addr=0x00820C38 + 4 * i,
            aliases=(f"TPC_GPC{i}_DISABLE",),
        )
        for i in range(TOTAL_GPC)
    ]


FUSE_CATALOG: tuple[FuseDefinition, ...] = tuple(
    [
        # --- FF-mapped floorsweep & subsystem option fuses ---
        FuseDefinition(
            key="fbio_disable",
            name="NV_FUSE_OPT_FBIO_DISABLE",
            fuse_type=FuseType.FF_OPTION,
            category="floorsweep",
            functional_group="memory",
            description="FBIO HBM2e I/O slice disable mask (24 bits; paired 2 bits per FBP)",
            ff_location=FFLocation(chain=1, bit=83, width=24),
            ff_overridable=True,
            burnable=True,
            canonical_field="FBIO_DISABLE",
            defect_key="fbio_defective",
            fused_addr=0x0082036C,
            defect_addr=0x008205D4,
            aliases=("FBIO_DISABLE", "FBIO_LAYOUT"),
        ),
        FuseDefinition(
            key="fbpa_disable",
            name="NV_FUSE_OPT_FBPA_DISABLE",
            fuse_type=FuseType.FF_OPTION,
            category="floorsweep",
            functional_group="memory",
            description="FBPA memory controller slice disable mask (24 bits; paired 2 bits per FBP)",
            ff_location=FFLocation(chain=1, bit=107, width=24),
            ff_overridable=True,
            burnable=True,
            canonical_field="FBPA_DISABLE",
            defect_key="fbpa_defective",
            fused_addr=0x00820368,
            defect_addr=0x008205D0,
            aliases=("FBPA_DISABLE", "FBPA_LAYOUT"),
        ),
        FuseDefinition(
            key="gpc_disable",
            name="NV_FUSE_OPT_GPC_DISABLE",
            fuse_type=FuseType.FF_OPTION,
            category="floorsweep",
            functional_group="compute",
            description="GPC disable mask (8 bits; 1=disabled)",
            ff_location=FFLocation(chain=1, bit=131, width=8),
            ff_overridable=True,
            burnable=True,
            canonical_field="GPC_DISABLE",
            defect_key="gpc_defective",
            fused_addr=0x00820350,
            defect_addr=0x008205C4,
            status_addr=0x00820C1C,
            aliases=("GPC_DISABLE",),
        ),
    ]
    + _tpc_fuse_definitions()
    + [
        FuseDefinition(
            key="pes_disable",
            name="NV_FUSE_OPT_PES_DISABLE",
            fuse_type=FuseType.FF_OPTION,
            category="floorsweep",
            functional_group="compute",
            description="PES disable mask (24 bits; 3 PES per GPC)",
            ff_location=FFLocation(chain=1, bit=251, width=24),
            ff_overridable=True,
            burnable=True,
            canonical_field="PES_DISABLE",
            defect_key="pes_defective",
            fused_addr=0x00820690,
            defect_addr=0x00820698,
            aliases=("PES_DISABLE",),
        ),
        FuseDefinition(
            key="fbp_disable",
            name="NV_FUSE_OPT_FBP_DISABLE",
            fuse_type=FuseType.FF_OPTION,
            category="floorsweep",
            functional_group="memory",
            description="FBP memory partition disable mask (12 bits; 1=disabled)",
            ff_location=FFLocation(chain=1, bit=307, width=12),
            ff_overridable=True,
            burnable=True,
            canonical_field="FBP_DISABLE",
            defect_key="fbp_defective",
            fused_addr=0x00820364,
            defect_addr=0x008205CC,
            status_addr=0x00820D38,
            aliases=("FBP_DISABLE",),
        ),
        FuseDefinition(
            key="rop_l2_disable",
            name="NV_FUSE_OPT_ROP_L2_DISABLE",
            fuse_type=FuseType.FF_OPTION,
            category="floorsweep",
            functional_group="memory",
            description="ROP/L2 cache slice disable mask (24 bits; paired 2 bits per FBP)",
            ff_location=FFLocation(chain=1, bit=319, width=24),
            ff_overridable=True,
            burnable=True,
            canonical_field="ROP_L2_DISABLE",
            defect_key="rop_l2_defective",
            fused_addr=0x008202C4,
            defect_addr=0x008205E8,
            aliases=("ROP_L2_DISABLE", "ROP_L2_LAYOUT"),
        ),
        FuseDefinition(
            key="nvlink_disable",
            name="NV_FUSE_OPT_NVLINK_DISABLE",
            fuse_type=FuseType.FF_OPTION,
            category="nvlink",
            functional_group="nvlink",
            description="NVLink additionally needs U11 bridge mod + firmware check bypass",
            ff_location=FFLocation(chain=1, bit=368, width=3),
            ff_overridable=True,
            burnable=True,
            canonical_field="NVLINK_DISABLE",
            defect_key="nvlink_defective",
            fused_addr=0x00820684,
            defect_addr=0x0082068C,
            aliases=("NVLINK_DISABLE",),
        ),
        FuseDefinition(
            key="sys_pipe_disable",
            name="NV_FUSE_OPT_SYS_PIPE_DISABLE",
            fuse_type=FuseType.FF_OPTION,
            category="mig",
            functional_group="mig",
            description="MIG/ECC/NVDEC unlock untested per Kate's writeup",
            ff_location=FFLocation(chain=1, bit=372, width=8),
            ff_overridable=True,
            burnable=True,
            canonical_field="SYS_PIPE_DISABLE",
            defect_key="sys_pipe_defective",
            fused_addr=0x008203A4,
            defect_addr=0x008203AC,
            aliases=("SYS_PIPE_DISABLE",),
        ),
        # --- Chain 2 identity (read-only, NEVER burnable) & PCIe speed/boot fuses ---
        FuseDefinition(
            key="pcie_devida",
            name="NV_FUSE_OPT_PCIE_DEVIDA",
            fuse_type=FuseType.FF_OPTION,
            category="identity",
            functional_group="identity",
            description="device ID (read-only)",
            ff_location=FFLocation(chain=2, bit=593, width=16),
            ff_overridable=False,
            burnable=False,
            fused_addr=0x008204D8,
            aliases=("PCIE_DEVIDA",),
        ),
        FuseDefinition(
            key="pcie_devidb",
            name="NV_FUSE_OPT_PCIE_DEVIDB",
            fuse_type=FuseType.FF_OPTION,
            category="identity",
            functional_group="identity",
            description="device ID (read-only)",
            ff_location=FFLocation(chain=2, bit=609, width=16),
            ff_overridable=False,
            burnable=False,
            fused_addr=0x0082056C,
            aliases=("PCIE_DEVIDB",),
        ),
        FuseDefinition(
            key="disable_gen3_speed_bit1",
            name="NV_FUSE_OPT_DISABLE_GEN3_SPEED",
            fuse_type=FuseType.FF_OPTION,
            category="pcie",
            functional_group="pcie",
            description="PCIe Gen3 restriction (bit 1)",
            ff_location=FFLocation(chain=2, bit=39 * 16 + 1, width=1),
            ff_overridable=True,
            burnable=True,
            fused_addr=0x00820250,
            aliases=("DISABLE_GEN3_SPEED", "OPT_DISABLE_GEN3_SPEED"),
        ),
        FuseDefinition(
            key="disable_gen3_speed_bit9",
            name="NV_FUSE_OPT_DISABLE_GEN3_SPEED",
            fuse_type=FuseType.FF_OPTION,
            category="pcie",
            functional_group="pcie",
            description="PCIe Gen3 restriction (bit 9)",
            ff_location=FFLocation(chain=2, bit=39 * 16 + 9, width=1),
            ff_overridable=True,
            burnable=True,
            fused_addr=0x00820250,
        ),
        FuseDefinition(
            key="pcie_magic_gen4_bit19",
            name="NV_FUSE_OPT_PCIE_MAGIC_BITS_D[19]",
            fuse_type=FuseType.FF_OPTION,
            category="pcie",
            functional_group="pcie",
            description="PCIe Gen4 MAGIC_BITS_D bit 19 (cleared for A100 0x00200000)",
            ff_location=FFLocation(chain=2, bit=46 * 16 + 13, width=1),
            ff_overridable=True,
            burnable=True,
            fused_addr=0x00820520,
        ),
        FuseDefinition(
            key="pcie_magic_gen4_bit22",
            name="NV_FUSE_OPT_PCIE_MAGIC_BITS_D[22]",
            fuse_type=FuseType.FF_OPTION,
            category="pcie",
            functional_group="pcie",
            description="PCIe Gen4 MAGIC_BITS_D bit 22 (cleared for A100 0x00200000)",
            ff_location=FFLocation(chain=2, bit=47 * 16 + 0, width=1),
            ff_overridable=True,
            burnable=True,
            fused_addr=0x00820520,
        ),
        FuseDefinition(
            key="pcie_magic_gen4_dis",
            name="NV_FUSE_OPT_PCIE_MAGIC_BITS_D[25]",
            fuse_type=FuseType.FF_OPTION,
            category="pcie",
            functional_group="pcie",
            description="Gen4 speed-disable flag (MAGIC_BITS_D bit 25)",
            ff_location=FFLocation(chain=2, bit=47 * 16 + 3, width=1),
            ff_overridable=True,
            burnable=True,
            fused_addr=0x00820520,
        ),
        FuseDefinition(
            key="pcie_magic_gen4_bit26",
            name="NV_FUSE_OPT_PCIE_MAGIC_BITS_D[26]",
            fuse_type=FuseType.FF_OPTION,
            category="pcie",
            functional_group="pcie",
            description="PCIe Gen4 MAGIC_BITS_D bit 26 (cleared for A100 0x00200000)",
            ff_location=FFLocation(chain=2, bit=47 * 16 + 4, width=1),
            ff_overridable=True,
            burnable=True,
            fused_addr=0x00820520,
        ),
        FuseDefinition(
            key="pcie_magic_gen4_bit28",
            name="NV_FUSE_OPT_PCIE_MAGIC_BITS_D[28]",
            fuse_type=FuseType.FF_OPTION,
            category="pcie",
            functional_group="pcie",
            description="PCIe Gen4 MAGIC_BITS_D bit 28 (cleared for A100 0x00200000)",
            ff_location=FFLocation(chain=2, bit=47 * 16 + 6, width=1),
            ff_overridable=True,
            burnable=True,
            fused_addr=0x00820520,
        ),
        FuseDefinition(
            key="pcie_boot_gen23_dis",
            name="NV_FUSE_OPT_PCIE_BOOT_GEN23_DISABLE",
            fuse_type=FuseType.FF_OPTION,
            category="pcie",
            functional_group="pcie",
            description="PCIe Gen2/Gen3 boot speed restriction",
            ff_location=FFLocation(chain=2, bit=49 * 16 + 6, width=1),
            ff_overridable=True,
            burnable=True,
            fused_addr=0x0082057C,
            aliases=("BOOT_GEN23_DIS", "OPT_PCIE_BOOT_GEN23_DIS"),
        ),
        FuseDefinition(
            key="pcie_boot_gen3_dis",
            name="NV_FUSE_OPT_PCIE_BOOT_GEN3_DISABLE",
            fuse_type=FuseType.FF_OPTION,
            category="pcie",
            functional_group="pcie",
            description="PCIe Gen3 boot speed restriction",
            ff_location=FFLocation(chain=2, bit=49 * 16 + 7, width=1),
            ff_overridable=True,
            burnable=True,
            fused_addr=0x00820580,
            aliases=("BOOT_GEN3_DIS", "OPT_PCIE_BOOT_GEN3_DIS"),
        ),
        FuseDefinition(
            key="magic_d",
            name="NV_FUSE_OPT_PCIE_MAGIC_BITS_D",
            fuse_type=FuseType.FF_OPTION,
            category="identity",
            functional_group="pcie",
            description="whole MAGIC_BITS_D word (Gen4 bits 19/22/25/26/28 are standard; other bits remain opt-in)",
            ff_location=FFLocation(chain=2, bit=730, width=32),
            ff_overridable=True,
            burnable=True,
            opt_in=True,
            fused_addr=0x00820520,
            aliases=("MAGIC_D", "OPT_PCIE_MAGIC_BITS_D"),
        ),
        # --- Non-FF OTP fuses (strictly read-only inspection; NOT FF-overridable) ---
        FuseDefinition(
            key="fuse_ecc_en",
            name="FUSE_ECC_EN",
            fuse_type=FuseType.OTP_ONLY,
            category="ECC & Feature Fuses",
            functional_group="ecc",
            description="OPT_ECC_EN; ECC enable OTP fuse (bit 0; 0=CMP/consumer, 1=A100/Pro)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x00820228,
            aliases=("OPT_ECC_EN",),
            cmp170hx_ref="0x0",
            a100_ref="0x1",
        ),
        FuseDefinition(
            key="fuse_feat_ovr_dis",
            name="FUSE_FEAT_OVR_DIS",
            fuse_type=FuseType.OTP_ONLY,
            category="ECC & Feature Fuses",
            functional_group="feature_override",
            description="OPT_FEATURE_FUSES_OVERRIDE_DISABLE; 0=HS feature overrides available",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x008203F0,
            aliases=("OPT_FEATURE_FUSES_OVERRIDE_DISABLE",),
            cmp170hx_ref="0x0",
            a100_ref="0x0",
        ),
        FuseDefinition(
            key="fuse_half_fbpa_en",
            name="FUSE_HALF_FBPA_EN",
            fuse_type=FuseType.OTP_ONLY,
            category="ECC & Feature Fuses",
            functional_group="memory",
            description="OPT_HALF_FBPA_ENABLE; per-FBPA half-capacity fuse mask (24-bit)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x0082049C,
            aliases=("OPT_HALF_FBPA_ENABLE",),
            cmp170hx_ref="0x0",
            a100_ref="0x0",
        ),
        FuseDefinition(
            key="fuse_fb_config",
            name="FUSE_FB_CONFIG",
            fuse_type=FuseType.OTP_ONLY,
            category="ECC & Feature Fuses",
            functional_group="memory",
            description="OPT_FB_CONFIG; 4-bit memory topology selector",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x00820328,
            aliases=("OPT_FB_CONFIG",),
            cmp170hx_ref="0x0",
            a100_ref="0x0",
        ),
        FuseDefinition(
            key="fuse_sku_id",
            name="FUSE_SKU_ID",
            fuse_type=FuseType.OTP_ONLY,
            category="ECC & Feature Fuses",
            functional_group="identity",
            description="OPT_SKU_ID; 8-bit SKU identifier (170HX 10G=0x68, A100=0x54/0x5b/0x7b)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x00821060,
            aliases=("OPT_SKU_ID",),
            cmp170hx_ref="0x68",
            a100_ref="0x54/0x5b/0x7b",
        ),
        # --- SM speed select OTP fuses (FMA / Tensor Core throttle; read-only) ---
        FuseDefinition(
            key="fuse_ss_dp",
            name="FUSE_SS_DP",
            fuse_type=FuseType.OTP_ONLY,
            category="SM Speed Select OTP Fuses",
            functional_group="speed_select",
            description="OPT_SM_SPEED_SELECT_DP; FP64 throttle (1-bit: 0=full, 1=reduced)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x00820224,
            aliases=("OPT_SM_SPEED_SELECT_DP",),
            cmp170hx_ref="0x1",
            a100_ref="0x0",
        ),
        FuseDefinition(
            key="fuse_ss_ffma",
            name="FUSE_SS_FFMA",
            fuse_type=FuseType.OTP_ONLY,
            category="SM Speed Select OTP Fuses",
            functional_group="speed_select",
            description="OPT_SM_SPEED_SELECT_FFMA; FP32 FMA throttle (3-bit: 0=full..5=max)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x0082059C,
            aliases=("OPT_SM_SPEED_SELECT_FFMA",),
            cmp170hx_ref="0x5",
            a100_ref="0x0",
        ),
        FuseDefinition(
            key="fuse_ss_fmla16",
            name="FUSE_SS_FMLA16",
            fuse_type=FuseType.OTP_ONLY,
            category="SM Speed Select OTP Fuses",
            functional_group="speed_select",
            description="OPT_SM_SPEED_SELECT_FMLA16; FP16 MLA/Tensor throttle (3-bit)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x008207D4,
            aliases=("OPT_SM_SPEED_SELECT_FMLA16",),
            cmp170hx_ref="0x5",
            a100_ref="0x0",
        ),
        FuseDefinition(
            key="fuse_ss_fmla32",
            name="FUSE_SS_FMLA32",
            fuse_type=FuseType.OTP_ONLY,
            category="SM Speed Select OTP Fuses",
            functional_group="speed_select",
            description="OPT_SM_SPEED_SELECT_FMLA32; FP32 MLA/Tensor throttle (3-bit)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x008207D8,
            aliases=("OPT_SM_SPEED_SELECT_FMLA32",),
            cmp170hx_ref="0x5",
            a100_ref="0x0",
        ),
        FuseDefinition(
            key="fuse_ss_imla0",
            name="FUSE_SS_IMLA0",
            fuse_type=FuseType.OTP_ONLY,
            category="SM Speed Select OTP Fuses",
            functional_group="speed_select",
            description="OPT_SM_SPEED_SELECT_IMLA0; INT MLA0 / DP4A throttle (3-bit)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x008207DC,
            aliases=("OPT_SM_SPEED_SELECT_IMLA0",),
            cmp170hx_ref="0x5",
            a100_ref="0x0",
        ),
        FuseDefinition(
            key="fuse_ss_imla1",
            name="FUSE_SS_IMLA1",
            fuse_type=FuseType.OTP_ONLY,
            category="SM Speed Select OTP Fuses",
            functional_group="speed_select",
            description="OPT_SM_SPEED_SELECT_IMLA1; INT MLA1 throttle (3-bit)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x008207E0,
            aliases=("OPT_SM_SPEED_SELECT_IMLA1",),
            cmp170hx_ref="0x5",
            a100_ref="0x0",
        ),
        FuseDefinition(
            key="fuse_ss_imla2",
            name="FUSE_SS_IMLA2",
            fuse_type=FuseType.OTP_ONLY,
            category="SM Speed Select OTP Fuses",
            functional_group="speed_select",
            description="OPT_SM_SPEED_SELECT_IMLA2; INT MLA2 throttle (3-bit)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x008207E4,
            aliases=("OPT_SM_SPEED_SELECT_IMLA2",),
            cmp170hx_ref="0x5",
            a100_ref="0x0",
        ),
        FuseDefinition(
            key="fuse_ss_imla3",
            name="FUSE_SS_IMLA3",
            fuse_type=FuseType.OTP_ONLY,
            category="SM Speed Select OTP Fuses",
            functional_group="speed_select",
            description="OPT_SM_SPEED_SELECT_IMLA3; INT MLA3 throttle (3-bit)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x008207E8,
            aliases=("OPT_SM_SPEED_SELECT_IMLA3",),
            cmp170hx_ref="0x5",
            a100_ref="0x0",
        ),
        FuseDefinition(
            key="fuse_ss_imla4",
            name="FUSE_SS_IMLA4",
            fuse_type=FuseType.OTP_ONLY,
            category="SM Speed Select OTP Fuses",
            functional_group="speed_select",
            description="OPT_SM_SPEED_SELECT_IMLA4; INT MLA4 throttle (3-bit)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x008207EC,
            aliases=("OPT_SM_SPEED_SELECT_IMLA4",),
            cmp170hx_ref="0x5",
            a100_ref="0x0",
        ),
        # --- Feature override & readout chain (L3/HS-gated runtime state; read-only) ---
        FuseDefinition(
            key="feat_ovr_ecc_plm",
            name="FEAT_OVR_ECC_PLM",
            fuse_type=FuseType.HS_OVERRIDE,
            category="Feature Override & Readout Chain",
            functional_group="feature_override",
            description="FEATURE_OVERRIDE_ECC PLM (0xffffff8f = L3/HS writable)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x00823800,
            cmp170hx_ref="0xffffff8f",
            a100_ref="0xffffff8f",
        ),
        FuseDefinition(
            key="feat_ovr_plm",
            name="FEAT_OVR_PLM",
            fuse_type=FuseType.HS_OVERRIDE,
            category="Feature Override & Readout Chain",
            functional_group="feature_override",
            description="FEATURE_OVERRIDE PLM (self-sealing, L3/HS writable)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x00823804,
            cmp170hx_ref="0xffffff8f",
            a100_ref="0xffffff8f",
        ),
        FuseDefinition(
            key="feat_ovr_ecc",
            name="FEAT_OVR_ECC",
            fuse_type=FuseType.HS_OVERRIDE,
            category="Feature Override & Readout Chain",
            functional_group="feature_override",
            description="Per-unit ECC override (SM_LRF/L1/LTC/DRAM/CBU)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x0082380C,
            cmp170hx_ref="0x00888888",
            a100_ref="0x00001111/0x00110111",
        ),
        FuseDefinition(
            key="feat_ovr_ecc_1",
            name="FEAT_OVR_ECC_1",
            fuse_type=FuseType.HS_OVERRIDE,
            category="Feature Override & Readout Chain",
            functional_group="feature_override",
            description="Per-unit ECC override 1 (icache/FECS/GPCCS/PMU/HUBMMU)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x00823810,
            cmp170hx_ref="0x002aaaaa",
            a100_ref="0x00104015/0x00104104",
        ),
        FuseDefinition(
            key="feat_ovr_ecc_2",
            name="FEAT_OVR_ECC_2",
            fuse_type=FuseType.HS_OVERRIDE,
            category="Feature Override & Readout Chain",
            functional_group="feature_override",
            description="Per-unit ECC override 2 (LTC_CBC/SM_URF)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x0082382C,
            cmp170hx_ref="0x0000000a",
            a100_ref="0x00000000/0x00000001",
        ),
        FuseDefinition(
            key="feat_readout_0",
            name="FEAT_READOUT_0",
            fuse_type=FuseType.STATUS_READOUT,
            category="Feature Override & Readout Chain",
            functional_group="feature_override",
            description="Effective Quadro[5:0] + ECC[31:12] status readout (RO)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x00823814,
            cmp170hx_ref="0x00000233",
            a100_ref="0xef8ff100",
        ),
        FuseDefinition(
            key="feat_readout_1",
            name="FEAT_READOUT_1",
            fuse_type=FuseType.STATUS_READOUT,
            category="Feature Override & Readout Chain",
            functional_group="feature_override",
            description="Effective SM speed-select throttle readout across all 9 units (RO)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x00823818,
            cmp170hx_ref="0x016db6ed",
            a100_ref="0x00000000",
        ),
        FuseDefinition(
            key="feat_readout_2",
            name="FEAT_READOUT_2",
            fuse_type=FuseType.STATUS_READOUT,
            category="Feature Override & Readout Chain",
            functional_group="feature_override",
            description="Effective row-remapper + ECC LTC_CBC/SM_URF readout (RO)",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x00823828,
            cmp170hx_ref="0x00000000",
            a100_ref="0x00000007",
        ),
        FuseDefinition(
            key="fbpa_ecc_ctrl",
            name="FBPA_ECC_CTRL",
            fuse_type=FuseType.STATUS_READOUT,
            category="Feature Override & Readout Chain",
            functional_group="ecc",
            description="Broadcast FBPA DRAM ECC control register",
            ff_overridable=False,
            burnable=False,
            fused_addr=0x009A0470,
            cmp170hx_ref="0x00000000",
            a100_ref="0x00000041",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Derived indexes & backward-compatible projections (all computed from FUSE_CATALOG)
# ---------------------------------------------------------------------------

FUSES_BY_KEY: dict[str, FuseDefinition] = {f.key: f for f in FUSE_CATALOG}

OPTION_FIELDS: tuple[OptionField, ...] = tuple(
    OptionField(
        key=f.key,
        name=f.name,
        chain=f.ff_location.chain,
        bit=f.ff_location.bit,
        width=f.ff_location.width,
        category=f.category,
        defect_key=f.defect_key,
        note=f.description if f.category != "floorsweep" else "",
        burnable=f.burnable,
        opt_in=f.opt_in,
    )
    for f in FUSE_CATALOG
    if f.ff_location is not None
)

FIELDS_BY_KEY: dict[str, OptionField] = {f.key: f for f in OPTION_FIELDS}

FFCALC_KEY_TO_FIELD: dict[str, str] = {
    f.key: f.canonical_field
    for f in FUSE_CATALOG
    if f.canonical_field is not None
}

FIELD_FUNCTIONAL_GROUP: dict[str, str] = {
    f.key: f.functional_group
    for f in FUSE_CATALOG
    if f.ff_location is not None
}

EXPERIMENTAL_INSPECT_REGS: tuple[ExperimentalRegister, ...] = tuple(
    ExperimentalRegister(
        name=f.name,
        address=f.fused_addr,
        category=f.category,
        cmp170hx_ref=f.cmp170hx_ref,
        a100_ref=f.a100_ref,
        description=f.description,
        aliases=f.aliases,
    )
    for f in FUSE_CATALOG
    if f.fuse_type is not FuseType.FF_OPTION and f.fused_addr is not None
)

EXPERIMENTAL_REGS_BY_NAME: dict[str, int] = {}
EXPERIMENTAL_BY_ADDR: dict[int, ExperimentalRegister] = {}
EXPERIMENTAL_BY_NAME: dict[str, ExperimentalRegister] = {}
for _exp in EXPERIMENTAL_INSPECT_REGS:
    EXPERIMENTAL_BY_ADDR[_exp.address] = _exp
    EXPERIMENTAL_REGS_BY_NAME[_exp.name] = _exp.address
    EXPERIMENTAL_BY_NAME[_exp.name] = _exp
    for _alias in _exp.aliases:
        EXPERIMENTAL_REGS_BY_NAME[_alias] = _exp.address
        EXPERIMENTAL_BY_NAME[_alias] = _exp

OPTION_REGS: dict[str, int] = {}
for _f in FUSE_CATALOG:
    if _f.fused_addr is not None:
        if _f.fuse_type is not FuseType.FF_OPTION:
            OPTION_REGS[_f.name] = _f.fused_addr
        for _alias in _f.aliases:
            OPTION_REGS[_alias] = _f.fused_addr
    if _f.defect_addr is not None and _f.defect_key:
        OPTION_REGS[_f.defect_key.upper()] = _f.defect_addr
    if _f.status_addr is not None and _f.canonical_field:
        OPTION_REGS[_f.canonical_field.replace("_DISABLE", "_STATUS")] = _f.status_addr


def register_name_for_address(address: int) -> str:
    """Return a human-readable register name for a known BAR0 address, or its hex string."""
    if address in CONTROLLER_REG_NAMES_BY_ADDR:
        return CONTROLLER_REG_NAMES_BY_ADDR[address]
    if address in EXPERIMENTAL_BY_ADDR:
        return EXPERIMENTAL_BY_ADDR[address].name
    for name, addr in OPTION_REGS.items():
        if addr == address:
            return name
    return f"0x{address:08x}"


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
