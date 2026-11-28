"""Verified GA100 (CMP 170HX 8 GB) fuse-controller protocol and Linux BAR0 adapter.

Register addresses, commands, and Goldilocks arming/programming sequences are
cross-verified across:
  - Kate Land's GA100VFI writeup (blog.kate.land)
  - sayhellotojungle/cmp170hx-ffcalc (live 17-record CMP 170HX burn)
  - kprinssu/ga100-ff-tools (live 3-card CMP 170HX arm + burn)
  - darkmaniac7/cmp170hx-8gb-vfi-tools (aligned 32-bit BAR0 store)

Every MMIO access uses AlignedMMIO (single 32-bit memoryview.cast("I") loads/stores,
NEVER struct.pack_into). DEBUGCTRL (0x00820060) is never written.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import fcntl
import mmap
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import time

from .ffrecord import (
    FF_ROW_MAX,
    FF_ROW_MIN,
    compute_sm_count,
    decode_a100_devid_ff_word,
    decode_ff,
)
from .fuses import (
    DEVICE_ID_DENIED_BLOCKS,
    DIE_ID_ROWS,
    EXPERIMENTAL_INSPECT_REGS,
    EXPERIMENTAL_REGS_BY_NAME,
    FFCALC_KEY_TO_FIELD,
    FF_EMPTY_WORD,
    FUSES_BY_KEY,
    OPTION_REGS,
    REG_CRC_STATUS,
    REG_CTRL_WR_SECURE,
    REG_DEBUGCTRL,
    REG_DFP_STATUS,
    REG_DSOV_STATUS,
    REG_EFP_STATUS,
    REG_EN_SW_OVERRIDE,
    REG_FUSEADDR,
    REG_FUSECTRL,
    REG_FUSEDATA,
    REG_FUSEWDATA,
    REG_GFW_BOOT,
    REG_IFF_RECORD,
    REG_PLM_F4,
    REG_PLM_F8,
    REG_PLM_FC,
    REG_PMC_BOOT_0,
    REG_RESENSE_DISABLE,
    REG_SBFE_SHADOW,
    REG_WDATA_WR_SECURE,
    TOTAL_FBP,
    TOTAL_FUSE_ROWS,
    TOTAL_GPC,
    TPC_DEFECTIVE_REGS,
    ExperimentalRegister,
    decode_experimental_register,
)
from .interfaces import FuseControllerHAL, FuseReader, FuseWriter, Health, Host, MMIOBus
from .mmio import AlignedMMIO, ReadOnlyMMIO
from .model import (
    BurnError,
    PreTriggerRefusal,
    SKU_CMP170HX_8GB,
    SUPPORTED_CMP170HX_SKUS,
    is_poison,
)

BAR0_MAP_SIZE = 0x01000000  # 16 MiB BAR0 window

# FUSECTRL commands & state field.
CMD_IDLE = 0x0
CMD_READ = 0x1
CMD_PROGRAM = 0x2                # busy: state field [20:16] = 0xC (e.g. 0xE00C0000) -> IDLE (4) in ~1 ms
CMD_SENSE_CTRL = 0x3             # restores EFP=1 while keeping DSOV=0
FUSECTRL_STATE_SHIFT = 16
FUSECTRL_STATE_MASK = 0x1F
FUSECTRL_IDLE_STATE = 4
FUSECTRL_CMD_MASK = 0x1F
DEBUGCTRL_RWL_BIT = 1 << 1       # 0 = main array; 1 = RIR macro (NEVER set)

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


class GA100FuseController(FuseControllerHAL):
    """Low-level GA100 fuse-controller HAL over any MMIOBus (real BAR0 or simulated PCIe).

    Works identically over real BAR0 mmap (``AlignedMMIO``) or a simulated PCIe MMIO bus.
    When ``read_only=True``, writes are strictly allowlisted to CMD_READ (and optionally
    CMD_SENSE_CTRL + volatile pre-sense latch); CMD_PROGRAM / FUSEWDATA(nonzero) /
    SBFE_SHADOW are physically blocked inside ``_wr``.
    """

    def __init__(
        self,
        mmio: MMIOBus,
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

    def _confirm_bad_prefixed_row_data(self, row: int, val: int) -> None:
        """Decide whether a 0xBADxxxxx FUSEDATA value is a fuse-unit error or real OTP data.

        FUSEDATA returns the raw 32-bit row. An FF record's top 16 bits are its block data,
        so a legitimate record with data 0xBAD0..0xBADF reads back as 0xBADxxxxx. Treat it
        as an error only if the controller is unhealthy straight after the read (FUSECTRL
        poisoned / not idle, or PMC_BOOT_0 bad: raised by the helpers) or a second FUSEDATA
        read disagrees. Otherwise it is data and the caller gets it unchanged.
        """
        ctrl = self._check_fusectrl_first()
        if not is_fusectrl_idle(ctrl):
            raise BurnError(
                f"FUSEDATA {val:#010x} reading row {row} with FUSECTRL not idle ({ctrl:#010x}): "
                "treating as fuse-unit error"
            )
        self._check_bar0_alive()
        again = self.rd(REG_FUSEDATA)
        if again != val:
            raise BurnError(
                f"FUSEDATA unstable reading row {row} ({val:#010x} then {again:#010x}): "
                "treating as fuse-unit error"
            )

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

    def read_health_registers(self) -> Health:
        """Read health registers in safe order: FUSECTRL FIRST, stopping on poison."""
        ctrl = self._check_fusectrl_first()
        dsov = self.rd(REG_DSOV_STATUS)
        if is_poison(dsov):
            return Health(ctrl, dsov, 0, 0, 0, 0)
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
                self._confirm_bad_prefixed_row_data(row, val)
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
        # A closed/partial PLM_f8 means the frame is too deep -> abort.
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
        allow_a100_devid: bool = False,
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
                if not (allow_a100_devid and decode_a100_devid_ff_word(word) is not None):
                    raise BurnError(
                        f"REFUSED: FF word 0x{word:08x} targets read-only PCIe device-ID block "
                        f"(chain {ff.chain} block {ff.block})"
                    )
            if is_poison(word):
                raise BurnError(
                    f"REFUSED: payload 0x{word:08x} has 0xBADx block data: deliberately not burned "
                    "(it would read back like a fuse-unit error at the moment of programming)"
                )

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


class GA100LiveAdapter(FuseReader, FuseWriter):
    """Linux live hardware adapter for CMP 170HX 8 GB (10de:20c2) via sysfs BAR0.

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
        writable: bool = False,
        assume_hold_active: bool = False,
        lock_dir: str | Path | None = None,
    ) -> None:
        self.bdf = normalize_bdf(bdf)
        self.writable = bool(writable)
        self.write_enabled = bool(writable)
        self.writer_verified = bool(writable)  # legacy alias; means write mode enabled, not bench-certified
        self.assume_hold_active = assume_hold_active
        self._bdf_lock: BDFLock | None = None
        self._fd: int | None = None
        self._mm = None

        self._sysfs = Path("/sys/bus/pci/devices") / self.bdf
        if not self._sysfs.exists():
            raise BurnError(f"PCI device not found in sysfs: {self._sysfs}")

        self._bdf_lock = BDFLock(self.bdf, lock_dir=lock_dir)
        try:
            # Verify PCI config memory decode is on before mapping BAR0.
            try:
                cfg = self._read_pci_config()
            except PermissionError as exc:
                raise BurnError(
                    f"root privileges are required to read PCI config ({self._sysfs / 'config'}) — re-run with sudo"
                ) from exc
            if struct.unpack_from("<I", cfg, 0)[0] == 0xFFFFFFFF:
                raise BurnError("PCI config space reads 0xFFFFFFFF (link down); park/power-cycle card")
            cmd = struct.unpack_from("<H", cfg, 4)[0]
            if (cmd & 0x0002) == 0:
                raise BurnError(
                    f"PCI memory decode is off (COMMAND=0x{cmd:04x}); "
                    f"enable with: sudo setpci -s {self.bdf} COMMAND=0x0002:0x0002"
                )

            res0 = self._sysfs / "resource0"
            try:
                self._fd = os.open(res0, os.O_RDWR | os.O_SYNC | getattr(os, "O_CLOEXEC", 0))
            except PermissionError as exc:
                raise BurnError(
                    f"root privileges are required to access PCI BAR0 ({res0}) — re-run with sudo"
                ) from exc
            self._mm = mmap.mmap(
                self._fd, BAR0_MAP_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
            )
            self._mmio = AlignedMMIO(self._mm)
            self.controller = GA100FuseController(self._mmio, read_only=not self.writable)
        except Exception:
            self.close()
            raise

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
        if getattr(self, "_bdf_lock", None) is not None:
            self._bdf_lock.release()
            self._bdf_lock = None

    def __enter__(self) -> "GA100LiveAdapter":
        return self

    def __exit__(self, *args) -> None:
        self.close()

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
        hold_active = self.assume_hold_active or (
            not is_poison(self.controller.rd(REG_FUSECTRL))
            and self.controller.rd(REG_DSOV_STATUS) == 0
        )
        return Host(
            bound_driver=bound_driver,
            nvidia_loaded=nvidia_loaded,
            nouveau_loaded=nouveau_loaded,
            lockdown=lockdown,
            hold_active=hold_active,
            boot_id=boot_id,
        )

    def health(self) -> Health:
        return self.controller.read_health_registers()

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

    def write_one_record(
        self, row: int, word: int, *, allow_a100_devid: bool = False
    ) -> None:
        if not self.writable:
            raise PreTriggerRefusal("GA100LiveAdapter opened read-only; pass --execute to enable writes")
        self.controller.program_row(row, word, allow_a100_devid=allow_a100_devid)

    def capture_cold_snapshot(
        self,
        *,
        previous_boot_id: str,
        read_rows: bool = True,
    ) -> dict:
        """Capture a cold-boot snapshot dict compatible with `vfi_burn verify-cold`."""
        host = self.host()
        pci = self.pci_id()
        if pci not in SUPPORTED_CMP170HX_SKUS:
            raise BurnError(
                f"unexpected PCI ID {pci} (expected {' or '.join(SUPPORTED_CMP170HX_SKUS)})"
            )
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


def read_ga100_card_state(
    backend,
    *,
    label: str = "live GPU",
) -> dict:
    """Read full read-only card state (Die IDs 8..15, FF rows 222..486, all 16 defect masks,
    fused option shadows, and optional PCIe LnkCap2) for the built-in unlock configurator.

    Refuses to read defect/shadow masks when ``EFP != 1`` (unsensed pre-SENSE_CTRL glitch
    window where option/defect shadows read 0) so genuine silicon defects are never missed.
    Caller must already have verified ``check_host_readonly_safe(backend.host())``.
    """
    pci = backend.pci_id()
    if pci not in SUPPORTED_CMP170HX_SKUS:
        raise BurnError(
            f"unexpected PCI ID {pci} at {label} (expected {' or '.join(SUPPORTED_CMP170HX_SKUS)})"
        )

    health = backend.health()
    if health.fusectrl == 0xFFFFFFFF or is_poison(health.fusectrl):
        raise BurnError(
            f"{label}: FUSECTRL reads 0x{health.fusectrl:08x} (BAR0 inaccessible or poisoned)"
        )
    if not is_fusectrl_idle(health.fusectrl):
        raise BurnError(
            f"{label}: FUSECTRL is not IDLE (0x{health.fusectrl:08x}); refusing read"
        )
    if health.crc != 0x55:
        raise BurnError(
            f"{label}: CRC_STATUS=0x{health.crc:02x} (expected 0x55); cannot trust fuse reads"
        )
    if health.efp != 1:
        raise BurnError(
            f"{label} option/defect shadow registers are UNSENSED (EFP={health.efp}, DSOV={health.dsov}): "
            "all OPT_*_DEFECTIVE registers read 0 until SENSE_CTRL runs, so genuine silicon defects "
            "cannot be read yet. Either configure unlocks on a cold card (before glitching) or run "
            "[4] Active Sense (SENSE_CTRL) first so EFP=1."
        )

    iff_record = 0
    try:
        raw_iff = backend.register(REG_IFF_RECORD)
        if raw_iff == 0xFFFFFFFF or is_poison(raw_iff):
            raise BurnError(
                f"{label}: poisoned/dead read 0x{raw_iff:08x} from IFF_RECORD (0x{REG_IFF_RECORD:08x})"
            )
        iff_record = raw_iff
    except BurnError:
        raise
    except Exception:
        iff_record = 0

    ff_last_row = FF_ROW_MAX
    iff_start = iff_record & 0xFFFF
    if iff_record != 0 and FF_ROW_MIN < iff_start <= 512:
        ff_last_row = min(FF_ROW_MAX, iff_start - 1)

    die_rows = {r: backend.die_row(r) for r in DIE_ID_ROWS}
    ff_rows = {r: backend.fuse_row(r) for r in range(FF_ROW_MIN, ff_last_row + 1)}

    defect_masks: dict[str, int] = {}
    for ff_key, vfi_name in FFCALC_KEY_TO_FIELD.items():
        addr = FUSES_BY_KEY[ff_key].defect_addr
        if addr is None:
            continue
        val = backend.register(addr)
        if val == 0xFFFFFFFF or is_poison(val):
            raise BurnError(
                f"{label}: poisoned/dead read 0x{val:08x} from defect register {vfi_name} (0x{addr:08x})"
            )
        defect_masks[vfi_name] = val

    shadow_masks: dict[str, int] = {}
    for ff_key in FFCALC_KEY_TO_FIELD.keys():
        addr = FUSES_BY_KEY[ff_key].fused_addr
        if addr is None:
            continue
        val = backend.register(addr)
        if val == 0xFFFFFFFF or is_poison(val):
            raise BurnError(
                f"{label}: poisoned/dead read 0x{val:08x} from fused shadow {ff_key} (0x{addr:08x})"
            )
        shadow_masks[ff_key] = val

    lnkcap2 = 0x06
    try:
        lc = backend.field("LnkCap2")
        if not is_poison(lc) and lc != 0xFFFFFFFF:
            lnkcap2 = lc
    except Exception:
        pass

    return {
        "pci_id": pci,
        "die_rows": die_rows,
        "ff_rows": ff_rows,
        "ff_last_row": ff_last_row,
        "iff_record": iff_record,
        "defect_masks": defect_masks,
        "shadow_masks": shadow_masks,
        "lnkcap2": lnkcap2,
    }


# ---------------------------------------------------------------------------
# Read-only PCI GPU discovery & interactive selector
# ---------------------------------------------------------------------------

_DEFAULT_PCI_IDS_PATHS: tuple[str, ...] = (
    "/usr/share/misc/pci.ids",
    "/usr/share/hwdata/pci.ids",
    "/usr/share/pci.ids",
)

_KNOWN_PCI_GPU_NAMES: dict[str, str] = {
    "10de:20c2": "NVIDIA CMP 170HX 8GB (GA100)",
    "10de:2082": "NVIDIA CMP 170HX 10GB (GA100)",
    "10de:20b0": "NVIDIA A100 SXM4 40GB",
    "10de:20b2": "NVIDIA A100 SXM4 80GB",
    "10de:20b5": "NVIDIA A100 PCIe 80GB",
    "10de:20b7": "NVIDIA A30 PCIe",
    "10de:20f1": "NVIDIA A100 PCIe 40GB",
    "10de:20f3": "NVIDIA A800 SXM4 80GB",
    "10de:20f5": "NVIDIA A800 PCIe 80GB",
    "10de:2204": "NVIDIA GeForce RTX 3090",
    "10de:2206": "NVIDIA GeForce RTX 3080",
    "10de:2684": "NVIDIA GeForce RTX 4090",
    "10de:2330": "NVIDIA H100 SXM5 80GB",
    "10de:2331": "NVIDIA H100 PCIe 80GB",
    "8086:e20b": "Intel Arc B580 (Battlemage)",
}


@dataclass(frozen=True)
class DiscoveredGPU:
    """Read-only summary of one PCI display/3D GPU discovered in Linux sysfs."""

    bdf: str
    pci_id: str
    subsystem_id: str | None
    pci_class: str
    name: str
    driver: str
    bar0_bytes: int | None
    bar1_bytes: int | None
    current_link_speed: str | None
    current_link_width: str | None
    max_link_speed: str | None
    max_link_width: str | None
    is_cmp170hx: bool
    fuse_state: str | None = None
    cur_sm: int | None = None
    max_sm: int | None = None
    cur_fbp: int | None = None
    max_fbp: int | None = None


def format_bytes_iec(size: int | None) -> str:
    """Format a byte count as a compact IEC size string (e.g. '16 GiB', '256 MiB')."""
    if size is None or size <= 0:
        return "?"
    for unit, div in (("TiB", 1 << 40), ("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if size >= div:
            val = size / div
            if val == int(val):
                return f"{int(val)} {unit}"
            return f"{val:.1f} {unit}"
    return f"{size} B"


def _read_sysfs_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _parse_hex_id(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return f"{int(raw, 16):04x}"
    except ValueError:
        return None


def _parse_bar_sizes(resource_text: str | None) -> tuple[int | None, int | None]:
    """Return ``(bar0_bytes, bar1_bytes)`` from sysfs ``resource`` (0x0000..0x0000 -> None)."""
    if not resource_text:
        return None, None
    lines = resource_text.splitlines()
    sizes: list[int | None] = []
    for idx in (0, 1):
        if idx >= len(lines):
            sizes.append(None)
            continue
        parts = lines[idx].split()
        if len(parts) < 2:
            sizes.append(None)
            continue
        try:
            start = int(parts[0], 16)
            end = int(parts[1], 16)
        except ValueError:
            sizes.append(None)
            continue
        sizes.append((end - start + 1) if end > start else None)
    return sizes[0], sizes[1]


def _lookup_pci_ids_names(
    wanted: set[tuple[str, str]],
    pci_ids_paths: Sequence[str | Path] | None = None,
) -> dict[str, str]:
    """Stream Linux ``pci.ids`` once to resolve device names for ``wanted`` ``(vendor, device)`` pairs."""
    if not wanted:
        return {}
    paths = [Path(p) for p in (pci_ids_paths if pci_ids_paths is not None else _DEFAULT_PCI_IDS_PATHS)]
    pci_ids = next((p for p in paths if p.is_file()), None)
    if pci_ids is None:
        return {}

    wanted_vendors = {v for v, _ in wanted}
    found: dict[str, str] = {}
    cur_vendor: str | None = None
    cur_vendor_name = ""
    try:
        with pci_ids.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line or line.startswith("#") or line.startswith("C "):
                    continue
                if not line.startswith("\t"):
                    if cur_vendor in wanted_vendors and all(
                        f"{v}:{d}" in found for v, d in wanted if v == cur_vendor
                    ):
                        wanted_vendors.discard(cur_vendor)
                        if not wanted_vendors:
                            break
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2 and len(parts[0]) == 4:
                        cur_vendor = parts[0].lower()
                        cur_vendor_name = parts[1].strip()
                    else:
                        cur_vendor = None
                elif cur_vendor in wanted_vendors and line.startswith("\t") and not line.startswith("\t\t"):
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2 and len(parts[0]) == 4:
                        dev = parts[0].lower()
                        if (cur_vendor, dev) in wanted:
                            dev_name = parts[1].strip()
                            if cur_vendor == "10de":
                                prefix = "" if "nvidia" in dev_name.lower() else "NVIDIA "
                            else:
                                vword = cur_vendor_name.split()[0] if cur_vendor_name else ""
                                prefix = f"{vword} " if (vword and vword.lower() not in dev_name.lower()) else ""
                            found[f"{cur_vendor}:{dev}"] = f"{prefix}{dev_name}"
    except OSError:
        return found
    return found


def _loaded_gpu_driver_modules(proc_modules_path: Path = Path("/proc/modules")) -> set[str]:
    found: set[str] = set()
    try:
        text = proc_modules_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return found
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] in ("nvidia", "nouveau"):
            found.add(parts[0])
    return found


def _probe_cmp170hx_bar0_readonly(
    dev_dir: Path,
    driver: str,
    *,
    proc_modules_path: Path = Path("/proc/modules"),
) -> tuple[str | None, int | None, int | None, int | None, int | None]:
    """Strictly O_RDONLY / PROT_READ BAR0 probe for a CMP 170HX (zero MMIO writes).

    Returns ``(fuse_state, cur_sm, max_sm, cur_fbp, max_fbp)``. Never raises.
    """
    if driver in ("nvidia", "nouveau"):
        return f"{driver} bound (skipping BAR0 probe)", None, None, None, None
    loaded = _loaded_gpu_driver_modules(proc_modules_path)
    if loaded:
        mods = "/".join(sorted(loaded))
        return f"{mods} module loaded (skipping BAR0 probe)", None, None, None, None

    try:
        with (dev_dir / "config").open("rb") as f:
            cfg = f.read(8)
        if len(cfg) < 6:
            return "PCI config short read", None, None, None, None
        if struct.unpack_from("<I", cfg, 0)[0] == 0xFFFFFFFF:
            return "PCI link down (0xFFFFFFFF)", None, None, None, None
        cmd = struct.unpack_from("<H", cfg, 4)[0]
        if (cmd & 0x0002) == 0:
            return "PCI memory decode OFF", None, None, None, None
    except PermissionError:
        return "run as root for SM/FBP/fuse readout", None, None, None, None
    except OSError as exc:
        return f"PCI config unreadable ({exc.strerror or exc})", None, None, None, None

    res0 = dev_dir / "resource0"
    fd: int | None = None
    mm = None
    try:
        fd = os.open(res0, os.O_RDONLY | os.O_SYNC | getattr(os, "O_CLOEXEC", 0))
        mm = mmap.mmap(fd, BAR0_MAP_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
        ro = ReadOnlyMMIO(mm)

        # Read FUSECTRL FIRST and halt immediately on dead/poisoned BAR0.
        ctrl = ro.read32(REG_FUSECTRL)
        if ctrl == 0xFFFFFFFF or is_poison(ctrl):
            return f"BAR0 dead/poisoned (FUSECTRL=0x{ctrl:08x})", None, None, None, None

        dsov = ro.read32(REG_DSOV_STATUS)
        efp = ro.read32(REG_EFP_STATUS)
        crc = ro.read32(REG_CRC_STATUS)
        if any(v == 0xFFFFFFFF or is_poison(v) for v in (dsov, efp, crc)):
            return "BAR0 poisoned during health probe", None, None, None, None

        if dsov == 0 and efp == 0:
            fuse_state = "GOLDILOCKS (unsensed, EFP=0)"
        elif dsov == 0 and efp == 1:
            fuse_state = "GOLDILOCKS (sensed, EFP=1)"
        elif dsov != 0 and efp == 1:
            fuse_state = "COLD (locked, EFP=1)"
        else:
            fuse_state = f"DSOV={dsov} EFP={efp} CRC=0x{crc:02x}"

        if efp != 1 or crc != 0x55:
            return fuse_state, None, None, None, None

        gpc_cur = ro.read32(OPTION_REGS["GPC_DISABLE"])
        tpc_cur = [ro.read32(OPTION_REGS[f"TPC_GPC{i}_DISABLE"]) for i in range(TOTAL_GPC)]
        fbp_cur = ro.read32(OPTION_REGS["FBP_DISABLE"])

        gpc_def = ro.read32(FUSES_BY_KEY["gpc_disable"].defect_addr)
        tpc_def = [ro.read32(FUSES_BY_KEY[f"tpc_gpc{i}_disable"].defect_addr) for i in range(TOTAL_GPC)]
        fbp_def = ro.read32(FUSES_BY_KEY["fbp_disable"].defect_addr)

        all_vals = [gpc_cur, fbp_cur, gpc_def, fbp_def, *tpc_cur, *tpc_def]
        if any(v == 0xFFFFFFFF or is_poison(v) for v in all_vals):
            return fuse_state, None, None, None, None

        cur_sm = compute_sm_count(gpc_cur, tpc_cur)
        max_sm = compute_sm_count(gpc_def, tpc_def)
        cur_fbp = sum(1 for i in range(TOTAL_FBP) if not ((fbp_cur >> i) & 1))
        max_fbp = sum(1 for i in range(TOTAL_FBP) if not ((fbp_def >> i) & 1))
        return fuse_state, cur_sm, max_sm, cur_fbp, max_fbp
    except PermissionError:
        return "run as root for SM/FBP/fuse readout", None, None, None, None
    except OSError as exc:
        return f"BAR0 unreadable ({exc.strerror or exc})", None, None, None, None
    finally:
        if mm is not None:
            try:
                mm.close()
            except Exception:
                pass
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def discover_pci_gpus(
    *,
    sysfs_root: str | Path = "/sys/bus/pci/devices",
    probe_bar0: bool = True,
    pci_ids_paths: Sequence[str | Path] | None = None,
    proc_modules_path: Path = Path("/proc/modules"),
) -> list[DiscoveredGPU]:
    """Discover PCI display/3D GPUs from Linux sysfs (pure read-only).

    Lists CMP 170HX 8 GB (``10de:20c2``) devices first, followed by other NVIDIA GPUs,
    then any other PCI display controllers (``class 0x03xxxx``), each sorted by BDF.
    For safe CMP 170HX cards (not bound to ``nvidia``/``nouveau``), optionally performs
    a zero-write ``O_RDONLY`` BAR0 shadow probe for current/healthy SM & FBP counts and
    COLD/GOLDILOCKS fuse state.
    """
    root = Path(sysfs_root)
    if not root.is_dir():
        return []

    raw_entries: list[tuple[Path, str, str, str, str | None, str]] = []
    wanted_names: set[tuple[str, str]] = set()
    for dev_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not dev_dir.is_dir():
            continue
        cls_raw = _read_sysfs_text(dev_dir / "class") or ""
        cls_low = cls_raw.lower()
        if not cls_low.startswith("0x03"):
            continue
        vendor = _parse_hex_id(_read_sysfs_text(dev_dir / "vendor"))
        device = _parse_hex_id(_read_sysfs_text(dev_dir / "device"))
        if not vendor or not device:
            continue
        try:
            bdf = normalize_bdf(dev_dir.name)
        except BurnError:
            continue
        sub_v = _parse_hex_id(_read_sysfs_text(dev_dir / "subsystem_vendor"))
        sub_d = _parse_hex_id(_read_sysfs_text(dev_dir / "subsystem_device"))
        sub_id = f"{sub_v}:{sub_d}" if (sub_v and sub_d) else None
        pci_id = f"{vendor}:{device}"
        if pci_id not in _KNOWN_PCI_GPU_NAMES:
            wanted_names.add((vendor, device))
        raw_entries.append((dev_dir, bdf, pci_id, cls_low, sub_id, vendor))

    resolved_names = _lookup_pci_ids_names(wanted_names, pci_ids_paths=pci_ids_paths)

    gpus: list[DiscoveredGPU] = []
    for dev_dir, bdf, pci_id, cls_low, sub_id, vendor in raw_entries:
        drv_link = dev_dir / "driver"
        driver = drv_link.resolve().name if drv_link.exists() else "unbound"
        bar0_b, bar1_b = _parse_bar_sizes(_read_sysfs_text(dev_dir / "resource"))
        cur_spd = _read_sysfs_text(dev_dir / "current_link_speed")
        cur_wid = _read_sysfs_text(dev_dir / "current_link_width")
        max_spd = _read_sysfs_text(dev_dir / "max_link_speed")
        max_wid = _read_sysfs_text(dev_dir / "max_link_width")
        is_cmp = pci_id in SUPPORTED_CMP170HX_SKUS
        name = _KNOWN_PCI_GPU_NAMES.get(pci_id) or resolved_names.get(pci_id) or f"PCI GPU ({pci_id})"

        fuse_state: str | None = None
        cur_sm: int | None = None
        max_sm: int | None = None
        cur_fbp: int | None = None
        max_fbp: int | None = None
        if is_cmp and probe_bar0:
            fuse_state, cur_sm, max_sm, cur_fbp, max_fbp = _probe_cmp170hx_bar0_readonly(
                dev_dir, driver, proc_modules_path=proc_modules_path
            )

        gpus.append(
            DiscoveredGPU(
                bdf=bdf,
                pci_id=pci_id,
                subsystem_id=sub_id,
                pci_class=cls_low,
                name=name,
                driver=driver,
                bar0_bytes=bar0_b,
                bar1_bytes=bar1_b,
                current_link_speed=cur_spd,
                current_link_width=cur_wid,
                max_link_speed=max_spd,
                max_link_width=max_wid,
                is_cmp170hx=is_cmp,
                fuse_state=fuse_state,
                cur_sm=cur_sm,
                max_sm=max_sm,
                cur_fbp=cur_fbp,
                max_fbp=max_fbp,
            )
        )

    def _sort_key(g: DiscoveredGPU) -> tuple[int, str]:
        if g.is_cmp170hx:
            return (0, g.bdf)
        if g.pci_id.startswith("10de:"):
            return (1, g.bdf)
        return (2, g.bdf)

    gpus.sort(key=_sort_key)
    return gpus


def format_gpu_entry(gpu: DiscoveredGPU, *, index: int | None = None, current: bool = False) -> list[str]:
    """Format a ``DiscoveredGPU`` as 2 scannable terminal lines (IDs + memory/link/driver/state)."""
    prefix = f"  [{index:2d}]" if index is not None else "  "
    cur_mark = "*" if current else " "
    sub_str = f"  sub={gpu.subsystem_id}" if gpu.subsystem_id else ""
    tag = "" if gpu.is_cmp170hx else "  [not CMP 170HX]"
    line1 = f"{prefix}{cur_mark}{gpu.bdf}  {gpu.pci_id}  {gpu.name}{sub_str}{tag}"

    details: list[str] = []
    if gpu.bar1_bytes is not None:
        details.append(f"BAR1={format_bytes_iec(gpu.bar1_bytes)}")
    elif gpu.bar0_bytes is not None:
        details.append(f"BAR0={format_bytes_iec(gpu.bar0_bytes)}")

    if gpu.cur_fbp is not None and gpu.max_fbp is not None:
        details.append(f"HBM/FBPs={gpu.cur_fbp}/{gpu.max_fbp}")
    if gpu.cur_sm is not None and gpu.max_sm is not None:
        details.append(f"SMs={gpu.cur_sm}/{gpu.max_sm}")

    cur_spd = gpu.current_link_speed.removesuffix(" PCIe") if gpu.current_link_speed else None
    max_spd = gpu.max_link_speed.removesuffix(" PCIe") if gpu.max_link_speed else None
    if gpu.current_link_width and cur_spd:
        link = f"PCIe x{gpu.current_link_width} @ {cur_spd}"
        if max_spd and max_spd != cur_spd:
            link += f" (LnkCap {max_spd})"
        details.append(link)
    elif gpu.max_link_width and max_spd:
        details.append(f"PCIe LnkCap x{gpu.max_link_width} @ {max_spd}")

    details.append(f"drv={gpu.driver}")
    if gpu.fuse_state:
        details.append(gpu.fuse_state)

    indent = " " * (len(prefix) + 1)
    line2 = f"{indent}" + " | ".join(details)
    return [line1, line2]


def prompt_select_gpu_bdf(
    *,
    current_bdf: str | None = None,
    input_fn=input,
    show=print,
    sysfs_root: str | Path = "/sys/bus/pci/devices",
    probe_bar0: bool = True,
) -> str | None:
    """Display detected PCI GPUs (IDs, BAR1 memory, SM/FBP counts, PCIe link, driver, fuse state)
    and let the operator pick by number or type a BDF manually."""
    gpus = discover_pci_gpus(sysfs_root=sysfs_root, probe_bar0=probe_bar0)
    if gpus:
        show("\nDetected PCI GPUs:")
        for idx, gpu in enumerate(gpus, start=1):
            is_cur = bool(current_bdf and gpu.bdf == current_bdf)
            for line in format_gpu_entry(gpu, index=idx, current=is_cur):
                show(line)
        default_hint = f" [{current_bdf}]" if current_bdf else ""
        prompt = f"Select GPU [1-{len(gpus)}] or enter PCI BDF manually{default_hint} (blank to cancel): "
    else:
        show("\nNo PCI GPUs detected in /sys/bus/pci/devices.")
        default_hint = f" [{current_bdf}]" if current_bdf else ""
        prompt = f"CMP 170HX PCI BDF (e.g. 0000:05:00.0){default_hint}: "

    while True:
        try:
            raw = input_fn(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            return current_bdf
        if gpus and raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(gpus):
                picked = gpus[idx - 1]
                if not picked.is_cmp170hx:
                    show(
                        f"REFUSED: [{idx}] {picked.bdf} is {picked.pci_id} ({picked.name}), "
                        f"not a supported CMP 170HX ({' / '.join(SUPPORTED_CMP170HX_SKUS)})."
                    )
                    continue
                return picked.bdf
            show(f"Selection {idx} outside 1..{len(gpus)}.")
            continue
        return normalize_bdf(raw)


# ---------------------------------------------------------------------------
# Safe host driver unload / pci-stub bind / BAR0 memory-decode enable helper
# ---------------------------------------------------------------------------

_NVIDIA_UNLOAD_ORDER: tuple[str, ...] = (
    "nvidia_drm",
    "nvidia_modeset",
    "nvidia_uvm",
    "nvidia_peermem",
    "nvidia",
    "nouveau",
)


def _loaded_forbidden_modules(proc_modules_path: Path = Path("/proc/modules")) -> list[str]:
    """Return loaded ``nvidia*`` / ``nouveau`` modules in safe reverse-dependency unload order."""
    loaded: set[str] = set()
    try:
        text = proc_modules_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        if name == "nvidia" or name.startswith("nvidia_") or name == "nouveau":
            loaded.add(name)
    ordered = [m for m in _NVIDIA_UNLOAD_ORDER if m in loaded]
    # Any additional nvidia_* modules not in the standard list come before core 'nvidia'.
    extras = sorted(m for m in loaded if m not in ordered)
    if extras:
        if "nvidia" in ordered:
            idx = ordered.index("nvidia")
            ordered = ordered[:idx] + extras + ordered[idx:]
        else:
            ordered = extras + ordered
    return ordered


def inspect_gpu_driver_blockers(
    bdf: str,
    *,
    sysfs_root: str | Path = "/sys/bus/pci/devices",
    proc_modules_path: Path = Path("/proc/modules"),
) -> tuple[str | None, list[str]]:
    """Return ``(bound_driver, loaded_forbidden_modules)`` for ``bdf`` without opening BAR0."""
    norm = normalize_bdf(bdf)
    dev_dir = Path(sysfs_root) / norm
    drv_link = dev_dir / "driver"
    bound = drv_link.resolve().name if drv_link.exists() else None
    loaded = _loaded_forbidden_modules(proc_modules_path)
    return bound, loaded


def find_nvidia_open_processes(proc_root: Path = Path("/proc")) -> list[tuple[int, str, str]]:
    """Find processes holding open ``/dev/nvidia*`` handles via ``/proc/<pid>/fd`` (zero subprocesses)."""
    holders: list[tuple[int, str, str]] = []
    if not proc_root.is_dir():
        return holders
    seen: set[tuple[int, str]] = set()
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return holders
    for pdir in entries:
        if not pdir.name.isdigit():
            continue
        pid = int(pdir.name)
        fd_dir = pdir / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        comm = _read_sysfs_text(pdir / "comm") or f"pid-{pid}"
        for fd_link in fds:
            try:
                target = os.readlink(fd_link)
            except OSError:
                continue
            if target.startswith("/dev/nvidia"):
                key = (pid, target)
                if key not in seen:
                    seen.add(key)
                    holders.append((pid, comm, target))
    holders.sort(key=lambda item: (item[0], item[2]))
    return holders


def unload_gpu_drivers(
    bdf: str,
    *,
    show=print,
    runner=None,
    sysfs_root: str | Path = "/sys/bus/pci/devices",
    pci_bus_root: str | Path = "/sys/bus/pci",
    proc_modules_path: Path = Path("/proc/modules"),
    proc_root: Path = Path("/proc"),
    check_root: bool = True,
) -> None:
    """Unload ``nvidia`` / ``nouveau`` kernel modules, bind ``bdf`` to ``pci-stub``, and
    re-enable PCI memory decode (``COMMAND`` bit 1) so BAR0 is safely readable.

    Never kills user processes automatically; if a module is in use, lists the exact PIDs
    holding ``/dev/nvidia*`` so the operator can stop them cleanly.
    """
    norm = normalize_bdf(bdf)
    if check_root and os.geteuid() != 0:
        raise BurnError(
            "root privileges are required to unload nvidia/nouveau drivers — re-run with sudo"
        )

    def _default_run(cmd: list[str]) -> tuple[int, str, str]:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

    run = runner or _default_run

    # 1. Unload loaded nvidia*/nouveau kernel modules in reverse-dependency order.
    loaded = _loaded_forbidden_modules(proc_modules_path)
    for mod in loaded:
        show(f"  Running: modprobe -r {mod}")
        rc, _out, err = run(["modprobe", "-r", mod])
        if rc != 0:
            holders = find_nvidia_open_processes(proc_root)
            detail_lines = [f"failed to unload kernel module '{mod}' (exit {rc}): {err or 'in use'}"]
            if holders:
                detail_lines.append("Processes currently holding /dev/nvidia*:")
                seen_comms: set[str] = set()
                for pid, comm, dev in holders[:12]:
                    detail_lines.append(f"  - pid {pid} ({comm}): {dev}")
                    seen_comms.add(comm)
                if "nvidia-persiste" in seen_comms or "nvidia-persistenced" in seen_comms:
                    detail_lines.append("Hint: stop nvidia-persistenced first: sudo systemctl stop nvidia-persistenced")
            else:
                detail_lines.append(
                    "Hint: stop any display manager / container / nvidia-persistenced using NVIDIA GPUs, then retry."
                )
            raise BurnError("\n".join(detail_lines))

    still_loaded = _loaded_forbidden_modules(proc_modules_path)
    if still_loaded:
        raise BurnError(
            f"nvidia/nouveau modules are still loaded after modprobe -r: {', '.join(still_loaded)}"
        )

    # 2. Ensure target BDF is unbound from any forbidden driver and bound to pci-stub (best effort).
    dev_dir = Path(sysfs_root) / norm
    if dev_dir.is_dir():
        drv_link = dev_dir / "driver"
        bound = drv_link.resolve().name if drv_link.exists() else None
        if bound not in (None, "pci-stub"):
            unbind_path = drv_link / "unbind"
            if unbind_path.exists():
                show(f"  Unbinding {norm} from {bound} ...")
                unbind_path.write_text(f"{norm}\n", encoding="ascii")

        # Try loading pci_stub and surgically binding only this BDF via driver_override + drivers_probe.
        run(["modprobe", "pci_stub"])
        bus_root = Path(pci_bus_root)
        pci_stub_dir = bus_root / "drivers" / "pci-stub"
        drv_override = dev_dir / "driver_override"
        drivers_probe = bus_root / "drivers_probe"
        if pci_stub_dir.is_dir() and drv_override.exists() and drivers_probe.exists():
            try:
                drv_override.write_text("pci-stub\n", encoding="ascii")
                drivers_probe.write_text(f"{norm}\n", encoding="ascii")
                show(f"  Bound {norm} to pci-stub.")
            except OSError as exc:
                show(f"  NOTE: could not bind {norm} to pci-stub ({exc}); leaving unbound.")

        # 3. Unloading nvidia often clears PCI COMMAND bit 1 (Memory Space Enable); re-enable if off.
        cfg_path = dev_dir / "config"
        enable_path = dev_dir / "enable"
        try:
            with cfg_path.open("rb") as f:
                cfg = f.read(6)
            if len(cfg) >= 6:
                cmd_val = struct.unpack_from("<H", cfg, 4)[0]
                if (cmd_val & 0x0002) == 0 and enable_path.exists():
                    show(f"  PCI memory decode is off (COMMAND=0x{cmd_val:04x}); enabling {norm} ...")
                    enable_path.write_text("1\n", encoding="ascii")
        except OSError:
            pass


def ensure_gpu_detached_interactive(
    bdf: str,
    *,
    input_fn=input,
    show=print,
    sysfs_root: str | Path = "/sys/bus/pci/devices",
    pci_bus_root: str | Path = "/sys/bus/pci",
    proc_modules_path: Path = Path("/proc/modules"),
    proc_root: Path = Path("/proc"),
    runner=None,
    check_root: bool = True,
) -> bool:
    """Check if ``nvidia``/``nouveau`` blocks safe BAR0 access on ``bdf`` and, if so, offer to
    unload the driver cleanly after explicit operator confirmation.

    Returns ``True`` when the host is clear to proceed, or ``False`` if the operator declined.
    """
    norm = normalize_bdf(bdf)
    bound, loaded = inspect_gpu_driver_blockers(
        norm, sysfs_root=sysfs_root, proc_modules_path=proc_modules_path
    )
    forbidden_bound = bound not in (None, "pci-stub")
    if not forbidden_bound and not loaded:
        return True

    show(f"\nGPU {norm} cannot be accessed safely while nvidia/nouveau is active:")
    if forbidden_bound:
        show(f"  - Bound driver   : {bound}")
    if loaded:
        show(f"  - Loaded modules : {', '.join(loaded)}")
    if check_root and os.geteuid() != 0:
        raise BurnError(
            "root privileges are required to unload nvidia/nouveau and access PCI BAR0 — re-run with sudo"
        )

    show("  NOTE: This unloads nvidia/nouveau for ALL NVIDIA GPUs on this host until reloaded or rebooted.")
    try:
        ans = input_fn(
            f"Unload nvidia/nouveau now, bind {norm} to pci-stub, and enable BAR0 decode? [y/N]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if ans not in ("y", "yes"):
        return False

    unload_gpu_drivers(
        norm,
        show=show,
        runner=runner,
        sysfs_root=sysfs_root,
        pci_bus_root=pci_bus_root,
        proc_modules_path=proc_modules_path,
        proc_root=proc_root,
        check_root=check_root,
    )
    return True


