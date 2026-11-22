"""Safe aligned 32-bit BAR0 MMIO helpers.

Enforces BURN_SPEC safety invariant #3:
Every MMIO access is a single aligned 32-bit load/store via
``memoryview(buf)[offset:offset+4].cast("I")``, NEVER ``struct.pack_into``.
Memoryviews are released immediately so underlying ``mmap`` objects can be closed cleanly.
"""
from __future__ import annotations

from .model import BurnError

REG_FUSECTRL = 0x00820000
REG_DEBUGCTRL = 0x00820060
REG_PLM_F8 = 0x008200F8

KNOWN_REGISTERS: dict[str, int] = {
    "FUSECTRL": REG_FUSECTRL,
    "DEBUGCTRL": REG_DEBUGCTRL,
    "PLM_F8": REG_PLM_F8,
}


def resolve_register_address(spec: str | int) -> int:
    """Resolve a known register name, experimental GA100 register name, or hex/decimal address."""
    if isinstance(spec, bool):
        raise BurnError("register address cannot be a boolean")
    if isinstance(spec, int):
        addr = spec
    elif isinstance(spec, str):
        key = spec.strip().upper()
        if key in KNOWN_REGISTERS:
            addr = KNOWN_REGISTERS[key]
        else:
            from .ga100 import EXPERIMENTAL_REGS_BY_NAME

            if key in EXPERIMENTAL_REGS_BY_NAME:
                addr = EXPERIMENTAL_REGS_BY_NAME[key]
            else:
                try:
                    addr = int(key, 0)
                except ValueError as exc:
                    known = ", ".join(sorted(KNOWN_REGISTERS))
                    raise BurnError(
                        f"unknown register {spec!r}; use a hex address, one of: {known}, "
                        "or an experimental GA100 register (e.g. FUSE_ECC_EN, FUSE_SS_DP, FEAT_READOUT_0)"
                    ) from exc
    else:
        raise BurnError(f"invalid register specifier: {spec!r}")
    if addr < 0 or addr & 3:
        raise BurnError(f"register address 0x{addr:x} must be non-negative and 4-byte aligned")
    return addr


class ReadOnlyMMIO:
    """Read-only 32-bit aligned MMIO wrapper. Physically cannot issue MMIO writes."""

    def __init__(self, buf) -> None:
        self._buf = buf
        self._size = len(buf)

    @property
    def size(self) -> int:
        return self._size

    def _validate_offset(self, offset: int) -> None:
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise BurnError(f"MMIO offset must be an integer, got {type(offset).__name__}")
        if offset < 0 or (offset & 3):
            raise BurnError(f"MMIO offset 0x{offset:x} is not 4-byte aligned")
        if offset + 4 > self._size:
            raise BurnError(
                f"MMIO offset 0x{offset:08x} exceeds mapping size 0x{self._size:08x}"
            )

    def read32(self, offset: int) -> int:
        self._validate_offset(offset)
        base = memoryview(self._buf)
        try:
            sliced = base[offset : offset + 4]
            try:
                word_view = sliced.cast("I")
                try:
                    return int(word_view[0])
                finally:
                    word_view.release()
            finally:
                sliced.release()
        finally:
            base.release()


class AlignedMMIO(ReadOnlyMMIO):
    """Read/write 32-bit aligned MMIO wrapper with single-store and DEBUGCTRL guards.

    Uses ``memoryview(buf)[offset:offset+4].cast("I")[0] = value`` to emit a single
    aligned 32-bit store without zeroing the register first.
    """

    def write32(self, offset: int, value: int, *, allow_debugctrl: bool = False) -> None:
        self._validate_offset(offset)
        if isinstance(value, bool) or not isinstance(value, int) or not (0 <= value <= 0xFFFFFFFF):
            raise BurnError(f"MMIO write value out of unsigned 32-bit range: {value!r}")
        if offset == REG_DEBUGCTRL and not allow_debugctrl:
            raise BurnError(
                "REFUSED implicit write to DEBUGCTRL (0x00820060); "
                "requires explicit allow_debugctrl=True after separate confirmation"
            )
        base = memoryview(self._buf)
        try:
            sliced = base[offset : offset + 4]
            try:
                word_view = sliced.cast("I")
                try:
                    word_view[0] = value
                finally:
                    word_view.release()
            finally:
                sliced.release()
        finally:
            base.release()
