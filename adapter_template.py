"""OWNER TEMPLATE ONLY: intentionally cannot talk to hardware as supplied.

How to use:
1. Copy this file to your private bench directory (outside git).
2. Implement ``MyLiveReader`` using ``ReadOnlyMMIO`` (from ``vfi_burn.mmio``) and your
   measured/verified read-only BAR0 paths from ``cmp170hx-8gb-vfi-tools``.
3. Test read-only inspection first:
      vfi-burn inspect --plan private/plan.json --adapter my_adapter:create_reader
      vfi-burn inspect --plan private/plan.json --adapter my_adapter:create_reader --sense
      vfi-burn manual --plan private/plan.json --adapter my_adapter:create_reader \
          --journal private/session.jsonl console
4. ONLY after you have independently verified FF record encoding, the FUSECTRL WRITE
   sequence, programming voltage, and FPF empty marker on your bench, implement
   ``MyLiveWriter.write_one_record`` using ``AlignedMMIO`` (single aligned 32-bit
   stores via memoryview.cast('I') — NEVER struct.pack_into) and set
   ``writer_verified = True``.
"""
from __future__ import annotations

from vfi_burn.interfaces import FuseReader, FuseWriter, Health, Host
from vfi_burn.mmio import AlignedMMIO, ReadOnlyMMIO


class MyLiveReader(FuseReader):
    """Read-only hardware adapter. Cannot invoke physical fuse writes."""

    simulation = False
    writer_verified = False

    def __init__(self, bar0_mmap=None) -> None:
        # Wrap BAR0 mmap in ReadOnlyMMIO so reads are always 4-byte aligned and
        # this class physically cannot issue MMIO writes.
        self._mmio = ReadOnlyMMIO(bar0_mmap) if bar0_mmap is not None else None

    def pci_id(self) -> str:
        raise NotImplementedError("read PCI ID 10de:20c2 from current bound sysfs device")

    def host(self) -> Host:
        raise NotImplementedError(
            "fresh reads: sysfs bound driver, /proc/modules (nvidia/nouveau), "
            "/sys/kernel/security/lockdown, held-frame flag, /proc/sys/kernel/random/boot_id"
        )

    def health(self) -> Health:
        # CRITICAL ORDER: read FUSECTRL FIRST, then DSOV, CRC, PLM,
        # PLM_f8, and EN_SW.
        raise NotImplementedError("fresh BAR0 register reads (FUSECTRL first)")

    def die_row(self, index: int) -> int:
        raise NotImplementedError("verified non-programming fuse row read (rows 8..15)")

    def field(self, name: str) -> int:
        raise NotImplementedError("read live decoded mask/PCIe status for this boot")

    def fuse_row(self, index: int) -> int:
        raise NotImplementedError("verified read-only FPF row access")

    def register(self, address: int) -> int:
        if self._mmio is not None:
            return self._mmio.read32(address)
        if address % 4:
            raise ValueError("MMIO register is not 32-bit aligned")
        raise NotImplementedError("32-bit aligned MMIO READ only")

    def active_sense_readonly(self) -> None:
        raise NotImplementedError("must prove non-committing active sense on the bench")


class MyLiveWriter(MyLiveReader, FuseWriter):
    """Opt-in live writer template. Disabled until writer_verified = True."""

    # DO NOT flip to True until FF encoding, FUSECTRL write opcodes, burn voltage,
    # and FPF empty marker have been independently verified on your hardware.
    writer_verified = False

    def __init__(self, bar0_mmap=None) -> None:
        super().__init__(bar0_mmap=bar0_mmap)
        # AlignedMMIO enforces single aligned 32-bit stores and blocks implicit DEBUGCTRL writes.
        self._rw_mmio = AlignedMMIO(bar0_mmap) if bar0_mmap is not None else None

    def write_one_record(self, row: int, word: int) -> None:
        raise NotImplementedError(
            "owner must implement verified single-record fuse programming sequence using self._rw_mmio.write32()"
        )


def create_reader() -> MyLiveReader:
    return MyLiveReader()


def create_writer() -> MyLiveWriter:
    return MyLiveWriter()
