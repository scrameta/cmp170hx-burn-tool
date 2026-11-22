"""Abstract base classes (ABC) defining clean interface/implementation boundaries.

Every hardware, HAL, MMIO, and simulator layer implements these explicit ABCs:
  - ``MMIOReadBus`` / ``MMIOBus``: aligned 32-bit BAR0 PCIe transport interface
  - ``FuseControllerHAL``: low-level fuse request -> PCIe BAR0 protocol interface
  - ``FuseReader`` / ``FuseWriter``: high-level card adapter interface consumed by the engine
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass
class Health:
    """Point-in-time snapshot of GA100 fuse-controller health and arming registers."""

    fusectrl: int
    dsov: int
    crc: int
    plm: int
    plm_f8: int
    en_sw: int = 0
    debugctrl: int = 0
    efp: int | None = None
    dfp: int | None = None
    plm_f4: int | None = None
    sbfe: int | None = None
    wdata_wr_secure: int | None = None
    resense_disable: int | None = None

    def serializable(self) -> dict:
        return asdict(self)


@dataclass
class Host:
    """Point-in-time snapshot of Linux host safety state for the target PCI GPU."""

    bound_driver: str | None
    nvidia_loaded: bool
    nouveau_loaded: bool
    lockdown: str
    hold_active: bool
    boot_id: str

    def serializable(self) -> dict:
        return asdict(self)


class MMIOReadBus(ABC):
    """Read-only 32-bit aligned PCIe BAR0 MMIO transport contract."""

    @abstractmethod
    def read32(self, offset: int) -> int:
        """Issue one aligned 32-bit MMIO read at BAR0 ``offset``."""


class MMIOBus(MMIOReadBus):
    """Read/write 32-bit aligned PCIe BAR0 MMIO transport contract."""

    @abstractmethod
    def write32(self, offset: int, value: int, *, allow_debugctrl: bool = False) -> None:
        """Issue one aligned 32-bit MMIO write at BAR0 ``offset``."""


class FuseControllerHAL(ABC):
    """Low-level Fuse HAL: translates fuse operations into PCIe BAR0 MMIO transactions."""

    @abstractmethod
    def rd(self, offset: int) -> int:
        """Read one 32-bit BAR0 register."""

    @abstractmethod
    def poll_idle(self, timeout_s: float = 0.2) -> int:
        """Poll FUSECTRL until IDLE (state=4) or raise BurnError on poison/timeout."""

    @abstractmethod
    def read_health_registers(self) -> Health:
        """Read health/arming registers in safe order (FUSECTRL first, stopping on poison)."""

    @abstractmethod
    def read_row(self, row: int, *, timeout_s: float = 0.2) -> int:
        """Read one 32-bit fuse array row via CMD_READ."""

    @abstractmethod
    def sense_ctrl(self) -> None:
        """Issue Goldilocks SENSE_CTRL (EFP 0->1 while preserving DSOV=0)."""

    @abstractmethod
    def armed_problems(self) -> list[str]:
        """Return [] when all programming prerequisites hold, else list every failing check."""

    @abstractmethod
    def arm(self) -> None:
        """Open volatile shadow gates and verify all armed invariants."""

    @abstractmethod
    def program_row(
        self,
        row: int,
        word: int,
        *,
        timeout_s: float = 10.0,
        pre_trigger_check=None,
    ) -> int:
        """Burn one 32-bit FF record into a blank FF row and verify readback."""


class FuseReader(ABC):
    """High-level read-only GPU adapter contract consumed by the safety engine.

    Implementations MUST perform fresh hardware (or simulator) reads; cached logs are forbidden.
    """

    simulation: bool = False

    @abstractmethod
    def pci_id(self) -> str:
        """Return PCI vendor:device ID (e.g. '10de:20c2')."""

    @abstractmethod
    def die_row(self, index: int) -> int:
        """Read one cold Die ID row (8..15)."""

    @abstractmethod
    def field(self, name: str) -> int:
        """Read a named option/status/PCIe field (e.g. 'GPC_DISABLE', 'LnkCap2', 'SM_COUNT')."""

    @abstractmethod
    def fuse_row(self, index: int) -> int:
        """Read one 32-bit fuse row via CMD_READ."""

    @abstractmethod
    def register(self, address: int) -> int:
        """Read one 32-bit aligned BAR0 register."""

    @abstractmethod
    def health(self) -> Health:
        """Return current fuse-controller health registers."""

    @abstractmethod
    def host(self) -> Host:
        """Return current host safety state."""

    @abstractmethod
    def active_sense_readonly(self) -> None:
        """Run non-committing active sense (SENSE_CTRL)."""

    def close(self) -> None:
        """Release any held OS resources (mmap, file descriptors, locks)."""

    def __enter__(self) -> "FuseReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()


class FuseWriter(ABC):
    """Single-record fuse programming contract."""

    write_enabled: bool = False
    writer_verified: bool = False

    @abstractmethod
    def write_one_record(self, row: int, word: int) -> None:
        """Program one 32-bit FF word into ``row``."""
