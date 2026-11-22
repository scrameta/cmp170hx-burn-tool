"""Backend boundary. No real fuse writer exists in this package."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
import importlib
import json
import os

from .model import BurnError, PreTriggerRefusal, require_private_file, u32

REQUIRED_READER_METHODS = (
    "pci_id",
    "die_row",
    "field",
    "fuse_row",
    "register",
    "health",
    "host",
    "active_sense_readonly",
)


@dataclass
class Health:
    fusectrl: int
    dsov: int
    crc: int
    plm: int
    plm_f8: int
    status_word: int
    vout_mv: float
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
    bound_driver: str | None
    nvidia_loaded: bool
    nouveau_loaded: bool
    lockdown: str
    vrm_tainted: bool
    hold_active: bool
    boot_id: str

    def serializable(self) -> dict:
        return asdict(self)


class Reader(Protocol):
    """Adapter functions MUST do fresh hardware reads; cached hold logs are forbidden."""

    def pci_id(self) -> str: ...
    def die_row(self, index: int) -> int: ...
    def field(self, name: str) -> int: ...
    def fuse_row(self, index: int) -> int: ...
    def register(self, address: int) -> int: ...
    def health(self) -> Health: ...
    def host(self) -> Host: ...
    def active_sense_readonly(self) -> None: ...


class Writer(Protocol):
    def write_one_record(self, row: int, word: int) -> None: ...


class MissingWriter:
    """Mandatory default for physical hardware; refuses any write attempt."""

    write_enabled = False
    writer_verified = False

    def write_one_record(self, row: int, word: int) -> None:
        raise PreTriggerRefusal(
            "REAL FUSE WRITES DISABLED: write mode not armed (pass --execute on an interactive TTY)"
        )


def has_write_capability(adapter) -> bool:
    """Return True if a live adapter has write capability armed (write_enabled or legacy writer_verified).

    NOTE: This indicates write mode/capability was explicitly armed by the caller or custom adapter,
    NOT that the hardware write sequence has been independently bench-certified.
    """
    if adapter is None or isinstance(adapter, MissingWriter):
        return False
    if getattr(adapter, "simulation", False):
        return False
    armed = (
        getattr(adapter, "write_enabled", False) is True
        or getattr(adapter, "writer_verified", False) is True
    )
    if not armed:
        return False
    return callable(getattr(adapter, "write_one_record", None))


# Backward-compatible alias for callers/adapters importing has_verified_writer.
has_verified_writer = has_write_capability


class FakeController:
    """In-memory simulation only; cannot access BAR0, PMBus, or physical fuses."""

    simulation = True
    write_enabled = False
    writer_verified = False

    def __init__(
        self,
        *,
        pci_id: str = "10de:20c2",
        die: dict[int, int] | None = None,
        fields: dict[str, int] | None = None,
        rows: dict[int, int] | None = None,
        health: Health | None = None,
        host: Host | None = None,
        registers: dict[int, int] | None = None,
    ) -> None:
        self._pci_id = pci_id
        self.die = dict(die) if die is not None else {i: 0x1000 + i for i in range(8, 16)}
        self.fields = dict(fields) if fields is not None else {}
        self.rows = dict(rows) if rows is not None else {}
        self.regs = dict(registers) if registers is not None else {}
        self.state = health or Health(
            0xE0040000, 0, 0x55, 0xFFFFFFFF, 0xFFFFFFFF, 0, 787.5, 0
        )
        self.machine = host or Host(
            "pci-stub", False, False, "[none]", False, True, "fake-boot-1"
        )
        self.events: list[tuple] = []
        self.after_sense = None
        self.after_write = None
        self.corrupt_readback = False
        self.fail_write = False

    def pci_id(self) -> str:
        return self._pci_id

    def die_row(self, index: int) -> int:
        if index not in self.die:
            raise BurnError(f"no fake die row {index}")
        return self.die[index]

    def field(self, name: str) -> int:
        if name not in self.fields:
            raise BurnError(f"no fake field {name!r}")
        return self.fields[name]

    def fuse_row(self, index: int) -> int:
        if index not in self.rows:
            raise BurnError(f"no fake fuse row {index}")
        value = self.rows[index]
        if self.corrupt_readback and self.events and self.events[-1][0] == "write":
            return value ^ 1
        return value

    def register(self, address: int) -> int:
        if address in self.regs:
            return self.regs[address]
        mapping = {0x00820000: "fusectrl", 0x008200F8: "plm_f8"}
        if address in mapping:
            return getattr(self.state, mapping[address])
        raise BurnError(f"no fake register 0x{address:08x}")

    def health(self) -> Health:
        return Health(**self.state.serializable())

    def host(self) -> Host:
        return Host(**self.machine.serializable())

    def active_sense_readonly(self) -> None:
        self.events.append(("sense",))
        if self.after_sense:
            self.after_sense(self)

    def write_one_record(self, row: int, word: int) -> None:
        self.events.append(("write", row, word))
        if self.fail_write:
            raise BurnError("injected write interruption")
        self.rows[row] = word
        if self.after_write:
            self.after_write(self)


def load_adapter(spec: str):
    """Import a local owner-supplied adapter module:factory.

    Validates required Reader methods and refuses simulation adapters.
    Write capability is never granted unless the adapter explicitly implements
    Writer with ``writer_verified = True`` AND the caller invokes ``--execute``.
    """
    if ":" not in spec:
        raise BurnError("adapter must be formatted as module:factory")
    module_name, factory_name = spec.split(":", 1)
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        raise BurnError(f"failed to import adapter module {module_name!r}: {exc}") from exc
    factory = getattr(mod, factory_name, None)
    if not callable(factory):
        raise BurnError(f"adapter factory {spec!r} not found or not callable")
    obj = factory()
    if getattr(obj, "simulation", False):
        raise BurnError("fake adapters must use the built-in simulator (--fake), not --adapter")
    missing = [m for m in REQUIRED_READER_METHODS if not callable(getattr(obj, m, None))]
    if missing:
        raise BurnError(f"adapter {spec!r} missing required Reader methods: {', '.join(missing)}")
    return obj


def write_private_json(path: str | Path, data: dict, *, overwrite: bool = False) -> None:
    """Atomically create a private mode-0600 JSON file inside a mode-0700 directory."""
    p = Path(path)
    if p.is_symlink():
        raise BurnError(f"refusing symlink output: {p}")
    if p.parent.is_symlink():
        raise BurnError(f"refusing symlink parent directory: {p.parent}")
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if p.parent.stat().st_mode & 0o077:
        raise BurnError(
            f"private output directory must have mode 0700: {p.parent} — run: chmod 700 {p.parent}"
        )
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_TRUNC if overwrite else os.O_EXCL
    try:
        fd = os.open(p, flags, 0o600)
    except FileExistsError as exc:
        raise BurnError(f"output file already exists (refusing to overwrite): {p}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        if not overwrite:
            p.unlink(missing_ok=True)
        raise


def snapshot_data(ctl: FakeController) -> dict:
    return {
        "pci_id": ctl.pci_id(),
        "die_rows": ctl.die,
        "fields": ctl.fields,
        "rows": ctl.rows,
        "health": ctl.state.serializable(),
        "host": ctl.machine.serializable(),
        "registers": {hex(k): v for k, v in ctl.regs.items()},
    }


def fake_from_file(path: str | Path) -> FakeController:
    p = require_private_file(path, "fake snapshot")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BurnError(f"invalid JSON in fake snapshot {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise BurnError("fake snapshot root must be a JSON object")
    try:
        health_raw = data["health"]
        health_kwargs = {
            **health_raw,
            # Optional Health fields are saved as JSON null by --save-fake; keep them None.
            **{
                k: (None if v is None else u32(v, f"health.{k}"))
                for k, v in health_raw.items()
                if k != "vout_mv"
            },
        }
        return FakeController(
            pci_id=str(data["pci_id"]),
            die={int(k): u32(v, f"die_rows.{k}") for k, v in data["die_rows"].items()},
            fields={str(k): u32(v, f"fields.{k}") for k, v in data["fields"].items()},
            rows={int(k): u32(v, f"rows.{k}") for k, v in data["rows"].items()},
            health=Health(**health_kwargs),
            host=Host(**data["host"]),
            registers={
                int(k, 0): u32(v, f"registers.{k}")
                for k, v in data.get("registers", {}).items()
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BurnError(f"malformed fake snapshot {p}: {exc}") from exc


def dump_fake(ctl: FakeController, path: str | Path, *, overwrite: bool = False) -> None:
    write_private_json(path, snapshot_data(ctl), overwrite=overwrite)
