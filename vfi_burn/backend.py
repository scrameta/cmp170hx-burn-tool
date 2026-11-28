"""Backend boundary, writer-capability guards, fake controller, and private JSON I/O."""
from __future__ import annotations

from pathlib import Path
import importlib
import json
import os
import secrets

from .interfaces import FuseReader, FuseWriter, Health, Host
from .model import BurnError, PreTriggerRefusal, require_private_file, u32

# Backward-compatible interface aliases.
Reader = FuseReader
Writer = FuseWriter

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


class MissingWriter(FuseWriter):
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


class FakeController(FuseReader, FuseWriter):
    """In-memory simulation only; cannot access BAR0 or physical fuses."""

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
            0xE0040000, 0, 0x55, 0xFFFFFFFF, 0xFFFFFFFF, 0
        )
        self.machine = host or Host(
            "pci-stub", False, False, "[none]", True, "fake-boot-1"
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
    """Atomically create (or replace when overwrite=True) a mode-0600 JSON file inside a mode-0700 directory."""
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
    if p.is_dir():
        raise BurnError(f"output path is a directory: {p}")
    if not overwrite:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(p, flags, 0o600)
        except FileExistsError as exc:
            raise BurnError(
                f"output file already exists (refusing to overwrite): {p} "
                "(pass --overwrite to replace)"
            ) from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            p.unlink(missing_ok=True)
            raise
        return

    # Atomic overwrite: stage in a mode-0600 sibling temp file, fsync, then os.replace.
    tmp_path: Path | None = None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(64):
        cand = p.parent / f".{p.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
        try:
            fd = os.open(cand, flags, 0o600)
            tmp_path = cand
            break
        except FileExistsError:
            continue
    if tmp_path is None:
        raise BurnError(f"could not allocate temporary file in {p.parent}")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if p.is_symlink():
            raise BurnError(f"refusing symlink output: {p}")
        os.replace(tmp_path, p)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def next_available_path(path: str | Path) -> str:
    """Return a non-existing sibling path by appending _2, _3, ... before the file suffix."""
    p = Path(path)
    n = 2
    while True:
        cand = p.parent / f"{p.stem}_{n}{p.suffix}"
        if not cand.exists() and not cand.is_symlink():
            return str(cand)
        n += 1


def write_private_json_interactive(
    path: str | Path,
    data: dict,
    *,
    input_fn,
    show,
    path_prompt: str = "Output path",
    describe_existing=None,
) -> str | None:
    """Write a private mode-0600 JSON file, prompting on collision instead of hard-failing.

    Returns the written path string, or ``None`` if the operator cancels.
    Never overwrites without an explicit affirmative choice; symlinks remain refused.
    """
    current = str(path)
    while True:
        p = Path(current)
        if p.is_symlink():
            raise BurnError(f"refusing symlink output: {p}")
        if not p.exists():
            try:
                write_private_json(current, data, overwrite=False)
                return current
            except BurnError as exc:
                if not p.exists():
                    raise
                show(f"NOTE: {exc}")
        if p.is_dir():
            show(f"Output path is a directory, not a file: {current}")
            try:
                alt = input_fn(f"{path_prompt} (blank to cancel): ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if not alt:
                return None
            current = alt
            continue

        if callable(describe_existing):
            note = describe_existing(p)
            if note:
                show(note)
        while True:
            try:
                choice = input_fn(
                    f"File already exists: {current} — [o]verwrite, [n]ew path, or [c]ancel [c]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if choice in ("o", "overwrite", "y", "yes"):
                write_private_json(current, data, overwrite=True)
                return current
            if choice in ("n", "new", "r", "rename"):
                suggested = next_available_path(current)
                try:
                    raw = input_fn(f"{path_prompt} [{suggested}]: ").strip()
                except (EOFError, KeyboardInterrupt):
                    return None
                current = raw if raw else suggested
                break
            if choice in ("", "c", "cancel"):
                return None
            show("Please enter 'o' (overwrite), 'n' (new path), or 'c' (cancel).")


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
            k: (None if v is None else u32(v, f"health.{k}"))
            for k, v in health_raw.items()
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
