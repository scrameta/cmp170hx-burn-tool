"""Fail-closed fuse-controller state machine. Hardware programming is NOT implemented."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import math
import os
import stat

from .backend import Health, Host, MissingWriter
from .model import (
    BurnError,
    Plan,
    Record,
    is_debugctrl_target,
    is_poison,
    require_private_file,
    u32,
)

FUSECTRL_IDLE = 0xE0040000
GOLDILOCKS_CRC = 0x55
GOLDILOCKS_PLM = 0xFFFFFFFF
LATCH_STATUS_WORD = 0x2042
HARD_VOUT_MIN_MV = 400.0
HARD_VOUT_MAX_MV = 1500.0

# Standard PMBus STATUS_WORD bit definitions (high byte + low byte).
PMBUS_STATUS_BITS: tuple[tuple[int, str], ...] = (
    (0x8000, "VOUT"),
    (0x4000, "IOUT/POUT"),
    (0x2000, "INPUT"),
    (0x1000, "MFR"),
    (0x0800, "POWER_GOOD#"),
    (0x0400, "FANS"),
    (0x0200, "OTHER"),
    (0x0100, "UNKNOWN"),
    (0x0080, "BUSY"),
    (0x0040, "OFF"),
    (0x0020, "VOUT_OV"),
    (0x0010, "IOUT_OC"),
    (0x0008, "VIN_UV"),
    (0x0004, "TEMP"),
    (0x0002, "CML"),
    (0x0001, "NONE_OF_THE_ABOVE"),
)

# Critical PMBus fault bits that always indicate VRM latch/off/fault (never overrideable).
PMBUS_CRITICAL_FAULT_MASK = 0x8840  # VOUT | POWER_GOOD# | OFF


def hex32(value: int) -> str:
    return f"0x{value:08x}"


def decode_pmbus_status_word(status_word: int) -> list[str]:
    return [name for mask, name in PMBUS_STATUS_BITS if status_word & mask]


def format_pmbus_status(status_word: int) -> str:
    flags = decode_pmbus_status_word(status_word)
    suffix = f" ({' | '.join(flags)})" if flags else ""
    return f"0x{status_word:04x}{suffix}"


@dataclass(frozen=True)
class Overrides:
    """Explicit, granular manual overrides. All default to False/None (fail-closed).

    Hard brick-prevention invariants (aligned 32-bit stores, empty target slot,
    no GPU driver, die/SKU match, poison/PLM_f8/latch halt, mandatory read-back,
    post-write gate, genuine defect bits) are NEVER overrideable.
    """

    slot: int | None = None
    skip_active_sense: bool = False
    no_test_receipt: bool = False
    mask_mismatch: bool = False
    vout_range: bool = False
    undetermined_ceiling: bool = False
    allow_manual_debugctrl: bool = False

    def active_names(self) -> list[str]:
        active: list[str] = []
        if self.slot is not None:
            active.append(f"slot={self.slot}")
        if self.skip_active_sense:
            active.append("skip_active_sense")
        if self.no_test_receipt:
            active.append("no_test_receipt")
        if self.mask_mismatch:
            active.append("mask_mismatch")
        if self.vout_range:
            active.append("vout_range")
        if self.undetermined_ceiling:
            active.append("undetermined_ceiling")
        if self.allow_manual_debugctrl:
            active.append("allow_manual_debugctrl")
        return active


def _poison(name: str, value: int) -> None:
    if is_poison(value):
        raise BurnError(f"{name} poison/fuse-unit error: {hex32(value)}")


def evaluate_health(
    h: Health,
    plan: Plan,
    *,
    allow_vout_override: bool = False,
    require_efp_one: bool = False,
) -> list[str]:
    """Evaluate all health gates and return every failing condition (never mutates state)."""
    issues: list[str] = []
    for key in ("fusectrl", "dsov", "crc", "plm", "plm_f8", "status_word", "en_sw", "debugctrl"):
        val = getattr(h, key)
        if is_poison(val):
            issues.append(f"{key} poison/fuse-unit error: {hex32(val)}")

    if h.status_word == LATCH_STATUS_WORD:
        issues.append(f"controller latch STATUS_WORD={format_pmbus_status(h.status_word)}")
    elif h.status_word & PMBUS_CRITICAL_FAULT_MASK:
        issues.append(f"PMBus VRM fault/off STATUS_WORD={format_pmbus_status(h.status_word)}")
    elif h.status_word & ~plan.allowed_status_mask:
        issues.append(f"unexpected PMBus STATUS_WORD={format_pmbus_status(h.status_word)}")

    if (h.plm_f8 & 1) == 0:
        issues.append(
            f"PLM_f8 bit 0 is closed ({hex32(h.plm_f8)}): "
            "frame too deep; re-hunt shallower/slower, do not go deeper"
        )
    if not (
        not is_poison(h.fusectrl)
        and ((h.fusectrl >> 16) & 0x1F) == 4
        and (h.fusectrl & 0x1F) == 0
    ):
        issues.append(f"FUSECTRL non-idle: {hex32(h.fusectrl)} (expected state=4/idle, e.g. {hex32(FUSECTRL_IDLE)})")
    if h.debugctrl != 0:
        issues.append(f"DEBUGCTRL nonzero: {hex32(h.debugctrl)} (must be 0; RWL=1 selects RIR macro)")
    if h.dfp is not None and h.dfp != 0:
        issues.append(f"DISABLE_FUSE_PROGRAM nonzero: {hex32(h.dfp)}")
    if require_efp_one and h.efp is not None and h.efp != 1:
        issues.append(f"EFP_STATUS={h.efp} (want 1 after SENSE_CTRL)")
    if (
        h.dsov != 0
        or h.crc != GOLDILOCKS_CRC
        or h.plm != GOLDILOCKS_PLM
        or h.plm_f8 != GOLDILOCKS_PLM
    ):
        issues.append(
            f"goldilocks lost: DSOV={hex32(h.dsov)}, CRC={hex32(h.crc)}, "
            f"PLM={hex32(h.plm)}, PLM_f8={hex32(h.plm_f8)}"
        )

    if isinstance(h.vout_mv, bool) or not math.isfinite(h.vout_mv):
        issues.append(f"invalid READ_VOUT reading: {h.vout_mv!r}")
    elif not (HARD_VOUT_MIN_MV <= h.vout_mv <= HARD_VOUT_MAX_MV):
        issues.append(
            f"rail at {h.vout_mv} mV (outside hard safety bounds "
            f"{HARD_VOUT_MIN_MV}..{HARD_VOUT_MAX_MV} mV; possible floor/latch)"
        )
    elif not allow_vout_override and not (
        plan.stock_mv - plan.vout_tolerance_mv
        <= h.vout_mv
        <= plan.stock_mv + plan.vout_tolerance_mv
    ):
        issues.append(
            f"rail at {h.vout_mv} mV, expected {plan.stock_mv} +/- {plan.vout_tolerance_mv} mV"
        )
    return issues


def check_health(
    h: Health,
    plan: Plan,
    *,
    allow_vout_override: bool = False,
    require_efp_one: bool = False,
) -> None:
    """No repair, reset, rail adjustment or retry is performed."""
    issues = evaluate_health(
        h,
        plan,
        allow_vout_override=allow_vout_override,
        require_efp_one=require_efp_one,
    )
    if issues:
        raise BurnError("; ".join(issues))


def evaluate_host(host: Host, *, require_hold: bool = True) -> list[str]:
    """Evaluate host safety invariants and return every failing condition."""
    issues: list[str] = []
    if require_hold and not host.hold_active:
        issues.append("no live held-frame indicator")
    if host.bound_driver not in (None, "pci-stub"):
        issues.append(f"GPU bound to forbidden driver: {host.bound_driver}")
    if host.nvidia_loaded or host.nouveau_loaded:
        issues.append("nvidia/nouveau driver loaded")
    if "[none]" not in host.lockdown.split():
        issues.append(f"kernel lockdown is not [none]: {host.lockdown}")
    if host.vrm_tainted:
        issues.append("VRM profile tainted")
    if not host.boot_id:
        issues.append("missing boot identity")
    return issues


def check_host(host: Host, *, require_hold: bool = True) -> None:
    issues = evaluate_host(host, require_hold=require_hold)
    if issues:
        raise BurnError("; ".join(issues))


def check_host_readonly_safe(host: Host) -> None:
    """Minimum host safety for read-only manual inspection (allows cold/faulted cards)."""
    issues: list[str] = []
    if host.bound_driver not in (None, "pci-stub"):
        issues.append(f"GPU bound to forbidden driver: {host.bound_driver}")
    if host.nvidia_loaded or host.nouveau_loaded:
        issues.append("nvidia/nouveau driver loaded")
    if issues:
        raise BurnError("; ".join(issues))


def check_die_rows(plan: Plan, reader) -> None:
    for i, value in plan.die_rows.items():
        actual = reader.die_row(i)
        _poison(f"die row {i}", actual)
        if actual != value:
            raise BurnError(
                f"DIE ID MISMATCH row {i}: expected {hex32(value)}, got {hex32(actual)}"
            )


def check_masks(expected_masks: dict[str, int], reader) -> None:
    for field, expected in expected_masks.items():
        actual = reader.field(field)
        _poison(f"field {field}", actual)
        if actual != expected:
            raise BurnError(
                f"IDENTITY MISMATCH {field}: expected {hex32(expected)}, got {hex32(actual)}"
            )


def preflight(
    plan: Plan,
    reader,
    *,
    active_sense: bool,
    match_masks: bool = True,
    expected_masks: dict[str, int] | None = None,
    overrides: Overrides | None = None,
) -> str:
    """Run fresh hardware/simulator preflight checks and return current boot_id."""
    ov = overrides or Overrides()
    host = reader.host()
    check_host(host)
    if reader.pci_id().lower() != plan.pci_id:
        raise BurnError(f"PCI identity mismatch: expected {plan.pci_id}, got {reader.pci_id()}")
    h = reader.health()
    check_health(h, plan, allow_vout_override=ov.vout_range)
    check_die_rows(plan, reader)
    if match_masks and not ov.mask_mismatch:
        check_masks(expected_masks if expected_masks is not None else plan.expected_masks, reader)
    if active_sense:
        reader.active_sense_readonly()
        # The sense itself may have destroyed this frame: read everything AGAIN and require EFP=1.
        check_health(
            reader.health(),
            plan,
            allow_vout_override=ov.vout_range,
            require_efp_one=True,
        )
        check_host(reader.host())
        for i, value in plan.die_rows.items():
            actual = reader.die_row(i)
            _poison(f"post-sense die row {i}", actual)
            if actual != value:
                raise BurnError(f"die ID changed after active sense: row {i}")
    return host.boot_id


def scan_slots(
    plan: Plan,
    reader,
    *,
    override_slot: int | None = None,
) -> tuple[int, list[tuple[int, int]]]:
    """Scan full declared FPF region twice. No heuristic for empties.

    When ``override_slot`` is provided (after explicit operator confirmation),
    the scan still enforces double-read stability, poison checks, row bounds,
    and that the chosen slot currently reads ``plan.empty_word``.
    """

    def scan() -> list[int]:
        return [reader.fuse_row(i) for i in range(plan.first_row, plan.last_row + 1)]

    a = scan()
    b = scan()
    if a != b:
        raise BurnError("FPF region changed between scans; cannot prove next free slot")
    for i, v in enumerate(a):
        _poison(f"FPF row {plan.first_row + i}", v)

    empty = plan.empty_word
    if override_slot is not None:
        if isinstance(override_slot, bool) or not isinstance(override_slot, int):
            raise BurnError("override slot must be an integer row index")
        if not (plan.first_row <= override_slot <= plan.last_row):
            raise BurnError(
                f"override slot {override_slot} outside FPF range {plan.first_row}..{plan.last_row}"
            )
        rel = override_slot - plan.first_row
        if a[rel] != empty:
            raise BurnError(
                f"REFUSED override slot {override_slot}: row is NOT empty "
                f"(reads {hex32(a[rel])}, expected {hex32(empty)})"
            )
        nearby = [
            (plan.first_row + i, v)
            for i, v in enumerate(a)
            if max(0, rel - 3) <= i < min(len(a), rel + 4)
        ]
        return override_slot, nearby

    first = next((i for i, x in enumerate(a) if x == empty), None)
    if first is None:
        raise BurnError("FPF region full; no free record")
    if any(v != empty for v in a[first:]):
        raise BurnError(
            "non-contiguous FPF records; next free slot is ambiguous "
            "(inspect FPF rows and use --override-slot ROW if intentional)"
        )
    pointer = plan.first_row + first
    nearby = [
        (plan.first_row + i, v)
        for i, v in enumerate(a)
        if max(0, first - 3) <= i < min(len(a), first + 4)
    ]
    return pointer, nearby


def _is_compute_mask_field(field: str) -> bool:
    upper = field.upper()
    return any(tok in upper for tok in ("GPC", "TPC", "SM"))


def reconcile(
    plan: Plan,
    record: Record,
    assumed_masks: dict[str, int],
    *,
    allow_manual_debugctrl: bool = False,
    allow_undetermined_ceiling: bool = False,
) -> None:
    """Prove defect bits and ceilings; never estimates unknown SM counts."""
    if record.word == plan.empty_word:
        raise BurnError(
            f"{record.name}: payload equals FPF empty_word ({hex32(plan.empty_word)})"
        )
    if is_poison(record.word):
        raise BurnError(f"{record.name}: payload {hex32(record.word)} matches poison prefix")
    if is_debugctrl_target(record.target) and not allow_manual_debugctrl:
        raise BurnError("DEBUGCTRL must never be included in an automatic plan")

    if record.mask_field is not None:
        field = record.mask_field
        if field not in assumed_masks or field not in plan.defect_masks:
            raise BurnError(f"{record.name}: missing cold mask or genuine defect mask for {field}")
        prior = assumed_masks[field]
        if prior != record.from_disable:
            raise BurnError(
                f"{record.name}: plan mask chain inconsistent for {field}: "
                f"{hex32(prior)} != {hex32(record.from_disable)}"
            )
        removed = record.from_disable & ~record.to_disable & 0xFFFFFFFF
        genuine = plan.defect_masks[field]
        if removed & genuine:
            raise BurnError(
                f"{record.name}: REFUSED; reenables genuinely defective {field} "
                f"bits {hex32(removed & genuine)}"
            )
        if _is_compute_mask_field(field) and record.resulting_sm_count is None:
            if not allow_undetermined_ceiling:
                raise BurnError(
                    f"{record.name}: resulting SM count undetermined; external review required"
                )
        assumed_masks[field] = record.to_disable
    elif record.defect_sensitive:
        raise BurnError(f"{record.name}: defect-sensitive change without mask map")

    if record.kind == "reconfig":
        if plan.override_sm_ceiling is None or not plan.reconfig_semantics_verified:
            raise BurnError("RECONFIG override ceiling/semantics unverified")

    if record.resulting_sm_count is not None and not allow_undetermined_ceiling:
        ceiling = plan.override_sm_ceiling if record.kind == "reconfig" else plan.base_sm_ceiling
        if ceiling is None or record.resulting_sm_count > ceiling:
            raise BurnError(
                f"{record.name}: predicted {record.resulting_sm_count} SM beyond supplied ceiling {ceiling}"
            )

    if record.kind == "pcie" and (
        record.link_cap2_before is None or record.link_cap2_after is None
    ):
        raise BurnError("PCIe record must declare reviewed LnkCap2 before/after")


def check_plan(
    plan: Plan,
    *,
    allow_manual_debugctrl: bool = False,
    allow_undetermined_ceiling: bool = False,
) -> dict[str, int]:
    assumed = plan.expected_masks.copy()
    record_cold: dict[str, int] = {}
    for r in plan.records:
        reconcile(
            plan,
            r,
            assumed,
            allow_manual_debugctrl=allow_manual_debugctrl,
            allow_undetermined_ceiling=allow_undetermined_ceiling,
        )
        record_cold.update(r.expected_cold)
    if plan.final_expected_cold:
        for field, value in record_cold.items():
            if field in plan.final_expected_cold and plan.final_expected_cold[field] != value:
                raise BurnError(
                    f"plan final_expected_cold.{field}={hex32(plan.final_expected_cold[field])} "
                    f"contradicts record expected_cold={hex32(value)}"
                )
    return assumed


class Journal:
    """Append-only, fsynced, exclusively locked JSONL with mode 0600 and no symlinks."""

    def __init__(self, file: str | Path) -> None:
        self.path = Path(file)
        if self.path.is_symlink():
            raise BurnError(f"refusing symlink journal: {self.path}")
        if self.path.parent.is_symlink():
            raise BurnError(f"refusing symlink journal parent directory: {self.path.parent}")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.path.parent.stat().st_mode & 0o077:
            raise BurnError(
                f"journal parent directory not private (must be mode 0700): {self.path.parent} "
                f"— run: chmod 700 {self.path.parent}"
            )
        self.fd: int | None = os.open(
            self.path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            if stat.S_IMODE(os.fstat(self.fd).st_mode) != 0o600:
                raise BurnError(f"journal is not mode 0600: {self.path}")
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BurnError(
                    f"journal {self.path} is locked by another vfi_burn session; "
                    "never run two sessions concurrently"
                ) from exc
        except Exception:
            os.close(self.fd)
            self.fd = None
            raise

    def append(self, **values) -> dict:
        if self.fd is None:
            raise BurnError("journal is closed")
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **values}
        line = (json.dumps(entry, sort_keys=True, default=str) + "\n").encode("utf-8")
        view = memoryview(line)
        try:
            while len(view):
                count = os.write(self.fd, view)
                if count <= 0:
                    raise BurnError("journal write failed; no further records allowed")
                view = view[count:]
        finally:
            view.release()
        os.fsync(self.fd)
        return entry

    def close(self) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _confirm_token(confirm, prompt: str, token: str) -> bool:
    return confirm(prompt, token) is True


def _require_override_confirm(
    confirm,
    journal: Journal,
    show,
    *,
    name: str,
    banner: str,
    token: str,
    details: dict | None = None,
) -> None:
    show(f"!!! MANUAL OVERRIDE [{name}]: {banner}")
    if not _confirm_token(confirm, f"Type '{token}' to confirm override: ", token):
        journal.append(event="override_declined", override=name, **(details or {}))
        raise BurnError(f"manual override {name!r} declined by operator")
    journal.append(event="override_confirmed", override=name, token=token, **(details or {}))


def _load_test_receipt(plan: Plan, test_receipt: str | Path) -> dict:
    receipt_path = require_private_file(test_receipt, "test receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BurnError(f"invalid JSON in test receipt {receipt_path}: {exc}") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("plan_sha256") != plan.content_hash
        or receipt.get("status") != "PASS"
        or receipt.get("stage") != "test"
    ):
        raise BurnError("cold verification receipt missing, failed or for another plan")
    if receipt.get("test_word") != hex32(plan.records[0].word):
        raise BurnError("test receipt record word disagrees with reviewed plan")
    test_row = receipt.get("test_row")
    if isinstance(test_row, bool) or not isinstance(test_row, int):
        raise BurnError("test receipt missing valid integer test_row")
    return receipt


def _capture_snapshot(reader) -> tuple[dict | None, dict | None]:
    health_dict = None
    host_dict = None
    try:
        health_dict = reader.health().serializable()
    except Exception:
        pass
    try:
        host_dict = reader.host().serializable()
    except Exception:
        pass
    return health_dict, host_dict


def _confirm_active_sense(confirm, journal: Journal, show, *, stage: str, phase: str) -> None:
    show(
        "ACTIVE SENSE: active_sense_readonly() does not burn fuses, but can collapse "
        "a too-deep held goldilocks frame (forcing power cycle + re-hunt)."
    )
    if not _confirm_token(confirm, "Type 'SENSE' to run active_sense_readonly(): ", "SENSE"):
        journal.append(event="declined", stage=stage, phase=phase, reason="SENSE")
        raise BurnError("active sense declined by operator")
    journal.append(event="active_sense_confirmed", stage=stage, phase=phase)


def process_records(
    plan: Plan,
    reader,
    writer,
    journal: Journal,
    *,
    stage: str,
    confirm,
    test_receipt=None,
    show=print,
    allow_manual_debugctrl: bool = False,
    overrides: Overrides | None = None,
    max_records: int | None = None,
    dry_run: bool = False,
    strict_confirm: bool = False,
    confirm_active_sense: bool = False,
    run_active_sense: bool | None = None,
    manual_record_sha256: str | None = None,
    is_manual_single: bool = False,
) -> int:
    """Execute one controlled burn (or simulation/dry-run) session.

    Never catches a failed write and continues. Pointer advances ONLY after
    mandatory read-back AND full post-write health/host gate pass.
    """
    if stage not in ("test", "real", "all"):
        raise BurnError("stage must be 'test', 'real', or 'all'")
    if max_records is not None and (isinstance(max_records, bool) or max_records < 1):
        raise BurnError("max_records must be a positive integer")

    ov = overrides or Overrides()
    effective_debugctrl = allow_manual_debugctrl or ov.allow_manual_debugctrl
    use_active_sense = (
        (not ov.skip_active_sense)
        if run_active_sense is None
        else (bool(run_active_sense) and not ov.skip_active_sense)
    )
    check_plan(
        plan,
        allow_manual_debugctrl=effective_debugctrl,
        allow_undetermined_ceiling=ov.undetermined_ceiling,
    )

    receipt: dict | None = None
    if stage == "real":
        if test_receipt is None:
            if not ov.no_test_receipt:
                raise BurnError(
                    "real stage forbidden: verified cold-boot test receipt required "
                    "(or supply --override-no-test-receipt if verified externally)"
                )
            if not dry_run:
                _require_override_confirm(
                    confirm,
                    journal,
                    show,
                    name="no_test_receipt",
                    banner="Proceeding to REAL stage without a vfi_burn cold-boot test receipt.",
                    token="OVERRIDE NO TEST RECEIPT",
                )
        else:
            receipt = _load_test_receipt(plan, test_receipt)

    if not dry_run:
        if ov.skip_active_sense:
            _require_override_confirm(
                confirm,
                journal,
                show,
                name="skip_active_sense",
                banner="Skipping active-sense dry run; passive goldilocks gates still apply.",
                token="OVERRIDE SKIP SENSE",
            )
        if ov.mask_mismatch:
            _require_override_confirm(
                confirm,
                journal,
                show,
                name="mask_mismatch",
                banner="Skipping initial expected-mask equality check against card fields.",
                token="OVERRIDE MASK MISMATCH",
            )
        if ov.undetermined_ceiling:
            _require_override_confirm(
                confirm,
                journal,
                show,
                name="undetermined_ceiling",
                banner="Allowing operator-supplied SM ceiling override (genuine defect bits remain REFUSED).",
                token="OVERRIDE CEILING",
            )

    if use_active_sense and confirm_active_sense:
        _confirm_active_sense(confirm, journal, show, stage=stage, phase="initial_preflight")

    # Initial live preflight (passive + optional active sense); masks verified below
    # after determining already-committed prefix rows on resumed real sessions.
    try:
        boot_id = preflight(
            plan,
            reader,
            active_sense=use_active_sense,
            match_masks=False,
            overrides=ov,
        )
    except Exception as exc:
        h_snap, host_snap = _capture_snapshot(reader)
        journal.append(
            event="preflight_failed",
            stage=stage,
            plan_sha256=plan.content_hash,
            phase="initial_preflight",
            active_sense=use_active_sense,
            health=h_snap,
            host=host_snap,
            error=str(exc),
        )
        raise

    if ov.vout_range and not dry_run:
        live_health = reader.health()
        if not (
            plan.stock_mv - plan.vout_tolerance_mv
            <= live_health.vout_mv
            <= plan.stock_mv + plan.vout_tolerance_mv
        ):
            _require_override_confirm(
                confirm,
                journal,
                show,
                name="vout_range",
                banner=(
                    f"Measured READ_VOUT is {live_health.vout_mv:.1f} mV "
                    f"(plan expects {plan.stock_mv} +/- {plan.vout_tolerance_mv} mV)."
                ),
                token=f"OVERRIDE VOUT {live_health.vout_mv:.1f}",
                details={"measured_vout_mv": live_health.vout_mv},
            )

    try:
        pointer, nearby = scan_slots(plan, reader, override_slot=ov.slot)
    except Exception as exc:
        h_snap, host_snap = _capture_snapshot(reader)
        journal.append(
            event="preflight_failed",
            stage=stage,
            plan_sha256=plan.content_hash,
            phase="slot_scan",
            active_sense=use_active_sense,
            health=h_snap,
            host=host_snap,
            error=str(exc),
        )
        raise

    show(
        f"Free FPF row {pointer}, nearby: "
        + ", ".join(f"{i}:{hex32(w)}" for i, w in nearby)
    )
    if ov.slot is not None and not dry_run:
        _require_override_confirm(
            confirm,
            journal,
            show,
            name="slot",
            banner=f"Using operator-selected empty FPF row {ov.slot} instead of contiguous scan.",
            token=f"OVERRIDE SLOT {ov.slot}",
            details={"slot": ov.slot},
        )

    assumed = plan.expected_masks.copy()
    if stage == "test":
        pending = list(plan.records[:1])
    elif stage == "all":
        pending = list(plan.records)
    else:
        reconcile(
            plan,
            plan.records[0],
            assumed,
            allow_manual_debugctrl=effective_debugctrl,
            allow_undetermined_ceiling=ov.undetermined_ceiling,
        )
        if is_manual_single:
            if receipt is not None:
                test_row = receipt["test_row"]
                actual_test = reader.fuse_row(test_row)
                if actual_test != plan.records[0].word or pointer < test_row + 1:
                    raise BurnError(
                        "cold-verified test record not at expected location; refuse manual write"
                    )
            pending = list(plan.records[1:])
        else:
            if receipt is not None:
                test_row = receipt["test_row"]
                actual_test = reader.fuse_row(test_row)
                if actual_test != plan.records[0].word or pointer < test_row + 1:
                    raise BurnError(
                        "cold-verified test record not at expected location; refuse real stage"
                    )
                already_written = pointer - (test_row + 1)
                if already_written > len(plan.records) - 1:
                    raise BurnError(
                        f"FPF pointer {pointer} is past end of plan "
                        f"(test_row={test_row}, {len(plan.records) - 1} real records)"
                    )
                for offset in range(1, already_written + 1):
                    row_idx = test_row + offset
                    expected_rec = plan.records[offset]
                    actual_word = reader.fuse_row(row_idx)
                    if actual_word != expected_rec.word:
                        raise BurnError(
                            f"committed FPF row {row_idx} reads {hex32(actual_word)}, "
                            f"expected plan record {expected_rec.name!r} ({hex32(expected_rec.word)})"
                        )
                    reconcile(
                        plan,
                        expected_rec,
                        assumed,
                        allow_manual_debugctrl=effective_debugctrl,
                        allow_undetermined_ceiling=ov.undetermined_ceiling,
                    )
                if already_written > 0:
                    show(
                        f"Resuming real stage: {already_written} record(s) already committed "
                        f"in rows {test_row + 1}..{pointer - 1}."
                    )
                pending = list(plan.records[1 + already_written :])
                if not pending:
                    if len(plan.records) == 1:
                        raise BurnError("no real records in plan")
                    show("All planned real records are already present in FPF rows; nothing to burn.")
                    return pointer
            else:
                pending = list(plan.records[1:])
                if not pending:
                    raise BurnError("no real records in plan")

    if not ov.mask_mismatch:
        try:
            check_masks(assumed, reader)
        except Exception as exc:
            h_snap, host_snap = _capture_snapshot(reader)
            journal.append(
                event="preflight_failed",
                stage=stage,
                plan_sha256=plan.content_hash,
                phase="mask_check",
                active_sense=use_active_sense,
                health=h_snap,
                host=host_snap,
                error=str(exc),
            )
            raise

    if max_records is not None:
        pending = pending[:max_records]

    start_health, start_host = _capture_snapshot(reader)
    journal.append(
        event="session_start",
        plan_sha256=plan.content_hash,
        pci_id=plan.pci_id,
        stage=stage,
        boot_id=boot_id,
        held_frame_source="operator_supplied_live_adapter"
        if not getattr(reader, "simulation", False)
        else "fake_simulation",
        pointer=pointer,
        pending_records=[r.name for r in pending],
        dry_run=dry_run,
        active_sense=use_active_sense,
        health=start_health,
        host=start_host,
        overrides=ov.active_names(),
        manual_record_sha256=manual_record_sha256,
    )

    if dry_run:
        dry_assumed = assumed.copy()
        for offset, record in enumerate(pending):
            reconcile(
                plan,
                record,
                dry_assumed,
                allow_manual_debugctrl=effective_debugctrl,
                allow_undetermined_ceiling=ov.undetermined_ceiling,
            )
            slot = pointer + offset
            show(
                f"DRY-RUN [{record.kind.upper()}]: would write row {slot} "
                f"{record.target}={hex32(record.word)} ({record.decode})"
            )
        journal.append(
            event="dry_run_complete",
            stage=stage,
            pointer=pointer,
            active_sense=use_active_sense,
            records=[r.name for r in pending],
        )
        return pointer

    for idx, record in enumerate(pending):
        reconcile(
            plan,
            record,
            assumed,
            allow_manual_debugctrl=effective_debugctrl,
            allow_undetermined_ceiling=ov.undetermined_ceiling,
        )
        if idx > 0:
            # Repeat live health, host, die ID, and slot-scan checks before every subsequent record.
            # SENSE_CTRL already ran once for this window (GA100 sets RESENSE_DISABLE=1 after it).
            try:
                preflight(
                    plan,
                    reader,
                    active_sense=False,
                    match_masks=False,
                    overrides=ov,
                )
                fresh_pointer, _ = scan_slots(
                    plan,
                    reader,
                    override_slot=pointer if ov.slot is not None else None,
                )
                if fresh_pointer != pointer:
                    raise BurnError(
                        f"FPF pointer moved externally {pointer} -> {fresh_pointer}; HALT"
                    )
            except Exception as exc:
                h_snap, host_snap = _capture_snapshot(reader)
                journal.append(
                    event="preflight_failed",
                    stage=stage,
                    plan_sha256=plan.content_hash,
                    phase=f"before_record:{record.name}",
                    active_sense=False,
                    health=h_snap,
                    host=host_snap,
                    error=str(exc),
                )
                raise

        previous = reader.fuse_row(pointer)
        _poison(f"FPF row {pointer}", previous)
        if previous != plan.empty_word:
            raise BurnError(
                f"row {pointer} not empty immediately before write "
                f"(reads {hex32(previous)}, expected {hex32(plan.empty_word)})"
            )

        mask_info = ""
        if record.mask_field is not None:
            mask_info = (
                f"; mask {record.mask_field}: "
                f"{hex32(record.from_disable)} -> {hex32(record.to_disable)}"
            )
            if record.resulting_sm_count is not None:
                mask_info += f" ({record.resulting_sm_count} SM)"
        show(
            f"{record.kind.upper()}: row {pointer} {record.target}={hex32(record.word)} "
            f"({record.decode}); defect_sensitive={record.defect_sensitive}{mask_info}; "
            f"current={hex32(previous)}"
        )

        if is_debugctrl_target(record.target):
            if not effective_debugctrl:
                raise BurnError("DEBUGCTRL must never be included in an automatic plan")
            for token in ("REVIEW DEBUGCTRL", "CONFIRM DEBUGCTRL"):
                show("WARNING: DEBUGCTRL (0x00820060) is a sensitive controller register.")
                if not _confirm_token(confirm, f"Type '{token}' separately: ", token):
                    journal.append(event="declined", row=pointer, record=record.name, reason=token)
                    raise BurnError("DEBUGCTRL separately confirmed manual action declined")

        if record.kind == "pcie":
            observed_link = reader.field("LnkCap2")
            _poison("LnkCap2", observed_link)
            show(
                f"LnkCap2 current={hex32(observed_link)}, "
                f"planned target={hex32(record.link_cap2_after)}"
            )
            if observed_link != record.link_cap2_before:
                raise BurnError(
                    f"current LnkCap2 ({hex32(observed_link)}) does not match reviewed "
                    f"before state ({hex32(record.link_cap2_before)})"
                )

        if record.kind in ("pcie", "reconfig"):
            token = f"REVIEW {record.kind.upper()}"
            if not _confirm_token(
                confirm,
                f"{token}: externally verify current link/ceiling and type '{token}': ",
                token,
            ):
                journal.append(event="declined", row=pointer, record=record.name, reason=token)
                raise BurnError("special record not confirmed")

        confirm_word = f"YES {pointer} {hex32(record.word)}" if strict_confirm else "YES"
        if not _confirm_token(
            confirm,
            f"PERMANENT: type '{confirm_word}' to burn row {pointer} (or Enter to stop): ",
            confirm_word,
        ):
            journal.append(event="declined", row=pointer, record=record.name)
            raise BurnError("operator declined")

        # Instantaneous pre-trigger gate: closes the window while the human was reading/typing.
        try:
            check_host(reader.host())
            check_health(reader.health(), plan, allow_vout_override=ov.vout_range)
            pre_slot = reader.fuse_row(pointer)
            _poison(f"pre-trigger FPF row {pointer}", pre_slot)
            if pre_slot != plan.empty_word:
                raise BurnError(
                    f"row {pointer} changed during confirmation ({hex32(pre_slot)})"
                )
        except Exception as exc:
            journal.append(
                event="aborted_pre_trigger",
                row=pointer,
                record=record.name,
                error=str(exc),
                write_issued=False,
            )
            raise BurnError(
                f"ABORTED before write (card state changed during confirmation; NO write issued): {exc}"
            ) from exc

        journal.append(
            event="attempt",
            stage=stage,
            plan_sha256=plan.content_hash,
            manual_record_sha256=manual_record_sha256,
            card=plan.pci_id,
            row=pointer,
            record=record.name,
            target=record.target,
            intended=hex32(record.word),
            previous=hex32(previous),
            confirmation=confirm_word,
            boot_id=boot_id,
            overrides=ov.active_names(),
        )

        readback: int | None = None
        health: Health | None = None
        host_after: Host | None = None
        try:
            writer.write_one_record(pointer, record.word)
            readback = reader.fuse_row(pointer)
            _poison(f"row {pointer} readback", readback)
            if readback != record.word:
                raise BurnError(
                    f"row {pointer} READBACK MISMATCH: {hex32(readback)} != {hex32(record.word)}"
                )
            health = reader.health()
            check_health(health, plan, allow_vout_override=ov.vout_range)
            host_after = reader.host()
            check_host(host_after)
            # Extra slot check: no pointer advance before every post-write gate passes.
            second_readback = reader.fuse_row(pointer)
            if second_readback != record.word:
                raise BurnError(
                    f"row {pointer} changed after post-write gate: {hex32(second_readback)}"
                )
        except Exception as exc:
            # Failure may be partial. Never re-issue the programming operation.
            if health is None:
                try:
                    health = reader.health()
                except Exception:
                    pass
            if host_after is None:
                try:
                    host_after = reader.host()
                except Exception:
                    pass
            if readback is None:
                try:
                    readback = reader.fuse_row(pointer)
                except Exception:
                    pass
            journal.append(
                event="HALT_UNVERIFIED",
                row=pointer,
                record=record.name,
                intended=hex32(record.word),
                readback=hex32(readback) if readback is not None else None,
                full_post_write_gate=health.serializable() if health else None,
                host_after=host_after.serializable() if host_after else None,
                error=str(exc),
                pointer_advanced=False,
                manual_record_sha256=manual_record_sha256,
            )
            raise BurnError(
                f"HALT: row {pointer} may be partially programmed; do NOT retry: {exc}"
            ) from exc

        journal.append(
            event="verified_readback_only",
            row=pointer,
            record=record.name,
            target=record.target,
            intended=hex32(record.word),
            readback=hex32(readback),
            result="MATCH",
            full_post_write_gate=health.serializable(),
            host_after=host_after.serializable(),
            pointer_advanced=True,
            cold_boot_verified=False,
            boot_id=boot_id,
            plan_sha256=plan.content_hash,
            manual_record_sha256=manual_record_sha256,
        )
        pointer += 1  # The ONLY advance, after read-back AND healthy gate.

        if stage == "all" and record.kind == "test" and idx + 1 < len(pending):
            show(
                f"\nTEST ROW READ-BACK VERIFIED (row {pointer - 1} = {hex32(readback)}, "
                f"CRC={hex32(health.crc)}, FUSECTRL={hex32(health.fusectrl)})."
            )
            show(
                "NOTE: Read-back confirms the test write landed in the fuse array, NOT a cold-boot re-parse.\n"
                "Continuing keeps your current Goldilocks glitch window (no re-glitch required)."
            )
            if not _confirm_token(
                confirm,
                "Type 'CONTINUE SAME WINDOW' to proceed to real records now "
                "(or press Enter to stop and cold-boot verify first): ",
                "CONTINUE SAME WINDOW",
            ):
                journal.append(
                    event="stage_all_paused_after_test",
                    test_row=pointer - 1,
                    next_pointer=pointer,
                    remaining_records=[r.name for r in pending[idx + 1 :]],
                    cold_boot_verified=False,
                )
                show(
                    "Paused cleanly after test row. Cold-boot, run verify-cold --stage test "
                    "--receipt-out <file>, then resume with --stage real --receipt <file>."
                )
                return pointer
            journal.append(
                event="same_window_continue_confirmed",
                test_row=pointer - 1,
                next_pointer=pointer,
                cold_boot_verified=False,
            )

    journal.append(
        event="session_complete_readback_only",
        stage=stage,
        last_pointer=pointer,
        records_written=len(pending),
        cold_boot_verified=False,
    )
    return pointer


def verify_cold_snapshot(plan: Plan, cold: dict, *, stage: str, journal: Journal) -> dict:
    """Offline compare; user-provided snapshot evidence is NOT authenticated hardware proof."""
    logged_comparison = False
    try:
        if stage not in ("test", "real"):
            raise BurnError("bad cold verification stage")
        if not isinstance(cold, dict):
            raise BurnError("cold snapshot must be a JSON object")
        if str(cold.get("pci_id", "")).lower() != plan.pci_id:
            raise BurnError("cold snapshot PCI ID mismatch")
        if (
            cold.get("cold_boot") is not True
            or not cold.get("boot_id")
            or not cold.get("previous_boot_id")
        ):
            raise BurnError("cold snapshot must declare two distinct boot IDs and cold_boot=true")
        if cold["boot_id"] == cold["previous_boot_id"]:
            raise BurnError("not a distinct cold boot")

        raw_die = cold.get("die_rows")
        if not isinstance(raw_die, dict):
            raise BurnError("cold snapshot missing die_rows object")
        actual_die = {int(k): u32(v, f"cold.die_rows.{k}") for k, v in raw_die.items()}
        if actual_die != plan.die_rows:
            raise BurnError("cold snapshot die ID mismatch")

        prior: list[dict] = []
        if journal.path.exists():
            prior = [
                json.loads(line)
                for line in journal.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        test_row: int | None = None
        if stage == "test":
            for item in reversed(prior):
                if (
                    item.get("event") == "verified_readback_only"
                    and item.get("record") == plan.records[0].name
                    and item.get("intended") == hex32(plan.records[0].word)
                    and item.get("plan_sha256") == plan.content_hash
                    and item.get("boot_id") == cold["previous_boot_id"]
                ):
                    test_row = item["row"]
                    break
            if test_row is None:
                raise BurnError("no matching completed test-row write in this private journal")
            if not any(
                item.get("event") == "session_start"
                and item.get("stage") in ("test", "all")
                and item.get("boot_id") == cold["previous_boot_id"]
                and item.get("plan_sha256") == plan.content_hash
                for item in prior
            ):
                raise BurnError("cold previous_boot_id does not match journal test session")
            raw_rows = cold.get("fpf_rows", {})
            if not isinstance(raw_rows, dict) or str(test_row) not in raw_rows:
                raise BurnError("cold snapshot lacks the written test FPF row")
            if u32(raw_rows[str(test_row)], f"cold.fpf_rows.{test_row}") != plan.records[0].word:
                raise BurnError("cold test FPF row does not persist")
        else:
            # If fpf_rows are supplied in real cold snapshot, verify every journaled write persisted.
            raw_rows = cold.get("fpf_rows")
            if isinstance(raw_rows, dict) and raw_rows:
                written = [
                    item
                    for item in prior
                    if item.get("event") == "verified_readback_only"
                    and item.get("plan_sha256") == plan.content_hash
                    and item.get("manual_record_sha256") is None
                ]
                for item in written:
                    row_key = str(item["row"])
                    if row_key in raw_rows:
                        observed_word = u32(raw_rows[row_key], f"cold.fpf_rows.{row_key}")
                        expected_word = u32(item["intended"], "journal.intended")
                        if observed_word != expected_word:
                            raise BurnError(
                                f"cold FPF row {row_key} ({hex32(observed_word)}) "
                                f"does not match written record {item['record']!r} ({hex32(expected_word)})"
                            )

        expected: dict[str, int] = {}
        records = plan.records[:1] if stage == "test" else plan.records
        for r in records:
            expected.update(r.expected_cold)
        if stage == "real":
            mandatory = {
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
            }
            absent = mandatory - set(plan.final_expected_cold)
            if absent:
                raise BurnError(
                    "incomplete full cold-boot expectations: " + ", ".join(sorted(absent))
                )
            # final_expected_cold takes precedence over intermediate per-record expected_cold.
            expected.update(plan.final_expected_cold)

        if not expected:
            raise BurnError("no independently reviewed cold-state expectations")

        observed = cold.get("fields", {})
        if not isinstance(observed, dict):
            raise BurnError("cold snapshot fields must be a JSON object")
        diffs = {
            field: (
                hex32(value),
                hex32(u32(observed[field], f"cold.fields.{field}"))
                if field in observed
                else "MISSING",
            )
            for field, value in expected.items()
            if field not in observed or u32(observed[field], f"cold.fields.{field}") != value
        }
        event = journal.append(
            event="cold_snapshot_comparison",
            stage=stage,
            plan_sha256=plan.content_hash,
            boot_id=cold["boot_id"],
            previous_boot_id=cold["previous_boot_id"],
            compared=sorted(expected),
            differences=diffs,
            status="PASS" if not diffs else "FAIL",
            evidence_kind="operator_provided_cold_snapshot_not_authenticated",
        )
        logged_comparison = True
        if diffs:
            raise BurnError(f"cold snapshot differs: {diffs}")
    except Exception as exc:
        if not logged_comparison:
            journal.append(
                event="cold_snapshot_rejected",
                stage=stage,
                plan_sha256=plan.content_hash,
                boot_id=cold.get("boot_id") if isinstance(cold, dict) else None,
                previous_boot_id=cold.get("previous_boot_id") if isinstance(cold, dict) else None,
                status="FAIL",
                error=str(exc),
                evidence_kind="operator_provided_cold_snapshot_not_authenticated",
            )
        raise

    return {
        "status": "PASS",
        "plan_sha256": plan.content_hash,
        "stage": stage,
        "test_row": test_row,
        "test_word": hex32(plan.records[0].word),
        "boot_id": cold["boot_id"],
        "previous_boot_id": cold["previous_boot_id"],
        "journal_timestamp": event["timestamp"],
        "evidence_kind": "operator_provided_cold_snapshot_not_authenticated",
    }
