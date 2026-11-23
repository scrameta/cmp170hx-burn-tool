"""Fail-closed fuse-controller state machine."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json

from .backend import Health, Host, MissingWriter
from .ffrecord import (
    FFCALC_KEY_TO_FIELD,
    TOTAL_GPC,
    apply_ff_word_to_masks,
    classify_ff_functional_groups,
    compute_sm_count,
    decode_ff,
    evaluate_poweroff_sanity,
    ff_word_defect_audit,
    ff_word_safety_audit,
    ff_word_touches_pcie,
    fields_in_block,
    replay_ff_rows,
    verify_noop_test_record,
)
from .fuse_rules import poweroff_domains_for_groups
from .journal import Journal, deferred_signals
from .model import (
    BurnError,
    Plan,
    PreTriggerRefusal,
    Record,
    is_debugctrl_target,
    is_poison,
    require_private_file,
    u32,
)

FUSECTRL_IDLE = 0xE0040000
GOLDILOCKS_CRC = 0x55
GOLDILOCKS_PLM = 0xFFFFFFFF


def hex32(value: int) -> str:
    return f"0x{value:08x}"


@dataclass(frozen=True)
class Overrides:
    """Explicit, granular manual overrides. All default to False/None (fail-closed).

    Hard brick-prevention invariants (aligned 32-bit stores, empty target slot,
    no GPU driver, die/SKU match, poison/PLM_f8 halt, mandatory read-back,
    post-write gate, known genuine defect bits, and read-only PCIe device-ID
    blocks c2/b37-38 & burnable=False bits) are NEVER overrideable.
    """

    slot: int | None = None
    skip_active_sense: bool = False
    no_test_receipt: bool = False
    mask_mismatch: bool = False
    undetermined_ceiling: bool = False
    allow_manual_debugctrl: bool = False
    baseline_mismatch: bool = False
    unmapped_ff: bool = False

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
        if self.undetermined_ceiling:
            active.append("undetermined_ceiling")
        if self.allow_manual_debugctrl:
            active.append("allow_manual_debugctrl")
        if self.baseline_mismatch:
            active.append("baseline_mismatch")
        if self.unmapped_ff:
            active.append("unmapped_ff")
        return active


def _poison(name: str, value: int) -> None:
    """Refuse a poisoned REGISTER value (FUSECTRL, option shadows, PCIe config, ...).

    Not for fuse-row data: ``FuseReader.fuse_row``/``die_row`` raise on a fuse-unit
    error themselves, and a returned row value is OTP data, which can legitimately
    start 0xBADx (an FF record whose block data is 0xBAD0..0xBADF).
    """
    if is_poison(value):
        raise BurnError(f"{name} poison/fuse-unit error: {hex32(value)}")


def evaluate_health(
    h: Health,
    plan: Plan | None = None,
    *,
    require_efp_one: bool = False,
) -> list[str]:
    """Evaluate all health gates and return every failing condition (never mutates state)."""
    del plan
    issues: list[str] = []
    for key in ("fusectrl", "dsov", "crc", "plm", "plm_f8", "en_sw", "debugctrl"):
        val = getattr(h, key)
        if is_poison(val):
            issues.append(f"{key} poison/fuse-unit error: {hex32(val)}")

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
    return issues


def check_health(
    h: Health,
    plan: Plan | None = None,
    *,
    require_efp_one: bool = False,
) -> None:
    """No repair, reset or retry is performed."""
    issues = evaluate_health(
        h,
        plan,
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
        if actual != value:
            raise BurnError(
                f"DIE ID MISMATCH row {i}: expected {hex32(value)}, got {hex32(actual)}"
            )


UNSENSED_SHADOWS_NOTE = (
    "Option shadows not sensed yet (DSOV=0, EFP=0, every compared option register reads 0): "
    "a Goldilocks frame before SENSE_CTRL. Identity masks are compared after SENSE_CTRL."
)


def option_shadows_unsensed(health: Health | None, readings: dict[str, int]) -> bool:
    """True only for the documented pre-SENSE Goldilocks signature.

    The glitch mis-senses the row 0/1 control fuses (DSOV=0, EFP=0) and leaves the
    option shadows unsensed, so they read 0 until SENSE_CTRL re-senses them
    (cmp170hx-8gb-vfi-tools docs/SWEET_SPOT.md, "the sentinel trap"). CMD_READ of
    fuse rows is unaffected. A normally booted card has DSOV=1, and a partly sensed
    frame has some nonzero shadow, so neither matches this and both are compared.
    """
    if health is None or health.dsov != 0 or health.efp != 0 or not readings:
        return False
    return all(v == 0 for v in readings.values())


def check_masks(
    expected_masks: dict[str, int],
    reader,
    *,
    health: Health | None = None,
    allow_unsensed: bool = False,
    show: Callable[[str], None] | None = None,
) -> bool:
    """Compare live option shadows with the plan's cold-boot masks.

    Returns True if compared, False if skipped because the shadows are provably
    unsensed (only when ``allow_unsensed``; read-only/dry-run paths). With
    ``health`` given and the unsensed signature present but not allowed, refuses
    with an explicit "run SENSE_CTRL first" error instead of a misleading mismatch.
    """
    readings: dict[str, int] = {}
    for field in expected_masks:
        actual = reader.field(field)
        _poison(f"field {field}", actual)
        readings[field] = actual
    if option_shadows_unsensed(health, readings):
        if not allow_unsensed:
            raise BurnError(
                "option shadows are unsensed (DSOV=0, EFP=0, all read 0): SENSE_CTRL has not run "
                "in this frame, so identity masks cannot be verified and nothing may be burned"
            )
        if show is not None:
            show(UNSENSED_SHADOWS_NOTE)
        return False
    for field, expected in expected_masks.items():
        actual = readings[field]
        if actual != expected:
            raise BurnError(
                f"IDENTITY MISMATCH {field}: expected {hex32(expected)}, got {hex32(actual)}"
            )
    return True


def preflight(
    plan: Plan,
    reader,
    *,
    active_sense: bool,
    match_masks: bool = True,
    check_die: bool = True,
    expected_masks: dict[str, int] | None = None,
    overrides: Overrides | None = None,
    show: Callable[[str], None] | None = None,
) -> str:
    """Run fresh hardware/simulator preflight checks and return current boot_id.

    Option-shadow masks are compared AFTER the active sense when one is requested:
    in a Goldilocks frame they are unsensed (read 0) until SENSE_CTRL. Without an
    active sense, the documented unsensed signature skips the comparison with a
    note (read-only preflight); everything else is compared as before.
    """
    ov = overrides or Overrides()
    host = reader.host()
    check_host(host)
    if reader.pci_id().lower() != plan.pci_id:
        raise BurnError(f"PCI identity mismatch: expected {plan.pci_id}, got {reader.pci_id()}")
    h = reader.health()
    check_health(h, plan)
    if check_die:
        check_die_rows(plan, reader)
    masks = expected_masks if expected_masks is not None else plan.expected_masks
    compare_masks = match_masks and not ov.mask_mismatch
    if compare_masks and not active_sense:
        check_masks(masks, reader, health=h, allow_unsensed=True, show=show)
    if active_sense:
        reader.active_sense_readonly()
        # The sense itself may have destroyed this frame: read everything AGAIN and require EFP=1.
        h_after = reader.health()
        check_health(
            h_after,
            plan,
            require_efp_one=True,
        )
        check_host(reader.host())
        if check_die:
            for i, value in plan.die_rows.items():
                actual = reader.die_row(i)
                if actual != value:
                    raise BurnError(f"die ID changed after active sense: row {i}")
        if compare_masks:
            # EFP=1 now, so the unsensed exemption can no longer apply: shadows must match.
            check_masks(masks, reader, health=h_after)
    return host.boot_id


def resolve_fpf_pointer(
    plan: Plan,
    full_rows: dict[int, int],
    *,
    override_slot: int | None = None,
) -> tuple[int, list[tuple[int, int]]]:
    """Validate contiguity and bounds on already-read FPF rows and return ``(pointer, nearby)``.

    Requires every row in ``plan.first_row..plan.last_row`` to be present in ``full_rows``
    (row-level fuse-unit errors are raised by ``FuseReader.fuse_row``).
    When ``override_slot`` is provided (after explicit operator confirmation), enforces
    row bounds and that the chosen slot reads ``plan.empty_word``.
    """
    a: list[int] = []
    for row_idx in range(plan.first_row, plan.last_row + 1):
        if row_idx not in full_rows:
            raise BurnError(f"FPF row {row_idx} missing from scan result")
        v = full_rows[row_idx]
        a.append(v)

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


def scan_fpf_rows(
    plan: Plan,
    reader,
    *,
    override_slot: int | None = None,
    passes: int = 1,
) -> tuple[int, list[tuple[int, int]], dict[int, int]]:
    """Scan full declared FPF region and return ``(pointer, nearby, full_rows)``.

    Defaults to a single full pass (``passes=1``), proving live prefix history and that all
    rows after the free pointer are empty without doubling ``CMD_READ`` traffic. Pass
    ``passes=2`` to additionally enforce multi-pass read stability.
    """
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 1:
        raise BurnError("passes must be a positive integer")

    def scan() -> list[int]:
        return [reader.fuse_row(i) for i in range(plan.first_row, plan.last_row + 1)]

    a = scan()
    for _ in range(passes - 1):
        b = scan()
        if a != b:
            raise BurnError("FPF region changed between scans; cannot prove next free slot")

    full_rows = {plan.first_row + i: v for i, v in enumerate(a)}
    pointer, nearby = resolve_fpf_pointer(plan, full_rows, override_slot=override_slot)
    return pointer, nearby, full_rows


def scan_slots(
    plan: Plan,
    reader,
    *,
    override_slot: int | None = None,
    passes: int = 1,
) -> tuple[int, list[tuple[int, int]]]:
    """Scan full declared FPF region (single pass by default). No heuristic for empties."""
    pointer, nearby, _ = scan_fpf_rows(plan, reader, override_slot=override_slot, passes=passes)
    return pointer, nearby


def verify_live_ff_prefix(
    plan: Plan,
    live_rows: dict[int, int],
    pointer: int,
    *,
    is_live: bool,
    stage: str,
    receipt: dict | None = None,
    is_manual_single: bool = False,
    allow_baseline_mismatch: bool = False,
) -> list[str]:
    """Verify every occupied live FF row matches the plan's baseline FF prefix snapshot.

    Returns a list of mismatch descriptions (empty when verified). Raises ``BurnError``
    unless ``allow_baseline_mismatch=True``.
    """
    issues: list[str] = []
    if plan.baseline_ff_rows is None:
        if is_live:
            issues.append(
                "plan lacks fpf.baseline_ff_rows snapshot; re-import from a full 512-row card "
                "dump so the live FF history can be verified (or use --override-baseline-mismatch)"
            )
    else:
        for row_idx, expected_word in sorted(plan.baseline_ff_rows.items()):
            actual_word = live_rows.get(row_idx)
            if actual_word != expected_word:
                actual_s = hex32(actual_word) if actual_word is not None else "MISSING"
                issues.append(
                    f"baseline FF row {row_idx} mismatch: live={actual_s} != plan={hex32(expected_word)}"
                )
        expected_next = (
            plan.expected_next_row
            if plan.expected_next_row is not None
            else (plan.first_row + len(plan.baseline_ff_rows))
        )
        if stage in ("test", "all"):
            if pointer != expected_next:
                issues.append(
                    f"stale plan or extra FF rows on card: live next free row is {pointer}, "
                    f"plan baseline expects {expected_next}"
                )
        else:
            if pointer < expected_next:
                issues.append(
                    f"live next free row {pointer} is behind plan baseline expected_next_row {expected_next}"
                )
            if receipt is not None and receipt.get("test_row") != expected_next:
                issues.append(
                    f"test receipt row {receipt.get('test_row')} disagrees with plan baseline "
                    f"expected_next_row {expected_next}"
                )
            if receipt is None and not is_manual_single and pointer > expected_next:
                # Without a receipt, any rows past expected_next must still match the plan prefix.
                extra_count = pointer - expected_next
                if extra_count > len(plan.records):
                    issues.append(
                        f"live next free row {pointer} has {extra_count} extra row(s) past "
                        f"baseline {expected_next} (exceeds plan length {len(plan.records)})"
                    )
                else:
                    for offset in range(extra_count):
                        row_idx = expected_next + offset
                        exp_rec = plan.records[offset]
                        got_w = live_rows.get(row_idx)
                        if got_w != exp_rec.word:
                            issues.append(
                                f"extra FF row {row_idx} reads {hex32(got_w) if got_w is not None else 'MISSING'}, "
                                f"expected plan record {exp_rec.name!r} ({hex32(exp_rec.word)})"
                            )

    if issues and not allow_baseline_mismatch:
        raise BurnError("FF PREFIX HISTORY MISMATCH: " + "; ".join(issues))
    return issues


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
    allow_unmapped_ff: bool = False,
    baseline_blocks: dict[int, dict[int, int]] | None = None,
    enforce_known_ff: bool = False,
    require_noop_test: bool = False,
) -> None:
    """Prove defect bits, approved FF effects, PCIe gates, and ceilings; never guesses unknown SM counts."""
    if record.word == plan.empty_word:
        raise BurnError(
            f"{record.name}: payload equals FPF empty_word ({hex32(plan.empty_word)})"
        )
    if is_poison(record.word):
        raise BurnError(
            f"{record.name}: payload {hex32(record.word)} has 0xBADx block data: deliberately "
            "not burned (it would read back like a fuse-unit error at the moment of programming)"
        )
    if is_debugctrl_target(record.target) and not allow_manual_debugctrl:
        raise BurnError("DEBUGCTRL must never be included in an automatic plan")

    # Word-derived checks: unconditionally decode the 32-bit GA100 FF word (IFF bit 5 == 0)
    # so checks cannot be bypassed by missing/incorrect plan metadata or synthetic fixtures.
    prior_masks = dict(assumed_masks)
    old_block: int | None = None
    touches_pcie = record.kind == "pcie"
    try:
        ff = decode_ff(record.word)
    except BurnError as exc:
        raise BurnError(f"{record.name}: not a valid GA100 FF record word ({exc})") from exc

    if baseline_blocks is not None:
        old_block = baseline_blocks.get(ff.chain, {}).get(ff.block)

    # Unconditional device-ID / burnable=False deny + unknown/unmapped FF safety audit.
    hard_denials, overridable_ff = ff_word_safety_audit(record.word, old_block)
    if hard_denials:
        raise BurnError(f"{record.name}: REFUSED; " + "; ".join(hard_denials))
    if enforce_known_ff and overridable_ff and not allow_unmapped_ff:
        raise BurnError(
            f"{record.name}: REFUSED; "
            + "; ".join(overridable_ff)
            + " (use --override-unmapped-ff only after independent verification)"
        )

    # Independent no-op proof for test records whenever a verified baseline is available.
    if record.kind == "test" and require_noop_test and baseline_blocks is not None:
        verify_noop_test_record(record.word, baseline_blocks)

    # Word-derived PCIe classification (cannot be evaded by setting kind="regular").
    if ff_word_touches_pcie(record.word, old_block):
        touches_pcie = True

    # Word-derived genuine defect audit. Known defects are NEVER overrideable; missing
    # defect masks are NOT bypassed by --override-undetermined-ceiling.
    violations, unknown = ff_word_defect_audit(record.word, plan.defect_masks)
    if violations:
        raise BurnError(f"{record.name}: REFUSED; " + "; ".join(violations))
    if unknown and not allow_unmapped_ff:
        raise BurnError(
            f"{record.name}: FF word touches {', '.join(sorted(set(unknown)))} but the plan has "
            "no genuine defect mask for it; cannot prove no defective unit is re-enabled"
        )
    word_mapped_fields = {
        FFCALC_KEY_TO_FIELD[f.key]
        for f in fields_in_block(ff.chain, ff.block)
        if f.key in FFCALC_KEY_TO_FIELD
    }
    word_masks = apply_ff_word_to_masks(record.word, assumed_masks)

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
        # Independent of from_disable: the declared end state must keep every defect disabled.
        if genuine & ~record.to_disable & 0xFFFFFFFF:
            raise BurnError(
                f"{record.name}: REFUSED; declared {field} end state {hex32(record.to_disable)} "
                f"leaves genuinely defective bits {hex32(genuine & ~record.to_disable)} enabled"
            )
        if field not in word_mapped_fields:
            raise BurnError(
                f"{record.name}: mask_update targets {field} but the FF word writes "
                f"{', '.join(sorted(word_mapped_fields)) or 'no known option field'}"
            )
        if field in word_masks and word_masks[field] != record.to_disable:
            raise BurnError(
                f"{record.name}: mask_update.to={hex32(record.to_disable)} for {field} disagrees with "
                f"what the FF word actually writes ({hex32(word_masks[field])})"
            )
        if _is_compute_mask_field(field) and record.resulting_sm_count is None:
            if not allow_undetermined_ceiling:
                raise BurnError(
                    f"{record.name}: resulting SM count undetermined; external review required"
                )
        assumed_masks[field] = record.to_disable
    elif record.defect_sensitive:
        raise BurnError(f"{record.name}: defect-sensitive change without mask map")

    # Propagate every word-derived mask (covers shared blocks touching several fields).
    untracked_compute = any(
        _is_compute_mask_field(k) and k not in prior_masks for k in word_mapped_fields
    )
    compute_changed = untracked_compute or any(
        _is_compute_mask_field(k) and prior_masks.get(k) != v for k, v in word_masks.items()
    )
    assumed_masks.update(word_masks)
    if compute_changed and not allow_undetermined_ceiling:
        tpc_keys = [f"TPC_GPC{i}_DISABLE" for i in range(TOTAL_GPC)]
        if "GPC_DISABLE" in assumed_masks and all(k in assumed_masks for k in tpc_keys):
            derived_sm = compute_sm_count(
                assumed_masks["GPC_DISABLE"], [assumed_masks[k] for k in tpc_keys]
            )
            if record.resulting_sm_count is not None and record.resulting_sm_count != derived_sm:
                raise BurnError(
                    f"{record.name}: declared resulting_sm_count={record.resulting_sm_count} but the "
                    f"FF word yields {derived_sm} SM"
                )
            ceiling = plan.override_sm_ceiling if record.kind == "reconfig" else plan.base_sm_ceiling
            if ceiling is None or derived_sm > ceiling:
                raise BurnError(
                    f"{record.name}: FF word yields {derived_sm} SM, beyond supplied ceiling {ceiling}"
                )
        elif record.resulting_sm_count is None:
            raise BurnError(
                f"{record.name}: changes GPC/TPC masks but the plan lacks GPC_DISABLE + all eight "
                "TPC_GPCi_DISABLE expected masks; resulting SM count undetermined"
            )

    if record.kind == "reconfig":
        if plan.override_sm_ceiling is None or not plan.reconfig_semantics_verified:
            raise BurnError("RECONFIG override ceiling/semantics unverified")

    if record.resulting_sm_count is not None and not allow_undetermined_ceiling:
        ceiling = plan.override_sm_ceiling if record.kind == "reconfig" else plan.base_sm_ceiling
        if ceiling is None or record.resulting_sm_count > ceiling:
            raise BurnError(
                f"{record.name}: predicted {record.resulting_sm_count} SM beyond supplied ceiling {ceiling}"
            )

    if touches_pcie and (
        record.link_cap2_before is None or record.link_cap2_after is None
    ):
        raise BurnError(
            "PCIe record (or FF word modifying PCIe speed/boot fuses) must declare reviewed LnkCap2 before/after"
        )

    if baseline_blocks is not None:
        baseline_blocks.setdefault(ff.chain, {})[ff.block] = ff.data


def check_plan(
    plan: Plan,
    *,
    allow_manual_debugctrl: bool = False,
    allow_undetermined_ceiling: bool = False,
    allow_unmapped_ff: bool = False,
) -> dict[str, int]:
    assumed = plan.expected_masks.copy()
    sim_blocks = (
        replay_ff_rows(plan.baseline_ff_rows) if plan.baseline_ff_rows is not None else None
    )
    has_baseline = sim_blocks is not None
    record_cold: dict[str, int] = {}
    for r in plan.records:
        reconcile(
            plan,
            r,
            assumed,
            allow_manual_debugctrl=allow_manual_debugctrl,
            allow_undetermined_ceiling=allow_undetermined_ceiling,
            allow_unmapped_ff=allow_unmapped_ff,
            baseline_blocks=sim_blocks,
            enforce_known_ff=has_baseline,
            require_noop_test=has_baseline,
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
    show("\n" + "!" * 72)
    show(f"!!! WARNING — MANUAL SAFETY OVERRIDE ENABLED: [{name}]")
    show(f"!!! {banner}")
    show("!!! OTP fuse burns are PERMANENT and IRREVERSIBLE.")
    show("!" * 72)
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


@dataclass(frozen=True)
class PlanProgress:
    """Passive detection of how much of a plan is already burned in card FPF rows."""

    state: str  # "fresh", "resume", "complete", or "mismatch"
    committed_count: int
    total_count: int
    start_row: int | None
    next_row: int | None
    detail: str = ""


def _journal_test_row(plan: Plan, journal_path: str | Path | None) -> int | None:
    if not journal_path:
        return None
    p = Path(journal_path)
    if not p.exists():
        return None
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    test_rec = plan.records[0]
    expected_word = hex32(test_rec.word)
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            item.get("event") == "verified_readback_only"
            and item.get("plan_sha256") == plan.content_hash
            and item.get("record") == test_rec.name
            and item.get("intended") == expected_word
            and isinstance(item.get("row"), int)
            and not isinstance(item.get("row"), bool)
        ):
            return int(item["row"])
    return None


def detect_plan_progress(
    plan: Plan,
    reader,
    *,
    journal_path: str | Path | None = None,
    receipt_path: str | Path | None = None,
    override_slot: int | None = None,
) -> PlanProgress:
    """Passively determine whether the card is fresh, mid-plan (resumable), or complete.

    Issues only minimal CMD_READs (1 row on a fresh card when ``baseline_ff_rows`` is
    present) and never triggers SENSE_CTRL or CMD_PROGRAM.
    """
    total = len(plan.records)

    def _match_from(start_row: int) -> PlanProgress:
        for offset in range(total):
            row_idx = start_row + offset
            if row_idx > plan.last_row:
                return PlanProgress(
                    state="fresh" if offset == 0 else "resume",
                    committed_count=offset,
                    total_count=total,
                    start_row=start_row,
                    next_row=row_idx,
                )
            word = reader.fuse_row(row_idx)
            if word == plan.empty_word:
                if offset == 0:
                    return PlanProgress(
                        state="fresh",
                        committed_count=0,
                        total_count=total,
                        start_row=start_row,
                        next_row=override_slot if override_slot is not None else start_row,
                    )
                return PlanProgress(
                    state="resume",
                    committed_count=offset,
                    total_count=total,
                    start_row=start_row,
                    next_row=row_idx,
                )
            expected = plan.records[offset]
            if word != expected.word:
                if offset == 0:
                    detail = (
                        f"FPF row {row_idx} reads {hex32(word)}, expected empty "
                        f"or plan test record {expected.name!r} ({hex32(expected.word)})"
                    )
                else:
                    detail = (
                        f"FPF row {row_idx} reads {hex32(word)}, expected empty "
                        f"or plan record {expected.name!r} ({hex32(expected.word)})"
                    )
                return PlanProgress(
                    state="mismatch",
                    committed_count=offset,
                    total_count=total,
                    start_row=start_row,
                    next_row=None,
                    detail=detail,
                )
        return PlanProgress(
            state="complete",
            committed_count=total,
            total_count=total,
            start_row=start_row,
            next_row=start_row + total,
        )

    # 1. Authoritative baseline snapshot (present on all live/imported plans): 1 read if fresh.
    if plan.baseline_ff_rows is not None:
        expected_next = (
            plan.expected_next_row
            if plan.expected_next_row is not None
            else (plan.first_row + len(plan.baseline_ff_rows))
        )
        return _match_from(expected_next)

    # 2. Without baseline_ff_rows: check receipt if supplied.
    if receipt_path is not None:
        receipt = _load_test_receipt(plan, receipt_path)
        return _match_from(receipt["test_row"])

    # 3. Check session journal for a prior verified write of this plan's test record.
    j_row = _journal_test_row(plan, journal_path)
    if j_row is not None and plan.first_row <= j_row <= plan.last_row:
        w = reader.fuse_row(j_row)
        if w == plan.records[0].word:
            return _match_from(j_row)

    # 4. Fallback scan until first empty row.
    if override_slot is not None:
        return PlanProgress(
            state="fresh",
            committed_count=0,
            total_count=total,
            start_row=override_slot,
            next_row=override_slot,
        )

    occupied: list[tuple[int, int]] = []
    pointer = plan.last_row + 1
    for r in range(plan.first_row, plan.last_row + 1):
        w = reader.fuse_row(r)
        if w == plan.empty_word:
            pointer = r
            break
        occupied.append((r, w))

    # Without baseline, receipt, or journal evidence, only auto-detect multi-record
    # plan suffixes (>= 2 matching records) so a factory row can never be mistaken for a no-op test.
    if total >= 2 and len(occupied) >= 2:
        max_k = min(total, len(occupied))
        for k in range(max_k, 1, -1):
            tail_words = [w for _, w in occupied[-k:]]
            plan_words = [rec.word for rec in plan.records[:k]]
            if tail_words == plan_words:
                start_row = occupied[-k][0]
                return PlanProgress(
                    state="complete" if k == total else "resume",
                    committed_count=k,
                    total_count=total,
                    start_row=start_row,
                    next_row=pointer,
                )

    return PlanProgress(
        state="fresh",
        committed_count=0,
        total_count=total,
        start_row=pointer,
        next_row=pointer,
    )


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


def _write_verify_or_halt(
    plan: Plan,
    reader,
    writer,
    journal: Journal,
    *,
    pointer: int,
    record: Record,
    ov: Overrides,
    manual_record_sha256: str | None,
) -> tuple[int, Health, Host]:
    """Issue ONE record write, then read-back + full post-write gate. Never retries."""
    readback: int | None = None
    health: Health | None = None
    host_after: Host | None = None
    try:
        writer.write_one_record(pointer, record.word)
    except PreTriggerRefusal as exc:
        # The writer guarantees CMD_PROGRAM was never issued: nothing can have burned.
        journal.append(
            event="aborted_pre_trigger",
            row=pointer,
            record=record.name,
            error=str(exc),
            write_issued=False,
            pointer_advanced=False,
            manual_record_sha256=manual_record_sha256,
        )
        raise BurnError(f"ABORTED before CMD_PROGRAM (NO write issued): {exc}") from exc
    except BaseException as exc:
        _journal_halt(journal, reader, pointer, record, exc, manual_record_sha256)
        if not isinstance(exc, Exception):
            raise
        raise BurnError(f"HALT: row {pointer} may be partially programmed; do NOT retry: {exc}") from exc
    try:
        readback = reader.fuse_row(pointer)
        if readback != record.word:
            raise BurnError(
                f"row {pointer} READBACK MISMATCH: {hex32(readback)} != {hex32(record.word)}"
            )
        health = reader.health()
        check_health(health, plan)
        host_after = reader.host()
        check_host(host_after)
        # Extra slot check: no pointer advance before every post-write gate passes.
        second_readback = reader.fuse_row(pointer)
        if second_readback != record.word:
            raise BurnError(
                f"row {pointer} changed after post-write gate: {hex32(second_readback)}"
            )
    except BaseException as exc:
        _journal_halt(
            journal, reader, pointer, record, exc, manual_record_sha256,
            readback=readback, health=health, host_after=host_after,
        )
        if not isinstance(exc, Exception):
            raise
        raise BurnError(f"HALT: row {pointer} may be partially programmed; do NOT retry: {exc}") from exc
    return readback, health, host_after


def _journal_halt(
    journal: Journal,
    reader,
    pointer: int,
    record: Record,
    exc: BaseException,
    manual_record_sha256: str | None,
    *,
    readback: int | None = None,
    health: Health | None = None,
    host_after: Host | None = None,
) -> None:
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
        error=f"{type(exc).__name__}: {exc}",
        pointer_advanced=False,
        manual_record_sha256=manual_record_sha256,
    )


def _run_poweroff_sanity(
    plan: Plan,
    current_rows: dict[int, int],
    journal: Journal,
    show,
    *,
    committed_records: list[Record],
    remaining_records: list[Record],
    baseline_blocks: dict[int, dict[int, int]] | None,
    reason: str,
) -> dict | None:
    """Evaluate current FPF rows (initial scan + read-back-verified writes) for a safe power-off boundary."""
    try:
        sanity = evaluate_poweroff_sanity(
            current_rows,
            plan,
            committed_records=committed_records,
            remaining_records=remaining_records,
            baseline_blocks=baseline_blocks,
        )
    except Exception as exc:
        show(f"\nNOTE: Pre-power-off fuse sanity check skipped ({exc}).")
        return None
    for line in sanity["lines"]:
        show(line)
    journal.append(
        event="poweroff_sanity_check",
        reason=reason,
        safe_to_power_off=sanity["safe_to_power_off"],
        committed_groups=sanity["committed_groups"],
        remaining_groups=sanity["remaining_groups"],
        split_groups=sanity["split_groups"],
        sm_count=sanity["sm_count"],
        errors=sanity["errors"],
        warnings=sanity["warnings"],
    )
    return sanity


def _has_split_poweroff_domains(
    committed_records: list[Record],
    remaining_records: list[Record],
    baseline_blocks: dict[int, dict[int, int]] | None,
) -> bool:
    """Return True if any atomic power-off domain spans both committed and remaining records."""
    walk = {c: dict(b) for c, b in baseline_blocks.items()} if baseline_blocks is not None else None

    def _collect(records: list[Record]) -> set[str]:
        domains: set[str] = set()
        for rec in records:
            if walk is not None:
                try:
                    ff = decode_ff(rec.word)
                    old_b = walk.get(ff.chain, {}).get(ff.block)
                    groups = classify_ff_functional_groups(rec.word, old_b)
                    walk.setdefault(ff.chain, {})[ff.block] = ff.data
                except BurnError:
                    groups = {"unknown"}
            else:
                groups = classify_ff_functional_groups(rec.word, None)
            domains.update(poweroff_domains_for_groups(groups))
        return domains

    committed_domains = _collect(committed_records)
    remaining_domains = _collect(remaining_records)
    return bool(committed_domains & remaining_domains)


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
    is_live = not getattr(reader, "simulation", False)
    strict_ff = is_live or (plan.baseline_ff_rows is not None)
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
        allow_unmapped_ff=ov.unmapped_ff,
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
                banner=(
                    "Allowing operator-supplied SM ceiling override "
                    "(known genuine defect bits and missing defect masks remain REFUSED)."
                ),
                token="OVERRIDE CEILING",
            )
        if ov.unmapped_ff:
            _require_override_confirm(
                confirm,
                journal,
                show,
                name="unmapped_ff",
                banner=(
                    "Allowing unknown FF blocks, changed unmapped/opt-in bits, or fields missing "
                    "defect masks (known genuine defect bits and read-only device-ID blocks remain REFUSED)."
                ),
                token="OVERRIDE UNMAPPED FF",
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

    try:
        pointer, nearby, live_rows = scan_fpf_rows(plan, reader, override_slot=ov.slot)
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

    try:
        baseline_issues = verify_live_ff_prefix(
            plan,
            live_rows,
            pointer,
            is_live=is_live,
            stage=stage,
            receipt=receipt,
            is_manual_single=is_manual_single,
            allow_baseline_mismatch=ov.baseline_mismatch,
        )
    except Exception as exc:
        h_snap, host_snap = _capture_snapshot(reader)
        journal.append(
            event="preflight_failed",
            stage=stage,
            plan_sha256=plan.content_hash,
            phase="baseline_ff_prefix",
            active_sense=use_active_sense,
            health=h_snap,
            host=host_snap,
            error=str(exc),
        )
        raise

    if baseline_issues and ov.baseline_mismatch and not dry_run:
        _require_override_confirm(
            confirm,
            journal,
            show,
            name="baseline_mismatch",
            banner="Live FF prefix disagrees with (or plan lacks) baseline_ff_rows: "
            + "; ".join(baseline_issues),
            token="OVERRIDE BASELINE MISMATCH",
            details={"issues": baseline_issues},
        )

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
    already_committed: list[Record] = []
    pre_commit_blocks = (
        replay_ff_rows(plan.baseline_ff_rows) if plan.baseline_ff_rows is not None else None
    )
    if stage == "test":
        pending = list(plan.records[:1])
        tail_after_stage: list[Record] = list(plan.records[1:])
    elif stage == "all":
        pending = list(plan.records)
        tail_after_stage = []
    else:
        reconcile(
            plan,
            plan.records[0],
            assumed,
            allow_manual_debugctrl=effective_debugctrl,
            allow_undetermined_ceiling=ov.undetermined_ceiling,
            allow_unmapped_ff=ov.unmapped_ff,
            baseline_blocks=pre_commit_blocks,
            enforce_known_ff=pre_commit_blocks is not None,
            require_noop_test=pre_commit_blocks is not None,
        )
        already_committed.append(plan.records[0])
        tail_after_stage = []
        if is_manual_single:
            if receipt is not None:
                test_row = receipt["test_row"]
                actual_test = live_rows[test_row] if test_row in live_rows else reader.fuse_row(test_row)
                if actual_test != plan.records[0].word or pointer < test_row + 1:
                    raise BurnError(
                        "cold-verified test record not at expected location; refuse manual write"
                    )
            pending = list(plan.records[1:])
        else:
            resume_test_row: int | None = None
            if receipt is not None:
                resume_test_row = receipt["test_row"]
                actual_test = (
                    live_rows[resume_test_row]
                    if resume_test_row in live_rows
                    else reader.fuse_row(resume_test_row)
                )
                if actual_test != plan.records[0].word or pointer < resume_test_row + 1:
                    raise BurnError(
                        "cold-verified test record not at expected location; refuse real stage"
                    )
            elif plan.baseline_ff_rows is not None:
                expected_next = (
                    plan.expected_next_row
                    if plan.expected_next_row is not None
                    else (plan.first_row + len(plan.baseline_ff_rows))
                )
                if pointer > expected_next:
                    resume_test_row = expected_next

            if resume_test_row is not None:
                already_written = pointer - (resume_test_row + 1)
                if already_written > len(plan.records) - 1:
                    raise BurnError(
                        f"FPF pointer {pointer} is past end of plan "
                        f"(test_row={resume_test_row}, {len(plan.records) - 1} real records)"
                    )
                for offset in range(1, already_written + 1):
                    row_idx = resume_test_row + offset
                    expected_rec = plan.records[offset]
                    actual_word = live_rows[row_idx] if row_idx in live_rows else reader.fuse_row(row_idx)
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
                        allow_unmapped_ff=ov.unmapped_ff,
                        baseline_blocks=pre_commit_blocks,
                        enforce_known_ff=pre_commit_blocks is not None,
                        require_noop_test=False,
                    )
                    already_committed.append(expected_rec)
                if already_written > 0:
                    show(
                        f"Resuming real stage: {already_written} record(s) already committed "
                        f"in rows {resume_test_row + 1}..{pointer - 1}."
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
            # A passive dry-run in a Goldilocks frame sees unsensed (zero) shadows: skip the
            # comparison with a journaled note. Real burns never skip it: --execute refuses
            # unsensed shadows, and after active sense EFP=1 so the exemption cannot apply.
            masks_compared = check_masks(
                assumed,
                reader,
                health=reader.health(),
                allow_unsensed=dry_run,
                show=show,
            )
            if not masks_compared:
                journal.append(
                    event="mask_check_deferred",
                    stage=stage,
                    plan_sha256=plan.content_hash,
                    reason="option shadows unsensed (DSOV=0, EFP=0, all compared fields read 0)",
                    dry_run=True,
                )
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

    deferred_by_max: list[Record] = []
    if max_records is not None:
        deferred_by_max = pending[max_records:]
        pending = pending[:max_records]

    # Build verified block image from live_rows when enforcing exact FF before/after proofs.
    working_blocks: dict[int, dict[int, int]] | None = None
    initial_live_blocks: dict[int, dict[int, int]] | None = None
    if strict_ff:
        working_blocks = replay_ff_rows(live_rows)
        initial_live_blocks = {c: dict(b) for c, b in working_blocks.items()}
    else:
        try:
            initial_live_blocks = replay_ff_rows(live_rows)
        except BurnError:
            initial_live_blocks = None

    start_health, start_host = _capture_snapshot(reader)
    journal.append(
        event="session_start",
        plan_sha256=plan.content_hash,
        pci_id=plan.pci_id,
        stage=stage,
        boot_id=boot_id,
        held_frame_source="operator_supplied_live_adapter" if is_live else "fake_simulation",
        baseline_ff_sha256=plan.baseline_ff_sha256,
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
        dry_blocks = {c: dict(b) for c, b in working_blocks.items()} if working_blocks is not None else None
        for offset, record in enumerate(pending):
            old_b = None
            if dry_blocks is not None:
                try:
                    ff_d = decode_ff(record.word)
                    old_b = dry_blocks.get(ff_d.chain, {}).get(ff_d.block)
                except BurnError:
                    old_b = None
            groups = sorted(classify_ff_functional_groups(record.word, old_b))
            reconcile(
                plan,
                record,
                dry_assumed,
                allow_manual_debugctrl=effective_debugctrl,
                allow_undetermined_ceiling=ov.undetermined_ceiling,
                allow_unmapped_ff=ov.unmapped_ff,
                baseline_blocks=dry_blocks,
                enforce_known_ff=strict_ff,
                require_noop_test=strict_ff,
            )
            slot = pointer + offset
            show(
                f"DRY-RUN [{record.kind.upper()}]: would write row {slot} "
                f"{record.target}={hex32(record.word)} [{'+'.join(groups)}] ({record.decode})"
            )
        journal.append(
            event="dry_run_complete",
            stage=stage,
            pointer=pointer,
            active_sense=use_active_sense,
            records=[r.name for r in pending],
        )
        return pointer

    written_records: list[Record] = []
    verified_rows: dict[int, int] = dict(live_rows)
    prev_groups: set[str] | None = None
    sanity_baseline = (
        replay_ff_rows(plan.baseline_ff_rows)
        if plan.baseline_ff_rows is not None
        else initial_live_blocks
    )

    for idx, record in enumerate(pending):
        rec_old_block: int | None = None
        if working_blocks is not None:
            try:
                ff_cur = decode_ff(record.word)
                rec_old_block = working_blocks.get(ff_cur.chain, {}).get(ff_cur.block)
            except BurnError:
                rec_old_block = None
        rec_touches_pcie = (record.kind == "pcie") or ff_word_touches_pcie(record.word, rec_old_block)
        rec_groups = classify_ff_functional_groups(record.word, rec_old_block)

        reconcile(
            plan,
            record,
            assumed,
            allow_manual_debugctrl=effective_debugctrl,
            allow_undetermined_ceiling=ov.undetermined_ceiling,
            allow_unmapped_ff=ov.unmapped_ff,
            baseline_blocks=working_blocks,
            enforce_known_ff=strict_ff,
            require_noop_test=strict_ff,
        )

        if idx > 0:
            # Repeat passive host, PCI identity, and health checks plus pointer bounds before
            # each subsequent record. SENSE_CTRL, die-ID verification, and the full FPF scan
            # already ran once at session start; BDF lock + target-row blank check + dual
            # read-back after each burn cover mid-session state without re-scanning all rows.
            try:
                preflight(
                    plan,
                    reader,
                    active_sense=False,
                    match_masks=False,
                    check_die=False,
                    overrides=ov,
                )
                if pointer > plan.last_row:
                    raise BurnError("FPF region full; no free record")
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

        prev_domains = poweroff_domains_for_groups(prev_groups) if prev_groups is not None else set()
        rec_domains = poweroff_domains_for_groups(rec_groups)
        if (
            prev_domains
            and rec_domains
            and not (prev_domains & rec_domains)
            and not _has_split_poweroff_domains(
                already_committed + written_records,
                pending[idx:] + deferred_by_max + tail_after_stage,
                sanity_baseline,
            )
        ):
            show(
                f"\n--- Functional group boundary: completed [{', '.join(sorted(prev_groups))}]; "
                f"next is [{', '.join(sorted(rec_groups))}] (clean point to stop & restart if desired) ---"
            )

        previous = reader.fuse_row(pointer)
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
        group_tag = "+".join(sorted(rec_groups))
        show(
            f"{record.kind.upper()}[{group_tag}]: row {pointer} {record.target}={hex32(record.word)} "
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
                    if written_records:
                        _run_poweroff_sanity(
                            plan, verified_rows, journal, show,
                            committed_records=already_committed + written_records,
                            remaining_records=pending[idx:] + deferred_by_max + tail_after_stage,
                            baseline_blocks=sanity_baseline,
                            reason="operator_declined",
                        )
                    raise BurnError("DEBUGCTRL separately confirmed manual action declined")

        if rec_touches_pcie:
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

        if rec_touches_pcie or record.kind == "reconfig":
            review_kind = "PCIE" if rec_touches_pcie else record.kind.upper()
            token = f"REVIEW {review_kind}"
            if not _confirm_token(
                confirm,
                f"{token}: externally verify current link/ceiling and type '{token}': ",
                token,
            ):
                journal.append(event="declined", row=pointer, record=record.name, reason=token)
                if written_records:
                    _run_poweroff_sanity(
                        plan, verified_rows, journal, show,
                        committed_records=already_committed + written_records,
                        remaining_records=pending[idx:] + deferred_by_max + tail_after_stage,
                        baseline_blocks=sanity_baseline,
                        reason="operator_declined",
                    )
                raise BurnError("special record not confirmed")

        confirm_word = f"YES {pointer} {hex32(record.word)}" if strict_confirm else "YES"
        if not _confirm_token(
            confirm,
            f"PERMANENT: type '{confirm_word}' to burn row {pointer} (or Enter to stop): ",
            confirm_word,
        ):
            journal.append(event="declined", row=pointer, record=record.name)
            if written_records:
                _run_poweroff_sanity(
                    plan, verified_rows, journal, show,
                    committed_records=already_committed + written_records,
                    remaining_records=pending[idx:] + deferred_by_max + tail_after_stage,
                    baseline_blocks=sanity_baseline,
                    reason="operator_declined",
                )
            raise BurnError("operator declined")

        # Instantaneous pre-trigger gate: closes the window while the human was reading/typing.
        try:
            check_host(reader.host())
            check_health(reader.health(), plan)
            pre_slot = reader.fuse_row(pointer)
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

        # Irreversible section: Ctrl-C / SSH hang-up are held back until the outcome of
        # this record (verified, aborted pre-trigger, or HALT) is fsynced to the journal.
        with deferred_signals():
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
            readback, health, host_after = _write_verify_or_halt(
                plan,
                reader,
                writer,
                journal,
                pointer=pointer,
                record=record,
                ov=ov,
                manual_record_sha256=manual_record_sha256,
            )
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
        verified_rows[pointer] = readback
        pointer += 1  # The ONLY advance, after read-back AND healthy gate.
        written_records.append(record)
        prev_groups = rec_groups

        if stage == "all" and record.kind == "test" and idx + 1 < len(pending):
            show(
                f"\nTEST ROW READ-BACK VERIFIED (row {pointer - 1} = {hex32(readback)}, "
                f"CRC={hex32(health.crc)}, FUSECTRL={hex32(health.fusectrl)})."
            )
            show(
                "NOTE: Read-back confirms the test write landed in the fuse array, NOT a cold-boot re-parse.\n"
                "You can continue burning the real records in this same glitch session, "
                "or stop now to power-cycle and cold-boot verify the test row first."
            )
            if not _confirm_token(
                confirm,
                "Type 'CONTINUE SAME WINDOW' to burn real records now "
                "(or press Enter to stop and cold-boot verify first): ",
                "CONTINUE SAME WINDOW",
            ):
                _run_poweroff_sanity(
                    plan, verified_rows, journal, show,
                    committed_records=already_committed + written_records,
                    remaining_records=pending[idx + 1 :] + deferred_by_max + tail_after_stage,
                    baseline_blocks=sanity_baseline,
                    reason="paused_after_test",
                )
                journal.append(
                    event="stage_all_paused_after_test",
                    test_row=pointer - 1,
                    next_pointer=pointer,
                    remaining_records=[r.name for r in pending[idx + 1 :]],
                    cold_boot_verified=False,
                )
                show(
                    "Paused cleanly after test row. Power-cycle to cold-boot verify "
                    "(menu [10] Verify Cold, or CLI: verify-cold --stage test --receipt-out <file>), "
                    "then resume with menu [6] Burn Plan (or CLI: --stage real --receipt <file>)."
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
        records_written=len(written_records),
        cold_boot_verified=False,
    )
    if written_records:
        _run_poweroff_sanity(
            plan, verified_rows, journal, show,
            committed_records=already_committed + written_records,
            remaining_records=deferred_by_max + tail_after_stage,
            baseline_blocks=sanity_baseline,
            reason="session_complete",
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

        ctrl_state = cold.get("controller_state")
        if isinstance(ctrl_state, dict) and ctrl_state:
            if "dsov" in ctrl_state and u32(ctrl_state["dsov"], "cold.controller_state.dsov") == 0:
                raise BurnError(
                    "cold snapshot controller_state has DSOV=0 (still in a held glitch window; "
                    "power-cycle card completely before capturing cold snapshot)"
                )
            if "en_sw" in ctrl_state and (u32(ctrl_state["en_sw"], "cold.controller_state.en_sw") & 1) == 1:
                raise BurnError(
                    "cold snapshot controller_state has EN_SW_OVERRIDE=1 (not a fresh cold boot)"
                )

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
        expected_written_rows: dict[int, int] = {}
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
            expected_written_rows[test_row] = plan.records[0].word
        else:
            # Every journaled write for this plan must persist across the cold boot.
            # Snapshots store only NON-ZERO rows, so an absent row means it reads 0x00000000
            # (did not persist) -- it must fail, not be skipped.
            raw_rows = cold.get("fpf_rows")
            if not isinstance(raw_rows, dict) or not raw_rows:
                raise BurnError(
                    "real-stage cold snapshot has no fpf_rows (captured with --no-rows?); "
                    "cannot prove the burned FF rows persisted"
                )
            written = [
                item
                for item in prior
                if item.get("event") == "verified_readback_only"
                and item.get("plan_sha256") == plan.content_hash
                and item.get("manual_record_sha256") is None
            ]
            if not written:
                raise BurnError("no journaled verified writes for this plan to cold-verify")
            for item in written:
                row_idx = int(item["row"])
                row_key = str(row_idx)
                expected_word = u32(item["intended"], "journal.intended")
                observed_word = (
                    u32(raw_rows[row_key], f"cold.fpf_rows.{row_key}") if row_key in raw_rows else 0
                )
                if observed_word != expected_word:
                    raise BurnError(
                        f"cold FPF row {row_key} ({hex32(observed_word)}) "
                        f"does not match written record {item['record']!r} ({hex32(expected_word)})"
                    )
                expected_written_rows[row_idx] = expected_word

        # When the plan includes baseline_ff_rows, verify the full FF row image in the snapshot.
        if plan.baseline_ff_rows is not None and isinstance(raw_rows, dict):
            parsed_cold_rows = {
                int(k): u32(v, f"cold.fpf_rows.{k}")
                for k, v in raw_rows.items()
                if u32(v, f"cold.fpf_rows.{k}") != plan.empty_word
            }
            expected_full = {**plan.baseline_ff_rows, **expected_written_rows}
            for r_idx, exp_w in sorted(expected_full.items()):
                got_w = parsed_cold_rows.get(r_idx, plan.empty_word)
                if got_w != exp_w:
                    raise BurnError(
                        f"cold snapshot FF row {r_idx} ({hex32(got_w)}) disagrees with "
                        f"expected baseline+burned image ({hex32(exp_w)})"
                    )
            unexpected = sorted(set(parsed_cold_rows) - set(expected_full))
            if unexpected:
                raise BurnError(
                    f"cold snapshot contains unexpected extra FF row(s): {unexpected[:5]}"
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
