"""Interactive manual fuse-controller console (BURN_SPEC §7).

Manual means the operator chooses what to read or which single record to write;
the safety gates (aligned 32-bit stores, empty target slot, mandatory read-back,
post-write poison/PLM/latch gate, and journal logging) never come off.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import shlex

from .backend import MissingWriter
from .engine import (
    Journal,
    Overrides,
    check_host_readonly_safe,
    check_plan,
    evaluate_health,
    evaluate_host,
    format_pmbus_status,
    hex32,
    process_records,
    scan_slots,
)
from .ffrecord import (
    FF_ROW_MAX,
    FF_ROW_MIN,
    TOTAL_GPC,
    analyze_ff_block,
    classify_ff_functional_groups,
    compute_sm_count,
    decode_all_fields,
    decode_ff,
    replay_ff_rows,
)
from .ga100 import EXPERIMENTAL_INSPECT_REGS, decode_experimental_register
from .mmio import REG_FUSECTRL, resolve_register_address
from .model import BurnError, Plan, Record, is_poison, u32

_EXP_BY_ADDR = {reg.address: reg for reg in EXPERIMENTAL_INSPECT_REGS}
_EXP_BY_NAME = {}
for _reg in EXPERIMENTAL_INSPECT_REGS:
    _EXP_BY_NAME[_reg.name] = _reg
    for _alias in _reg.aliases:
        _EXP_BY_NAME[_alias] = _reg


def read_experimental_registers(
    reader, *, fusectrl: int | None = None
) -> tuple[dict[str, int], bool]:
    """Read EXPERIMENTAL_INSPECT_REGS safely (read-only).

    Checks FUSECTRL first and immediately stops without touching other BAR0 registers
    if FUSECTRL is poisoned or link-down (0xFFFFFFFF).
    Returns ``(readings_by_name, skipped_due_to_poison)``.
    """
    if fusectrl is None:
        try:
            fusectrl = reader.register(REG_FUSECTRL)
        except Exception:
            try:
                fusectrl = reader.health().fusectrl
            except Exception:
                fusectrl = None
    if fusectrl is not None and (fusectrl == 0xFFFFFFFF or is_poison(fusectrl)):
        return {}, True

    readings: dict[str, int] = {}
    for reg in EXPERIMENTAL_INSPECT_REGS:
        val: int | None = None
        try:
            val = reader.register(reg.address)
        except Exception:
            try:
                val = reader.field(reg.name)
            except Exception:
                val = None
        if val is not None:
            readings[reg.name] = val
    return readings, False


def format_experimental_report(
    reader,
    *,
    fusectrl: int | None = None,
    readings: dict[str, int] | None = None,
    poisoned: bool | None = None,
) -> list[str]:
    """Format the full read-only experimental GA100 register readout & reference table."""
    if readings is None or poisoned is None:
        readings, poisoned = read_experimental_registers(reader, fusectrl=fusectrl)
    lines = [
        "=== EXPERIMENTAL READ-ONLY REGISTERS (UNMAPPED IN FF PLANS — NEVER BURNED) ===",
        "  NOTE: Diagnostic BAR0 readouts from GA100 cross-variant research (170HX vs A100).",
        "        These are NOT mapped to FF burn chains and are never written by vfi_burn.",
    ]
    if poisoned:
        lines.append(
            "  SKIPPED: FUSECTRL is poisoned/link-down — refusing further BAR0 reads for safety."
        )
        return lines

    if not readings:
        lines.append(
            "  NOTE: Target snapshot has no experimental registers populated (showing reference map)."
        )

    current_cat: str | None = None
    for reg in EXPERIMENTAL_INSPECT_REGS:
        if reg.category != current_cat:
            current_cat = reg.category
            lines.append(f"  [{current_cat}]")
        ref_str = f"170HX={reg.cmp170hx_ref} / A100={reg.a100_ref}"
        if reg.name in readings:
            val = readings[reg.name]
            note = decode_experimental_register(reg, val)
            lines.append(
                f"    {reg.name:18} ({hex32(reg.address)}) = {hex32(val)}  | {note}  [{ref_str}]"
            )
        else:
            lines.append(
                f"    {reg.name:18} ({hex32(reg.address)}) = {'n/a':10}  | {reg.description}  [{ref_str}]"
            )
    return lines


def _format_experimental_compact_lines(readings: dict[str, int]) -> list[str]:
    """Build compact summary lines for format_status_report when experimental regs are present."""
    parts: list[str] = []
    if "FUSE_ECC_EN" in readings:
        v = readings["FUSE_ECC_EN"]
        parts.append(f"FUSE_ECC_EN={hex32(v)} ({decode_experimental_register(_EXP_BY_NAME['FUSE_ECC_EN'], v)})")
    if "FUSE_SKU_ID" in readings:
        v = readings["FUSE_SKU_ID"]
        parts.append(f"FUSE_SKU_ID={hex32(v)} ({decode_experimental_register(_EXP_BY_NAME['FUSE_SKU_ID'], v)})")
    ss_names = (
        "FUSE_SS_FFMA",
        "FUSE_SS_FMLA16",
        "FUSE_SS_FMLA32",
        "FUSE_SS_IMLA0",
        "FUSE_SS_IMLA1",
        "FUSE_SS_IMLA2",
        "FUSE_SS_IMLA3",
        "FUSE_SS_IMLA4",
    )
    ss_vals = [readings[n] for n in ss_names if n in readings]
    if "FUSE_SS_DP" in readings or ss_vals:
        dp_part = f"DP={readings['FUSE_SS_DP'] & 1}" if "FUSE_SS_DP" in readings else ""
        if ss_vals:
            uniq = sorted(set(ss_vals))
            if len(uniq) == 1:
                tag = "FULL" if uniq[0] == 0 else ("MAX THROTTLE" if uniq[0] == 5 else f"level {uniq[0]}")
                vec_part = f"FMA/MLA(x{len(ss_vals)})=all 0x{uniq[0]:x} [{tag}]"
            else:
                vec_part = f"FMA/MLA(x{len(ss_vals)})=mixed ({','.join(f'0x{x:x}' for x in uniq)})"
        else:
            vec_part = ""
        ss_join = ", ".join(p for p in (dp_part, vec_part) if p)
        parts.append(f"SM_SPEED_SELECT({ss_join})")
    for rname in ("FEAT_READOUT_0", "FEAT_READOUT_1", "FBPA_ECC_CTRL"):
        if rname in readings:
            v = readings[rname]
            parts.append(f"{rname}={hex32(v)} ({decode_experimental_register(_EXP_BY_NAME[rname], v)})")
    if not parts:
        return []
    lines = ["  Experimental : (read-only, NOT in FF burn plans — use 'experimental' / '--experimental' for full table)"]
    for p in parts:
        lines.append(f"    - {p}")
    return lines


def format_status_report(plan: Plan, reader, *, allow_vout_override: bool = False) -> list[str]:
    """Build a human-readable diagnostic report without aborting on bad health."""
    lines: list[str] = []
    pci = reader.pci_id()
    host = reader.host()
    health = reader.health()
    host_issues = evaluate_host(host, require_hold=True)
    health_issues = evaluate_health(health, plan, allow_vout_override=allow_vout_override)

    lines.append("=== CARD & HOST STATUS ===")
    lines.append(
        f"  PCI ID       : {pci} "
        f"({'OK' if pci.lower() == plan.pci_id else f'MISMATCH expected {plan.pci_id}'})"
    )
    lines.append(
        f"  Host         : boot={host.boot_id} | driver={host.bound_driver or 'none'} | "
        f"nvidia={host.nvidia_loaded} nouveau={host.nouveau_loaded} | "
        f"lockdown={host.lockdown} | vrm_tainted={host.vrm_tainted} | hold={host.hold_active}"
    )
    lines.append(
        f"  FUSECTRL     : {hex32(health.fusectrl)}  "
        f"DSOV={hex32(health.dsov)}  CRC={hex32(health.crc)}  "
        f"EN_SW={hex32(health.en_sw)}  DEBUGCTRL={hex32(health.debugctrl)}"
    )
    plm_f4_str = f"  PLM_f4={hex32(health.plm_f4)}" if health.plm_f4 is not None else ""
    lines.append(
        f"  PLM / PLM_f8 : PLM={hex32(health.plm)}  PLM_f8={hex32(health.plm_f8)} "
        f"(bit0={'OPEN' if health.plm_f8 & 1 else 'CLOSED!'}){plm_f4_str}"
    )
    if health.efp is not None or health.sbfe is not None or health.wdata_wr_secure is not None:
        arm_parts = []
        if health.efp is not None:
            arm_parts.append(f"EFP={health.efp}")
        if health.dfp is not None:
            arm_parts.append(f"DFP={health.dfp}")
        if health.sbfe is not None:
            arm_parts.append(f"SBFE={health.sbfe}")
        if health.wdata_wr_secure is not None:
            arm_parts.append(f"WSEC={health.wdata_wr_secure}")
        if health.resense_disable is not None:
            arm_parts.append(f"RESENSE_DIS={health.resense_disable}")
        lines.append("  Arming State : " + "  ".join(arm_parts))
    assumed_rail = (not getattr(reader, "simulation", False)) and (
        not getattr(reader, "rail_telemetry_live", True)
    )
    rail_note = "  [ASSUMED / UNMEASURED — no live PMBus]" if assumed_rail else ""
    lines.append(
        f"  VRM / Rail   : STATUS_WORD={format_pmbus_status(health.status_word)}  "
        f"READ_VOUT={health.vout_mv:.1f} mV "
        f"(plan {plan.stock_mv} +/- {plan.vout_tolerance_mv} mV){rail_note}"
    )

    die_bad: list[str] = []
    for idx, expected in sorted(plan.die_rows.items()):
        try:
            actual = reader.die_row(idx)
            if is_poison(actual):
                die_bad.append(f"row {idx}=POISON({hex32(actual)})")
            elif actual != expected:
                die_bad.append(f"row {idx}={hex32(actual)}!=exp({hex32(expected)})")
        except Exception as exc:
            die_bad.append(f"row {idx}=ERR({exc})")
    lines.append(
        "  Die ID (8-15): MATCH" if not die_bad else "  Die ID (8-15): MISMATCH — " + ", ".join(die_bad)
    )

    mask_parts: list[str] = []
    for field, expected in sorted(plan.expected_masks.items()):
        try:
            actual = reader.field(field)
            tag = "POISON!" if is_poison(actual) else ("OK" if actual == expected else f"exp {hex32(expected)}")
            mask_parts.append(f"{field}={hex32(actual)} ({tag})")
        except Exception as exc:
            mask_parts.append(f"{field}=ERR({exc})")
    if mask_parts:
        lines.append("  Masks        : " + "; ".join(mask_parts))

    exp_readings, _ = read_experimental_registers(reader, fusectrl=health.fusectrl)
    if exp_readings:
        lines.extend(_format_experimental_compact_lines(exp_readings))

    all_issues = host_issues + health_issues + die_bad
    if not all_issues:
        lines.append("  GATE SUMMARY : PASS (passive goldilocks & identity clean)")
    else:
        lines.append("  GATE SUMMARY : NOT WRITE-READY:")
        for issue in all_issues:
            lines.append(f"    - {issue}")
    return lines


def _bit_indices(mask: int) -> list[int]:
    return [i for i in range(32) if (mask >> i) & 1]


def _format_unit_list(prefix: str, bits: list[int]) -> str:
    if not bits:
        return "none"
    parts: list[str] = []
    start = prev = bits[0]
    for b in bits[1:]:
        if b == prev + 1:
            prev = b
        else:
            parts.append(f"{prefix}{start}" if start == prev else f"{prefix}{start}..{prefix}{prev}")
            start = prev = b
    parts.append(f"{prefix}{start}" if start == prev else f"{prefix}{start}..{prefix}{prev}")
    return ", ".join(parts)


def _unit_prefix_for_mask(field_name: str) -> str:
    up = field_name.upper()
    if up.startswith("TPC_GPC") and up.endswith("_DISABLE"):
        return "TPC"
    if up.endswith("_DISABLE"):
        return up[: -len("_DISABLE")]
    return "bit"


def _safe_replay_rows(rows: dict[int, int]) -> tuple[dict[int, dict[int, int]], int]:
    """Replay valid GA100 FF rows into {chain: {block: data16}}, counting skipped non-FF rows."""
    blocks: dict[int, dict[int, int]] = {}
    skipped = 0
    for r in sorted(rows):
        if not (FF_ROW_MIN <= r <= FF_ROW_MAX):
            continue
        w = rows[r]
        if w == 0:
            continue
        try:
            rec = decode_ff(w)
        except BurnError:
            skipped += 1
            continue
        blocks.setdefault(rec.chain, {})[rec.block] = rec.data
    return blocks, skipped


def format_plan_review(
    plan: Plan,
    *,
    reader=None,
    override_slot: int | None = None,
) -> list[str]:
    """Build a complete pre-burn plan review: FF rows taken/remaining, enabled/disabled outcome,
    and an audit of unknown, unmapped, or poorly understood fuse changes."""
    final_masks = check_plan(plan)
    lines: list[str] = []

    # Optional read-only scan of current FF rows if a live or fake reader is attached.
    current_rows: dict[int, int] | None = None
    next_free_row: int | None = None
    slot_error: str | None = None
    if reader is not None:
        check_host_readonly_safe(reader.host())
        try:
            rows: dict[int, int] = {}
            for row_idx in range(plan.first_row, plan.last_row + 1):
                val = reader.fuse_row(row_idx)
                if is_poison(val):
                    raise BurnError(f"row {row_idx} reads poisoned ({hex32(val)})")
                rows[row_idx] = val
            current_rows = rows
            try:
                next_free_row, _ = scan_slots(plan, reader, override_slot=override_slot)
            except BurnError as exc:
                slot_error = str(exc)
        except BurnError as exc:
            slot_error = str(exc)

    total_slots = plan.last_row - plan.first_row + 1
    n_total = len(plan.records)
    n_test = sum(1 for r in plan.records if r.kind == "test")
    n_real = n_total - n_test

    # ------------------------------------------------------------------ 1. Header
    mode_tag = "WITH TARGET FF SCAN (read-only)" if current_rows is not None else "OFFLINE ONLY"
    lines.append(f"=== PLAN REVIEW & EXPECTED OUTCOME ({mode_tag}) ===")
    lines.append(f"  Plan SHA256  : {plan.content_hash}")
    lines.append(f"  Target SKU   : {plan.pci_id} (CMP 170HX 8 GB) | Rail: {plan.stock_mv} +/- {plan.vout_tolerance_mv} mV")
    lines.append(
        f"  SM Ceilings  : base={plan.base_sm_ceiling} | override={plan.override_sm_ceiling} | "
        f"reconfig_verified={plan.reconfig_semantics_verified}"
    )
    lines.append(
        "  Die ID (8-15): "
        + " ".join(f"{k}:{hex32(v)}" for k, v in sorted(plan.die_rows.items()))
    )
    lines.append(
        "  Defect Masks : "
        + (", ".join(f"{k}={hex32(v)}" for k, v in sorted(plan.defect_masks.items())) or "none")
    )

    # ------------------------------------------------------------------ 2. FF Capacity
    lines.append("")
    lines.append("=== FF REGION CAPACITY (ROWS TAKEN & REMAINING) ===")
    lines.append(
        f"  FF Window       : rows {plan.first_row}..{plan.last_row} ({total_slots} total slots, empty={hex32(plan.empty_word)})"
    )
    if current_rows is not None:
        used_rows = [r for r, v in sorted(current_rows.items()) if v != plan.empty_word]
        used_count = len(used_rows)
        used_span = f"rows {used_rows[0]}..{used_rows[-1]}" if used_rows else "none (all empty)"
        lines.append(f"  Currently Used  : {used_count} / {total_slots} rows ({used_span})")
        if next_free_row is not None:
            last_take = next_free_row + n_total - 1
            remaining_after = plan.last_row - last_take
            row_span = f"row {next_free_row}" if n_total == 1 else f"rows {next_free_row}..{last_take}"
            lines.append(f"  Rows To Take    : {n_total} ({n_test} test + {n_real} real) -> {row_span}")
            if remaining_after < 0:
                lines.append(
                    f"  Remaining After : WARNING — EXCEEDS FF WINDOW by {-remaining_after} row(s)! CANNOT BURN."
                )
            elif remaining_after < 16:
                lines.append(
                    f"  Remaining After : {remaining_after} / {total_slots} free rows (CAUTION: low spare capacity)"
                )
            else:
                lines.append(f"  Remaining After : {remaining_after} / {total_slots} free rows remaining")
        else:
            lines.append(f"  Rows To Take    : {n_total} ({n_test} test + {n_real} real)")
            lines.append(f"  Remaining After : WARNING — free-slot scan issue: {slot_error}")
    else:
        max_rem = total_slots - n_total
        lines.append(f"  Rows To Take    : {n_total} consecutive empty rows ({n_test} test + {n_real} real)")
        if slot_error:
            lines.append(f"  Remaining After : <= {max_rem} / {total_slots} (target scan unavailable: {slot_error})")
        else:
            lines.append(
                f"  Remaining After : <= {max_rem} / {total_slots} "
                "(select a target or fake snapshot to prove exact starting row)"
            )

    # ------------------------------------------------------------------ 3. Simulate FF Replay & Block Audit
    sim_blocks: dict[int, dict[int, int]] | None = None
    before_ff_fields: dict[str, int] | None = None
    after_ff_fields: dict[str, int] | None = None
    if current_rows is not None:
        sim_blocks, _ = _safe_replay_rows(current_rows)
        before_ff_fields = decode_all_fields(sim_blocks)
    elif plan.baseline_ff_rows is not None:
        sim_blocks, _ = _safe_replay_rows(plan.baseline_ff_rows)
        before_ff_fields = decode_all_fields(sim_blocks)
        if next_free_row is None and plan.expected_next_row is not None:
            next_free_row = plan.expected_next_row

    record_analyses: list[tuple[int, Record, dict | None, str | None]] = []
    audit_warnings: list[str] = []

    for idx, rec in enumerate(plan.records):
        row_label = f"row {next_free_row + idx}" if next_free_row is not None else f"slot +{idx}"
        try:
            ff = decode_ff(rec.word)
        except BurnError as exc:
            record_analyses.append((idx, rec, None, str(exc)))
            audit_warnings.append(
                f"[{row_label} / {rec.name}] INVALID FF WORD {hex32(rec.word)}: {exc}"
            )
            continue

        old_blk = sim_blocks.get(ff.chain, {}).get(ff.block, 0) if sim_blocks is not None else None
        analysis = analyze_ff_block(ff.chain, ff.block, ff.data, old_blk)
        if sim_blocks is not None:
            sim_blocks.setdefault(ff.chain, {})[ff.block] = ff.data
        record_analyses.append((idx, rec, analysis, None))

        for w in analysis["warnings"]:
            audit_warnings.append(f"[{row_label} / {rec.name}] {w}")
        if rec.kind == "reconfig" and not plan.reconfig_semantics_verified:
            audit_warnings.append(
                f"[{row_label} / {rec.name}] RECONFIG record semantics are NOT marked verified in plan"
            )

    if sim_blocks is not None:
        after_ff_fields = decode_all_fields(sim_blocks)

    # ------------------------------------------------------------------ 4. Expected Outcome (Enabled / Disabled)
    lines.append("")
    lines.append("=== EXPECTED OUTCOME (WHAT WILL BE ENABLED / DISABLED) ===")
    outcome_lines: list[str] = []

    # 4a. SM Count estimate (from full masks, FF replay, or declared resulting_sm_count)
    sm_before: int | None = None
    sm_after: int | None = None
    tpc_keys = [f"TPC_GPC{i}_DISABLE" for i in range(TOTAL_GPC)]
    if "GPC_DISABLE" in plan.expected_masks and all(k in plan.expected_masks for k in tpc_keys):
        sm_before = compute_sm_count(
            plan.expected_masks["GPC_DISABLE"], [plan.expected_masks[k] for k in tpc_keys]
        )
        sm_after = compute_sm_count(
            final_masks["GPC_DISABLE"], [final_masks[k] for k in tpc_keys]
        )
    elif before_ff_fields is not None and after_ff_fields is not None:
        ff_tpc_keys = [f"tpc_gpc{i}_disable" for i in range(TOTAL_GPC)]
        b_sm = compute_sm_count(before_ff_fields["gpc_disable"], [before_ff_fields[k] for k in ff_tpc_keys])
        a_sm = compute_sm_count(after_ff_fields["gpc_disable"], [after_ff_fields[k] for k in ff_tpc_keys])
        if b_sm != a_sm:
            sm_before, sm_after = b_sm, a_sm
    if sm_before is not None and sm_after is not None:
        delta = sm_after - sm_before
        sign = "+" if delta >= 0 else ""
        outcome_lines.append(
            f"  [+] Compute SMs      : {sm_before} -> {sm_after} SMs ({sign}{delta} SMs; "
            f"base ceiling={plan.base_sm_ceiling})"
        )
    else:
        declared_sms = [r.resulting_sm_count for r in plan.records if r.resulting_sm_count is not None]
        if declared_sms:
            outcome_lines.append(
                f"  [+] Compute SMs      : target {declared_sms[-1]} SMs (declared in plan; "
                f"base ceiling={plan.base_sm_ceiling})"
            )

    # 4b. Mask deltas from expected_masks -> final_masks
    reported_fields_upper: set[str] = set()
    for field in sorted(plan.expected_masks):
        before_v = plan.expected_masks[field]
        after_v = final_masks[field]
        if before_v == after_v:
            continue
        reported_fields_upper.add(field.upper())
        prefix = _unit_prefix_for_mask(field)
        en_bits = _bit_indices(before_v & ~after_v)
        dis_bits = _bit_indices(~before_v & after_v)
        parts: list[str] = []
        if en_bits:
            parts.append(f"ENABLED {_format_unit_list(prefix, en_bits)}")
        if dis_bits:
            parts.append(f"DISABLED {_format_unit_list(prefix, dis_bits)}")
        defect = plan.defect_masks.get(field, 0)
        def_note = f" (defects {hex32(defect)} preserved)" if defect else ""
        tag = "[+]" if en_bits and not dis_bits else ("[-]" if dis_bits and not en_bits else "[~]")
        outcome_lines.append(
            f"  {tag} {field:16} : {hex32(before_v)} -> {hex32(after_v)} | {'; '.join(parts)}{def_note}"
        )

    # 4c. FF-replay non-floorsweep or extra field deltas (PCIe, NVLink, MIG/SysPipe, etc.)
    if before_ff_fields is not None and after_ff_fields is not None:
        pcie_keys = (
            "disable_gen3_speed_bit1",
            "disable_gen3_speed_bit9",
            "pcie_magic_gen4_dis",
            "pcie_boot_gen23_dis",
            "pcie_boot_gen3_dis",
        )
        pcie_en = [k for k in pcie_keys if before_ff_fields[k] == 1 and after_ff_fields[k] == 0]
        pcie_dis = [k for k in pcie_keys if before_ff_fields[k] == 0 and after_ff_fields[k] == 1]
        if pcie_en:
            outcome_lines.append(f"  [+] PCIe Speed Flags : ENABLED (cleared {', '.join(pcie_en)})")
        if pcie_dis:
            outcome_lines.append(f"  [-] PCIe Speed Flags : RESTRICTED (set {', '.join(pcie_dis)})")

        pcie_set = set(pcie_keys)
        for key in sorted(before_ff_fields):
            if key in pcie_set:
                continue
            bv, av = before_ff_fields[key], after_ff_fields[key]
            if bv == av or key.upper() in reported_fields_upper:
                continue
            if key == "magic_d":
                # Only report if bits outside bit 25 (Gen4 flag) changed
                if ((bv ^ av) & ~(1 << 25)) == 0:
                    continue
                outcome_lines.append(
                    f"  [!] PCIE_MAGIC_BITS_D: {hex32(bv)} -> {hex32(av)} (EXPERIMENTAL bits changed!)"
                )
            else:
                prefix = _unit_prefix_for_mask(key)
                en_bits = _bit_indices(bv & ~av)
                dis_bits = _bit_indices(~bv & av)
                parts = []
                if en_bits:
                    parts.append(f"ENABLED {_format_unit_list(prefix, en_bits)}")
                if dis_bits:
                    parts.append(f"DISABLED {_format_unit_list(prefix, dis_bits)}")
                caution = ""
                tag = "[+]" if en_bits and not dis_bits else ("[-]" if dis_bits and not en_bits else "[~]")
                if key == "sys_pipe_disable":
                    caution = " (EXPERIMENTAL / UNTESTED)"
                    tag = "[!]"
                elif key == "nvlink_disable":
                    caution = " (needs U11 bridge mod)"
                    tag = "[!]"
                outcome_lines.append(
                    f"  {tag} {key.upper():16} : {hex32(bv)} -> {hex32(av)} | {'; '.join(parts)}{caution}"
                )

    # 4d. Declared PCIe LnkCap2 / Reconfig / No-op Test summaries
    for idx, rec, analysis, _ in record_analyses:
        if rec.kind == "pcie" and rec.link_cap2_before is not None and rec.link_cap2_after is not None:
            outcome_lines.append(
                f"  [+] PCIe Link Cap2   : {hex32(rec.link_cap2_before)} -> {hex32(rec.link_cap2_after)} ({rec.decode})"
            )
        elif rec.kind == "test":
            if analysis is not None and analysis["is_noop"] is True:
                outcome_lines.append(
                    f"  [=] Test Row ({rec.name}): NO-OP (0 effective fuse bits changed; verifies burn/readback/CRC)"
                )
            else:
                outcome_lines.append(f"  [=] Test Row ({rec.name}): {rec.decode}")
        elif rec.kind == "reconfig":
            outcome_lines.append(f"  [!] Reconfig ({rec.name}): {rec.decode}")

    if outcome_lines:
        lines.extend(outcome_lines)
    else:
        lines.append("  (No mask or FF-field state changes detected.)")

    # ------------------------------------------------------------------ 5. Unknown / Poorly Understood Fuse Audit
    lines.append("")
    lines.append("=== UNKNOWN / UNMAPPED / POORLY UNDERSTOOD FUSE AUDIT ===")
    if audit_warnings:
        for w in audit_warnings:
            lines.append(f"  WARNING: {w}")
    else:
        lines.append(
            "  PASS: All planned changes map to well-understood GA100 fields; "
            "no unmapped, unknown, or experimental bits change."
        )

    # ------------------------------------------------------------------ 6. Per-Record Write Plan
    lines.append("")
    lines.append(f"=== PER-RECORD WRITE PLAN ({n_total} records) ===")
    for idx, rec, analysis, decode_err in record_analyses:
        slot_str = f"row {next_free_row + idx:3d}" if next_free_row is not None else f"slot +{idx:<2d}"
        if decode_err is not None:
            ff_tag = "INVALID/IFF!"
            grp_tag = "unknown"
        else:
            c, b, nd, od = analysis["chain"], analysis["block"], analysis["new_data"], analysis["old_data"]
            if od is not None:
                delta_s = "NO-OP" if od == nd else f"0x{od:04x}->0x{nd:04x}"
            else:
                delta_s = f"0x{nd:04x}"
            ff_tag = f"c{c}/b{b:<3d} {delta_s}"
            grp_tag = "+".join(sorted(classify_ff_functional_groups(rec.word, od)))
        lines.append(
            f"  {slot_str} | {rec.kind:8} | {grp_tag:10} | {rec.name:18} | {hex32(rec.word)} ({ff_tag}) | {rec.decode}"
        )

    return lines


def format_fuse_controller_state(plan: Plan, reader, *, allow_vout_override: bool = False) -> list[str]:
    """Decode and format the live/fake GA100 fuse controller registers, PLM, VRM, and armed state."""
    health = reader.health()
    host = reader.host()
    ctrl = health.fusectrl
    state_val = (ctrl >> 16) & 0x1F
    cmd_val = ctrl & 0x1F
    idle_str = "IDLE (state=4)" if (not is_poison(ctrl) and state_val == 4 and cmd_val == 0) else (
        "POISON!" if is_poison(ctrl) else f"NON-IDLE (state={state_val}, cmd={cmd_val})"
    )
    rwl = (health.debugctrl >> 1) & 1
    efp_str = str(health.efp) if health.efp is not None else "n/a"
    dfp_str = str(health.dfp) if health.dfp is not None else "n/a"
    sbfe_str = str(health.sbfe) if health.sbfe is not None else "n/a"
    wsec_str = str(health.wdata_wr_secure) if health.wdata_wr_secure is not None else "n/a"
    resense_str = str(health.resense_disable) if health.resense_disable is not None else "n/a"
    plm_f4_str = hex32(health.plm_f4) if health.plm_f4 is not None else "n/a"

    health_issues = evaluate_health(health, plan, allow_vout_override=allow_vout_override)
    host_issues = evaluate_host(host, require_hold=True)
    if health_issues or host_issues:
        verdict = "NOT READY (" + "; ".join(health_issues + host_issues) + ")"
    elif health.efp == 1 and health.sbfe == 0 and (health.en_sw & 1) == 1:
        verdict = "ARMED & WRITE-READY (EFP=1, DSOV=0, CRC=0x55, SBFE=0, PLMs OPEN)"
    else:
        verdict = "PASSIVE GOLDILOCKS OK (DSOV=0, CRC=0x55, PLM_f8 OPEN; needs SENSE_CTRL + arm before CMD_PROGRAM)"

    assumed_rail = (not getattr(reader, "simulation", False)) and (
        not getattr(reader, "rail_telemetry_live", True)
    )
    rail_tag = " [ASSUMED/UNMEASURED]" if assumed_rail else ""
    lines = [
        "=== FUSE CONTROLLER STATE (BAR0 0x0082xxxx) ===",
        f"  FUSECTRL (0x820000) : {hex32(ctrl)}  [{idle_str}]",
        f"  DEBUGCTRL(0x820060) : {hex32(health.debugctrl)}  [RWL={rwl} — NEVER WRITTEN]",
        f"  CRC / DSOV          : CRC(0x820070)={hex32(health.crc)} ({'OK' if health.crc == 0x55 else 'BAD!'}) | DSOV(0x820084)={hex32(health.dsov)} ({'GOLDILOCKS' if health.dsov == 0 else 'LOCKED'})",
        f"  Sense / Arm Flags   : EFP(0x820078)={efp_str}  DFP(0x82007C)={dfp_str}  RESENSE_DIS(0x821058)={resense_str}",
        f"  Override / Shadow   : EN_SW(0x820040)={hex32(health.en_sw)}  SBFE(0x821108)={sbfe_str}  WDATA_WR_SEC(0x82048C)={wsec_str}",
        f"  PLM Gates           : PLM_f4(0x8200F4)={plm_f4_str}  PLM_f8(0x8200F8)={hex32(health.plm_f8)} ({'OPEN' if health.plm_f8 & 1 else 'CLOSED!'})  PLM_fc(0x8200FC)={hex32(health.plm)}",
        f"  VRM / Host          : READ_VOUT={health.vout_mv:.1f} mV{rail_tag}  STATUS_WORD={format_pmbus_status(health.status_word)}  driver={host.bound_driver or 'none'}  hold={host.hold_active}",
        f"  Controller Verdict  : {verdict}",
    ]
    return lines


def format_pcie_write_plan(
    row: int,
    word: int,
    health,
    *,
    skip_sense: bool = False,
    old_block: int | None = None,
) -> list[str]:
    """Format the exact ordered 32-bit aligned PCIe BAR0 MMIO writes for programming one FF row."""
    lines = [
        f"=== EXACT PCIE BAR0 MMIO WRITES TO BE ISSUED (row {row}, word {hex32(word)}) ===",
    ]
    try:
        ff = decode_ff(word)
        analysis = analyze_ff_block(ff.chain, ff.block, ff.data, old_block)
        fields_str = ", ".join(f.name for f in analysis["fields"]) or "NONE (UNKNOWN BLOCK!)"
        delta_str = (
            f" | block 0x{old_block:04x}->0x{ff.data:04x}"
            if old_block is not None
            else f" | block data=0x{ff.data:04x}"
        )
        lines.append(
            f"  FF Decode : chain={ff.chain} block={ff.block} data=0x{ff.data:04x} (IFF bit5=0){delta_str}"
        )
        lines.append(f"  Fields    : {fields_str}")
        for w in analysis["warnings"]:
            lines.append(f"  WARNING   : {w}")
    except BurnError as exc:
        lines.append(f"  WARNING   : INVALID FF WORD — {exc}")

    already_sensed = health.efp == 1
    sense_note = (
        "SKIPPED (EFP=1 already; only one SENSE_CTRL per glitch window)"
        if already_sensed
        else ("SKIPPED (--override-skip-sense)" if skip_sense else "RUNS ONCE (restores EFP 0->1, keeps DSOV=0)")
    )

    lines.append("  --- Phase 1: Arm Fuse Controller (aligned 32-bit BAR0 stores) ---")
    if already_sensed or skip_sense:
        lines.append(f"   [1] SENSE_CTRL (0x00820000 <- 0x00000003) : {sense_note}")
    else:
        lines.append("   [1] WR32 BAR0+0x00820040  EN_SW_OVERRIDE   <- 0x00000001  (verify bit0=1)")
        lines.append("   [2] WR32 BAR0+0x0082048C  WDATA_WR_SECURE  <- 0x00000000  (open PLM_f4 -> 0xffffffff)")
        lines.append("   [3] WR32 BAR0+0x0082000C  FUSEWDATA        <- 0x00000000  (no-burn safety latch; verify rd==0)")
        lines.append("   [4] WR32 BAR0+0x00820000  FUSECTRL         <- 0x00000000  (CMD_IDLE)")
        lines.append("   [5] WR32 BAR0+0x00820000  FUSECTRL         <- 0x00000003  (CMD_SENSE_CTRL; poll IDLE, verify EFP=1/DSOV=0/CRC=0x55)")
    lines.append("   [6] WR32 BAR0+0x00820040  EN_SW_OVERRIDE   <- 0x00000001  (ensure volatile SW override=1)")
    lines.append("   [7] WR32 BAR0+0x00821108  SBFE_SHADOW      <- 0x00000000  (clear SBFE shadow)")
    lines.append("   [8] WR32 BAR0+0x0082048C  WDATA_WR_SECURE  <- 0x00000000  (verify all 12 armed invariants)")
    lines.append("  --- Phase 2: Single-Record Program (CMD_PROGRAM = 2) ---")
    lines.append("   [9] RD32 BAR0+0x00820004  FUSEADDR         -> save orig_addr")
    lines.append("  [10] WR32 BAR0+0x00820000  FUSECTRL         <- 0x00000000  (CMD_IDLE)")
    lines.append(f"  [11] WR32 BAR0+0x00820004  FUSEADDR         <- {hex32(row)}  (select EMPTY FF row {row}; verify latch)")
    lines.append(f"  [12] WR32 BAR0+0x0082000C  FUSEWDATA        <- {hex32(word)}  (stage FF word; verify readback)")
    lines.append("  [13] WR32 BAR0+0x00820000  FUSECTRL         <- 0x00000002  (CMD_PROGRAM; poll IDLE, halt on 0xBADF/0xBAD0)")
    lines.append("  [14] WR32 BAR0+0x00820004  FUSEADDR         <- orig_addr   (restore FUSEADDR)")
    lines.append("  [15] WR32 BAR0+0x0082000C  FUSEWDATA        <- 0x00000000  (restore no-burn safety latch)")
    lines.append("  --- Phase 3: Mandatory Read-Back (CMD_READ = 1) & Post-Write Gate ---")
    lines.append("  [16] WR32 BAR0+0x00820000  FUSECTRL         <- 0x00000000  (CMD_IDLE)")
    lines.append(f"  [17] WR32 BAR0+0x00820004  FUSEADDR         <- {hex32(row)}  (select row {row})")
    lines.append(f"  [18] WR32 BAR0+0x00820000  FUSECTRL         <- 0x00000001  (CMD_READ; verify FUSEDATA(0x820008) == {hex32(word)})")
    lines.append("  [19] POST-WRITE GATE       full health/PMBus/host check + second CMD_READ verify")
    lines.append("  GUARANTEES: single aligned 32-bit stores only | DEBUGCTRL (0x820060) NEVER written | FUSEDATA (0x820008) read-only")
    return lines


class ManualConsole:
    """Simple, clear, journaled manual fuse-controller console."""

    def __init__(
        self,
        plan: Plan,
        reader,
        writer,
        journal: Journal,
        *,
        execute: bool = False,
        test_receipt=None,
        overrides: Overrides | None = None,
        strict_confirm: bool = False,
        input_fn=input,
        show=print,
    ) -> None:
        self.plan = plan
        self.reader = reader
        self.writer = writer if writer is not None else MissingWriter()
        self.journal = journal
        self.execute = execute
        self.test_receipt = test_receipt
        self.overrides = overrides or Overrides()
        self.strict_confirm = strict_confirm
        self.input_fn = input_fn
        self.show = show
        self.baseline_rows: dict[int, int] = {}
        self.baseline_fields: dict[str, int] = {}
        self._staged: dict | None = None

    def _confirm(self, prompt: str, token: str) -> bool:
        try:
            return self.input_fn(prompt).strip() == token
        except (EOFError, KeyboardInterrupt):
            return False

    def _capture_baseline(self) -> None:
        for row in range(self.plan.first_row, self.plan.last_row + 1):
            try:
                self.baseline_rows[row] = self.reader.fuse_row(row)
            except Exception:
                pass
        for field in self.plan.expected_masks:
            try:
                self.baseline_fields[field] = self.reader.field(field)
            except Exception:
                pass

    def _mode_label(self) -> str:
        if getattr(self.reader, "simulation", False):
            return "SIMULATION (fake backend)"
        if self.execute:
            return "LIVE EXECUTE (verified writer enabled)"
        return "LIVE READ-ONLY (writes disabled; 'apply'/'write' runs dry-run only)"

    def run(self) -> int:
        check_host_readonly_safe(self.reader.host())
        self._capture_baseline()
        self.journal.append(
            event="manual_console_start",
            plan_sha256=self.plan.content_hash,
            mode=self._mode_label(),
            overrides=self.overrides.active_names(),
        )
        self.show(f"vfi_burn manual console | {self._mode_label()}")
        if self.overrides.active_names():
            self.show("Active overrides: " + ", ".join(self.overrides.active_names()))
        self.show("Type 'help' for commands, 'preview <word>' then 'apply' to burn without re-typing, 'quit' to exit.")
        for line in format_status_report(
            self.plan, self.reader, allow_vout_override=self.overrides.vout_range
        ):
            self.show(line)

        while True:
            prompt = (
                f"vfi-manual [STAGED row {self._staged['row']} {hex32(self._staged['word'])}]> "
                if self._staged is not None
                else "vfi-manual> "
            )
            try:
                raw = self.input_fn(prompt)
            except EOFError:
                self.show("")
                break
            except KeyboardInterrupt:
                self.show("\n(Use 'quit' to exit cleanly.)")
                continue
            raw = raw.strip()
            if not raw:
                continue
            try:
                should_continue = self.dispatch(raw)
            except BurnError as exc:
                self.show(f"REFUSED: {exc}")
                should_continue = True
            except Exception as exc:
                self.show(f"ERROR: {exc}")
                should_continue = True
            if not should_continue:
                break

        self.journal.append(event="manual_console_end", plan_sha256=self.plan.content_hash)
        return 0

    def dispatch(self, line: str) -> bool:
        parts = shlex.split(line)
        if not parts:
            return True
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("quit", "exit", "q"):
            return False
        if cmd in ("help", "?"):
            self._cmd_help()
            return True
        if cmd == "status":
            for out in format_status_report(
                self.plan, self.reader, allow_vout_override=self.overrides.vout_range
            ):
                self.show(out)
            self.journal.append(event="manual_status")
            return True
        if cmd in ("state", "fuse-state"):
            self._cmd_state()
            return True
        if cmd in ("experimental", "exp"):
            self._cmd_experimental()
            return True
        if cmd == "preview":
            self._cmd_preview(args)
            return True
        if cmd == "apply":
            self._cmd_apply(args)
            return True
        if cmd in ("clear", "unstage"):
            self._cmd_clear()
            return True
        if cmd == "staged":
            self._cmd_staged()
            return True
        if cmd == "sense":
            self._cmd_sense()
            return True
        if cmd == "scan":
            self._cmd_scan()
            return True
        if cmd == "row":
            self._cmd_row(args)
            return True
        if cmd == "rows":
            self._cmd_rows(args)
            return True
        if cmd == "reg":
            self._cmd_reg(args)
            return True
        if cmd == "field":
            self._cmd_field(args)
            return True
        if cmd == "diff":
            self._cmd_diff()
            return True
        if cmd == "overrides":
            self._cmd_overrides()
            return True
        if cmd == "write":
            self._cmd_write(args)
            return True

        raise BurnError(f"unknown command {cmd!r}; type 'help' for available commands")

    def _cmd_help(self) -> None:
        self.show(
            "Commands:\n"
            "  status                         Show full health, host, die ID, and mask status\n"
            "  state                          Show detailed GA100 fuse controller registers & armed state\n"
            "  experimental (exp)             Read-only table of experimental GA100 regs (ECC/SS/SKU/readouts)\n"
            "  preview <word> [row]           Inspect fuse state + exact PCIe BAR0 MMIO writes & stage\n"
            "  preview <target> <word> <desc> Inspect exact PCIe BAR0 writes & stage with custom label\n"
            "  apply                          Execute the exact staged preview (no re-typing; gated)\n"
            "  staged / clear                 Show or discard the currently staged write\n"
            "  sense                          Run opt-in active_sense_readonly() and diff health\n"
            "  scan                           Double-scan FPF window and prove next free row\n"
            "  row <index>                    Read one FPF row (with poison & diff check)\n"
            "  rows <start> <end>             Read a range of FPF rows (max 128)\n"
            "  reg <addr|NAME>                Read aligned 32-bit MMIO reg (e.g. FUSECTRL, FUSE_ECC_EN)\n"
            "  field <name>                   Read named decoded field (e.g. GPC_DISABLE, FUSE_ECC_EN)\n"
            "  diff                           Diff current FPF rows & fields vs console start\n"
            "  overrides                      Show active overrides & unbreakable invariants\n"
            "  write <target> <word> <decode> One-step preview + gated write (prefer preview -> apply)\n"
            "  quit                           Exit manual console"
        )

    def _cmd_state(self) -> None:
        for out in format_fuse_controller_state(
            self.plan, self.reader, allow_vout_override=self.overrides.vout_range
        ):
            self.show(out)
        self.journal.append(event="manual_fuse_state", health=self.reader.health().serializable())

    def _cmd_experimental(self) -> None:
        readings, poisoned = read_experimental_registers(self.reader)
        for out in format_experimental_report(self.reader, readings=readings, poisoned=poisoned):
            self.show(out)
        self.journal.append(
            event="manual_experimental_read",
            skipped_poison=poisoned,
            registers={k: hex32(v) for k, v in readings.items()},
        )

    def _show_write_preview(self, word: int, row: int | None = None) -> int:
        if row is None:
            row, _ = scan_slots(self.plan, self.reader, override_slot=self.overrides.slot)
        old_blk = None
        try:
            ff = decode_ff(word)
            cur_rows = {
                r: self.reader.fuse_row(r)
                for r in range(self.plan.first_row, self.plan.last_row + 1)
            }
            blocks, _ = _safe_replay_rows(cur_rows)
            old_blk = blocks.get(ff.chain, {}).get(ff.block, 0)
        except Exception:
            old_blk = None
        for out in format_fuse_controller_state(
            self.plan, self.reader, allow_vout_override=self.overrides.vout_range
        ):
            self.show(out)
        for out in format_pcie_write_plan(
            row,
            word,
            self.reader.health(),
            skip_sense=self.overrides.skip_active_sense,
            old_block=old_blk,
        ):
            self.show(out)
        return row

    def _auto_label_for_word(self, word: int) -> tuple[str, str]:
        try:
            ff = decode_ff(word)
            analysis = analyze_ff_block(ff.chain, ff.block, ff.data, None)
            fields = analysis["fields"]
            if fields:
                # Prefer a floorsweep field name if any overlap (ensures defect-mask guard triggers).
                primary = next((f for f in fields if f.category == "floorsweep"), fields[0])
                names = ", ".join(f.name for f in fields)
                return primary.key.upper(), f"FF c{ff.chain}/b{ff.block}=0x{ff.data:04x} ({names})"
            return f"FF_C{ff.chain}_B{ff.block}", f"FF c{ff.chain}/b{ff.block}=0x{ff.data:04x} (unknown block)"
        except BurnError as exc:
            return "INVALID_FF_WORD", f"invalid FF word ({exc})"

    def _cmd_preview(self, args: list[str]) -> None:
        if not args:
            raise BurnError("usage: preview <32-bit-hex-word> [row]  OR  preview <target> <word> <decode>")
        explicit_row: int | None = None
        if len(args) >= 3:
            target = args[0]
            word = u32(args[1], "word")
            decode = " ".join(args[2:]).strip()
            if not decode:
                raise BurnError("preview decode description must be non-empty")
        else:
            word = u32(args[0], "word")
            explicit_row = int(args[1], 0) if len(args) == 2 else None
            target, decode = self._auto_label_for_word(word)

        resolved_row = self._show_write_preview(word, row=explicit_row)
        staged_ok = True
        if explicit_row is not None:
            try:
                proved_row, _ = scan_slots(self.plan, self.reader, override_slot=self.overrides.slot)
                if explicit_row != proved_row:
                    staged_ok = False
                    self.show(
                        f"NOTE: previewed explicit row {explicit_row}, but next proved free row is {proved_row} "
                        "(not staged — omit [row] or set --override-slot to stage)."
                    )
            except BurnError:
                staged_ok = False

        if staged_ok:
            self._staged = {
                "row": resolved_row,
                "target": target,
                "word": word,
                "decode": decode,
            }
            self.show(
                f"STAGED: row {resolved_row} | target={target} | word={hex32(word)} | decode={decode!r}"
            )
            self.show(
                "Type 'apply' when happy to execute this exact staged write (no re-typing needed), "
                "or 'clear' to discard."
            )
        self.journal.append(
            event="manual_preview_writes",
            row=resolved_row,
            target=target,
            word=hex32(word),
            decode=decode,
            staged=staged_ok,
        )

    def _cmd_staged(self) -> None:
        if self._staged is None:
            self.show("Nothing staged. Use 'preview <word>' or 'preview <target> <word> <decode>' to stage a write.")
            return
        s = self._staged
        self.show(
            f"STAGED: row {s['row']} | target={s['target']} | word={hex32(s['word'])} | decode={s['decode']!r}"
        )

    def _cmd_clear(self) -> None:
        if self._staged is None:
            self.show("Nothing was staged.")
            return
        old = self._staged
        self._staged = None
        self.journal.append(
            event="manual_staged_cleared",
            row=old["row"],
            target=old["target"],
            word=hex32(old["word"]),
        )
        self.show(f"Cleared staged write (row {old['row']} {hex32(old['word'])}).")

    def _cmd_apply(self, args: list[str]) -> None:
        if args:
            raise BurnError(
                "usage: apply (takes NO arguments — executes the exact staged preview to prevent typos; "
                "use 'preview ...' to change what is staged)"
            )
        if self._staged is None:
            raise BurnError(
                "nothing staged — run 'preview <word>' or 'preview <target> <word> <decode>' first"
            )
        staged = self._staged
        self._staged = None  # consume immediately so a stale staged write can never apply twice

        cur_row, _ = scan_slots(self.plan, self.reader, override_slot=self.overrides.slot)
        if cur_row != staged["row"]:
            raise BurnError(
                f"next proved free row changed since preview (staged row {staged['row']}, now row {cur_row}); "
                "re-run 'preview' before applying"
            )
        self.show(
            f"APPLYING STAGED WRITE: row {staged['row']} | target={staged['target']} | "
            f"word={hex32(staged['word'])} | decode={staged['decode']!r}"
        )
        self._execute_manual_record(
            staged["target"],
            staged["word"],
            staged["decode"],
            show_preview=False,
        )

    def _cmd_sense(self) -> None:
        before = self.reader.health()
        pre_issues = evaluate_health(
            before, self.plan, allow_vout_override=self.overrides.vout_range
        )
        if pre_issues:
            self.journal.append(
                event="manual_active_sense_refused",
                before=before.serializable(),
                issues=pre_issues,
            )
            raise BurnError(
                "refusing active sense when passive health is already degraded: "
                + "; ".join(pre_issues)
            )
        self.show(
            "ACTIVE SENSE: active_sense_readonly() does not burn fuses, but can collapse "
            "a too-deep held goldilocks frame (forcing power cycle + re-hunt)."
        )
        if not self._confirm("Type 'SENSE' to run active_sense_readonly(): ", "SENSE"):
            self.journal.append(event="declined", command="manual_active_sense", reason="SENSE")
            raise BurnError("active sense declined by operator")
        self.reader.active_sense_readonly()
        after = self.reader.health()
        post_issues = evaluate_health(
            after, self.plan, allow_vout_override=self.overrides.vout_range
        )
        self.journal.append(
            event="manual_active_sense",
            before=before.serializable(),
            after=after.serializable(),
            issues=post_issues,
        )
        self.show(
            f"Active sense completed: FUSECTRL {hex32(before.fusectrl)} -> {hex32(after.fusectrl)} | "
            f"PLM_f8 {hex32(before.plm_f8)} -> {hex32(after.plm_f8)} | "
            f"EN_SW {hex32(before.en_sw)} -> {hex32(after.en_sw)} | "
            f"VOUT {before.vout_mv:.1f} -> {after.vout_mv:.1f} mV"
        )
        if post_issues:
            self.show("WARNING — frame degraded by active sense (too deep; do NOT write):")
            for issue in post_issues:
                self.show(f"  - {issue}")
        else:
            self.show("PASS: active sense clean; frame survived.")

    def _cmd_scan(self) -> None:
        rows = [
            (i, self.reader.fuse_row(i))
            for i in range(self.plan.first_row, self.plan.last_row + 1)
        ]
        for idx, val in rows:
            tag = (
                "POISON!"
                if is_poison(val)
                else ("EMPTY" if val == self.plan.empty_word else "WRITTEN")
            )
            self.show(f"  row {idx:4d}: {hex32(val)}  [{tag}]")
        try:
            pointer, _ = scan_slots(self.plan, self.reader, override_slot=self.overrides.slot)
            self.show(f"Proved next free row: {pointer}")
            self.journal.append(event="manual_scan", proved_pointer=pointer)
        except BurnError as exc:
            self.show(f"Free-slot proof FAILED: {exc}")
            self.journal.append(event="manual_scan", error=str(exc))

    def _cmd_row(self, args: list[str]) -> None:
        if len(args) != 1:
            raise BurnError("usage: row <index>")
        idx = int(args[0], 0)
        val = self.reader.fuse_row(idx)
        warn = "  [POISON!]" if is_poison(val) else ""
        changed = ""
        if idx in self.baseline_rows and self.baseline_rows[idx] != val:
            changed = f"  (was {hex32(self.baseline_rows[idx])})"
        self.show(f"row {idx} = {hex32(val)}{warn}{changed}")
        self.journal.append(event="manual_read_row", row=idx, value=hex32(val))

    def _cmd_rows(self, args: list[str]) -> None:
        if len(args) != 2:
            raise BurnError("usage: rows <start> <end>")
        start, end = int(args[0], 0), int(args[1], 0)
        if end < start or (end - start) >= 128:
            raise BurnError("invalid range (must satisfy start <= end and <= 128 rows)")
        values = {}
        for idx in range(start, end + 1):
            val = self.reader.fuse_row(idx)
            values[str(idx)] = hex32(val)
            warn = "  [POISON!]" if is_poison(val) else ""
            self.show(f"  row {idx:4d}: {hex32(val)}{warn}")
        self.journal.append(event="manual_read_rows", start=start, end=end, values=values)

    def _cmd_reg(self, args: list[str]) -> None:
        if len(args) != 1:
            raise BurnError("usage: reg <addr|FUSECTRL|DEBUGCTRL|PLM_F8|FUSE_ECC_EN|...>")
        addr = resolve_register_address(args[0])
        val = self.reader.register(addr)
        warn = "  [POISON!]" if is_poison(val) else ""
        exp = _EXP_BY_ADDR.get(addr)
        extra = (
            f"  [EXPERIMENTAL: {exp.name} — {decode_experimental_register(exp, val)}; "
            f"170HX={exp.cmp170hx_ref} / A100={exp.a100_ref}]"
            if exp is not None
            else ""
        )
        self.show(f"reg {hex32(addr)} = {hex32(val)}{warn}{extra}")
        self.journal.append(event="manual_read_reg", address=hex32(addr), value=hex32(val))

    def _cmd_field(self, args: list[str]) -> None:
        if len(args) != 1:
            raise BurnError("usage: field <name>")
        name = args[0]
        val = self.reader.field(name)
        warn = "  [POISON!]" if is_poison(val) else ""
        exp = _EXP_BY_NAME.get(name.strip().upper())
        extra = (
            f"  [EXPERIMENTAL: {decode_experimental_register(exp, val)}; "
            f"170HX={exp.cmp170hx_ref} / A100={exp.a100_ref}]"
            if exp is not None
            else ""
        )
        self.show(f"{name} = {hex32(val)}{warn}{extra}")
        self.journal.append(event="manual_read_field", field=name, value=hex32(val))

    def _cmd_diff(self) -> None:
        row_diffs: dict[str, list[str]] = {}
        field_diffs: dict[str, list[str]] = {}
        for row, orig in sorted(self.baseline_rows.items()):
            cur = self.reader.fuse_row(row)
            if cur != orig:
                self.show(f"  row {row}: {hex32(orig)} -> {hex32(cur)}")
                row_diffs[str(row)] = [hex32(orig), hex32(cur)]
        for field, orig in sorted(self.baseline_fields.items()):
            cur = self.reader.field(field)
            if cur != orig:
                self.show(f"  field {field}: {hex32(orig)} -> {hex32(cur)}")
                field_diffs[field] = [hex32(orig), hex32(cur)]
        if not row_diffs and not field_diffs:
            self.show("No changes since console started.")
        self.journal.append(event="manual_diff", row_diffs=row_diffs, field_diffs=field_diffs)

    def _cmd_overrides(self) -> None:
        active = self.overrides.active_names()
        self.show("Active overrides: " + (", ".join(active) if active else "none (strict defaults)"))
        self.show(
            "Never-overrideable invariants: aligned 32-bit stores, no GPU driver, SKU/die ID match, "
            "empty target slot, poison/PLM_f8/latch halt, mandatory read-back, post-write gate, "
            "known genuine defect bits, and read-only PCIe device-ID blocks (c2/b37-38)."
        )

    def _is_floorsweep_target(self, target: str, word: int | None = None) -> bool:
        upper = target.strip().upper()
        known = {k.upper() for k in self.plan.defect_masks} | {
            k.upper() for k in self.plan.expected_masks
        }
        if upper in known:
            return True
        if any(tok in upper for tok in ("GPC", "TPC", "SM_", "FBP", "DISABLE", "RECONFIG")):
            return True
        if word is not None:
            try:
                ff = decode_ff(word)
                if any(f.category == "floorsweep" for f in analyze_ff_block(ff.chain, ff.block, ff.data, None)["fields"]):
                    return True
            except BurnError:
                pass
        return False

    def _cmd_write(self, args: list[str]) -> None:
        if len(args) < 3:
            raise BurnError("usage: write <target> <32-bit-hex-word> <human-readable-decode>")
        target = args[0]
        word = u32(args[1], "word")
        decode = " ".join(args[2:]).strip()
        if not decode:
            raise BurnError("manual write requires a non-empty decode description")
        self._execute_manual_record(target, word, decode, show_preview=True)

    def _execute_manual_record(
        self,
        target: str,
        word: int,
        decode: str,
        *,
        show_preview: bool,
    ) -> None:
        if self._is_floorsweep_target(target, word):
            if not self.overrides.undetermined_ceiling:
                self.journal.append(
                    event="manual_console_write_refused",
                    target=target,
                    word=hex32(word),
                    reason="floorsweep_target_requires_mask_update",
                )
                raise BurnError(
                    f"target {target!r} looks like a floorsweep/mask field, and the console "
                    "one-liner cannot reconcile mask_update against card.defect_masks. "
                    "Use 'vfi_burn manual write-record --record <file.json>' with an explicit "
                    "mask_update, or restart console with --override-undetermined-ceiling."
                )
            token = f"UNRECONCILED TARGET {target.upper()}"
            self.show(
                f"!!! WARNING: console one-liner has no mask_update/SM metadata for {target!r}; "
                "genuine defect bits are still refused from the FF word itself."
            )
            if not self._confirm(f"Type '{token}' to confirm unreconciled manual target: ", token):
                self.journal.append(
                    event="override_declined",
                    override="unreconciled_console_target",
                    target=target,
                )
                raise BurnError("unreconciled floorsweep target declined by operator")
            self.journal.append(
                event="override_confirmed",
                override="unreconciled_console_target",
                target=target,
                token=token,
            )

        record = Record(
            name=f"MANUAL_{target}",
            target=target,
            word=word,
            decode=decode,
            kind="regular",
            defect_sensitive=False,
            mask_field=None,
            from_disable=None,
            to_disable=None,
            resulting_sm_count=None,
            expected_cold={},
            link_cap2_before=None,
            link_cap2_after=None,
        )
        payload_hash = hashlib.sha256(
            f"{record.target}:{hex32(record.word)}:{record.decode}".encode("utf-8")
        ).hexdigest()
        single = replace(self.plan, records=(self.plan.records[0], record))
        dry_run = (not self.execute) and (not getattr(self.reader, "simulation", False))
        if dry_run:
            self.show("NOTE: console is READ-ONLY — running write as DRY-RUN preview only.")

        if show_preview:
            try:
                self._show_write_preview(word)
            except BurnError:
                pass  # let process_records surface the exact preflight/slot-proof error & journal it

        self.journal.append(
            event="manual_console_write_request",
            target=record.target,
            word=hex32(record.word),
            decode=record.decode,
            dry_run=dry_run,
            manual_record_sha256=payload_hash,
        )
        process_records(
            single,
            self.reader,
            self.writer,
            self.journal,
            stage="real",
            confirm=self._confirm,
            test_receipt=self.test_receipt,
            show=self.show,
            overrides=self.overrides,
            dry_run=dry_run,
            strict_confirm=self.strict_confirm,
            confirm_active_sense=self.execute,
            run_active_sense=not dry_run,
            manual_record_sha256=payload_hash,
            is_manual_single=True,
        )
