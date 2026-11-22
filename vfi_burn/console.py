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
    evaluate_health,
    evaluate_host,
    format_pmbus_status,
    hex32,
    process_records,
    scan_slots,
)
from .mmio import resolve_register_address
from .model import BurnError, Plan, Record, is_poison, u32


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
    lines.append(
        f"  VRM / Rail   : STATUS_WORD={format_pmbus_status(health.status_word)}  "
        f"READ_VOUT={health.vout_mv:.1f} mV "
        f"(plan {plan.stock_mv} +/- {plan.vout_tolerance_mv} mV)"
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

    all_issues = host_issues + health_issues + die_bad
    if not all_issues:
        lines.append("  GATE SUMMARY : PASS (passive goldilocks & identity clean)")
    else:
        lines.append("  GATE SUMMARY : NOT WRITE-READY:")
        for issue in all_issues:
            lines.append(f"    - {issue}")
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
        return "LIVE READ-ONLY (writes disabled; 'write' runs dry-run only)"

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
        self.show("Type 'help' for commands, 'quit' to exit.")
        for line in format_status_report(
            self.plan, self.reader, allow_vout_override=self.overrides.vout_range
        ):
            self.show(line)

        while True:
            try:
                raw = self.input_fn("vfi-manual> ")
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
            "  sense                          Run opt-in active_sense_readonly() and diff health\n"
            "  scan                           Double-scan FPF window and prove next free row\n"
            "  row <index>                    Read one FPF row (with poison & diff check)\n"
            "  rows <start> <end>             Read a range of FPF rows (max 128)\n"
            "  reg <addr|FUSECTRL|DEBUGCTRL|PLM_F8>  Read aligned 32-bit MMIO register\n"
            "  field <name>                   Read named decoded field (e.g. GPC_DISABLE, LnkCap2)\n"
            "  diff                           Diff current FPF rows & fields vs console start\n"
            "  overrides                      Show active overrides & unbreakable invariants\n"
            "  write <target> <word> <decode> Write ONE record through full safety gates\n"
            "  quit                           Exit manual console"
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
            raise BurnError("usage: reg <addr|FUSECTRL|DEBUGCTRL|PLM_F8>")
        addr = resolve_register_address(args[0])
        val = self.reader.register(addr)
        warn = "  [POISON!]" if is_poison(val) else ""
        self.show(f"reg {hex32(addr)} = {hex32(val)}{warn}")
        self.journal.append(event="manual_read_reg", address=hex32(addr), value=hex32(val))

    def _cmd_field(self, args: list[str]) -> None:
        if len(args) != 1:
            raise BurnError("usage: field <name>")
        name = args[0]
        val = self.reader.field(name)
        warn = "  [POISON!]" if is_poison(val) else ""
        self.show(f"{name} = {hex32(val)}{warn}")
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
            "genuine defect bits."
        )

    def _is_floorsweep_target(self, target: str) -> bool:
        upper = target.strip().upper()
        known = {k.upper() for k in self.plan.defect_masks} | {
            k.upper() for k in self.plan.expected_masks
        }
        if upper in known:
            return True
        return any(tok in upper for tok in ("GPC", "TPC", "SM_", "FBP", "DISABLE", "RECONFIG"))

    def _cmd_write(self, args: list[str]) -> None:
        if len(args) < 3:
            raise BurnError("usage: write <target> <32-bit-hex-word> <human-readable-decode>")
        target = args[0]
        word = u32(args[1], "word")
        decode = " ".join(args[2:]).strip()
        if not decode:
            raise BurnError("manual write requires a non-empty decode description")

        if self._is_floorsweep_target(target):
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
                f"!!! WARNING: console one-liner skips defect-mask reconciliation for {target!r}!"
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
