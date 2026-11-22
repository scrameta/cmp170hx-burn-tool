"""High-level menu-driven interface for vfi_burn.

Designed for bench use over SSH/tmux: plain numbered menus (no fragile curses),
persistent status header, read-only by default on live hardware, and every safety
gate, typed confirmation, and fsynced journal log preserved identically to the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import json
import sys

from .backend import (
    MissingWriter,
    fake_from_file,
    has_verified_writer,
    load_adapter,
    snapshot_data,
    write_private_json,
)
from .console import (
    ManualConsole,
    format_experimental_report,
    format_plan_review,
    format_status_report,
    read_experimental_registers,
)
from .engine import (
    Journal,
    Overrides,
    check_host_readonly_safe,
    check_plan,
    evaluate_health,
    hex32,
    preflight,
    process_records,
    scan_slots,
    verify_cold_snapshot,
)
from .ffrecord import decode_ff, fields_in_block, import_ffcalc_plan
from .ga100 import GA100LiveAdapter
from .model import BurnError, Plan, require_private_file, u32


@dataclass
class MenuSession:
    plan_path: str | None = None
    bdf: str | None = None
    pico_port: str | None = None
    assume_vout_mv: float | None = None
    adapter_spec: str | None = None
    fake_path: str | None = None
    journal_path: str = "private/session.jsonl"
    receipt_path: str | None = None
    execute_live: bool = False
    overrides: Overrides = field(default_factory=Overrides)


class BurnMenu:
    """Interactive menu-driven controller wrapping all vfi_burn workflows safely."""

    def __init__(
        self,
        session: MenuSession | None = None,
        *,
        input_fn=None,
        show=None,
        check_tty: bool = True,
    ) -> None:
        self.s = session or MenuSession()
        self.input_fn = input_fn if input_fn is not None else (lambda prompt="": input(prompt))
        self.show = show if show is not None else (lambda *args, **kwargs: print(*args, **kwargs))
        self.check_tty = check_tty
        self._cached_fake = None  # keeps fake state across menu actions in simulation mode

    def _ask(self, prompt: str, default: str | None = None) -> str:
        suffix = f" [{default}]" if default else ""
        raw = self.input_fn(f"{prompt}{suffix}: ").strip()
        return raw if raw else (default or "")

    def _ask_clearable(self, prompt: str, current: str | None = None) -> str | None:
        hint = f" [{current}, or '-' to clear]" if current else " [blank for none]"
        raw = self.input_fn(f"{prompt}{hint}: ").strip()
        if raw == "-":
            return None
        if not raw:
            return current
        return raw

    def _confirm(self, prompt: str, token: str) -> bool:
        try:
            return self.input_fn(prompt).strip() == token
        except (EOFError, KeyboardInterrupt):
            return False

    def _load_plan(self) -> Plan:
        if not self.s.plan_path:
            raise BurnError("no plan selected yet — choose [1] Setup to load or import a plan")
        p = Plan.load(self.s.plan_path)
        check_plan(p)
        return p

    def _target_label(self) -> str:
        if self.s.fake_path:
            return f"FAKE ({self.s.fake_path})"
        if self.s.bdf:
            rail = f"port={self.s.pico_port}" if self.s.pico_port else (
                f"assume_vout={self.s.assume_vout_mv}mV" if self.s.assume_vout_mv else "NO RAIL MONITOR"
            )
            return f"LIVE BDF={self.s.bdf} ({rail})"
        if self.s.adapter_spec:
            return f"ADAPTER ({self.s.adapter_spec})"
        return "not selected"

    def _mode_label(self) -> str:
        if self.s.fake_path:
            return "SIMULATION (fake backend — no GPU touched)"
        if self.s.execute_live:
            return "LIVE EXECUTE ENABLED (permanent fuse writes armed behind per-record confirms)"
        return "LIVE READ-ONLY / DRY-RUN (safe default)"

    def _open_backend(self, *, writable: bool = False):
        if self.s.fake_path:
            if self._cached_fake is None:
                self._cached_fake = fake_from_file(self.s.fake_path)
            return self._cached_fake, False  # False = do not close cached fake
        if self.s.bdf:
            return (
                GA100LiveAdapter(
                    self.s.bdf,
                    pico_port=self.s.pico_port,
                    writable=writable,
                    assume_vout_mv=self.s.assume_vout_mv,
                ),
                True,
            )
        if self.s.adapter_spec:
            return load_adapter(self.s.adapter_spec), True
        raise BurnError("no target selected yet — choose [1] Setup to select --bdf, --adapter, or --fake")

    def _close_if_owned(self, backend, owned: bool) -> None:
        if owned and backend is not None and callable(getattr(backend, "close", None)):
            backend.close()

    def _check_execute_policy(self) -> None:
        if self.s.fake_path:
            return
        if not self.s.execute_live:
            raise BurnError(
                "Live Execute mode is OFF (read-only/dry-run protection). "
                "Enable it in [1] Setup -> Toggle Live Execute mode first."
            )
        if self.check_tty and not sys.stdin.isatty():
            raise BurnError("REFUSED live write without an interactive TTY")

    def _require_execute_ready(self, backend) -> None:
        self._check_execute_policy()
        if not self.s.fake_path and not has_verified_writer(backend):
            raise BurnError("supplied adapter does not implement a verified Writer (writer_verified=True)")

    def _render_header(self) -> None:
        plan_info = self.s.plan_path or "not selected"
        if self.s.plan_path:
            try:
                p = Plan.load(self.s.plan_path)
                plan_info = f"{self.s.plan_path} ({len(p.records)} records, sha={p.content_hash[:12]})"
            except Exception as exc:
                plan_info = f"{self.s.plan_path} (INVALID: {exc})"
        ov_list = self.s.overrides.active_names()
        ov_str = ", ".join(ov_list) if ov_list else "none (strict defaults)"
        self.show("\n" + "=" * 72)
        self.show("  vfi_burn — CMP 170HX 8 GB FF Fuse-Burn Safety Controller")
        self.show("  WARNING: UNTESTED ON PHYSICAL HARDWARE — VERIFY EVERYTHING BY HAND")
        self.show("=" * 72)
        self.show(f"  Mode      : {self._mode_label()}")
        self.show(f"  Target    : {self._target_label()}")
        self.show(f"  Plan      : {plan_info}")
        self.show(f"  Journal   : {self.s.journal_path}" + (f" | Receipt: {self.s.receipt_path}" if self.s.receipt_path else ""))
        self.show(f"  Overrides : {ov_str}")
        self.show("-" * 72)
        self.show("   [1] Setup / Load Plan, Import ffcalc, Target BDF/Port & Mode")
        self.show("   [2] Review Plan & Expected Outcome (enabled/disabled, unknown fuses, FF rows)")
        self.show("   [3] Live Status & Passive Inspection (safe on cold/faulted cards)")
        self.show("   [4] Run Active Sense (SENSE_CTRL: EFP 0->1, checks PLM_f8)")
        self.show("   [5] Dry-Run Burn Preview (passive preflight + row plan, NO writes)")
        self.show("   [6] Burn Plan — Single Window (test row -> CONTINUE SAME WINDOW -> real)")
        self.show("   [7] Burn Plan — Split Window (test only, or real with receipt)")
        self.show("   [8] Configure Manual Overrides (granular, audited)")
        self.show("   [9] Manual Low-Level Console (read rows/regs/fields, diff, single write)")
        self.show("  [10] Decode Any 32-Bit FF Word (offline)")
        self.show("  [11] Cold-Boot Snapshot & Verify (snapshot / verify-cold)")
        self.show("  [12] Experimental Registers (read-only: ECC, SM speed select, SKU, readouts)")
        self.show("   [0] Quit")
        self.show("-" * 72)

    def run(self) -> int:
        while True:
            self._render_header()
            try:
                choice = self.input_fn("Select [0-12]: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.show("\nExiting vfi_burn menu.")
                return 0
            if choice in ("0", "q", "quit", "exit"):
                self.show("Exiting vfi_burn menu.")
                return 0
            try:
                self._dispatch(choice)
            except KeyboardInterrupt:
                self.show("\nABORTED action (Ctrl-C). Returning to menu.")
            except (BurnError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
                self.show(f"\nREFUSED: {exc}")

    def _dispatch(self, choice: str) -> None:
        handlers = {
            "1": self._menu_setup,
            "2": self._menu_review_plan,
            "3": self._menu_status_inspect,
            "4": self._menu_active_sense,
            "5": self._menu_dry_run,
            "6": self._menu_burn_single_window,
            "7": self._menu_burn_split,
            "8": self._menu_overrides,
            "9": self._menu_manual_console,
            "10": self._menu_decode_word,
            "11": self._menu_cold_verify,
            "12": self._menu_experimental_registers,
        }
        fn = handlers.get(choice)
        if fn is None:
            self.show(f"Unknown selection {choice!r}; pick 0-12.")
            return
        fn()

    # ------------------------------------------------------------------ [1] Setup
    def _menu_setup(self) -> None:
        while True:
            self.show("\n--- [1] Setup ---")
            self.show(f"  [a] Select plan JSON             (current: {self.s.plan_path or 'none'})")
            self.show("  [b] Import cmp170hx-ffcalc plan + 512-row dump -> new 0600 plan")
            self.show(f"  [c] Select LIVE GPU (--bdf / --port) (current: {self.s.bdf or 'none'})")
            self.show(f"  [d] Select FAKE snapshot (--fake)    (current: {self.s.fake_path or 'none'})")
            self.show(f"  [e] Select journal path          (current: {self.s.journal_path})")
            self.show(f"  [f] Select test receipt path     (current: {self.s.receipt_path or 'none'})")
            self.show(f"  [g] Toggle Live Execute mode     (current: {'ON' if self.s.execute_live else 'OFF (safe)'})")
            self.show("  [0] Back to main menu")
            sub = self.input_fn("Setup [a-g, 0]: ").strip().lower()
            if sub in ("0", "", "b0", "back"):
                return
            if sub == "a":
                path = self._ask("Path to mode-0600 plan JSON", self.s.plan_path)
                if path:
                    p = Plan.load(path)
                    check_plan(p)
                    self.s.plan_path = path
                    self.show(f"Loaded valid plan ({len(p.records)} records, sha={p.content_hash[:12]}).")
            elif sub == "b":
                bp = self._ask("ffcalc burn.json path")
                dp = self._ask("ffcalc 512-row dump.json path")
                out = self._ask("Output mode-0600 plan path", "private/plan.json")
                noop = self._ask("Prepend safe no-op test row? (y/n)", "y").lower().startswith("y")
                plan_dict = import_ffcalc_plan(bp, dp, prepend_noop_test=noop)
                write_private_json(out, plan_dict)
                loaded = Plan.load(out)
                check_plan(loaded)
                self.s.plan_path = out
                self.show(f"Imported & loaded {out} ({len(loaded.records)} records).")
            elif sub == "c":
                bdf = self._ask("CMP 170HX PCI BDF (e.g. 0000:05:00.0)", self.s.bdf)
                port = self._ask_clearable("Pico serial port", self.s.pico_port)
                vout = None
                if not port:
                    default_vout = str(self.s.assume_vout_mv) if self.s.assume_vout_mv is not None else "787.5"
                    vout_s = self._ask("Externally verified NVVDD in mV (e.g. 787.5)", default_vout)
                    vout = float(vout_s) if vout_s else None
                self.s.bdf = bdf or None
                self.s.pico_port = port
                self.s.assume_vout_mv = vout
                self.s.fake_path = None
                self.s.adapter_spec = None
                self._cached_fake = None
                self.s.execute_live = False  # always reset to safe read-only on target change
                self.show("Target set to live hardware (Live Execute reset to OFF).")
            elif sub == "d":
                fp = self._ask("Path to mode-0600 fake JSON snapshot", self.s.fake_path or "private/fake.json")
                if fp:
                    ctl = fake_from_file(fp)
                    self.s.fake_path = fp
                    self.s.bdf = None
                    self.s.adapter_spec = None
                    self._cached_fake = ctl
                    self.show(f"Loaded fake simulation backend from {fp}.")
            elif sub == "e":
                jp = self._ask("Private mode-0600 journal JSONL path", self.s.journal_path)
                if jp:
                    self.s.journal_path = jp
            elif sub == "f":
                self.s.receipt_path = self._ask_clearable("Cold-verified test receipt path", self.s.receipt_path)
            elif sub == "g":
                if self.s.execute_live:
                    self.s.execute_live = False
                    self.show("Live Execute mode turned OFF (read-only/dry-run).")
                else:
                    self.show(
                        "WARNING: Enabling Live Execute permits physical CMD_PROGRAM writes "
                        "(still gated by SENSE_CTRL, pre-trigger checks, and per-record YES <row> <word>)."
                    )
                    if self._confirm("Type 'ENABLE LIVE WRITES' to arm Live Execute: ", "ENABLE LIVE WRITES"):
                        self.s.execute_live = True
                        self.show("Live Execute mode ENABLED.")
                    else:
                        self.show("Kept Live Execute mode OFF.")

    # ------------------------------------------------------------------ [2] Review Plan
    def _menu_review_plan(self) -> None:
        p = self._load_plan()
        backend = None
        owned = False
        has_target = bool(self.s.fake_path or self.s.bdf or self.s.adapter_spec)
        if has_target:
            try:
                backend, owned = self._open_backend(writable=False)
            except Exception as exc:
                self.show(f"\nNOTE: target unavailable for FF row scan ({exc}); showing offline review.")
                backend, owned = None, False
        try:
            self.show("")
            for line in format_plan_review(p, reader=backend, override_slot=self.s.overrides.slot):
                self.show(line)
        finally:
            self._close_if_owned(backend, owned)

    # ------------------------------------------------------------------ [3] Status & Inspect
    def _menu_status_inspect(self) -> None:
        p = self._load_plan()
        backend, owned = self._open_backend(writable=False)
        try:
            check_host_readonly_safe(backend.host())
            for line in format_status_report(p, backend, allow_vout_override=self.s.overrides.vout_range):
                self.show(line)
            with Journal(self.s.journal_path) as j:
                try:
                    boot = preflight(p, backend, active_sense=False, overrides=self.s.overrides)
                    row, surrounding = scan_slots(p, backend, override_slot=self.s.overrides.slot)
                    self.show(f"\nPASSIVE INSPECTION OK | boot={boot} | proved next free row={row}")
                    for i, v in surrounding:
                        tag = "EMPTY" if v == p.empty_word else "WRITTEN"
                        self.show(f"  row {i:4d}: {hex32(v)}  [{tag}]")
                    j.append(
                        event="inspect_result",
                        plan_sha256=p.content_hash,
                        boot_id=boot,
                        active_sense=False,
                        proved_pointer=row,
                        status="PASS",
                    )
                except BurnError as exc:
                    j.append(
                        event="inspect_failed",
                        plan_sha256=p.content_hash,
                        active_sense=False,
                        status="FAIL",
                        error=str(exc),
                    )
                    self.show(f"\nPassive preflight / slot proof did NOT pass: {exc}")
        finally:
            self._close_if_owned(backend, owned)

    # ------------------------------------------------------------------ [4] Active Sense
    def _menu_active_sense(self) -> None:
        p = self._load_plan()
        backend, owned = self._open_backend(writable=False)
        try:
            check_host_readonly_safe(backend.host())
            before = backend.health()
            pre_issues = evaluate_health(before, p, allow_vout_override=self.s.overrides.vout_range)
            if pre_issues:
                raise BurnError("passive health not ready for SENSE_CTRL: " + "; ".join(pre_issues))
            self.show(
                "\nACTIVE SENSE (SENSE_CTRL): does not burn fuses, restores EFP 0->1 while keeping DSOV=0, "
                "but can collapse a too-deep held frame (only one SENSE_CTRL per glitch window)."
            )
            with Journal(self.s.journal_path) as j:
                if not self._confirm("Type 'SENSE' to run active_sense_readonly(): ", "SENSE"):
                    j.append(event="declined", command="menu_sense", reason="SENSE")
                    self.show("Active sense cancelled.")
                    return
                j.append(event="active_sense_confirmed", command="menu_sense")
                backend.active_sense_readonly()
                after = backend.health()
                post_issues = evaluate_health(
                    after, p, allow_vout_override=self.s.overrides.vout_range, require_efp_one=True
                )
                j.append(
                    event="manual_active_sense",
                    before=before.serializable(),
                    after=after.serializable(),
                    issues=post_issues,
                )
                self.show(
                    f"After SENSE_CTRL: FUSECTRL={hex32(after.fusectrl)} DSOV={after.dsov} "
                    f"CRC={hex32(after.crc)} PLM_f8={hex32(after.plm_f8)} EFP={after.efp} SBFE={after.sbfe}"
                )
                if post_issues:
                    self.show("WARNING — frame degraded after SENSE_CTRL:")
                    for issue in post_issues:
                        self.show(f"  - {issue}")
                else:
                    self.show("PASS: active sense clean; frame is armed and write-capable.")
        finally:
            self._close_if_owned(backend, owned)

    # ------------------------------------------------------------------ [5] Dry-Run
    def _menu_dry_run(self) -> None:
        p = self._load_plan()
        backend, owned = self._open_backend(writable=False)
        try:
            with Journal(self.s.journal_path) as j:
                process_records(
                    p,
                    backend,
                    MissingWriter(),
                    j,
                    stage="all",
                    confirm=self._confirm,
                    overrides=self.s.overrides,
                    dry_run=True,
                    run_active_sense=False,
                    show=self.show,
                )
            self.show("DRY-RUN COMPLETE (passive preflight only; no fuses touched).")
        finally:
            self._close_if_owned(backend, owned)

    # ------------------------------------------------------------------ [6] Single-Window Burn
    def _menu_burn_single_window(self) -> None:
        self._run_burn_stage("all", receipt=None, max_records=None)

    # ------------------------------------------------------------------ [7] Split-Window Burn
    def _menu_burn_split(self) -> None:
        self.show("\n--- [7] Split-Window Burn ---")
        self.show("  [1] Stage 'test' (burn ONLY the first/test record, then stop)")
        self.show("  [2] Stage 'real' (burn real records; requires cold-verified test receipt)")
        pick = self.input_fn("Choose [1-2, Enter to cancel]: ").strip()
        if pick == "1":
            self._run_burn_stage("test", receipt=None, max_records=None)
        elif pick == "2":
            max_s = self._ask("Max records this session (blank for all)", "")
            max_n = int(max_s) if max_s else None
            self._run_burn_stage("real", receipt=self.s.receipt_path, max_records=max_n)

    def _run_burn_stage(self, stage: str, *, receipt: str | None, max_records: int | None) -> None:
        p = self._load_plan()
        self._check_execute_policy()
        is_fake = bool(self.s.fake_path)
        want_write = not is_fake and self.s.execute_live
        backend, owned = self._open_backend(writable=want_write)
        try:
            self._require_execute_ready(backend)
            writer = backend
            with Journal(self.s.journal_path) as j:
                process_records(
                    p,
                    backend,
                    writer,
                    j,
                    stage=stage,
                    confirm=self._confirm,
                    test_receipt=receipt,
                    overrides=self.s.overrides,
                    max_records=max_records,
                    dry_run=False,
                    strict_confirm=not is_fake,
                    confirm_active_sense=not is_fake,
                    show=self.show,
                )
            if is_fake:
                self.show("SIMULATION COMPLETE (in-memory fake rows updated; no GPU contacted).")
            else:
                self.show("READ-BACK VERIFIED (NOT cold-boot verified). Cold boot and run [11] Verify Cold.")
        finally:
            self._close_if_owned(backend, owned)

    # ------------------------------------------------------------------ [8] Overrides
    def _menu_overrides(self) -> None:
        while True:
            ov = self.s.overrides
            self.show("\n--- [8] Granular Manual Overrides (each still requires typed confirm at burn time) ---")
            self.show(f"  [1] Override target FF slot (ambiguous gap)   : {ov.slot if ov.slot is not None else 'OFF'}")
            self.show(f"  [2] Skip automatic SENSE_CTRL                 : {'ON' if ov.skip_active_sense else 'OFF'}")
            self.show(f"  [3] Allow real stage without test receipt     : {'ON' if ov.no_test_receipt else 'OFF'}")
            self.show(f"  [4] Skip initial expected_masks check         : {'ON' if ov.mask_mismatch else 'OFF'}")
            self.show(f"  [5] Allow VOUT outside stock +/- tol          : {'ON' if ov.vout_range else 'OFF'}")
            self.show(f"  [6] Override undetermined/exceeded SM ceiling : {'ON' if ov.undetermined_ceiling else 'OFF'}")
            self.show(f"  [7] Manual-only: allow DEBUGCTRL target       : {'ON' if ov.allow_manual_debugctrl else 'OFF'}")
            self.show("  [r] Reset ALL overrides to strict defaults")
            self.show("  [0] Back to main menu")
            self.show("  (Never overrideable: aligned 32-bit stores, no nvidia/nouveau, Die ID, rows 222..486,")
            self.show("   IFF bit 5=0, blank slot, genuine defects, hard 400..1500mV, read-back & post-write gate.)")
            pick = self.input_fn("Toggle [1-7, r, 0]: ").strip().lower()
            if pick in ("0", ""):
                return
            if pick == "1":
                val = self._ask_clearable(
                    "Target EMPTY FF row (222..486)",
                    str(ov.slot) if ov.slot is not None else None,
                )
                self.s.overrides = replace(ov, slot=int(val, 0) if val is not None else None)
            elif pick == "2":
                self.s.overrides = replace(ov, skip_active_sense=not ov.skip_active_sense)
            elif pick == "3":
                self.s.overrides = replace(ov, no_test_receipt=not ov.no_test_receipt)
            elif pick == "4":
                self.s.overrides = replace(ov, mask_mismatch=not ov.mask_mismatch)
            elif pick == "5":
                self.s.overrides = replace(ov, vout_range=not ov.vout_range)
            elif pick == "6":
                self.s.overrides = replace(ov, undetermined_ceiling=not ov.undetermined_ceiling)
            elif pick == "7":
                self.s.overrides = replace(ov, allow_manual_debugctrl=not ov.allow_manual_debugctrl)
            elif pick == "r":
                self.s.overrides = Overrides()
                self.show("All manual overrides reset to OFF.")

    # ------------------------------------------------------------------ [9] Manual Console
    def _menu_manual_console(self) -> None:
        p = self._load_plan()
        is_fake = bool(self.s.fake_path)
        execute_live = (not is_fake) and self.s.execute_live
        if execute_live:
            self._check_execute_policy()
        backend, owned = self._open_backend(writable=execute_live)
        try:
            if execute_live:
                self._require_execute_ready(backend)
            writer = backend if (is_fake or execute_live) else MissingWriter()
            with Journal(self.s.journal_path) as j:
                ManualConsole(
                    p,
                    backend,
                    writer,
                    j,
                    execute=execute_live,
                    test_receipt=self.s.receipt_path,
                    overrides=self.s.overrides,
                    strict_confirm=execute_live,
                    input_fn=self.input_fn,
                    show=self.show,
                ).run()
        finally:
            self._close_if_owned(backend, owned)

    # ------------------------------------------------------------------ [10] Decode FF Word
    def _menu_decode_word(self) -> None:
        raw = self._ask("Enter 32-bit FF word (e.g. 0x0fe805c1)")
        if not raw:
            return
        word = u32(raw, "word")
        ff = decode_ff(word)
        self.show(f"FF word {hex32(word)}: chain={ff.chain} block={ff.block} data=0x{ff.data:04x} (IFF bit5=0)")
        overlapping = fields_in_block(ff.chain, ff.block)
        if overlapping:
            self.show("Overlapping GA100 option fields:")
            for f in overlapping:
                self.show(f"  - {f.key:24} ({f.name}) bits {f.bit}..{f.bit + f.width - 1} [{f.category}]")
        else:
            self.show("Overlapping GA100 option fields: none in known map")

    # ------------------------------------------------------------------ [11] Cold Snapshot & Verify
    def _menu_cold_verify(self) -> None:
        self.show("\n--- [11] Cold-Boot Snapshot & Verify ---")
        self.show("  [1] Capture cold snapshot from live card (--bdf) -> mode-0600 JSON")
        self.show("  [2] Verify cold snapshot against plan (verify-cold)")
        pick = self.input_fn("Choose [1-2, Enter to cancel]: ").strip()
        if pick == "1":
            if not self.s.bdf:
                raise BurnError("select a live GPU (--bdf) in [1] Setup first")
            prev = self._ask("Previous burn session boot_id (from journal)")
            out = self._ask("Output cold snapshot path", "private/cold.json")
            backend, owned = self._open_backend(writable=False)
            try:
                check_host_readonly_safe(backend.host())
                snap = backend.capture_cold_snapshot(previous_boot_id=prev, read_rows=True)
                write_private_json(out, snap)
                self.show(f"Saved cold snapshot to {out} (mode 0600, SM_COUNT={snap['fields']['SM_COUNT']}).")
            finally:
                self._close_if_owned(backend, owned)
        elif pick == "2":
            p = self._load_plan()
            cold_path = require_private_file(self._ask("Cold snapshot JSON path", "private/cold.json"), "cold snapshot")
            stage = self._ask("Verification stage (test or real)", "real")
            receipt_out = self._ask("Optional receipt output path (blank to skip)", "private/test.receipt" if stage == "test" else "")
            cold_data = json.loads(cold_path.read_text(encoding="utf-8"))
            with Journal(self.s.journal_path) as j:
                receipt = verify_cold_snapshot(p, cold_data, stage=stage, journal=j)
            if receipt_out:
                write_private_json(receipt_out, receipt)
                if stage == "test":
                    self.s.receipt_path = receipt_out
            self.show("PASS: cold snapshot matches declared expectations (not independently authenticated).")

    # ------------------------------------------------------------------ [12] Experimental Registers
    def _menu_experimental_registers(self) -> None:
        backend, owned = self._open_backend(writable=False)
        try:
            check_host_readonly_safe(backend.host())
            readings, poisoned = read_experimental_registers(backend)
            self.show("")
            for line in format_experimental_report(
                backend, readings=readings, poisoned=poisoned
            ):
                self.show(line)
            with Journal(self.s.journal_path) as j:
                j.append(
                    event="manual_experimental_read",
                    pci_id=backend.pci_id(),
                    skipped_poison=poisoned,
                    registers={k: hex32(v) for k, v in readings.items()},
                )
        finally:
            self._close_if_owned(backend, owned)
