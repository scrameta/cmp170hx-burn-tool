"""Standalone CLI for vfi_burn.

Supports:
  - Direct Linux GA100 hardware access via ``--bdf`` (+ optional Pico ``--port``)
  - Custom owner-supplied adapters via ``--adapter module:factory``
  - Offline simulation via ``--fake``
  - ``cmp170hx-ffcalc`` plan import, FF record decoding, and cold-boot snapshot capture

Physical burning is passive DRY-RUN by default; live writes require ``--execute``
on an interactive TTY plus explicit per-record row+word confirmation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
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
from .console import ManualConsole, format_status_report
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
from .mmio import resolve_register_address
from .model import BurnError, Plan, Record, is_poison, require_private_file, u32


def prompt_confirm(prompt: str, token: str) -> bool:
    try:
        return input(prompt).strip() == token
    except (EOFError, KeyboardInterrupt):
        return False


def _add_live_hw_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--port",
        default=None,
        help="optional Pico serial port (e.g. /dev/ttyACM0) for live PMBus READ_VOUT/STATUS_WORD",
    )
    parser.add_argument(
        "--assume-vout-mv",
        type=float,
        default=None,
        help="only when --port is omitted: externally measured NVVDD rail in mV (e.g. 787.5)",
    )


def _add_override_args(parser: argparse.ArgumentParser, *, allow_debugctrl: bool = False) -> None:
    grp = parser.add_argument_group(
        "manual overrides (all off by default; each requires explicit typed confirmation and is journaled)"
    )
    grp.add_argument(
        "--override-slot",
        type=lambda s: int(s, 0),
        default=None,
        metavar="ROW",
        help="use an operator-selected EMPTY FPF/FF row when contiguous scan is ambiguous",
    )
    grp.add_argument(
        "--override-skip-sense",
        action="store_true",
        help="skip automatic SENSE_CTRL (passive goldilocks checks still enforced)",
    )
    grp.add_argument(
        "--override-no-test-receipt",
        action="store_true",
        help="allow real stage without a vfi_burn test receipt (if test row was verified externally)",
    )
    grp.add_argument(
        "--override-mask-mismatch",
        action="store_true",
        help="skip initial expected_masks equality check (e.g. after external/manual changes)",
    )
    grp.add_argument(
        "--override-vout-range",
        action="store_true",
        help="allow READ_VOUT outside stock_mv +/- tolerance_mv (hard 400..1500 mV and latch limits remain)",
    )
    grp.add_argument(
        "--override-undetermined-ceiling",
        action="store_true",
        help="allow operator judgment when SM count is undetermined/above ceiling (genuine defects remain REFUSED)",
    )
    if allow_debugctrl:
        grp.add_argument(
            "--override-allow-debugctrl",
            action="store_true",
            help="manual-only: permit DEBUGCTRL (0x820060) target behind two separate typed confirmations",
        )


def _overrides_from_args(args: argparse.Namespace) -> Overrides:
    return Overrides(
        slot=getattr(args, "override_slot", None),
        skip_active_sense=bool(getattr(args, "override_skip_sense", False)),
        no_test_receipt=bool(getattr(args, "override_no_test_receipt", False)),
        mask_mismatch=bool(getattr(args, "override_mask_mismatch", False)),
        vout_range=bool(getattr(args, "override_vout_range", False)),
        undetermined_ceiling=bool(getattr(args, "override_undetermined_ceiling", False)),
        allow_manual_debugctrl=bool(getattr(args, "override_allow_debugctrl", False)),
    )


def _require_live_execute_ready(backend) -> None:
    if not sys.stdin.isatty():
        raise BurnError(
            "REFUSED --execute without an interactive TTY; unattended/piped burns are forbidden"
        )
    if not has_verified_writer(backend):
        raise BurnError(
            "REFUSED --execute: supplied adapter does not implement a verified Writer "
            "(must define write_one_record(row, word) AND set writer_verified = True "
            "after independent bench verification)"
        )


def _open_backend(args: argparse.Namespace, *, writable: bool = False):
    fake_path = getattr(args, "fake", None)
    if fake_path:
        return fake_from_file(fake_path)
    bdf = getattr(args, "bdf", None)
    if bdf:
        return GA100LiveAdapter(
            bdf,
            pico_port=getattr(args, "port", None),
            writable=writable,
            assume_vout_mv=getattr(args, "assume_vout_mv", None),
        )
    return load_adapter(args.adapter)


def _print_plan_summary(p: Plan, final_masks: dict[str, int]) -> None:
    print(f"OFFLINE PLAN REVIEW | SHA256: {p.content_hash}")
    print(f"  SKU          : {p.pci_id} (CMP 170HX 8 GB)")
    print(
        f"  FF Window    : rows {p.first_row}..{p.last_row} "
        f"({p.last_row - p.first_row + 1} slots) | empty_word={hex32(p.empty_word)}"
    )
    print(
        f"  Rail / VRM   : stock={p.stock_mv} +/- {p.vout_tolerance_mv} mV | "
        f"allowed_status_mask=0x{p.allowed_status_mask:04x}"
    )
    print(
        f"  SM Ceilings  : base={p.base_sm_ceiling} | override={p.override_sm_ceiling} | "
        f"reconfig_verified={p.reconfig_semantics_verified}"
    )
    print(
        "  Die ID (8-15): "
        + " ".join(f"{k}:{hex32(v)}" for k, v in sorted(p.die_rows.items()))
    )
    print(
        "  Defect Masks : "
        + (", ".join(f"{k}={hex32(v)}" for k, v in sorted(p.defect_masks.items())) or "none")
    )
    print(
        "  Mask Plan    : "
        + ", ".join(
            f"{k}: {hex32(p.expected_masks[k])} -> {hex32(final_masks[k])}"
            for k in sorted(p.expected_masks)
        )
    )
    print(f"  Records ({len(p.records)}):")
    for i, r in enumerate(p.records):
        extra = ""
        if r.mask_field is not None:
            extra = f" [{r.mask_field}: {hex32(r.from_disable)}->{hex32(r.to_disable)}"
            if r.resulting_sm_count is not None:
                extra += f", {r.resulting_sm_count} SM"
            extra += "]"
        elif r.kind == "pcie":
            extra = f" [LnkCap2: {hex32(r.link_cap2_before)}->{hex32(r.link_cap2_after)}]"
        try:
            ff = decode_ff(r.word)
            ff_tag = f" (c{ff.chain}/b{ff.block}=0x{ff.data:04x})"
        except BurnError:
            ff_tag = " [WARNING: IFF bit 5 SET — not a valid GA100 FF record!]"
        print(
            f"    {i:2d}: {r.kind:8} {r.name:22} target={r.target:18} "
            f"word={hex32(r.word)}{ff_tag}{extra} | {r.decode}"
        )
    print("OFFLINE ONLY: no hardware contacted.")


def _warn_if_unhealthy_readonly(plan: Plan, backend) -> None:
    try:
        issues = evaluate_health(backend.health(), plan)
    except Exception as exc:
        print(f"NOTE: health read failed ({exc}); proceeding with read-only diagnostic", file=sys.stderr)
        return
    if issues:
        print("NOTE: card is NOT in a write-ready goldilocks state:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vfi_burn",
        description="CMP 170HX 8 GB FF fuse-burn safety controller (passive dry-run by default)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_plan(p: argparse.ArgumentParser) -> None:
        p.add_argument("--plan", required=True, help="private mode-0600 per-card plan JSON")

    plan_p = sub.add_parser("plan", help="pure offline plan validation and review; never touches hardware")
    add_plan(plan_p)

    decode_p = sub.add_parser("decode", help="decode a 32-bit GA100 FF record word into chain/block/data/fields")
    decode_p.add_argument("word", help="32-bit FF record word in hex or decimal (e.g. 0x0fe805c1)")

    imp_p = sub.add_parser(
        "import-ffcalc",
        help="convert a cmp170hx-ffcalc burn.json + 512-row card dump.json into a mode-0600 vfi_burn plan",
    )
    imp_p.add_argument("--burn-plan", required=True, help="mode-0600 cmp170hx-ffcalc plans/<uuid>.burn.json")
    imp_p.add_argument("--dump", required=True, help="mode-0600 cmp170hx-ffcalc cards/<uuid>.json (with all 512 rows)")
    imp_p.add_argument("--out", required=True, help="output path for new mode-0600 vfi_burn plan JSON")
    imp_p.add_argument("--stock-mv", type=float, default=787.5)
    imp_p.add_argument("--tolerance-mv", type=float, default=25.0)
    imp_p.add_argument("--base-sm", type=int, default=128)
    imp_p.add_argument("--override-sm", type=int, default=None)
    imp_p.add_argument(
        "--prepend-noop-test",
        action="store_true",
        help="prepend a safe zero-effect no-op FF test row (recommended so SM/PES/ROP_L2 burn together)",
    )

    snap_p = sub.add_parser(
        "snapshot",
        help="capture a mode-0600 cold-boot snapshot from a live CMP 170HX for verify-cold",
    )
    snap_p.add_argument("--bdf", required=True, help="PCI BDF of CMP 170HX (e.g. 0000:05:00.0)")
    _add_live_hw_args(snap_p)
    snap_p.add_argument("--previous-boot-id", required=True, help="boot_id from the burn session journal")
    snap_p.add_argument("--out", required=True, help="output path for new mode-0600 cold snapshot JSON")
    snap_p.add_argument("--no-rows", action="store_true", help="skip reading FF rows 222..486 (options + die ID only)")

    inspect_p = sub.add_parser("inspect", help="read-only preflight, health report, and free-slot proof")
    add_plan(inspect_p)
    ig = inspect_p.add_mutually_exclusive_group(required=True)
    ig.add_argument("--fake", help="synthetic JSON controller snapshot")
    ig.add_argument("--bdf", help="live CMP 170HX PCI BDF (e.g. 0000:05:00.0)")
    ig.add_argument("--adapter", help="custom owner-supplied live module:factory")
    _add_live_hw_args(inspect_p)
    inspect_p.add_argument(
        "--sense",
        action="store_true",
        help="explicitly invoke SENSE_CTRL active sense (confirmed with 'SENSE' on live hardware)",
    )
    inspect_p.add_argument(
        "--journal",
        help="private mode-0600 audit JSONL (recommended so inspect/sense results and failures are logged)",
    )

    burn_p = sub.add_parser(
        "burn",
        help="live hardware burn workflow (passive DRY-RUN by default; requires --execute + TTY)",
    )
    add_plan(burn_p)
    bg = burn_p.add_mutually_exclusive_group(required=True)
    bg.add_argument("--bdf", help="live CMP 170HX PCI BDF (e.g. 0000:05:00.0)")
    bg.add_argument("--adapter", help="custom owner-supplied live module:factory")
    _add_live_hw_args(burn_p)
    burn_p.add_argument(
        "--stage",
        required=True,
        choices=("test", "real", "all"),
        help="'all' burns test row then asks CONTINUE SAME WINDOW to burn real records without losing the glitch; 'test'/'real' splits across a cold boot",
    )
    burn_p.add_argument("--journal", required=True, help="private mode-0600 audit JSONL")
    burn_p.add_argument("--receipt", help="cold-verified test receipt (required for --stage real)")
    burn_p.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="stop cleanly after writing N records",
    )
    burn_p.add_argument(
        "--sense",
        action="store_true",
        help="also run confirmed SENSE_CTRL during dry-run (always runs on --execute unless overridden)",
    )
    burn_p.add_argument(
        "--execute",
        action="store_true",
        help="actually program FF fuses (without this flag, passive dry-run only)",
    )
    _add_override_args(burn_p, allow_debugctrl=False)

    sim_p = sub.add_parser("simulate", help="exercise the exact write loop on FAKE data only")
    add_plan(sim_p)
    sim_p.add_argument("--fake", required=True, help="synthetic mode-0600 JSON snapshot")
    sim_p.add_argument("--stage", required=True, choices=("test", "real", "all"))
    sim_p.add_argument("--journal", required=True, help="private mode-0600 audit JSONL")
    sim_p.add_argument("--receipt", help="cold-verified test receipt (required for --stage real)")
    sim_p.add_argument("--max-records", type=int, default=None, help="stop cleanly after N records")
    sim_p.add_argument("--dry-run", action="store_true", help="preview only; do not update fake rows")
    sim_p.add_argument("--save-fake", help="write updated fake snapshot to a NEW mode-0600 file")
    _add_override_args(sim_p, allow_debugctrl=False)

    manual_p = sub.add_parser(
        "manual",
        help="read-only diagnostics, interactive console, and gated single-record write",
    )
    add_plan(manual_p)
    manual_p.add_argument("--journal", required=True, help="private mode-0600 audit JSONL")
    mg = manual_p.add_mutually_exclusive_group(required=True)
    mg.add_argument("--fake", help="synthetic JSON controller snapshot")
    mg.add_argument("--bdf", help="live CMP 170HX PCI BDF (e.g. 0000:05:00.0)")
    mg.add_argument("--adapter", help="custom owner-supplied live module:factory")
    _add_live_hw_args(manual_p)
    man_sub = manual_p.add_subparsers(dest="action", required=True)

    man_sub.add_parser("status", help="print full health, host, die ID, and mask status (never writes)")

    rr = man_sub.add_parser("read-row", help="read a single fuse row (works even if card is faulted/cold)")
    rr.add_argument("row", type=lambda s: int(s, 0))

    rrs = man_sub.add_parser("read-rows", help="read a range of fuse rows (start..end inclusive)")
    rrs.add_argument("start", type=lambda s: int(s, 0))
    rrs.add_argument("end", type=lambda s: int(s, 0))

    rg = man_sub.add_parser("read-reg", help="read aligned 32-bit MMIO reg (hex or FUSECTRL/DEBUGCTRL/PLM_F8)")
    rg.add_argument("address")

    rf = man_sub.add_parser("read-field", help="read named decoded field")
    rf.add_argument("name")

    mc = man_sub.add_parser("console", help="interactive manual fuse-controller console")
    mc.add_argument("--receipt", help="cold-verified test receipt for manual writes")
    mc.add_argument("--execute", action="store_true", help="enable live writes on a verified writer")
    _add_override_args(mc, allow_debugctrl=True)

    mw = man_sub.add_parser(
        "write-record",
        help="write one operator-supplied record JSON (passive dry-run on live hardware unless --execute)",
    )
    mw.add_argument("--record", required=True, help="private mode-0600 JSON record object")
    mw.add_argument("--receipt", help="cold-verified test row receipt")
    mw.add_argument("--execute", action="store_true", help="execute on live verified adapter")
    mw.add_argument("--dry-run", action="store_true", help="preview only without writing")
    mw.add_argument("--save-fake", help="write updated fake snapshot to a NEW mode-0600 file")
    _add_override_args(mw, allow_debugctrl=True)

    msw = man_sub.add_parser(
        "simulate-write-record",
        help="single reviewed record on FAKE backend only (backward-compatible alias)",
    )
    msw.add_argument("--record", required=True, help="private mode-0600 JSON record object")
    msw.add_argument("--receipt", help="cold-verified test row receipt")
    msw.add_argument("--save-fake", help="write updated fake snapshot to a NEW mode-0600 file")
    _add_override_args(msw, allow_debugctrl=True)

    verify_p = sub.add_parser("verify-cold", help="compare user-supplied cold snapshot with expected fields")
    add_plan(verify_p)
    verify_p.add_argument("--cold", required=True, help="private mode-0600 cold snapshot JSON")
    verify_p.add_argument("--stage", required=True, choices=("test", "real"))
    verify_p.add_argument("--journal", required=True, help="private mode-0600 audit JSONL")
    verify_p.add_argument("--receipt-out", help="create new mode-0600 PASS receipt")

    return ap


def _handle_manual_write(args, p: Plan, backend, *, allow_live: bool) -> int:
    is_fake = bool(args.fake)
    if not is_fake and not allow_live:
        raise BurnError("manual physical writes require 'manual write-record --execute' with a verified writer")

    record_path = require_private_file(args.record, "manual record file")
    try:
        record = Record.parse(json.loads(record_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise BurnError(f"invalid JSON in manual record {record_path}: {exc}") from exc
    if record.kind == "test":
        raise BurnError("manual write-record cannot insert a 'test' record; use stage=test")

    overrides = _overrides_from_args(args)
    if not allow_live:
        overrides = replace(overrides, allow_manual_debugctrl=True)

    execute_live = (not is_fake) and bool(getattr(args, "execute", False))
    dry_run = bool(getattr(args, "dry_run", False)) or ((not is_fake) and (not execute_live))
    if execute_live and not dry_run:
        _require_live_execute_ready(backend)

    writer = backend if (is_fake or execute_live) else MissingWriter()
    rec_hash = hashlib.sha256(record_path.read_bytes()).hexdigest()
    single = replace(p, records=(p.records[0], record))
    with Journal(args.journal) as j:
        j.append(
            event="manual_selected_record",
            source_file_sha256=rec_hash,
            record=record.name,
            target=record.target,
            word=hex32(record.word),
            original_card_plan_sha256=p.content_hash,
            dry_run=dry_run,
            note="manual record is not part of the original approved list",
        )
        process_records(
            single,
            backend,
            writer,
            j,
            stage="real",
            confirm=prompt_confirm,
            test_receipt=args.receipt,
            overrides=overrides,
            dry_run=dry_run,
            strict_confirm=execute_live,
            confirm_active_sense=execute_live,
            run_active_sense=not dry_run,
            manual_record_sha256=rec_hash,
            is_manual_single=True,
        )
    if is_fake and getattr(args, "save_fake", None) and not dry_run:
        write_private_json(args.save_fake, snapshot_data(backend))
    if dry_run:
        print("DRY-RUN COMPLETE: no fuses written.")
    elif is_fake:
        print("SIMULATION ONLY: fake record updated; no GPU contacted.")
    else:
        print("READ-BACK VERIFIED (NOT cold-boot verified). Cold boot and verify before trusting unlock.")
    return 0


def _capture_inspect_snapshot(backend) -> tuple[dict | None, dict | None]:
    health_snap, host_snap = None, None
    try:
        health_snap = backend.health().serializable()
    except Exception:
        pass
    try:
        host_snap = backend.host().serializable()
    except Exception:
        pass
    return health_snap, host_snap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    backend = None
    try:
        if args.cmd == "decode":
            word = u32(args.word, "word")
            ff = decode_ff(word)
            print(f"FF word {hex32(word)}: chain={ff.chain} block={ff.block} data=0x{ff.data:04x} (IFF bit5=0)")
            overlapping = fields_in_block(ff.chain, ff.block)
            if overlapping:
                print("Overlapping GA100 option fields:")
                for f in overlapping:
                    print(f"  - {f.key:24} ({f.name}) bits {f.bit}..{f.bit + f.width - 1} [{f.category}]")
            else:
                print("Overlapping GA100 option fields: none in known map")
            return 0

        if args.cmd == "import-ffcalc":
            plan_dict = import_ffcalc_plan(
                args.burn_plan,
                args.dump,
                stock_mv=args.stock_mv,
                tolerance_mv=args.tolerance_mv,
                base_sm_ceiling=args.base_sm,
                override_sm_ceiling=args.override_sm,
                prepend_noop_test=args.prepend_noop_test,
            )
            write_private_json(args.out, plan_dict)
            # Validate the written plan immediately with Plan.load + check_plan!
            loaded = Plan.load(args.out)
            final_masks = check_plan(loaded)
            print(f"Imported ffcalc plan to {args.out} (mode 0600):")
            _print_plan_summary(loaded, final_masks)
            return 0

        if args.cmd == "snapshot":
            backend = GA100LiveAdapter(
                args.bdf,
                pico_port=args.port,
                writable=False,
                assume_vout_mv=args.assume_vout_mv,
            )
            check_host_readonly_safe(backend.host())
            snap = backend.capture_cold_snapshot(
                previous_boot_id=args.previous_boot_id,
                read_rows=not args.no_rows,
            )
            write_private_json(args.out, snap)
            print(
                f"Saved cold snapshot {args.out} (mode 0600) | boot_id={snap['boot_id']} "
                f"rows_nonzero={len(snap['fpf_rows'])} SM_COUNT={snap['fields']['SM_COUNT']}"
            )
            return 0

        p = Plan.load(args.plan)
        final_masks = check_plan(p)

        if args.cmd == "plan":
            _print_plan_summary(p, final_masks)
            return 0

        if args.cmd in ("inspect", "burn", "manual", "simulate"):
            want_write = (
                (args.cmd == "burn" and bool(args.execute))
                or (
                    args.cmd == "manual"
                    and args.action in ("write-record", "console")
                    and bool(getattr(args, "execute", False))
                    and not getattr(args, "fake", None)
                )
            )
            backend = _open_backend(args, writable=want_write)

        if args.cmd == "inspect":
            for line in format_status_report(p, backend):
                print(line)
            journal = Journal(args.journal) if args.journal else None
            try:
                if args.sense and not args.fake:
                    print(
                        "ACTIVE SENSE: active_sense_readonly() (SENSE_CTRL) does not burn fuses, "
                        "but can collapse a too-deep held goldilocks frame (forcing power cycle + re-hunt)."
                    )
                    if not prompt_confirm("Type 'SENSE' to run active_sense_readonly(): ", "SENSE"):
                        if journal is not None:
                            journal.append(
                                event="declined",
                                command="inspect",
                                plan_sha256=p.content_hash,
                                reason="SENSE",
                            )
                        raise BurnError("active sense declined by operator")
                    if journal is not None:
                        journal.append(
                            event="active_sense_confirmed",
                            command="inspect",
                            plan_sha256=p.content_hash,
                        )
                boot = preflight(p, backend, active_sense=args.sense)
                row, surrounding = scan_slots(p, backend)
                if journal is not None:
                    h_snap, host_snap = _capture_inspect_snapshot(backend)
                    journal.append(
                        event="inspect_result",
                        plan_sha256=p.content_hash,
                        boot_id=boot,
                        active_sense=args.sense,
                        proved_pointer=row,
                        health=h_snap,
                        host=host_snap,
                        status="PASS",
                    )
            except Exception as exc:
                if journal is not None:
                    h_snap, host_snap = _capture_inspect_snapshot(backend)
                    journal.append(
                        event="inspect_failed",
                        plan_sha256=p.content_hash,
                        active_sense=args.sense,
                        health=h_snap,
                        host=host_snap,
                        status="FAIL",
                        error=str(exc),
                    )
                raise
            finally:
                if journal is not None:
                    journal.close()
            print(f"{'FAKE' if args.fake else 'LIVE READ-ONLY'} | boot {boot}; first free row {row}")
            for i, v in surrounding:
                tag = "EMPTY" if v == p.empty_word else "WRITTEN"
                print(f"  row {i:4d}: {hex32(v)}  [{tag}]")
            print(
                "FULL write-capable preflight NOT established (re-run with --sense to test active sense)"
                if not args.sense
                else "Active-sense checks passed; frame is write-capable (no write issued)"
            )
            return 0

        if args.cmd == "burn":
            overrides = _overrides_from_args(args)
            dry_run = not args.execute
            if not dry_run:
                _require_live_execute_ready(backend)
            run_sense = (not dry_run) or bool(args.sense)
            writer = backend if not dry_run else MissingWriter()
            with Journal(args.journal) as j:
                process_records(
                    p,
                    backend,
                    writer,
                    j,
                    stage=args.stage,
                    confirm=prompt_confirm,
                    test_receipt=args.receipt,
                    overrides=overrides,
                    max_records=args.max_records,
                    dry_run=dry_run,
                    strict_confirm=not dry_run,
                    confirm_active_sense=True,
                    run_active_sense=run_sense,
                )
            if dry_run:
                sense_note = (
                    "active-sense & slot proof passed"
                    if run_sense and not overrides.skip_active_sense
                    else "passive preflight & slot proof passed (add --sense to test active sense)"
                )
                print(f"DRY-RUN COMPLETE: {sense_note}; no fuses written. Pass --execute to burn.")
            else:
                print("READ-BACK VERIFIED (NOT cold-boot verified). Cold boot and run verify-cold.")
            return 0

        if args.cmd == "simulate":
            overrides = _overrides_from_args(args)
            with Journal(args.journal) as j:
                process_records(
                    p,
                    backend,
                    backend,
                    j,
                    stage=args.stage,
                    confirm=prompt_confirm,
                    test_receipt=args.receipt,
                    overrides=overrides,
                    max_records=args.max_records,
                    dry_run=args.dry_run,
                )
            if args.save_fake and not args.dry_run:
                write_private_json(args.save_fake, snapshot_data(backend))
            print(
                "SIMULATION DRY-RUN ONLY: no fake rows modified."
                if args.dry_run
                else "SIMULATION ONLY: fake records updated; no GPU contacted."
            )
            return 0

        if args.cmd == "manual":
            if args.action == "simulate-write-record":
                return _handle_manual_write(args, p, backend, allow_live=False)
            if args.action == "write-record":
                return _handle_manual_write(args, p, backend, allow_live=True)
            if args.action == "console":
                overrides = _overrides_from_args(args)
                is_fake = bool(args.fake)
                execute_live = (not is_fake) and bool(args.execute)
                if execute_live:
                    _require_live_execute_ready(backend)
                writer = backend if (is_fake or execute_live) else MissingWriter()
                with Journal(args.journal) as j:
                    return ManualConsole(
                        p,
                        backend,
                        writer,
                        j,
                        execute=execute_live,
                        test_receipt=args.receipt,
                        overrides=overrides,
                        strict_confirm=execute_live,
                    ).run()

            check_host_readonly_safe(backend.host())
            if args.action == "status":
                for line in format_status_report(p, backend):
                    print(line)
                with Journal(args.journal) as j:
                    j.append(event="manual_status", pci_id=backend.pci_id())
                return 0

            _warn_if_unhealthy_readonly(p, backend)
            if args.action == "read-row":
                value = backend.fuse_row(args.row)
                warn = "  [POISON!]" if is_poison(value) else ""
                print(f"row {args.row} = {hex32(value)}{warn}")
                with Journal(args.journal) as j:
                    j.append(event="manual_read_row", row=args.row, value=hex32(value))
            elif args.action == "read-rows":
                if args.end < args.start or (args.end - args.start) >= 128:
                    raise BurnError("invalid row range (must satisfy start <= end and <= 128 rows)")
                values = {}
                for row_idx in range(args.start, args.end + 1):
                    val = backend.fuse_row(row_idx)
                    values[str(row_idx)] = hex32(val)
                    warn = "  [POISON!]" if is_poison(val) else ""
                    print(f"  row {row_idx:4d}: {hex32(val)}{warn}")
                with Journal(args.journal) as j:
                    j.append(event="manual_read_rows", start=args.start, end=args.end, values=values)
            elif args.action == "read-reg":
                addr = resolve_register_address(args.address)
                value = backend.register(addr)
                warn = "  [POISON!]" if is_poison(value) else ""
                print(f"reg {hex32(addr)} = {hex32(value)}{warn}")
                with Journal(args.journal) as j:
                    j.append(event="manual_read_reg", address=hex32(addr), value=hex32(value))
            elif args.action == "read-field":
                value = backend.field(args.name)
                warn = "  [POISON!]" if is_poison(value) else ""
                print(f"{args.name} = {hex32(value)}{warn}")
                with Journal(args.journal) as j:
                    j.append(event="manual_read_field", field=args.name, value=hex32(value))
            return 0

        if args.cmd == "verify-cold":
            cold_path = require_private_file(args.cold, "cold snapshot")
            try:
                cold_data = json.loads(cold_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise BurnError(f"invalid JSON in cold snapshot {cold_path}: {exc}") from exc
            with Journal(args.journal) as j:
                receipt = verify_cold_snapshot(p, cold_data, stage=args.stage, journal=j)
            if args.receipt_out:
                write_private_json(args.receipt_out, receipt)
            print(
                "PASS: supplied cold snapshot matches declared expectations; "
                "evidence is not independently authenticated"
            )
            return 0

    except KeyboardInterrupt:
        print("\nABORTED by operator (Ctrl-C).", file=sys.stderr)
        return 130
    except (BurnError, ValueError, KeyError, NotImplementedError, OSError, json.JSONDecodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    finally:
        if backend is not None and callable(getattr(backend, "close", None)):
            backend.close()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
