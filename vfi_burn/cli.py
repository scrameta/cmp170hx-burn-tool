"""Standalone CLI argument parser and dispatcher for vfi_burn.

Responsibilities of this module are strictly limited to:
  1. Command-line grammar definition via ``argparse`` (``build_parser``)
  2. Mapping parsed ``argparse.Namespace`` objects to typed requests (``dispatch_args``)
  3. Top-level exit-code and error handling (``main``)

All command execution, backend lifecycle, and hardware/simulation logic lives in
``vfi_burn.commands``.
"""
from __future__ import annotations

import argparse
import json
import sys

from .commands import (
    CommandIO,
    TargetSpec,
    default_prompt_confirm as prompt_confirm,
    execute_burn,
    execute_decode,
    execute_import_ffcalc,
    execute_import_ga100_ff_tools,
    execute_inspect,
    execute_manual,
    execute_menu,
    execute_plan_review,
    execute_simulate,
    execute_snapshot,
    execute_verify_cold,
)
from .engine import Overrides
from .menu import MenuSession
from .model import BurnError


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
        "--override-undetermined-ceiling",
        action="store_true",
        help="allow operator judgment when SM count is undetermined/above ceiling (genuine defects remain REFUSED)",
    )
    grp.add_argument(
        "--override-baseline-mismatch",
        action="store_true",
        help="allow proceeding when live FF prefix differs from (or plan lacks) baseline_ff_rows",
    )
    grp.add_argument(
        "--override-unmapped-ff",
        action="store_true",
        help="allow unknown FF blocks, changed unmapped/opt-in bits, or fields missing defect masks (known defects & device-ID blocks remain REFUSED)",
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
        undetermined_ceiling=bool(getattr(args, "override_undetermined_ceiling", False)),
        allow_manual_debugctrl=bool(getattr(args, "override_allow_debugctrl", False)),
        baseline_mismatch=bool(getattr(args, "override_baseline_mismatch", False)),
        unmapped_ff=bool(getattr(args, "override_unmapped_ff", False)),
    )


def _target_from_args(args: argparse.Namespace) -> TargetSpec:
    return TargetSpec(
        bdf=getattr(args, "bdf", None),
        adapter=getattr(args, "adapter", None),
        fake=getattr(args, "fake", None),
        sim_pcie=getattr(args, "sim_pcie", None),
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vfi_burn",
        description=(
            "CMP 170HX 8 GB FF fuse-burn safety controller "
            "(run with no subcommand or 'menu' for the interactive menu; passive dry-run by default)"
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=False)

    def add_plan(p: argparse.ArgumentParser) -> None:
        p.add_argument("--plan", required=True, help="private mode-0600 per-card plan JSON")

    menu_p = sub.add_parser(
        "menu",
        help="high-level interactive menu interface (also launched when vfi_burn is run with no arguments)",
    )
    menu_p.add_argument("--plan", default=None, help="optional pre-loaded mode-0600 per-card plan JSON")
    menu_g = menu_p.add_mutually_exclusive_group(required=False)
    menu_g.add_argument("--bdf", default=None, help="live CMP 170HX PCI BDF (e.g. 0000:05:00.0)")
    menu_g.add_argument("--adapter", default=None, help="custom owner-supplied live module:factory")
    menu_g.add_argument("--fake", default=None, help="synthetic mode-0600 JSON controller snapshot")
    menu_p.add_argument(
        "--journal",
        default="private/session.jsonl",
        help="private mode-0600 audit JSONL (default: private/session.jsonl)",
    )
    menu_p.add_argument("--receipt", default=None, help="optional cold-verified test receipt path")
    menu_p.add_argument(
        "--execute",
        action="store_true",
        help="start with Live Execute mode armed (still requires TTY + per-record confirmations)",
    )
    _add_override_args(menu_p, allow_debugctrl=True)

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
    imp_p.add_argument(
        "--base-sm",
        type=int,
        required=True,
        help="REQUIRED operator-supplied non-defective SM ceiling for this SKU (e.g. 122); "
        "the card cannot tell you this",
    )
    imp_p.add_argument("--override-sm", type=int, default=None)
    imp_p.add_argument(
        "--link-cap2-before",
        type=lambda s: int(s, 0),
        default=0x06,
        help="reviewed current PCIe LnkCap2 (read it with 'manual read-field LnkCap2'; default 0x06)",
    )
    imp_p.add_argument(
        "--link-cap2-after",
        type=lambda s: int(s, 0),
        default=0x1E,
        help="expected LnkCap2 after PCIe records burn (default 0x1E); ignored if no PCIe records",
    )
    imp_p.add_argument(
        "--prepend-noop-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prepend a proved zero-effect no-op FF test row (default: enabled; disabling requires burn[0] to already be a no-op)",
    )

    imp_ga_p = sub.add_parser(
        "import-ga100-ff-tools",
        help="convert one or more ga100-ff-tools plan JSONs + ff_dump.py + read_defects.py into a mode-0600 vfi_burn plan",
    )
    imp_ga_p.add_argument(
        "--plans",
        nargs="+",
        required=True,
        help="one or more mode-0600 ga100-ff-tools plan JSONs from one ffcard.py derive run "
        "(e.g. *_plan_0_noop.json *_plan_sm_hbm.json [*_plan_pcie.json *_plan_nvlink.json])",
    )
    imp_ga_p.add_argument(
        "--dump",
        required=True,
        help="mode-0600 ga100-ff-tools ff_dump.py JSON (covering rows 222..486+)",
    )
    imp_ga_p.add_argument(
        "--defects",
        required=True,
        help="mode-0600 ga100-ff-tools read_defects.py JSON",
    )
    imp_ga_p.add_argument("--out", required=True, help="output path for new mode-0600 vfi_burn plan JSON")
    imp_ga_p.add_argument(
        "--base-sm",
        type=int,
        required=True,
        help="REQUIRED operator-supplied non-defective SM ceiling for this SKU (e.g. 122); "
        "the card cannot tell you this",
    )
    imp_ga_p.add_argument("--override-sm", type=int, default=None)
    imp_ga_p.add_argument(
        "--die-rows-from",
        default=None,
        help="mode-0600 JSON supplying Die ID rows 8..15 (vfi_burn snapshot, ff_dump.py --start 8, "
        "ffcalc dump, or plain {\"8\":...,\"15\":...}); unnecessary if --dump already includes rows 8..15",
    )
    imp_ga_p.add_argument(
        "--bdf",
        default=None,
        help="read Die ID rows 8..15 read-only from the live CMP 170HX GPU (e.g. 0000:05:00.0)",
    )
    imp_ga_p.add_argument(
        "--link-cap2-before",
        type=lambda s: int(s, 0),
        default=0x06,
        help="reviewed current PCIe LnkCap2 (default 0x06)",
    )
    imp_ga_p.add_argument(
        "--link-cap2-after",
        type=lambda s: int(s, 0),
        default=0x1E,
        help="expected LnkCap2 after PCIe records burn (default 0x1E); ignored if no PCIe records",
    )
    imp_ga_p.add_argument(
        "--prepend-noop-test",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="auto by default: reuse included 0_noop test record if present, else prepend a proved no-op test row "
        "(--no-prepend-noop-test refuses unless the first record is already a verified no-op)",
    )

    snap_p = sub.add_parser(
        "snapshot",
        help="capture a mode-0600 cold-boot snapshot from a live CMP 170HX for verify-cold",
    )
    snap_p.add_argument("--bdf", required=True, help="PCI BDF of CMP 170HX (e.g. 0000:05:00.0)")
    snap_p.add_argument("--previous-boot-id", required=True, help="boot_id from the burn session journal")
    snap_p.add_argument("--out", required=True, help="output path for new mode-0600 cold snapshot JSON")
    snap_p.add_argument("--no-rows", action="store_true", help="skip reading FF rows 222..486 (options + die ID only)")

    inspect_p = sub.add_parser("inspect", help="read-only preflight, health report, and free-slot proof")
    add_plan(inspect_p)
    ig = inspect_p.add_mutually_exclusive_group(required=True)
    ig.add_argument("--fake", help="synthetic JSON controller snapshot")
    ig.add_argument("--sim-pcie", help="mode-0600 JSON snapshot run through simulated PCIe BAR0 + GA100 HAL")
    ig.add_argument("--bdf", help="live CMP 170HX PCI BDF (e.g. 0000:05:00.0)")
    ig.add_argument("--adapter", help="custom owner-supplied live module:factory")
    inspect_p.add_argument(
        "--sense",
        action="store_true",
        help="explicitly invoke SENSE_CTRL active sense (confirmed with 'SENSE' on live hardware)",
    )
    inspect_p.add_argument(
        "--experimental",
        action="store_true",
        help="also print the full read-only experimental GA100 register table (FUSE_ECC_EN, FUSE_SS_*, SKU, readouts)",
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
    burn_p.add_argument(
        "--stage",
        required=True,
        choices=("test", "real", "all"),
        help="'test' burns only the no-op test row; 'real' burns real records (requires test receipt unless overridden); 'all' runs no-op test then lets you stop or continue through functional groups (runs pre-power-off sanity check whenever stopped/finished)",
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

    sim_p = sub.add_parser("simulate", help="exercise the exact write loop on FAKE or simulated PCIe data only")
    add_plan(sim_p)
    sg = sim_p.add_mutually_exclusive_group(required=True)
    sg.add_argument("--fake", help="synthetic mode-0600 JSON snapshot (lightweight fake controller)")
    sg.add_argument("--sim-pcie", help="mode-0600 JSON snapshot executed through simulated PCIe BAR0 + GA100 HAL")
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
    mg.add_argument("--sim-pcie", help="mode-0600 JSON snapshot executed through simulated PCIe BAR0 + GA100 HAL")
    mg.add_argument("--bdf", help="live CMP 170HX PCI BDF (e.g. 0000:05:00.0)")
    mg.add_argument("--adapter", help="custom owner-supplied live module:factory")
    man_sub = manual_p.add_subparsers(dest="action", required=True)

    man_sub.add_parser("status", help="print full health, host, die ID, and mask status (never writes)")
    man_sub.add_parser(
        "experimental",
        help="print full read-only experimental GA100 register table (FUSE_ECC_EN, FUSE_SS_*, SKU, readouts)",
    )

    rr = man_sub.add_parser("read-row", help="read a single fuse row (works even if card is faulted/cold)")
    rr.add_argument("row", type=lambda s: int(s, 0))

    rrs = man_sub.add_parser("read-rows", help="read a range of fuse rows (start..end inclusive)")
    rrs.add_argument("start", type=lambda s: int(s, 0))
    rrs.add_argument("end", type=lambda s: int(s, 0))

    rg = man_sub.add_parser(
        "read-reg",
        help="read aligned 32-bit MMIO reg (hex, FUSECTRL/DEBUGCTRL/PLM_F8, or experimental name like FUSE_ECC_EN)",
    )
    rg.add_argument("address")

    rf = man_sub.add_parser("read-field", help="read named decoded field (including experimental names)")
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


def dispatch_args(args: argparse.Namespace, *, io: CommandIO | None = None) -> int:
    """Translate parsed CLI arguments into decoupled command execution calls."""
    io = io or CommandIO()
    cmd = args.cmd

    if cmd is None:
        return execute_menu(MenuSession())

    if cmd == "menu":
        return execute_menu(
            MenuSession(
                plan_path=args.plan,
                bdf=args.bdf,
                adapter_spec=args.adapter,
                fake_path=args.fake,
                journal_path=args.journal,
                receipt_path=args.receipt,
                execute_live=bool(args.execute),
                overrides=_overrides_from_args(args),
            )
        )

    if cmd == "decode":
        return execute_decode(args.word, io=io)

    if cmd == "import-ffcalc":
        return execute_import_ffcalc(
            burn_plan=args.burn_plan,
            dump=args.dump,
            out=args.out,
            base_sm=args.base_sm,
            override_sm=args.override_sm,
            prepend_noop_test=args.prepend_noop_test,
            link_cap2_before=args.link_cap2_before,
            link_cap2_after=args.link_cap2_after,
            io=io,
        )

    if cmd == "import-ga100-ff-tools":
        return execute_import_ga100_ff_tools(
            plans=args.plans,
            dump=args.dump,
            defects=args.defects,
            out=args.out,
            base_sm=args.base_sm,
            override_sm=args.override_sm,
            die_rows_from=args.die_rows_from,
            bdf=args.bdf,
            prepend_noop_test=args.prepend_noop_test,
            link_cap2_before=args.link_cap2_before,
            link_cap2_after=args.link_cap2_after,
            io=io,
        )

    if cmd == "snapshot":
        return execute_snapshot(
            bdf=args.bdf,
            previous_boot_id=args.previous_boot_id,
            out=args.out,
            read_rows=not args.no_rows,
            io=io,
        )

    if cmd == "plan":
        return execute_plan_review(
            plan_path=args.plan,
            target=_target_from_args(args),
            io=io,
        )

    if cmd == "inspect":
        return execute_inspect(
            plan_path=args.plan,
            target=_target_from_args(args),
            sense=bool(args.sense),
            experimental=bool(args.experimental),
            journal_path=args.journal,
            io=io,
        )

    if cmd == "burn":
        return execute_burn(
            plan_path=args.plan,
            target=_target_from_args(args),
            stage=args.stage,
            journal_path=args.journal,
            receipt=args.receipt,
            max_records=args.max_records,
            sense=bool(args.sense),
            execute=bool(args.execute),
            overrides=_overrides_from_args(args),
            io=io,
        )

    if cmd == "simulate":
        return execute_simulate(
            plan_path=args.plan,
            target=_target_from_args(args),
            stage=args.stage,
            journal_path=args.journal,
            receipt=args.receipt,
            max_records=args.max_records,
            dry_run=bool(args.dry_run),
            save_fake=args.save_fake,
            overrides=_overrides_from_args(args),
            io=io,
        )

    if cmd == "manual":
        return execute_manual(
            plan_path=args.plan,
            target=_target_from_args(args),
            action=args.action,
            journal_path=args.journal,
            row=getattr(args, "row", None),
            start=getattr(args, "start", None),
            end=getattr(args, "end", None),
            address=getattr(args, "address", None),
            field_name=getattr(args, "name", None),
            record_file=getattr(args, "record", None),
            receipt=getattr(args, "receipt", None),
            execute=bool(getattr(args, "execute", False)),
            dry_run=bool(getattr(args, "dry_run", False)),
            save_fake=getattr(args, "save_fake", None),
            overrides=_overrides_from_args(args),
            io=io,
        )

    if cmd == "verify-cold":
        return execute_verify_cold(
            plan_path=args.plan,
            cold_path=args.cold,
            stage=args.stage,
            journal_path=args.journal,
            receipt_out=args.receipt_out,
            io=io,
        )

    return 2


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return dispatch_args(args)
    except KeyboardInterrupt:
        print("\nABORTED by operator (Ctrl-C).", file=sys.stderr)
        return 130
    except (BurnError, ValueError, KeyError, NotImplementedError, OSError, json.JSONDecodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
