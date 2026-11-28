"""Command execution layer decoupled from CLI argument parsing (argparse).

Each ``execute_*`` function accepts explicit typed parameters and an injectable
``CommandIO`` object, performing backend lifecycle management, safety checks,
journaling, and execution without touching ``argparse``.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable

from .backend import (
    MissingWriter,
    fake_from_file,
    has_verified_writer,
    load_adapter,
    snapshot_data,
    write_private_json,
    write_private_json_interactive,
)
from .console import (
    ManualConsole,
    format_experimental_report,
    format_plan_review,
    format_status_report,
    read_experimental_registers,
)
from .engine import (
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
from .ffrecord import decode_ff, fields_in_block
from .fuses import (
    EXPERIMENTAL_BY_ADDR,
    EXPERIMENTAL_BY_NAME,
    decode_experimental_register,
)
from .ga100 import GA100LiveAdapter, ensure_gpu_detached_interactive, prompt_select_gpu_bdf
from .journal import Journal
from .menu import BurnMenu, MenuSession
from .menuconfig import run_menuconfig
from .mmio import resolve_register_address
from .model import BurnError, Plan, Record, is_poison, require_private_file, sku_display_name, u32
from .planner import UnlockConfig, build_plan_from_config, capture_card_snapshot
from .sim_pcie import SimulatedGA100Adapter, sim_pcie_from_file


def default_prompt_confirm(prompt: str, token: str) -> bool:
    try:
        return input(prompt).strip() == token
    except (EOFError, KeyboardInterrupt):
        return False


@dataclass(frozen=True)
class TargetSpec:
    """Declarative specification of the target backend (live BDF, adapter, fake, or sim-pcie)."""

    bdf: str | None = None
    adapter: str | None = None
    fake: str | None = None
    sim_pcie: str | None = None

    @property
    def is_simulated(self) -> bool:
        return bool(self.fake or self.sim_pcie)

    @property
    def has_target(self) -> bool:
        return bool(self.bdf or self.adapter or self.fake or self.sim_pcie)


@dataclass
class CommandIO:
    """Injectable I/O callbacks for command execution (stdout, stderr, confirmation, input, TTY)."""

    show: Callable[[str], None] = print
    err: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr)
    confirm: Callable[[str, str], bool] = default_prompt_confirm
    is_tty: Callable[[], bool] = lambda: sys.stdin.isatty()
    input_fn: Callable[[str], str] = input


@contextmanager
def open_target(spec: TargetSpec, *, writable: bool = False):
    """Open the requested target backend and guarantee cleanup on exit."""
    if spec.sim_pcie:
        backend = sim_pcie_from_file(spec.sim_pcie)
    elif spec.fake:
        backend = fake_from_file(spec.fake)
    elif spec.bdf:
        backend = GA100LiveAdapter(spec.bdf, writable=writable)
    elif spec.adapter:
        backend = load_adapter(spec.adapter)
    else:
        raise BurnError("no target specified (expected --bdf, --adapter, --fake, or --sim-pcie)")
    try:
        yield backend
    finally:
        if callable(getattr(backend, "close", None)):
            backend.close()


def _save_sim_snapshot(backend, path: str, *, overwrite: bool = False) -> None:
    if isinstance(backend, SimulatedGA100Adapter):
        write_private_json(path, backend.to_snapshot_dict(), overwrite=overwrite)
    else:
        write_private_json(path, snapshot_data(backend), overwrite=overwrite)


def _require_live_execute_ready(backend, io: CommandIO) -> None:
    if not io.is_tty():
        raise BurnError(
            "REFUSED --execute without an interactive TTY; unattended/piped burns are forbidden"
        )
    if not has_verified_writer(backend):
        raise BurnError(
            "REFUSED --execute: supplied adapter does not have write capability armed "
            "(must define write_one_record(row, word) AND set write_enabled = True or legacy writer_verified = True)"
        )


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


def _warn_if_unhealthy_readonly(plan: Plan, backend, io: CommandIO) -> None:
    try:
        issues = evaluate_health(backend.health(), plan)
    except Exception as exc:
        io.err(f"NOTE: health read failed ({exc}); proceeding with read-only diagnostic")
        return
    if issues:
        io.err("NOTE: card is NOT in a write-ready goldilocks state:")
        for issue in issues:
            io.err(f"  - {issue}")


def _print_plan_summary(p: Plan, final_masks: dict[str, int], io: CommandIO) -> None:
    io.show(f"OFFLINE PLAN REVIEW | SHA256: {p.content_hash}")
    io.show(f"  SKU          : {p.pci_id} ({sku_display_name(p.pci_id)})")
    io.show(
        f"  FF Window    : rows {p.first_row}..{p.last_row} "
        f"({p.last_row - p.first_row + 1} slots) | empty_word={hex32(p.empty_word)}"
    )
    io.show(
        f"  SM Ceilings  : base={p.base_sm_ceiling} | override={p.override_sm_ceiling} | "
        f"reconfig_verified={p.reconfig_semantics_verified}"
    )
    io.show(
        "  Die ID (8-15): "
        + " ".join(f"{k}:{hex32(v)}" for k, v in sorted(p.die_rows.items()))
    )
    io.show(
        "  Defect Masks : "
        + (", ".join(f"{k}={hex32(v)}" for k, v in sorted(p.defect_masks.items())) or "none")
    )
    io.show(
        "  Mask Plan    : "
        + ", ".join(
            f"{k}: {hex32(p.expected_masks[k])} -> {hex32(final_masks[k])}"
            for k in sorted(p.expected_masks)
        )
    )
    io.show(f"  Records ({len(p.records)}):")
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
        io.show(
            f"    {i:2d}: {r.kind:8} {r.name:22} target={r.target:18} "
            f"word={hex32(r.word)}{ff_tag}{extra} | {r.decode}"
        )
    io.show("OFFLINE ONLY: no hardware contacted.")


# ---------------------------------------------------------------------------
# Command executors (each has zero argparse dependency)
# ---------------------------------------------------------------------------


def execute_menu(session: MenuSession | None = None) -> int:
    return BurnMenu(session or MenuSession()).run()


def execute_decode(word_raw: str, *, io: CommandIO | None = None) -> int:
    io = io or CommandIO()
    word = u32(word_raw, "word")
    ff = decode_ff(word)
    io.show(f"FF word {hex32(word)}: chain={ff.chain} block={ff.block} data=0x{ff.data:04x} (IFF bit5=0)")
    overlapping = fields_in_block(ff.chain, ff.block)
    if overlapping:
        io.show("Overlapping GA100 option fields:")
        for f in overlapping:
            io.show(f"  - {f.key:24} ({f.name}) bits {f.bit}..{f.bit + f.width - 1} [{f.category}]")
    else:
        io.show("Overlapping GA100 option fields: none in known map")
    return 0


def _check_generated_plan(loaded: Plan, io: CommandIO) -> dict[str, int]:
    final_masks = check_plan(loaded, allow_unmapped_ff=True)
    try:
        check_plan(loaded, allow_unmapped_ff=False)
    except BurnError as exc:
        io.show(
            f"NOTE: plan contains opt-in/override-gated record(s) ({exc}); "
            "burning will require --override-unmapped-ff (or menu [7] -> [8])."
        )
    return final_masks


def _describe_existing_plan(path: Path, card) -> str | None:
    try:
        existing = Plan.load(path)
    except Exception:
        return None
    if existing.pci_id != card.pci_id or existing.die_rows != card.die_rows:
        return (
            f"WARNING: existing file {path} is a plan for a DIFFERENT card "
            f"({existing.pci_id} {sku_display_name(existing.pci_id)}, "
            f"sha={existing.content_hash[:12]})."
        )
    return (
        f"Existing file {path} is an earlier plan for this same card "
        f"({existing.pci_id} {sku_display_name(existing.pci_id)}, "
        f"{len(existing.records)} records, sha={existing.content_hash[:12]})."
    )


def execute_configure(
    *,
    target: TargetSpec,
    out: str,
    unlock_sm_hbm: bool = False,
    pcie_mode: str | None = None,
    nvlink: bool | None = None,
    a100_devid: int | None = None,
    enable_mig_sys_pipe: bool = False,
    use_curses: bool | None = None,
    overwrite: bool = False,
    io: CommandIO | None = None,
    sysfs_root: str = "/sys/bus/pci/devices",
) -> int:
    """Capture live/simulated card state read-only and build a mode-0600 vfi_burn plan
    interactively (kernel menuconfig) or non-interactively via explicit unlock flags."""
    io = io or CommandIO()
    if not target.has_target:
        picked = prompt_select_gpu_bdf(
            current_bdf=None,
            input_fn=io.input_fn,
            show=io.show,
            sysfs_root=sysfs_root,
        )
        if not picked:
            io.show("No GPU selected; cancelled.")
            return 1
        target = replace(target, bdf=picked)

    if target.bdf and not target.is_simulated:
        if not ensure_gpu_detached_interactive(
            target.bdf,
            input_fn=io.input_fn,
            show=io.show,
            sysfs_root=sysfs_root,
        ):
            io.show("Cancelled — nvidia/nouveau must be unloaded before reading BAR0.")
            return 1

    with open_target(target, writable=False) as backend:
        check_host_readonly_safe(backend.host())
        label = target.bdf or target.adapter or target.sim_pcie or target.fake or "GPU"
        card = capture_card_snapshot(backend, label=label)

    non_interactive = (
        unlock_sm_hbm
        or (pcie_mode is not None)
        or (nvlink is not None)
        or (a100_devid is not None)
        or enable_mig_sys_pipe
    )
    if non_interactive:
        cfg = UnlockConfig.from_card(card)
        if unlock_sm_hbm:
            cfg.unlock_all_healthy_sm_hbm()
        if pcie_mode is not None:
            cfg.set_pcie_mode(pcie_mode)
        if nvlink is not None:
            cfg.set_nvlink(nvlink)
        if a100_devid is not None:
            cfg.set_a100_devid(a100_devid)
        if enable_mig_sys_pipe:
            cfg.set_sys_pipe_unlock(True)
    else:
        custom_io = (io.input_fn is not input) or (io.show is not print)
        cfg = run_menuconfig(
            card,
            use_curses=False if (custom_io and use_curses is None) else use_curses,
            input_fn=io.input_fn if custom_io else None,
            show=io.show if custom_io else None,
        )
        if cfg is None:
            io.show("Unlock configuration cancelled; no plan written.")
            return 1

    plan_dict = build_plan_from_config(card, cfg)
    if overwrite:
        write_private_json(out, plan_dict, overwrite=True)
        saved = out
    elif not non_interactive:
        saved = write_private_json_interactive(
            out,
            plan_dict,
            input_fn=io.input_fn,
            show=io.show,
            path_prompt="Save mode-0600 plan JSON path",
            describe_existing=lambda p: _describe_existing_plan(p, card),
        )
        if not saved:
            io.show("Plan not saved; cancelled.")
            return 1
    else:
        write_private_json(out, plan_dict, overwrite=False)
        saved = out
    loaded = Plan.load(saved)
    final_masks = _check_generated_plan(loaded, io)
    io.show(f"Generated plan {saved} (mode 0600) | {cfg.summary_line()}:")
    _print_plan_summary(loaded, final_masks, io)
    if target.bdf:
        io.show(f"Next: sudo python3 -m vfi_burn menu --bdf {target.bdf} --plan {saved}")
    elif target.fake:
        io.show(f"Next: python3 -m vfi_burn menu --fake {target.fake} --plan {saved}")
    return 0


def execute_snapshot(
    *,
    bdf: str,
    previous_boot_id: str,
    out: str,
    read_rows: bool = True,
    overwrite: bool = False,
    io: CommandIO | None = None,
) -> int:
    io = io or CommandIO()
    with GA100LiveAdapter(bdf, writable=False) as backend:
        check_host_readonly_safe(backend.host())
        snap = backend.capture_cold_snapshot(
            previous_boot_id=previous_boot_id,
            read_rows=read_rows,
        )
    write_private_json(out, snap, overwrite=overwrite)
    io.show(
        f"Saved cold snapshot {out} (mode 0600) | boot_id={snap['boot_id']} "
        f"rows_nonzero={len(snap['fpf_rows'])} SM_COUNT={snap['fields']['SM_COUNT']}"
    )
    return 0


def execute_plan_review(
    *,
    plan_path: str,
    target: TargetSpec | None = None,
    io: CommandIO | None = None,
) -> int:
    io = io or CommandIO()
    p = Plan.load(plan_path)
    check_plan(p, allow_unmapped_ff=True)
    if target is not None and target.has_target:
        with open_target(target, writable=False) as backend:
            for line in format_plan_review(p, reader=backend):
                io.show(line)
        return 0
    for line in format_plan_review(p, reader=None):
        io.show(line)
    io.show("OFFLINE PLAN REVIEW — OFFLINE ONLY: no hardware contacted.")
    return 0


def execute_inspect(
    *,
    plan_path: str,
    target: TargetSpec,
    sense: bool = False,
    experimental: bool = False,
    journal_path: str | None = None,
    io: CommandIO | None = None,
) -> int:
    io = io or CommandIO()
    p = Plan.load(plan_path)
    check_plan(p, allow_unmapped_ff=True)
    with open_target(target, writable=False) as backend:
        for line in format_status_report(p, backend):
            io.show(line)
        exp_readings: dict[str, int] | None = None
        if experimental:
            exp_readings, exp_poisoned = read_experimental_registers(backend)
            for line in format_experimental_report(
                backend, readings=exp_readings, poisoned=exp_poisoned
            ):
                io.show(line)
        journal = Journal(journal_path) if journal_path else None
        try:
            if sense and not target.is_simulated:
                io.show(
                    "ACTIVE SENSE: active_sense_readonly() (SENSE_CTRL) does not burn fuses, "
                    "but can collapse a too-deep held goldilocks frame (forcing power cycle + re-hunt)."
                )
                if not io.confirm("Type 'SENSE' to run active_sense_readonly(): ", "SENSE"):
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
            boot = preflight(p, backend, active_sense=sense, show=io.show)
            row, surrounding = scan_slots(p, backend)
            if journal is not None:
                h_snap, host_snap = _capture_inspect_snapshot(backend)
                extra_j = (
                    {"experimental_registers": {k: hex32(v) for k, v in exp_readings.items()}}
                    if exp_readings is not None
                    else {}
                )
                journal.append(
                    event="inspect_result",
                    plan_sha256=p.content_hash,
                    boot_id=boot,
                    active_sense=sense,
                    proved_pointer=row,
                    health=h_snap,
                    host=host_snap,
                    status="PASS",
                    **extra_j,
                )
        except Exception as exc:
            if journal is not None:
                h_snap, host_snap = _capture_inspect_snapshot(backend)
                journal.append(
                    event="inspect_failed",
                    plan_sha256=p.content_hash,
                    active_sense=sense,
                    health=h_snap,
                    host=host_snap,
                    status="FAIL",
                    error=str(exc),
                )
            raise
        finally:
            if journal is not None:
                journal.close()
        tag_mode = "FAKE" if target.is_simulated else "LIVE READ-ONLY"
        io.show(f"{tag_mode} | boot {boot}; first free row {row}")
        for i, v in surrounding:
            tag = "EMPTY" if v == p.empty_word else "WRITTEN"
            io.show(f"  row {i:4d}: {hex32(v)}  [{tag}]")
        io.show(
            "FULL write-capable preflight NOT established (re-run with --sense to test active sense)"
            if not sense
            else "Active-sense checks passed; frame is write-capable (no write issued)"
        )
    return 0


def execute_burn(
    *,
    plan_path: str,
    target: TargetSpec,
    stage: str,
    journal_path: str,
    receipt: str | None = None,
    max_records: int | None = None,
    sense: bool = False,
    execute: bool = False,
    overrides: Overrides | None = None,
    io: CommandIO | None = None,
) -> int:
    io = io or CommandIO()
    p = Plan.load(plan_path)
    ov = overrides or Overrides()
    check_plan(
        p,
        allow_undetermined_ceiling=ov.undetermined_ceiling,
        allow_unmapped_ff=ov.unmapped_ff,
    )
    dry_run = not execute
    with open_target(target, writable=execute) as backend:
        if not dry_run:
            _require_live_execute_ready(backend, io)
        run_sense = (not dry_run) or bool(sense)
        writer = backend if not dry_run else MissingWriter()
        with Journal(journal_path) as j:
            process_records(
                p,
                backend,
                writer,
                j,
                stage=stage,
                confirm=io.confirm,
                test_receipt=receipt,
                show=io.show,
                overrides=ov,
                max_records=max_records,
                dry_run=dry_run,
                strict_confirm=not dry_run,
                confirm_active_sense=True,
                run_active_sense=run_sense,
            )
    if dry_run:
        sense_note = (
            "active-sense & slot proof passed"
            if run_sense and not ov.skip_active_sense
            else "passive preflight & slot proof passed (add --sense to test active sense)"
        )
        io.show(f"DRY-RUN COMPLETE: {sense_note}; no fuses written. Pass --execute to burn.")
    else:
        io.show("READ-BACK VERIFIED (NOT cold-boot verified). Cold boot and run verify-cold.")
    return 0


def execute_simulate(
    *,
    plan_path: str,
    target: TargetSpec,
    stage: str,
    journal_path: str,
    receipt: str | None = None,
    max_records: int | None = None,
    dry_run: bool = False,
    save_fake: str | None = None,
    overwrite: bool = False,
    overrides: Overrides | None = None,
    io: CommandIO | None = None,
) -> int:
    io = io or CommandIO()
    if not target.is_simulated:
        raise BurnError("simulate requires --fake or --sim-pcie")
    p = Plan.load(plan_path)
    ov = overrides or Overrides()
    check_plan(
        p,
        allow_undetermined_ceiling=ov.undetermined_ceiling,
        allow_unmapped_ff=ov.unmapped_ff,
    )
    with open_target(target, writable=False) as backend:
        with Journal(journal_path) as j:
            process_records(
                p,
                backend,
                backend,
                j,
                stage=stage,
                confirm=io.confirm,
                test_receipt=receipt,
                show=io.show,
                overrides=ov,
                max_records=max_records,
                dry_run=dry_run,
            )
        if save_fake and not dry_run:
            _save_sim_snapshot(backend, save_fake, overwrite=overwrite)
    io.show(
        "SIMULATION DRY-RUN ONLY: no fake rows modified."
        if dry_run
        else "SIMULATION ONLY: fake records updated; no GPU contacted."
    )
    return 0


def _execute_manual_write(
    *,
    plan: Plan,
    target: TargetSpec,
    backend,
    journal_path: str,
    record_file: str,
    receipt: str | None,
    execute: bool,
    dry_run: bool,
    save_fake: str | None,
    overwrite: bool = False,
    overrides: Overrides,
    allow_live: bool,
    io: CommandIO,
) -> int:
    is_sim = target.is_simulated
    if not is_sim and not allow_live:
        raise BurnError("manual physical writes require 'manual write-record --execute' with a verified writer")

    record_path = require_private_file(record_file, "manual record file")
    try:
        record = Record.parse(json.loads(record_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise BurnError(f"invalid JSON in manual record {record_path}: {exc}") from exc
    if record.kind == "test":
        raise BurnError("manual write-record cannot insert a 'test' record; use stage=test")

    ov = overrides if allow_live else replace(overrides, allow_manual_debugctrl=True)
    execute_live = (not is_sim) and bool(execute)
    effective_dry = bool(dry_run) or ((not is_sim) and (not execute_live))
    if execute_live and not effective_dry:
        _require_live_execute_ready(backend, io)

    writer = backend if (is_sim or execute_live) else MissingWriter()
    rec_hash = hashlib.sha256(record_path.read_bytes()).hexdigest()
    single = replace(plan, records=(plan.records[0], record))
    with Journal(journal_path) as j:
        j.append(
            event="manual_selected_record",
            source_file_sha256=rec_hash,
            record=record.name,
            target=record.target,
            word=hex32(record.word),
            original_card_plan_sha256=plan.content_hash,
            dry_run=effective_dry,
            note="manual record is not part of the original approved list",
        )
        process_records(
            single,
            backend,
            writer,
            j,
            stage="real",
            confirm=io.confirm,
            test_receipt=receipt,
            show=io.show,
            overrides=ov,
            dry_run=effective_dry,
            strict_confirm=execute_live,
            confirm_active_sense=execute_live,
            run_active_sense=not effective_dry,
            manual_record_sha256=rec_hash,
            is_manual_single=True,
        )
    if is_sim and save_fake and not effective_dry:
        _save_sim_snapshot(backend, save_fake, overwrite=overwrite)
    if effective_dry:
        io.show("DRY-RUN COMPLETE: no fuses written.")
    elif is_sim:
        io.show("SIMULATION ONLY: fake record updated; no GPU contacted.")
    else:
        io.show("READ-BACK VERIFIED (NOT cold-boot verified). Cold boot and verify before trusting unlock.")
    return 0


def execute_manual(
    *,
    plan_path: str,
    target: TargetSpec,
    action: str,
    journal_path: str,
    row: int | None = None,
    start: int | None = None,
    end: int | None = None,
    address: str | None = None,
    field_name: str | None = None,
    record_file: str | None = None,
    receipt: str | None = None,
    execute: bool = False,
    dry_run: bool = False,
    save_fake: str | None = None,
    overwrite: bool = False,
    overrides: Overrides | None = None,
    io: CommandIO | None = None,
) -> int:
    io = io or CommandIO()
    p = Plan.load(plan_path)
    ov = overrides or Overrides()
    check_plan(
        p,
        allow_manual_debugctrl=ov.allow_manual_debugctrl,
        allow_undetermined_ceiling=ov.undetermined_ceiling,
        allow_unmapped_ff=ov.unmapped_ff,
    )
    want_write = action in ("write-record", "console") and bool(execute) and not target.is_simulated

    with open_target(target, writable=want_write) as backend:
        if action == "simulate-write-record":
            return _execute_manual_write(
                plan=p,
                target=target,
                backend=backend,
                journal_path=journal_path,
                record_file=record_file or "",
                receipt=receipt,
                execute=False,
                dry_run=False,
                save_fake=save_fake,
                overwrite=overwrite,
                overrides=ov,
                allow_live=False,
                io=io,
            )
        if action == "write-record":
            return _execute_manual_write(
                plan=p,
                target=target,
                backend=backend,
                journal_path=journal_path,
                record_file=record_file or "",
                receipt=receipt,
                execute=execute,
                dry_run=dry_run,
                save_fake=save_fake,
                overwrite=overwrite,
                overrides=ov,
                allow_live=True,
                io=io,
            )
        if action == "console":
            is_sim = target.is_simulated
            execute_live = (not is_sim) and bool(execute)
            if execute_live:
                _require_live_execute_ready(backend, io)
            writer = backend if (is_sim or execute_live) else MissingWriter()
            with Journal(journal_path) as j:
                return ManualConsole(
                    p,
                    backend,
                    writer,
                    j,
                    execute=execute_live,
                    test_receipt=receipt,
                    overrides=ov,
                    strict_confirm=execute_live,
                    show=io.show,
                ).run()

        check_host_readonly_safe(backend.host())
        if action == "status":
            for line in format_status_report(p, backend):
                io.show(line)
            with Journal(journal_path) as j:
                j.append(event="manual_status", pci_id=backend.pci_id())
            return 0
        if action == "experimental":
            readings, poisoned = read_experimental_registers(backend)
            for line in format_experimental_report(
                backend, readings=readings, poisoned=poisoned
            ):
                io.show(line)
            with Journal(journal_path) as j:
                j.append(
                    event="manual_experimental_read",
                    pci_id=backend.pci_id(),
                    skipped_poison=poisoned,
                    registers={k: hex32(v) for k, v in readings.items()},
                )
            return 0

        _warn_if_unhealthy_readonly(p, backend, io)
        if action == "read-row":
            assert row is not None
            value = backend.fuse_row(row)
            warn = "  [0xBADx prefix: returned by the reader as OTP data]" if is_poison(value) else ""
            io.show(f"row {row} = {hex32(value)}{warn}")
            with Journal(journal_path) as j:
                j.append(event="manual_read_row", row=row, value=hex32(value))
        elif action == "read-rows":
            assert start is not None and end is not None
            if end < start or (end - start) >= 128:
                raise BurnError("invalid row range (must satisfy start <= end and <= 128 rows)")
            values = {}
            for row_idx in range(start, end + 1):
                val = backend.fuse_row(row_idx)
                values[str(row_idx)] = hex32(val)
                warn = "  [0xBADx prefix: returned by the reader as OTP data]" if is_poison(val) else ""
                io.show(f"  row {row_idx:4d}: {hex32(val)}{warn}")
            with Journal(journal_path) as j:
                j.append(event="manual_read_rows", start=start, end=end, values=values)
        elif action == "read-reg":
            assert address is not None
            addr = resolve_register_address(address)
            value = backend.register(addr)
            warn = "  [POISON!]" if is_poison(value) else ""
            exp = EXPERIMENTAL_BY_ADDR.get(addr)
            extra = (
                f"  [EXPERIMENTAL: {exp.name} — {decode_experimental_register(exp, value)}; "
                f"170HX={exp.cmp170hx_ref} / A100={exp.a100_ref}]"
                if exp is not None
                else ""
            )
            io.show(f"reg {hex32(addr)} = {hex32(value)}{warn}{extra}")
            with Journal(journal_path) as j:
                j.append(event="manual_read_reg", address=hex32(addr), value=hex32(value))
        elif action == "read-field":
            assert field_name is not None
            value = backend.field(field_name)
            warn = "  [POISON!]" if is_poison(value) else ""
            exp = EXPERIMENTAL_BY_NAME.get(field_name.strip().upper())
            extra = (
                f"  [EXPERIMENTAL: {decode_experimental_register(exp, value)}; "
                f"170HX={exp.cmp170hx_ref} / A100={exp.a100_ref}]"
                if exp is not None
                else ""
            )
            io.show(f"{field_name} = {hex32(value)}{warn}{extra}")
            with Journal(journal_path) as j:
                j.append(event="manual_read_field", field=field_name, value=hex32(value))
        else:
            raise BurnError(f"unknown manual action {action!r}")
    return 0


def execute_verify_cold(
    *,
    plan_path: str,
    cold_path: str,
    stage: str,
    journal_path: str,
    receipt_out: str | None = None,
    overwrite: bool = False,
    io: CommandIO | None = None,
) -> int:
    io = io or CommandIO()
    p = Plan.load(plan_path)
    check_plan(p, allow_unmapped_ff=True)
    cp = require_private_file(cold_path, "cold snapshot")
    try:
        cold_data = json.loads(cp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BurnError(f"invalid JSON in cold snapshot {cp}: {exc}") from exc
    with Journal(journal_path) as j:
        receipt = verify_cold_snapshot(p, cold_data, stage=stage, journal=j)
    if receipt_out:
        write_private_json(receipt_out, receipt, overwrite=overwrite)
    io.show(
        "PASS: supplied cold snapshot matches declared expectations; "
        "evidence is not independently authenticated"
    )
    return 0
