"""Kernel menuconfig-style interactive configurator for CMP 170HX 8 GB fuse unlocks.

Uses Python's built-in ``curses`` library when running on an interactive TTY (SSH/tmux/console)
for true ``make menuconfig`` arrow-key / Space / Enter navigation, and seamlessly falls back
to an identical numbered plain-text tree when running without a TTY or under ``--no-curses``.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Callable

from .ffrecord import encode_ff
from .fuses import (
    A100_TARGET_DEVICE_IDS,
    BURNABLE_TARGET_DEVICE_IDS,
    SM_PER_TPC,
    TOTAL_FBP,
    TOTAL_GPC,
    TPC_PER_GPC,
)
from .model import BurnError, SKU_CMP170HX_10GB, SKU_CMP170HX_8GB
from .planner import CardSnapshot, UnlockConfig, derive_ff_records

try:
    import curses
except ImportError:  # pragma: no cover - standard on Linux
    curses = None  # type: ignore[assignment]


@dataclass
class _MenuEntry:
    label: str
    help_text: str
    on_toggle: Callable[[], str | None] | None = None
    on_enter: Callable[[], str | None] | None = None
    submenu: str | None = None
    locked: bool = False


def _compute_preview_lines(card: CardSnapshot, config: UnlockConfig) -> list[str]:
    lines = [
        f"Card: {card.pci_id} | Next free FF row: {card.next_free_row} ({card.free_row_count} free slots)",
        f"Config: {config.summary_line()}",
        "",
    ]
    if config.a100_devid is not None:
        if config.a100_devid in A100_TARGET_DEVICE_IDS:
            lines.extend(
                [
                    "!!! WARNING — A100 PCIe DEVICE ID CONVERSION SELECTED !!!",
                    f"!!! Target: {card.pci_id} -> 10de:{config.a100_devid:04x} ({BURNABLE_TARGET_DEVICE_IDS[config.a100_devid]})",
                    "!!! - Card WILL NOT POST on stock CMP 170HX VBIOS after cold boot; back up CMP ROM",
                    "!!!   (nvflash --save) BEFORE burning and flash matching A100 VBIOS right after.",
                    "!!! - Host BIOS MUST support 128 GB 64-bit BAR1 (Above 4G, CSM OFF, ReBAR ON);",
                    "!!!   POST failure requires offline CH341A SPI recovery.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "!!! WARNING — CMP 170HX PCIe DEVICE ID REVERT SELECTED !!!",
                    f"!!! Target: {card.pci_id} -> 10de:{config.a100_devid:04x} ({BURNABLE_TARGET_DEVICE_IDS[config.a100_devid]})",
                    "!!! - If an A100 VBIOS is currently installed, card WILL NOT POST on it after cold boot;",
                    "!!!   flash your backed-up matching CMP 170HX VBIOS right after cold power cycle.",
                    "",
                ]
            )
    try:
        items, issues = derive_ff_records(card, config)
    except BurnError as exc:
        lines.append(f"ERROR deriving FF records: {exc}")
        return lines

    if not items:
        lines.append("No fuse changes selected (card would remain unchanged).")
        return lines

    lines.append(
        f"Planned FF writes: 1 automatic no-op test row (row {card.next_free_row}) + "
        f"{len(items)} real record(s) (rows {card.next_free_row + 1}..{card.next_free_row + len(items)})"
    )
    lines.append(
        f"Free FF rows remaining after burn: {card.free_row_count - (len(items) + 1)}"
    )
    lines.append("")
    for idx, it in enumerate(items, start=1):
        row = card.next_free_row + idx
        note = f"  [{it['note']}]" if it.get("note") else ""
        fields_s = ",".join(it["fields"]) or "raw"
        old_s = f"0x{it['old']:04X}" if it.get("old") is not None else "FPF   "
        lines.append(
            f"  row {row:3d}: 0x{it['word']:08X}  c{it['chain']}/b{it['block']:<2d} "
            f"{old_s}->0x{it['new']:04X}  ({fields_s}){note}"
        )
    if issues:
        lines.append("")
        lines.append("Coherence warnings:")
        for iss in issues:
            lines.append(f"  [{iss.severity.upper()}] {iss.message}")
    else:
        lines.append("")
        lines.append("Coherence check: PASS (0 warnings)")
    return lines


def _build_entries(screen: str, card: CardSnapshot, config: UnlockConfig) -> tuple[str, list[_MenuEntry]]:
    """Build title and selectable entries for the given menuconfig screen."""
    if screen == "root":
        all_sm_hbm = config.is_all_healthy_sm_hbm_unlocked()
        cur_sm, tgt_sm, max_sm = (
            card.current_sm_count(),
            config.target_sm_count(),
            card.max_healthy_sm_count(),
        )
        cur_fbp, tgt_fbp, max_fbp = (
            card.current_fbp_count(),
            config.target_fbp_count(),
            card.max_healthy_fbp_count(),
        )
        try:
            items, issues = derive_ff_records(card, config)
            ff_summary = f"+{len(items)} FF row(s)"
            if issues:
                ff_summary += f", {len(issues)} warning(s)"
        except BurnError as exc:
            ff_summary = f"ERROR: {exc}"

        def _toggle_all_sm_hbm() -> str:
            if config.is_all_healthy_sm_hbm_unlocked():
                config.reset_compute()
                config.reset_memory()
                return "Reset Compute + Memory back to current card state."
            config.unlock_all_healthy_sm_hbm()
            return f"Unlocked all healthy Compute ({config.target_sm_count()} SMs) + Memory ({config.target_fbp_count()} FBPs)."

        def _toggle_nvlink() -> str:
            if card.nvlink_all_defective():
                return "NVLink is defective in OPT_NVLINK_DEFECTIVE — cannot enable."
            config.set_nvlink(not config.nvlink_enabled)
            return f"NVLink set to {'ON' if config.nvlink_enabled else 'OFF'}."

        nvl_tag = "[X]" if card.nvlink_all_defective() else ("[*]" if config.nvlink_enabled else "[ ]")
        nvl_note = (
            "DEFECTIVE (locked)"
            if card.nvlink_all_defective()
            else ("ON -> LOCK" if (card.nvlink_currently_enabled() and not config.nvlink_enabled)
                  else ("OFF -> UNLOCK" if (not card.nvlink_currently_enabled() and config.nvlink_enabled)
                        else ("currently ON" if config.nvlink_enabled else "currently OFF")))
        )

        entries = [
            _MenuEntry(
                label=f"[{'*' if all_sm_hbm else ' '}] Unlock ALL healthy Compute + Memory (SM + HBM)",
                help_text=(
                    "One-step toggle: unlocks every non-defective GPC, TPC, PES, FBP, FBPA, FBIO, "
                    "and ROP_L2 partition on this die. Press Space again to reset Compute+Memory "
                    "to current card state. Use the submenus below to customize individual units."
                ),
                on_toggle=_toggle_all_sm_hbm,
                on_enter=_toggle_all_sm_hbm,
            ),
            _MenuEntry(
                label=(
                    f"    Compute (GPC / TPC / PES): {cur_sm}->{tgt_sm} SMs "
                    f"(healthy max {max_sm}, {config.target_gpc_count()}/{TOTAL_GPC} GPCs)  --->"
                ),
                help_text=(
                    "Enter to unlock or lock individual GPCs (0..7) and individual TPCs (0..7, 2 SMs each). "
                    "PES slices are kept coherent automatically so enabled GPCs always boot cleanly."
                ),
                submenu="compute",
            ),
            _MenuEntry(
                label=(
                    f"    Memory / HBM (FBP / FBPA / FBIO / ROP_L2): {cur_fbp}->{tgt_fbp} FBPs "
                    f"(healthy max {max_fbp}/{TOTAL_FBP})  --->"
                ),
                help_text=(
                    "Enter to unlock or lock individual memory partitions (FBP0..FBP11). "
                    "Paired FBPA, FBIO, and ROP_L2 slices are kept coherent automatically."
                ),
                submenu="memory",
            ),
            _MenuEntry(
                label=f"    PCIe Link Speed: {config.target_pcie_label()}  --->",
                help_text=(
                    "Choose PCIe speed unlock: Keep current, Gen3 (2 FF rows), "
                    "Gen4 (Gen3 + PCIE_MAGIC_BITS_D -> 0x00200000, 4 FF rows), "
                    "or Lock back to Gen1/Gen2."
                ),
                submenu="pcie",
            ),
            _MenuEntry(
                label=f"{nvl_tag} Enable NVLink ({nvl_note}; needs U11 bridge mod + FW bypass)",
                help_text=(
                    "Clears non-defective bits of OPT_NVLINK_DISABLE in chain 1 block 23 while "
                    "preserving OPT_SYS_PIPE_DISABLE and unmapped bit 371."
                ),
                on_toggle=_toggle_nvlink,
                on_enter=_toggle_nvlink,
                locked=card.nvlink_all_defective(),
            ),
            _MenuEntry(
                label=f"    PCIe Device ID (A100 / CMP 170HX): {config.target_devid_label()}  --->",
                help_text=(
                    "WARNING: Sets fused DEVIDA (c2/b37) + DEVIDB (c2/b38) to an A100 ID "
                    "(20F1/20B5, PCIe) or reverts back to CMP 170HX (20C2 8 GB / 2082 10 GB). "
                    "Requires matching VBIOS after cold boot (A100 also needs 128 GB BAR1)."
                ),
                submenu="devid",
            ),
            _MenuEntry(
                label=(
                    (
                        f"[X] [EXPERIMENTAL] MIG / SYS_PIPE (already max healthy: "
                        f"{card.current_sys_pipe_count()}/{card.max_healthy_sys_pipe_count()} pipes)"
                    )
                    if card.sys_pipe_unlockable_mask() == 0
                    else (
                        f"[{'*' if config.sys_pipe_unlock else ' '}] [EXPERIMENTAL] Unlock MIG / SYS_PIPE "
                        f"({card.current_sys_pipe_count()}->{card.max_healthy_sys_pipe_count()}/8 pipes, c1/b23)"
                    )
                ),
                help_text=(
                    "EXPERIMENTAL: Clears non-defective bits of OPT_SYS_PIPE_DISABLE (c1/b23) "
                    "which controls the 8 GA100 hardware compute/MIG pipes (SYS_PIPE0..7). "
                    "Never clears genuinely defective bits in OPT_SYS_PIPE_DEFECTIVE."
                ),
                on_toggle=(
                    (lambda: "All non-defective SYS_PIPE units are already enabled on this card.")
                    if card.sys_pipe_unlockable_mask() == 0
                    else (
                        lambda: (
                            config.set_sys_pipe_unlock(not config.sys_pipe_unlock),
                            f"MIG / SYS_PIPE unlock set to {'ON' if config.sys_pipe_unlock else 'OFF'}.",
                        )[1]
                    )
                ),
                on_enter=(
                    (lambda: "All non-defective SYS_PIPE units are already enabled on this card.")
                    if card.sys_pipe_unlockable_mask() == 0
                    else (
                        lambda: (
                            config.set_sys_pipe_unlock(not config.sys_pipe_unlock),
                            f"MIG / SYS_PIPE unlock set to {'ON' if config.sys_pipe_unlock else 'OFF'}.",
                        )[1]
                    )
                ),
                locked=(card.sys_pipe_unlockable_mask() == 0),
            ),
            _MenuEntry(
                label=f"    Preview Exact FF Records ({ff_summary})  --->",
                help_text="Inspect every 32-bit FF word and row that will be written (plus coherence audit).",
                submenu="preview",
            ),
        ]
        return "Main Configuration", entries

    if screen == "compute":
        entries = [
            _MenuEntry(
                label=f"    [*] Unlock ALL healthy GPCs & TPCs ({card.max_healthy_sm_count()} SMs)",
                help_text="Enable every non-defective GPC, TPC, and paired PES slice.",
                on_toggle=lambda: (config.unlock_all_healthy_compute(), f"Compute set to {config.target_sm_count()} SMs.")[1],
                on_enter=lambda: (config.unlock_all_healthy_compute(), f"Compute set to {config.target_sm_count()} SMs.")[1],
            ),
            _MenuEntry(
                label=f"    [ ] Reset Compute to current card state ({card.current_sm_count()} SMs)",
                help_text="Discard all Compute (GPC/TPC/PES) changes.",
                on_toggle=lambda: (config.reset_compute(), "Compute reset to current card state.")[1],
                on_enter=lambda: (config.reset_compute(), "Compute reset to current card state.")[1],
            ),
        ]
        for g in range(TOTAL_GPC):
            is_def = card.gpc_defective(g)
            cur_on = card.gpc_currently_enabled(g)
            tgt_on = config.gpc_enabled[g]
            healthy_tpc_cnt = sum(1 for t in range(TPC_PER_GPC) if not card.tpc_defective(g, t))
            tgt_tpc_cnt = sum(1 for t in range(TPC_PER_GPC) if tgt_on and config.tpc_enabled[g][t])
            if is_def:
                entries.append(
                    _MenuEntry(
                        label=f"[X] GPC{g}  (DEFECTIVE in OPT_GPC_DEFECTIVE — locked)",
                        help_text=f"GPC{g} is marked genuinely defective in OPT_GPC_DEFECTIVE and cannot be enabled.",
                        on_toggle=lambda g=g: f"GPC{g} is defective in OPT_GPC_DEFECTIVE — cannot enable.",
                        locked=True,
                    )
                )
            else:
                state_tag = (
                    "OFF -> UNLOCK" if (not cur_on and tgt_on)
                    else ("ON -> LOCK" if (cur_on and not tgt_on)
                          else ("currently ON" if tgt_on else "currently OFF"))
                )
                def_tpcs = TPC_PER_GPC - healthy_tpc_cnt
                def_suffix = f", {def_tpcs} TPC defective" if def_tpcs else ""
                sm_cnt = tgt_tpc_cnt * SM_PER_TPC
                entries.append(
                    _MenuEntry(
                        label=(
                            f"[{'*' if tgt_on else ' '}] GPC{g}  "
                            f"({tgt_tpc_cnt}/{healthy_tpc_cnt} healthy TPCs = {sm_cnt:2d} SMs{def_suffix}) "
                            f"[{state_tag}]  --->"
                        ),
                        help_text=(
                            f"Space: toggle entire GPC{g} ON/OFF (with paired PES slices). "
                            f"Enter: open GPC{g} to toggle individual TPC0..TPC7 (2 SMs each)."
                        ),
                        on_toggle=lambda g=g: (
                            config.set_gpc(g, not config.gpc_enabled[g]),
                            f"GPC{g} {'enabled' if config.gpc_enabled[g] else 'disabled'} "
                            f"(total {config.target_sm_count()} SMs).",
                        )[1],
                        submenu=f"gpc:{g}",
                    )
                )
        return f"Compute — GPC / TPC / PES ({config.target_sm_count()}/{card.max_healthy_sm_count()} SMs)", entries

    if screen.startswith("gpc:"):
        g = int(screen.split(":", 1)[1])
        entries = []
        for t in range(TPC_PER_GPC):
            is_def = card.tpc_defective(g, t)
            cur_on = card.tpc_currently_enabled(g, t)
            tgt_on = config.gpc_enabled[g] and config.tpc_enabled[g][t]
            if is_def:
                entries.append(
                    _MenuEntry(
                        label=f"[X] GPC{g} / TPC{t}  (DEFECTIVE in OPT_TPC_GPC{g}_DEFECTIVE — locked)",
                        help_text=f"TPC{t} on GPC{g} is genuinely defective and cannot be enabled.",
                        on_toggle=lambda g=g, t=t: f"GPC{g}/TPC{t} is defective — cannot enable.",
                        locked=True,
                    )
                )
            else:
                state_tag = (
                    "OFF -> UNLOCK" if (not cur_on and tgt_on)
                    else ("ON -> LOCK" if (cur_on and not tgt_on)
                          else ("currently ON" if tgt_on else "currently OFF"))
                )
                def _toggle_tpc(g: int = g, t: int = t) -> str:
                    now = not (config.gpc_enabled[g] and config.tpc_enabled[g][t])
                    config.set_tpc(g, t, now)
                    return (
                        f"GPC{g}/TPC{t} {'enabled' if now else 'disabled'} "
                        f"(total {config.target_sm_count()} SMs)."
                    )
                entries.append(
                    _MenuEntry(
                        label=f"[{'*' if tgt_on else ' '}] GPC{g} / TPC{t}  (2 SMs)  [{state_tag}]",
                        help_text=f"Toggle TPC{t} on GPC{g} (2 SMs). Disabling all TPCs automatically disables GPC{g}.",
                        on_toggle=_toggle_tpc,
                        on_enter=_toggle_tpc,
                    )
                )
        return f"GPC{g} — Individual TPCs (2 SMs each)", entries

    if screen == "memory":
        entries = [
            _MenuEntry(
                label=f"    [*] Unlock ALL healthy FBPs ({card.max_healthy_fbp_count()}/{TOTAL_FBP} FBPs)",
                help_text="Enable every non-defective FBP along with its paired FBPA, FBIO, and ROP_L2 slices.",
                on_toggle=lambda: (config.unlock_all_healthy_memory(), f"Memory set to {config.target_fbp_count()} FBPs.")[1],
                on_enter=lambda: (config.unlock_all_healthy_memory(), f"Memory set to {config.target_fbp_count()} FBPs.")[1],
            ),
            _MenuEntry(
                label=f"    [ ] Reset Memory to current card state ({card.current_fbp_count()}/{TOTAL_FBP} FBPs)",
                help_text="Discard all Memory / HBM changes.",
                on_toggle=lambda: (config.reset_memory(), "Memory reset to current card state.")[1],
                on_enter=lambda: (config.reset_memory(), "Memory reset to current card state.")[1],
            ),
        ]
        for i in range(TOTAL_FBP):
            is_def = card.fbp_defective(i)
            cur_on = card.fbp_currently_enabled(i)
            tgt_on = config.fbp_enabled[i]
            if is_def:
                entries.append(
                    _MenuEntry(
                        label=f"[X] FBP{i:<2d} (slices {2 * i:2d},{2 * i + 1:2d})  (DEFECTIVE in OPT_FBP_DEFECTIVE — locked)",
                        help_text=f"FBP{i} is marked genuinely defective in OPT_FBP_DEFECTIVE and cannot be enabled.",
                        on_toggle=lambda i=i: f"FBP{i} is defective in OPT_FBP_DEFECTIVE — cannot enable.",
                        locked=True,
                    )
                )
            else:
                state_tag = (
                    "OFF -> UNLOCK" if (not cur_on and tgt_on)
                    else ("ON -> LOCK" if (cur_on and not tgt_on)
                          else ("currently ON" if tgt_on else "currently OFF"))
                )
                def _toggle_fbp(i: int = i) -> str:
                    config.set_fbp(i, not config.fbp_enabled[i])
                    return f"FBP{i} {'enabled' if config.fbp_enabled[i] else 'disabled'} ({config.target_fbp_count()} FBPs)."
                entries.append(
                    _MenuEntry(
                        label=f"[{'*' if tgt_on else ' '}] FBP{i:<2d} (paired FBPA/FBIO/ROP_L2 slices {2 * i:2d},{2 * i + 1:2d})  [{state_tag}]",
                        help_text=f"Toggle FBP{i} ON/OFF together with its paired FBPA, FBIO, and ROP_L2 slices.",
                        on_toggle=_toggle_fbp,
                        on_enter=_toggle_fbp,
                    )
                )
        return f"Memory / HBM — FBPs ({config.target_fbp_count()}/{card.max_healthy_fbp_count()} healthy)", entries

    if screen == "pcie":
        choices = [
            (
                "keep",
                f"Keep current card state ({card.current_pcie_label()})",
                "Emit zero PCIe FF records; leave chain 2 blocks untouched.",
            ),
            (
                "gen3",
                "PCIe Gen3  (safe: clears DISABLE_GEN3_SPEED + BOOT_GEN23/GEN3 — c2/b39, c2/b49)",
                "Unlocks PCIe Gen3 speed using well-understood chain 2 blocks 39 and 49.",
            ),
            (
                "gen4",
                "PCIe Gen4  (Gen3 + PCIE_MAGIC_BITS_D c2/b47, c2/b46 -> 0x00200000, 4 FF rows)",
                "Unlocks full PCIe Gen4 (clears DISABLE_GEN3_SPEED, BOOT_GEN23/GEN3, and the 5 Gen4 "
                "MAGIC_BITS_D bits 19/22/25/26/28 to match A100 0x00200000; 4 FF rows).",
            ),
            (
                "lock_gen12",
                "Lock to PCIe Gen1/Gen2  (sets DISABLE_GEN3_SPEED + BOOT_GEN23/GEN3)",
                "Re-applies Gen3/boot speed restrictions on chain 2 blocks 39 and 49.",
            ),
        ]
        entries = []
        for mode, lbl, hlp in choices:
            sel = "*" if config.pcie_mode == mode else " "
            entries.append(
                _MenuEntry(
                    label=f"({sel}) {lbl}",
                    help_text=hlp,
                    on_toggle=lambda m=mode: (config.set_pcie_mode(m), f"PCIe speed set to {config.target_pcie_label()}.")[1],
                    on_enter=lambda m=mode: (config.set_pcie_mode(m), f"PCIe speed set to {config.target_pcie_label()}.")[1],
                )
            )
        return "PCIe Link Speed (radio selection)", entries

    if screen == "devid":
        a100_warning = (
            "WARNING: Requires matching A100 VBIOS flash after cold boot (stock CMP ROM will NOT POST) "
            "AND host BIOS 128 GB BAR1 support (Above 4G ON, CSM OFF, ReBAR ON; else CH341A SPI recovery)."
        )
        cmp_warning = (
            "WARNING: Reverts fused DEVIDA (c2/b37) + DEVIDB (c2/b38) back to CMP 170HX. "
            "If an A100 VBIOS is currently installed, flash your backed-up CMP 170HX VBIOS after cold boot."
        )
        rec_10g = "  [matches 10 GB / 40 GB HBM2e card]" if card.pci_id == SKU_CMP170HX_10GB else ""
        rec_8g = "  [matches 8 GB / 80 GB HBM2e card]" if card.pci_id == SKU_CMP170HX_8GB else ""
        choices: list[tuple[int | None, str, str]] = [
            (
                None,
                f"Keep current PCIe Device ID ({card.pci_id}) — no VBIOS change",
                "Leave DEVIDA (c2/b37) and DEVIDB (c2/b38) untouched; keeps current VBIOS compatibility.",
            ),
            (
                0x20F1,
                f"A100 PCIe 40 GB  (10de:20f1) — DEVIDA+DEVIDB (2 FF rows){rec_10g}",
                a100_warning,
            ),
            (
                0x20B5,
                f"A100 PCIe 80 GB  (10de:20b5) — DEVIDA+DEVIDB (2 FF rows){rec_8g}",
                a100_warning,
            ),
            (
                0x20C2,
                "CMP 170HX 8 GB   (10de:20c2) — revert DEVIDA+DEVIDB (2 FF rows; needs CMP 8 GB VBIOS)",
                cmp_warning,
            ),
            (
                0x2082,
                "CMP 170HX 10 GB  (10de:2082) — revert DEVIDA+DEVIDB (2 FF rows; needs CMP 10 GB VBIOS)",
                cmp_warning,
            ),
        ]
        entries = []
        for target_id, lbl, hlp in choices:
            sel = "*" if config.a100_devid == target_id else " "

            def _pick(tid: int | None = target_id) -> str:
                config.set_a100_devid(tid)
                if tid is None:
                    return f"PCIe Device ID reset to keep ({card.pci_id})."
                if tid in A100_TARGET_DEVICE_IDS:
                    return (
                        f"WARNING: Device ID set to 10de:{tid:04x} ({BURNABLE_TARGET_DEVICE_IDS[tid]}). "
                        "Needs A100 VBIOS after cold boot + 128 GB BAR1!"
                    )
                return (
                    f"WARNING: Device ID set to revert 10de:{tid:04x} ({BURNABLE_TARGET_DEVICE_IDS[tid]}). "
                    "Needs matching CMP 170HX VBIOS after cold boot!"
                )

            entries.append(
                _MenuEntry(
                    label=f"({sel}) {lbl}",
                    help_text=hlp,
                    on_toggle=_pick,
                    on_enter=_pick,
                )
            )
        return "PCIe Device ID — A100 / CMP 170HX (WARNING: must match flashed VBIOS)", entries

    if screen == "preview":
        preview_lines = _compute_preview_lines(card, config)
        entries = [
            _MenuEntry(label=line or " ", help_text="Preview of derived FF records.", locked=True)
            for line in preview_lines
        ]
        return "Preview Exact FF Records", entries

    return "Main Configuration", []


def run_menuconfig_plain(
    card: CardSnapshot,
    config: UnlockConfig | None = None,
    *,
    input_fn=None,
    show=None,
) -> UnlockConfig | None:
    """Plain-text numbered menuconfig fallback (works over serial/pipes/tests)."""
    cfg = config if config is not None else UnlockConfig.from_card(card)
    ask = input_fn if input_fn is not None else (lambda prompt="": input(prompt))
    out = show if show is not None else (lambda *a, **kw: print(*a, **kw))
    stack: list[str] = ["root"]
    status = ""

    while True:
        screen = stack[-1]
        title, entries = _build_entries(screen, card, cfg)
        out("\n" + "=" * 76)
        out(f"  vfi_burn menuconfig — {title}")
        out(f"  {cfg.summary_line()}")
        out("=" * 76)
        if status:
            out(f"  >> {status}")
            status = ""
        if screen == "preview":
            for e in entries:
                out(f"  {e.label}")
            out("-" * 76)
            raw = ask("[Enter/b] Back, [s] Save & generate plan, [q] Cancel: ").strip().lower()
        else:
            for idx, e in enumerate(entries, start=1):
                out(f"  [{idx:2d}] {e.label}")
            out("-" * 76)
            extra = "t<N>=toggle GPC, " if screen == "compute" else ""
            back_hint = "[b] Back, " if len(stack) > 1 else ""
            raw = ask(
                f"Pick [1-{len(entries)}] ({extra}?<N>=help, {back_hint}[p] Preview, [s] Save plan, [q] Cancel): "
            ).strip()

        low = raw.lower()
        if low in ("q", "quit", "cancel"):
            return None
        if low in ("s", "save"):
            try:
                items, issues = derive_ff_records(card, cfg)
            except BurnError as exc:
                status = f"REFUSED: {exc}"
                continue
            if not items:
                status = "No changes selected yet — toggle at least one unlock/lock before saving."
                continue
            errs = [iss.message for iss in issues if iss.severity == "error"]
            if errs:
                status = "REFUSED (incoherent): " + "; ".join(errs)
                continue
            return cfg
        if low in ("b", "back", "0") or (screen == "preview" and low == ""):
            if len(stack) > 1:
                stack.pop()
            else:
                return None
            continue
        if low == "p" and screen != "preview":
            stack.append("preview")
            continue
        if low.startswith("?") and len(low) > 1 and low[1:].isdigit():
            i = int(low[1:])
            if 1 <= i <= len(entries):
                status = entries[i - 1].help_text
            continue
        if screen == "compute" and low.startswith("t") and len(low) > 1 and low[1:].isdigit():
            i = int(low[1:])
            if 1 <= i <= len(entries) and entries[i - 1].on_toggle is not None:
                status = entries[i - 1].on_toggle() or ""
            continue
        if low.isdigit():
            i = int(low)
            if 1 <= i <= len(entries):
                entry = entries[i - 1]
                if entry.submenu is not None:
                    stack.append(entry.submenu)
                elif entry.on_enter is not None:
                    status = entry.on_enter() or ""
                elif entry.on_toggle is not None:
                    status = entry.on_toggle() or ""
            else:
                status = f"Selection {i} outside 1..{len(entries)}."


def _run_menuconfig_curses(card: CardSnapshot, config: UnlockConfig) -> UnlockConfig | None:  # pragma: no cover
    assert curses is not None

    def _main(stdscr) -> UnlockConfig | None:
        curses.curs_set(0)
        stdscr.keypad(True)
        has_colors = False
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)   # header/footer bar
                curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)   # selected row
                curses.init_pair(3, curses.COLOR_RED, -1)                    # defective/warning
                curses.init_pair(4, curses.COLOR_YELLOW, -1)                 # status line
                has_colors = True
            except curses.error:
                has_colors = False

        stack: list[tuple[str, int, int]] = [("root", 0, 0)]  # (screen, cursor, scroll)
        status = "Space=Toggle  Enter=Submenu  Left/Esc=Back  ?=Help  P=Preview  S=Save Plan  Q=Cancel"
        help_popup: str | None = None

        while True:
            screen, cur, scroll = stack[-1]
            title, entries = _build_entries(screen, card, config)
            if entries:
                cur = max(0, min(cur, len(entries) - 1))
            else:
                cur = 0

            stdscr.erase()
            h, w = stdscr.getmaxyx()
            hdr_attr = (curses.color_pair(1) | curses.A_BOLD) if has_colors else curses.A_REVERSE
            sel_attr = (curses.color_pair(2) | curses.A_BOLD) if has_colors else curses.A_REVERSE
            def_attr = curses.color_pair(3) if has_colors else curses.A_DIM
            stat_attr = (curses.color_pair(4) | curses.A_BOLD) if has_colors else curses.A_BOLD

            def _put(y: int, x: int, text: str, attr: int = 0) -> None:
                if 0 <= y < h and x < w:
                    try:
                        stdscr.addnstr(y, x, text, max(0, w - x - 1), attr)
                    except curses.error:
                        pass

            # Top 2 header rows
            _put(0, 0, f" vfi_burn menuconfig — {title} ".ljust(w), hdr_attr)
            _put(1, 0, f" {config.summary_line()} ".ljust(w), hdr_attr)

            list_top = 3
            list_height = max(1, h - 6)
            if cur < scroll:
                scroll = cur
            elif cur >= scroll + list_height:
                scroll = cur - list_height + 1

            for row_off in range(list_height):
                idx = scroll + row_off
                if idx >= len(entries):
                    break
                e = entries[idx]
                is_sel = (idx == cur) and screen != "preview"
                attr = sel_attr if is_sel else (def_attr if e.locked else 0)
                prefix = " > " if is_sel else "   "
                _put(list_top + row_off, 0, f"{prefix}{e.label}".ljust(w - 1), attr)

            if help_popup:
                _put(h - 3, 0, f" HELP: {help_popup} ", stat_attr)
                help_popup = None
            else:
                _put(h - 3, 0, f" {status} ", stat_attr)

            footer = " [Up/Down] Move  [Space] Toggle  [Enter] Open  [Esc/Left] Back  [?] Help  [P] Preview  [S] Save  [Q] Cancel "
            _put(h - 1, 0, footer.ljust(w), hdr_attr)
            stdscr.refresh()

            ch = stdscr.getch()
            stack[-1] = (screen, cur, scroll)

            if ch in (ord("q"), ord("Q")):
                return None
            if ch in (curses.KEY_UP, ord("k")):
                if screen == "preview":
                    scroll = max(0, scroll - 1)
                else:
                    cur = max(0, cur - 1)
                stack[-1] = (screen, cur, scroll)
            elif ch in (curses.KEY_DOWN, ord("j")):
                if screen == "preview":
                    scroll = min(max(0, len(entries) - list_height), scroll + 1)
                else:
                    cur = min(max(0, len(entries) - 1), cur + 1)
                stack[-1] = (screen, cur, scroll)
            elif ch in (curses.KEY_LEFT, 27, curses.KEY_BACKSPACE, 127, ord("b"), ord("B")):
                if len(stack) > 1:
                    stack.pop()
                elif ch == 27:
                    return None
            elif ch in (ord("?"), ord("h"), ord("H")):
                if entries:
                    help_popup = entries[cur].help_text
            elif ch in (ord("p"), ord("P")):
                if screen != "preview":
                    stack.append(("preview", 0, 0))
            elif ch in (ord("s"), ord("S")):
                try:
                    items, issues = derive_ff_records(card, config)
                except BurnError as exc:
                    status = f"REFUSED: {exc}"
                    continue
                if not items:
                    status = "No fuse changes selected — toggle at least one item before saving."
                    continue
                errs = [iss.message for iss in issues if iss.severity == "error"]
                if errs:
                    status = "REFUSED (incoherent): " + "; ".join(errs)
                    continue
                return config
            elif ch == ord(" "):
                if entries and screen != "preview":
                    e = entries[cur]
                    if e.on_toggle is not None:
                        status = e.on_toggle() or status
                    elif e.submenu is not None:
                        stack.append((e.submenu, 0, 0))
            elif ch in (curses.KEY_ENTER, 10, 13, curses.KEY_RIGHT):
                if entries and screen != "preview":
                    e = entries[cur]
                    if e.submenu is not None:
                        stack.append((e.submenu, 0, 0))
                    elif e.on_enter is not None:
                        status = e.on_enter() or status
                    elif e.on_toggle is not None:
                        status = e.on_toggle() or status

    return curses.wrapper(_main)


def run_menuconfig(
    card: CardSnapshot,
    config: UnlockConfig | None = None,
    *,
    use_curses: bool | None = None,
    input_fn=None,
    show=None,
) -> UnlockConfig | None:
    """Launch the kernel menuconfig-style unlock configurator.

    Uses ``curses`` automatically on an interactive TTY unless ``use_curses=False`` or
    custom ``input_fn``/``show`` callbacks are provided; falls back cleanly to plain-text mode.
    """
    cfg = config if config is not None else UnlockConfig.from_card(card)
    want_curses = (
        use_curses
        if use_curses is not None
        else (
            curses is not None
            and input_fn is None
            and show is None
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        )
    )
    if want_curses and curses is not None:  # pragma: no cover - interactive TTY
        try:
            return _run_menuconfig_curses(card, cfg)
        except curses.error:
            pass
    return run_menuconfig_plain(card, cfg, input_fn=input_fn, show=show)
