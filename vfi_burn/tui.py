"""Zero-dependency stdlib curses full-screen TUI for vfi_burn (Option B).

Provides the headless / SSH / no-browser interface over the exact same
``WorkflowController`` used by the Local Web UI:
  - 5-step guided pipeline + Tools tab with contextual step help
  - Two-pane Midnight-Commander-style file selector with metadata & 0600 fix
  - Integrated kernel-menuconfig configurator (split Compute/Memory + Est. RAM)
  - Live Goldilocks Watch (1.5 s read-only BAR0 polling, auto-stops on poison)
  - Streamlined group-mode burn flow & one-step cold-boot verification
"""
from __future__ import annotations

from pathlib import Path
import sys
import textwrap

from .backend import write_private_json_interactive
from .console import ManualConsole
from .engine import CONFIRM_MODES, Journal
from .menu import BurnMenu, MenuSession
from .menuconfig import run_menuconfig
from .model import BurnError, Plan
from .planner import build_plan_from_config
from .workflow import (
    STEP_HELP,
    WorkflowController,
    fix_file_permissions,
    list_directory_entries,
)

try:
    import curses
except ImportError:  # pragma: no cover - standard on Linux
    curses = None  # type: ignore[assignment]


TABS: tuple[tuple[str, str], ...] = (
    ("target", "1 Target"),
    ("configure", "2 Plan"),
    ("preflight", "3 Preflight"),
    ("burn", "4 Burn"),
    ("cold_verify", "5 Cold Verify"),
    ("tools", "6 Tools"),
)

_FILTERS: tuple[tuple[str, str], ...] = (
    ("all", "All"),
    ("plan", "Plans"),
    ("cold_snapshot", "Cold Snaps"),
    ("test_receipt", "Receipts"),
    ("fake_snapshot", "Sim Snaps"),
)


def _safe_put(win, y: int, x: int, text: str, attr: int = 0) -> None:
    h, w = win.getmaxyx()
    if 0 <= y < h and 0 <= x < w:
        try:
            win.addnstr(y, x, text, max(0, w - x - 1), attr)
        except Exception:
            pass


def _prompt_modal_curses(stdscr, title: str, prompt: str, *, default: str = "") -> str | None:
    """Centered single-line text input dialog inside curses."""
    h, w = stdscr.getmaxyx()
    bw = min(max(50, w - 8), 80)
    bh = 8
    by = max(1, (h - bh) // 2)
    bx = max(1, (w - bw) // 2)
    win = curses.newwin(bh, bw, by, bx)
    win.keypad(True)
    buf = list(default)
    curses.curs_set(1)
    try:
        while True:
            win.erase()
            win.box()
            _safe_put(win, 0, 2, f" {title} ", curses.A_BOLD)
            wrapped = textwrap.wrap(prompt, max(10, bw - 4))[:3]
            for i, line in enumerate(wrapped):
                _safe_put(win, 1 + i, 2, line)
            text = "".join(buf)
            vis = text[-(bw - 6):] if len(text) > bw - 6 else text
            _safe_put(win, 5, 2, "> " + vis.ljust(bw - 6), curses.A_REVERSE)
            _safe_put(win, 6, 2, "[Enter] OK   [Esc] Cancel", curses.A_DIM)
            win.move(5, min(bw - 2, 4 + len(vis)))
            win.refresh()
            ch = win.getch()
            if ch in (27,):
                return None
            if ch in (10, 13, curses.KEY_ENTER):
                return "".join(buf).strip()
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                if buf:
                    buf.pop()
            elif 32 <= ch <= 126:
                buf.append(chr(ch))
    finally:
        curses.curs_set(0)


def _scroll_viewer_curses(stdscr, title: str, lines: list[str], colors: dict[str, int]) -> None:
    """Full-screen scrollable text viewer inside curses."""
    scroll = 0
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        _safe_put(stdscr, 0, 0, f" {title} ".ljust(w), colors["hdr"])
        body_h = max(1, h - 3)
        max_scroll = max(0, len(lines) - body_h)
        scroll = max(0, min(scroll, max_scroll))
        for row in range(body_h):
            idx = scroll + row
            if idx >= len(lines):
                break
            _safe_put(stdscr, 1 + row, 1, lines[idx])
        _safe_put(
            stdscr,
            h - 1,
            0,
            f" [Up/Down/PgUp/PgDn] Scroll ({scroll + 1}/{max(1, len(lines))})   [Esc/q/Enter] Back ".ljust(w),
            colors["hdr"],
        )
        stdscr.refresh()
        ch = stdscr.getch()
        if ch in (27, ord("q"), ord("Q"), 10, 13, curses.KEY_LEFT):
            return
        if ch in (curses.KEY_UP, ord("k")):
            scroll -= 1
        elif ch in (curses.KEY_DOWN, ord("j")):
            scroll += 1
        elif ch == curses.KEY_PPAGE:
            scroll -= body_h
        elif ch == curses.KEY_NPAGE:
            scroll += body_h


def run_file_selector_curses(
    stdscr,
    colors: dict[str, int],
    *,
    title: str = "Select File",
    start_dir: str = "private",
    kind_filter: str = "all",
    target_pci_id: str | None = None,
) -> str | None:
    """Two-pane Midnight-Commander-style curses file selector with live metadata preview."""
    cur_dir = start_dir
    filt_idx = next((i for i, (k, _) in enumerate(_FILTERS) if k == kind_filter), 0)
    cur = 0
    scroll = 0
    status = "Enter=Open/Select  f=Filter  c=chmod 600  p=private/  Esc=Cancel"

    while True:
        filt_key, filt_lbl = _FILTERS[filt_idx]
        try:
            listing = list_directory_entries(
                cur_dir, kind_filter=filt_key, target_pci_id=target_pci_id
            )
            cur_dir = listing["directory"]
            entries = listing["entries"]
        except Exception as exc:
            entries = []
            status = f"Error: {exc}"

        cur = max(0, min(cur, len(entries) - 1)) if entries else 0
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        split_x = max(28, int(w * 0.56))
        body_top = 2
        body_h = max(3, h - 5)

        _safe_put(stdscr, 0, 0, f" {title} — Filter: [{filt_lbl}] ".ljust(w), colors["hdr"])
        _safe_put(stdscr, 1, 0, f" Dir: {cur_dir}".ljust(w), colors["sub"])

        if cur < scroll:
            scroll = cur
        elif cur >= scroll + body_h:
            scroll = cur - body_h + 1

        # Left pane: directory entries
        for row in range(body_h):
            idx = scroll + row
            if idx >= len(entries):
                break
            e = entries[idx]
            tag = "[DIR] " if e["is_dir"] else (
                "[PLAN]" if e["kind"] == "plan" else
                "[COLD]" if e["kind"] == "cold_snapshot" else
                "[RCPT]" if e["kind"] == "test_receipt" else
                "[SIM] " if e["kind"] == "fake_snapshot" else
                "[FILE]"
            )
            perm = "" if e["is_dir"] else (" 0600" if e["is_private_0600"] else f" !{e['mode_octal']}")
            line = f" {tag} {e['name']}{perm}"
            attr = colors["sel"] if idx == cur else (colors["warn"] if (not e["is_dir"] and not e["is_private_0600"]) else 0)
            _safe_put(stdscr, body_top + row, 0, line[: split_x - 1].ljust(split_x - 1), attr)

        # Vertical divider
        for y in range(body_top, body_top + body_h):
            _safe_put(stdscr, y, split_x, "|", curses.A_DIM)

        # Right pane: live preview of selected item
        px = split_x + 2
        pw = max(12, w - px - 1)
        if entries:
            sel = entries[cur]
            preview_lines = [
                f"Name   : {sel['name']}",
                f"Type   : {sel['summary']}",
            ]
            if not sel["is_dir"]:
                preview_lines.append(
                    f"Perms  : {sel['mode_octal']} "
                    + ("(OK 0600)" if sel["is_private_0600"] else "(Press 'c' to chmod 600)")
                )
                preview_lines.append(f"Size   : {sel['size_label']}  Modified: {sel['mtime_label']}")
            if sel.get("sku_match") == "match":
                preview_lines.append("Card   : MATCHES SELECTED GPU")
            elif sel.get("sku_match") in ("mismatch", "die_mismatch"):
                preview_lines.append("Card   : WARNING — DIFFERENT CARD")
            det = sel.get("details") or {}
            if det.get("configured_summary"):
                preview_lines.append("")
                preview_lines.append("Plan Summary:")
                preview_lines.extend(textwrap.wrap(str(det["configured_summary"]), pw))
            row_y = body_top
            for pl in preview_lines:
                if row_y >= body_top + body_h:
                    break
                attr = colors["ok"] if "MATCHES" in pl else (colors["danger"] if "DIFFERENT" in pl else 0)
                _safe_put(stdscr, row_y, px, pl, attr)
                row_y += 1

        _safe_put(stdscr, h - 2, 0, f" {status} ", colors["warn"])
        _safe_put(
            stdscr,
            h - 1,
            0,
            " [Up/Down] Move  [Enter] Open/Select  [f] Filter  [c] chmod 600  [p] private/  [Esc] Cancel ".ljust(w),
            colors["hdr"],
        )
        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (27, ord("q"), ord("Q")):
            return None
        if ch in (curses.KEY_UP, ord("k")):
            cur = max(0, cur - 1)
        elif ch in (curses.KEY_DOWN, ord("j")):
            cur = min(max(0, len(entries) - 1), cur + 1)
        elif ch in (ord("f"), ord("F")):
            filt_idx = (filt_idx + 1) % len(_FILTERS)
            cur = scroll = 0
        elif ch in (ord("p"), ord("P")):
            cur_dir = "private"
            cur = scroll = 0
        elif ch in (ord("c"), ord("C")) and entries:
            sel = entries[cur]
            if not sel["is_dir"]:
                try:
                    fix_file_permissions(sel["path"])
                    status = f"Set mode 0600 on {sel['name']}."
                except Exception as exc:
                    status = f"chmod failed: {exc}"
        elif ch in (10, 13, curses.KEY_ENTER, curses.KEY_RIGHT) and entries:
            sel = entries[cur]
            if sel["is_dir"]:
                cur_dir = sel["path"]
                cur = scroll = 0
            elif sel["is_symlink"]:
                status = "Refused: private files cannot be symlinks."
            else:
                return sel["path"]
        elif ch in (curses.KEY_BACKSPACE, 127, curses.KEY_LEFT):
            parent = Path(cur_dir).resolve().parent
            cur_dir = str(parent)
            cur = scroll = 0


class BurnTUI:
    """Full-screen stdlib curses TUI wrapping WorkflowController."""

    def __init__(self, controller: WorkflowController) -> None:
        self.ctl = controller
        self.tab_idx = 0
        self.status = "Use 1-6 or Tab to switch steps. Press ? for step help, q to quit."
        self.watch_active = False
        self.cold_stage: str | None = None
        self.last_cold_result: dict | None = None
        self.gpus_cache: list[dict] = []
        self.gpu_cursor = 0
        self.ov_cursor = 0

    def _init_colors(self) -> dict[str, int]:
        colors = {
            "hdr": curses.A_REVERSE | curses.A_BOLD,
            "sub": curses.A_BOLD,
            "sel": curses.A_REVERSE | curses.A_BOLD,
            "ok": curses.A_BOLD,
            "warn": curses.A_BOLD,
            "danger": curses.A_BOLD,
            "tab_on": curses.A_REVERSE | curses.A_BOLD,
        }
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
                curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)
                curses.init_pair(3, curses.COLOR_GREEN, -1)
                curses.init_pair(4, curses.COLOR_YELLOW, -1)
                curses.init_pair(5, curses.COLOR_RED, -1)
                curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_RED)
                colors["hdr"] = curses.color_pair(1) | curses.A_BOLD
                colors["sel"] = curses.color_pair(2) | curses.A_BOLD
                colors["ok"] = curses.color_pair(3) | curses.A_BOLD
                colors["warn"] = curses.color_pair(4) | curses.A_BOLD
                colors["danger"] = curses.color_pair(5) | curses.A_BOLD
                colors["caution"] = curses.color_pair(6) | curses.A_BOLD
                colors["tab_on"] = curses.color_pair(2) | curses.A_BOLD
            except curses.error:
                colors["caution"] = curses.A_REVERSE | curses.A_BOLD
        else:
            colors["caution"] = curses.A_REVERSE | curses.A_BOLD
        return colors

    def _suspend_for_terminal(self, fn) -> None:
        curses.endwin()
        try:
            fn()
            try:
                input("\nPress Enter to return to vfi_burn TUI...")
            except (EOFError, KeyboardInterrupt):
                pass
        finally:
            curses.doupdate()

    def run(self) -> int:
        if curses is None or not sys.stdin.isatty() or not sys.stdout.isatty():
            # Graceful fallback to plain numbered menu on non-TTY / serial.
            return BurnMenu(
                MenuSession(
                    plan_path=self.ctl.state.plan_path,
                    bdf=self.ctl.state.bdf,
                    adapter_spec=self.ctl.state.adapter_spec,
                    fake_path=self.ctl.state.fake_path,
                    journal_path=self.ctl.state.journal_path,
                    receipt_path=self.ctl.state.receipt_path,
                    execute_live=self.ctl.execute_live,
                    overrides=self.ctl.overrides,
                )
            ).run()
        return curses.wrapper(self._main)

    def _main(self, stdscr) -> int:
        curses.curs_set(0)
        stdscr.keypad(True)
        colors = self._init_colors()
        self.gpus_cache = self.ctl.list_gpus(probe_bar0=True)

        # Jump straight to recommended step if a previous session was restored.
        ov0 = self.ctl.get_overview()
        if ov0["has_target"] and ov0["recommended_step"] in [k for k, _ in TABS]:
            self.tab_idx = [k for k, _ in TABS].index(ov0["recommended_step"])

        while True:
            step_key, _ = TABS[self.tab_idx]
            stdscr.timeout(1500 if (self.watch_active and step_key == "preflight") else -1)
            ov = self.ctl.get_overview()
            self._render(stdscr, ov, colors)
            ch = stdscr.getch()
            if ch == -1:
                # Timeout refresh during Live Watch
                if (ov.get("live_status") or {}).get("badge") == "POISON":
                    self.watch_active = False
                    self.status = "WATCH STOPPED: 0xBADxxxxx poison detected!"
                continue

            if ch in (ord("q"), ord("Q")):
                return 0
            if ord("1") <= ch <= ord("6"):
                self.tab_idx = ch - ord("1")
                if TABS[self.tab_idx][0] == "target":
                    self.gpus_cache = self.ctl.list_gpus(probe_bar0=True)
                continue
            if ch == 9:  # Tab
                self.tab_idx = (self.tab_idx + 1) % len(TABS)
                continue
            if ch == curses.KEY_BTAB:
                self.tab_idx = (self.tab_idx - 1) % len(TABS)
                continue
            if ch == ord("?"):
                info = STEP_HELP[step_key]
                _scroll_viewer_curses(
                    stdscr,
                    f"Help — {info['title']}",
                    [
                        "WHAT THIS STEP IS FOR:",
                        *textwrap.wrap(info["purpose"], 76),
                        "",
                        "WHEN TO USE IT:",
                        *textwrap.wrap(info["when"], 76),
                        "",
                        "BENCH TIP:",
                        *textwrap.wrap(info["tip"], 76),
                    ],
                    colors,
                )
                continue

            try:
                self._handle_key(stdscr, step_key, ch, ov, colors)
            except BurnError as exc:
                self.status = f"REFUSED: {exc}"
            except Exception as exc:
                self.status = f"ERROR: {exc}"

    def _render(self, stdscr, ov: dict, colors: dict[str, int]) -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        step_key, _ = TABS[self.tab_idx]

        # Row 0: Caution banner (preserved until bench sign-off)
        _safe_put(
            stdscr, 0, 0,
            " WARNING: UNTESTED ON PHYSICAL HARDWARE — VERIFY EVERYTHING BY HAND ".ljust(w),
            colors["caution"],
        )
        # Row 1: Persistent Status Bar
        badge = (ov.get("live_status") or {}).get("badge", "NO TARGET")
        plan_lbl = ov["plan"]["path"] if ov.get("plan") else "none"
        mode_s = "SIM" if ov["is_simulated"] else ("LIVE-ARMED" if ov["execute_live"] else "READ-ONLY")
        _safe_put(
            stdscr, 1, 0,
            f" Target: {ov['target_label']} | Card: {badge} | Plan: {plan_lbl} | Mode: {mode_s} ".ljust(w),
            colors["hdr"],
        )
        # Row 2: Tab Bar
        x = 1
        for i, (k, lbl) in enumerate(TABS):
            st = (ov.get("step_states") or {}).get(k, "")
            mark = "*" if st == "done" else ("!" if st == "warning" else " ")
            tok = f" [{lbl}{mark}] "
            attr = colors["tab_on"] if i == self.tab_idx else 0
            _safe_put(stdscr, 2, x, tok, attr)
            x += len(tok) + 1
        # Row 3: Advisor line
        _safe_put(stdscr, 3, 1, f"Next: {ov['headline']}", colors["ok"])
        # Row 4: Contextual Help Strip (2 compact lines)
        info = STEP_HELP[step_key]
        _safe_put(stdscr, 4, 1, f"Why : {info['purpose']}", curses.A_DIM)
        _safe_put(stdscr, 5, 1, f"When: {info['when']}", curses.A_DIM)
        _safe_put(stdscr, 6, 0, "-" * max(0, w - 1), curses.A_DIM)

        y = 7
        max_y = h - 3

        if step_key == "target":
            _safe_put(stdscr, y, 1, "Detected PCI GPUs (Up/Down + Enter to select):", curses.A_BOLD)
            y += 1
            if not self.gpus_cache:
                _safe_put(stdscr, y, 3, "(No PCI GPUs in sysfs — press [s] for Sim-PCIe or [f] for Fake snapshot)", curses.A_DIM)
                y += 1
            else:
                self.gpu_cursor = max(0, min(self.gpu_cursor, len(self.gpus_cache) - 1))
                for i, g in enumerate(self.gpus_cache):
                    if y >= max_y - 3:
                        break
                    sel_m = ">>" if i == self.gpu_cursor else "  "
                    cur_m = "*" if g["selected"] else " "
                    sm_s = f"{g['cur_sm']}/{g['max_sm']}SM" if g["cur_sm"] is not None else "--"
                    fbp_s = f"{g['cur_fbp']}/{g['max_fbp']}FBP" if g["cur_fbp"] is not None else "--"
                    line = f"{sel_m}[{cur_m}] {g['bdf']}  {g['pci_id']} {g['name'][:26]:26} drv={g['driver']:10} {sm_s} {fbp_s} {g['fuse_state'] or ''}"
                    _safe_put(stdscr, y, 1, line, colors["sel"] if i == self.gpu_cursor else 0)
                    y += 1
            y += 1
            if y < max_y:
                _safe_put(stdscr, y, 1, f"Journal : {ov['journal_path']}")
                y += 1
            if y < max_y:
                _safe_put(stdscr, y, 1, f"Receipt : {ov['receipt_path'] or 'none (auto-attached in Step 5)'}")
                y += 1
            if y < max_y and ov["driver_blockers"].get("needed"):
                _safe_put(stdscr, y, 1, "DRIVER BLOCKER: Press [d] to unload nvidia/nouveau & bind pci-stub!", colors["warn"])
            footer = " [Enter] Select GPU  [r] Rescan  [d] Detach Driver  [s] Sim-PCIe File  [f] Fake File  [c] Receipt File  [q] Quit "

        elif step_key == "configure":
            if ov.get("plan"):
                pl = ov["plan"]
                _safe_put(stdscr, y, 1, f"Active Plan: {pl['path']} ({pl['pci_id']} {pl['sku_name']}, {pl['record_count']} rec, sha={pl['sha12']})", colors["ok"])
                y += 1
                s = pl["summary"]
                summary_lines = [
                    f"  CUDA Cores / SMs : {s['cuda_cores_before']} -> {s['cuda_cores_after']} CUDA | {s['sm_before']} -> {s['sm_after']} SMs (max {s['sm_max_healthy']})",
                    f"  Est. VRAM        : {s['vram_label']}",
                    f"  Memory Bus Width : {s['bus_width_before']}-bit -> {s['bus_width_after']}-bit ({s['fbp_before']} -> {s['fbp_after']}/12 FBPs)",
                    f"  PCIe Link Speed  : {s['pcie_before']} -> {s['pcie_after']}",
                ]
                for sl in summary_lines:
                    if y < max_y:
                        _safe_put(stdscr, y, 1, sl)
                        y += 1
            else:
                _safe_put(stdscr, y, 1, "No plan loaded yet. Press [c] to run menuconfig on the card, or [o] to open a plan file.", colors["warn"])
            footer = " [c] Configure Unlocks (menuconfig)  [o] Open Plan (File Picker)  [v] Full Plan Review  [q] Quit "

        elif step_key == "preflight":
            watch_lbl = "ON (polling 1.5s)" if self.watch_active else "OFF"
            _safe_put(stdscr, y, 1, f"Live Goldilocks Watch: {watch_lbl}", colors["ok"] if self.watch_active else 0)
            y += 1
            ls = ov.get("live_status")
            if ls and ls.get("available"):
                hmap = ls["health"]
                _safe_put(stdscr, y, 1, f"State     : {ls['badge']} — {ls['detail']}", colors["ok"] if ls["badge"] in ("GOLDILOCKS", "ARMED") else colors["warn"])
                y += 1
                _safe_put(stdscr, y, 1, f"Registers : FUSECTRL={hmap['fusectrl']}  DSOV={hmap['dsov']}  CRC={hmap['crc']}  PLM_f8={hmap['plm_f8']} ({'OPEN' if hmap['plm_f8_open'] else 'CLOSED'})")
                y += 1
                _safe_put(stdscr, y, 1, f"Arming    : EFP={hmap['efp']}  EN_SW={hmap['en_sw']}  SBFE={hmap['sbfe']}  DEBUGCTRL={hmap['debugctrl']}  PLM_f4={hmap['plm_f4']}")
                y += 1
                issues = (ls.get("host_issues") or []) + (ls.get("health_issues") or [])
                if issues:
                    _safe_put(stdscr, y, 1, "Issues    : " + "; ".join(issues)[: max(20, w - 14)], colors["warn"])
                else:
                    _safe_put(stdscr, y, 1, "Preflight : ALL PASSIVE GATES PASS", colors["ok"])
            else:
                _safe_put(stdscr, y, 1, ov.get("live_error") or "Select a target in Step 1 first.", colors["warn"])
            footer = " [w] Toggle Live Watch  [r] Full Status Report  [d] Passive Dry-Run  [s] Active Sense (SENSE_CTRL)  [q] Quit "

        elif step_key == "burn":
            _safe_put(stdscr, y, 1, f"Confirm Cadence: [{ov['confirm_mode'].upper()}]   Mode: {ov['mode_label']}", colors["danger"] if ov["execute_live"] else colors["ok"])
            y += 1
            try:
                bp = self.ctl.get_burn_preview()
                _safe_put(stdscr, y, 1, f"Card Progress  : {bp['state'].upper()} ({bp['committed_count']}/{bp['total_count']} burned, next row={bp['next_row']})")
                y += 1
                if bp["needs_receipt"]:
                    _safe_put(stdscr, y, 1, "WARNING: Cold-verified test receipt required to resume — run Step 5 first!", colors["warn"])
                    y += 1
                y += 1
                for g in bp["groups"]:
                    if y >= max_y:
                        break
                    span = f"row {g['start_row']}" if g["count"] == 1 else f"rows {g['start_row']}..{g['end_row']}"
                    _safe_put(stdscr, y, 1, f"  GROUP [{g['token']}]  {g['title']} ({g['count']} rec, {span})", curses.A_BOLD)
                    y += 1
                    for rec in g["records"][:3]:
                        if y >= max_y:
                            break
                        _safe_put(stdscr, y, 4, f"row {rec['row']:3d}: {rec['target']}={rec['word']}  ({rec['decode'][:44]})")
                        y += 1
            except Exception as exc:
                _safe_put(stdscr, y, 1, f"Not ready: {exc}", colors["warn"])
            footer = " [m] Cycle Cadence (group/plan/row)  [a] Arm/Disarm Live Writes  [b] START BURN  [q] Quit "

        elif step_key == "cold_verify":
            st_lbl = self.cold_stage or "auto"
            _safe_put(stdscr, y, 1, f"Stage: [{st_lbl}]   (press [t] to cycle auto / test / real)")
            y += 1
            try:
                pf = self.ctl.preflight_cold_verify()
                if pf["ready"]:
                    _safe_put(stdscr, y, 1, f"READY: Cold boot confirmed (stage={pf['inferred_stage']}, boot changed).", colors["ok"])
                    y += 1
                else:
                    for prob in pf["problems"][:3]:
                        if y < max_y:
                            _safe_put(stdscr, y, 1, f"NOT READY: {prob}", colors["warn"])
                            y += 1
            except Exception as exc:
                _safe_put(stdscr, y, 1, f"Preflight: {exc}", colors["warn"])
                y += 1
            if self.last_cold_result is not None and y + 1 < max_y:
                res = self.last_cold_result
                y += 1
                tag = "PASS" if res["passed"] else f"FAIL: {res.get('error')}"
                _safe_put(stdscr, y, 1, f"Last Result ({res['stage']}): {tag}", colors["ok"] if res["passed"] else colors["danger"])
                y += 1
                if res.get("receipt_path") and y < max_y:
                    _safe_put(stdscr, y, 1, f"Receipt attached: {res['receipt_path']}", colors["ok"])
                    y += 1
                for row in (res.get("comparison") or []):
                    if y >= max_y:
                        break
                    mark = "OK  " if row["match"] else "DIFF"
                    _safe_put(stdscr, y, 2, f"[{mark}] {row['field']:20} exp={row['expected']:12} got={row['observed']}", colors["ok"] if row["match"] else colors["danger"])
                    y += 1
            footer = " [v] One-Step Capture & Verify  [o] Verify Snapshot File  [t] Cycle Stage  [q] Quit "

        else:  # tools
            ov_map = ov["overrides"]
            ov_keys = [
                ("skip_active_sense", "Skip automatic SENSE_CTRL"),
                ("no_test_receipt", "Allow real stage without test receipt"),
                ("mask_mismatch", "Skip initial expected_masks check"),
                ("undetermined_ceiling", "Override undetermined/exceeded SM ceiling"),
                ("baseline_mismatch", "Allow baseline FF prefix mismatch"),
                ("unmapped_ff", "Allow unknown/unmapped FF & missing mask"),
                ("allow_manual_debugctrl", "Manual-only: allow DEBUGCTRL target"),
            ]
            self.ov_cursor = max(0, min(self.ov_cursor, len(ov_keys) - 1))
            _safe_put(stdscr, y, 1, "Granular Manual Overrides (Up/Down + Space to toggle, x to reset):", curses.A_BOLD)
            y += 1
            for i, (k, lbl) in enumerate(ov_keys):
                if y >= max_y:
                    break
                mark = "*" if ov_map.get(k) else " "
                attr = colors["sel"] if i == self.ov_cursor else (colors["warn"] if ov_map.get(k) else 0)
                _safe_put(stdscr, y, 2, f"[{mark}] {lbl}", attr)
                y += 1
            footer = " [Space] Toggle Override  [x] Reset Overrides  [d] Decode FF  [e] Experimental Regs  [j] Journal  [m] Manual Console  [q] Quit "

        _safe_put(stdscr, h - 2, 0, f" {self.status} ", colors["warn"])
        _safe_put(stdscr, h - 1, 0, footer.ljust(w), colors["hdr"])
        stdscr.refresh()

    def _handle_key(self, stdscr, step_key: str, ch: int, ov: dict, colors: dict[str, int]) -> None:
        live_pci = (ov.get("live_status") or {}).get("pci_id")

        if step_key == "target":
            if ch in (curses.KEY_UP, ord("k")) and self.gpus_cache:
                self.gpu_cursor = max(0, self.gpu_cursor - 1)
            elif ch in (curses.KEY_DOWN, ord("j")) and self.gpus_cache:
                self.gpu_cursor = min(len(self.gpus_cache) - 1, self.gpu_cursor + 1)
            elif ch in (10, 13, curses.KEY_ENTER) and self.gpus_cache:
                bdf = self.gpus_cache[self.gpu_cursor]["bdf"]
                self.ctl.select_gpu(bdf)
                self.gpus_cache = self.ctl.list_gpus(probe_bar0=True)
                self.status = f"Selected live GPU {bdf} (Read-Only mode)."
            elif ch in (ord("r"), ord("R")):
                self.gpus_cache = self.ctl.list_gpus(probe_bar0=True)
                self.status = f"Rescanned PCI bus: {len(self.gpus_cache)} GPU(s) found."
            elif ch in (ord("d"), ord("D")):
                logs = self.ctl.detach_gpu_driver()
                self.gpus_cache = self.ctl.list_gpus(probe_bar0=True)
                self.status = logs[-1] if logs else "GPU detached and ready for BAR0."
            elif ch in (ord("s"), ord("S"), ord("f"), ord("F")):
                mode = "sim_pcie" if ch in (ord("s"), ord("S")) else "fake"
                picked = run_file_selector_curses(
                    stdscr, colors, title=f"Select {mode} Snapshot", kind_filter="fake_snapshot"
                )
                if picked:
                    self.ctl.select_simulation(picked, mode=mode)
                    self.status = f"Loaded {mode} snapshot: {picked}"
            elif ch in (ord("c"), ord("C")):
                picked = run_file_selector_curses(
                    stdscr, colors, title="Select Test Receipt", kind_filter="test_receipt"
                )
                if picked:
                    self.ctl.set_receipt_path(picked)
                    self.status = f"Attached receipt: {picked}"

        elif step_key == "configure":
            if ch in (ord("o"), ord("O")):
                picked = run_file_selector_curses(
                    stdscr, colors, title="Open Burn Plan JSON", kind_filter="plan", target_pci_id=live_pci
                )
                if picked:
                    res = self.ctl.load_plan(picked)
                    self.status = res.get("note") or f"Loaded plan {picked}."
            elif ch in (ord("v"), ord("V")):
                rev = self.ctl.get_plan_review()
                _scroll_viewer_curses(stdscr, f"Plan Review — {rev['path']}", rev["lines"], colors)
            elif ch in (ord("c"), ord("C")):
                self.ctl.start_configure(refresh=True)
                card = self.ctl._active_card
                assert card is not None
                curses.endwin()
                cfg = run_menuconfig(card, config=self.ctl._active_config, use_curses=True)
                curses.doupdate()
                if cfg is None:
                    self.status = "Unlock configuration cancelled."
                else:
                    self.ctl._active_config = cfg
                    default_out = self.ctl.default_plan_save_path()
                    out = _prompt_modal_curses(stdscr, "Save Mode-0600 Plan JSON", "Output plan path:", default=default_out)
                    if out:
                        res = self.ctl.save_configured_plan(out, overwrite=False)
                        if res.get("collision"):
                            ans = _prompt_modal_curses(
                                stdscr, "File Exists", f"{res['message']} Type OVERWRITE to replace:"
                            )
                            if ans == "OVERWRITE":
                                res = self.ctl.save_configured_plan(out, overwrite=True)
                            else:
                                self.status = "Save cancelled (existing file kept)."
                                return
                        self.status = f"Saved & loaded {res['path']} ({res['record_count']} records)."

        elif step_key == "preflight":
            if ch in (ord("w"), ord("W")):
                self.watch_active = not self.watch_active
                self.status = f"Live Goldilocks Watch {'ON' if self.watch_active else 'OFF'}."
            elif ch in (ord("r"), ord("R")):
                st = self.ctl.read_passive_status(include_full_report=True)
                lines = st.get("report_lines") or [f"State: {st.get('badge')} — {st.get('detail')}"]
                _scroll_viewer_curses(stdscr, "Passive Status Report", lines, colors)
            elif ch in (ord("d"), ord("D")):
                res = self.ctl.run_dry_run()
                _scroll_viewer_curses(stdscr, "Passive Dry-Run Preview", res["lines"], colors)
            elif ch in (ord("s"), ord("S")):
                ans = _prompt_modal_curses(
                    stdscr,
                    "Active Sense (SENSE_CTRL)",
                    "Restores EFP 0->1 while keeping DSOV=0 (1 per glitch window). Type 'SENSE':",
                )
                if ans is not None:
                    res = self.ctl.run_active_sense(ans)
                    self.status = "Active sense PASS — frame armed." if res["passed"] else f"Issues: {'; '.join(res['issues'])}"

        elif step_key == "burn":
            if ch in (ord("m"), ord("M")):
                modes = ("group", "plan", "row")
                nxt = modes[(modes.index(ov["confirm_mode"]) + 1) % len(modes)] if ov["confirm_mode"] in modes else "group"
                self.ctl.set_confirm_mode(nxt)
                self.status = f"Burn confirmation cadence set to '{nxt}'."
            elif ch in (ord("a"), ord("A")):
                if ov["is_simulated"]:
                    self.status = "Simulation mode does not require arming live writes."
                elif ov["execute_live"]:
                    self.ctl.set_execute_live(False)
                    self.status = "Live Execute turned OFF (safe Read-Only)."
                else:
                    ans = _prompt_modal_curses(
                        stdscr,
                        "Arm Live Hardware Writes",
                        "Type 'ENABLE LIVE WRITES' to arm physical CMD_PROGRAM:",
                    )
                    if ans is not None:
                        self.ctl.set_execute_live(True, ans)
                        self.status = "Live Execute ARMED."
            elif ch in (ord("b"), ord("B")):
                def _run_burn_in_term() -> None:
                    print("\n=== vfi_burn — Burn Session ===")
                    try:
                        res = self.ctl.execute_burn(
                            confirm_cb=lambda p, tok: input(p).strip() == tok,
                            show_cb=print,
                        )
                        print(f"\nDONE: {res}")
                        self.status = "Burn session finished (read-back verified). Cold-boot and run Step 5."
                    except Exception as exc:
                        print(f"\nSTOPPED: {exc}")
                        self.status = f"Burn stopped: {exc}"
                self._suspend_for_terminal(_run_burn_in_term)

        elif step_key == "cold_verify":
            if ch in (ord("t"), ord("T")):
                order = (None, "test", "real")
                self.cold_stage = order[(order.index(self.cold_stage) + 1) % len(order)]
                self.status = f"Cold verify stage set to {self.cold_stage or 'auto'}."
            elif ch in (ord("v"), ord("V")):
                res = self.ctl.run_cold_verify_live(stage=self.cold_stage)
                self.last_cold_result = res
                self.status = (
                    f"Cold Verify PASS ({res['stage']})!"
                    + (f" Receipt attached: {res['receipt_path']}" if res.get("receipt_path") else "")
                    if res["passed"]
                    else f"Cold Verify FAILED: {res.get('error')}"
                )
            elif ch in (ord("o"), ord("O")):
                picked = run_file_selector_curses(
                    stdscr, colors, title="Select Cold Snapshot JSON", kind_filter="cold_snapshot", target_pci_id=live_pci
                )
                if picked:
                    res = self.ctl.run_cold_verify_file(picked, stage=self.cold_stage)
                    self.last_cold_result = res
                    self.status = (
                        f"Cold Verify PASS ({res['stage']})!"
                        if res["passed"]
                        else f"Cold Verify FAILED: {res.get('error')}"
                    )

        elif step_key == "tools":
            ov_keys = [
                "skip_active_sense",
                "no_test_receipt",
                "mask_mismatch",
                "undetermined_ceiling",
                "baseline_mismatch",
                "unmapped_ff",
                "allow_manual_debugctrl",
            ]
            if ch in (curses.KEY_UP, ord("k")):
                self.ov_cursor = max(0, self.ov_cursor - 1)
            elif ch in (curses.KEY_DOWN, ord("j")):
                self.ov_cursor = min(len(ov_keys) - 1, self.ov_cursor + 1)
            elif ch == ord(" "):
                k = ov_keys[self.ov_cursor]
                cur_val = bool(ov["overrides"].get(k))
                self.ctl.update_overrides({k: not cur_val})
                self.status = f"Override '{k}' -> {'ON' if not cur_val else 'OFF'}."
            elif ch in (ord("x"), ord("X")):
                self.ctl.update_overrides({"reset": True})
                self.status = "All manual overrides reset to OFF."
            elif ch in (ord("d"), ord("D")):
                raw = _prompt_modal_curses(stdscr, "Decode 32-Bit FF Word", "Enter hex/dec FF word (e.g. 0x0fe805c1):")
                if raw:
                    d = self.ctl.decode_ff_word(raw)
                    lines = [
                        f"Word : {d['word']} (chain={d['chain']}, block={d['block']}, data={d['data']}, IFF=0)",
                        "",
                        "Overlapping GA100 option fields:",
                        *(
                            [f"  - {f['key']:24} ({f['name']}) bits {f['bits']} [{f['category']}]" for f in d["fields"]]
                            or ["  (none in known map)"]
                        ),
                    ]
                    _scroll_viewer_curses(stdscr, f"Decode {d['word']}", lines, colors)
            elif ch in (ord("e"), ord("E")):
                exp = self.ctl.read_experimental()
                _scroll_viewer_curses(stdscr, "Experimental Read-Only Registers", exp["lines"], colors)
            elif ch in (ord("j"), ord("J")):
                j = self.ctl.read_journal(limit=60)
                lines = [str(ev) for ev in j.get("recent_events") or []] or ["(empty journal)"]
                _scroll_viewer_curses(stdscr, f"Journal — {ov['journal_path']}", lines, colors)
            elif ch in (ord("m"), ord("M")):
                def _run_manual() -> None:
                    p = self.ctl._load_plan()
                    execute_live = (not self.ctl.is_simulated) and self.ctl.execute_live
                    backend, owned = self.ctl._open_backend(writable=execute_live)
                    try:
                        writer = backend if (self.ctl.is_simulated or execute_live) else None
                        from .backend import MissingWriter
                        with Journal(self.ctl.state.journal_path) as j:
                            ManualConsole(
                                p,
                                backend,
                                writer or MissingWriter(),
                                j,
                                execute=execute_live,
                                test_receipt=self.ctl.state.receipt_path,
                                overrides=self.ctl.overrides,
                                strict_confirm=execute_live,
                            ).run()
                    finally:
                        self.ctl._close_if_owned(backend, owned)
                self._suspend_for_terminal(_run_manual)


def run_tui(controller: WorkflowController, *, use_curses: bool = True) -> int:
    """Launch the full-screen stdlib curses TUI (or plain menu fallback)."""
    if not use_curses:
        return BurnMenu(
            MenuSession(
                plan_path=controller.state.plan_path,
                bdf=controller.state.bdf,
                adapter_spec=controller.state.adapter_spec,
                fake_path=controller.state.fake_path,
                journal_path=controller.state.journal_path,
                receipt_path=controller.state.receipt_path,
                execute_live=controller.execute_live,
                overrides=controller.overrides,
            )
        ).run()
    return BurnTUI(controller).run()
