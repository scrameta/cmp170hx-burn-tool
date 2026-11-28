"""Shared UI-agnostic workflow, session, file-catalog, and cold-verify controller.

Both the Local Web UI (``vfi_burn/web.py``) and the stdlib curses TUI
(``vfi_burn/tui.py``) sit directly on top of ``WorkflowController`` so every safety
gate, file-permission check, Before/After capability calculation, confirmation
cadence, and cold-boot verification rule is implemented once.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable

from .backend import (
    MissingWriter,
    fake_from_file,
    has_verified_writer,
    load_adapter,
    write_private_json,
)
from .console import (
    format_experimental_report,
    format_plan_review,
    format_status_report,
    read_experimental_registers,
)
from .engine import (
    CONFIRM_MODES,
    GOLDILOCKS_CRC,
    GOLDILOCKS_PLM,
    Overrides,
    build_approval_groups,
    check_host_readonly_safe,
    check_plan,
    cold_snapshot_controller_problems,
    cold_snapshot_warnings,
    detect_plan_progress,
    evaluate_health,
    evaluate_host,
    hex32,
    last_session_boot_id,
    lnkcap2_speeds,
    preflight,
    process_records,
    scan_slots,
    verify_cold_snapshot,
)
from .ffrecord import decode_ff, fields_in_block, replay_ff_rows
from .fuses import (
    BUS_BITS_PER_FBP,
    CMP_VBIOS_GB_PER_FBP,
    CUDA_CORES_PER_SM,
    TOTAL_FBP,
    a100_gb_per_fbp_for_sku,
)
from .ga100 import (
    GA100LiveAdapter,
    discover_pci_gpus,
    format_bytes_iec,
    inspect_gpu_driver_blockers,
    normalize_bdf,
    unload_gpu_drivers,
)
from .journal import Journal
from .mmio import resolve_register_address
from .model import (
    BurnError,
    Plan,
    is_poison,
    require_private_file,
    sku_display_name,
    u32,
)
from .planner import (
    CardSnapshot,
    UnlockConfig,
    build_plan_from_config,
    capture_card_snapshot,
    derive_ff_records,
    summarize_plan_before_after,
)
from .sim_pcie import sim_pcie_from_file


DEFAULT_SESSION_PATH = "private/.vfi_session.json"
DEFAULT_JOURNAL_PATH = "private/session.jsonl"

# Shared contextual help text ("What this step is for & when to run it") used by Web UI & TUI.
STEP_HELP: dict[str, dict[str, str]] = {
    "target": {
        "title": "1. Target GPU & Host Prep",
        "purpose": "Pick the physical GPU (PCI BDF) or an offline simulation snapshot, and detach nvidia/nouveau before touching BAR0.",
        "when": "Run this first on every fresh host boot, or whenever you switch cards. On a cold-booted card, detaching nvidia/nouveau lets you read fuses safely without glitching.",
        "tip": "Simulation mode (--fake / --sim-pcie) lets you rehearse the full configure -> burn -> cold-verify workflow with zero hardware risk.",
    },
    "configure": {
        "title": "2. Configure Unlocks & Review Plan",
        "purpose": "Read your card's factory FF rows and genuine defect masks read-only, choose which healthy Compute, Memory, PCIe, and feature blocks to enable, and save a mode-0600 burn plan.",
        "when": "Run once on a normal cold-booted card (no glitch needed) before burning. You can also open an already-saved plan JSON from disk if you are resuming after a reboot.",
        "tip": "Compute and Memory have independent Enable-All switches. Always inspect the Before/After summary (CUDA cores, VRAM, bus width, PCIe) and FF preview before saving.",
    },
    "preflight": {
        "title": "3. Glitch Watch & Preflight",
        "purpose": "Monitor live fuse-controller registers (DSOV, CRC=0x55, PLM_f8) while hunting your voltage glitch, run passive dry-runs, or trigger a standalone SENSE_CTRL check.",
        "when": "Run right after holding a Goldilocks glitch window and before burning. Enable Live Watch while tuning your glitcher so you see DSOV=0 / CRC=0x55 / PLM_f8 OPEN turn green immediately.",
        "tip": "Live Watch is strictly read-only and halts automatically on any 0xBADxxxxx poison read. Only one SENSE_CTRL is allowed per glitch window.",
    },
    "burn": {
        "title": "4. Burn Plan",
        "purpose": "Arm Live Writes and program the planned FF rows one CMD_PROGRAM at a time with mandatory dual read-back and post-write health verification on every single row.",
        "when": "Run only once Step 3 confirms a clean Goldilocks window. Group mode (recommended) burns the no-op test row first, lets you pause or continue, then approves one atomic group at a time.",
        "tip": "Never power off mid-group if the power-off sanity check warns of a split domain. On any read-back or health fault, vfi_burn halts immediately without retrying.",
    },
    "cold_verify": {
        "title": "5. Cold-Boot Verify",
        "purpose": "Prove that burned FF rows actually persisted across a full wall power-cycle and produced the exact expected SMs, FBPs, PCIe LnkCap2, and shadow registers.",
        "when": "Run after a full power-off / cold boot following either the no-op test row (generates & auto-attaches your test receipt so you can resume) or after finishing the full plan.",
        "tip": "This step checks boot_id and controller reset state automatically BEFORE saving, then captures, compares, and binds the test receipt in one action.",
    },
    "tools": {
        "title": "Tools & Diagnostics",
        "purpose": "Low-level read-only diagnostics (32-bit FF word decoder, experimental GA100 registers, FF row/register reader, journal viewer) and granular manual overrides.",
        "when": "Use anytime you want to inspect raw silicon registers, decode a community FF word, review your audit journal, or enable an explicit bench override.",
        "tip": "Hard brick-prevention invariants (aligned 32-bit stores, genuine defect masks, Die ID match, blank-slot checks, dual read-back) can never be overridden.",
    },
}


def _host_boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


@dataclass
class WorkspaceState:
    """Persisted operator workspace (mode 0600). Never persists ``execute_live=True``."""

    bdf: str | None = None
    fake_path: str | None = None
    sim_pcie_path: str | None = None
    adapter_spec: str | None = None
    plan_path: str | None = None
    journal_path: str = DEFAULT_JOURNAL_PATH
    receipt_path: str | None = None
    confirm_mode: str = "group"

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "bdf": self.bdf,
            "fake_path": self.fake_path,
            "sim_pcie_path": self.sim_pcie_path,
            "adapter_spec": self.adapter_spec,
            "plan_path": self.plan_path,
            "journal_path": self.journal_path,
            "receipt_path": self.receipt_path,
            "confirm_mode": self.confirm_mode,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SESSION_PATH) -> "WorkspaceState":
        p = Path(path)
        if not p.exists() or p.is_symlink():
            return cls()
        try:
            st = p.stat()
            if st.st_mode & 0o077:
                return cls()
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cls()
            cm = str(data.get("confirm_mode") or "group")
            if cm not in CONFIRM_MODES:
                cm = "group"
            return cls(
                bdf=data.get("bdf") or None,
                fake_path=data.get("fake_path") or None,
                sim_pcie_path=data.get("sim_pcie_path") or None,
                adapter_spec=data.get("adapter_spec") or None,
                plan_path=data.get("plan_path") or None,
                journal_path=str(data.get("journal_path") or DEFAULT_JOURNAL_PATH),
                receipt_path=data.get("receipt_path") or None,
                confirm_mode=cm,
            )
        except Exception:
            return cls()

    def save(self, path: str | Path = DEFAULT_SESSION_PATH) -> None:
        try:
            write_private_json(path, self.to_dict(), overwrite=True)
        except Exception:
            # Workspace persistence is best-effort (never blocks an explicit CLI/test run).
            pass


# ---------------------------------------------------------------------------
# File catalog & safe JSON metadata preview (shared by Web & Curses pickers)
# ---------------------------------------------------------------------------


def _classify_json_file(
    p: Path,
    *,
    target_pci_id: str | None = None,
    target_die_rows: dict[int, int] | None = None,
) -> tuple[str, str, str | None, dict]:
    """Peek into a JSON file safely (without mutating anything) and describe its type."""
    try:
        st = p.stat()
        if st.st_size > 4 * 1024 * 1024:
            return "json", f"JSON ({format_bytes_iec(st.st_size)})", None, {}
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return "json", f"Unreadable JSON ({exc})", None, {}
    if not isinstance(raw, dict):
        return "json", "JSON data", None, {}

    # 1. vfi_burn Plan
    if raw.get("schema_version") == 1 and isinstance(raw.get("card"), dict) and isinstance(raw.get("records"), list):
        card = raw["card"]
        pci = str(card.get("pci_id", "")).lower()
        n_rec = len(raw["records"])
        meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        cfg_sum = meta.get("configured_summary") or ""
        sku = sku_display_name(pci) if pci else "unknown SKU"
        sku_match: str | None = None
        if target_pci_id and pci:
            sku_match = "match" if pci == target_pci_id.lower() else "mismatch"
            if sku_match == "match" and target_die_rows and isinstance(card.get("die_rows"), dict):
                try:
                    file_die = {int(k): u32(v, "die") for k, v in card["die_rows"].items()}
                    if file_die != target_die_rows:
                        sku_match = "die_mismatch"
                except Exception:
                    pass
        summary = f"Plan · {pci} ({sku}) · {n_rec} rec(s)"
        details = {
            "pci_id": pci,
            "sku_name": sku,
            "record_count": n_rec,
            "configured_summary": cfg_sum,
            "created_by": meta.get("source_format") or "",
        }
        return "plan", summary, sku_match, details

    # 2. Test/Real Cold-Verify Receipt
    if raw.get("status") in ("PASS", "FAIL") and raw.get("stage") in ("test", "real") and raw.get("plan_sha256"):
        stg = raw.get("stage")
        sha = str(raw.get("plan_sha256", ""))[:12]
        row = raw.get("test_row")
        summary = f"Receipt ({stg.upper()} {raw.get('status')}) · plan {sha}"
        if row is not None:
            summary += f" · row {row}"
        return "test_receipt", summary, None, {
            "stage": stg,
            "status": raw.get("status"),
            "plan_sha256": sha,
            "test_row": row,
            "boot_id": raw.get("boot_id"),
        }

    # 3. Cold-Boot Snapshot
    if raw.get("cold_boot") is True and "boot_id" in raw and isinstance(raw.get("fields"), dict):
        pci = str(raw.get("pci_id", "")).lower()
        sm = raw["fields"].get("SM_COUNT")
        sku = sku_display_name(pci) if pci else "unknown"
        sku_match = None
        if target_pci_id and pci:
            sku_match = "match" if pci == target_pci_id.lower() else "mismatch"
        summary = f"Cold Snapshot · {pci} ({sku})"
        if sm is not None:
            summary += f" · SM={sm}"
        return "cold_snapshot", summary, sku_match, {
            "pci_id": pci,
            "sku_name": sku,
            "boot_id": raw.get("boot_id"),
            "previous_boot_id": raw.get("previous_boot_id"),
            "sm_count": sm,
            "row_count": len(raw.get("fpf_rows") or {}),
        }

    # 4. Fake / Simulated PCIe Snapshot
    if "pci_id" in raw and ("rows" in raw or "fields" in raw or "health" in raw):
        pci = str(raw.get("pci_id", "")).lower()
        sku = sku_display_name(pci) if pci else "unknown"
        h = raw.get("health") if isinstance(raw.get("health"), dict) else {}
        dsov = h.get("dsov")
        state = "GOLDILOCKS" if dsov == 0 else ("COLD" if dsov == 1 else "SIM")
        summary = f"Sim Snapshot ({state}) · {pci} ({sku})"
        return "fake_snapshot", summary, None, {
            "pci_id": pci,
            "sku_name": sku,
            "state": state,
            "row_count": len(raw.get("rows") or {}),
        }

    return "json", "JSON object", None, {"keys": list(raw.keys())[:8]}


def inspect_file_metadata(
    path: str | Path,
    *,
    target_pci_id: str | None = None,
    target_die_rows: dict[int, int] | None = None,
) -> dict:
    """Return UI-ready metadata for a directory or file entry."""
    p = Path(path)
    is_sym = p.is_symlink()
    try:
        st = p.lstat() if is_sym else p.stat()
    except OSError as exc:
        return {
            "name": p.name,
            "path": str(p),
            "is_dir": False,
            "is_symlink": is_sym,
            "kind": "error",
            "summary": f"Unaccessible ({exc})",
            "is_private_0600": False,
            "mode_octal": "0000",
            "size": 0,
            "mtime": 0,
            "sku_match": None,
            "details": {},
        }
    is_dir = p.is_dir() and not is_sym
    mode = st.st_mode & 0o777
    is_0600 = (not is_sym) and (not is_dir) and ((st.st_mode & 0o077) == 0)
    kind = "dir" if is_dir else "file"
    summary = "Directory" if is_dir else f"{format_bytes_iec(st.st_size)}"
    sku_match: str | None = None
    details: dict = {}

    if is_sym:
        kind = "symlink"
        summary = "Symlink (refused for private files)"
    elif not is_dir:
        suf = p.suffix.lower()
        if suf == ".jsonl":
            kind = "journal"
            try:
                cnt = sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
                summary = f"Audit Journal · {cnt} event(s)"
                details = {"event_count": cnt}
            except OSError:
                summary = "Audit Journal"
        elif suf in (".json", ".receipt"):
            kind, summary, sku_match, details = _classify_json_file(
                p, target_pci_id=target_pci_id, target_die_rows=target_die_rows
            )

    return {
        "name": p.name,
        "path": str(p),
        "is_dir": is_dir,
        "is_symlink": is_sym,
        "kind": kind,
        "summary": summary,
        "is_private_0600": is_0600,
        "mode_octal": f"{mode:04o}",
        "size": st.st_size,
        "size_label": format_bytes_iec(st.st_size) if not is_dir else "",
        "mtime": int(st.st_mtime),
        "mtime_label": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "sku_match": sku_match,
        "details": details,
    }


def list_directory_entries(
    directory: str | Path = "private",
    *,
    kind_filter: str | None = None,
    target_pci_id: str | None = None,
    target_die_rows: dict[int, int] | None = None,
) -> dict:
    """List browsable entries in ``directory`` with metadata for Web & Curses file selectors."""
    base = Path(directory)
    if not base.exists():
        # Fall back gracefully to CWD if private/ doesn't exist yet.
        base = Path.cwd()
    base = base.resolve()
    parent = base.parent if base.parent != base else None

    dirs: list[dict] = []
    files: list[dict] = []
    try:
        children = list(base.iterdir())
    except OSError as exc:
        raise BurnError(f"cannot read directory {base}: {exc}") from exc

    for child in children:
        # Hide internal session file and temporary atomic-write files from picker clutter.
        if child.name.startswith(".vfi_session") or (child.name.startswith(".") and child.name.endswith(".tmp")):
            continue
        meta = inspect_file_metadata(
            child, target_pci_id=target_pci_id, target_die_rows=target_die_rows
        )
        if meta["is_dir"]:
            dirs.append(meta)
        else:
            if kind_filter and kind_filter != "all":
                allowed = {k.strip() for k in kind_filter.split(",") if k.strip()}
                if meta["kind"] not in allowed:
                    continue
            files.append(meta)

    dirs.sort(key=lambda d: d["name"].lower())
    files.sort(key=lambda f: (-f["mtime"], f["name"].lower()))

    entries: list[dict] = []
    if parent is not None:
        entries.append({
            "name": "..",
            "path": str(parent),
            "is_dir": True,
            "is_symlink": False,
            "kind": "parent",
            "summary": f"Parent directory ({parent})",
            "is_private_0600": False,
            "mode_octal": "",
            "size": 0,
            "size_label": "",
            "mtime": 0,
            "mtime_label": "",
            "sku_match": None,
            "details": {},
        })
    entries.extend(dirs)
    entries.extend(files)
    return {
        "directory": str(base),
        "parent": str(parent) if parent is not None else None,
        "entries": entries,
    }


def fix_file_permissions(path: str | Path) -> dict:
    """Safely enforce mode 0600 on a private file (refusing symlinks and directories)."""
    p = Path(path)
    if p.is_symlink():
        raise BurnError(f"refusing to chmod symlink: {p}")
    if not p.is_file():
        raise BurnError(f"not a regular file: {p}")
    os.chmod(p, 0o600)
    return inspect_file_metadata(p)


# ---------------------------------------------------------------------------
# Journal analysis & Cold-Verify diff helpers
# ---------------------------------------------------------------------------


def summarize_journal(
    journal_path: str | Path | None,
    *,
    plan_sha256: str | None = None,
    limit: int = 30,
) -> dict:
    """Parse an audit JSONL journal into structured lifecycle & recent-event summaries."""
    result = {
        "exists": False,
        "total_events": 0,
        "has_unresolved_halt": False,
        "has_unresolved_attempt": False,
        "last_session_boot_id": None,
        "last_burn_boot_id": None,
        "last_burn_stage": None,
        "burned_for_plan": 0,
        "test_burned_for_plan": False,
        "real_burned_for_plan": False,
        "cold_verified_test": False,
        "cold_verified_real": False,
        "recent_events": [],
    }
    if not journal_path:
        return result
    p = Path(journal_path)
    if not p.exists():
        return result
    result["exists"] = True
    try:
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return result

    parsed: list[dict] = []
    pending_attempt: dict | None = None
    burned_rows: set[int] = set()
    for ln in lines:
        try:
            item = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        parsed.append(item)
        ev = item.get("event")
        if ev == "HALT_UNVERIFIED":
            result["has_unresolved_halt"] = True
        elif ev == "attempt":
            pending_attempt = item
        elif ev in ("aborted_pre_trigger", "verified_readback_only"):
            pending_attempt = None

        if ev == "session_start" and item.get("boot_id"):
            result["last_session_boot_id"] = str(item["boot_id"])
            if not item.get("dry_run"):
                if plan_sha256 is None or item.get("plan_sha256") == plan_sha256:
                    result["last_burn_boot_id"] = str(item["boot_id"])
                    result["last_burn_stage"] = item.get("stage")
        elif ev == "verified_readback_only":
            if item.get("boot_id"):
                result["last_burn_boot_id"] = str(item["boot_id"])
            if plan_sha256 and item.get("plan_sha256") == plan_sha256 and item.get("manual_record_sha256") is None:
                row = item.get("row")
                if isinstance(row, int) and not isinstance(row, bool):
                    burned_rows.add(row)
                rec_name = str(item.get("record") or "")
                if "noop" in rec_name.lower() or "test" in rec_name.lower() or len(burned_rows) == 1:
                    result["test_burned_for_plan"] = True
                if len(burned_rows) > 1:
                    result["real_burned_for_plan"] = True
        elif ev == "cold_snapshot_comparison" and item.get("status") == "PASS":
            if plan_sha256 is None or item.get("plan_sha256") == plan_sha256:
                if item.get("stage") == "test":
                    result["cold_verified_test"] = True
                elif item.get("stage") == "real":
                    result["cold_verified_real"] = True

    result["total_events"] = len(parsed)
    result["has_unresolved_attempt"] = pending_attempt is not None
    result["burned_for_plan"] = len(burned_rows)
    result["recent_events"] = parsed[-limit:][::-1]
    return result


def build_cold_comparison_table(plan: Plan, cold: dict, *, stage: str) -> list[dict]:
    """Build a field-by-field Expected vs Observed table for Cold-Boot Verify UIs."""
    expected: dict[str, int] = {}
    records = plan.records[:1] if stage == "test" else plan.records
    for r in records:
        expected.update(r.expected_cold)
    if stage == "real":
        expected.update(plan.final_expected_cold)
    observed = cold.get("fields", {}) if isinstance(cold.get("fields"), dict) else {}

    rows: list[dict] = []
    for field_name, exp_val in sorted(expected.items()):
        if field_name not in observed:
            rows.append({
                "field": field_name,
                "expected": hex32(exp_val) if field_name != "SM_COUNT" else str(exp_val),
                "observed": "MISSING",
                "match": False,
            })
            continue
        try:
            got_val = u32(observed[field_name], f"cold.fields.{field_name}")
        except BurnError:
            rows.append({
                "field": field_name,
                "expected": hex32(exp_val),
                "observed": str(observed[field_name]),
                "match": False,
            })
            continue
        if field_name == "LnkCap2":
            match = lnkcap2_speeds(got_val) == lnkcap2_speeds(exp_val)
        else:
            match = got_val == exp_val
        rows.append({
            "field": field_name,
            "expected": str(exp_val) if field_name == "SM_COUNT" else hex32(exp_val),
            "observed": str(got_val) if field_name == "SM_COUNT" else hex32(got_val),
            "match": match,
        })
    return rows


# ---------------------------------------------------------------------------
# Unified WorkflowController (used by Web UI and Curses TUI)
# ---------------------------------------------------------------------------


class WorkflowController:
    """Single source of truth for vfi_burn interactive sessions (Web UI & Curses TUI)."""

    def __init__(
        self,
        *,
        session_path: str | Path | None = DEFAULT_SESSION_PATH,
        initial_state: WorkspaceState | None = None,
        sysfs_root: str | Path = "/sys/bus/pci/devices",
        check_tty: bool = False,
        persist_session: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._session_path = Path(session_path) if session_path else None
        self._persist = persist_session and (self._session_path is not None)
        self._sysfs_root = sysfs_root
        self.check_tty = check_tty

        if initial_state is not None:
            self.state = initial_state
        elif self._session_path is not None:
            self.state = WorkspaceState.load(self._session_path)
        else:
            self.state = WorkspaceState()

        # Runtime-only state (never persisted across restarts).
        self.execute_live: bool = False
        self.overrides: Overrides = Overrides()
        self._cached_sim = None
        self._cached_sim_key: tuple[str, str] | None = None
        self._active_card: CardSnapshot | None = None
        self._active_config: UnlockConfig | None = None

    def _save_state(self) -> None:
        if self._persist and self._session_path is not None:
            self.state.save(self._session_path)

    # ------------------------------------------------------------------ Target helpers
    @property
    def is_simulated(self) -> bool:
        return bool(self.state.fake_path or self.state.sim_pcie_path)

    @property
    def has_target(self) -> bool:
        return bool(
            self.state.bdf
            or self.state.fake_path
            or self.state.sim_pcie_path
            or self.state.adapter_spec
        )

    def target_label(self) -> str:
        if self.state.sim_pcie_path:
            return f"SIM-PCIe ({self.state.sim_pcie_path})"
        if self.state.fake_path:
            return f"FAKE ({self.state.fake_path})"
        if self.state.bdf:
            return f"LIVE {self.state.bdf}"
        if self.state.adapter_spec:
            return f"ADAPTER ({self.state.adapter_spec})"
        return "not selected"

    def mode_label(self) -> str:
        if self.is_simulated:
            return "SIMULATION (no GPU touched)"
        if self.execute_live:
            return "LIVE ARMED (permanent writes enabled behind confirmations)"
        return "LIVE READ-ONLY / DRY-RUN (safe default)"

    def _open_backend(self, *, writable: bool = False):
        if self.state.sim_pcie_path:
            key = ("sim_pcie", self.state.sim_pcie_path)
            if self._cached_sim is None or self._cached_sim_key != key:
                self._cached_sim = sim_pcie_from_file(self.state.sim_pcie_path)
                self._cached_sim_key = key
            return self._cached_sim, False
        if self.state.fake_path:
            key = ("fake", self.state.fake_path)
            if self._cached_sim is None or self._cached_sim_key != key:
                self._cached_sim = fake_from_file(self.state.fake_path)
                self._cached_sim_key = key
            return self._cached_sim, False
        if self.state.bdf:
            return GA100LiveAdapter(self.state.bdf, writable=writable), True
        if self.state.adapter_spec:
            return load_adapter(self.state.adapter_spec), True
        raise BurnError("no target selected yet — pick a GPU or simulation snapshot in Step 1 (Target)")

    @staticmethod
    def _close_if_owned(backend, owned: bool) -> None:
        if owned and backend is not None and callable(getattr(backend, "close", None)):
            backend.close()

    def _load_plan(self, *, for_review: bool = False) -> Plan:
        if not self.state.plan_path:
            raise BurnError("no plan loaded yet — create one in Step 2 (Configure) or open an existing plan JSON")
        p = Plan.load(self.state.plan_path)
        check_plan(
            p,
            allow_unmapped_ff=(for_review or self.overrides.unmapped_ff),
            allow_undetermined_ceiling=self.overrides.undetermined_ceiling,
        )
        return p

    def _bdf_sysfs_pci_id(self, bdf: str | None) -> str | None:
        if not bdf:
            return None
        try:
            dev_dir = Path(self._sysfs_root) / bdf
            v = int((dev_dir / "vendor").read_text(encoding="utf-8").strip(), 0)
            d = int((dev_dir / "device").read_text(encoding="utf-8").strip(), 0)
            return f"{v:04x}:{d:04x}"
        except (OSError, ValueError):
            return None

    # ------------------------------------------------------------------ Step 1: Target & Host
    def list_gpus(self, *, probe_bar0: bool = True) -> list[dict]:
        with self._lock:
            gpus = discover_pci_gpus(sysfs_root=self._sysfs_root, probe_bar0=probe_bar0)
            out: list[dict] = []
            for g in gpus:
                a100_gb = a100_gb_per_fbp_for_sku(g.pci_id)
                out.append({
                    "bdf": g.bdf,
                    "pci_id": g.pci_id,
                    "name": g.name,
                    "is_cmp170hx": g.is_cmp170hx,
                    "driver": g.driver,
                    "bar0_size": format_bytes_iec(g.bar0_bytes),
                    "bar1_size": format_bytes_iec(g.bar1_bytes),
                    "link_current": f"{g.cur_speed or '?'} x{g.cur_width or '?'}",
                    "link_max": f"{g.max_speed or '?'} x{g.max_width or '?'}",
                    "cur_sm": g.cur_sm,
                    "max_sm": g.max_sm,
                    "cur_cuda": g.cur_sm * CUDA_CORES_PER_SM if g.cur_sm is not None else None,
                    "max_cuda": g.max_sm * CUDA_CORES_PER_SM if g.max_sm is not None else None,
                    "cur_fbp": g.cur_fbp,
                    "max_fbp": g.max_fbp,
                    "cur_bus_bits": g.cur_fbp * BUS_BITS_PER_FBP if g.cur_fbp is not None else None,
                    "cur_vram_cmp_gb": g.cur_fbp * CMP_VBIOS_GB_PER_FBP if g.cur_fbp is not None else None,
                    "cur_vram_a100_gb": (g.cur_fbp * a100_gb) if (g.cur_fbp is not None and a100_gb) else None,
                    "fuse_state": g.fuse_state,
                    "probe_note": g.probe_note,
                    "selected": (not self.is_simulated) and (self.state.bdf == g.bdf),
                })
            return out

    def select_gpu(self, bdf: str) -> dict:
        with self._lock:
            norm = normalize_bdf(bdf)
            self.state.bdf = norm
            self.state.fake_path = None
            self.state.sim_pcie_path = None
            self.state.adapter_spec = None
            self._cached_sim = None
            self._cached_sim_key = None
            self._active_card = None
            self._active_config = None
            self.execute_live = False  # always disarm on target switch
            self._save_state()
            return self.get_overview()

    def select_simulation(self, path: str, *, mode: str = "sim_pcie") -> dict:
        with self._lock:
            p = str(require_private_file(path, "simulation snapshot"))
            if mode == "fake":
                ctl = fake_from_file(p)
                self.state.fake_path = p
                self.state.sim_pcie_path = None
                self._cached_sim = ctl
                self._cached_sim_key = ("fake", p)
            else:
                ctl = sim_pcie_from_file(p)
                self.state.sim_pcie_path = p
                self.state.fake_path = None
                self._cached_sim = ctl
                self._cached_sim_key = ("sim_pcie", p)
            self.state.bdf = None
            self.state.adapter_spec = None
            self._active_card = None
            self._active_config = None
            self.execute_live = False
            self._save_state()
            return self.get_overview()

    def inspect_driver_blockers(self) -> dict:
        with self._lock:
            if not self.state.bdf or self.is_simulated:
                return {"needed": False, "bound_driver": None, "loaded_modules": []}
            bound, loaded = inspect_gpu_driver_blockers(
                self.state.bdf, sysfs_root=self._sysfs_root
            )
            needed = (bound not in (None, "pci-stub")) or bool(loaded)
            return {
                "needed": needed,
                "bdf": self.state.bdf,
                "bound_driver": bound,
                "loaded_modules": loaded,
            }

    def detach_gpu_driver(self) -> list[str]:
        with self._lock:
            if not self.state.bdf or self.is_simulated:
                raise BurnError("select a live GPU BDF first")
            logs: list[str] = []
            unload_gpu_drivers(
                self.state.bdf,
                show=logs.append,
                sysfs_root=self._sysfs_root,
            )
            return logs

    def set_execute_live(self, enabled: bool, confirm_phrase: str = "") -> dict:
        with self._lock:
            if not enabled:
                self.execute_live = False
                return {"execute_live": False, "mode_label": self.mode_label()}
            if not self.is_simulated and confirm_phrase.strip() != "ENABLE LIVE WRITES":
                raise BurnError("type 'ENABLE LIVE WRITES' to arm physical fuse writes")
            self.execute_live = True
            return {"execute_live": True, "mode_label": self.mode_label()}

    def set_confirm_mode(self, mode: str) -> dict:
        with self._lock:
            m = str(mode).strip().lower()
            if m not in CONFIRM_MODES:
                raise BurnError(f"confirm_mode must be one of {', '.join(CONFIRM_MODES)}")
            self.state.confirm_mode = m
            self._save_state()
            return {"confirm_mode": m}

    def set_journal_path(self, path: str) -> dict:
        with self._lock:
            cleaned = str(path).strip()
            if not cleaned:
                raise BurnError("journal path must be nonempty")
            self.state.journal_path = cleaned
            self._save_state()
            return {"journal_path": cleaned}

    def set_receipt_path(self, path: str | None) -> dict:
        with self._lock:
            if not path:
                self.state.receipt_path = None
            else:
                require_private_file(path, "test receipt")
                self.state.receipt_path = str(path)
            self._save_state()
            return {"receipt_path": self.state.receipt_path}

    def update_overrides(self, patch: dict) -> dict:
        with self._lock:
            ov = self.overrides
            if patch.get("reset"):
                self.overrides = Overrides()
            else:
                kwargs: dict = {}
                if "slot" in patch:
                    raw = patch["slot"]
                    kwargs["slot"] = None if raw in (None, "", "-") else int(str(raw), 0)
                for key in (
                    "skip_active_sense",
                    "no_test_receipt",
                    "mask_mismatch",
                    "undetermined_ceiling",
                    "allow_manual_debugctrl",
                    "baseline_mismatch",
                    "unmapped_ff",
                ):
                    if key in patch:
                        kwargs[key] = bool(patch[key])
                self.overrides = replace(ov, **kwargs)
            return self.overrides_dict()

    def overrides_dict(self) -> dict:
        ov = self.overrides
        return {
            "slot": ov.slot,
            "skip_active_sense": ov.skip_active_sense,
            "no_test_receipt": ov.no_test_receipt,
            "mask_mismatch": ov.mask_mismatch,
            "undetermined_ceiling": ov.undetermined_ceiling,
            "allow_manual_debugctrl": ov.allow_manual_debugctrl,
            "baseline_mismatch": ov.baseline_mismatch,
            "unmapped_ff": ov.unmapped_ff,
            "active_names": ov.active_names(),
        }

    # ------------------------------------------------------------------ Step 2: Configure & Plan
    def load_plan(self, path: str) -> dict:
        with self._lock:
            p = Plan.load(path)
            check_plan(
                p,
                allow_unmapped_ff=True,
                allow_undetermined_ceiling=self.overrides.undetermined_ceiling,
            )
            note: str | None = None
            try:
                check_plan(
                    p,
                    allow_unmapped_ff=False,
                    allow_undetermined_ceiling=self.overrides.undetermined_ceiling,
                )
            except BurnError as exc:
                note = (
                    f"Plan contains opt-in/override-gated record(s) ({exc}); "
                    "burning requires enabling the 'unmapped_ff' override."
                )
            self.state.plan_path = str(path)
            self._save_state()
            return {
                "plan_path": self.state.plan_path,
                "note": note,
                "review": self.get_plan_review(),
            }

    def get_plan_review(self) -> dict:
        with self._lock:
            p = self._load_plan(for_review=True)
            backend = None
            owned = False
            if self.has_target:
                try:
                    backend, owned = self._open_backend(writable=False)
                except Exception:
                    backend, owned = None, False
            try:
                lines = format_plan_review(
                    p,
                    reader=backend,
                    override_slot=self.overrides.slot,
                    allow_undetermined_ceiling=self.overrides.undetermined_ceiling,
                )
            finally:
                self._close_if_owned(backend, owned)
            summary = summarize_plan_before_after(p)
            base_blocks = replay_ff_rows(p.baseline_ff_rows) if p.baseline_ff_rows is not None else None
            start_r = p.expected_next_row if p.expected_next_row is not None else p.first_row
            groups = build_approval_groups(list(p.records), start_r, base_blocks, is_live=not self.is_simulated)
            return {
                "path": self.state.plan_path,
                "pci_id": p.pci_id,
                "sku_name": sku_display_name(p.pci_id),
                "sha256": p.content_hash,
                "sha12": p.content_hash[:12],
                "record_count": len(p.records),
                "summary": summary.to_dict(),
                "summary_lines": summary.format_lines(),
                "groups": [g.to_dict() for g in groups],
                "lines": lines,
            }

    def default_plan_save_path(self) -> str:
        if self._active_card is not None:
            if self.state.plan_path:
                try:
                    existing = Plan.load(self.state.plan_path)
                    if existing.pci_id == self._active_card.pci_id and existing.die_rows == self._active_card.die_rows:
                        return self.state.plan_path
                except Exception:
                    pass
            dev_slug = self._active_card.pci_id.split(":")[-1].lower()
            return f"private/plan_{dev_slug}.json"
        return self.state.plan_path or "private/plan.json"

    def start_configure(self, *, refresh: bool = True) -> dict:
        with self._lock:
            if self._active_config is None or refresh:
                backend, owned = self._open_backend(writable=False)
                try:
                    check_host_readonly_safe(backend.host())
                    self._active_card = capture_card_snapshot(backend, label=self.target_label())
                    self._active_config = UnlockConfig.from_card(self._active_card)
                finally:
                    self._close_if_owned(backend, owned)
            return self.get_config_state()

    def get_config_state(self) -> dict:
        with self._lock:
            if self._active_config is None or self._active_card is None:
                return {"loaded": False}
            data = self._active_config.to_dict()
            data["loaded"] = True
            data["default_save_path"] = self.default_plan_save_path()
            return data

    def patch_config(self, patch: dict) -> dict:
        with self._lock:
            if self._active_config is None:
                self.start_configure(refresh=True)
            assert self._active_config is not None
            msg = self._active_config.apply_patch(patch)
            state = self.get_config_state()
            state["message"] = msg
            return state

    def save_configured_plan(self, out_path: str | None = None, *, overwrite: bool = False) -> dict:
        with self._lock:
            if self._active_card is None or self._active_config is None:
                raise BurnError("no active configuration loaded — read card state first")
            target_path = (out_path or self.default_plan_save_path()).strip()
            if not target_path:
                raise BurnError("output plan path must be nonempty")
            p = Path(target_path)
            if p.exists() and not overwrite:
                same_card = None
                try:
                    ex = Plan.load(p)
                    same_card = (
                        ex.pci_id == self._active_card.pci_id
                        and ex.die_rows == self._active_card.die_rows
                    )
                except Exception:
                    same_card = None
                return {
                    "saved": False,
                    "collision": True,
                    "path": target_path,
                    "same_card": same_card,
                    "message": (
                        f"File {target_path} already exists "
                        + (
                            "(earlier plan for THIS SAME card)."
                            if same_card is True
                            else ("(WARNING: plan for a DIFFERENT card!)." if same_card is False else ".")
                        )
                    ),
                }
            plan_dict = build_plan_from_config(self._active_card, self._active_config)
            write_private_json(target_path, plan_dict, overwrite=overwrite)
            loaded = Plan.load(target_path)
            check_plan(loaded, allow_unmapped_ff=True, allow_undetermined_ceiling=self.overrides.undetermined_ceiling)
            self.state.plan_path = target_path
            self._save_state()
            return {
                "saved": True,
                "collision": False,
                "path": target_path,
                "sha12": loaded.content_hash[:12],
                "record_count": len(loaded.records),
                "summary": summarize_plan_before_after(loaded).to_dict(),
            }

    # ------------------------------------------------------------------ Step 3: Preflight & Glitch Watch
    def _classify_health_badge(self, h) -> tuple[str, str]:
        for key in ("fusectrl", "dsov", "crc", "plm", "plm_f8", "en_sw", "debugctrl"):
            if is_poison(getattr(h, key)):
                return "POISON", f"{key}={hex32(getattr(h, key))}"
        if h.dsov == 0 and h.crc == GOLDILOCKS_CRC and h.plm == GOLDILOCKS_PLM and h.plm_f8 == GOLDILOCKS_PLM:
            if h.efp == 1 and h.en_sw == 1 and (h.sbfe in (None, 0)):
                return "ARMED", "DSOV=0 CRC=0x55 EFP=1 PLM_f8=OPEN (Armed)"
            return "GOLDILOCKS", "DSOV=0 CRC=0x55 PLM_f8=OPEN (Goldilocks)"
        if h.dsov == 1:
            return "COLD", f"Normal cold-boot state (DSOV=1, CRC={hex32(h.crc)})"
        return "DEGRADED", f"DSOV={h.dsov} CRC={hex32(h.crc)} PLM_f8={hex32(h.plm_f8)}"

    def read_passive_status(self, *, include_full_report: bool = False) -> dict:
        """Fast read-only status probe (safe for Live Goldilocks Watch polling)."""
        with self._lock:
            if not self.has_target:
                return {"available": False, "badge": "NO TARGET", "detail": "Select a GPU in Step 1."}
            backend, owned = self._open_backend(writable=False)
            try:
                host = backend.host()
                check_host_readonly_safe(host)
                h = backend.health()
                pci = backend.pci_id()
                badge, detail = self._classify_health_badge(h)
                plan_obj: Plan | None = None
                if self.state.plan_path:
                    try:
                        plan_obj = Plan.load(self.state.plan_path)
                    except Exception:
                        plan_obj = None
                health_issues = evaluate_health(h, plan_obj)
                host_issues = evaluate_host(host, require_hold=True)
                report_lines: list[str] = []
                proved_row: int | None = None
                surrounding: list[dict] = []
                if include_full_report and plan_obj is not None:
                    report_lines = format_status_report(plan_obj, backend)
                    try:
                        row, nearby = scan_slots(plan_obj, backend, override_slot=self.overrides.slot)
                        proved_row = row
                        surrounding = [
                            {"row": i, "word": hex32(v), "empty": v == plan_obj.empty_word}
                            for i, v in nearby
                        ]
                    except Exception:
                        pass
                return {
                    "available": True,
                    "timestamp": time.time(),
                    "pci_id": pci,
                    "sku_name": sku_display_name(pci),
                    "badge": badge,
                    "detail": detail,
                    "ready_for_sense": len(health_issues) == 0 and len(host_issues) == 0,
                    "health": {
                        "fusectrl": hex32(h.fusectrl),
                        "dsov": h.dsov,
                        "crc": hex32(h.crc),
                        "plm": hex32(h.plm),
                        "plm_f8": hex32(h.plm_f8),
                        "plm_f8_open": bool(h.plm_f8 & 1),
                        "plm_f4": hex32(h.plm_f4) if h.plm_f4 is not None else None,
                        "en_sw": h.en_sw,
                        "debugctrl": hex32(h.debugctrl),
                        "efp": h.efp,
                        "dfp": h.dfp,
                        "sbfe": h.sbfe,
                        "wdata_wr_secure": h.wdata_wr_secure,
                        "resense_disable": h.resense_disable,
                    },
                    "host": host.serializable(),
                    "health_issues": health_issues,
                    "host_issues": host_issues,
                    "proved_row": proved_row,
                    "surrounding": surrounding,
                    "report_lines": report_lines,
                }
            finally:
                self._close_if_owned(backend, owned)

    def run_dry_run(self) -> dict:
        with self._lock:
            p = self._load_plan()
            backend, owned = self._open_backend(writable=False)
            lines: list[str] = []
            try:
                check_host_readonly_safe(backend.host())
                progress = detect_plan_progress(
                    p,
                    backend,
                    journal_path=self.state.journal_path,
                    receipt_path=self.state.receipt_path,
                    override_slot=self.overrides.slot,
                )
                if progress.state == "complete":
                    return {
                        "state": "complete",
                        "lines": [
                            f"All {progress.total_count}/{progress.total_count} plan record(s) are already burned."
                        ],
                    }
                if progress.state == "mismatch":
                    raise BurnError(f"dry-run mismatch: {progress.detail}")
                stage = "real" if progress.state == "resume" else "all"
                receipt = self.state.receipt_path if stage == "real" else None
                dry_ov = self.overrides
                if stage == "real" and receipt is None and not dry_ov.no_test_receipt:
                    dry_ov = replace(dry_ov, no_test_receipt=True)
                with Journal(self.state.journal_path) as j:
                    process_records(
                        p,
                        backend,
                        MissingWriter(),
                        j,
                        stage=stage,
                        confirm=lambda _p, _t: False,
                        test_receipt=receipt,
                        overrides=dry_ov,
                        dry_run=True,
                        run_active_sense=False,
                        show=lines.append,
                    )
                return {"state": progress.state, "stage": stage, "lines": lines}
            finally:
                self._close_if_owned(backend, owned)

    def run_active_sense(self, confirm_phrase: str) -> dict:
        with self._lock:
            p = self._load_plan()
            if confirm_phrase.strip() != "SENSE":
                raise BurnError("type 'SENSE' to run active_sense_readonly()")
            backend, owned = self._open_backend(writable=False)
            try:
                check_host_readonly_safe(backend.host())
                before = backend.health()
                pre_issues = evaluate_health(before, p)
                if pre_issues:
                    raise BurnError("passive health not ready for SENSE_CTRL: " + "; ".join(pre_issues))
                with Journal(self.state.journal_path) as j:
                    j.append(event="active_sense_confirmed", command="workflow_sense")
                    backend.active_sense_readonly()
                    after = backend.health()
                    post_issues = evaluate_health(after, p, require_efp_one=True)
                    j.append(
                        event="manual_active_sense",
                        before=before.serializable(),
                        after=after.serializable(),
                        issues=post_issues,
                    )
                badge, detail = self._classify_health_badge(after)
                return {
                    "passed": len(post_issues) == 0,
                    "badge": badge,
                    "detail": detail,
                    "issues": post_issues,
                    "after": after.serializable(),
                }
            finally:
                self._close_if_owned(backend, owned)

    # ------------------------------------------------------------------ Step 4: Burn
    def get_burn_preview(self) -> dict:
        with self._lock:
            p = self._load_plan()
            backend, owned = self._open_backend(writable=False)
            try:
                check_host_readonly_safe(backend.host())
                progress = detect_plan_progress(
                    p,
                    backend,
                    journal_path=self.state.journal_path,
                    receipt_path=self.state.receipt_path,
                    override_slot=self.overrides.slot,
                )
                base_blocks = replay_ff_rows(p.baseline_ff_rows) if p.baseline_ff_rows is not None else None
                pending_recs = list(p.records[progress.committed_count :]) if progress.state != "complete" else []
                start_row = progress.next_row if progress.next_row is not None else p.first_row
                groups = build_approval_groups(
                    pending_recs, start_row, base_blocks, is_live=not self.is_simulated
                )
                needs_receipt = (
                    progress.state == "resume"
                    and not self.state.receipt_path
                    and not self.overrides.no_test_receipt
                )
                return {
                    "state": progress.state,
                    "committed_count": progress.committed_count,
                    "total_count": progress.total_count,
                    "start_row": progress.start_row,
                    "next_row": progress.next_row,
                    "detail": progress.detail,
                    "stage": "real" if progress.state == "resume" else "all",
                    "needs_receipt": needs_receipt,
                    "receipt_path": self.state.receipt_path,
                    "confirm_mode": self.state.confirm_mode,
                    "execute_live": self.execute_live,
                    "is_simulated": self.is_simulated,
                    "summary": summarize_plan_before_after(p).to_dict(),
                    "groups": [g.to_dict() for g in groups],
                }
            finally:
                self._close_if_owned(backend, owned)

    def execute_burn(
        self,
        *,
        confirm_cb: Callable[[str, str], bool],
        on_event: Callable[[dict], None] | None = None,
        show_cb: Callable[[str], None] | None = None,
    ) -> dict:
        """Run the unified burn flow using the active ``confirm_mode`` (default ``group``)."""
        show = show_cb or (lambda _m: None)
        with self._lock:
            p = self._load_plan()
            if not self.is_simulated:
                if not self.execute_live:
                    raise BurnError(
                        "Live Execute mode is OFF (read-only protection). "
                        "Arm Live Writes first before burning physical hardware."
                    )
                if self.check_tty and not os.isatty(0):
                    raise BurnError("REFUSED live write without an interactive TTY")
            want_write = (not self.is_simulated) and self.execute_live
            backend, owned = self._open_backend(writable=want_write)
            try:
                if not self.is_simulated and not has_verified_writer(backend):
                    raise BurnError("adapter does not have write capability armed")
                check_host_readonly_safe(backend.host())
                progress = detect_plan_progress(
                    p,
                    backend,
                    journal_path=self.state.journal_path,
                    receipt_path=self.state.receipt_path,
                    override_slot=self.overrides.slot,
                )
                if progress.state == "complete":
                    return {"state": "complete", "records_written": 0, "next_pointer": progress.next_row}
                if progress.state == "mismatch":
                    raise BurnError(f"cannot burn plan: {progress.detail}")
                stage = "real" if progress.state == "resume" else "all"
                receipt = self.state.receipt_path if stage == "real" else None
                with Journal(self.state.journal_path) as j:
                    next_ptr = process_records(
                        p,
                        backend,
                        backend,
                        j,
                        stage=stage,
                        confirm=confirm_cb,
                        test_receipt=receipt,
                        overrides=self.overrides,
                        max_records=None,
                        dry_run=False,
                        strict_confirm=(not self.is_simulated) and (self.state.confirm_mode == "row"),
                        confirm_active_sense=not self.is_simulated,
                        confirm_mode=self.state.confirm_mode,
                        on_event=on_event,
                        show=show,
                    )
                return {
                    "state": "burned",
                    "stage": stage,
                    "next_pointer": next_ptr,
                    "is_simulated": self.is_simulated,
                }
            finally:
                self._close_if_owned(backend, owned)

    # ------------------------------------------------------------------ Step 5: One-Step Cold Verify
    def preflight_cold_verify(self) -> dict:
        """Check whether the live card is ready for cold-boot verification BEFORE capturing."""
        with self._lock:
            p = self._load_plan(for_review=True)
            j_sum = summarize_journal(self.state.journal_path, plan_sha256=p.content_hash)
            prev_boot = j_sum["last_burn_boot_id"] or last_session_boot_id(self.state.journal_path)
            inferred_stage = "real" if j_sum["real_burned_for_plan"] else "test"
            problems: list[str] = []
            notes: list[str] = []
            cur_boot: str | None = None
            ctrl_state: dict | None = None

            if not prev_boot:
                problems.append(
                    "No prior burn session found in the active journal yet. Burn at least the test row first."
                )

            if self.has_target:
                try:
                    backend, owned = self._open_backend(writable=False)
                    try:
                        host = backend.host()
                        check_host_readonly_safe(host)
                        cur_boot = host.boot_id
                        h = backend.health()
                        ctrl_state = {
                            "dsov": h.dsov,
                            "en_sw": h.en_sw,
                            "sbfe": h.sbfe,
                            "wdata_wr_secure": h.wdata_wr_secure,
                            "plm_f4": h.plm_f4,
                        }
                        # Drop None entries for cold_snapshot_controller_problems check.
                        clean_ctrl = {k: v for k, v in ctrl_state.items() if v is not None}
                        c_probs, c_note = cold_snapshot_controller_problems(clean_ctrl)
                        problems.extend(c_probs)
                        if c_note:
                            notes.append(c_note)
                    finally:
                        self._close_if_owned(backend, owned)
                except Exception as exc:
                    problems.append(f"Target readout failed: {exc}")
            else:
                cur_boot = _host_boot_id()
                problems.append("No target selected — select your GPU in Step 1 (or verify an existing snapshot file).")

            if prev_boot and cur_boot and prev_boot == cur_boot:
                problems.append(
                    "Host boot_id has NOT changed since the burn session — power off at the wall, "
                    "cold boot the machine, and run Cold Verify again."
                )

            return {
                "ready": len(problems) == 0,
                "inferred_stage": inferred_stage,
                "previous_boot_id": prev_boot,
                "current_boot_id": cur_boot,
                "problems": problems,
                "notes": notes,
                "controller_state": ctrl_state,
                "journal_summary": j_sum,
            }

    def run_cold_verify_live(
        self,
        *,
        stage: str | None = None,
        previous_boot_id: str | None = None,
        out_path: str | None = None,
        receipt_out: str | None = None,
    ) -> dict:
        """One-step live cold-boot capture + verification + auto-attached test receipt."""
        with self._lock:
            p = self._load_plan(for_review=True)
            j_sum = summarize_journal(self.state.journal_path, plan_sha256=p.content_hash)
            eff_stage = stage or ("real" if j_sum["real_burned_for_plan"] else "test")
            prev = previous_boot_id or j_sum["last_burn_boot_id"] or last_session_boot_id(self.state.journal_path)
            if not prev:
                raise BurnError("previous burn session boot_id not found in journal")

            backend, owned = self._open_backend(writable=False)
            try:
                check_host_readonly_safe(backend.host())
                if callable(getattr(backend, "capture_cold_snapshot", None)):
                    snap = backend.capture_cold_snapshot(previous_boot_id=prev, read_rows=True)
                else:
                    raise BurnError(
                        "selected backend does not support live capture_cold_snapshot() — "
                        "use 'Verify Existing Snapshot File' instead"
                    )
            finally:
                self._close_if_owned(backend, owned)

            warnings = cold_snapshot_warnings(snap)
            dev_slug = p.pci_id.split(":")[-1].lower()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = out_path or f"private/cold_{dev_slug}_{eff_stage}_{ts}.json"
            write_private_json(snap_path, snap, overwrite=True)

            return self._verify_cold_dict(
                p, snap, stage=eff_stage, snapshot_path=snap_path,
                receipt_out=receipt_out, warnings=warnings,
            )

    def run_cold_verify_file(
        self,
        cold_path: str,
        *,
        stage: str | None = None,
        receipt_out: str | None = None,
    ) -> dict:
        """Verify an existing mode-0600 cold snapshot file with structured diff output."""
        with self._lock:
            p = self._load_plan(for_review=True)
            j_sum = summarize_journal(self.state.journal_path, plan_sha256=p.content_hash)
            eff_stage = stage or ("real" if j_sum["real_burned_for_plan"] else "test")
            cp = require_private_file(cold_path, "cold snapshot")
            snap = json.loads(cp.read_text(encoding="utf-8"))
            warnings = cold_snapshot_warnings(snap) if isinstance(snap, dict) else []
            return self._verify_cold_dict(
                p, snap, stage=eff_stage, snapshot_path=str(cp),
                receipt_out=receipt_out, warnings=warnings,
            )

    def _verify_cold_dict(
        self,
        p: Plan,
        snap: dict,
        *,
        stage: str,
        snapshot_path: str,
        receipt_out: str | None,
        warnings: list[str],
    ) -> dict:
        comparison = build_cold_comparison_table(p, snap, stage=stage)
        try:
            with Journal(self.state.journal_path) as j:
                receipt = verify_cold_snapshot(p, snap, stage=stage, journal=j)
        except BurnError as exc:
            return {
                "passed": False,
                "stage": stage,
                "snapshot_path": snapshot_path,
                "receipt_path": None,
                "error": str(exc),
                "warnings": warnings,
                "comparison": comparison,
                "summary": summarize_plan_before_after(p).to_dict(),
            }

        saved_receipt: str | None = None
        if stage == "test" or receipt_out:
            dev_slug = p.pci_id.split(":")[-1].lower()
            r_path = receipt_out or f"private/test_{dev_slug}_{p.content_hash[:8]}.receipt"
            write_private_json(r_path, receipt, overwrite=True)
            saved_receipt = r_path
            if stage == "test":
                self.state.receipt_path = saved_receipt
                self._save_state()

        return {
            "passed": True,
            "stage": stage,
            "snapshot_path": snapshot_path,
            "receipt_path": saved_receipt,
            "receipt": receipt,
            "warnings": warnings,
            "comparison": comparison,
            "summary": summarize_plan_before_after(p).to_dict(),
        }

    # ------------------------------------------------------------------ Step 6: Tools & Diagnostics
    def decode_ff_word(self, raw: str) -> dict:
        word = u32(raw.strip(), "word")
        ff = decode_ff(word)
        overlapping = fields_in_block(ff.chain, ff.block)
        return {
            "word": hex32(word),
            "chain": ff.chain,
            "block": ff.block,
            "data": f"0x{ff.data:04x}",
            "iff_bit": 0,
            "fields": [
                {
                    "key": f.key,
                    "name": f.name,
                    "bits": f"{f.bit}..{f.bit + f.width - 1}",
                    "category": f.category,
                    "burnable": f.burnable,
                }
                for f in overlapping
            ],
        }

    def read_experimental(self) -> dict:
        with self._lock:
            backend, owned = self._open_backend(writable=False)
            try:
                check_host_readonly_safe(backend.host())
                readings, poisoned = read_experimental_registers(backend)
                lines = format_experimental_report(backend, readings=readings, poisoned=poisoned)
                with Journal(self.state.journal_path) as j:
                    j.append(
                        event="manual_experimental_read",
                        pci_id=backend.pci_id(),
                        skipped_poison=poisoned,
                        registers={k: hex32(v) for k, v in readings.items()},
                    )
                return {
                    "poisoned": poisoned,
                    "readings": {k: hex32(v) for k, v in readings.items()},
                    "lines": lines,
                }
            finally:
                self._close_if_owned(backend, owned)

    def read_ff_rows_range(self, start: int = 222, end: int = 270) -> list[dict]:
        with self._lock:
            if not (0 <= start <= end <= 511):
                raise BurnError("row range must satisfy 0 <= start <= end <= 511")
            if (end - start) > 290:
                raise BurnError("row range too large (max 290 rows per request)")
            backend, owned = self._open_backend(writable=False)
            try:
                check_host_readonly_safe(backend.host())
                rows: list[dict] = []
                for r in range(start, end + 1):
                    w = backend.fuse_row(r)
                    decoded = ""
                    if w != 0 and r >= 222:
                        try:
                            ff = decode_ff(w)
                            flds = ",".join(f.key for f in fields_in_block(ff.chain, ff.block))
                            decoded = f"c{ff.chain}/b{ff.block}=0x{ff.data:04x}" + (f" ({flds})" if flds else "")
                        except BurnError:
                            decoded = "IFF/non-FF"
                    rows.append({
                        "row": r,
                        "word": hex32(w),
                        "empty": w == 0,
                        "decoded": decoded,
                    })
                return rows
            finally:
                self._close_if_owned(backend, owned)

    def read_register_value(self, target: str) -> dict:
        with self._lock:
            addr = resolve_register_address(target)
            backend, owned = self._open_backend(writable=False)
            try:
                check_host_readonly_safe(backend.host())
                val = backend.register(addr)
                return {
                    "target": target,
                    "address": hex32(addr),
                    "value": hex32(val),
                    "poison": is_poison(val),
                }
            finally:
                self._close_if_owned(backend, owned)

    def read_journal(self, limit: int = 50) -> dict:
        with self._lock:
            plan_sha = None
            if self.state.plan_path:
                try:
                    plan_sha = Plan.load(self.state.plan_path).content_hash
                except Exception:
                    plan_sha = None
            return summarize_journal(self.state.journal_path, plan_sha256=plan_sha, limit=limit)

    # ------------------------------------------------------------------ Overview & Smart Next Step
    def get_overview(self) -> dict:
        """Return complete workspace status + smart recommended next step for UI header/stepper."""
        with self._lock:
            plan_info: dict | None = None
            plan_obj: Plan | None = None
            plan_error: str | None = None
            if self.state.plan_path:
                try:
                    plan_obj = Plan.load(self.state.plan_path)
                    summ = summarize_plan_before_after(plan_obj)
                    plan_info = {
                        "path": self.state.plan_path,
                        "pci_id": plan_obj.pci_id,
                        "sku_name": sku_display_name(plan_obj.pci_id),
                        "record_count": len(plan_obj.records),
                        "sha12": plan_obj.content_hash[:12],
                        "sha256": plan_obj.content_hash,
                        "summary": summ.to_dict(),
                    }
                except Exception as exc:
                    plan_error = str(exc)

            j_sum = summarize_journal(
                self.state.journal_path,
                plan_sha256=plan_obj.content_hash if plan_obj else None,
            )

            live_status: dict | None = None
            live_error: str | None = None
            progress_info: dict | None = None
            if self.has_target:
                try:
                    live_status = self.read_passive_status(include_full_report=False)
                except Exception as exc:
                    live_error = str(exc)

            if self.has_target and plan_obj is not None and live_status is not None and live_status.get("badge") != "POISON":
                try:
                    backend, owned = self._open_backend(writable=False)
                    try:
                        prog = detect_plan_progress(
                            plan_obj,
                            backend,
                            journal_path=self.state.journal_path,
                            receipt_path=self.state.receipt_path,
                            override_slot=self.overrides.slot,
                        )
                        progress_info = {
                            "state": prog.state,
                            "committed_count": prog.committed_count,
                            "total_count": prog.total_count,
                            "start_row": prog.start_row,
                            "next_row": prog.next_row,
                            "detail": prog.detail,
                        }
                    finally:
                        self._close_if_owned(backend, owned)
                except Exception:
                    progress_info = None

            # Smart Next-Step Advisor
            rec_step = "target"
            headline = "Select a GPU or simulation snapshot to get started"
            detail = "Pick a live CMP 170HX BDF (and detach nvidia/nouveau) or load a simulation snapshot."
            step_states = {
                "target": "current",
                "configure": "pending",
                "preflight": "pending",
                "burn": "pending",
                "cold_verify": "pending",
                "tools": "ready",
            }

            if self.has_target:
                step_states["target"] = "done"
                cur_boot = (live_status or {}).get("host", {}).get("boot_id") or _host_boot_id()
                badge = (live_status or {}).get("badge", "UNKNOWN")
                prev_burn_boot = j_sum["last_burn_boot_id"]

                if j_sum["has_unresolved_halt"]:
                    rec_step = "tools"
                    headline = "Journal contains HALT_UNVERIFIED — inspect before any further burn"
                    detail = "A previous write halted without full verification. Review the journal and card rows before continuing."
                    step_states["burn"] = "warning"
                elif plan_obj is None:
                    rec_step = "configure"
                    step_states["configure"] = "current"
                    headline = "Configure unlocks or open an existing plan JSON"
                    detail = "Read your card's defect masks read-only in Step 2 to generate a plan, or browse private/ for a saved plan."
                else:
                    step_states["configure"] = "done"
                    if j_sum["cold_verified_real"]:
                        rec_step = "cold_verify"
                        step_states["preflight"] = "done"
                        step_states["burn"] = "done"
                        step_states["cold_verify"] = "done"
                        headline = "Plan complete & cold-boot verified!"
                        detail = "All burned rows persisted across a cold boot and matched expected shadow registers."
                    elif j_sum["burned_for_plan"] > 0 and prev_burn_boot and cur_boot and prev_burn_boot != cur_boot:
                        # Rebooted after burning!
                        if j_sum["real_burned_for_plan"] and not j_sum["cold_verified_real"]:
                            rec_step = "cold_verify"
                            step_states["preflight"] = "done"
                            step_states["burn"] = "done"
                            step_states["cold_verify"] = "current"
                            headline = "Cold boot detected — run One-Step Cold Verify (real stage)"
                            detail = "Your host rebooted since burning. Verify that all programmed rows persisted and took effect."
                        elif j_sum["test_burned_for_plan"] and not j_sum["cold_verified_test"]:
                            rec_step = "cold_verify"
                            step_states["preflight"] = "done"
                            step_states["burn"] = "ready"
                            step_states["cold_verify"] = "current"
                            headline = "Cold boot detected — verify your no-op test row now"
                            detail = "Run One-Step Cold Verify (test stage) to prove persistence and auto-attach your test receipt."
                        elif badge in ("GOLDILOCKS", "ARMED"):
                            rec_step = "burn"
                            step_states["preflight"] = "done"
                            step_states["burn"] = "current"
                            headline = "Goldilocks window active — resume burning remaining records"
                            detail = "Test row is cold-verified and Goldilocks is live. Proceed to Step 4 to burn the remaining groups."
                        else:
                            rec_step = "preflight"
                            step_states["preflight"] = "current"
                            headline = "Test row cold-verified — glitch card to resume remaining records"
                            detail = "Enable Live Watch in Step 3 while hunting a Goldilocks window, then resume Step 4."
                    elif progress_info and progress_info["state"] == "complete":
                        rec_step = "cold_verify"
                        step_states["preflight"] = "done"
                        step_states["burn"] = "done"
                        step_states["cold_verify"] = "current"
                        headline = "All plan rows burned — power off at the wall, then run Cold Verify"
                        detail = "Read-back passed, but fuses are only confirmed after a full cold power cycle."
                    elif badge in ("GOLDILOCKS", "ARMED"):
                        rec_step = "burn"
                        step_states["preflight"] = "done"
                        step_states["burn"] = "current"
                        headline = f"{badge} detected — ready to dry-run or burn"
                        detail = "Health gates look good. Run a quick passive dry-run in Step 3 or proceed directly to Step 4 (Burn)."
                    else:
                        rec_step = "preflight"
                        step_states["preflight"] = "current"
                        headline = "Plan ready — hunt Goldilocks window with Live Watch"
                        detail = f"Card is currently {badge}. Open Step 3 and toggle Live Watch while glitching until DSOV=0 / CRC=0x55 / PLM_f8 OPEN."

            sku_mismatch = False
            live_pci = (live_status or {}).get("pci_id") or self._bdf_sysfs_pci_id(self.state.bdf)
            if plan_info and live_pci and plan_info["pci_id"] != live_pci.lower():
                sku_mismatch = True

            return {
                "target_label": self.target_label(),
                "mode_label": self.mode_label(),
                "has_target": self.has_target,
                "is_simulated": self.is_simulated,
                "execute_live": self.execute_live,
                "bdf": self.state.bdf,
                "fake_path": self.state.fake_path,
                "sim_pcie_path": self.state.sim_pcie_path,
                "journal_path": self.state.journal_path,
                "receipt_path": self.state.receipt_path,
                "confirm_mode": self.state.confirm_mode,
                "plan": plan_info,
                "plan_error": plan_error,
                "sku_mismatch": sku_mismatch,
                "live_status": live_status,
                "live_error": live_error,
                "progress": progress_info,
                "journal_summary": j_sum,
                "overrides": self.overrides_dict(),
                "driver_blockers": self.inspect_driver_blockers(),
                "recommended_step": rec_step,
                "headline": headline,
                "detail": detail,
                "step_states": step_states,
                "step_help": STEP_HELP,
            }
