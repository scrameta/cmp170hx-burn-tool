"""Tests for split Compute/Memory + Est. RAM, group/plan burn confirmations,
WorkflowController, file catalog, one-step cold verify, and Local Web UI server."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from vfi_burn.backend import write_private_json
from vfi_burn.cli import build_parser, main
from vfi_burn.engine import Journal, build_approval_groups, process_records
from vfi_burn.ffrecord import decode_all_fields, replay_ff_rows
from vfi_burn.menuconfig import run_menuconfig
from vfi_burn.model import BurnError, Plan
from vfi_burn.planner import (
    UnlockConfig,
    build_plan_from_config,
    capture_card_snapshot,
    summarize_plan_before_after,
)
from vfi_burn.web import VFIWebServer
from vfi_burn.workflow import (
    WorkflowController,
    WorkspaceState,
    fix_file_permissions,
    inspect_file_metadata,
    list_directory_entries,
)

# Reuse synthetic GA100 state builder from test_planner_menuconfig
from tests.test_planner_menuconfig import _build_sim_adapter, _synthetic_card_state


class TestSplitComputeMemoryAndExpectedRAM(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.work.chmod(0o700)

    def tearDown(self):
        self.tmp.cleanup()

    def test_vram_bus_width_and_cuda_cores_for_8gb_and_10gb_skus(self):
        # 8 GB card (10de:20c2): 16 Gb HBM2e dies -> 1 GB/FBP CMP, 8 GB/FBP A100
        st8 = _synthetic_card_state(tpc_defs={1: 0x03})
        ad8 = _build_sim_adapter(st8, goldilocks=False)
        card8 = capture_card_snapshot(ad8)
        cfg8 = UnlockConfig.from_card(card8)
        cfg8.unlock_all_healthy_compute()
        # Memory must remain untouched when only Compute is enabled
        self.assertTrue(cfg8.is_all_healthy_compute_unlocked())
        self.assertFalse(cfg8.is_all_healthy_memory_unlocked())
        self.assertEqual(cfg8.target_fbp_count(), card8.current_fbp_count())

        s8 = cfg8.before_after_summary()
        self.assertEqual(s8.cuda_cores_before, s8.sm_before * 64)
        self.assertEqual(s8.cuda_cores_after, s8.sm_after * 64)
        self.assertEqual(s8.a100_gb_per_fbp, 8)
        self.assertEqual(s8.vram_cmp_before_gb, s8.fbp_before * 1)
        self.assertEqual(s8.vram_a100_before_gb, s8.fbp_before * 8)
        self.assertEqual(s8.bus_width_before, s8.fbp_before * 512)

        # Now toggle Memory independently -> 10 healthy FBPs (5120-bit, 10 GB CMP / 80 GB A100)
        cfg8.unlock_all_healthy_memory()
        s8_mem = cfg8.before_after_summary()
        self.assertEqual(s8_mem.fbp_after, 10)
        self.assertEqual(s8_mem.bus_width_after, 5120)
        self.assertEqual(s8_mem.vram_cmp_after_gb, 10)
        self.assertEqual(s8_mem.vram_a100_after_gb, 80)
        self.assertIn("80 GB", s8_mem.vram_label())
        self.assertIn("5120-bit", "\n".join(s8_mem.format_lines()))

        # 10 GB card (10de:2082): 8 Gb HBM2e dies -> 1 GB/FBP CMP, 4 GB/FBP A100
        st10 = _synthetic_card_state()
        st10["pci_id"] = "10de:2082"
        ad10 = _build_sim_adapter(st10, goldilocks=False)
        card10 = capture_card_snapshot(ad10)
        cfg10 = UnlockConfig.from_card(card10)
        cfg10.unlock_all_healthy_memory()
        s10 = cfg10.before_after_summary()
        self.assertEqual(s10.a100_gb_per_fbp, 4)
        self.assertEqual(s10.vram_cmp_after_gb, 10)
        self.assertEqual(s10.vram_a100_after_gb, 40)

    def test_menuconfig_independent_compute_and_memory_flags_and_ram_display(self):
        st = _synthetic_card_state()
        ad = _build_sim_adapter(st, goldilocks=False)
        card = capture_card_snapshot(ad)

        # Pick [3] (Unlock ALL healthy Memory ONLY), then preview [p], back [""], save [s]
        shown: list[str] = []
        ans = iter(["3", "p", "", "s"])
        cfg = run_menuconfig(card, use_curses=False, input_fn=lambda _p="": next(ans), show=shown.append)
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertFalse(cfg.is_all_healthy_compute_unlocked())
        self.assertTrue(cfg.is_all_healthy_memory_unlocked())
        out = "\n".join(shown)
        self.assertIn("Unlock ALL healthy Compute", out)
        self.assertIn("Unlock ALL healthy Memory", out)
        self.assertIn("5120-bit", out)
        self.assertIn("80 GB", out)
        self.assertIn("CUDA cores", out)

    def test_cli_configure_split_compute_and_memory_flags(self):
        st = _synthetic_card_state()
        ad = _build_sim_adapter(st, goldilocks=False)
        snap = self.work / "cold.json"
        write_private_json(snap, ad.to_snapshot_dict())
        out_compute = self.work / "plan_compute_only.json"
        out_memory = self.work / "plan_memory_only.json"

        rc1 = main(["configure", "--sim-pcie", str(snap), "--out", str(out_compute), "--compute"])
        self.assertEqual(rc1, 0)
        p1 = Plan.load(out_compute)
        s1 = summarize_plan_before_after(p1)
        self.assertGreater(s1.sm_after, s1.sm_before)
        self.assertEqual(s1.fbp_after, s1.fbp_before)

        rc2 = main(["configure", "--sim-pcie", str(snap), "--out", str(out_memory), "--memory"])
        self.assertEqual(rc2, 0)
        p2 = Plan.load(out_memory)
        s2 = summarize_plan_before_after(p2)
        self.assertEqual(s2.sm_after, s2.sm_before)
        self.assertGreater(s2.fbp_after, s2.fbp_before)


class TestGroupAndPlanConfirmationModes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.work.chmod(0o700)
        self.state = _synthetic_card_state()
        self.cold = _build_sim_adapter(self.state, goldilocks=False)
        card = capture_card_snapshot(self.cold)
        cfg = UnlockConfig.from_card(card)
        cfg.unlock_all_healthy_compute()
        cfg.unlock_all_healthy_memory()
        cfg.set_pcie_mode("gen4")
        cfg.set_a100_devid(0x20B5)
        self.plan_path = self.work / "full_plan.json"
        write_private_json(self.plan_path, build_plan_from_config(card, cfg))
        self.plan = Plan.load(self.plan_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_group_mode_deduplicates_pcie_and_devid_and_eliminates_hex_copypaste(self):
        ad = _build_sim_adapter(self.state, goldilocks=True)
        tokens_seen: list[str] = []
        events_seen: list[str] = []

        def _confirm(prompt: str, token: str) -> bool:
            tokens_seen.append(token)
            return True

        with Journal(self.work / "group.jsonl") as j:
            ptr = process_records(
                self.plan,
                ad,
                ad,
                j,
                stage="all",
                confirm=_confirm,
                confirm_mode="group",
                strict_confirm=True,
                on_event=lambda ev: events_seen.append(ev["type"]),
                show=lambda _m: None,
            )

        self.assertEqual(ptr, self.plan.expected_next_row + len(self.plan.records))
        # Expect: BURN TEST, CONTINUE SAME WINDOW, BURN COMPUTE+MEM, BURN PCIE (ONCE!), CONFIRM A100 DEVID 0x20B5 (ONCE!)
        self.assertEqual(
            tokens_seen,
            [
                "BURN TEST",
                "CONTINUE SAME WINDOW",
                "BURN COMPUTE+MEM",
                "BURN PCIE",
                "CONFIRM A100 DEVID 0x20B5",
            ],
        )
        # Zero per-row hex copy-paste prompts and zero duplicate REVIEW PCIE!
        self.assertFalse(any(t.startswith("YES ") for t in tokens_seen))
        self.assertEqual(tokens_seen.count("BURN PCIE"), 1)
        self.assertEqual(tokens_seen.count("CONFIRM A100 DEVID 0x20B5"), 1)

        # Every single row must still be written and verified
        self.assertEqual(events_seen.count("record_verified"), len(self.plan.records))
        self.assertIn("session_start", events_seen)
        self.assertIn("session_complete", events_seen)

        # Verify silicon outcome
        final_fields = decode_all_fields(replay_ff_rows(ad.pcie.rows))
        self.assertEqual(final_fields["gpc_disable"], 0x80)
        self.assertEqual(final_fields["fbp_disable"], 0xC00)
        self.assertEqual(final_fields["disable_gen3_speed_bit1"], 0)
        self.assertEqual(final_fields["pcie_boot_gen3_dis"], 0)

    def test_plan_mode_requires_test_devid_and_single_plan_approval(self):
        ad = _build_sim_adapter(self.state, goldilocks=True)
        tokens_seen: list[str] = []
        with Journal(self.work / "plan_mode.jsonl") as j:
            ptr = process_records(
                self.plan,
                ad,
                ad,
                j,
                stage="all",
                confirm=lambda _p, tok: (tokens_seen.append(tok), True)[1],
                confirm_mode="plan",
                show=lambda _m: None,
            )
        self.assertEqual(ptr, self.plan.expected_next_row + len(self.plan.records))
        self.assertEqual(
            tokens_seen,
            [
                "BURN TEST",
                "CONTINUE SAME WINDOW",
                "CONFIRM A100 DEVID 0x20B5",
                f"BURN PLAN {self.plan.content_hash[:8]}",
            ],
        )

    def test_group_mode_still_refuses_lnkcap2_mismatch_on_pcie(self):
        from vfi_burn.sim_pcie import build_pcie_config_space

        ad = _build_sim_adapter(self.state, goldilocks=True)
        # Corrupt PCIe config space LnkCap2 so it disagrees with plan's reviewed before state
        ad.pcie.pci_config = build_pcie_config_space(lnkcap2=0x1E)
        with Journal(self.work / "pcie_bad.jsonl") as j:
            with self.assertRaises(BurnError) as ctx:
                process_records(
                    self.plan,
                    ad,
                    ad,
                    j,
                    stage="all",
                    confirm=lambda _p, _t: True,
                    confirm_mode="group",
                    show=lambda _m: None,
                )
        self.assertIn("LnkCap2", str(ctx.exception))


class TestWorkflowControllerFileCatalogAndColdVerify(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.work.chmod(0o700)
        self.state = _synthetic_card_state()
        self.cold_adapter = _build_sim_adapter(self.state, goldilocks=False)
        self.cold_snap = self.work / "cold_sim.json"
        write_private_json(self.cold_snap, self.cold_adapter.to_snapshot_dict())
        self.gold_adapter = _build_sim_adapter(self.state, goldilocks=True)
        self.sim_snap = self.work / "gold_sim.json"
        write_private_json(self.sim_snap, self.gold_adapter.to_snapshot_dict())
        self.session_file = self.work / ".vfi_session.json"
        self.journal_file = self.work / "session.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_workspace_persistence_and_file_catalog_and_chmod600(self):
        ctl = WorkflowController(session_path=self.session_file)
        ctl.set_journal_path(str(self.journal_file))
        ctl.select_simulation(str(self.cold_snap), mode="sim_pcie")
        ctl.set_execute_live(True, "ENABLE LIVE WRITES")
        self.assertTrue(ctl.execute_live)

        # Reload session -> target & journal persisted, execute_live MUST default back to False!
        ctl2 = WorkflowController(session_path=self.session_file)
        self.assertEqual(ctl2.state.sim_pcie_path, str(self.cold_snap))
        self.assertEqual(ctl2.state.journal_path, str(self.journal_file))
        self.assertFalse(ctl2.execute_live)

        # Configure & save plan via WorkflowController on cold card
        cfg_data = ctl2.start_configure()
        self.assertTrue(cfg_data["loaded"])
        ctl2.patch_config({"action": "toggle_all_compute"})
        plan_path = self.work / "my_plan.json"
        saved = ctl2.save_configured_plan(str(plan_path))
        self.assertTrue(saved["saved"])

        # Deliberately loosen permissions, inspect in file catalog, then fix via fix_file_permissions
        os.chmod(plan_path, 0o644)
        listing = list_directory_entries(self.work, kind_filter="plan", target_pci_id="10de:20c2")
        plan_entries = [e for e in listing["entries"] if e["name"] == "my_plan.json"]
        self.assertEqual(len(plan_entries), 1)
        self.assertFalse(plan_entries[0]["is_private_0600"])
        self.assertEqual(plan_entries[0]["sku_match"], "match")

        fixed = fix_file_permissions(plan_path)
        self.assertTrue(fixed["is_private_0600"])

        # Symlink must be refused by fix_file_permissions
        sym = self.work / "bad_link.json"
        sym.symlink_to(plan_path)
        with self.assertRaises(BurnError):
            fix_file_permissions(sym)

    def test_one_step_cold_verify_file_auto_attaches_test_receipt(self):
        ctl = WorkflowController(session_path=self.session_file)
        ctl.set_journal_path(str(self.journal_file))
        # Step 1 & 2: Configure on cold card
        ctl.select_simulation(str(self.cold_snap), mode="sim_pcie")
        ctl.start_configure()
        ctl.patch_config({"action": "set_nvlink", "enabled": True})
        plan_path = self.work / "plan_nvl.json"
        ctl.save_configured_plan(str(plan_path))
        plan = Plan.load(plan_path)

        # Step 3 & 4: Switch to Goldilocks card and burn ONLY test row (decline CONTINUE SAME WINDOW)
        ctl.select_simulation(str(self.sim_snap), mode="sim_pcie")
        ctl.execute_burn(
            confirm_cb=lambda _p, tok: tok != "CONTINUE SAME WINDOW",
        )
        # Step 5: Verify cold snapshot from a distinct cold boot
        burn_boot = ctl.read_journal()["last_burn_boot_id"]
        self.assertIsNotNone(burn_boot)
        test_row = plan.expected_next_row
        cold_snap = {
            "pci_id": plan.pci_id,
            "cold_boot": True,
            "boot_id": "cold-boot-2",
            "previous_boot_id": burn_boot,
            "controller_state": {
                "dsov": 1,
                "en_sw": 0,
                "sbfe": 1,
                "wdata_wr_secure": 1,
                "plm_f4": 0x00000001,
            },
            "die_rows": {str(k): f"0x{v:08x}" for k, v in plan.die_rows.items()},
            "fpf_rows": {
                **{str(k): f"0x{v:08x}" for k, v in (plan.baseline_ff_rows or {}).items()},
                str(test_row): f"0x{plan.records[0].word:08x}",
            },
            "fields": {
                k: f"0x{v:08x}" for k, v in plan.records[0].expected_cold.items()
            },
        }
        cold_path = self.work / "cold_test.json"
        write_private_json(cold_path, cold_snap)
        receipt_out = self.work / "test.receipt"

        res = ctl.run_cold_verify_file(str(cold_path), stage="test", receipt_out=str(receipt_out))
        self.assertTrue(res["passed"], res)
        self.assertEqual(res["receipt_path"], str(receipt_out))
        # Receipt must be auto-attached to workspace state!
        self.assertEqual(ctl.state.receipt_path, str(receipt_out))
        self.assertTrue(all(r["match"] for r in res["comparison"]))


class TestWebServerAPIsAndBurnFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.work.chmod(0o700)
        self.state = _synthetic_card_state()
        self.cold_adapter = _build_sim_adapter(self.state, goldilocks=False)
        self.cold_snap = self.work / "cold_sim.json"
        write_private_json(self.cold_snap, self.cold_adapter.to_snapshot_dict())
        self.gold_adapter = _build_sim_adapter(self.state, goldilocks=True)
        self.sim_snap = self.work / "gold_sim.json"
        write_private_json(self.sim_snap, self.gold_adapter.to_snapshot_dict())

        self.ctl = WorkflowController(
            session_path=self.work / ".vfi_session.json",
            persist_session=True,
        )
        self.ctl.set_journal_path(str(self.work / "web_session.jsonl"))
        self.server = VFIWebServer(("127.0.0.1", 0), self.ctl, token="test-secret-token")
        self.port = self.server.bound_port
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.tmp.cleanup()

    def _req(self, method: str, path: str, body: dict | None = None, *, token: str | None = "test-secret-token", host: str | None = None):
        url = f"http://127.0.0.1:{self.port}{path}"
        headers: dict[str, str] = {}
        if token is not None:
            headers["X-VFI-Token"] = token
        if host is not None:
            headers["Host"] = host
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        req = Request(url, data=data, headers=headers, method=method)
        with urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()

    def test_static_assets_and_auth_and_host_guards(self):
        for asset in ("/", "/styles.css", "/app.js"):
            st, raw = self._req("GET", asset, token=None)
            self.assertEqual(st, 200)
            self.assertGreater(len(raw), 100)

        # Missing token -> 401
        with self.assertRaises(HTTPError) as ctx:
            self._req("GET", "/api/overview", token=None)
        self.assertEqual(ctx.exception.code, 401)

        # Wrong Host header -> 403 (DNS rebinding guard)
        with self.assertRaises(HTTPError) as ctx2:
            self._req("GET", "/api/overview", host="evil.example.com:8765")
        self.assertEqual(ctx2.exception.code, 403)

    def test_end_to_end_web_configure_and_group_burn(self):
        # 1. Select cold sim target & configure Compute + PCIe Gen3 + save plan
        st, raw = self._req("POST", "/api/target/sim", {"path": str(self.cold_snap), "mode": "sim_pcie"})
        self.assertEqual(st, 200)
        ov = json.loads(raw)["data"]
        self.assertTrue(ov["has_target"])

        self._req("POST", "/api/configure/start", {"refresh": True})
        _, cfg_raw = self._req("POST", "/api/configure/patch", {"action": "toggle_all_compute"})
        cfg_data = json.loads(cfg_raw)["data"]
        self.assertTrue(cfg_data["all_healthy_compute"])
        self.assertFalse(cfg_data["all_healthy_memory"])
        self._req("POST", "/api/configure/patch", {"action": "set_pcie_mode", "mode": "gen3"})
        plan_path = self.work / "web_plan.json"
        _, save_raw = self._req("POST", "/api/configure/save", {"path": str(plan_path)})
        self.assertTrue(json.loads(save_raw)["data"]["saved"])

        # 2. File browser API & decode API
        _, files_raw = self._req("GET", f"/api/files?dir={self.work}&kind=plan")
        names = [e["name"] for e in json.loads(files_raw)["data"]["entries"]]
        self.assertIn("web_plan.json", names)

        _, dec_raw = self._req("GET", "/api/tools/decode?word=0x0fe805c1")
        self.assertEqual(json.loads(dec_raw)["data"]["chain"], 1)

        # 3. Switch to Goldilocks sim target -> Passive status + Dry-Run
        self._req("POST", "/api/target/sim", {"path": str(self.sim_snap), "mode": "sim_pcie"})
        _, st_raw = self._req("GET", "/api/status")
        self.assertEqual(json.loads(st_raw)["data"]["badge"], "GOLDILOCKS")
        _, dry_raw = self._req("POST", "/api/preflight/dry-run", {})
        self.assertGreater(len(json.loads(dry_raw)["data"]["lines"]), 0)

        # 4. Start interactive burn in default group mode and resolve each single-use confirm_id
        self._req("POST", "/api/burn/start", {})
        resolved_tokens: list[str] = []
        used_id: str | None = None
        deadline = time.time() + 10.0
        while time.time() < deadline:
            _, b_raw = self._req("GET", "/api/burn/status")
            b_st = json.loads(b_raw)["data"]
            pc = b_st.get("pending_confirm")
            if pc is not None:
                used_id = pc["confirm_id"]
                resolved_tokens.append(pc["token"])
                self._req("POST", "/api/burn/confirm", {"confirm_id": pc["confirm_id"], "response": pc["token"]})
            elif not b_st["running"]:
                break
            time.sleep(0.02)

        self.assertFalse(json.loads(self._req("GET", "/api/burn/status")[1])["data"]["running"])
        self.assertEqual(
            resolved_tokens,
            ["BURN TEST", "CONTINUE SAME WINDOW", "BURN COMPUTE", "BURN PCIE"],
        )
        # Replaying an already-resolved confirm_id MUST fail (single-use nonce)
        assert used_id is not None
        with self.assertRaises(HTTPError) as replay_ctx:
            self._req("POST", "/api/burn/confirm", {"confirm_id": used_id, "response": "BURN PCIE"})
        self.assertEqual(replay_ctx.exception.code, 400)

    def test_cli_parser_has_web_tui_and_confirm_mode(self):
        p = build_parser()
        a_web = p.parse_args(["web", "--port", "9000", "--confirm-mode", "group"])
        self.assertEqual(a_web.cmd, "web")
        self.assertEqual(a_web.port, 9000)
        self.assertEqual(a_web.confirm_mode, "group")

        a_tui = p.parse_args(["tui", "--no-curses"])
        self.assertEqual(a_tui.cmd, "tui")
        self.assertTrue(a_tui.no_curses)

        a_burn = p.parse_args(["burn", "--plan", "p.json", "--bdf", "05:00.0", "--stage", "all", "--journal", "j.jsonl"])
        self.assertEqual(a_burn.confirm_mode, "group")


if __name__ == "__main__":
    unittest.main()
