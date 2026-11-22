from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vfi_burn.cli import main
from vfi_burn.menu import BurnMenu, MenuSession

ROOT = Path(__file__).resolve().parents[1]


class TestBurnMenu(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.work.chmod(0o700)

        self.plan_path = self.work / "plan.json"
        self.plan_path.write_text((ROOT / "examples/plan.example.json").read_text())
        self.plan_path.chmod(0o600)

        self.fake_path = self.work / "fake.json"
        self.fake_path.write_text((ROOT / "examples/fake.example.json").read_text())
        self.fake_path.chmod(0o600)

        self.cold_path = self.work / "cold.json"
        self.cold_path.write_text((ROOT / "examples/cold.example.json").read_text())
        self.cold_path.chmod(0o600)

        self.journal_path = self.work / "session.jsonl"

    def _run_menu(self, session: MenuSession, responses: list[str], *, check_tty: bool = False):
        it = iter(responses)
        lines: list[str] = []

        def fake_input(prompt: str = "") -> str:
            lines.append(f"PROMPT: {prompt}")
            try:
                return next(it)
            except StopIteration:
                raise EOFError

        menu = BurnMenu(session, input_fn=fake_input, show=lines.append, check_tty=check_tty)
        rc = menu.run()
        return rc, "\n".join(lines), menu

    def _journal_events(self) -> list[dict]:
        if not self.journal_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_review_plan_decode_status_sense_and_dry_run(self):
        session = MenuSession(
            plan_path=str(self.plan_path),
            fake_path=str(self.fake_path),
            journal_path=str(self.journal_path),
        )
        rc, out, menu = self._run_menu(
            session,
            [
                "2",             # Review plan offline
                "10", "0x0fe805c1",  # Decode FF word
                "3",             # Status & passive inspect
                "4", "NO",       # Active sense cancelled
                "4", "SENSE",    # Active sense confirmed
                "5",             # Passive dry-run preview
                "0",             # Quit
            ],
        )
        self.assertEqual(rc, 0)
        self.assertIn("PLAN REVIEW", out)
        self.assertIn("chain=1 block=23 data=0x0fe8", out)
        self.assertIn("PASSIVE INSPECTION OK", out)
        self.assertIn("Active sense cancelled.", out)
        self.assertIn("PASS: active sense clean", out)
        self.assertIn("DRY-RUN COMPLETE", out)

        # Dry-run must NOT have modified fake rows.
        self.assertEqual(menu._cached_fake.fuse_row(223), 0)
        events = [e["event"] for e in self._journal_events()]
        self.assertIn("inspect_result", events)
        self.assertIn("declined", events)
        self.assertIn("manual_active_sense", events)
        self.assertIn("dry_run_complete", events)
        self.assertNotIn("attempt", events)

    def test_single_window_burn_flow_on_fake(self):
        session = MenuSession(
            plan_path=str(self.plan_path),
            fake_path=str(self.fake_path),
            journal_path=str(self.journal_path),
        )
        # examples/plan.example.json has 2 records:
        # 0: FAKE_TEST (0x13572468) at row 223 -> YES
        # -> CONTINUE SAME WINDOW checkpoint
        # 1: FAKE_COMPUTE (0x24681357) at row 224 -> YES
        rc, out, menu = self._run_menu(
            session,
            [
                "6",
                "YES",                   # confirm FAKE_TEST
                "CONTINUE SAME WINDOW",  # keep same glitch window
                "YES",                   # confirm FAKE_COMPUTE
                "0",
            ],
        )
        self.assertEqual(rc, 0, out)
        self.assertIn("SIMULATION COMPLETE", out)
        self.assertEqual(menu._cached_fake.fuse_row(223), 0x13572468)
        self.assertEqual(menu._cached_fake.fuse_row(224), 0x24681357)

        events = [e["event"] for e in self._journal_events()]
        self.assertIn("same_window_continue_confirmed", events)
        self.assertEqual(events.count("verified_readback_only"), 2)

    def test_split_window_burn_with_cold_verify_receipt(self):
        receipt_out = self.work / "test.receipt"
        session = MenuSession(
            plan_path=str(self.plan_path),
            fake_path=str(self.fake_path),
            journal_path=str(self.journal_path),
        )
        rc, out, menu = self._run_menu(
            session,
            [
                "7", "1", "YES",                                           # split stage=test
                "11", "2", str(self.cold_path), "test", str(receipt_out),  # verify cold -> sets receipt_path
                "7", "2", "", "YES",                                       # split stage=real
                "0",
            ],
        )
        self.assertEqual(rc, 0, out)
        self.assertEqual(session.receipt_path, str(receipt_out))
        self.assertTrue(receipt_out.exists())
        self.assertEqual(menu._cached_fake.fuse_row(223), 0x13572468)
        self.assertEqual(menu._cached_fake.fuse_row(224), 0x24681357)

    def test_overrides_and_manual_console_submenus(self):
        session = MenuSession(
            plan_path=str(self.plan_path),
            fake_path=str(self.fake_path),
            journal_path=str(self.journal_path),
        )
        rc, out, _ = self._run_menu(
            session,
            [
                "8", "1", "225", "2", "5", "0",  # set slot=225, skip_active_sense, vout_range
                "9", "status", "quit",           # enter manual console and exit cleanly
                "8", "1", "-", "r", "0",         # clear slot with '-', reset all overrides
                "0",
            ],
        )
        self.assertEqual(rc, 0, out)
        self.assertIn("GATE SUMMARY : PASS", out)
        self.assertIn("All manual overrides reset to OFF.", out)
        self.assertEqual(session.overrides.active_names(), [])

    def test_live_execute_guard_refuses_before_opening_bar0(self):
        # Nonexistent BDF ensures any accidental BAR0/sysfs open would fail with "PCI device not found".
        session = MenuSession(
            plan_path=str(self.plan_path),
            bdf="0000:ff:00.0",
            assume_vout_mv=787.5,
            journal_path=str(self.journal_path),
            execute_live=False,
        )
        rc, out, _ = self._run_menu(
            session,
            [
                "6",                                 # must refuse: Live Execute mode is OFF
                "1", "g", "WRONG TOKEN",             # wrong token leaves execute OFF
                "g", "ENABLE LIVE WRITES",           # arms live execute
                "c", "0000:fe:00.0", "", "787.5",    # changing target resets execute back to OFF
                "0",
                "0",
            ],
            check_tty=True,
        )
        self.assertEqual(rc, 0)
        self.assertIn("Live Execute mode is OFF", out)
        self.assertNotIn("PCI device not found", out)
        self.assertIn("Kept Live Execute mode OFF.", out)
        self.assertIn("Live Execute mode ENABLED.", out)
        self.assertIn("Live Execute reset to OFF", out)
        self.assertFalse(session.execute_live)
        self.assertEqual(session.bdf, "0000:fe:00.0")

    def test_cli_launches_menu_with_no_args_and_menu_subcommand(self):
        for argv in (
            [],
            [
                "menu",
                "--plan",
                str(self.plan_path),
                "--fake",
                str(self.fake_path),
                "--journal",
                str(self.journal_path),
            ],
        ):
            out, err = io.StringIO(), io.StringIO()
            with patch("builtins.input", side_effect=["0"]):
                with redirect_stdout(out), redirect_stderr(err):
                    rc = main(argv)
            self.assertEqual(rc, 0, err.getvalue())
            self.assertIn("vfi_burn — CMP 170HX 8 GB FF Fuse-Burn Safety Controller", out.getvalue())
            self.assertIn("Exiting vfi_burn menu.", out.getvalue())

    def test_plan_review_outcome_unknown_fuses_and_ff_row_capacity(self):
        from vfi_burn.ffrecord import analyze_ff_block, encode_ff

        # 1. Verify analyze_ff_block distinguishes well-known Gen4 bit vs experimental magic_d bits
        # Chain 2 block 47 bit 3 is pcie_magic_gen4_dis (well understood); bit 4 is experimental magic_d.
        clean_gen4 = analyze_ff_block(2, 47, 0x0000, old_data=0x0008)
        self.assertEqual(clean_gen4["warnings"], [])
        exp_magic = analyze_ff_block(2, 47, 0x0010, old_data=0x0000)
        self.assertTrue(any("EXPERIMENTAL" in w for w in exp_magic["warnings"]))

        # Unmapped bit 371 in chain 1 block 23 (bit 3 within block 23: between nvlink 368..370 and sys_pipe 372..379)
        unmapped = analyze_ff_block(1, 23, 0x0008, old_data=0x0000)
        self.assertIn(371, unmapped["changed_unmapped_bits"])
        self.assertTrue(any("UNMAPPED" in w for w in unmapped["warnings"]))

        # 2. Build a realistic synthetic GA100 FF plan + fake snapshot and test Menu [2] Review Plan
        # Existing FF rows:
        #   row 222: chain 1 block 8 (gpc_disable bits 131..138 -> bits 3..10 in blk 8) = 0x0428 (gpc_disable=0x85)
        #   row 223: chain 2 block 47 (pcie_magic_gen4_dis bit 3) = 0x0008
        #   row 224: chain 1 block 23 (nvlink 368..370 + unmapped 371 + sys_pipe 372..379) = 0x0ff7
        w_gpc_old = encode_ff(1, 8, 0x0428)
        w_pcie_old = encode_ff(2, 47, 0x0008)
        w_sys_old = encode_ff(1, 23, 0x0FF7)

        w_noop = encode_ff(1, 8, 0x0428)         # exact rewrite -> NO-OP
        w_gpc_new = encode_ff(1, 8, 0x0420)      # clears GPC0 disable bit (0x85 -> 0x84)
        w_pcie_new = encode_ff(2, 47, 0x0000)    # clears Gen4 disable bit only (well-understood)
        w_sys_new = encode_ff(1, 23, 0x000F)     # clears sys_pipe_disable (experimental) + flips unmapped bit 371!
        w_unknown = encode_ff(1, 99, 0x1234)     # unknown block not in GA100 option map

        plan_dict = {
            "schema_version": 1,
            "card": {
                "pci_id": "10de:20c2",
                "die_rows": {str(i): f"0x100{i:x}" for i in range(8, 16)},
                "expected_masks": {"GPC_DISABLE": "0x85"},
                "defect_masks": {"GPC_DISABLE": "0x04"},
            },
            "fpf": {"first_row": 222, "last_row": 486, "empty_word": "0x00000000"},
            "environment": {"stock_mv": 787.5, "tolerance_mv": 25},
            "ceiling": {"base_sm": 122, "override_sm": 128},
            "records": [
                {
                    "name": "NOOP_TEST",
                    "kind": "test",
                    "target": "NOOP_CHAIN1_BLK8",
                    "word": f"0x{w_noop:08x}",
                    "decode": "No-op rewrite of chain 1 block 8",
                    "defect_sensitive": False,
                    "expected_cold": {"GPC_DISABLE": "0x85"},
                },
                {
                    "name": "UNLOCK_GPC0",
                    "kind": "regular",
                    "target": "GPC_DISABLE",
                    "word": f"0x{w_gpc_new:08x}",
                    "decode": "Enable GPC0 (0x85 -> 0x84)",
                    "defect_sensitive": True,
                    "mask_update": {"field": "GPC_DISABLE", "from": "0x85", "to": "0x84"},
                    "resulting_sm_count": 112,
                    "expected_cold": {"GPC_DISABLE": "0x84"},
                },
                {
                    "name": "UNLOCK_PCIE_GEN4",
                    "kind": "pcie",
                    "target": "PCIE_GEN4",
                    "word": f"0x{w_pcie_new:08x}",
                    "decode": "Clear Gen4 disable flag",
                    "defect_sensitive": False,
                    "link_cap2_before": "0x00000002",
                    "link_cap2_after": "0x0000001e",
                    "expected_cold": {},
                },
                {
                    "name": "EXP_SYS_PIPE",
                    "kind": "regular",
                    "target": "SYS_PIPE_DISABLE",
                    "word": f"0x{w_sys_new:08x}",
                    "decode": "Enable sys_pipe (experimental) + bit 371",
                    "defect_sensitive": False,
                    "expected_cold": {},
                },
                {
                    "name": "UNKNOWN_BLK99",
                    "kind": "regular",
                    "target": "UNKNOWN_FUSE",
                    "word": f"0x{w_unknown:08x}",
                    "decode": "Unmapped chain 1 block 99",
                    "defect_sensitive": False,
                    "expected_cold": {},
                },
            ],
        }
        fake_dict = json.loads(self.fake_path.read_text())
        fake_dict["rows"] = {str(r): "0x00000000" for r in range(222, 487)}
        fake_dict["rows"]["222"] = f"0x{w_gpc_old:08x}"
        fake_dict["rows"]["223"] = f"0x{w_pcie_old:08x}"
        fake_dict["rows"]["224"] = f"0x{w_sys_old:08x}"
        r_plan = self.work / "rich_plan.json"
        r_fake = self.work / "rich_fake.json"
        r_plan.write_text(json.dumps(plan_dict))
        r_fake.write_text(json.dumps(fake_dict))
        r_plan.chmod(0o600)
        r_fake.chmod(0o600)

        session = MenuSession(
            plan_path=str(r_plan),
            fake_path=str(r_fake),
            journal_path=str(self.journal_path),
        )
        rc, out, _ = self._run_menu(session, ["2", "0"])
        self.assertEqual(rc, 0, out)
        # Capacity checks: 3 used (222..224), 5 to take (225..229), 257 / 265 remaining
        self.assertIn("Currently Used  : 3 / 265 rows (rows 222..224)", out)
        self.assertIn("Rows To Take    : 5 (1 test + 4 real) -> rows 225..229", out)
        self.assertIn("Remaining After : 257 / 265 free rows remaining", out)
        # Outcome checks
        self.assertIn("ENABLED GPC0 (defects 0x00000004 preserved)", out)
        self.assertIn("PCIe Speed Flags : ENABLED (cleared pcie_magic_gen4_dis)", out)
        self.assertIn("NO-OP (0 effective fuse bits changed", out)
        # Unknown / unmapped / experimental fuse audit checks
        self.assertIn("UNMAPPED / UNDOCUMENTED BITS CHANGING", out)
        self.assertIn("371", out)
        self.assertIn("NV_FUSE_OPT_SYS_PIPE_DISABLE", out)
        self.assertIn("UNKNOWN FF BLOCK (chain=1, block=99", out)
        # Well-understood Gen4 bit should NOT trigger a warning
        self.assertNotIn("[row 227 / UNLOCK_PCIE_GEN4]", out)


if __name__ == "__main__":
    unittest.main()
