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


if __name__ == "__main__":
    unittest.main()
