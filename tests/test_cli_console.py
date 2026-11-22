from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from vfi_burn.backend import FakeController, dump_fake
from vfi_burn.cli import main
from vfi_burn.console import ManualConsole
from vfi_burn.engine import Journal, Overrides
from vfi_burn.model import Plan

ROOT = Path(__file__).resolve().parents[1]


class TestCLIAndConsole(unittest.TestCase):
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

        self.journal_path = self.work / "session.jsonl"

    def _run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_plan_and_inspect_cli(self):
        code, out, err = self._run_cli(["plan", "--plan", str(self.plan_path)])
        self.assertEqual(code, 0, err)
        self.assertIn("OFFLINE PLAN REVIEW", out)
        self.assertIn("FAKE_COMPUTE", out)

        code, out, err = self._run_cli(
            ["inspect", "--plan", str(self.plan_path), "--fake", str(self.fake_path), "--sense"]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("GATE SUMMARY : PASS", out)
        self.assertIn("first free row 223", out)

    def test_manual_read_works_even_on_faulted_or_cold_card(self):
        # Simulate a card whose PLM_f8 dropped and hold is inactive (cold/faulted).
        faulted = json.loads(self.fake_path.read_text())
        faulted["health"]["plm_f8"] = "0xFFFFFFFE"
        faulted["host"]["hold_active"] = False
        faulted["rows"]["223"] = "0xBADF510E"
        faulted_path = self.work / "faulted.json"
        faulted_path.write_text(json.dumps(faulted))
        faulted_path.chmod(0o600)

        code, out, err = self._run_cli(
            [
                "manual",
                "--plan",
                str(self.plan_path),
                "--fake",
                str(faulted_path),
                "--journal",
                str(self.journal_path),
                "read-row",
                "223",
            ]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("0xbadf510e", out)
        self.assertIn("POISON!", out)
        self.assertIn("NOT in a write-ready goldilocks state", err)

        code, out, err = self._run_cli(
            [
                "manual",
                "--plan",
                str(self.plan_path),
                "--fake",
                str(faulted_path),
                "--journal",
                str(self.journal_path),
                "read-reg",
                "PLM_F8",
            ]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("0xfffffffe", out)

        code, out, err = self._run_cli(
            [
                "manual",
                "--plan",
                str(self.plan_path),
                "--fake",
                str(faulted_path),
                "--journal",
                str(self.journal_path),
                "status",
            ]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("NOT WRITE-READY", out)

    def test_manual_read_still_refuses_when_nvidia_driver_loaded(self):
        bad_host = json.loads(self.fake_path.read_text())
        bad_host["host"]["nvidia_loaded"] = True
        bad_path = self.work / "nvidia.json"
        bad_path.write_text(json.dumps(bad_host))
        bad_path.chmod(0o600)

        code, out, err = self._run_cli(
            [
                "manual",
                "--plan",
                str(self.plan_path),
                "--fake",
                str(bad_path),
                "--journal",
                str(self.journal_path),
                "read-row",
                "222",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("nvidia/nouveau", err)

    def test_simulate_dry_run_does_not_mutate(self):
        code, out, err = self._run_cli(
            [
                "simulate",
                "--plan",
                str(self.plan_path),
                "--fake",
                str(self.fake_path),
                "--stage",
                "test",
                "--journal",
                str(self.journal_path),
                "--dry-run",
            ]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("DRY-RUN [TEST]: would write row 223", out)
        self.assertIn("SIMULATION DRY-RUN ONLY", out)

    def test_interactive_manual_console_workflow(self):
        plan = Plan.load(self.plan_path)
        ctl = FakeController(
            fields={"GPC_DISABLE": 0x85},
            rows={222: 0xABCD00DE, **{i: 0 for i in range(223, 228)}},
        )
        commands = iter(
            [
                "help",
                "status",
                "sense",
                "SENSE",
                "scan",
                "row 222",
                "rows 222 224",
                "reg FUSECTRL",
                "field GPC_DISABLE",
                "overrides",
                "diff",
                "write CUSTOM_FUSE 0x11223344 manual test fuse",
                "OVERRIDE NO TEST RECEIPT",
                "YES",
                "diff",
                "quit",
            ]
        )
        lines: list[str] = []
        with Journal(self.journal_path) as j:
            console = ManualConsole(
                plan,
                ctl,
                ctl,
                j,
                overrides=Overrides(no_test_receipt=True),
                input_fn=lambda prompt="": next(commands),
                show=lines.append,
            )
            rc = console.run()
        self.assertEqual(rc, 0)
        output = "\n".join(lines)
        self.assertIn("PASS: active sense clean", output)
        self.assertIn("Proved next free row: 223", output)
        self.assertIn("row 223: 0x00000000 -> 0x11223344", output)
        self.assertEqual(ctl.rows[223], 0x11223344)
        events = [json.loads(line)["event"] for line in self.journal_path.read_text().splitlines()]
        self.assertIn("manual_diff", events)

    def test_console_refuses_floorsweep_target_without_override_and_confirms_with_override(self):
        plan = Plan.load(self.plan_path)
        ctl = FakeController(
            fields={"GPC_DISABLE": 0x85},
            rows={222: 0xABCD00DE, **{i: 0 for i in range(223, 228)}},
        )
        # 1. Default console refuses one-liner targeting GPC_DISABLE (no mask_update).
        cmds = iter(["write GPC_DISABLE 0x11223344 enable gpc0", "quit"])
        lines: list[str] = []
        with Journal(self.journal_path) as j:
            ManualConsole(
                plan, ctl, ctl, j,
                overrides=Overrides(no_test_receipt=True),
                input_fn=lambda prompt="": next(cmds),
                show=lines.append,
            ).run()
        self.assertIn("floorsweep/mask field", "\n".join(lines))
        self.assertEqual(ctl.rows[223], 0)
        events = [json.loads(line) for line in self.journal_path.read_text().splitlines()]
        self.assertTrue(any(e["event"] == "manual_console_write_refused" for e in events))

        # 2. With undetermined_ceiling override, requires UNRECONCILED TARGET GPC_DISABLE confirmation.
        cmds2 = iter([
            "write GPC_DISABLE 0x11223344 enable gpc0",
            "UNRECONCILED TARGET GPC_DISABLE",
            "OVERRIDE NO TEST RECEIPT",
            "OVERRIDE CEILING",
            "YES",
            "quit",
        ])
        lines2: list[str] = []
        with Journal(self.journal_path) as j:
            ManualConsole(
                plan, ctl, ctl, j,
                overrides=Overrides(no_test_receipt=True, undetermined_ceiling=True),
                input_fn=lambda prompt="": next(cmds2),
                show=lines2.append,
            ).run()
        self.assertEqual(ctl.rows[223], 0x11223344)

    def test_inspect_logs_pass_and_fail_when_journal_supplied(self):
        code, out, err = self._run_cli(
            [
                "inspect",
                "--plan",
                str(self.plan_path),
                "--fake",
                str(self.fake_path),
                "--sense",
                "--journal",
                str(self.journal_path),
            ]
        )
        self.assertEqual(code, 0, err)
        entries = [json.loads(line) for line in self.journal_path.read_text().splitlines()]
        self.assertEqual(entries[-1]["event"], "inspect_result")
        self.assertEqual(entries[-1]["status"], "PASS")
        self.assertTrue(entries[-1]["active_sense"])

        bad = json.loads(self.fake_path.read_text())
        bad["health"]["plm_f8"] = "0xFFFFFFFE"
        bad_path = self.work / "bad_inspect.json"
        bad_path.write_text(json.dumps(bad))
        bad_path.chmod(0o600)
        code, out, err = self._run_cli(
            [
                "inspect",
                "--plan",
                str(self.plan_path),
                "--fake",
                str(bad_path),
                "--journal",
                str(self.journal_path),
            ]
        )
        self.assertEqual(code, 2)
        entries = [json.loads(line) for line in self.journal_path.read_text().splitlines()]
        self.assertEqual(entries[-1]["event"], "inspect_failed")
        self.assertEqual(entries[-1]["status"], "FAIL")

    def test_dump_fake_is_atomic_0600_and_refuses_overwrite(self):
        ctl = FakeController()
        out_file = self.work / "dumped.json"
        dump_fake(ctl, out_file)
        self.assertEqual(out_file.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(Exception):
            dump_fake(ctl, out_file)

    def test_burn_dry_run_and_execute_guards(self):
        # Create a temporary non-simulated live adapter module (read-only by default).
        mod_path = self.work / "tmp_live_adapter.py"
        mod_path.write_text(
            "from vfi_burn.backend import FakeController\n"
            "EVENTS = []\n"
            "class LiveStub(FakeController):\n"
            "    simulation = False\n"
            "    writer_verified = False\n"
            "    def active_sense_readonly(self):\n"
            "        EVENTS.append('sense')\n"
            "        super().active_sense_readonly()\n"
            "def make():\n"
            "    return LiveStub(fields={'GPC_DISABLE': 0x85},\n"
            "                    rows={222: 0xABCD00DE, 223: 0, 224: 0, 225: 0, 226: 0, 227: 0})\n"
        )
        import sys
        sys.path.insert(0, str(self.work))
        self.addCleanup(lambda: sys.path.remove(str(self.work)))
        self.addCleanup(lambda: sys.modules.pop("tmp_live_adapter", None))

        # Default burn (no --execute, no --sense) runs passive dry-run WITHOUT firing active_sense_readonly()!
        code, out, err = self._run_cli(
            [
                "burn",
                "--plan",
                str(self.plan_path),
                "--adapter",
                "tmp_live_adapter:make",
                "--stage",
                "test",
                "--journal",
                str(self.journal_path),
            ]
        )
        self.assertEqual(code, 0, err)
        self.assertIn("DRY-RUN [TEST]: would write row 223", out)
        self.assertIn("passive preflight", out)
        import tmp_live_adapter
        self.assertEqual(tmp_live_adapter.EVENTS, [])

        # burn --execute refuses when stdin is not a TTY or writer_verified is False.
        code, out, err = self._run_cli(
            [
                "burn",
                "--plan",
                str(self.plan_path),
                "--adapter",
                "tmp_live_adapter:make",
                "--stage",
                "test",
                "--journal",
                str(self.journal_path),
                "--execute",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("REFUSED --execute", err)

    def test_manual_console_fuse_state_and_pcie_write_preview(self):
        from vfi_burn.ffrecord import encode_ff

        plan = Plan.load(self.plan_path)
        w_pcie = encode_ff(2, 47, 0x0000)
        ctl = FakeController(
            fields={"GPC_DISABLE": 0x85},
            rows={222: 0xABCD00DE, 223: 0x13572468, 224: 0, 225: 0, 226: 0, 227: 0},
        )
        lines: list[str] = []
        cmds = iter([
            "state",
            f"preview 0x{w_pcie:08x}",
            f"write PCIE_GEN4 0x{w_pcie:08x} Clear Gen4 disable flag",
            "",  # cancel at confirmation after seeing exact PCIe writes!
            "quit",
        ])
        with Journal(self.journal_path) as j:
            rc = ManualConsole(
                plan,
                ctl,
                ctl,
                j,
                overrides=Overrides(no_test_receipt=True),
                input_fn=lambda prompt="": next(cmds),
                show=lines.append,
            ).run()
        self.assertEqual(rc, 0)
        out = "\n".join(lines)
        self.assertIn("FUSE CONTROLLER STATE (BAR0 0x0082xxxx)", out)
        self.assertIn("FUSECTRL (0x820000)", out)
        self.assertIn("DEBUGCTRL(0x820060)", out)
        self.assertIn("EXACT PCIE BAR0 MMIO WRITES TO BE ISSUED (row 224", out)
        self.assertIn("WR32 BAR0+0x00820040  EN_SW_OVERRIDE   <- 0x00000001", out)
        self.assertIn("WR32 BAR0+0x00820004  FUSEADDR         <- 0x000000e0", out)
        self.assertIn(f"WR32 BAR0+0x0082000C  FUSEWDATA        <- 0x{w_pcie:08x}", out)
        self.assertIn("WR32 BAR0+0x00820000  FUSECTRL         <- 0x00000002  (CMD_PROGRAM", out)
        self.assertIn("DEBUGCTRL (0x820060) NEVER written", out)
        # Cancelled before confirming -> row 224 must still be empty!
        self.assertEqual(ctl.fuse_row(224), 0)
        events = [json.loads(l)["event"] for l in self.journal_path.read_text().splitlines()]
        self.assertIn("manual_fuse_state", events)
        self.assertIn("manual_preview_writes", events)

    def test_manual_console_preview_then_apply_workflow(self):
        from vfi_burn.ffrecord import encode_ff

        plan = Plan.load(self.plan_path)
        w_pcie = encode_ff(2, 47, 0x0000)
        w_gpc = encode_ff(1, 8, 0x0420)  # floorsweep GPC word
        ctl = FakeController(
            fields={"GPC_DISABLE": 0x85},
            rows={222: 0xABCD00DE, 223: 0x13572468, 224: 0, 225: 0, 226: 0, 227: 0},
        )
        lines: list[str] = []
        cmds = iter([
            "apply",                                                # nothing staged -> refused
            f"preview 0x{w_pcie:08x}",                              # stage auto-labeled PCIe word
            "staged",
            "clear",                                                # discard staged write
            "apply",                                                # nothing staged -> refused
            f"preview PCIE_GEN4 0x{w_pcie:08x} Clear Gen4 disable", # stage with explicit label
            "apply 0xdeadbeef",                                     # extra arg -> refused (keeps stage!)
            "apply",                                                # apply exact staged preview
            "OVERRIDE NO TEST RECEIPT",                             # confirm no_test_receipt override
            "YES",                                                  # confirm burn row 224
            "apply",                                                # consumed -> refused (no double-burn!)
            f"preview 0x{w_gpc:08x}",                               # floorsweep word
            "apply",                                                # refused by floorsweep defect-mask guard!
            "quit",
        ])
        with Journal(self.journal_path) as j:
            rc = ManualConsole(
                plan,
                ctl,
                ctl,
                j,
                overrides=Overrides(no_test_receipt=True),
                input_fn=lambda prompt="": next(cmds),
                show=lines.append,
            ).run()
        self.assertEqual(rc, 0)
        out = "\n".join(lines)
        self.assertIn("STAGED: row 224 | target=PCIE_GEN4", out)
        self.assertIn("Cleared staged write", out)
        self.assertIn("takes NO arguments", out)
        self.assertIn("APPLYING STAGED WRITE: row 224", out)
        self.assertIn("looks like a floorsweep/mask field", out)
        self.assertEqual(ctl.fuse_row(224), w_pcie)
        self.assertEqual(ctl.fuse_row(225), 0)  # floorsweep word was refused!


if __name__ == "__main__":
    unittest.main()
