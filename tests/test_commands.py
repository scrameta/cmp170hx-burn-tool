"""Tests for decoupled command execution (vfi_burn.commands) and CLI --sim-pcie support."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from vfi_burn.backend import write_private_json
from vfi_burn.cli import build_parser, main
from vfi_burn.commands import (
    CommandIO,
    TargetSpec,
    execute_decode,
    execute_inspect,
    execute_simulate,
)
from vfi_burn.ffrecord import encode_ff
from vfi_burn.model import canonical_ff_prefix_sha256
from vfi_burn.sim_pcie import SimulatedGA100Adapter


class TestDecoupledCommands(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.work.chmod(0o700)

        die = {i: 0x1000 + i for i in range(8, 16)}
        base_rows = {
            222: encode_ff(1, 7, 0x0123),
            223: encode_ff(1, 19, 0x4290),
        }
        self.w_noop = encode_ff(1, 7, 0x0123)
        self.w_fbp = encode_ff(1, 19, 0x4200)

        plan_obj = {
            "schema_version": 1,
            "card": {
                "pci_id": "10de:20c2",
                "die_rows": {str(k): f"0x{v:08x}" for k, v in die.items()},
                "expected_masks": {"FBP_DISABLE": "0x852"},
                "defect_masks": {"FBP_DISABLE": "0x840", "FBPA_DISABLE": "0x0", "ROP_L2_DISABLE": "0x0"},
            },
            "fpf": {
                "first_row": 222,
                "last_row": 230,
                "empty_word": "0x00000000",
                "expected_next_row": 224,
                "baseline_ff_rows": {str(r): f"0x{w:08x}" for r, w in base_rows.items()},
                "baseline_ff_sha256": canonical_ff_prefix_sha256(base_rows),
            },
            "ceiling": {"base_sm": 122},
            "records": [
                {
                    "name": "NOOP_TEST",
                    "kind": "test",
                    "target": "NOOP_C1_B7",
                    "word": f"0x{self.w_noop:08x}",
                    "decode": "no-op test row",
                    "defect_sensitive": False,
                    "expected_cold": {"FBP_DISABLE": "0x852"},
                },
                {
                    "name": "UNLOCK_FBP",
                    "kind": "regular",
                    "target": "FBP_DISABLE",
                    "word": f"0x{self.w_fbp:08x}",
                    "decode": "enable non-defective FBPs",
                    "defect_sensitive": True,
                    "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
                    "expected_cold": {"FBP_DISABLE": "0x840"},
                },
            ],
        }
        self.plan_path = self.work / "plan.json"
        write_private_json(self.plan_path, plan_obj)

        sim = SimulatedGA100Adapter(
            die=die,
            rows=base_rows,
            fields={"FBP_DISABLE": 0x852},
        )
        self.sim_path = self.work / "sim.json"
        write_private_json(self.sim_path, sim.to_snapshot_dict())
        self.journal_path = self.work / "session.jsonl"

    def test_commands_run_directly_without_argparse(self):
        out: list[str] = []
        errs: list[str] = []
        cmd_io = CommandIO(show=out.append, err=errs.append, confirm=lambda *_: True, is_tty=lambda: True)

        rc = execute_decode(f"0x{self.w_fbp:08x}", io=cmd_io)
        self.assertEqual(rc, 0)
        self.assertTrue(any("chain=1 block=19" in line for line in out))

        rc2 = execute_inspect(
            plan_path=str(self.plan_path),
            target=TargetSpec(sim_pcie=str(self.sim_path)),
            sense=True,
            journal_path=str(self.journal_path),
            io=cmd_io,
        )
        self.assertEqual(rc2, 0)
        self.assertTrue(any("first free row 224" in line for line in out))

        saved = self.work / "sim_after.json"
        rc3 = execute_simulate(
            plan_path=str(self.plan_path),
            target=TargetSpec(sim_pcie=str(self.sim_path)),
            stage="all",
            journal_path=str(self.journal_path),
            save_fake=str(saved),
            io=cmd_io,
        )
        self.assertEqual(rc3, 0)
        self.assertTrue(saved.exists())
        saved_data = json.loads(saved.read_text(encoding="utf-8"))
        self.assertEqual(saved_data["rows"]["224"], self.w_noop)
        self.assertEqual(saved_data["rows"]["225"], self.w_fbp)

    def test_cli_parser_supports_sim_pcie_and_dispatches(self):
        ap = build_parser()
        args = ap.parse_args(
            [
                "inspect",
                "--plan",
                str(self.plan_path),
                "--sim-pcie",
                str(self.sim_path),
                "--sense",
            ]
        )
        self.assertEqual(args.cmd, "inspect")
        self.assertEqual(args.sim_pcie, str(self.sim_path))

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = main(
                [
                    "inspect",
                    "--plan",
                    str(self.plan_path),
                    "--sim-pcie",
                    str(self.sim_path),
                    "--sense",
                ]
            )
        self.assertEqual(rc, 0, err.getvalue())
        self.assertIn("first free row 224", out.getvalue())

    def test_commands_module_does_not_import_argparse(self):
        commands_src = (Path(__file__).resolve().parents[1] / "vfi_burn" / "commands.py").read_text(encoding="utf-8")
        self.assertNotIn("import argparse", commands_src)


if __name__ == "__main__":
    unittest.main()
