"""Tests for simulated PCIe execution, HAL BAR0 transaction tracing, and SimulatedGA100Adapter."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from vfi_burn.console import format_pcie_write_plan
from vfi_burn.engine import Journal, process_records
from vfi_burn.ffrecord import encode_ff
from vfi_burn.fuses import (
    REG_DEBUGCTRL,
    REG_FUSEADDR,
    REG_FUSECTRL,
    REG_FUSEDATA,
    REG_FUSEWDATA,
)
from vfi_burn.ga100 import CMD_PROGRAM, CMD_READ, CMD_SENSE_CTRL, GA100FuseController
from vfi_burn.interfaces import FuseReader, FuseWriter, Health, MMIOBus
from vfi_burn.model import BurnError, Plan, canonical_ff_prefix_sha256
from vfi_burn.sim_pcie import (
    SimulatedGA100Adapter,
    SimulatedGA100PCIe,
    TracingMMIOBus,
    trace_hal_program_sequence,
)


class TestSimulatedPCIeAndHALTracing(unittest.TestCase):
    def test_simulated_pcie_implements_mmio_bus_and_guards(self):
        pcie = SimulatedGA100PCIe()
        self.assertIsInstance(pcie, MMIOBus)
        with self.assertRaises(BurnError):
            pcie.read32(2)
        with self.assertRaises(BurnError):
            pcie.write32(3, 1)
        with self.assertRaises(BurnError):
            pcie.write32(0, 0x1_0000_0000)
        with self.assertRaises(BurnError):
            pcie.write32(REG_DEBUGCTRL, 1)

    def test_tracing_bus_records_hal_pcie_transactions(self):
        pcie = SimulatedGA100PCIe(rows={222: encode_ff(1, 7, 0x0123)})
        bus = TracingMMIOBus(pcie)
        ctl = GA100FuseController(bus, read_only=False, sleep=lambda _: None)

        ctl.sense_ctrl()
        word = encode_ff(1, 19, 0x4200)
        got = ctl.program_row(223, word)
        self.assertEqual(got, word)

        writes = bus.writes()
        self.assertTrue(any(tx.offset == REG_FUSECTRL and tx.value == CMD_SENSE_CTRL for tx in writes))
        self.assertTrue(any(tx.offset == REG_FUSEADDR and tx.value == 223 for tx in writes))
        self.assertTrue(any(tx.offset == REG_FUSEWDATA and tx.value == word for tx in writes))
        self.assertTrue(any(tx.offset == REG_FUSECTRL and tx.value == CMD_PROGRAM for tx in writes))
        self.assertTrue(any(tx.offset == REG_FUSECTRL and tx.value == CMD_READ for tx in writes))
        self.assertFalse(any(tx.offset == REG_DEBUGCTRL for tx in writes))
        self.assertFalse(any(tx.offset == REG_FUSEDATA for tx in writes))

    def test_simulated_adapter_runs_full_engine_burn_through_real_hal(self):
        die = {i: 0x1000 + i for i in range(8, 16)}
        base_rows = {
            222: encode_ff(1, 7, 0x0123),
            223: encode_ff(1, 19, 0x4290),  # FBP_DISABLE = 0x852
        }
        w_noop = encode_ff(1, 7, 0x0123)
        w_fbp = encode_ff(1, 19, 0x4200)    # FBP_DISABLE = 0x840 (preserves defect 0x840)

        adapter = SimulatedGA100Adapter(
            die=die,
            rows=base_rows,
            fields={"FBP_DISABLE": 0x852},
            lnkcap2=0x06,
        )
        self.assertIsInstance(adapter, FuseReader)
        self.assertIsInstance(adapter, FuseWriter)
        self.assertTrue(adapter.simulation)
        self.assertEqual(adapter.field("LnkCap2"), 0x06)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            work.chmod(0o700)
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
                        "word": f"0x{w_noop:08x}",
                        "decode": "no-op test row",
                        "defect_sensitive": False,
                        "expected_cold": {"FBP_DISABLE": "0x852"},
                    },
                    {
                        "name": "UNLOCK_FBP",
                        "kind": "regular",
                        "target": "FBP_DISABLE",
                        "word": f"0x{w_fbp:08x}",
                        "decode": "enable non-defective FBPs",
                        "defect_sensitive": True,
                        "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
                        "expected_cold": {"FBP_DISABLE": "0x840"},
                    },
                ],
            }
            plan_path = work / "plan.json"
            plan_path.write_text(json.dumps(plan_obj))
            plan_path.chmod(0o600)
            plan = Plan.load(plan_path)
            journal_path = work / "session.jsonl"

            with Journal(journal_path) as j:
                next_ptr = process_records(
                    plan,
                    adapter,
                    adapter,
                    j,
                    stage="all",
                    confirm=lambda *_: True,
                    show=lambda *_: None,
                )

            self.assertEqual(next_ptr, 226)
            self.assertEqual(adapter.pcie.sense_calls, 1)
            self.assertEqual(adapter.pcie.program_calls, 2)
            self.assertEqual(adapter.pcie.rows[224], w_noop)
            self.assertEqual(adapter.pcie.rows[225], w_fbp)
            # Confirm real PCIe BAR0 transactions were generated by GA100FuseController.
            self.assertGreater(len(adapter.bus.writes()), 10)
            self.assertEqual(adapter.pcie.regs[REG_FUSEWDATA], 0)

    def test_trace_hal_program_sequence_and_format_write_plan(self):
        health_unsensed = Health(0xE0040000, 0, 0x55, 0xFFFFFFFF, 0xFFFFFFFF, 0, efp=0)
        word = encode_ff(1, 19, 0x4200)
        sense_w, prog_w, rb_w = trace_hal_program_sequence(224, word, health_unsensed)
        self.assertTrue(any(tx.offset == REG_FUSECTRL and tx.value == CMD_SENSE_CTRL for tx in sense_w))
        self.assertTrue(any(tx.offset == REG_FUSEWDATA and tx.value == word for tx in prog_w))
        self.assertTrue(any(tx.offset == REG_FUSECTRL and tx.value == CMD_PROGRAM for tx in prog_w))
        self.assertTrue(any(tx.offset == REG_FUSECTRL and tx.value == CMD_READ for tx in rb_w))

        # Already sensed -> no SENSE_CTRL writes generated.
        health_sensed = Health(0xE0040000, 0, 0x55, 0xFFFFFFFF, 0xFFFFFFFF, 1, efp=1, sbfe=0)
        sense_w2, _, _ = trace_hal_program_sequence(224, word, health_sensed)
        self.assertEqual(sense_w2, [])

        # Preview reflects HAL refusal for denied PCIe device-ID block (c2/b37).
        denied_word = encode_ff(2, 37, 0x1234)
        out = "\n".join(format_pcie_write_plan(224, denied_word, health_unsensed))
        self.assertIn("HAL REFUSAL", out)

    def test_snapshot_round_trip(self):
        from vfi_burn.backend import write_private_json
        from vfi_burn.sim_pcie import sim_pcie_from_file

        adapter = SimulatedGA100Adapter(
            rows={222: encode_ff(1, 7, 0x0123)},
            fields={"FBP_DISABLE": 0x852},
        )
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            work.chmod(0o700)
            snap_path = work / "sim.json"
            write_private_json(snap_path, adapter.to_snapshot_dict())
            reloaded = sim_pcie_from_file(snap_path)
            self.assertEqual(reloaded.fuse_row(222), encode_ff(1, 7, 0x0123))
            self.assertEqual(reloaded.field("FBP_DISABLE"), 0x852)


if __name__ == "__main__":
    unittest.main()
