"""Goldilocks frames have UNSENSED option shadows (they read 0) until SENSE_CTRL.

The glitch mis-senses the row 0/1 control fuses (DSOV=0, EFP=0) and leaves the option
shadows unsensed; SENSE_CTRL re-senses them (EFP 0->1, DSOV stays 0). CMD_READ of fuse
rows is unaffected. Before this fix, inspect / menu preflight / passive dry-run compared
the zero shadows with the cold-boot masks and reported IDENTITY MISMATCH on a good frame,
and preflight compared them BEFORE its own SENSE even when a sense was requested.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from vfi_burn.console import format_status_report
from vfi_burn.engine import (
    Journal,
    Overrides,
    UNSENSED_SHADOWS_NOTE,
    option_shadows_unsensed,
    preflight,
    process_records,
)
from vfi_burn.ffrecord import FF_ROW_MAX, FF_ROW_MIN, encode_ff
from vfi_burn.fuses import DIE_ID_ROWS, OPTION_REGS, REG_DSOV_STATUS
from vfi_burn.ga100 import CMD_PROGRAM
from vfi_burn.interfaces import Health
from vfi_burn.model import BurnError, Plan, canonical_ff_prefix_sha256
from vfi_burn.sim_pcie import SimulatedGA100Adapter

BASE_ROWS = {222: encode_ff(1, 7, 0x0123), 223: encode_ff(1, 19, 0x4290)}
EXPECTED = {"FBP_DISABLE": 0x852, "FBPA_DISABLE": 0x2460, "ROP_L2_DISABLE": 0x0}
DEFECT = {"FBP_DISABLE": 0x840, "FBPA_DISABLE": 0x0, "ROP_L2_DISABLE": 0x0}


def _health(dsov, efp):
    return Health(fusectrl=0xE0040000, dsov=dsov, crc=0x55, plm=0xFFFFFFFF, plm_f8=0xFFFFFFFF,
                  en_sw=0, debugctrl=0, efp=efp)


class TestSignature(unittest.TestCase):
    def test_only_the_documented_pre_sense_state_counts_as_unsensed(self):
        zeros = {"FBP_DISABLE": 0, "FBPA_DISABLE": 0}
        self.assertTrue(option_shadows_unsensed(_health(0, 0), zeros))
        self.assertFalse(option_shadows_unsensed(_health(1, 0), zeros), "locked/normal boot: compare")
        self.assertFalse(option_shadows_unsensed(_health(0, 1), zeros), "already sensed: compare")
        self.assertFalse(option_shadows_unsensed(_health(0, None), zeros), "unknown EFP: compare")
        self.assertFalse(option_shadows_unsensed(None, zeros))
        self.assertFalse(option_shadows_unsensed(_health(0, 0), {"FBP_DISABLE": 0, "FBPA_DISABLE": 0x40}),
                         "partly sensed / odd frame: compare (and fail)")
        self.assertFalse(option_shadows_unsensed(_health(0, 0), {}))


class TestGoldilocksPreflight(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work = Path(self._tmp.name)
        self.work.chmod(0o700)
        self.journal_path = self.work / "journal.jsonl"
        self.die = {i: 0x1000 + i for i in DIE_ID_ROWS}

    def plan(self, expected=None):
        noop = encode_ff(1, 7, 0x0123)
        fbp = encode_ff(1, 19, 0x4200)
        obj = {
            "schema_version": 1,
            "card": {
                "pci_id": "10de:20c2",
                "die_rows": {str(k): f"0x{v:08x}" for k, v in self.die.items()},
                "expected_masks": {k: f"0x{v:x}" for k, v in (expected or EXPECTED).items()},
                "defect_masks": {k: f"0x{v:x}" for k, v in DEFECT.items()},
            },
            "fpf": {
                "first_row": FF_ROW_MIN, "last_row": FF_ROW_MAX, "empty_word": "0x00000000",
                "expected_next_row": 224,
                "baseline_ff_rows": {str(r): f"0x{w:08x}" for r, w in BASE_ROWS.items()},
                "baseline_ff_sha256": canonical_ff_prefix_sha256(BASE_ROWS),
            },
            "ceiling": {"base_sm": 122},
            "records": [
                {"name": "NOOP_TEST", "kind": "test", "target": "NOOP_C1_B7", "word": f"0x{noop:08x}",
                 "decode": "no-op", "defect_sensitive": False, "expected_cold": {"FBP_DISABLE": "0x852"}},
                {"name": "UNLOCK_FBP", "kind": "regular", "target": "FBP_DISABLE", "word": f"0x{fbp:08x}",
                 "decode": "FBP 0x852 -> 0x840", "defect_sensitive": True,
                 "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
                 "expected_cold": {"FBP_DISABLE": "0x840"}},
            ],
        }
        p = self.work / f"plan{len(list(self.work.iterdir()))}.json"
        p.write_text(json.dumps(obj))
        p.chmod(0o600)
        return Plan.load(p)

    def goldilocks(self):
        """Pre-SENSE Goldilocks frame (DSOV=0, EFP=0): option shadows read 0 until SENSE."""
        ad = SimulatedGA100Adapter(die=self.die, rows=dict(BASE_ROWS), fields=dict(EXPECTED))
        self.assertEqual(ad.field("FBP_DISABLE"), 0, "simulator must model unsensed shadows")
        return ad

    def logged(self):
        return [json.loads(l) for l in self.journal_path.read_text().splitlines() if l.strip()]

    # ---- read-only preflight (inspect, menu [3]) -------------------------------------
    def test_passive_preflight_in_goldilocks_defers_masks_with_note(self):
        notes = []
        preflight(self.plan(), self.goldilocks(), active_sense=False, show=notes.append)
        self.assertIn(UNSENSED_SHADOWS_NOTE, notes)

    def test_preflight_with_sense_compares_masks_after_sense(self):
        ad = self.goldilocks()
        preflight(self.plan(), ad, active_sense=True)  # previously: IDENTITY MISMATCH before sensing
        self.assertEqual(ad.pcie.sense_calls, 1)

    def test_wrong_masks_still_refused_after_sense(self):
        ad = self.goldilocks()
        with self.assertRaisesRegex(BurnError, "IDENTITY MISMATCH FBP_DISABLE"):
            preflight(self.plan(expected={**EXPECTED, "FBP_DISABLE": 0x850}), ad, active_sense=True)

    def test_sense_that_leaves_shadows_zero_is_refused(self):
        ad = self.goldilocks()
        real_sense = ad.active_sense_readonly

        def sense_without_shadows():
            real_sense()
            ad.pcie.unsensed = set(OPTION_REGS.values())  # EFP=1 but shadows still 0
        ad.active_sense_readonly = sense_without_shadows
        with self.assertRaisesRegex(BurnError, "IDENTITY MISMATCH"):
            preflight(self.plan(), ad, active_sense=True)

    def test_already_sensed_frame_keeps_full_identity_check(self):
        # Same window after an earlier --sense run: EFP=1, shadows sensed -> no exemption.
        sensed = SimulatedGA100Adapter(die=self.die, rows=dict(BASE_ROWS),
                                       fields={**EXPECTED, "FBP_DISABLE": 0x850}, health=_health(0, 1))
        self.assertEqual(sensed.field("FBP_DISABLE"), 0x850)
        with self.assertRaisesRegex(BurnError, "IDENTITY MISMATCH FBP_DISABLE"):
            preflight(self.plan(), sensed, active_sense=False)

    def test_partly_sensed_goldilocks_frame_is_not_exempt(self):
        ad = self.goldilocks()
        ad.pcie.unsensed.discard(OPTION_REGS["FBPA_DISABLE"])  # one shadow nonzero
        with self.assertRaisesRegex(BurnError, "IDENTITY MISMATCH FBP_DISABLE"):
            preflight(self.plan(), ad, active_sense=False)

    # ---- burn session ------------------------------------------------------------------
    def test_passive_dry_run_in_goldilocks_completes_and_journals_deferral(self):
        ad = self.goldilocks()
        with Journal(self.journal_path) as j:
            process_records(self.plan(), ad, ad, j, stage="all", confirm=lambda *_: True,
                            show=lambda *_: None, dry_run=True, run_active_sense=False)
        events = [e["event"] for e in self.logged()]
        self.assertIn("mask_check_deferred", events)
        self.assertEqual(ad.pcie.sense_calls, 0)
        self.assertEqual(ad.pcie.program_calls, 0)

    def test_burn_without_sense_on_unsensed_shadows_is_refused_before_any_program(self):
        ad = self.goldilocks()
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "SENSE_CTRL has not run"):
                process_records(self.plan(), ad, ad, j, stage="all", confirm=lambda *_: True,
                                show=lambda *_: None, overrides=Overrides(skip_active_sense=True))
        self.assertEqual(sum(1 for tx in ad.bus.writes() if tx.value == CMD_PROGRAM), 0)

    def test_burn_with_sense_compares_masks_and_burns(self):
        ad = self.goldilocks()
        with Journal(self.journal_path) as j:
            process_records(self.plan(), ad, ad, j, stage="all", confirm=lambda *_: True,
                            show=lambda *_: None)
        events = [e["event"] for e in self.logged()]
        self.assertNotIn("mask_check_deferred", events)
        self.assertEqual(ad.pcie.program_calls, 2)
        self.assertEqual(ad.field("FBP_DISABLE"), 0x852, "shadows are sensed after SENSE_CTRL")

    def test_status_report_labels_unsensed_shadows_before_sense_and_ok_after_sense(self):
        ad = self.goldilocks()
        plan = self.plan()
        before = "\n".join(format_status_report(plan, ad))
        self.assertIn("FBP_DISABLE=0x00000000 (unsensed pre-SENSE; cold exp 0x00000852)", before)
        self.assertNotIn("(exp 0x00000852)", before)

        # Partly sensed frame must NOT get the unsensed label.
        ad.pcie.unsensed.discard(OPTION_REGS["FBPA_DISABLE"])
        partly = "\n".join(format_status_report(plan, ad))
        self.assertIn("FBP_DISABLE=0x00000000 (exp 0x00000852)", partly)
        self.assertNotIn("unsensed pre-SENSE", partly)

        # After SENSE_CTRL, shadows match cold expectations as OK.
        ad.active_sense_readonly()
        after = "\n".join(format_status_report(plan, ad))
        self.assertIn("FBP_DISABLE=0x00000852 (OK)", after)
        self.assertNotIn("unsensed pre-SENSE", after)

    def test_sim_adapter_explicit_dsov_one_register_does_not_mark_shadows_unsensed(self):
        ad = SimulatedGA100Adapter(
            die=self.die,
            rows=dict(BASE_ROWS),
            fields=dict(EXPECTED),
            registers={REG_DSOV_STATUS: 1},
        )
        self.assertEqual(ad.field("FBP_DISABLE"), 0x852)


if __name__ == "__main__":
    unittest.main()
