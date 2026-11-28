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
    POST_SENSE_TPC_CLEARED_NOTE,
    UNSENSED_SHADOWS_NOTE,
    option_shadows_unsensed,
    post_sense_tpc_shadows_cleared,
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
        self.assertIn("GATE SUMMARY : PASS", before)

        # Partly sensed frame must NOT get the unsensed label and must NOT report GATE SUMMARY : PASS.
        ad.pcie.unsensed.discard(OPTION_REGS["FBPA_DISABLE"])
        partly = "\n".join(format_status_report(plan, ad))
        self.assertIn("FBP_DISABLE=0x00000000 (exp 0x00000852)", partly)
        self.assertNotIn("unsensed pre-SENSE", partly)
        self.assertIn("GATE SUMMARY : NOT WRITE-READY:", partly)
        self.assertIn("mask FBP_DISABLE=0x00000000!=exp(0x00000852)", partly)

        # After SENSE_CTRL, shadows match cold expectations as OK.
        ad.active_sense_readonly()
        after = "\n".join(format_status_report(plan, ad))
        self.assertIn("FBP_DISABLE=0x00000852 (OK)", after)
        self.assertNotIn("unsensed pre-SENSE", after)
        self.assertIn("GATE SUMMARY : PASS", after)

    def test_sim_adapter_explicit_dsov_one_register_does_not_mark_shadows_unsensed(self):
        ad = SimulatedGA100Adapter(
            die=self.die,
            rows=dict(BASE_ROWS),
            fields=dict(EXPECTED),
            registers={REG_DSOV_STATUS: 1},
        )
        self.assertEqual(ad.field("FBP_DISABLE"), 0x852)


class TestPostSenseClearedTpcShadows(unittest.TestCase):
    """Observed CMP 170HX 8 GB quirk: CMD_SENSE_CTRL clears OPT_TPC_GPC0..7_DISABLE to 0."""

    FULL_EXPECTED = {
        "FBP_DISABLE": 0x852,
        "FBPA_DISABLE": 0x00C0330C,
        "GPC_DISABLE": 0x85,
        "ROP_L2_DISABLE": 0x00C0330C,
        "TPC_GPC0_DISABLE": 0xFF,
        "TPC_GPC1_DISABLE": 0x01,
        "TPC_GPC2_DISABLE": 0xFF,
        "TPC_GPC3_DISABLE": 0x01,
        "TPC_GPC4_DISABLE": 0x00,
        "TPC_GPC5_DISABLE": 0x00,
        "TPC_GPC6_DISABLE": 0x01,
        "TPC_GPC7_DISABLE": 0xFF,
    }
    FULL_DEFECT = {
        "FBP_DISABLE": 0x840,
        "FBPA_DISABLE": 0x00C03000,
        "GPC_DISABLE": 0x04,
        "ROP_L2_DISABLE": 0x00C03000,
        "TPC_GPC0_DISABLE": 0x00,
        "TPC_GPC1_DISABLE": 0x00,
        "TPC_GPC2_DISABLE": 0xFF,
        "TPC_GPC3_DISABLE": 0x00,
        "TPC_GPC4_DISABLE": 0x00,
        "TPC_GPC5_DISABLE": 0x00,
        "TPC_GPC6_DISABLE": 0x00,
        "TPC_GPC7_DISABLE": 0x80,
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work = Path(self._tmp.name)
        self.work.chmod(0o700)
        self.journal_path = self.work / "journal.jsonl"
        self.die = {i: 0x1000 + i for i in DIE_ID_ROWS}

    def _post_sense_health(self, *, dsov=0, efp=1, en_sw=1, resense_disable=1):
        return Health(
            fusectrl=0xE0040000,
            dsov=dsov,
            crc=0x55,
            plm=0xFFFFFFFF,
            plm_f8=0xFFFFFFFF,
            en_sw=en_sw,
            debugctrl=0,
            efp=efp,
            resense_disable=resense_disable,
        )

    BASE_ROWS_8GB = {222: encode_ff(1, 7, 0x0198), 223: encode_ff(1, 19, 0x4290)}

    def _plan(self, expected=None):
        noop = encode_ff(1, 7, 0x0198)
        fbp = encode_ff(1, 19, 0x4200)
        exp = expected or self.FULL_EXPECTED
        obj = {
            "schema_version": 1,
            "card": {
                "pci_id": "10de:20c2",
                "die_rows": {str(k): f"0x{v:08x}" for k, v in self.die.items()},
                "expected_masks": {k: f"0x{v:x}" for k, v in sorted(exp.items())},
                "defect_masks": {k: f"0x{v:x}" for k, v in sorted(self.FULL_DEFECT.items())},
            },
            "fpf": {
                "first_row": FF_ROW_MIN,
                "last_row": FF_ROW_MAX,
                "empty_word": "0x00000000",
                "expected_next_row": 224,
                "baseline_ff_rows": {str(r): f"0x{w:08x}" for r, w in self.BASE_ROWS_8GB.items()},
                "baseline_ff_sha256": canonical_ff_prefix_sha256(self.BASE_ROWS_8GB),
            },
            "ceiling": {"base_sm": 110},
            "records": [
                {
                    "name": "NOOP_TEST",
                    "kind": "test",
                    "target": "NOOP_C1_B7",
                    "word": f"0x{noop:08x}",
                    "decode": "no-op",
                    "defect_sensitive": False,
                    "expected_cold": {"FBP_DISABLE": "0x852"},
                },
                {
                    "name": "UNLOCK_FBP",
                    "kind": "regular",
                    "target": "FBP_DISABLE",
                    "word": f"0x{fbp:08x}",
                    "decode": "FBP 0x852 -> 0x840",
                    "defect_sensitive": True,
                    "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
                    "expected_cold": {"FBP_DISABLE": "0x840"},
                },
            ],
        }
        p = self.work / f"plan_tpc_{len(list(self.work.iterdir()))}.json"
        p.write_text(json.dumps(obj))
        p.chmod(0o600)
        return Plan.load(p)

    def _sensed_8gb_goldilocks(self, *, fields=None):
        """Model the real CMP 170HX 8 GB frame: shadows already sensed pre-SENSE, TPCs zero on SENSE."""
        defect_regs = {
            OPTION_REGS["TPC_GPC2_DEFECTIVE"]: 0xFF,
            OPTION_REGS["TPC_GPC7_DEFECTIVE"]: 0x80,
            OPTION_REGS["FBP_DEFECTIVE"]: 0x840,
            OPTION_REGS["TPC_GPC0_STATUS"]: 0xFF,
            OPTION_REGS["TPC_GPC2_STATUS"]: 0xFF,
        }
        return SimulatedGA100Adapter(
            die=self.die,
            rows=dict(self.BASE_ROWS_8GB),
            fields=dict(fields or self.FULL_EXPECTED),
            registers=defect_regs,
            unsensed_option_regs=(),
            clear_tpc_shadows_on_sense=True,
        )

    def _logged(self):
        return [json.loads(l) for l in self.journal_path.read_text().splitlines() if l.strip()]

    def test_signature_requires_exact_post_sense_state(self):
        exp = {"FBP_DISABLE": 0x852, "TPC_GPC0_DISABLE": 0xFF, "TPC_GPC1_DISABLE": 0x01}
        cleared = {"FBP_DISABLE": 0x852, "TPC_GPC0_DISABLE": 0, "TPC_GPC1_DISABLE": 0}
        ok_h = self._post_sense_health()
        self.assertTrue(post_sense_tpc_shadows_cleared(ok_h, exp, cleared))

        # Controller state gates (fail closed)
        self.assertFalse(post_sense_tpc_shadows_cleared(None, exp, cleared))
        self.assertFalse(post_sense_tpc_shadows_cleared(self._post_sense_health(dsov=1), exp, cleared))
        self.assertFalse(post_sense_tpc_shadows_cleared(self._post_sense_health(efp=0), exp, cleared))
        self.assertFalse(
            post_sense_tpc_shadows_cleared(self._post_sense_health(en_sw=0), exp, cleared),
            "permanently-clear-DSOV cold boot (EN_SW=0) must NOT match",
        )
        self.assertFalse(post_sense_tpc_shadows_cleared(self._post_sense_health(resense_disable=0), exp, cleared))
        self.assertFalse(post_sense_tpc_shadows_cleared(self._post_sense_health(resense_disable=None), exp, cleared))

        # Partial TPC mismatch or non-TPC mismatch must fail closed
        self.assertFalse(
            post_sense_tpc_shadows_cleared(ok_h, exp, {"FBP_DISABLE": 0x852, "TPC_GPC0_DISABLE": 0, "TPC_GPC1_DISABLE": 1})
        )
        self.assertFalse(
            post_sense_tpc_shadows_cleared(ok_h, exp, {"FBP_DISABLE": 0x840, "TPC_GPC0_DISABLE": 0, "TPC_GPC1_DISABLE": 0})
        )
        # All-zero non-TPC shadow cannot vouch for a sensed frame
        self.assertFalse(
            post_sense_tpc_shadows_cleared(
                ok_h,
                {"FBP_DISABLE": 0, "TPC_GPC0_DISABLE": 0xFF},
                {"FBP_DISABLE": 0, "TPC_GPC0_DISABLE": 0},
            )
        )
        # TPC-only or non-TPC-only plans must not match
        self.assertFalse(
            post_sense_tpc_shadows_cleared(ok_h, {"TPC_GPC0_DISABLE": 0xFF}, {"TPC_GPC0_DISABLE": 0})
        )
        self.assertFalse(
            post_sense_tpc_shadows_cleared(ok_h, {"FBP_DISABLE": 0x852}, {"FBP_DISABLE": 0x852})
        )
        # No TPC expected nonzero -> normal match path, not cleared quirk
        self.assertFalse(
            post_sense_tpc_shadows_cleared(
                ok_h,
                {"FBP_DISABLE": 0x852, "TPC_GPC0_DISABLE": 0},
                {"FBP_DISABLE": 0x852, "TPC_GPC0_DISABLE": 0},
            )
        )

    def test_manual_sense_then_inspect_dry_run_and_burn(self):
        ad = self._sensed_8gb_goldilocks()
        plan = self._plan()

        # [3] Pre-SENSE: all 16 masks match OK.
        pre_report = "\n".join(format_status_report(plan, ad))
        self.assertIn("TPC_GPC0_DISABLE=0x000000ff (OK)", pre_report)
        self.assertIn("GATE SUMMARY : PASS", pre_report)
        preflight(plan, ad, active_sense=False)

        # [4] Active Sense: clears TPC_GPC0..7_DISABLE & STATUS to 0; defects & non-TPC stay intact.
        ad.active_sense_readonly()
        self.assertEqual(ad.field("TPC_GPC0_DISABLE"), 0)
        self.assertEqual(ad.field("TPC_GPC7_DISABLE"), 0)
        self.assertEqual(ad.register(OPTION_REGS["TPC_GPC2_DEFECTIVE"]), 0xFF)
        self.assertEqual(ad.register(OPTION_REGS["TPC_GPC7_DEFECTIVE"]), 0x80)
        self.assertEqual(ad.field("FBP_DISABLE"), 0x852)

        # [3] Post-SENSE: status labels TPCs as cleared post-SENSE and passes preflight with note.
        post_report = "\n".join(format_status_report(plan, ad))
        self.assertIn("TPC_GPC0_DISABLE=0x00000000 (cleared post-SENSE; cold exp 0x000000ff)", post_report)
        self.assertIn("TPC_GPC4_DISABLE=0x00000000 (OK)", post_report)
        self.assertIn("FBP_DISABLE=0x00000852 (OK)", post_report)
        self.assertIn("GATE SUMMARY : PASS", post_report)
        notes: list[str] = []
        preflight(plan, ad, active_sense=False, show=notes.append)
        self.assertIn(POST_SENSE_TPC_CLEARED_NOTE, notes)

        # [5] Post-SENSE passive dry-run: succeeds and journals post_sense_tpc_shadows_cleared.
        with Journal(self.journal_path) as j:
            process_records(
                plan, ad, ad, j, stage="all", confirm=lambda *_: True,
                show=lambda *_: None, dry_run=True, run_active_sense=False,
            )
        events = [e["event"] for e in self._logged()]
        self.assertIn("post_sense_tpc_shadows_cleared", events)
        self.assertNotIn("mask_check_deferred", events)
        self.assertEqual(ad.pcie.program_calls, 0)

        # [6] Post-SENSE burn: succeeds without needing mask_mismatch override.
        with Journal(self.journal_path) as j:
            process_records(
                plan, ad, ad, j, stage="all", confirm=lambda *_: True,
                show=lambda *_: None,
            )
        self.assertEqual(ad.pcie.program_calls, 2)

    def test_direct_burn_from_pre_sense_verifies_tpc_before_sense_and_succeeds(self):
        ad = self._sensed_8gb_goldilocks()
        plan = self._plan()
        with Journal(self.journal_path) as j:
            process_records(
                plan, ad, ad, j, stage="all", confirm=lambda *_: True,
                show=lambda *_: None,
            )
        events = [e["event"] for e in self._logged()]
        self.assertIn("post_sense_tpc_shadows_cleared", events)
        self.assertEqual(ad.pcie.sense_calls, 1)
        self.assertEqual(ad.pcie.program_calls, 2)

    def test_pre_sense_tpc_mismatch_refused_before_sense_ctrl(self):
        # If TPC_GPC0_DISABLE already disagrees before SENSE_CTRL on a sensed pre-SENSE frame,
        # refuse BEFORE issuing SENSE_CTRL (which would clear TPC_GPC0..7_DISABLE to 0).
        bad_fields = {**self.FULL_EXPECTED, "TPC_GPC0_DISABLE": 0x00}
        ad = self._sensed_8gb_goldilocks(fields=bad_fields)
        with self.assertRaisesRegex(BurnError, "IDENTITY MISMATCH TPC_GPC0_DISABLE"):
            preflight(self._plan(), ad, active_sense=True)
        self.assertEqual(ad.pcie.sense_calls, 0)

        ad2 = self._sensed_8gb_goldilocks(fields=bad_fields)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "IDENTITY MISMATCH TPC_GPC0_DISABLE"):
                process_records(
                    self._plan(), ad2, ad2, j, stage="all",
                    confirm=lambda *_: True, show=lambda *_: None,
                )
        self.assertEqual(ad2.pcie.sense_calls, 0)
        self.assertEqual(ad2.pcie.program_calls, 0)

    def test_post_sense_partial_tpc_or_non_tpc_mismatch_still_refused(self):
        ad = self._sensed_8gb_goldilocks()
        ad.active_sense_readonly()

        # Partial TPC mismatch: one TPC nonzero while others zero -> refused & NOT WRITE-READY.
        ad.pcie.regs[OPTION_REGS["TPC_GPC2_DISABLE"]] = 0xFF
        report = "\n".join(format_status_report(self._plan(), ad))
        self.assertIn("GATE SUMMARY : NOT WRITE-READY:", report)
        with self.assertRaisesRegex(BurnError, "IDENTITY MISMATCH TPC_GPC0_DISABLE"):
            preflight(self._plan(), ad, active_sense=False)

        # Non-TPC mismatch: FBP_DISABLE differs -> refused & NOT WRITE-READY.
        ad.pcie.regs[OPTION_REGS["TPC_GPC2_DISABLE"]] = 0x00
        ad.pcie.regs[OPTION_REGS["FBP_DISABLE"]] = 0x840
        report2 = "\n".join(format_status_report(self._plan(), ad))
        self.assertIn("GATE SUMMARY : NOT WRITE-READY:", report2)
        with self.assertRaisesRegex(BurnError, "IDENTITY MISMATCH FBP_DISABLE"):
            preflight(self._plan(), ad, active_sense=False)

    def test_cold_boot_permanent_dsov_zero_with_zero_tpcs_still_refused(self):
        # EN_SW=0 (cold boot on permanently clear DSOV card) must never trigger post-SENSE exemption.
        cleared_fields = {
            **self.FULL_EXPECTED,
            **{f"TPC_GPC{i}_DISABLE": 0 for i in range(8)},
        }
        ad = SimulatedGA100Adapter(
            die=self.die,
            rows=dict(BASE_ROWS),
            fields=cleared_fields,
            health=self._post_sense_health(en_sw=0, resense_disable=1),
        )
        with self.assertRaisesRegex(BurnError, "IDENTITY MISMATCH TPC_GPC0_DISABLE"):
            preflight(self._plan(), ad, active_sense=False)


if __name__ == "__main__":
    unittest.main()
