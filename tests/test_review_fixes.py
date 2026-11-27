"""Regression tests for the review fixes. Each test fails on the pre-review code."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest

from vfi_burn.backend import FakeController
from vfi_burn.engine import Journal, process_records, reconcile, verify_cold_snapshot
from vfi_burn.ffrecord import FF_ROW_MIN, build_vfi_plan_from_records, encode_ff
from vfi_burn.ga100 import (
    REG_CTRL_WR_SECURE,
    REG_FUSEWDATA,
    GA100FuseController,
    GA100LiveAdapter,
)
from vfi_burn.mmio import REG_PLM_F8
from vfi_burn.model import BurnError, Plan, PreTriggerRefusal, Record, is_poison

from test_ga100 import FakeGA100MMIO

ROOT = Path(__file__).resolve().parents[1]
TPC_KEYS = [f"TPC_GPC{i}_DISABLE" for i in range(8)]


def _rec(name, word, **kw):
    base = dict(
        name=name, target=kw.pop("target", "RAW"), word=word, decode="test", kind="regular",
        defect_sensitive=False, mask_field=None, from_disable=None, to_disable=None,
        resulting_sm_count=None, expected_cold={}, link_cap2_before=None, link_cap2_after=None,
    )
    base.update(kw)
    return Record(**base)


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.work.chmod(0o700)
        data = json.loads((ROOT / "examples/plan.example.json").read_text())
        p = self.work / "plan.json"
        p.write_text(json.dumps(data))
        p.chmod(0o600)
        self.plan = Plan.load(p)
        self.journal_path = self.work / "j.jsonl"

    def logged(self):
        return [json.loads(x) for x in self.journal_path.read_text().splitlines()]

    def fake(self):
        return FakeController(fields={"GPC_DISABLE": 0x85},
                              rows={222: 0xABCD00DE, **{i: 0 for i in range(223, 228)}})


class TestWordDerivedDefectCheck(Base):
    """Genuine defects must be refused from the FF word itself, whatever the metadata says."""

    def plan_with(self, **kw):
        return replace(self.plan, **kw)

    def test_defect_reenable_without_mask_metadata_is_refused_even_with_override(self):
        # chain 1 block 8 holds GPC_DISABLE bits 0..7 at block bits 3..10 (+ FBPA bits 21..23).
        # GPC defect mask 0x04; data 0x0408 sets GPC bits 0 and 7 only -> re-enables defective GPC2.
        plan = self.plan_with(defect_masks={"GPC_DISABLE": 0x04, "FBPA_DISABLE": 0})
        rec = _rec("SNEAKY", encode_ff(1, 8, 0x0408), target="CONSOLE_ONE_LINER")
        for allow in (False, True):  # --override-undetermined-ceiling must NOT unlock this
            with self.subTest(allow_undetermined_ceiling=allow):
                with self.assertRaisesRegex(BurnError, "genuinely defective GPC_DISABLE"):
                    reconcile(plan, rec, {"GPC_DISABLE": 0x85}, allow_undetermined_ceiling=allow)

    def test_unknown_defect_mask_for_touched_field_is_refused(self):
        plan = self.plan_with(defect_masks={"GPC_DISABLE": 0x04})  # no FBPA mask, block 8 has FBPA bits
        rec = _rec("X", encode_ff(1, 8, 0x0428))
        with self.assertRaisesRegex(BurnError, "no genuine defect mask"):
            reconcile(plan, rec, {"GPC_DISABLE": 0x85})

    def test_mask_update_must_match_what_the_word_writes(self):
        plan = self.plan_with(defect_masks={"GPC_DISABLE": 0x04, "FBPA_DISABLE": 0})
        # Declares 0x85 -> 0x84, but data 0x0020 writes GPC=0x04 (enables GPC0 and GPC7).
        rec = _rec("LIAR", encode_ff(1, 8, 0x0020), defect_sensitive=True,
                   mask_field="GPC_DISABLE", from_disable=0x85, to_disable=0x84,
                   resulting_sm_count=75)
        with self.assertRaisesRegex(BurnError, "disagrees with what the FF word actually writes"):
            reconcile(plan, rec, {"GPC_DISABLE": 0x85})

    def test_shared_tpc_block_cannot_bypass_sm_ceiling(self):
        # chain 1 block 12 = bits 192..207: TPC_GPC0[5..7], TPC_GPC1[0..7], TPC_GPC2[0..4].
        masks = {"GPC_DISABLE": 0x00, **{k: 0xFF for k in TPC_KEYS}}
        plan = self.plan_with(
            expected_masks=masks, base_sm_ceiling=10,
            defect_masks={"GPC_DISABLE": 0, "TPC_GPC0_DISABLE": 0, "TPC_GPC1_DISABLE": 0,
                          "TPC_GPC2_DISABLE": 0},
        )
        rec = _rec("SHARED_TPC", encode_ff(1, 12, 0x0000))  # multi-field: no mask_update possible
        with self.assertRaisesRegex(BurnError, r"yields 32 SM, beyond supplied ceiling 10"):
            reconcile(plan, rec, dict(masks))


    def test_single_field_gpc_record_also_gets_word_derived_sm_check(self):
        masks = {"GPC_DISABLE": 0x85, **{k: 0x00 for k in TPC_KEYS}}  # 5 GPCs x 16 = 80 SM
        plan = self.plan_with(expected_masks=masks, base_sm_ceiling=90,
                              defect_masks={"GPC_DISABLE": 0x04, "FBPA_DISABLE": 0})
        # 0x85 -> 0x84 enables GPC0 -> 96 SM; metadata under-declares 80 to slip past the ceiling.
        rec = _rec("UNDER_DECLARED", encode_ff(1, 8, 0x0420), defect_sensitive=True,
                   mask_field="GPC_DISABLE", from_disable=0x85, to_disable=0x84,
                   resulting_sm_count=80)
        with self.assertRaisesRegex(BurnError, "declared resulting_sm_count=80 but the FF word yields 96"):
            reconcile(plan, rec, dict(masks))

    def test_mask_update_for_different_field_than_mapped_ff_block_is_refused(self):
        plan = self.plan_with(defect_masks={"GPC_DISABLE": 0x04, "FBP_DISABLE": 0x840, "ROP_L2_DISABLE": 0})
        # FF word writes FBP_DISABLE/ROP_L2_DISABLE (c1/b19), NOT GPC_DISABLE (c1/b8).
        rec = _rec("WRONG_BLOCK", encode_ff(1, 19, 0x4200), defect_sensitive=True,
                   mask_field="GPC_DISABLE", from_disable=0x85, to_disable=0x84,
                   resulting_sm_count=75)
        with self.assertRaisesRegex(BurnError, "mask_update targets GPC_DISABLE but the FF word writes"):
            reconcile(plan, rec, {"GPC_DISABLE": 0x85, "FBP_DISABLE": 0x852, "ROP_L2_DISABLE": 0})

    def test_mask_update_on_unmapped_ff_block_is_refused(self):
        plan = self.plan_with(defect_masks={"GPC_DISABLE": 0x04})
        # Chain 1 block 99 maps no known option field, so it cannot back a GPC_DISABLE mask_update.
        rec = _rec("UNMAPPED_BLOCK", encode_ff(1, 99, 0x0420), defect_sensitive=True,
                   mask_field="GPC_DISABLE", from_disable=0x85, to_disable=0x84,
                   resulting_sm_count=75)
        with self.assertRaisesRegex(BurnError, "mask_update targets GPC_DISABLE but the FF word writes no known option field"):
            reconcile(plan, rec, {"GPC_DISABLE": 0x85})

    def test_untracked_compute_field_missing_from_expected_masks_is_refused(self):
        # Block 15 (bits 240..255) carries TPC_GPC6[5..7], TPC_GPC7[0..7], and PES[0..4];
        # defect masks are present, but expected_masks lacks TPC_GPC6/TPC_GPC7.
        plan = self.plan_with(
            expected_masks={"GPC_DISABLE": 0x00},
            defect_masks={"TPC_GPC6_DISABLE": 0, "TPC_GPC7_DISABLE": 0, "PES_DISABLE": 0},
        )
        rec = _rec("UNTRACKED_TPC7", encode_ff(1, 15, 0x0000))
        with self.assertRaisesRegex(BurnError, "resulting SM count undetermined"):
            reconcile(plan, rec, {"GPC_DISABLE": 0x00})


class TestSignalsDeferredDuringWrite(Base):
    def test_ctrl_c_mid_write_is_held_until_outcome_is_journaled(self):
        ctl = self.fake()
        real_write = ctl.write_one_record

        def write_then_sigint(row, word):
            os.kill(os.getpid(), signal.SIGINT)  # operator hits Ctrl-C mid-program
            real_write(row, word)

        ctl.write_one_record = write_then_sigint
        with Journal(self.journal_path) as j:
            with self.assertRaises(KeyboardInterrupt):
                process_records(self.plan, ctl, ctl, j, stage="test",
                                confirm=lambda *_: True, show=lambda *_: None)
        events = [e["event"] for e in self.logged()]
        self.assertIn("verified_readback_only", events)  # outcome recorded before the interrupt
        self.assertEqual(ctl.rows[223], self.plan.records[0].word)


class TestGA100Controller(Base):
    def armed(self, rows=None):
        mmio = FakeGA100MMIO(rows={8: 0x1008, 222: encode_ff(1, 5, 0x9867), **(rows or {})})
        ctl = GA100FuseController(mmio, read_only=False, sleep=lambda _: None)
        ctl.sense_ctrl()
        return mmio, ctl

    def test_no_burn_latch_restored_after_program(self):
        mmio, ctl = self.armed()
        word = encode_ff(1, 19, 0x4200)
        ctl.program_row(223, word)
        self.assertEqual(mmio.rows[223], word)
        self.assertEqual(mmio.regs[REG_FUSEWDATA], 0, "FUSEWDATA must not stay loaded with the word")

    def test_non_blank_target_is_a_pre_trigger_refusal_with_no_program(self):
        mmio, ctl = self.armed(rows={223: 0x11})
        with self.assertRaises(PreTriggerRefusal):
            ctl.program_row(223, encode_ff(1, 19, 0x4200))
        self.assertEqual(mmio.program_calls, 0)

    def test_arm_does_not_reopen_partial_plm_f8_in_software(self):
        mmio, ctl = self.armed()
        mmio.regs[REG_PLM_F8] = 0x7FFFFFFF  # bit 0 still open, but not the full goldilocks value
        with self.assertRaisesRegex(BurnError, "refusing to re-open"):
            ctl.arm()
        self.assertNotIn(REG_CTRL_WR_SECURE, {a for a, _ in mmio.writes})

    def test_pcie_option_regs_read_bit0_directly(self):
        # Real CMP 170HX BAR0 readouts (ffprobe_all.txt / pcie_state.py) hold 0x1 when locked
        # and 0x0 when cleared in bit 0 (NOT shifted to FF chain-2 block-local bits 1/9/6/7).
        adapter = GA100LiveAdapter.__new__(GA100LiveAdapter)
        mmio = FakeGA100MMIO()
        adapter.controller = GA100FuseController(mmio, read_only=True)
        mmio.regs[0x00820250] = 0x00000001
        mmio.regs[0x0082057C] = 0x00000001
        mmio.regs[0x00820580] = 0x00000001
        self.assertEqual(adapter.field("DISABLE_GEN3_SPEED"), 1)
        self.assertEqual(adapter.field("BOOT_GEN23_DIS"), 1)
        self.assertEqual(adapter.field("BOOT_GEN3_DIS"), 1)
        mmio.regs[0x00820250] = 0x00000000
        mmio.regs[0x00820580] = 0x00000000
        self.assertEqual(adapter.field("DISABLE_GEN3_SPEED"), 0)
        self.assertEqual(adapter.field("BOOT_GEN3_DIS"), 0)


class TestEngineGates(Base):
    def test_pre_trigger_refusal_is_not_journaled_as_partial_halt(self):
        ctl = self.fake()

        def refuse(row, word):
            raise PreTriggerRefusal("SBFE did not clear")

        ctl.write_one_record = refuse
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "NO write issued"):
                process_records(self.plan, ctl, ctl, j, stage="test",
                                confirm=lambda *_: True, show=lambda *_: None)
        events = self.logged()
        self.assertTrue(any(e["event"] == "aborted_pre_trigger" and e["write_issued"] is False for e in events))
        self.assertFalse(any(e["event"] == "HALT_UNVERIFIED" for e in events))

    def test_live_reader_refuses_non_ff_word_before_sense(self):
        ctl = self.fake()
        ctl.simulation = False
        bad = replace(self.plan, records=(replace(self.plan.records[0], word=0x13572468),
                                          self.plan.records[1]))  # IFF bit 5 set
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "not a valid GA100 FF record"):
                process_records(bad, ctl, ctl, j, stage="test", confirm=lambda *_: True,
                                show=lambda *_: None, dry_run=True)
        self.assertEqual(ctl.events, [])

    def test_poison_matches_every_bad_prefix(self):
        for v in (0xBADF5040, 0xBAD00001, 0xBAD1ACC0, 0xBADE0000):
            self.assertTrue(is_poison(v), hex(v))
        self.assertFalse(is_poison(0xBAE00000))

    def test_plan_rows_must_be_inside_ff_region(self):
        data = json.loads((ROOT / "examples/plan.example.json").read_text())
        data["fpf"]["first_row"] = 200
        p = self.work / "bad.json"
        p.write_text(json.dumps(data))
        p.chmod(0o600)
        with self.assertRaisesRegex(BurnError, "inside the GA100 FF region"):
            Plan.load(p)

    def test_cold_verify_real_fails_when_burned_row_did_not_persist(self):
        ctl = self.fake()
        full = replace(self.plan, final_expected_cold={
            "GPC_DISABLE": 0x84, "FBP_DISABLE": 0, "FBIO_LAYOUT": 1, "ROP_L2_LAYOUT": 1,
            "FBPA_LAYOUT": 1, "MAGIC_D": 1, "BOOT_GEN3_DIS": 0, "DISABLE_GEN3_SPEED": 0,
            "LnkCap2": 3, "SM_COUNT": 75,
        })
        with Journal(self.journal_path) as j:
            process_records(full, ctl, ctl, j, stage="all", confirm=lambda *_: True, show=lambda *_: None)
        cold = dict(
            pci_id=full.pci_id, die_rows={str(k): v for k, v in full.die_rows.items()},
            fields=dict(full.final_expected_cold), cold_boot=True,
            boot_id="boot-2", previous_boot_id="fake-boot-1",
            fpf_rows={"223": hex(full.records[0].word)},  # row 224 reads 0 after cold boot
        )
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, r"cold FPF row 224 \(0x00000000\)"):
                verify_cold_snapshot(full, cold, stage="real", journal=j)
        cold["fpf_rows"] = {}
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "no fpf_rows"):
                verify_cold_snapshot(full, cold, stage="real", journal=j)


class TestFakeRoundTrip(Base):
    def test_save_fake_reloads_with_optional_health_fields(self):
        from vfi_burn.backend import dump_fake, fake_from_file
        ctl = self.fake()  # Health.efp/dfp/... default to None -> JSON null
        out = self.work / "after.json"
        dump_fake(ctl, out)
        self.assertIsNone(fake_from_file(out).state.efp)


class TestPlanBuilder(Base):
    def base_args(self, *, expected_masks=None, defect_masks=None, row=224, old=0x4290):
        die_rows = {i: 0x1000 + i for i in range(8, 16)}
        ff_rows = {r: 0 for r in range(FF_ROW_MIN, 487)}
        ff_rows[FF_ROW_MIN] = encode_ff(1, 19, 0x4290)
        return dict(
            ff_rows=ff_rows,
            die_rows_int=die_rows,
            expected_masks=expected_masks if expected_masks is not None else {
                "FBP_DISABLE": 0x852,
                "ROP_L2_DISABLE": 0,
            },
            defect_masks=defect_masks if defect_masks is not None else {
                "FBP_DISABLE": 0x840,
                "ROP_L2_DISABLE": 0,
            },
            planned_items=[
                {
                    "row": row,
                    "chain": 1,
                    "block": 19,
                    "old": old,
                    "new": 0x4200,
                    "word": encode_ff(1, 19, 0x4200),
                    "fields": ["fbp_disable"],
                }
            ],
        )

    def test_base_sm_ceiling_auto_computed_from_defects(self):
        # 1 defective GPC + 3 defective TPCs on GPC 1 -> (7*8 - 3)*2 = 106 SMs
        plan = build_vfi_plan_from_records(
            **self.base_args(
                defect_masks={
                    "GPC_DISABLE": 0x01,
                    "TPC_GPC1_DISABLE": 0x07,
                    "FBP_DISABLE": 0x840,
                    "ROP_L2_DISABLE": 0,
                }
            )
        )
        self.assertEqual(plan["ceiling"]["base_sm"], 106)

    def test_stale_rows_and_old_values_are_refused(self):
        with self.assertRaisesRegex(BurnError, "not contiguous with expected row 224"):
            build_vfi_plan_from_records(**self.base_args(row=230))
        with self.assertRaisesRegex(BurnError, "planned old=0x4291"):
            build_vfi_plan_from_records(**self.base_args(old=0x4291))

    def test_dump_disagreeing_with_ff_replay_fails_closed(self):
        with self.assertRaisesRegex(BurnError, "disagrees with FF replay"):
            build_vfi_plan_from_records(
                **self.base_args(expected_masks={"FBP_DISABLE": 0x850, "ROP_L2_DISABLE": 0})
            )

    def test_lnkcap2_expectation_unchanged_without_pcie_records(self):
        plan = build_vfi_plan_from_records(**self.base_args(), link_cap2_before=0x02)
        self.assertEqual(plan["final_expected_cold"]["LnkCap2"], "0x2")


if __name__ == "__main__":
    unittest.main()
