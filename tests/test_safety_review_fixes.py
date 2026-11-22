"""Regression tests for all CMP170HX safety review fixes.

Every refusal test asserts BOTH that BurnError/PreTriggerRefusal is raised AND that
the irreversible CMD_PROGRAM primitive was called ZERO times (or exactly once with no
retry/subsequent record for post-trigger failure tests).
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from vfi_burn.backend import Host, has_write_capability
from vfi_burn.console import format_status_report
from vfi_burn.engine import (
    Journal,
    Overrides,
    check_plan,
    process_records,
    reconcile,
    verify_cold_snapshot,
)
from vfi_burn.ffrecord import (
    FF_ROW_MIN,
    encode_ff,
    ff_word_touches_pcie,
    import_ffcalc_plan,
    replay_ff_rows,
)
from vfi_burn.ga100 import (
    BDFLock,
    GA100FuseController,
    REG_FUSECTRL,
    REG_FUSEWDATA,
    REG_PLM_F8,
)
from vfi_burn.model import (
    BurnError,
    Plan,
    PreTriggerRefusal,
    Record,
    canonical_ff_prefix_sha256,
)

from test_ga100 import FakeGA100MMIO


DIE_ROWS = {i: 0x1000 + i for i in range(8, 16)}


class LiveMMIOAdapter:
    """Realistic non-simulation adapter backed by FakeGA100MMIO that counts CMD_PROGRAM calls."""

    simulation = False
    write_enabled = True
    writer_verified = True

    def __init__(
        self,
        mmio: FakeGA100MMIO,
        *,
        boot_id: str = "boot-live-1",
        fields: dict[str, int] | None = None,
        pre_trigger_check=None,
    ) -> None:
        self.mmio = mmio
        self.ctl = GA100FuseController(mmio, read_only=False, sleep=lambda _: None)
        self._boot = boot_id
        self._fields = dict(fields or {"FBP_DISABLE": 0x852, "LnkCap2": 0x06})
        self._pre_trigger_check = pre_trigger_check

    def pci_id(self) -> str:
        return "10de:20c2"

    def die_row(self, i: int) -> int:
        return self.ctl.read_row(i)

    def fuse_row(self, i: int) -> int:
        return self.ctl.read_row(i)

    def register(self, a: int) -> int:
        return self.ctl.rd(a)

    def field(self, name: str) -> int:
        if name in self._fields:
            return self._fields[name]
        return 0

    def health(self):
        return self.ctl.read_health_registers()

    def host(self) -> Host:
        return Host("pci-stub", False, False, "[none]", True, self._boot)

    def active_sense_readonly(self) -> None:
        self.ctl.sense_ctrl()

    def write_one_record(self, row: int, word: int) -> None:
        self.ctl.program_row(row, word, pre_trigger_check=self._pre_trigger_check)


class SafetyReviewTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.work.chmod(0o700)
        self.journal_path = self.work / "session.jsonl"

    def _write_json(self, name: str, obj: dict) -> Path:
        p = self.work / name
        p.write_text(json.dumps(obj))
        p.chmod(0o600)
        return p

    def _logged(self) -> list[dict]:
        if not self.journal_path.exists():
            return []
        return [json.loads(line) for line in self.journal_path.read_text().splitlines() if line.strip()]

    def _base_rows(self) -> dict[int, int]:
        return {
            222: encode_ff(1, 7, 0x0123),   # FBPA block 7 (known, burnable)
            223: encode_ff(1, 19, 0x4290),  # FBP_DISABLE=0x852, ROP_L2[0]=0
            224: encode_ff(1, 23, 0x0FEF),  # NVLINK/SYS_PIPE with unmapped bit 3 = 1
            225: encode_ff(2, 39, 0x0203),  # Gen3 speed bits 1 & 9 = 1, pcie_devidb[15] (bit 0) = 1
            226: encode_ff(2, 47, 0x0008),  # pcie_magic_gen4_dis (bit 3) = 1
        }

    def _make_plan(
        self,
        records: list[dict],
        *,
        baseline_rows: dict[int, int] | None = None,
        include_baseline: bool = True,
        expected_masks: dict[str, str] | None = None,
        defect_masks: dict[str, str] | None = None,
        extra_fpf: dict | None = None,
        name: str = "plan.json",
    ) -> Plan:
        base = self._base_rows() if baseline_rows is None else baseline_rows
        fpf: dict = {
            "first_row": 222,
            "last_row": 240,
            "empty_word": "0x00000000",
        }
        if include_baseline:
            fpf["expected_next_row"] = 222 + len(base)
            fpf["baseline_ff_rows"] = {str(r): f"0x{w:08x}" for r, w in sorted(base.items())}
            fpf["baseline_ff_sha256"] = canonical_ff_prefix_sha256(base)
        if extra_fpf:
            fpf.update(extra_fpf)
        obj = {
            "schema_version": 1,
            "card": {
                "pci_id": "10de:20c2",
                "die_rows": {str(k): f"0x{v:08x}" for k, v in DIE_ROWS.items()},
                "expected_masks": expected_masks if expected_masks is not None else {"FBP_DISABLE": "0x852"},
                "defect_masks": defect_masks
                if defect_masks is not None
                else {
                    "FBP_DISABLE": "0x840",
                    "FBIO_DISABLE": "0x0",
                    "FBPA_DISABLE": "0x0",
                    "ROP_L2_DISABLE": "0x0",
                    "NVLINK_DISABLE": "0x0",
                    "SYS_PIPE_DISABLE": "0x0",
                },
            },
            "fpf": fpf,
            "ceiling": {"base_sm": 122},
            "records": records,
        }
        return Plan.load(self._write_json(name, obj))

    def _noop_rec(self) -> dict:
        w = encode_ff(1, 7, 0x0123)
        return {
            "name": "NOOP_TEST",
            "kind": "test",
            "target": "NOOP_C1_B7",
            "word": f"0x{w:08x}",
            "decode": "no-op test row",
            "defect_sensitive": False,
            "expected_cold": {"FBP_DISABLE": "0x852"},
        }

    def _live_mmio(self, rows: dict[int, int] | None = None) -> FakeGA100MMIO:
        base = self._base_rows() if rows is None else rows
        return FakeGA100MMIO(rows={**DIE_ROWS, **base}, goldilocks=True)


class TestP0_1_UnknownUnmappedAndDeviceID(SafetyReviewTestBase):
    """P0-1: Unknown blocks, changed unmapped bits, and read-only device-ID blocks."""

    def test_unknown_ff_block_refused_zero_program_calls(self):
        plan = self._make_plan([
            self._noop_rec(),
            {
                "name": "UNKNOWN_BLK",
                "kind": "regular",
                "target": "UNKNOWN_C1_B99",
                "word": f"0x{encode_ff(1, 99, 0x1234):08x}",
                "decode": "valid FF word targeting unknown block",
                "defect_sensitive": False,
            },
        ])
        mmio = self._live_mmio()
        ad = LiveMMIOAdapter(mmio)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "unknown FF block"):
                process_records(
                    plan, ad, ad, j, stage="all", confirm=lambda *_: True, show=lambda *_: None
                )
        self.assertEqual(mmio.program_calls, 0)

    def test_changed_unmapped_bit_in_known_block_refused_zero_program_calls(self):
        # Row 224 has c1/b23=0x0FEF (unmapped bit 3 is 1). Flip bit 3 -> 0 (0x0FE0).
        plan = self._make_plan([
            self._noop_rec(),
            {
                "name": "FLIP_UNMAPPED",
                "kind": "regular",
                "target": "NVLINK_DISABLE",
                "word": f"0x{encode_ff(1, 23, 0x0FE0):08x}",
                "decode": "flips unmapped bit 371 in c1/b23",
                "defect_sensitive": False,
            },
        ])
        mmio = self._live_mmio()
        ad = LiveMMIOAdapter(mmio)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "unmapped/undocumented chain bits 371"):
                process_records(
                    plan, ad, ad, j, stage="all", confirm=lambda *_: True, show=lambda *_: None
                )
        self.assertEqual(mmio.program_calls, 0)

    def test_unmapped_ff_override_requires_big_warning_and_approval(self):
        plan = self._make_plan([
            self._noop_rec(),
            {
                "name": "UNKNOWN_BLK",
                "kind": "regular",
                "target": "UNKNOWN_C1_B99",
                "word": f"0x{encode_ff(1, 99, 0x1234):08x}",
                "decode": "experimental unknown block",
                "defect_sensitive": False,
            },
        ])
        mmio = self._live_mmio()
        ad = LiveMMIOAdapter(mmio)
        shown: list[str] = []
        # Declining the OVERRIDE UNMAPPED FF token stops with zero writes.
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "declined"):
                process_records(
                    plan,
                    ad,
                    ad,
                    j,
                    stage="all",
                    overrides=Overrides(unmapped_ff=True),
                    confirm=lambda p, t: t != "OVERRIDE UNMAPPED FF",
                    show=shown.append,
                )
        self.assertEqual(mmio.program_calls, 0)
        self.assertTrue(any("MANUAL SAFETY OVERRIDE ENABLED: [unmapped_ff]" in s for s in shown))

    def test_device_id_blocks_37_38_unconditionally_denied_even_with_override_and_in_controller(self):
        for blk in (37, 38):
            with self.subTest(block=blk):
                devid_word = encode_ff(2, blk, 0x20B0)
                plan = self._make_plan([
                    self._noop_rec(),
                    {
                        "name": f"DEVID_B{blk}",
                        "kind": "regular",
                        "target": "DISGUISED_REGULAR",
                        "word": f"0x{devid_word:08x}",
                        "decode": "disguised device ID change",
                        "defect_sensitive": False,
                    },
                ])
                mmio = self._live_mmio()
                ad = LiveMMIOAdapter(mmio)
                with Journal(self.journal_path) as j:
                    with self.assertRaisesRegex(BurnError, "device-ID block"):
                        process_records(
                            plan,
                            ad,
                            ad,
                            j,
                            stage="all",
                            overrides=Overrides(unmapped_ff=True, undetermined_ceiling=True),
                            confirm=lambda *_: True,
                            show=lambda *_: None,
                        )
                self.assertEqual(mmio.program_calls, 0)

                # Defense-in-depth directly inside GA100FuseController.program_row()!
                mmio2 = self._live_mmio()
                ctl = GA100FuseController(mmio2, read_only=False, sleep=lambda _: None)
                ctl.sense_ctrl()
                with self.assertRaisesRegex(PreTriggerRefusal, "device-ID block"):
                    ctl.program_row(230, devid_word)
                self.assertEqual(mmio2.program_calls, 0)

    def test_block_39_preserves_pcie_devidb_bit15_and_refuses_changing_it(self):
        # Row 225 has c2/b39=0x0203 (bit 0 = pcie_devidb[15] = 1, bits 1 & 9 = disable_gen3_speed = 1).
        # 1) Clearing bit 0 (0x0000) modifies burnable=False pcie_devidb[15] -> UNCONDITIONALLY REFUSED!
        bad_plan = self._make_plan([
            self._noop_rec(),
            {
                "name": "CORRUPT_DEVIDB_MSB",
                "kind": "pcie",
                "target": "DISABLE_GEN3_SPEED",
                "word": f"0x{encode_ff(2, 39, 0x0000):08x}",
                "decode": "accidentally clears bit 0 (pcie_devidb[15])",
                "defect_sensitive": False,
                "link_cap2_before": "0x06",
                "link_cap2_after": "0x1e",
            },
        ])
        mmio_bad = self._live_mmio()
        ad_bad = LiveMMIOAdapter(mmio_bad)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "read-only field NV_FUSE_OPT_PCIE_DEVIDB"):
                process_records(
                    bad_plan,
                    ad_bad,
                    ad_bad,
                    j,
                    stage="all",
                    overrides=Overrides(unmapped_ff=True),
                    confirm=lambda *_: True,
                    show=lambda *_: None,
                )
        self.assertEqual(mmio_bad.program_calls, 0)

        # 2) Preserving bit 0 (0x0001) clears ONLY disable_gen3_speed bits 1 & 9 -> ALLOWED!
        good_word = encode_ff(2, 39, 0x0001)
        good_plan = self._make_plan([
            self._noop_rec(),
            {
                "name": "UNLOCK_GEN3",
                "kind": "pcie",
                "target": "DISABLE_GEN3_SPEED",
                "word": f"0x{good_word:08x}",
                "decode": "clear Gen3 speed disable bits while preserving DEVIDB[15]",
                "defect_sensitive": False,
                "link_cap2_before": "0x06",
                "link_cap2_after": "0x1e",
            },
        ], name="good_b39.json")
        mmio_good = self._live_mmio()
        ad_good = LiveMMIOAdapter(mmio_good)
        with Journal(self.work / "good_b39.jsonl") as j:
            ptr = process_records(
                good_plan, ad_good, ad_good, j, stage="all", confirm=lambda *_: True, show=lambda *_: None
            )
        self.assertEqual(ptr, 229)
        self.assertEqual(mmio_good.program_calls, 2)
        self.assertEqual(mmio_good.rows[228], good_word)


class TestP0_2_BaselineFFPrefixHistory(SafetyReviewTestBase):
    """P0-2: Live preflight proves complete existing FF row prefix matches plan baseline."""

    def test_changed_prior_ff_word_with_identical_known_masks_refused(self):
        # Card has row 224 = c1/b23=0x0FEE instead of plan baseline 0x0FEF.
        # FBP_DISABLE (the checked option mask) is identical (0x852).
        plan = self._make_plan([self._noop_rec()])
        altered = self._base_rows()
        altered[224] = encode_ff(1, 23, 0x0FEE)
        mmio = self._live_mmio(rows=altered)
        ad = LiveMMIOAdapter(mmio)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "FF PREFIX HISTORY MISMATCH: baseline FF row 224"):
                process_records(
                    plan, ad, ad, j, stage="test", confirm=lambda *_: True, show=lambda *_: None
                )
        self.assertEqual(mmio.program_calls, 0)

    def test_stale_plan_after_external_burn_refused(self):
        # Card already has an unexpected extra row at row 227 (where plan expects first free row).
        plan = self._make_plan([self._noop_rec()])
        stale = self._base_rows()
        stale[227] = encode_ff(1, 7, 0x0123)
        mmio = self._live_mmio(rows=stale)
        ad = LiveMMIOAdapter(mmio)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "live next free row is 228, plan baseline expects 227"):
                process_records(
                    plan, ad, ad, j, stage="test", confirm=lambda *_: True, show=lambda *_: None
                )
        self.assertEqual(mmio.program_calls, 0)

    def test_tampered_baseline_sha256_rejected_at_plan_load(self):
        with self.assertRaisesRegex(BurnError, "baseline_ff_sha256 does not match"):
            self._make_plan(
                [self._noop_rec()],
                extra_fpf={"baseline_ff_sha256": "0" * 64},
                name="bad_sha.json",
            )


class TestP0_3_NoopTestProof(SafetyReviewTestBase):
    """P0-3: Test record must be independently proved a 0-bit-diff no-op on an existing block."""

    def test_test_record_changing_one_bit_refused_zero_program_calls(self):
        # Baseline row 222 is c1/b7=0x0123; test record writes 0x0122 (differs by 1 bit!).
        bad_test = self._noop_rec()
        bad_test["word"] = f"0x{encode_ff(1, 7, 0x0122):08x}"
        plan = self._make_plan([bad_test])
        mmio = self._live_mmio()
        ad = LiveMMIOAdapter(mmio)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "is NOT a no-op"):
                process_records(
                    plan, ad, ad, j, stage="test", confirm=lambda *_: True, show=lambda *_: None
                )
        self.assertEqual(mmio.program_calls, 0)

    def test_test_record_for_block_absent_from_prefix_refused(self):
        # Block 6 (chain 1) is NOT in baseline rows; cannot assume underlying OTP default is 0.
        absent_test = self._noop_rec()
        absent_test["word"] = f"0x{encode_ff(1, 6, 0x0000):08x}"
        plan = self._make_plan([absent_test])
        mmio = self._live_mmio()
        ad = LiveMMIOAdapter(mmio)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "not present in the card's existing FF prefix"):
                process_records(
                    plan, ad, ad, j, stage="test", confirm=lambda *_: True, show=lambda *_: None
                )
        self.assertEqual(mmio.program_calls, 0)

    def test_importer_defaults_to_noop_test_and_refuses_promoting_real_record(self):
        rows = {str(i): f"0x{v:08x}" for i, v in DIE_ROWS.items()}
        rows[str(FF_ROW_MIN)] = f"0x{encode_ff(1, 19, 0x4290):08x}"
        for r in range(FF_ROW_MIN + 1, 487):
            rows[str(r)] = "0x00000000"
        dump_p = self._write_json("dump.json", {
            "ff_rows": rows,
            "option_words": {
                "fbp_disable": 0x852,
                "fbp_defective": 0x840,
                "rop_l2_disable": 0,
                "rop_l2_defective": 0,
            },
        })
        burn_p = self._write_json("burn.json", {
            "burn": [{
                "row": 223,
                "chain": 1,
                "block": 19,
                "old": "0x4290",
                "new": "0x4200",
                "record": f"0x{encode_ff(1, 19, 0x4200):08x}",
                "fields": ["fbp_disable"],
            }],
        })
        # Default import prepends a proved no-op test AND stores baseline_ff_rows + sha256!
        plan_dict = import_ffcalc_plan(burn_p, dump_p, base_sm_ceiling=122)
        self.assertEqual(len(plan_dict["records"]), 2)
        self.assertEqual(plan_dict["records"][0]["kind"], "test")
        self.assertIn("NO-OP", plan_dict["records"][0]["decode"])
        self.assertEqual(plan_dict["fpf"]["expected_next_row"], 223)
        self.assertIn("222", plan_dict["fpf"]["baseline_ff_rows"])
        self.assertEqual(
            plan_dict["fpf"]["baseline_ff_sha256"],
            canonical_ff_prefix_sha256({222: encode_ff(1, 19, 0x4290)}),
        )
        # Disabling prepend_noop_test when burn[0] changes bits MUST be refused!
        with self.assertRaisesRegex(BurnError, "is NOT a no-op"):
            import_ffcalc_plan(burn_p, dump_p, base_sm_ceiling=122, prepend_noop_test=False)


class TestP1_1_PoweroffSanityCheck(SafetyReviewTestBase):
    """P1-1: Functional-group pre-power-off sanity check detects split vs clean boundaries."""

    def test_partial_memory_group_warns_do_not_power_off(self):
        # Plan has TWO memory records (FBP_DISABLE + FBPA_DISABLE); stop after 1st memory record!
        fbp_word = encode_ff(1, 19, 0x4200)   # FBP 0x852 -> 0x840 (memory)
        fbpa_word = encode_ff(1, 7, 0x0103)   # FBPA change in c1/b7 (memory)
        plan = self._make_plan([
            self._noop_rec(),
            {
                "name": "MEM_FBP",
                "kind": "regular",
                "target": "FBP_DISABLE",
                "word": f"0x{fbp_word:08x}",
                "decode": "FBP 0x852 -> 0x840",
                "defect_sensitive": True,
                "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
            },
            {
                "name": "MEM_FBPA",
                "kind": "regular",
                "target": "FBPA_DISABLE",
                "word": f"0x{fbpa_word:08x}",
                "decode": "FBPA layout update",
                "defect_sensitive": False,
            },
        ])
        mmio = self._live_mmio()
        ad = LiveMMIOAdapter(mmio)
        shown: list[str] = []
        with Journal(self.journal_path) as j:
            # max_records=2 burns NOOP_TEST + MEM_FBP, leaving MEM_FBPA pending!
            process_records(
                plan,
                ad,
                ad,
                j,
                stage="all",
                max_records=2,
                confirm=lambda *_: True,
                show=shown.append,
            )
        out = "\n".join(shown)
        self.assertIn("PARTIAL 'MEMORY' FUNCTIONAL GROUP BURNED", out)
        self.assertIn("DO NOT POWER OFF YET", out)
        sanity_events = [e for e in self._logged() if e["event"] == "poweroff_sanity_check"]
        self.assertEqual(len(sanity_events), 1)
        self.assertFalse(sanity_events[0]["safe_to_power_off"])
        self.assertEqual(sanity_events[0]["split_groups"], ["memory"])

    def test_clean_boundary_after_complete_memory_group_before_pcie_is_safe(self):
        fbp_word = encode_ff(1, 19, 0x4200)   # only memory record
        pcie_word = encode_ff(2, 47, 0x0000)  # pcie record (Gen4)
        plan = self._make_plan([
            self._noop_rec(),
            {
                "name": "MEM_FBP",
                "kind": "regular",
                "target": "FBP_DISABLE",
                "word": f"0x{fbp_word:08x}",
                "decode": "FBP 0x852 -> 0x840",
                "defect_sensitive": True,
                "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
            },
            {
                "name": "PCIE_GEN4",
                "kind": "pcie",
                "target": "PCIE_MAGIC_GEN4_DIS",
                "word": f"0x{pcie_word:08x}",
                "decode": "clear Gen4 disable",
                "defect_sensitive": False,
                "link_cap2_before": "0x06",
                "link_cap2_after": "0x1e",
            },
        ])
        mmio = self._live_mmio()
        ad = LiveMMIOAdapter(mmio)
        shown: list[str] = []
        with Journal(self.journal_path) as j:
            # Stop after NOOP_TEST + MEM_FBP (complete memory group), leaving PCIE_GEN4 pending.
            process_records(
                plan,
                ad,
                ad,
                j,
                stage="all",
                max_records=2,
                confirm=lambda *_: True,
                show=shown.append,
            )
        out = "\n".join(shown)
        self.assertIn("SAFE TO POWER OFF", out)
        sanity_events = [e for e in self._logged() if e["event"] == "poweroff_sanity_check"]
        self.assertTrue(sanity_events[0]["safe_to_power_off"])
        self.assertEqual(sanity_events[0]["split_groups"], [])


class TestP1_2_WordDerivedPCIeHandling(SafetyReviewTestBase):
    """P1-2: PCIe safety gates derive from the FF word, not plan kind metadata."""

    def test_pcie_word_relabelled_regular_without_lnkcap2_is_refused(self):
        # Row 226 has c2/b47=0x0008; writing 0x0000 clears pcie_magic_gen4_dis (category="pcie").
        plan = self._make_plan([
            self._noop_rec(),
            {
                "name": "SNEAKY_PCIE",
                "kind": "regular",  # mislabelled!
                "target": "NOT_PCIE_TRUST_ME",
                "word": f"0x{encode_ff(2, 47, 0x0000):08x}",
                "decode": "mislabelled PCIe Gen4 unlock",
                "defect_sensitive": False,
            },
        ])
        mmio = self._live_mmio()
        ad = LiveMMIOAdapter(mmio)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "LnkCap2 before/after"):
                process_records(
                    plan, ad, ad, j, stage="all", confirm=lambda *_: True, show=lambda *_: None
                )
        self.assertEqual(mmio.program_calls, 0)

    def test_pcie_word_relabelled_regular_still_enforces_lnkcap2_and_review_pcie(self):
        plan = self._make_plan([
            self._noop_rec(),
            {
                "name": "SNEAKY_PCIE_WITH_CAPS",
                "kind": "regular",  # mislabelled!
                "target": "DISGUISED",
                "word": f"0x{encode_ff(2, 47, 0x0000):08x}",
                "decode": "mislabelled PCIe Gen4 unlock",
                "defect_sensitive": False,
                "link_cap2_before": "0x06",
                "link_cap2_after": "0x1e",
            },
        ])
        # 1) Wrong live LnkCap2 refuses before CMD_PROGRAM on record 2 (test row may burn, PCIe row must not).
        mmio1 = self._live_mmio()
        ad1 = LiveMMIOAdapter(mmio1, fields={"FBP_DISABLE": 0x852, "LnkCap2": 0x02})
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "current LnkCap2"):
                process_records(
                    plan, ad1, ad1, j, stage="all", confirm=lambda *_: True, show=lambda *_: None
                )
        self.assertEqual(mmio1.program_calls, 1)  # only NOOP_TEST burned; PCIe record refused!

        # 2) Matching LnkCap2 STILL demands REVIEW PCIE token!
        mmio2 = self._live_mmio()
        ad2 = LiveMMIOAdapter(mmio2, fields={"FBP_DISABLE": 0x852, "LnkCap2": 0x06})
        tokens: list[str] = []
        def capture_and_decline_pcie(prompt, token):
            tokens.append(token)
            return token != "REVIEW PCIE"
        with Journal(self.work / "j_pcie2.jsonl") as j:
            with self.assertRaisesRegex(BurnError, "special record not confirmed"):
                process_records(
                    plan, ad2, ad2, j, stage="all", confirm=capture_and_decline_pcie, show=lambda *_: None
                )
        self.assertIn("REVIEW PCIE", tokens)
        self.assertEqual(mmio2.program_calls, 1)  # PCIe record never programmed!


class TestP1_3_MissingDefectMaskNotBypassedByCeilingOverride(SafetyReviewTestBase):
    """P1-3: --override-undetermined-ceiling must NEVER bypass missing defect masks."""

    def test_missing_defect_mask_refused_even_with_override_undetermined_ceiling(self):
        # Plan omits ROP_L2_DISABLE from defect_masks; block 19 touches FBP_DISABLE + ROP_L2_DISABLE.
        plan = self._make_plan(
            [
                self._noop_rec(),
                {
                    "name": "UNLOCK_FBP",
                    "kind": "regular",
                    "target": "FBP_DISABLE",
                    "word": f"0x{encode_ff(1, 19, 0x4200):08x}",
                    "decode": "FBP unlock with missing ROP_L2 defect mask",
                    "defect_sensitive": True,
                    "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
                },
            ],
            defect_masks={"FBP_DISABLE": "0x840", "FBPA_DISABLE": "0x0"},
        )
        mmio = self._live_mmio()
        ad = LiveMMIOAdapter(mmio)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "no genuine defect mask"):
                process_records(
                    plan,
                    ad,
                    ad,
                    j,
                    stage="all",
                    overrides=Overrides(undetermined_ceiling=True),
                    confirm=lambda *_: True,
                    show=lambda *_: None,
                )
        self.assertEqual(mmio.program_calls, 0)


class TestP1_4_BDFLockAndFinalPreTriggerGate(SafetyReviewTestBase):
    """P1-4: Per-BDF exclusive lock, final pre-trigger gate, and no retry after uncertain write."""

    def test_two_sessions_same_bdf_different_journals_excluded(self):
        lock_dir = self.work / "locks"
        lock1 = BDFLock("0000:05:00.0", lock_dir=lock_dir)
        try:
            with self.assertRaisesRegex(BurnError, "already locked by another vfi_burn process"):
                BDFLock("05:00.0", lock_dir=lock_dir)
            # Different BDF succeeds concurrently.
            lock_other = BDFLock("0000:06:00.0", lock_dir=lock_dir)
            lock_other.release()
        finally:
            lock1.release()
        # Once released, re-acquiring succeeds.
        lock2 = BDFLock("0000:05:00.0", lock_dir=lock_dir)
        lock2.release()

    def test_plm_closure_or_poison_after_staging_aborts_pre_trigger_zero_program_calls(self):
        for case, reg, bad_val in [
            ("plm_f8_closed", REG_PLM_F8, 0xFFFFFFFE),
            ("fusectrl_poison", REG_FUSECTRL, 0xBADF510E),
        ]:
            with self.subTest(case=case):
                mmio = self._live_mmio()
                orig_write32 = mmio.write32

                def sabotage_after_wdata(offset, value, **kw):
                    orig_write32(offset, value, **kw)
                    # Simulate fragile window collapsing right after FUSEWDATA is staged!
                    if offset == REG_FUSEWDATA and value != 0:
                        mmio.regs[reg] = bad_val

                mmio.write32 = sabotage_after_wdata
                ctl = GA100FuseController(mmio, read_only=False, sleep=lambda _: None)
                ctl.sense_ctrl()
                with self.assertRaisesRegex(PreTriggerRefusal, "final pre-trigger controller gate failed"):
                    ctl.program_row(227, encode_ff(1, 7, 0x0123))
                self.assertEqual(mmio.program_calls, 0)
                if reg != REG_FUSECTRL:
                    self.assertEqual(mmio.regs[REG_FUSEWDATA], 0)

    def test_pre_trigger_rail_callback_failure_aborts_with_zero_program_calls(self):
        mmio = self._live_mmio()
        ctl = GA100FuseController(mmio, read_only=False, sleep=lambda _: None)
        ctl.sense_ctrl()
        def bad_rail():
            raise BurnError("live rail collapse before CMD_PROGRAM")
        with self.assertRaisesRegex(PreTriggerRefusal, "live rail collapse"):
            ctl.program_row(227, encode_ff(1, 7, 0x0123), pre_trigger_check=bad_rail)
        self.assertEqual(mmio.program_calls, 0)
        self.assertEqual(mmio.regs[REG_FUSEWDATA], 0)

    def test_uncertain_post_trigger_failure_invokes_program_once_no_retry_no_second_record(self):
        plan = self._make_plan([
            self._noop_rec(),
            {
                "name": "UNLOCK_FBP",
                "kind": "regular",
                "target": "FBP_DISABLE",
                "word": f"0x{encode_ff(1, 19, 0x4200):08x}",
                "decode": "FBP unlock",
                "defect_sensitive": True,
                "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
            },
        ])
        mmio = self._live_mmio()
        orig_write32 = mmio.write32

        def partial_program(offset, value, **kw):
            orig_write32(offset, value, **kw)
            # Corrupt programmed word in OTP to simulate a partial/uncertain burn!
            if offset == REG_FUSECTRL and (value & 0x1F) == 2:
                mmio.rows[227] ^= 0x1

        mmio.write32 = partial_program
        ad = LiveMMIOAdapter(mmio)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "HALT: row 227 may be partially programmed; do NOT retry"):
                process_records(
                    plan, ad, ad, j, stage="all", confirm=lambda *_: True, show=lambda *_: None
                )
        # CMD_PROGRAM invoked EXACTLY ONCE; never retried and 2nd record never touched!
        self.assertEqual(mmio.program_calls, 1)
        self.assertEqual(mmio.rows[228], 0)
        events = [e["event"] for e in self._logged()]
        self.assertIn("HALT_UNVERIFIED", events)


class TestP2_WriteCapabilityAndColdSnapshotHardening(SafetyReviewTestBase):
    """P2: Honest write_enabled capability check and cold-snapshot controller/row checks."""

    def test_has_write_capability_accepts_write_enabled_without_writer_verified(self):
        class HonestAdapter:
            simulation = False
            write_enabled = True
            def write_one_record(self, row, word):
                pass

        self.assertTrue(has_write_capability(HonestAdapter()))

    def test_cold_verify_rejects_held_glitch_window_and_altered_baseline_rows(self):
        plan = self._make_plan([self._noop_rec()])
        mmio = self._live_mmio()
        ad = LiveMMIOAdapter(mmio, boot_id="boot-1")
        with Journal(self.journal_path) as j:
            process_records(plan, ad, ad, j, stage="test", confirm=lambda *_: True, show=lambda *_: None)

        base_fpf = {str(r): f"0x{w:08x}" for r, w in self._base_rows().items()}
        base_fpf["227"] = f"0x{plan.records[0].word:08x}"

        # 1) Snapshot with DSOV=0 (still in held glitch window!) -> rejected!
        held_cold = {
            "pci_id": "10de:20c2",
            "cold_boot": True,
            "boot_id": "boot-2",
            "previous_boot_id": "boot-1",
            "die_rows": {str(k): f"0x{v:08x}" for k, v in DIE_ROWS.items()},
            "fpf_rows": dict(base_fpf),
            "fields": {"FBP_DISABLE": "0x852"},
            "controller_state": {"fusectrl": "0xe0040000", "dsov": "0x00000000", "en_sw": "0x00000001"},
        }
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "DSOV=0"):
                verify_cold_snapshot(plan, held_cold, stage="test", journal=j)

        # 2) Snapshot with unexpected extra FF row 228 -> rejected!
        extra_fpf = dict(base_fpf)
        extra_fpf["228"] = f"0x{encode_ff(1, 7, 0x0123):08x}"
        extra_cold = {
            **held_cold,
            "fpf_rows": extra_fpf,
            "controller_state": {"fusectrl": "0xe0040000", "dsov": "0x00000001", "en_sw": "0x00000000"},
        }
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "unexpected extra FF row"):
                verify_cold_snapshot(plan, extra_cold, stage="test", journal=j)

        # 3) Clean cold snapshot with DSOV=1 and exact baseline+test rows -> PASS!
        good_cold = {
            **held_cold,
            "controller_state": {"fusectrl": "0xe0040000", "dsov": "0x00000001", "en_sw": "0x00000000"},
        }
        with Journal(self.journal_path) as j:
            receipt = verify_cold_snapshot(plan, good_cold, stage="test", journal=j)
        self.assertEqual(receipt["status"], "PASS")


class TestArchReviewSafetyFixes(SafetyReviewTestBase):
    """Regression tests for post-architecture code-review safety findings."""

    def _sm_hbm_fixture(self) -> tuple[dict[int, int], dict[str, str], dict[str, str], dict[str, int]]:
        base_rows = {
            222: encode_ff(1, 7, 0x0123),   # FBPA (no-op target)
            223: encode_ff(1, 12, 0x0008),  # TPC_GPC1_DISABLE bit 0 = 1 (compute)
            224: encode_ff(1, 20, 0x0001),  # ROP_L2_DISABLE bit 1 = 1 (memory)
            225: encode_ff(2, 47, 0x0008),  # pcie_magic_gen4_dis = 1 (pcie)
        }
        expected_masks = {
            "GPC_DISABLE": "0x0",
            **{f"TPC_GPC{i}_DISABLE": ("0x1" if i == 1 else "0x0") for i in range(8)},
            "ROP_L2_DISABLE": "0x2",
        }
        defect_masks = {
            "GPC_DISABLE": "0x0",
            **{f"TPC_GPC{i}_DISABLE": "0x0" for i in range(8)},
            "FBPA_DISABLE": "0x0",
            "ROP_L2_DISABLE": "0x0",
        }
        live_fields = {
            "GPC_DISABLE": 0x0,
            **{f"TPC_GPC{i}_DISABLE": (0x1 if i == 1 else 0x0) for i in range(8)},
            "ROP_L2_DISABLE": 0x2,
            "LnkCap2": 0x06,
        }
        return base_rows, expected_masks, defect_masks, live_fields

    def test_tpc_committed_with_rop_l2_pending_is_not_safe_to_power_off(self):
        # Finding 1: compute (TPC) committed while memory (ROP_L2) is pending is a GSP-boot-failure
        # state and must NOT print "clean point to stop" or report SAFE TO POWER OFF.
        base_rows, expected_masks, defect_masks, live_fields = self._sm_hbm_fixture()
        tpc_unlock = encode_ff(1, 12, 0x0000)   # compute: TPC_GPC1_DISABLE 0x1 -> 0x0 (128 SMs)
        rop_unlock = encode_ff(1, 20, 0x0000)   # memory: ROP_L2_DISABLE 0x2 -> 0x0
        pcie_unlock = encode_ff(2, 47, 0x0000)  # pcie: Gen4 unlock
        plan = self._make_plan(
            [
                self._noop_rec(),
                {
                    "name": "UNLOCK_TPC1",
                    "kind": "regular",
                    "target": "TPC_GPC1_DISABLE",
                    "word": f"0x{tpc_unlock:08x}",
                    "decode": "TPC_GPC1 0x1 -> 0x0",
                    "defect_sensitive": True,
                    "mask_update": {"field": "TPC_GPC1_DISABLE", "from": "0x1", "to": "0x0"},
                    "resulting_sm_count": 128,
                },
                {
                    "name": "UNLOCK_ROP_L2",
                    "kind": "regular",
                    "target": "ROP_L2_DISABLE",
                    "word": f"0x{rop_unlock:08x}",
                    "decode": "ROP_L2 0x2 -> 0x0",
                    "defect_sensitive": True,
                    "mask_update": {"field": "ROP_L2_DISABLE", "from": "0x2", "to": "0x0"},
                },
                {
                    "name": "UNLOCK_PCIE_GEN4",
                    "kind": "pcie",
                    "target": "PCIE_MAGIC_GEN4_DIS",
                    "word": f"0x{pcie_unlock:08x}",
                    "decode": "clear Gen4 disable",
                    "defect_sensitive": False,
                    "link_cap2_before": "0x06",
                    "link_cap2_after": "0x1e",
                },
            ],
            baseline_rows=base_rows,
            expected_masks=expected_masks,
            defect_masks=defect_masks,
            extra_fpf={"last_row": 240},
        )
        # Allow 128 SMs ceiling for this synthetic plan.
        plan = replace(plan, base_sm_ceiling=128)

        # 1) Stop after NOOP + TPC (ROP_L2 still pending): DO NOT POWER OFF YET + split sm_hbm!
        mmio1 = self._live_mmio(base_rows)
        ad1 = LiveMMIOAdapter(mmio1, fields=live_fields)
        shown1: list[str] = []
        with Journal(self.journal_path) as j:
            process_records(
                plan,
                ad1,
                ad1,
                j,
                stage="all",
                max_records=2,
                confirm=lambda *_: True,
                show=shown1.append,
            )
        out1 = "\n".join(shown1)
        self.assertIn("DO NOT POWER OFF YET", out1)
        self.assertNotIn("SAFE TO POWER OFF", out1)
        self.assertIn("SM_HBM", out1)
        self.assertIn("GSP boot failure", out1)
        sanity1 = [e for e in self._logged() if e["event"] == "poweroff_sanity_check"]
        self.assertEqual(len(sanity1), 1)
        self.assertFalse(sanity1[0]["safe_to_power_off"])
        self.assertEqual(sanity1[0]["split_groups"], ["sm_hbm"])

        # 2) Burn through NOOP + TPC + ROP_L2 (stopping before PCIE):
        #    - NO "clean point to stop" banner between TPC (compute) and ROP_L2 (memory).
        #    - Pre-power-off sanity check before PCIE IS safe to power off!
        mmio2 = self._live_mmio(base_rows)
        ad2 = LiveMMIOAdapter(mmio2, fields=live_fields)
        shown2: list[str] = []
        j2_path = self.work / "sm_hbm_full.jsonl"
        with Journal(j2_path) as j2:
            process_records(
                plan,
                ad2,
                ad2,
                j2,
                stage="all",
                max_records=3,
                confirm=lambda *_: True,
                show=shown2.append,
            )
        out2 = "\n".join(shown2)
        self.assertNotIn("clean point to stop & restart", out2)
        self.assertIn("SAFE TO POWER OFF", out2)

        # 3) Burn all 4 records: "clean point to stop & restart" appears ONLY before PCIE,
        #    never between TPC (compute) and ROP_L2 (memory).
        mmio3 = self._live_mmio(base_rows)
        ad3 = LiveMMIOAdapter(mmio3, fields=live_fields)
        shown3: list[str] = []
        with Journal(self.work / "sm_hbm_all4.jsonl") as j3:
            process_records(
                plan,
                ad3,
                ad3,
                j3,
                stage="all",
                confirm=lambda *_: True,
                show=shown3.append,
            )
        boundary_lines = [line for line in shown3 if "clean point to stop & restart" in line]
        self.assertEqual(len(boundary_lines), 1)
        self.assertIn("next is [pcie]", boundary_lines[0])

    def test_interleaved_compute_pcie_memory_does_not_claim_clean_boundary_before_pcie(self):
        # Even if PCIE sits between TPC (compute) and ROP_L2 (memory), sm_hbm is still split
        # before PCIE, so the mid-loop banner must NOT claim a clean point to stop.
        base_rows, expected_masks, defect_masks, live_fields = self._sm_hbm_fixture()
        plan = self._make_plan(
            [
                self._noop_rec(),
                {
                    "name": "UNLOCK_TPC1",
                    "kind": "regular",
                    "target": "TPC_GPC1_DISABLE",
                    "word": f"0x{encode_ff(1, 12, 0x0000):08x}",
                    "decode": "TPC_GPC1 0x1 -> 0x0",
                    "defect_sensitive": True,
                    "mask_update": {"field": "TPC_GPC1_DISABLE", "from": "0x1", "to": "0x0"},
                    "resulting_sm_count": 128,
                },
                {
                    "name": "UNLOCK_PCIE_GEN4",
                    "kind": "pcie",
                    "target": "PCIE_MAGIC_GEN4_DIS",
                    "word": f"0x{encode_ff(2, 47, 0x0000):08x}",
                    "decode": "clear Gen4 disable",
                    "defect_sensitive": False,
                    "link_cap2_before": "0x06",
                    "link_cap2_after": "0x1e",
                },
                {
                    "name": "UNLOCK_ROP_L2",
                    "kind": "regular",
                    "target": "ROP_L2_DISABLE",
                    "word": f"0x{encode_ff(1, 20, 0x0000):08x}",
                    "decode": "ROP_L2 0x2 -> 0x0",
                    "defect_sensitive": True,
                    "mask_update": {"field": "ROP_L2_DISABLE", "from": "0x2", "to": "0x0"},
                },
            ],
            baseline_rows=base_rows,
            expected_masks=expected_masks,
            defect_masks=defect_masks,
            name="interleaved.json",
        )
        plan = replace(plan, base_sm_ceiling=128)
        mmio = self._live_mmio(base_rows)
        ad = LiveMMIOAdapter(mmio, fields=live_fields)
        shown: list[str] = []
        with Journal(self.journal_path) as j:
            process_records(plan, ad, ad, j, stage="all", confirm=lambda *_: True, show=shown.append)
        self.assertFalse(any("clean point to stop & restart" in line for line in shown))

    def test_missing_block_39_keeps_pcie_devidb_hard_deny_even_with_override(self):
        # Finding 2: When chain 2 block 39 is absent from the FF prefix, reconcile() and
        # process_records() must pass old_block=None (not 0), keeping the unconditional hard deny
        # on pcie_devidb[15] (bit 0) even if --override-unmapped-ff is supplied.
        rows_without_b39 = {
            222: encode_ff(1, 7, 0x0123),
            223: encode_ff(1, 19, 0x4290),
        }
        gen3_word = encode_ff(2, 39, 0x0580)  # standard Gen3 unlock with bit 0 = 0
        plan = self._make_plan(
            [
                self._noop_rec(),
                {
                    "name": "GEN3_MISSING_B39",
                    "kind": "pcie",
                    "target": "DISABLE_GEN3_SPEED",
                    "word": f"0x{gen3_word:08x}",
                    "decode": "Gen3 unlock 0x0580 when c2/b39 absent from FF prefix",
                    "defect_sensitive": False,
                    "link_cap2_before": "0x06",
                    "link_cap2_after": "0x0e",
                },
            ],
            baseline_rows=rows_without_b39,
            name="missing_b39.json",
        )

        # 1) Direct reconcile() refuses even with allow_unmapped_ff=True.
        with self.assertRaisesRegex(BurnError, "NV_FUSE_OPT_PCIE_DEVIDB.*no verified baseline block"):
            reconcile(
                plan,
                plan.records[1],
                plan.expected_masks.copy(),
                allow_unmapped_ff=True,
                baseline_blocks=replay_ff_rows(rows_without_b39),
                enforce_known_ff=True,
            )

        # 2) process_records() refuses with Overrides(unmapped_ff=True) and ZERO program calls.
        mmio = self._live_mmio(rows_without_b39)
        ad = LiveMMIOAdapter(mmio)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "NV_FUSE_OPT_PCIE_DEVIDB.*no verified baseline block"):
                process_records(
                    plan,
                    ad,
                    ad,
                    j,
                    stage="all",
                    overrides=Overrides(unmapped_ff=True),
                    confirm=lambda *_: True,
                    show=lambda *_: None,
                )
        self.assertEqual(mmio.program_calls, 0)

        # 3) import_ffcalc_plan() also refuses chain 2 block 39 when absent from dump FF prefix.
        ff_rows = {str(i): f"0x{v:08x}" for i, v in DIE_ROWS.items()}
        ff_rows["222"] = f"0x{encode_ff(1, 7, 0x0123):08x}"
        for r in range(223, 487):
            ff_rows[str(r)] = "0x00000000"
        dp = self._write_json(
            "dump_no_b39.json",
            {"sku": "8gb", "ff_rows": ff_rows, "option_words": {}},
        )
        bp = self._write_json(
            "burn_no_b39.json",
            {
                "sku": "8gb",
                "burn": [
                    {
                        "row": 223,
                        "chain": 2,
                        "block": 39,
                        "old": "0x0000",
                        "new": "0x0580",
                        "record": f"0x{gen3_word:08x}",
                        "fields": ["disable_gen3_speed_bit1", "disable_gen3_speed_bit9"],
                    }
                ],
            },
        )
        with self.assertRaisesRegex(BurnError, "NV_FUSE_OPT_PCIE_DEVIDB.*no verified baseline block"):
            import_ffcalc_plan(bp, dp, base_sm_ceiling=122)

    def test_chain2_block46_magic_d_is_treated_as_pcie_and_enforces_lnkcap2_and_review(self):
        # Finding 3: ga100-ff-tools Gen4 record on chain 2 block 46 (0xA000 -> 0x8000, magic_d)
        # must be treated as PCIe via functional_group == "pcie", enforcing LnkCap2 and REVIEW PCIE
        # while still requiring --override-unmapped-ff because magic_d is opt_in.
        b46_old = encode_ff(2, 46, 0xA000)
        b46_new = encode_ff(2, 46, 0x8000)
        self.assertTrue(ff_word_touches_pcie(b46_new, 0xA000))
        self.assertTrue(ff_word_touches_pcie(b46_new, None))

        base_rows = {
            **self._base_rows(),
            227: b46_old,
        }

        # 1) Without LnkCap2 before/after, even with kind="regular" and Overrides(unmapped_ff=True),
        #    reconcile / process_records refuses before any write.
        no_lnk_plan = self._make_plan(
            [
                self._noop_rec(),
                {
                    "name": "GEN4_B46_NO_LNK",
                    "kind": "regular",
                    "target": "MAGIC_D",
                    "word": f"0x{b46_new:08x}",
                    "decode": "chain 2 block 46 0xA000 -> 0x8000",
                    "defect_sensitive": False,
                },
            ],
            baseline_rows=base_rows,
            name="b46_no_lnk.json",
        )
        mmio0 = self._live_mmio(base_rows)
        ad0 = LiveMMIOAdapter(mmio0)
        with Journal(self.work / "j_b46_0.jsonl") as j0:
            with self.assertRaisesRegex(BurnError, "LnkCap2 before/after"):
                process_records(
                    no_lnk_plan,
                    ad0,
                    ad0,
                    j0,
                    stage="all",
                    overrides=Overrides(unmapped_ff=True),
                    confirm=lambda *_: True,
                    show=lambda *_: None,
                )
        self.assertEqual(mmio0.program_calls, 0)

        # 2) With LnkCap2 declared, WITHOUT --override-unmapped-ff it is still refused (magic_d opt_in).
        with_lnk_plan = self._make_plan(
            [
                self._noop_rec(),
                {
                    "name": "GEN4_B46",
                    "kind": "regular",  # intentionally mislabelled regular; word must still force PCIe gates
                    "target": "MAGIC_D",
                    "word": f"0x{b46_new:08x}",
                    "decode": "chain 2 block 46 0xA000 -> 0x8000",
                    "defect_sensitive": False,
                    "link_cap2_before": "0x06",
                    "link_cap2_after": "0x1e",
                },
            ],
            baseline_rows=base_rows,
            name="b46_with_lnk.json",
        )
        mmio_no_ov = self._live_mmio(base_rows)
        ad_no_ov = LiveMMIOAdapter(mmio_no_ov)
        with Journal(self.work / "j_b46_no_ov.jsonl") as j_no_ov:
            with self.assertRaisesRegex(BurnError, "experimental opt-in field"):
                process_records(
                    with_lnk_plan,
                    ad_no_ov,
                    ad_no_ov,
                    j_no_ov,
                    stage="all",
                    confirm=lambda *_: True,
                    show=lambda *_: None,
                )
        self.assertEqual(mmio_no_ov.program_calls, 0)

        # 3) With Overrides(unmapped_ff=True), wrong live LnkCap2 refuses before burning block 46.
        mmio1 = self._live_mmio(base_rows)
        ad1 = LiveMMIOAdapter(mmio1, fields={"FBP_DISABLE": 0x852, "LnkCap2": 0x02})
        with Journal(self.work / "j_b46_1.jsonl") as j1:
            with self.assertRaisesRegex(BurnError, "current LnkCap2"):
                process_records(
                    with_lnk_plan,
                    ad1,
                    ad1,
                    j1,
                    stage="all",
                    overrides=Overrides(unmapped_ff=True),
                    confirm=lambda *_: True,
                    show=lambda *_: None,
                )
        self.assertEqual(mmio1.program_calls, 1)  # NOOP_TEST only; block 46 refused

        # 4) Matching live LnkCap2 demands REVIEW PCIE confirmation token!
        mmio2 = self._live_mmio(base_rows)
        ad2 = LiveMMIOAdapter(mmio2, fields={"FBP_DISABLE": 0x852, "LnkCap2": 0x06})
        tokens: list[str] = []

        def decline_review_pcie(prompt, token):
            tokens.append(token)
            return token != "REVIEW PCIE"

        with Journal(self.work / "j_b46_2.jsonl") as j2:
            with self.assertRaisesRegex(BurnError, "special record not confirmed"):
                process_records(
                    with_lnk_plan,
                    ad2,
                    ad2,
                    j2,
                    stage="all",
                    overrides=Overrides(unmapped_ff=True),
                    confirm=decline_review_pcie,
                    show=lambda *_: None,
                )
        self.assertIn("REVIEW PCIE", tokens)
        self.assertEqual(mmio2.program_calls, 1)

        # 5) import_ffcalc_plan classifies block 46 magic_d as kind="pcie" with LnkCap2 fields.
        ff_rows = {str(i): f"0x{v:08x}" for i, v in DIE_ROWS.items()}
        ff_rows["222"] = f"0x{encode_ff(1, 7, 0x0123):08x}"
        ff_rows["223"] = f"0x{b46_old:08x}"
        for r in range(224, 487):
            ff_rows[str(r)] = "0x00000000"
        dp = self._write_json("dump_b46.json", {"sku": "8gb", "ff_rows": ff_rows, "option_words": {}})
        bp = self._write_json(
            "burn_b46.json",
            {
                "sku": "8gb",
                "burn": [
                    {
                        "row": 224,
                        "chain": 2,
                        "block": 46,
                        "old": "0xa000",
                        "new": "0x8000",
                        "record": f"0x{b46_new:08x}",
                        "fields": ["magic_d"],
                    }
                ],
            },
        )
        imported = import_ffcalc_plan(bp, dp, base_sm_ceiling=122)
        self.assertEqual(imported["records"][1]["kind"], "pcie")
        self.assertIn("link_cap2_before", imported["records"][1])
        self.assertIn("link_cap2_after", imported["records"][1])
        self.assertEqual(imported["final_expected_cold"]["LnkCap2"], "0x1e")


if __name__ == "__main__":
    unittest.main()
