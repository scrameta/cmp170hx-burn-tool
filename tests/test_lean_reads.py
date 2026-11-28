"""Tests for lean FPF read behavior: single-pass initial scan, no inter-record re-scans,
journal/verified-row power-off sanity check, and single-pass console inspection."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from vfi_burn.console import ManualConsole, format_plan_review
from vfi_burn.engine import Journal, Overrides, process_records, resolve_fpf_pointer, scan_fpf_rows
from vfi_burn.ffrecord import FF_ROW_MAX, FF_ROW_MIN, encode_ff
from vfi_burn.fuses import (
    DIE_ID_ROWS,
    REG_DSOV_STATUS,
    REG_FUSECTRL,
)
from vfi_burn.ga100 import CMD_PROGRAM, CMD_READ
from vfi_burn.model import BurnError, Plan, canonical_ff_prefix_sha256
from vfi_burn.sim_pcie import SimulatedGA100Adapter


FF_REGION_LEN = FF_ROW_MAX - FF_ROW_MIN + 1  # 265 rows (222..486)


def _count_cmd(adapter: SimulatedGA100Adapter, cmd: int) -> int:
    return sum(
        1
        for tx in adapter.bus.writes()
        if tx.offset == REG_FUSECTRL and tx.value == cmd
    )


class TestLeanReads(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)
        self.work.chmod(0o700)
        self.journal_path = self.work / "journal.jsonl"
        self.die = {i: 0x1000 + i for i in DIE_ID_ROWS}
        # Baseline FF prefix covering chain 1 blocks 7 (FBPA), 19 (FBP + ROP_L2[0]),
        # 20 (ROP_L2), and 23 (NVLink/SYS_PIPE) so multi-record plans have clean targets.
        self.base_rows = {
            222: encode_ff(1, 7, 0x0123),
            223: encode_ff(1, 19, 0x4290),   # FBP_DISABLE = 0x852, ROP_L2 bit 0 = 0
            224: encode_ff(1, 20, 0x0000),   # ROP_L2_DISABLE upper bits
            225: encode_ff(1, 23, 0x0007),   # NVLINK_DISABLE = 0x7
        }
        self.expected_masks = {
            "FBP_DISABLE": 0x852,
            "FBPA_DISABLE": 0x2460,
            "ROP_L2_DISABLE": 0x0,
            "NVLINK_DISABLE": 0x7,
            "SYS_PIPE_DISABLE": 0x0,
        }
        self.defect_masks = {
            "FBP_DISABLE": 0x840,
            "FBPA_DISABLE": 0x0,
            "ROP_L2_DISABLE": 0x0,
            "NVLINK_DISABLE": 0x0,
            "SYS_PIPE_DISABLE": 0x0,
        }

    def tearDown(self):
        self._tmp.cleanup()

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

    def _fbp_rec(self, name: str = "UNLOCK_FBP") -> dict:
        w = encode_ff(1, 19, 0x4200)  # FBP 0x852 -> 0x840 (keeps defect 0x840)
        return {
            "name": name,
            "kind": "regular",
            "target": "FBP_DISABLE",
            "word": f"0x{w:08x}",
            "decode": "FBP 0x852 -> 0x840",
            "defect_sensitive": True,
            "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
            "expected_cold": {"FBP_DISABLE": "0x840"},
        }

    def _make_plan(
        self,
        records: list[dict],
        *,
        base_rows: dict[int, int] | None = None,
        last_row: int = FF_ROW_MAX,
        expected_next_row: int | None = None,
    ) -> Plan:
        brows = self.base_rows if base_rows is None else base_rows
        next_row = (
            expected_next_row
            if expected_next_row is not None
            else (FF_ROW_MIN + len(brows))
        )
        obj = {
            "schema_version": 1,
            "card": {
                "pci_id": "10de:20c2",
                "die_rows": {str(k): f"0x{v:08x}" for k, v in self.die.items()},
                "expected_masks": {k: f"0x{v:x}" for k, v in self.expected_masks.items()},
                "defect_masks": {k: f"0x{v:x}" for k, v in self.defect_masks.items()},
            },
            "fpf": {
                "first_row": FF_ROW_MIN,
                "last_row": last_row,
                "empty_word": "0x00000000",
                "expected_next_row": next_row,
                "baseline_ff_rows": {str(r): f"0x{w:08x}" for r, w in brows.items()},
                "baseline_ff_sha256": canonical_ff_prefix_sha256(brows),
            },
            "ceiling": {"base_sm": 122},
            "records": records,
        }
        p = self.work / f"plan_{len(records)}_{last_row}.json"
        p.write_text(json.dumps(obj), encoding="utf-8")
        p.chmod(0o600)
        return Plan.load(p)

    def _make_adapter(self, rows: dict[int, int] | None = None) -> SimulatedGA100Adapter:
        return SimulatedGA100Adapter(
            die=self.die,
            rows=dict(self.base_rows if rows is None else rows),
            fields=dict(self.expected_masks),
            lnkcap2=0x06,
        )

    def _logged(self) -> list[dict]:
        if not self.journal_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # ------------------------------------------------------------------
    # Exact CMD_READ budgets across full 265-row FF region
    # ------------------------------------------------------------------

    def test_2_record_session_exact_cmd_read_count(self):
        plan = self._make_plan([self._noop_rec(), self._fbp_rec()])
        ad = self._make_adapter()
        with Journal(self.journal_path) as j:
            ptr = process_records(
                plan, ad, ad, j,
                stage="all",
                confirm=lambda *_: True,
                show=lambda *_: None,
            )
        self.assertEqual(ptr, 228)
        self.assertEqual(_count_cmd(ad, CMD_PROGRAM), 2)
        # 16 (initial + post-sense die rows) + 265 (1 full pass) + 2 * 6 (target reads) + 0 (poweroff sanity)
        self.assertEqual(_count_cmd(ad, CMD_READ), 16 + FF_REGION_LEN + 2 * 6)
        self.assertEqual(_count_cmd(ad, CMD_READ), 293)

        sanity = [e for e in self._logged() if e["event"] == "poweroff_sanity_check"]
        self.assertEqual(len(sanity), 1)
        self.assertTrue(sanity[0]["safe_to_power_off"])

    def test_17_record_session_exact_cmd_read_count(self):
        # 1 no-op test + 16 valid non-defective records (17 records total).
        records: list[dict] = [self._noop_rec(), self._fbp_rec()]
        for i in range(15):
            # Alternate valid NVLINK disable masks (all bits non-defective, chain 1 block 23).
            val = 0x0007 if (i % 2 == 0) else 0x0000
            w = encode_ff(1, 23, val)
            records.append(
                {
                    "name": f"REC_{i + 2}",
                    "kind": "regular",
                    "target": "NVLINK_DISABLE",
                    "word": f"0x{w:08x}",
                    "decode": f"nvlink step {i}",
                    "defect_sensitive": False,
                }
            )
        self.assertEqual(len(records), 17)
        plan = self._make_plan(records)
        ad = self._make_adapter()
        with Journal(self.journal_path) as j:
            ptr = process_records(
                plan, ad, ad, j,
                stage="all",
                confirm=lambda *_: True,
                show=lambda *_: None,
            )
        self.assertEqual(ptr, 226 + 17)
        self.assertEqual(_count_cmd(ad, CMD_PROGRAM), 17)
        # 16 (die rows) + 265 (single initial scan) + 17 * 6 (target checks/readbacks) = 383
        self.assertEqual(_count_cmd(ad, CMD_READ), 16 + FF_REGION_LEN + 17 * 6)
        self.assertEqual(_count_cmd(ad, CMD_READ), 383)

        verified = [e for e in self._logged() if e["event"] == "verified_readback_only"]
        self.assertEqual(len(verified), 17)
        sanity = [e for e in self._logged() if e["event"] == "poweroff_sanity_check"]
        self.assertEqual(len(sanity), 1)
        self.assertTrue(sanity[0]["safe_to_power_off"])

    # ------------------------------------------------------------------
    # Safety invariants preserved by single-pass initial scan
    # ------------------------------------------------------------------

    def test_single_pass_initial_scan_still_catches_stray_trailing_row(self):
        plan = self._make_plan([self._noop_rec(), self._fbp_rec()])
        rows = dict(self.base_rows)
        rows[300] = encode_ff(1, 19, 0x4290)  # stray record past pointer
        ad = self._make_adapter(rows=rows)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "non-contiguous"):
                process_records(
                    plan, ad, ad, j,
                    stage="all",
                    confirm=lambda *_: True,
                    show=lambda *_: None,
                )
        self.assertEqual(_count_cmd(ad, CMD_PROGRAM), 0)
        # 16 die reads + 265 single-pass FPF scan (no second pass).
        self.assertEqual(_count_cmd(ad, CMD_READ), 16 + FF_REGION_LEN)

    def test_single_pass_initial_scan_still_catches_baseline_mismatch(self):
        plan = self._make_plan([self._noop_rec(), self._fbp_rec()])
        rows = dict(self.base_rows)
        rows[222] = encode_ff(1, 7, 0x0124)  # altered prefix row
        ad = self._make_adapter(rows=rows)
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "FF PREFIX HISTORY MISMATCH"):
                process_records(
                    plan, ad, ad, j,
                    stage="all",
                    confirm=lambda *_: True,
                    show=lambda *_: None,
                )
        self.assertEqual(_count_cmd(ad, CMD_PROGRAM), 0)

    def test_resolve_fpf_pointer_rejects_missing_row(self):
        plan = self._make_plan([self._noop_rec()])
        partial = {r: 0 for r in range(FF_ROW_MIN, FF_ROW_MAX)}  # missing FF_ROW_MAX
        with self.assertRaisesRegex(BurnError, f"row {FF_ROW_MAX} missing"):
            resolve_fpf_pointer(plan, partial)

    # ------------------------------------------------------------------
    # Inter-record safety checks without full FPF re-scan
    # ------------------------------------------------------------------

    def test_inter_record_halts_on_health_loss(self):
        plan = self._make_plan([self._noop_rec(), self._fbp_rec()])
        ad = self._make_adapter()

        def confirm_and_collapse(prompt: str, token: str) -> bool:
            if token == "CONTINUE SAME WINDOW":
                ad.pcie.regs[REG_DSOV_STATUS] = 1
            return True

        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "goldilocks lost|no live held-frame"):
                process_records(
                    plan, ad, ad, j,
                    stage="all",
                    confirm=confirm_and_collapse,
                    show=lambda *_: None,
                )
        self.assertEqual(_count_cmd(ad, CMD_PROGRAM), 1)
        preflight_fails = [
            e for e in self._logged()
            if e["event"] == "preflight_failed" and e["phase"] == "before_record:UNLOCK_FBP"
        ]
        self.assertEqual(len(preflight_fails), 1)

    def test_inter_record_halts_if_next_target_row_becomes_nonempty(self):
        plan = self._make_plan([self._noop_rec(), self._fbp_rec()])
        ad = self._make_adapter()

        def confirm_and_dirty_next_row(prompt: str, token: str) -> bool:
            if token == "CONTINUE SAME WINDOW":
                ad.pcie.rows[227] = encode_ff(1, 7, 0x9999)
            return True

        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "row 227 not empty immediately before write"):
                process_records(
                    plan, ad, ad, j,
                    stage="all",
                    confirm=confirm_and_dirty_next_row,
                    show=lambda *_: None,
                )
        self.assertEqual(_count_cmd(ad, CMD_PROGRAM), 1)

    def test_refuses_before_burning_when_pending_records_exceed_fpf_region(self):
        # Plan has 2 records, but operator overrides slot to 227 (the last row):
        # up-front capacity check refuses BEFORE burning record 0 (0 CMD_PROGRAM calls).
        plan = self._make_plan([self._noop_rec(), self._fbp_rec()], last_row=227)
        ad = self._make_adapter()
        with Journal(self.journal_path) as j:
            with self.assertRaisesRegex(BurnError, "do not fit in the remaining FF region"):
                process_records(
                    plan, ad, ad, j,
                    stage="all",
                    overrides=Overrides(slot=227, baseline_mismatch=True),
                    confirm=lambda *_: True,
                    show=lambda *_: None,
                )
        self.assertEqual(_count_cmd(ad, CMD_PROGRAM), 0)

    # ------------------------------------------------------------------
    # Power-off sanity check computed from baseline + verified writes (0 reads)
    # ------------------------------------------------------------------

    def test_poweroff_sanity_zero_reads_and_succeeds_even_if_reader_dies_at_exit(self):
        fbpa_word = encode_ff(1, 7, 0x0103)  # second memory record left pending
        plan = self._make_plan([
            self._noop_rec(),
            self._fbp_rec(),
            {
                "name": "MEM_FBPA",
                "kind": "regular",
                "target": "FBPA_DISABLE",
                "word": f"0x{fbpa_word:08x}",
                "decode": "FBPA update",
                "defect_sensitive": False,
            },
        ])
        ad = self._make_adapter()
        orig_fuse_row = ad.fuse_row
        reads_after_2nd_write = [0]
        writes_done = [0]
        orig_write = ad.write_one_record

        def track_write(row: int, word: int) -> None:
            orig_write(row, word)
            writes_done[0] += 1

        # After the 2nd record's 2 post-write readbacks, fail any further fuse_row calls.
        post_second_readbacks = [0]

        def guarded_fuse_row(idx: int) -> int:
            if writes_done[0] >= 2:
                post_second_readbacks[0] += 1
                if post_second_readbacks[0] > 2:
                    reads_after_2nd_write[0] += 1
                    raise BurnError("simulated window collapse at session exit")
            return orig_fuse_row(idx)

        ad.write_one_record = track_write
        ad.fuse_row = guarded_fuse_row

        shown: list[str] = []
        with Journal(self.journal_path) as j:
            process_records(
                plan, ad, ad, j,
                stage="all",
                max_records=2,
                confirm=lambda *_: True,
                show=shown.append,
            )

        self.assertEqual(reads_after_2nd_write[0], 0)
        out = "\n".join(shown)
        self.assertIn("PARTIAL 'MEMORY' FUNCTIONAL GROUP BURNED", out)
        self.assertIn("DO NOT POWER OFF YET", out)
        sanity = [e for e in self._logged() if e["event"] == "poweroff_sanity_check"]
        self.assertEqual(len(sanity), 1)
        self.assertFalse(sanity[0]["safe_to_power_off"])
        self.assertEqual(sanity[0]["split_groups"], ["memory"])

    # ------------------------------------------------------------------
    # Resumed real stage reuses initial scan rows (no extra fuse_row reads)
    # ------------------------------------------------------------------

    def test_resumed_real_stage_reuses_initial_scan_rows(self):
        w_nv = encode_ff(1, 23, 0x0000)
        plan = self._make_plan([
            self._noop_rec(),
            self._fbp_rec(),
            {
                "name": "UNLOCK_NVLINK",
                "kind": "regular",
                "target": "NVLINK_DISABLE",
                "word": f"0x{w_nv:08x}",
                "decode": "clear nvlink disable",
                "defect_sensitive": False,
            },
        ])
        # Card already has test row (226) and first real record (227) committed; 228 is next free.
        rows = dict(self.base_rows)
        rows[226] = plan.records[0].word
        rows[227] = plan.records[1].word
        ad = SimulatedGA100Adapter(
            die=self.die,
            rows=rows,
            fields={**self.expected_masks, "FBP_DISABLE": 0x840},
            lnkcap2=0x06,
        )
        receipt = {
            "status": "PASS",
            "stage": "test",
            "plan_sha256": plan.content_hash,
            "test_row": 226,
            "test_word": f"0x{plan.records[0].word:08x}",
        }
        receipt_path = self.work / "receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        receipt_path.chmod(0o600)

        with Journal(self.journal_path) as j:
            ptr = process_records(
                plan, ad, ad, j,
                stage="real",
                test_receipt=receipt_path,
                confirm=lambda *_: True,
                show=lambda *_: None,
            )
        self.assertEqual(ptr, 229)
        self.assertEqual(_count_cmd(ad, CMD_PROGRAM), 1)
        # 16 die reads + 265 single initial scan + 0 resume re-reads + 6 target reads for record 2.
        self.assertEqual(_count_cmd(ad, CMD_READ), 16 + FF_REGION_LEN + 6)

    # ------------------------------------------------------------------
    # Console review / preview / scan single-pass deduplication
    # ------------------------------------------------------------------

    def test_console_review_preview_and_scan_single_pass(self):
        plan = self._make_plan([self._noop_rec(), self._fbp_rec()])
        ad = self._make_adapter()

        # 1. format_plan_review: 8 die-ID identity reads + 1 pass (265 reads), not 3 passes (795).
        ad.bus.clear()
        lines = format_plan_review(plan, reader=ad)
        self.assertTrue(any("rows 226..227" in ln for ln in lines))
        self.assertEqual(_count_cmd(ad, CMD_READ), 8 + FF_REGION_LEN)

        # 2. ManualConsole preview (auto row), preview (explicit row), and scan: 1 pass each.
        with Journal(self.journal_path) as j:
            console = ManualConsole(
                plan, ad, ad, j,
                execute=False,
                overrides=Overrides(),
                show=lambda *_: None,
                input_fn=lambda _: "",
            )

            ad.bus.clear()
            console._cmd_preview([f"0x{plan.records[1].word:08x}"])
            self.assertEqual(_count_cmd(ad, CMD_READ), FF_REGION_LEN)
            self.assertIsNotNone(console._staged)
            self.assertEqual(console._staged["row"], 226)

            ad.bus.clear()
            console._cmd_preview([f"0x{plan.records[1].word:08x}", "226"])
            self.assertEqual(_count_cmd(ad, CMD_READ), FF_REGION_LEN)
            self.assertIsNotNone(console._staged)
            self.assertEqual(console._staged["row"], 226)

            ad.bus.clear()
            console._cmd_scan()
            self.assertEqual(_count_cmd(ad, CMD_READ), FF_REGION_LEN)

    def test_scan_fpf_rows_passes_parameter(self):
        plan = self._make_plan([self._noop_rec()])
        ad = self._make_adapter()
        ad.bus.clear()
        ptr, _, rows = scan_fpf_rows(plan, ad)
        self.assertEqual(ptr, 226)
        self.assertEqual(len(rows), FF_REGION_LEN)
        self.assertEqual(_count_cmd(ad, CMD_READ), FF_REGION_LEN)

        ad.bus.clear()
        ptr2, _, _ = scan_fpf_rows(plan, ad, passes=2)
        self.assertEqual(ptr2, 226)
        self.assertEqual(_count_cmd(ad, CMD_READ), 2 * FF_REGION_LEN)


if __name__ == "__main__":
    unittest.main()
