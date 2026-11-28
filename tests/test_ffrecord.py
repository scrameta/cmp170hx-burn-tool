from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from vfi_burn.cli import main
from vfi_burn.engine import check_plan
from vfi_burn.ffrecord import (
    FF_ROW_MIN,
    build_vfi_plan_from_records,
    compute_sm_count,
    decode_ff,
    encode_ff,
    fields_in_block,
    make_noop_test_record,
    replay_ff_rows,
    simulate_ff_write,
)
from vfi_burn.model import BurnError, Plan


class TestFFRecord(unittest.TestCase):
    def test_kate_blog_and_ffcalc_verified_vectors(self):
        # Kate Land blog example: 0x0fe805c1 -> chain 1, block 23, data 0x0fe8
        self.assertEqual(encode_ff(1, 23, 0x0FE8), 0x0FE805C1)
        dec = decode_ff(0x0FE805C1)
        self.assertEqual((dec.chain, dec.block, dec.data), (1, 23, 0x0FE8))
        keys = {f.key for f in fields_in_block(1, 23)}
        self.assertIn("nvlink_disable", keys)
        self.assertIn("sys_pipe_disable", keys)

        # cmp170hx-ffcalc live-burned example: 0x80070141 -> chain 1, block 5, data 0x8007
        self.assertEqual(encode_ff(1, 5, 0x8007), 0x80070141)
        dec2 = decode_ff(0x80070141)
        self.assertEqual((dec2.chain, dec2.block, dec2.data), (1, 5, 0x8007))
        self.assertIn("fbio_disable", {f.key for f in fields_in_block(1, 5)})

    def test_decode_rejects_empty_and_iff_bit(self):
        with self.assertRaisesRegex(BurnError, "empty FF marker"):
            decode_ff(0x00000000)
        with self.assertRaisesRegex(BurnError, "IFF bit 5"):
            decode_ff(0x0FE805C1 | (1 << 5))

    def test_replay_later_record_wins_and_simulate_tracks_changes(self):
        # Chain 1 block 19 holds FBP_DISABLE (bits 307..318 -> block 19 bits 3..14).
        # FBP_DISABLE = 0x852 -> 0x852 << 3 = 0x4290 in block 19.
        # Replacement FBP_DISABLE = 0x840 -> 0x840 << 3 = 0x4200 in block 19.
        w_old = encode_ff(1, 19, 0x4290)
        w_new = encode_ff(1, 19, 0x4200)
        rows = {222: w_old, 223: 0}
        sim = simulate_ff_write(rows, 223, w_new)
        self.assertEqual(sim["old_block"], 0x4290)
        self.assertEqual(sim["new_block"], 0x4200)
        self.assertEqual(sim["changes"]["fbp_disable"], (0x852, 0x840))

        rows[223] = w_new
        blocks = replay_ff_rows(rows)
        self.assertEqual(blocks[1][19], 0x4200)

    def test_compute_sm_count(self):
        # All 8 GPCs enabled, 8 TPCs per GPC -> 128 SMs
        self.assertEqual(compute_sm_count(0x00, [0] * 8), 128)
        # 1 GPC disabled (bit 0) -> 7 * 16 = 112 SMs
        self.assertEqual(compute_sm_count(0x01, [0] * 8), 112)
        # 1 GPC disabled + 2 TPCs disabled on GPC 1 -> 112 - 4 = 108 SMs
        self.assertEqual(compute_sm_count(0x01, [0, 0x03, 0, 0, 0, 0, 0, 0]), 108)

    def test_build_vfi_plan_from_records_and_cli_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            work.chmod(0o700)
            die_rows = {i: 0x1000 + i for i in range(8, 16)}
            ff_rows = {r: 0 for r in range(FF_ROW_MIN, 487)}
            ff_rows[FF_ROW_MIN] = encode_ff(1, 7, 0x0123)
            ff_rows[FF_ROW_MIN + 1] = encode_ff(1, 19, 0x4290)

            plan_dict = build_vfi_plan_from_records(
                ff_rows=ff_rows,
                die_rows_int=die_rows,
                expected_masks={
                    "GPC_DISABLE": 0,
                    "FBP_DISABLE": 0x852,
                    "FBPA_DISABLE": 0x2460,
                    "ROP_L2_DISABLE": 0,
                },
                defect_masks={
                    "GPC_DISABLE": 0,
                    "FBP_DISABLE": 0x840,
                    "FBPA_DISABLE": 0,
                    "ROP_L2_DISABLE": 0,
                },
                planned_items=[
                    {
                        "row": FF_ROW_MIN + 3,
                        "chain": 1,
                        "block": 19,
                        "old": 0x4290,
                        "new": 0x4200,
                        "word": encode_ff(1, 19, 0x4200),
                        "fields": ["fbp_disable"],
                    }
                ],
                prepend_noop_test=True,
            )
            outp = work / "vfi_plan.json"
            outp.write_text(json.dumps(plan_dict))
            outp.chmod(0o600)

            loaded = Plan.load(outp)
            final_masks = check_plan(loaded)
            self.assertEqual(loaded.records[0].kind, "test")
            self.assertIn("NO-OP", loaded.records[0].decode)
            self.assertEqual(loaded.records[1].kind, "regular")
            self.assertEqual(final_masks["FBP_DISABLE"], 0x840)
            # base_sm_ceiling auto-computed from defect_masks (all 8 GPCs healthy -> 128)
            self.assertEqual(loaded.base_sm_ceiling, 128)

            # Verify decode CLI
            out2, err2 = io.StringIO(), io.StringIO()
            with redirect_stdout(out2), redirect_stderr(err2):
                rc2 = main(["decode", "0x0fe805c1"])
            self.assertEqual(rc2, 0, err2.getvalue())
            self.assertIn("chain=1 block=23 data=0x0fe8", out2.getvalue())
            self.assertIn("nvlink_disable", out2.getvalue())

    def test_build_plan_refuses_defective_bit_clear(self):
        die_rows = {i: 0x1000 + i for i in range(8, 16)}
        ff_rows = {r: 0 for r in range(FF_ROW_MIN, 487)}
        ff_rows[222] = encode_ff(1, 19, 0x4290)
        # Tries to clear FBP_DISABLE to 0x000 (clearing genuinely defective bits 0x840!).
        with self.assertRaisesRegex(BurnError, "genuinely defective"):
            build_vfi_plan_from_records(
                ff_rows=ff_rows,
                die_rows_int=die_rows,
                expected_masks={"FBP_DISABLE": 0x852},
                defect_masks={"FBP_DISABLE": 0x840},
                planned_items=[
                    {
                        "row": 224,
                        "chain": 1,
                        "block": 19,
                        "old": 0x4290,
                        "new": 0x0000,
                        "word": encode_ff(1, 19, 0x0000),
                        "fields": ["fbp_disable"],
                    }
                ],
            )

    def test_make_noop_test_record(self):
        rows = {222: encode_ff(1, 7, 0xABCD)}
        rec = make_noop_test_record(rows)
        self.assertEqual(rec["kind"], "test")
        self.assertEqual(int(rec["word"], 0), encode_ff(1, 7, 0xABCD))


if __name__ == "__main__":
    unittest.main()
