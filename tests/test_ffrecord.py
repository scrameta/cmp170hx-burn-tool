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
    compute_sm_count,
    decode_ff,
    encode_ff,
    fields_in_block,
    import_ffcalc_plan,
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

    def test_import_ffcalc_plan_and_cli_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            work.chmod(0o700)
            # Build minimal synthetic 512-row ffcalc dump + burn plan matching Kate's FBP example.
            ff_rows = {str(i): f"0x{0x1000 + i:08x}" for i in range(8, 16)}
            ff_rows[str(FF_ROW_MIN)] = f"0x{encode_ff(1, 7, 0x0123):08x}"
            ff_rows[str(FF_ROW_MIN + 1)] = f"0x{encode_ff(1, 19, 0x4290):08x}"
            for r in range(FF_ROW_MIN + 2, 487):
                ff_rows[str(r)] = "0x00000000"

            dump = {
                "sku": "8gb",
                "uuid": "GPU-00000000-0000-0000-0000-000000000001",
                "ff_rows": ff_rows,
                "option_words": {
                    "gpc_disable": "0x0",
                    "gpc_defective": "0x0",
                    "fbp_disable": "0x852",
                    "fbp_defective": "0x840",
                },
            }
            burn_plan = {
                "sku": "8gb",
                "uuid": dump["uuid"],
                "burn": [
                    {
                        "row": FF_ROW_MIN + 2,
                        "chain": 1,
                        "block": 19,
                        "old": "0x4290",
                        "new": "0x4200",
                        "record": f"0x{encode_ff(1, 19, 0x4200):08x}",
                        "fields": ["fbp_disable"],
                    }
                ],
                "targets": {"fbp_disable": "0x840"},
            }
            dp = work / "dump.json"
            bp = work / "burn.json"
            outp = work / "vfi_plan.json"
            dp.write_text(json.dumps(dump))
            bp.write_text(json.dumps(burn_plan))
            dp.chmod(0o600)
            bp.chmod(0o600)

            out_buf, err_buf = io.StringIO(), io.StringIO()
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                rc = main([
                    "import-ffcalc",
                    "--burn-plan", str(bp),
                    "--dump", str(dp),
                    "--out", str(outp),
                    "--prepend-noop-test",
                ])
            self.assertEqual(rc, 0, err_buf.getvalue())
            loaded = Plan.load(outp)
            final_masks = check_plan(loaded)
            self.assertEqual(loaded.records[0].kind, "test")
            self.assertIn("NO-OP", loaded.records[0].decode)
            self.assertEqual(loaded.records[1].kind, "regular")
            self.assertEqual(final_masks["FBP_DISABLE"], 0x840)

            # Verify decode CLI
            out2, err2 = io.StringIO(), io.StringIO()
            with redirect_stdout(out2), redirect_stderr(err2):
                rc2 = main(["decode", "0x0fe805c1"])
            self.assertEqual(rc2, 0, err2.getvalue())
            self.assertIn("chain=1 block=23 data=0x0fe8", out2.getvalue())
            self.assertIn("nvlink_disable", out2.getvalue())

    def test_import_ffcalc_refuses_defective_bit_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            work.chmod(0o700)
            ff_rows = {str(i): f"0x{0x1000 + i:08x}" for i in range(8, 16)}
            ff_rows["222"] = f"0x{encode_ff(1, 19, 0x4290):08x}"
            dump = {
                "sku": "8gb",
                "ff_rows": ff_rows,
                "option_words": {"fbp_disable": "0x852", "fbp_defective": "0x840"},
            }
            # Tries to clear FBP_DISABLE to 0x000 (clearing genuinely defective bits 0x840!).
            burn_plan = {
                "sku": "8gb",
                "burn": [{
                    "row": 223, "chain": 1, "block": 19, "old": "0x4290", "new": "0x0000",
                    "record": f"0x{encode_ff(1, 19, 0x0000):08x}", "fields": ["fbp_disable"],
                }],
            }
            dp, bp = work / "d.json", work / "b.json"
            dp.write_text(json.dumps(dump)); dp.chmod(0o600)
            bp.write_text(json.dumps(burn_plan)); bp.chmod(0o600)
            with self.assertRaisesRegex(BurnError, "genuinely defective"):
                import_ffcalc_plan(bp, dp)

    def test_make_noop_test_record(self):
        rows = {222: encode_ff(1, 7, 0xABCD)}
        rec = make_noop_test_record(rows)
        self.assertEqual(rec["kind"], "test")
        self.assertEqual(int(rec["word"], 0), encode_ff(1, 7, 0xABCD))


if __name__ == "__main__":
    unittest.main()
