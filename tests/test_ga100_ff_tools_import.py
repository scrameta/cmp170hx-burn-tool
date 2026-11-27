"""Tests for importing kprinssu/ga100-ff-tools plans into vfi_burn (schema_version=1)."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vfi_burn.cli import main
from vfi_burn.commands import (
    CommandIO,
    TargetSpec,
    execute_import_ga100_ff_tools,
    execute_plan_review,
    execute_simulate,
)
from vfi_burn.engine import Overrides, check_plan
from vfi_burn.ffrecord import (
    GA100_FF_TOOLS_FIELD_TO_KEY,
    decode_all_fields,
    encode_ff,
    ga100_ff_tools_fingerprint,
    import_ga100_ff_tools_plan,
    load_die_rows_from_file,
    replay_ff_rows,
)
from vfi_burn.fuses import FFCALC_KEY_TO_FIELD, FUSES_BY_KEY
from vfi_burn.interfaces import Health, Host
from vfi_burn.menu import BurnMenu, MenuSession
from vfi_burn.model import BurnError, Plan


SYNTHETIC_DIE_ROWS = {r: 0x3100 + r for r in range(8, 16)}


def _build_synthetic_ga100_fixture(work: Path, *, include_die_in_dump: bool = False) -> dict:
    """Build synthetic ga100-ff-tools dump, defects, and split plan JSONs (no real card data)."""
    # Baseline blocks covering SM/HBM, PCIe Gen3+Gen4, NVLink, and read-only Device IDs.
    # Chain 1:
    #   blk  7: FBPA bits (0x0198 -> 0x0180)
    #   blk  8: FBPA + GPC_DISABLE (bits 3..10): 0xFC2E -> GPC_DISABLE=0x85; 0xF826 -> GPC_DISABLE=0x04
    #   blk 19: FBP_DISABLE + ROP_L2: 0x4290 -> FBP=0x852; 0x4200 -> FBP=0x840
    #   blk 23: NVLINK (0..2) + unmapped(3)=1 + SYS_PIPE (4..11)=0xFE: 0x0FEF -> 0x0FE8
    # Chain 2:
    #   blk 37/38: PCIE_DEVIDA/B = 0x20C2 (bit 1..16 -> 0x4184)
    #   blk 39: DISABLE_GEN3_SPEED (bit1+bit9): 0x0782 -> 0x0580
    #   blk 46: MAGIC_BITS_D bit19: 0xA000 -> 0x8000
    #   blk 47: MAGIC_BITS_D bits 25,22,26,28: 0x0059 -> 0x0000
    #   blk 49: BOOT_GEN23 + BOOT_GEN3: 0x00E4 -> 0x0024
    base_blocks = [
        (1, 7, 0x0198),
        (1, 8, 0xFC2E),
        (1, 19, 0x4290),
        (1, 23, 0x0FEF),
        (2, 37, 0x4184),
        (2, 38, 0x4184),
        (2, 39, 0x0782),
        (2, 46, 0xA000),
        (2, 47, 0x0059),
        (2, 49, 0x00E4),
    ]
    rows_map: dict[int, int] = {}
    if include_die_in_dump:
        rows_map.update(SYNTHETIC_DIE_ROWS)
    for idx, (c, b, d) in enumerate(base_blocks):
        rows_map[222 + idx] = encode_ff(c, b, d)
    last_row = 222 + len(base_blocks) - 1  # 231
    for r in range(last_row + 1, 488):
        rows_map[r] = 0

    decoded_base = decode_all_fields(replay_ff_rows(rows_map))
    fp = ga100_ff_tools_fingerprint(rows_map, last_row)
    baseline = {"last_row": last_row, "fingerprint": fp}
    unchanged = {
        "OPT_SYS_PIPE_DISABLE": f"0x{decoded_base['sys_pipe_disable']:X}",
        "OPT_PCIE_DEVIDA": "0x20C2",
        "OPT_PCIE_DEVIDB": "0x20C2",
    }

    row_list = [
        {"row": r, "raw": f"0x{rows_map[r]:08X}"}
        for r in sorted(rows_map)
    ]
    dump_obj = {
        "bdf": "0000:05:00.0",
        "read_only": True,
        "shadow_only": False,
        "fusectrl": "0x00040000",
        "crc_status": "0x55",
        "controller_idle": True,
        "shadow": {
            "OPT_GPC_DISABLE": f"0x{decoded_base['gpc_disable']:08X}",
            "OPT_FBP_DISABLE": f"0x{decoded_base['fbp_disable']:08X}",
            "OPT_FBPA_DISABLE": f"0x{decoded_base['fbpa_disable']:08X}",
            "OPT_FBIO_DISABLE": f"0x{decoded_base['fbio_disable']:08X}",
            "OPT_PCIE_DEVIDA": "0x000020C2",
            "OPT_PCIE_DEVIDB": "0x000020C2",
            # Runtime STATUS register intentionally differs from fused shadow (must be ignored).
            "STATUS_OPT_GPC": "0x000000FF",
            **{
                f"OPT_TPC_DISABLE_GPC{i}": f"0x{decoded_base[f'tpc_gpc{i}_disable']:08X}"
                for i in range(8)
            },
        },
        "rows": row_list,
        "blank_rows": list(range(last_row + 1, 488)),
        "read_errors": [],
    }

    # Apply planned SM/HBM changes to compute exact expect_fields & matching defect masks.
    after_sm = dict(rows_map)
    after_sm[last_row + 1] = encode_ff(1, 7, 0x0198)  # 0_noop
    after_sm[last_row + 2] = encode_ff(1, 7, 0x0180)
    after_sm[last_row + 3] = encode_ff(1, 8, 0xF826)
    after_sm[last_row + 4] = encode_ff(1, 19, 0x4200)
    dec_sm = decode_all_fields(replay_ff_rows(after_sm))

    defects_obj = {
        "bdf": "0000:05:00.0",
        "registers": [
            {"name": "OPT_GPC_DEFECTIVE", "value": f"0x{dec_sm['gpc_disable']:08X}"},
            {"name": "OPT_FBP_DEFECTIVE", "value": f"0x{dec_sm['fbp_disable']:08X}"},
            {"name": "OPT_FBPA_DEFECTIVE", "value": f"0x{dec_sm['fbpa_disable']:08X}"},
            {"name": "OPT_FBIO_DEFECTIVE", "value": f"0x{dec_sm['fbio_disable']:08X}"},
            {"name": "OPT_PES_DEFECTIVE", "value": f"0x{dec_sm['pes_disable']:08X}"},
            {"name": "OPT_ROP_L2_DEFECTIVE", "value": f"0x{dec_sm['rop_l2_disable']:08X}"},
            {"name": "OPT_NVLINK_DEFECTIVE", "value": "0x00000000"},
            {"name": "OPT_PES_DISABLE", "value": f"0x{decoded_base['pes_disable']:08X}"},
            {"name": "OPT_ROP_L2_DISABLE", "value": f"0x{decoded_base['rop_l2_disable']:08X}"},
            *[
                {"name": f"OPT_TPC_GPC{i}_DEFECTIVE", "value": f"0x{dec_sm[f'tpc_gpc{i}_disable']:08X}"}
                for i in range(8)
            ],
        ],
    }

    noop_plan = {
        "mode": "noop",
        "order": 0,
        "card": "syn_8gb",
        "baseline": baseline,
        "unchanged_fields": unchanged,
        "records": [
            {
                "chain": 1,
                "block": 7,
                "old_data": "0x0198",
                "new_data": "0x0198",
                "encoded": f"0x{encode_ff(1, 7, 0x0198):08X}",
                "note": "mechanism test, value unchanged",
                "row": last_row + 1,
            }
        ],
    }

    sm_plan = {
        "mode": "sm_hbm",
        "order": 1,
        "card": "syn_8gb",
        "baseline": baseline,
        "unchanged_fields": unchanged,
        "records": [
            {
                "chain": 1,
                "block": 7,
                "old_data": "0x0198",
                "new_data": "0x0180",
                "encoded": f"0x{encode_ff(1, 7, 0x0180):08X}",
                "row": last_row + 2,
            },
            {
                "chain": 1,
                "block": 8,
                "old_data": "0xFC2E",
                "new_data": "0xF826",
                "encoded": f"0x{encode_ff(1, 8, 0xF826):08X}",
                "row": last_row + 3,
            },
            {
                "chain": 1,
                "block": 19,
                "old_data": "0x4290",
                "new_data": "0x4200",
                "encoded": f"0x{encode_ff(1, 19, 0x4200):08X}",
                "row": last_row + 4,
            },
        ],
        "expect_fields": {
            "OPT_GPC_DISABLE": f"0x{dec_sm['gpc_disable']:X}",
            "OPT_FBP_DISABLE": f"0x{dec_sm['fbp_disable']:X}",
            "OPT_FBPA_DISABLE": f"0x{dec_sm['fbpa_disable']:X}",
            "OPT_FBIO_DISABLE": f"0x{dec_sm['fbio_disable']:X}",
            "OPT_ROP_L2_DISABLE": f"0x{dec_sm['rop_l2_disable']:X}",
        },
    }

    pcie_plan = {
        "mode": "pcie_gen3_gen4",
        "order": 2,
        "card": "syn_8gb",
        "baseline": baseline,
        "unchanged_fields": unchanged,
        "records": [
            {
                "chain": 2,
                "block": 39,
                "old_data": "0x0782",
                "new_data": "0x0580",
                "encoded": f"0x{encode_ff(2, 39, 0x0580):08X}",
                "note": "DISABLE_GEN3_SPEED (bit1+bit9)",
                "row": last_row + 5,
            },
            {
                "chain": 2,
                "block": 49,
                "old_data": "0x00E4",
                "new_data": "0x0024",
                "encoded": f"0x{encode_ff(2, 49, 0x0024):08X}",
                "note": "BOOT_GEN23 + BOOT_GEN3 disable",
                "row": last_row + 6,
            },
            {
                "chain": 2,
                "block": 47,
                "old_data": "0x0059",
                "new_data": "0x0000",
                "encoded": f"0x{encode_ff(2, 47, 0x0000):08X}",
                "note": "PCIE_MAGIC_BITS_D bits 25,22,26,28 (Gen4)",
                "row": last_row + 7,
            },
            {
                "chain": 2,
                "block": 46,
                "old_data": "0xA000",
                "new_data": "0x8000",
                "encoded": f"0x{encode_ff(2, 46, 0x8000):08X}",
                "note": "PCIE_MAGIC_BITS_D bit19 (Gen4)",
                "row": last_row + 8,
            },
        ],
        "expect_registers": {
            "DISABLE_GEN3_SPEED 0x820250": "0x00000000",
            "BOOT_GEN23_DISABLE 0x82057C": "0x00000000",
            "BOOT_GEN3_DISABLE 0x820580": "0x00000000",
            "PCIE_MAGIC_BITS_D 0x820520": "0x00200000",
        },
    }

    nvlink_plan = {
        "mode": "nvlink",
        "order": 3,
        "card": "syn_8gb",
        "baseline": baseline,
        "unchanged_fields": unchanged,
        "records": [
            {
                "chain": 1,
                "block": 23,
                "old_data": "0x0FEF",
                "new_data": "0x0FE8",
                "encoded": f"0x{encode_ff(1, 23, 0x0FE8):08X}",
                "row": last_row + 9,
            }
        ],
        "expect_fields": {
            "OPT_NVLINK_DISABLE": "0x0",
        },
    }

    def _w(name: str, obj: dict) -> Path:
        p = work / name
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        p.chmod(0o600)
        return p

    die_snap = _w(
        "die_snapshot.json",
        {
            "pci_id": "10de:20c2",
            "die_rows": {str(r): f"0x{v:08x}" for r, v in SYNTHETIC_DIE_ROWS.items()},
        },
    )
    return {
        "dump": _w("card-dump.json", dump_obj),
        "defects": _w("card-defects.json", defects_obj),
        "noop": _w("syn_8gb_plan_0_noop.json", noop_plan),
        "sm": _w("syn_8gb_plan_sm_hbm.json", sm_plan),
        "pcie": _w("syn_8gb_plan_pcie.json", pcie_plan),
        "nvlink": _w("syn_8gb_plan_nvlink.json", nvlink_plan),
        "die_snap": die_snap,
        "rows_map": rows_map,
        "last_row": last_row,
        "decoded_base": decoded_base,
    }


class TestGA100FFToolsImport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.work.chmod(0o700)
        self.fx = _build_synthetic_ga100_fixture(self.work)

    def _write_json(self, name: str, obj: dict) -> Path:
        p = self.work / name
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        p.chmod(0o600)
        return p

    def test_plan_load_refuses_raw_ga100_ff_tools_and_ffcalc_with_actionable_hint(self):
        with self.assertRaisesRegex(BurnError, "only schema_version=1 is understood.*import-ga100-ff-tools"):
            Plan.load(self.fx["sm"])

        raw_ffcalc = self._write_json(
            "raw_ffcalc.json",
            {"uuid": "GPU-SYN", "burn": [{"row": 232, "chain": 1, "block": 7, "new": 0x0180, "record": "0x018001c1"}]},
        )
        with self.assertRaisesRegex(BurnError, "only schema_version=1 is understood.*import-ffcalc"):
            Plan.load(raw_ffcalc)

    def test_import_noop_plus_sm_hbm_passes_strict_check_without_duplicate_noop(self):
        # Pass plans out of order; importer must sort by starting row and reuse 0_noop as record 0.
        plan_dict = import_ga100_ff_tools_plan(
            [self.fx["sm"], self.fx["noop"]],
            self.fx["dump"],
            self.fx["defects"],
            base_sm_ceiling=122,
            die_rows_path=self.fx["die_snap"],
        )
        out = self._write_json("vfi_sm.json", plan_dict)
        loaded = Plan.load(out)
        check_plan(loaded, allow_unmapped_ff=False)
        self.assertEqual(len(loaded.records), 4)
        self.assertEqual(loaded.records[0].kind, "test")
        self.assertEqual(loaded.records[0].word, encode_ff(1, 7, 0x0198))
        self.assertEqual(loaded.expected_next_row, self.fx["last_row"] + 1)
        self.assertEqual(loaded.die_rows, SYNTHETIC_DIE_ROWS)
        self.assertEqual(plan_dict["source_format"], "ga100-ff-tools")
        self.assertEqual(plan_dict["source_ga100_ff_tools_modes"], ["noop", "sm_hbm"])

    def test_import_sm_hbm_only_auto_prepends_noop_into_reserved_slot(self):
        # sm_hbm starts at last_row + 2 because 0_noop occupied last_row + 1; auto-prepend fills it.
        plan_dict = import_ga100_ff_tools_plan(
            [self.fx["sm"]],
            self.fx["dump"],
            self.fx["defects"],
            base_sm_ceiling=122,
            die_rows_path=self.fx["die_snap"],
        )
        out = self._write_json("vfi_sm_only.json", plan_dict)
        loaded = Plan.load(out)
        check_plan(loaded, allow_unmapped_ff=False)
        self.assertEqual(len(loaded.records), 4)
        self.assertEqual(loaded.records[0].kind, "test")
        self.assertTrue(loaded.records[0].name.startswith("NOOP_TEST_"))

        # Explicitly disabling prepend when record 0 is NOT a no-op must refuse.
        with self.assertRaisesRegex(BurnError, "NOT a no-op"):
            import_ga100_ff_tools_plan(
                [self.fx["sm"]],
                self.fx["dump"],
                self.fx["defects"],
                base_sm_ceiling=122,
                die_rows_path=self.fx["die_snap"],
                prepend_noop_test=False,
            )

    def test_import_all_four_plans_review_and_sim_pcie_burn_with_override_gate(self):
        out = self.work / "vfi_all4.json"
        shown: list[str] = []
        io_obj = CommandIO(show=shown.append, confirm=lambda *_: True)
        rc = execute_import_ga100_ff_tools(
            plans=[
                str(self.fx["nvlink"]),
                str(self.fx["pcie"]),
                str(self.fx["sm"]),
                str(self.fx["noop"]),
            ],
            dump=str(self.fx["dump"]),
            defects=str(self.fx["defects"]),
            out=str(out),
            base_sm=122,
            die_rows_from=str(self.fx["die_snap"]),
            io=io_obj,
        )
        self.assertEqual(rc, 0)
        self.assertTrue(any("opt-in/override-gated" in line for line in shown))

        loaded = Plan.load(out)
        self.assertEqual(len(loaded.records), 9)  # 1 noop + 3 sm + 4 pcie + 1 nvlink

        # Offline plan review must succeed and surface both experimental warnings and BURN GATE.
        review_out: list[str] = []
        rc_rev = execute_plan_review(
            plan_path=str(out),
            io=CommandIO(show=review_out.append),
        )
        self.assertEqual(rc_rev, 0)
        rev_text = "\n".join(review_out)
        self.assertIn("BURN GATE: strict burn requires --override-unmapped-ff", rev_text)
        self.assertIn("PCIE_MAGIC_BITS_D", rev_text)

        # Build a simulated PCIe snapshot matching this synthetic card and test simulate:
        # without --override-unmapped-ff it MUST refuse; with it, all 9 records burn cleanly.
        sim_snap = self._write_json(
            "sim_card.json",
            {
                "pci_id": "10de:20c2",
                "die_rows": {str(r): f"0x{v:08x}" for r, v in SYNTHETIC_DIE_ROWS.items()},
                "fields": {
                    "GPC_DISABLE": f"0x{self.fx['decoded_base']['gpc_disable']:x}",
                    "FBP_DISABLE": f"0x{self.fx['decoded_base']['fbp_disable']:x}",
                    "FBPA_DISABLE": f"0x{self.fx['decoded_base']['fbpa_disable']:x}",
                    "FBIO_DISABLE": f"0x{self.fx['decoded_base']['fbio_disable']:x}",
                    "PES_DISABLE": f"0x{self.fx['decoded_base']['pes_disable']:x}",
                    "ROP_L2_DISABLE": f"0x{self.fx['decoded_base']['rop_l2_disable']:x}",
                    "NVLINK_DISABLE": f"0x{self.fx['decoded_base']['nvlink_disable']:x}",
                    "LnkCap2": "0x06",
                    **{
                        f"TPC_GPC{i}_DISABLE": f"0x{self.fx['decoded_base'][f'tpc_gpc{i}_disable']:x}"
                        for i in range(8)
                    },
                },
                "rows": {str(r): f"0x{v:08x}" for r, v in self.fx["rows_map"].items() if 222 <= r <= 486},
                "health": {
                    "fusectrl": "0xE0040000",
                    "dsov": 0,
                    "crc": "0x55",
                    "plm": "0xFFFFFFFF",
                    "plm_f8": "0xFFFFFFFF",
                    "en_sw": 0,
                },
                "host": {
                    "bound_driver": "pci-stub",
                    "nvidia_loaded": False,
                    "nouveau_loaded": False,
                    "lockdown": "[none]",
                    "hold_active": True,
                    "boot_id": "boot-ga100-sim",
                },
                "registers": {"0x820060": "0x00000000"},
            },
        )
        journal = self.work / "sim.jsonl"

        with self.assertRaisesRegex(BurnError, "opt-in field"):
            execute_simulate(
                plan_path=str(out),
                target=TargetSpec(sim_pcie=str(sim_snap)),
                stage="all",
                journal_path=str(journal),
                io=CommandIO(show=lambda *_: None, confirm=lambda *_: True),
            )

        sim_out: list[str] = []
        rc_sim = execute_simulate(
            plan_path=str(out),
            target=TargetSpec(sim_pcie=str(sim_snap)),
            stage="all",
            journal_path=str(journal),
            overrides=Overrides(unmapped_ff=True),
            io=CommandIO(show=sim_out.append, confirm=lambda *_: True),
        )
        self.assertEqual(rc_sim, 0, "\n".join(sim_out))
        self.assertTrue(any("SIMULATION ONLY: fake records updated" in line for line in sim_out))
        events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        verified = [e for e in events if e.get("event") == "verified_readback_only"]
        self.assertEqual(len(verified), 9)

    def test_die_rows_sources_and_missing_refusal(self):
        # 1. Missing die rows (not in dump, no --die-rows-from, no --bdf) must fail closed.
        with self.assertRaisesRegex(BurnError, "Die ID rows 8..15"):
            import_ga100_ff_tools_plan(
                [self.fx["noop"], self.fx["sm"]],
                self.fx["dump"],
                self.fx["defects"],
                base_sm_ceiling=122,
            )

        # 2. Dump captured with ff_dump.py --start 8 works with no extra die file.
        sub = self.work / "with_die"
        sub.mkdir(mode=0o700)
        fx_die = _build_synthetic_ga100_fixture(sub, include_die_in_dump=True)
        plan_dict = import_ga100_ff_tools_plan(
            [fx_die["noop"], fx_die["sm"]],
            fx_die["dump"],
            fx_die["defects"],
            base_sm_ceiling=122,
        )
        self.assertEqual(
            {int(k): int(v, 16) for k, v in plan_dict["card"]["die_rows"].items()},
            SYNTHETIC_DIE_ROWS,
        )

        # 3. load_die_rows_from_file accepts ff_dump --start 8, ffcalc dump, and plain dict.
        self.assertEqual(load_die_rows_from_file(fx_die["dump"]), SYNTHETIC_DIE_ROWS)
        plain = self._write_json("plain_die.json", {str(k): f"0x{v:08x}" for k, v in SYNTHETIC_DIE_ROWS.items()})
        self.assertEqual(load_die_rows_from_file(plain), SYNTHETIC_DIE_ROWS)
        ffcalc_d = self._write_json(
            "ffcalc_die.json",
            {"ff_rows": {str(k): f"0x{v:08x}" for k, v in SYNTHETIC_DIE_ROWS.items()}},
        )
        self.assertEqual(load_die_rows_from_file(ffcalc_d), SYNTHETIC_DIE_ROWS)

    def test_refuses_fingerprint_mismatch_mixed_baselines_defects_and_shadow_mismatch(self):
        # Baseline fingerprint mismatch
        bad_sm = json.loads(self.fx["sm"].read_text())
        bad_sm["baseline"]["fingerprint"] = "0000000000000000"
        bad_sm_p = self._write_json("bad_fp.json", bad_sm)
        with self.assertRaisesRegex(BurnError, "fingerprint"):
            import_ga100_ff_tools_plan(
                [bad_sm_p],
                self.fx["dump"],
                self.fx["defects"],
                base_sm_ceiling=122,
                die_rows_path=self.fx["die_snap"],
            )

        # Mixed baselines across plans
        with self.assertRaisesRegex(BurnError, "different baseline"):
            import_ga100_ff_tools_plan(
                [self.fx["noop"], bad_sm_p],
                self.fx["dump"],
                self.fx["defects"],
                base_sm_ceiling=122,
                die_rows_path=self.fx["die_snap"],
            )

        # Genuine defect violation (mark GPC0 defective while sm_hbm clears GPC0)
        bad_def = json.loads(self.fx["defects"].read_text())
        for reg in bad_def["registers"]:
            if reg["name"] == "OPT_GPC_DEFECTIVE":
                reg["value"] = "0x00000005"  # bit 0 + bit 2 defective; sm_hbm enables bit 0
        bad_def_p = self._write_json("bad_defects.json", bad_def)
        with self.assertRaisesRegex(BurnError, "genuinely defective GPC_DISABLE"):
            import_ga100_ff_tools_plan(
                [self.fx["noop"], self.fx["sm"]],
                self.fx["dump"],
                bad_def_p,
                base_sm_ceiling=122,
                die_rows_path=self.fx["die_snap"],
            )

        # Shadow vs FF replay mismatch (e.g., unsensed dump)
        bad_dump = json.loads(self.fx["dump"].read_text())
        bad_dump["shadow"]["OPT_GPC_DISABLE"] = "0x00000000"
        bad_dump_p = self._write_json("bad_shadow.json", bad_dump)
        with self.assertRaisesRegex(BurnError, "shadow OPT_GPC_DISABLE.*disagrees"):
            import_ga100_ff_tools_plan(
                [self.fx["noop"], self.fx["sm"]],
                bad_dump_p,
                self.fx["defects"],
                base_sm_ceiling=122,
                die_rows_path=self.fx["die_snap"],
            )

        # expect_fields mismatch
        bad_exp = json.loads(self.fx["sm"].read_text())
        bad_exp["expect_fields"]["OPT_GPC_DISABLE"] = "0x0"
        bad_exp_p = self._write_json("bad_expect.json", bad_exp)
        with self.assertRaisesRegex(BurnError, "expect_fields.OPT_GPC_DISABLE"):
            import_ga100_ff_tools_plan(
                [self.fx["noop"], bad_exp_p],
                self.fx["dump"],
                self.fx["defects"],
                base_sm_ceiling=122,
                die_rows_path=self.fx["die_snap"],
            )

    def test_cli_import_ga100_ff_tools_and_live_bdf_die_read(self):
        out = self.work / "cli_out.json"

        class _FakeLive:
            def __init__(self, bdf, writable=False):
                assert not writable
                self.bdf = bdf

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def host(self):
                return Host(
                    bound_driver="pci-stub",
                    nvidia_loaded=False,
                    nouveau_loaded=False,
                    lockdown="[none]",
                    hold_active=True,
                    boot_id="boot-1",
                )

            def pci_id(self):
                return "10de:20c2"

            def die_row(self, idx: int) -> int:
                return SYNTHETIC_DIE_ROWS[idx]

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("vfi_burn.commands.GA100LiveAdapter", _FakeLive):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "import-ga100-ff-tools",
                        "--plans",
                        str(self.fx["noop"]),
                        str(self.fx["sm"]),
                        "--dump",
                        str(self.fx["dump"]),
                        "--defects",
                        str(self.fx["defects"]),
                        "--bdf",
                        "0000:05:00.0",
                        "--base-sm",
                        "122",
                        "--out",
                        str(out),
                    ]
                )
        self.assertEqual(rc, 0, stderr.getvalue())
        self.assertIn("Imported ga100-ff-tools plan", stdout.getvalue())
        loaded = Plan.load(out)
        self.assertEqual(loaded.die_rows, SYNTHETIC_DIE_ROWS)
        self.assertEqual(len(loaded.records), 4)

    def _make_fake_live_cls(
        self,
        *,
        rows_map: dict[int, int],
        defects_by_vfi: dict[str, int],
        shadow_decoded: dict[str, int] | None = None,
        efp: int = 1,
        dsov: int = 1,
    ):
        dec = shadow_decoded if shadow_decoded is not None else decode_all_fields(replay_ff_rows(rows_map))
        reg_map: dict[int, int] = {}
        for ff_key, vfi_name in FFCALC_KEY_TO_FIELD.items():
            d_addr = FUSES_BY_KEY[ff_key].defect_addr
            if d_addr is not None:
                reg_map[d_addr] = defects_by_vfi.get(vfi_name, 0)
        for opt_name, ff_key in GA100_FF_TOOLS_FIELD_TO_KEY.items():
            f_addr = FUSES_BY_KEY[ff_key].fused_addr
            if f_addr is not None:
                reg_map[f_addr] = dec[ff_key]

        class _FakeLiveFull:
            def __init__(self, bdf, writable=False):
                assert not writable
                self.bdf = bdf

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def host(self):
                return Host(
                    bound_driver="pci-stub",
                    nvidia_loaded=False,
                    nouveau_loaded=False,
                    lockdown="[none]",
                    hold_active=True,
                    boot_id="boot-live-1",
                )

            def pci_id(self):
                return "10de:20c2"

            def health(self):
                return Health(
                    fusectrl=0xE0040000,
                    dsov=dsov,
                    crc=0x55,
                    plm=0xFFFFFFFF,
                    plm_f8=0xFFFFFFFF,
                    en_sw=0,
                    efp=efp,
                )

            def die_row(self, idx: int) -> int:
                return SYNTHETIC_DIE_ROWS[idx]

            def fuse_row(self, idx: int) -> int:
                return rows_map.get(idx, 0)

            def register(self, addr: int) -> int:
                return reg_map[addr]

        return _FakeLiveFull

    def _plans_only_dir(self) -> Path:
        """Directory containing ONLY the 4 ga100-ff-tools plan JSONs (no dump or defects files)."""
        d = self.work / "plans_only"
        d.mkdir(mode=0o700)
        for key in ("noop", "sm", "pcie", "nvlink"):
            src = self.fx[key]
            dst = d / src.name
            dst.write_bytes(src.read_bytes())
            dst.chmod(0o600)
        return d

    def _defects_by_vfi(self) -> dict[str, int]:
        raw = json.loads(self.fx["defects"].read_text())["registers"]
        reg_vals = {r["name"]: int(r["value"], 16) for r in raw}
        out = {
            "GPC_DISABLE": reg_vals["OPT_GPC_DEFECTIVE"],
            "FBP_DISABLE": reg_vals["OPT_FBP_DEFECTIVE"],
            "FBPA_DISABLE": reg_vals["OPT_FBPA_DEFECTIVE"],
            "FBIO_DISABLE": reg_vals["OPT_FBIO_DEFECTIVE"],
            "PES_DISABLE": reg_vals["OPT_PES_DEFECTIVE"],
            "ROP_L2_DISABLE": reg_vals["OPT_ROP_L2_DEFECTIVE"],
            "NVLINK_DISABLE": reg_vals["OPT_NVLINK_DEFECTIVE"],
            "SYS_PIPE_DISABLE": 0x0,
        }
        for i in range(8):
            out[f"TPC_GPC{i}_DISABLE"] = reg_vals[f"OPT_TPC_GPC{i}_DEFECTIVE"]
        return out

    def test_menu_setup_option_h_auto_detects_dump_and_defects(self):
        out = self.work / "menu_ga100.json"
        responses = iter(
            [
                "1",  # Setup
                "h",  # Import ga100-ff-tools
                f"{self.fx['noop']} {self.fx['sm']}",  # plans (dump+defects auto-detected in same dir!)
                str(self.fx["die_snap"]),              # die rows JSON
                str(out),                              # output path
                "122",                                 # base SM ceiling
                "0",                                   # back to main menu
                "2",                                   # Review Plan
                "0",                                   # quit
            ]
        )
        lines: list[str] = []
        session = MenuSession(journal_path=str(self.work / "menu.jsonl"))
        menu = BurnMenu(session, input_fn=lambda p="": next(responses), show=lines.append, check_tty=False)
        rc = menu.run()
        output = "\n".join(lines)
        self.assertEqual(rc, 0, output)
        self.assertEqual(session.plan_path, str(out))
        self.assertIn("Auto-detected card dump", output)
        self.assertIn("Imported & loaded", output)
        self.assertIn("=== PLAN REVIEW & EXPECTED OUTCOME", output)

    def test_cli_import_directory_only_from_live_gpu_without_dump_or_defects_files(self):
        plans_dir = self._plans_only_dir()
        out = self.work / "cli_live_dir.json"
        fake_cls = self._make_fake_live_cls(
            rows_map=self.fx["rows_map"],
            defects_by_vfi=self._defects_by_vfi(),
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("vfi_burn.commands.GA100LiveAdapter", fake_cls):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = main(
                    [
                        "import-ga100-ff-tools",
                        "--plans",
                        str(plans_dir),
                        "--bdf",
                        "0000:05:00.0",
                        "--base-sm",
                        "122",
                        "--out",
                        str(out),
                    ]
                )
        self.assertEqual(rc, 0, stderr.getvalue())
        loaded = Plan.load(out)
        self.assertEqual(len(loaded.records), 9)  # all 4 plans: 1 noop + 3 sm + 4 pcie + 1 nvlink
        self.assertIn("SYS_PIPE_DISABLE", loaded.defect_masks)  # read live from 0x8203AC!
        self.assertEqual(loaded.die_rows, SYNTHETIC_DIE_ROWS)

    def test_import_single_pcie_plan_on_fresh_and_partially_burned_card(self):
        plans_dir = self._plans_only_dir()
        pcie_only = plans_dir / self.fx["pcie"].name

        # 1. Fresh card (rows 222..231 occupied, next free = 232; pcie numbered 236..239)
        fake_fresh = self._make_fake_live_cls(
            rows_map=self.fx["rows_map"],
            defects_by_vfi=self._defects_by_vfi(),
        )
        out_fresh = self.work / "pcie_fresh.json"
        with patch("vfi_burn.commands.GA100LiveAdapter", fake_fresh):
            rc = execute_import_ga100_ff_tools(
                plans=[str(pcie_only)],
                bdf="0000:05:00.0",
                base_sm=122,
                out=str(out_fresh),
                io=CommandIO(show=lambda *_: None),
            )
        self.assertEqual(rc, 0)
        p_fresh = Plan.load(out_fresh)
        self.assertEqual(len(p_fresh.records), 5)  # 1 auto-prepended no-op + 4 pcie
        self.assertEqual(p_fresh.expected_next_row, self.fx["last_row"] + 1)

        # 2. Partially burned card (0_noop + sm_hbm already burned at rows 232..235; next free = 236)
        #    Test with BOTH cold-booted shadows and same-window (pre-burn) shadows!
        partial_rows = dict(self.fx["rows_map"])
        partial_rows[self.fx["last_row"] + 1] = encode_ff(1, 7, 0x0198)
        partial_rows[self.fx["last_row"] + 2] = encode_ff(1, 7, 0x0180)
        partial_rows[self.fx["last_row"] + 3] = encode_ff(1, 8, 0xF826)
        partial_rows[self.fx["last_row"] + 4] = encode_ff(1, 19, 0x4200)
        for label, shadow_dec in (
            ("cold_booted", None),
            ("same_window", self.fx["decoded_base"]),
        ):
            with self.subTest(shadows=label):
                fake_partial = self._make_fake_live_cls(
                    rows_map=partial_rows,
                    defects_by_vfi=self._defects_by_vfi(),
                    shadow_decoded=shadow_dec,
                )
                out_part = self.work / f"pcie_partial_{label}.json"
                with patch("vfi_burn.commands.GA100LiveAdapter", fake_partial):
                    rc2 = execute_import_ga100_ff_tools(
                        plans=[str(pcie_only)],
                        bdf="0000:05:00.0",
                        base_sm=122,
                        out=str(out_part),
                        io=CommandIO(show=lambda *_: None),
                    )
                self.assertEqual(rc2, 0)
                p_part = Plan.load(out_part)
                self.assertEqual(p_part.expected_next_row, self.fx["last_row"] + 5)
                self.assertEqual(len(p_part.records), 5)

    def test_unsensed_live_gpu_efp_zero_refuses_defect_read_but_allows_defects_file(self):
        plans_dir = self._plans_only_dir()
        fake_unsensed = self._make_fake_live_cls(
            rows_map=self.fx["rows_map"],
            defects_by_vfi={k: 0 for k in self._defects_by_vfi()},
            efp=0,
            dsov=0,
        )
        out = self.work / "unsensed.json"
        # Without --defects, EFP=0 MUST refuse rather than trusting zeroed defect shadows.
        with patch("vfi_burn.commands.GA100LiveAdapter", fake_unsensed):
            with self.assertRaisesRegex(BurnError, "UNSENSED.*EFP=0"):
                execute_import_ga100_ff_tools(
                    plans=[str(plans_dir)],
                    bdf="0000:05:00.0",
                    base_sm=122,
                    out=str(out),
                    io=CommandIO(show=lambda *_: None),
                )

        # With cold --defects supplied, CMD_READ of live FF + Die ID rows succeeds even at EFP=0!
        with patch("vfi_burn.commands.GA100LiveAdapter", fake_unsensed):
            rc = execute_import_ga100_ff_tools(
                plans=[str(plans_dir / self.fx["noop"].name), str(plans_dir / self.fx["sm"].name)],
                defects=str(self.fx["defects"]),
                bdf="0000:05:00.0",
                base_sm=122,
                out=str(out),
                io=CommandIO(show=lambda *_: None),
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(Plan.load(out).records), 4)

    def test_menu_option_a_accepts_ga100_directory_with_live_gpu_and_subset_pick(self):
        plans_dir = self._plans_only_dir()
        out = self.work / "menu_dir_live.json"
        fake_cls = self._make_fake_live_cls(
            rows_map=self.fx["rows_map"],
            defects_by_vfi=self._defects_by_vfi(),
        )
        responses = iter(
            [
                "1",             # Setup
                "a",             # Select plan JSON -> point straight at directory!
                str(plans_dir),  # ga100-ff-tools directory with only 4 plan files
                "y",             # Import into a vfi_burn plan now?
                "1,3",           # Pick [1] 0_noop + [3] pcie
                "0000:05:00.0",  # Live GPU PCI BDF (no dump/defects JSON prompts!)
                str(out),        # Output path
                "122",           # Base SM ceiling
                "0",             # Back to main menu
                "2",             # Review Plan
                "0",             # Quit
            ]
        )
        lines: list[str] = []
        session = MenuSession(journal_path=str(self.work / "menu_dir.jsonl"))
        with patch("vfi_burn.menu.GA100LiveAdapter", fake_cls):
            menu = BurnMenu(session, input_fn=lambda p="": next(responses), show=lines.append, check_tty=False)
            rc = menu.run()
        output = "\n".join(lines)
        self.assertEqual(rc, 0, output)
        self.assertEqual(session.plan_path, str(out))
        self.assertEqual(session.bdf, "0000:05:00.0")
        self.assertFalse(session.execute_live)
        self.assertIn("Found 4 ga100-ff-tools plan(s)", output)
        self.assertIn("Imported & loaded", output)
        self.assertEqual(len(Plan.load(out).records), 5)  # 1 noop + 4 pcie


if __name__ == "__main__":
    unittest.main()
