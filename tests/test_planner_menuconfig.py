"""Tests for the native GA100 unlock planner, kernel menuconfig UI, and CLI/menu integration.

All fixtures are 100% synthetic (no real card die IDs or dumps).
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from vfi_burn.backend import write_private_json
from vfi_burn.cli import build_parser, main
from vfi_burn.commands import CommandIO, TargetSpec, execute_configure
from vfi_burn.engine import Overrides, check_plan, process_records
from vfi_burn.ffrecord import (
    FF_ROW_MIN,
    decode_all_fields,
    decode_ff,
    encode_ff,
    replay_ff_rows,
)
from vfi_burn.fuses import (
    DIE_ID_ROWS,
    FFCALC_KEY_TO_FIELD,
    FIELDS_BY_KEY,
    FUSES_BY_KEY,
    REG_CRC_STATUS,
    REG_DSOV_STATUS,
    REG_EFP_STATUS,
    REG_FUSECTRL,
    REG_PLM_F8,
    REG_PLM_FC,
    REG_RESENSE_DISABLE,
    REG_SBFE_SHADOW,
)
from vfi_burn.ga100 import read_ga100_card_state
from vfi_burn.interfaces import Health, Host
from vfi_burn.journal import Journal
from vfi_burn.menu import BurnMenu, MenuSession
from vfi_burn.menuconfig import run_menuconfig
from vfi_burn.model import BurnError, FF_ROW_MAX, Plan, SKU_CMP170HX_8GB
from vfi_burn.planner import (
    CardSnapshot,
    UnlockConfig,
    build_plan_from_config,
    capture_card_snapshot,
    derive_ff_records,
    snapshot_from_raw_state,
)
from vfi_burn.sim_pcie import SimulatedGA100Adapter


def _pack_fields_into_blocks(field_values: dict[str, int], extra_blocks: dict[tuple[int, int], int] | None = None) -> dict[int, dict[int, int]]:
    """Build synthetic chain->block->16bit data from named fields plus optional raw block seeds."""
    blocks: dict[int, dict[int, int]] = {}
    if extra_blocks:
        for (c, b), val in extra_blocks.items():
            blocks.setdefault(c, {})[b] = val & 0xFFFF
    for key, target_val in field_values.items():
        f = FIELDS_BY_KEY[key]
        for offset in range(f.width):
            bit_pos = f.bit + offset
            blk, local = divmod(bit_pos, 16)
            cur = blocks.setdefault(f.chain, {}).get(blk, 0)
            if (target_val >> offset) & 1:
                cur |= 1 << local
            else:
                cur &= ~(1 << local)
            blocks[f.chain][blk] = cur & 0xFFFF
    return blocks


def _synthetic_card_state(
    *,
    gpc_def: int = 0x80,          # GPC7 genuinely defective
    tpc_defs: dict[int, int] | None = None,  # default: GPC1 has TPC0,TPC1 defective (0x03)
    fbp_def: int = 0xC00,         # FBP10, FBP11 genuinely defective
    nvlink_def: int = 0x0,
    gpc_cur: int = 0xE0,          # GPC5, GPC6 factory-disabled (healthy); GPC7 defective
    tpc_curs: dict[int, int] | None = None,
    fbp_cur: int = 0xF00,         # FBP8, FBP9 factory-disabled (healthy); FBP10,11 defective
    nvlink_cur: int = 0x7,        # NVLink factory-disabled
    pcie_restricted: bool = True,
    include_b39: bool = True,
    include_gen4_blocks: bool = True,
) -> dict:
    """Create a complete, coherent synthetic CMP 170HX card state (rows, defects, shadows)."""
    tpc_def_map = {g: 0xFF if ((gpc_def >> g) & 1) else 0x00 for g in range(8)}
    if tpc_defs:
        tpc_def_map.update(tpc_defs)
    tpc_cur_map = {g: 0xFF if ((gpc_cur >> g) & 1) else tpc_def_map[g] for g in range(8)}
    if tpc_curs:
        tpc_cur_map.update(tpc_curs)

    def _pair_mask(fbp_mask: int) -> int:
        out = 0
        for i in range(12):
            if (fbp_mask >> i) & 1:
                out |= 0x3 << (2 * i)
        return out

    pes_def = 0
    pes_cur = 0
    for g in range(8):
        if (gpc_def >> g) & 1:
            pes_def |= 0x7 << (3 * g)
        if (gpc_cur >> g) & 1:
            pes_cur |= 0x7 << (3 * g)

    field_values = {
        "gpc_disable": gpc_cur,
        "pes_disable": pes_cur,
        "fbp_disable": fbp_cur,
        "fbpa_disable": _pair_mask(fbp_cur),
        "fbio_disable": _pair_mask(fbp_cur),
        "rop_l2_disable": _pair_mask(fbp_cur),
        "nvlink_disable": nvlink_cur,
        "sys_pipe_disable": 0xFE,
    }
    for g in range(8):
        field_values[f"tpc_gpc{g}_disable"] = tpc_cur_map[g]

    # Seed c1/b23 with unmapped bit 371 (local bit 3 = 0x0008) to verify preservation.
    extra_blocks: dict[tuple[int, int], int] = {
        (1, 23): 0x0008,
    }
    if include_b39:
        # c2/b39: preserve pcie_devidb[15] (bit 10 -> 0x0400) + bit 8 (0x0100) + bit 7 (0x0080)
        # Restricted adds bit 1 (0x0002) and bit 9 (0x0200) -> 0x0782; unrestricted -> 0x0580
        extra_blocks[(2, 39)] = 0x0782 if pcie_restricted else 0x0580
        # c2/b49: preserve 0x0024; restricted sets bit 6 (0x0040) and bit 7 (0x0080) -> 0x00E4
        extra_blocks[(2, 49)] = 0x00E4 if pcie_restricted else 0x0024
    if include_gen4_blocks:
        extra_blocks[(2, 46)] = 0xA000 if pcie_restricted else 0x8000
        extra_blocks[(2, 47)] = 0x0059 if pcie_restricted else 0x0000

    blocks = _pack_fields_into_blocks(field_values, extra_blocks=extra_blocks)
    ff_rows = {r: 0 for r in range(FF_ROW_MIN, FF_ROW_MAX + 1)}
    row = FF_ROW_MIN
    for chain in sorted(blocks):
        for blk in sorted(blocks[chain]):
            ff_rows[row] = encode_ff(chain, blk, blocks[chain][blk])
            row += 1

    decoded = decode_all_fields(replay_ff_rows(ff_rows))
    defect_masks = {
        "GPC_DISABLE": gpc_def,
        "PES_DISABLE": pes_def,
        "FBP_DISABLE": fbp_def,
        "FBPA_DISABLE": _pair_mask(fbp_def),
        "FBIO_DISABLE": _pair_mask(fbp_def),
        "ROP_L2_DISABLE": _pair_mask(fbp_def),
        "NVLINK_DISABLE": nvlink_def,
        "SYS_PIPE_DISABLE": 0,
    }
    for g in range(8):
        defect_masks[f"TPC_GPC{g}_DISABLE"] = tpc_def_map[g]

    shadow_masks = {ff_key: decoded[ff_key] for ff_key in FFCALC_KEY_TO_FIELD}
    die_rows = {r: 0x5A000000 + r for r in DIE_ID_ROWS}
    return {
        "pci_id": SKU_CMP170HX_8GB,
        "die_rows": die_rows,
        "ff_rows": ff_rows,
        "defect_masks": defect_masks,
        "shadow_masks": shadow_masks,
        "lnkcap2": 0x06 if pcie_restricted else 0x0E,
    }


def _build_sim_adapter(state: dict, *, goldilocks: bool = False) -> SimulatedGA100Adapter:
    """Construct a SimulatedGA100Adapter populated with all defect + fused shadow BAR0 registers."""
    regs: dict[int, int] = {}
    for ff_key, vfi_name in FFCALC_KEY_TO_FIELD.items():
        fdef = FUSES_BY_KEY[ff_key]
        if fdef.defect_addr is not None:
            regs[fdef.defect_addr] = state["defect_masks"][vfi_name]
        if fdef.fused_addr is not None:
            regs[fdef.fused_addr] = state["shadow_masks"][ff_key]

    if goldilocks:
        health = Health(
            fusectrl=0xE0040000,
            dsov=0,
            crc=0x55,
            plm=0xFFFFFFFF,
            plm_f8=0xFFFFFFFF,
            en_sw=0,
            debugctrl=0,
            efp=0,
            dfp=0,
            plm_f4=0,
            sbfe=0,
            wdata_wr_secure=1,
            resense_disable=0,
        )
    else:
        health = Health(
            fusectrl=0xE0040000,
            dsov=1,
            crc=0x55,
            plm=0,
            plm_f8=0,
            en_sw=0,
            debugctrl=0,
            efp=1,
            dfp=0,
            plm_f4=0,
            sbfe=1,
            wdata_wr_secure=1,
            resense_disable=1,
        )
    return SimulatedGA100Adapter(
        pci_id=state["pci_id"],
        die=state["die_rows"],
        rows=state["ff_rows"],
        fields={"LnkCap2": state["lnkcap2"]},
        health=health,
        host=Host("pci-stub", False, False, "[none]", True, "sim-boot-1"),
        registers=regs,
        lnkcap2=state["lnkcap2"],
    )


class TestCardStateReadout(unittest.TestCase):
    def test_capture_cold_card_snapshot_succeeds(self):
        state = _synthetic_card_state(tpc_defs={1: 0x03})
        adapter = _build_sim_adapter(state, goldilocks=False)
        card = capture_card_snapshot(adapter, label="sim-cold")
        self.assertEqual(card.pci_id, SKU_CMP170HX_8GB)
        # Current: GPC0..4 enabled (5 GPCs), GPC1 has 2 TPCs defective -> (5*8 - 2)*2 = 76 SMs
        self.assertEqual(card.current_sm_count(), 76)
        # Healthy max: GPC0..6 healthy (7 GPCs), GPC1 has 2 TPCs defective -> (7*8 - 2)*2 = 108 SMs
        self.assertEqual(card.max_healthy_sm_count(), 108)
        self.assertEqual(card.current_fbp_count(), 8)
        self.assertEqual(card.max_healthy_fbp_count(), 10)
        self.assertEqual(card.current_pcie_label(), "Gen1/Gen2 (restricted)")
        self.assertFalse(card.nvlink_currently_enabled())

    def test_refuses_unsensed_goldilocks_card_with_efp_zero(self):
        state = _synthetic_card_state()
        adapter = _build_sim_adapter(state, goldilocks=True)
        with self.assertRaisesRegex(BurnError, r"UNSENSED.*EFP=0"):
            capture_card_snapshot(adapter, label="sim-goldilocks-unsensed")

    def test_refuses_shadow_disagreeing_with_ff_replay(self):
        state = _synthetic_card_state()
        state["shadow_masks"]["fbp_disable"] ^= 0x1
        with self.assertRaisesRegex(BurnError, "disagrees with FF replay"):
            snapshot_from_raw_state(**state)

    def test_refuses_wrong_pci_id_and_bad_crc(self):
        state = _synthetic_card_state()
        adapter = _build_sim_adapter(state, goldilocks=False)
        adapter.pcie.regs[REG_CRC_STATUS] = 0x00
        with self.assertRaisesRegex(BurnError, "CRC_STATUS"):
            read_ga100_card_state(adapter, label="bad-crc")

        adapter2 = _build_sim_adapter({**state, "pci_id": "10de:20b0"}, goldilocks=False)
        with self.assertRaisesRegex(BurnError, "unexpected PCI ID"):
            read_ga100_card_state(adapter2, label="wrong-pci")

    def test_iff_record_caps_ff_last_row_below_486_and_diagnoses_iff_shadow_override(self):
        from vfi_burn.fuses import REG_IFF_RECORD
        from vfi_burn.model import SKU_CMP170HX_10GB

        # Simulate a 10GB card whose persistent IFF table starts at row 468 (IFF_RECORD=0x01fb01d4),
        # with non-zero IFF data occupying rows 468..486.
        state = _synthetic_card_state(tpc_defs={1: 0x03})
        state["pci_id"] = SKU_CMP170HX_10GB
        for r in range(468, 487):
            state["ff_rows"][r] = 0x0820043D  # non-zero IFF word (bit 5 set)

        adapter = _build_sim_adapter(state, goldilocks=False)
        adapter.pcie.regs[REG_IFF_RECORD] = 0x01FB01D4  # low16=468, hi16=507

        card = capture_card_snapshot(adapter, label="10gb-iff468")
        self.assertEqual(card.ff_last_row, 467)
        self.assertEqual(card.iff_record, 0x01FB01D4)
        self.assertEqual(card.free_row_count, 467 - card.next_free_row + 1)

        cfg = UnlockConfig.from_card(card)
        cfg.set_nvlink(True)
        plan_dict = build_plan_from_config(card, cfg)
        self.assertEqual(plan_dict["fpf"]["last_row"], 467)
        self.assertEqual(plan_dict["card"]["pci_id"], SKU_CMP170HX_10GB)

        # If persistent IFF overrides a shadow register so it disagrees with FF replay, report IFF clearly
        adapter.pcie.regs[FUSES_BY_KEY["gpc_disable"].fused_addr] ^= 0x1
        with self.assertRaisesRegex(BurnError, r"persistent IFF records.*0x01fb01d4.*rows 468\.\.507"):
            capture_card_snapshot(adapter, label="10gb-iff-override")

        # Genuine gap inside 222..467 still refuses and lists exact non-zero runs
        state_gap = _synthetic_card_state()
        state_gap["ff_rows"][350] = encode_ff(1, 23, 0x0FE8)
        with self.assertRaisesRegex(BurnError, r"non-contiguous.*non-zero runs:.*350"):
            snapshot_from_raw_state(**state_gap)


class TestUnlockAndLockPlanner(unittest.TestCase):
    def setUp(self):
        self.state = _synthetic_card_state(tpc_defs={1: 0x03})
        self.card = snapshot_from_raw_state(**self.state)

    def test_unlock_all_healthy_sm_hbm_pcie_gen3_nvlink(self):
        cfg = UnlockConfig.from_card(self.card)
        cfg.unlock_all_healthy_sm_hbm()
        cfg.set_pcie_mode("gen3")
        cfg.set_nvlink(True)

        items, issues = derive_ff_records(self.card, cfg)
        self.assertGreater(len(items), 0)
        self.assertEqual([i for i in issues if i.severity == "error"], [])

        # Verify Gen3 touched c2/b39 & c2/b49, preserved pcie_devidb[15], and did NOT touch c2/b46-47
        touched = {(it["chain"], it["block"]): it["new"] for it in items}
        self.assertEqual(touched[(2, 39)], 0x0580)
        self.assertEqual(touched[(2, 49)], 0x0024)
        self.assertNotIn((2, 46), touched)
        self.assertNotIn((2, 47), touched)

        # Verify NVLink cleared bits 0..2 while preserving sys_pipe_disable (0xFE) & bit 371 (0x0008)
        self.assertEqual(touched[(1, 23)] & 0x000F, 0x0008)
        self.assertEqual((touched[(1, 23)] >> 4) & 0xFF, 0xFE)

        plan_dict = build_plan_from_config(self.card, cfg)
        self.assertEqual(plan_dict["ceiling"]["base_sm"], 108)
        self.assertEqual(plan_dict["records"][0]["kind"], "test")
        self.assertEqual(plan_dict["final_expected_cold"]["LnkCap2"], "0xe")

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "plan.json"
            Path(tmp).chmod(0o700)
            write_private_json(p, plan_dict)
            loaded = Plan.load(p)
            # Gen3 + SM+HBM + NVLink requires NO experimental override!
            final_masks = check_plan(loaded, allow_unmapped_ff=False)
            self.assertEqual(final_masks["GPC_DISABLE"], 0x80)
            self.assertEqual(final_masks["TPC_GPC1_DISABLE"], 0x03)
            self.assertEqual(final_masks["FBP_DISABLE"], 0xC00)
            self.assertEqual(final_masks["NVLINK_DISABLE"], 0x0)

    def test_pcie_gen4_clears_magic_d_blocks_without_override(self):
        cfg = UnlockConfig.from_card(self.card)
        cfg.set_pcie_mode("gen4")
        items, _ = derive_ff_records(self.card, cfg)
        touched = {(it["chain"], it["block"]): it["new"] for it in items}
        self.assertEqual(touched[(2, 39)], 0x0580)
        self.assertEqual(touched[(2, 49)], 0x0024)
        self.assertEqual(touched[(2, 46)], 0x8000)
        self.assertEqual(touched[(2, 47)], 0x0000)

        plan_dict = build_plan_from_config(self.card, cfg)
        self.assertEqual(plan_dict["final_expected_cold"]["LnkCap2"], "0x1e")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "gen4.json"
            Path(tmp).chmod(0o700)
            write_private_json(p, plan_dict)
            loaded = Plan.load(p)
            # Promoted Gen4 MAGIC_BITS_D bits (19/22/25/26/28) pass strict check_plan with NO override!
            check_plan(loaded, allow_unmapped_ff=False)

    def test_lock_active_gpc_tpc_fbp_pcie_and_nvlink(self):
        # Start from an already-unlocked Gen3 + NVLink card and lock individual items!
        state = _synthetic_card_state(
            gpc_cur=0x80,
            tpc_defs={1: 0x03},
            fbp_cur=0xC00,
            nvlink_cur=0x0,
            pcie_restricted=False,
        )
        card = snapshot_from_raw_state(**state)
        cfg = UnlockConfig.from_card(card)

        # Lock GPC0 entirely -> masks GPC0, all 8 TPCs of GPC0, and all 3 PES slices of GPC0
        cfg.set_gpc(0, False)
        # Lock a single TPC (GPC2/TPC5)
        cfg.set_tpc(2, 5, False)
        # Lock FBP3 -> masks FBP3 and both paired FBPA/FBIO/ROP_L2 slices
        cfg.set_fbp(3, False)
        # Lock PCIe back to Gen1/Gen2 and lock NVLink OFF
        cfg.set_pcie_mode("lock_gen12")
        cfg.set_nvlink(False)

        masks = cfg.desired_floorsweep_masks()
        self.assertTrue((masks["gpc_disable"] >> 0) & 1)
        self.assertEqual(masks["tpc_gpc0_disable"], 0xFF)
        self.assertEqual(masks["pes_disable"] & 0x7, 0x7)
        self.assertEqual(masks["tpc_gpc2_disable"], 1 << 5)
        self.assertTrue((masks["fbp_disable"] >> 3) & 1)
        self.assertEqual((masks["fbpa_disable"] >> 6) & 0x3, 0x3)
        self.assertEqual((masks["fbio_disable"] >> 6) & 0x3, 0x3)
        self.assertEqual((masks["rop_l2_disable"] >> 6) & 0x3, 0x3)
        self.assertEqual(masks["nvlink_disable"], 0x7)

        items, issues = derive_ff_records(card, cfg)
        self.assertEqual([i for i in issues if i.severity == "error"], [])
        touched = {(it["chain"], it["block"]): it["new"] for it in items}
        self.assertEqual(touched[(2, 39)], 0x0782)
        self.assertEqual(touched[(2, 49)], 0x00E4)
        self.assertEqual(touched[(1, 23)] & 0x7, 0x7)

    def test_tpc_coherence_and_defect_refusals(self):
        cfg = UnlockConfig.from_card(self.card)
        # Cannot enable defective GPC7, defective GPC1/TPC0, or defective FBP10
        with self.assertRaisesRegex(BurnError, "defective"):
            cfg.set_gpc(7, True)
        with self.assertRaisesRegex(BurnError, "defective"):
            cfg.set_tpc(1, 0, True)
        with self.assertRaisesRegex(BurnError, "defective"):
            cfg.set_fbp(10, True)

        # Enabling one TPC on disabled healthy GPC5 automatically enables GPC5!
        self.assertFalse(cfg.gpc_enabled[5])
        cfg.set_tpc(5, 3, True)
        self.assertTrue(cfg.gpc_enabled[5])
        self.assertEqual(sum(cfg.tpc_enabled[5]), 1)
        # Disabling that last TPC automatically disables GPC5 again!
        cfg.set_tpc(5, 3, False)
        self.assertFalse(cfg.gpc_enabled[5])

    def test_nvlink_all_defective_refused(self):
        state = _synthetic_card_state(nvlink_def=0x7, nvlink_cur=0x7)
        card = snapshot_from_raw_state(**state)
        cfg = UnlockConfig.from_card(card)
        with self.assertRaisesRegex(BurnError, "defective"):
            cfg.set_nvlink(True)

    def test_missing_pcie_block_hard_denied(self):
        state = _synthetic_card_state(include_b39=False)
        card = snapshot_from_raw_state(**state)
        cfg = UnlockConfig.from_card(card)
        cfg.set_pcie_mode("gen3")
        with self.assertRaisesRegex(BurnError, "absent from this card's FF prefix"):
            derive_ff_records(card, cfg)

    def test_absent_unchanged_trailing_block_c1_b17_allowed_but_modifying_it_refused(self):
        # Card where GPC2 is defective (0x04), GPC5/6 are factory-disabled (gpc_cur=0x64),
        # and GPC7 is ALREADY enabled (so c1/b17 holding GPC7 PES bits 272..274 is absent from FF!).
        state = _synthetic_card_state(gpc_def=0x04, gpc_cur=0x64, tpc_defs={5: 0x02})
        # Remove c1/b17 from ff_rows and repack contiguously from row 222.
        raw_nonzero = [
            w for r, w in sorted(state["ff_rows"].items())
            if w != 0 and (decode_ff(w).chain, decode_ff(w).block) != (1, 17)
        ]
        ff_rows = {r: 0 for r in range(FF_ROW_MIN, FF_ROW_MAX + 1)}
        for idx, w in enumerate(raw_nonzero):
            ff_rows[FF_ROW_MIN + idx] = w
        state["ff_rows"] = ff_rows
        card = snapshot_from_raw_state(**state)
        self.assertNotIn(17, card.current_blocks.get(1, {}))

        # Unlocking all healthy compute changes GPC5/6 PES bits in c1/b16 only (GPC7 PES stays 0) -> SUCCEEDS!
        cfg = UnlockConfig.from_card(card)
        cfg.unlock_all_healthy_compute()
        items, issues = derive_ff_records(card, cfg)
        self.assertEqual([i for i in issues if i.severity == "error"], [])
        self.assertNotIn((1, 17), {(it["chain"], it["block"]) for it in items})
        plan_dict = build_plan_from_config(card, cfg)
        self.assertEqual(plan_dict["ceiling"]["base_sm"], 110)

        # Explicitly trying to LOCK GPC7 requires modifying pes_disable bits in absent c1/b17 -> STILL REFUSED!
        cfg_lock7 = UnlockConfig.from_card(card)
        cfg_lock7.set_gpc(7, False)
        with self.assertRaisesRegex(BurnError, r"cannot modify pes_disable.*c1/b17 is absent"):
            derive_ff_records(card, cfg_lock7)

    def test_empty_selection_refused(self):
        cfg = UnlockConfig.from_card(self.card)
        with self.assertRaisesRegex(BurnError, "no fuse changes selected"):
            build_plan_from_config(self.card, cfg)

    def test_plan_load_defaults_ceiling_when_omitted_and_rejects_bad_schema(self):
        cfg = UnlockConfig.from_card(self.card)
        cfg.set_nvlink(True)
        plan_dict = build_plan_from_config(self.card, cfg)
        plan_dict.pop("ceiling")
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp).chmod(0o700)
            p = Path(tmp) / "no_ceiling.json"
            write_private_json(p, plan_dict)
            loaded = Plan.load(p)
            self.assertEqual(loaded.base_sm_ceiling, 128)

            bad = Path(tmp) / "bad_schema.json"
            write_private_json(bad, {"schema_version": 99})
            with self.assertRaisesRegex(BurnError, r"configure.*Setup"):
                Plan.load(bad)


class TestMenuconfigPlainUI(unittest.TestCase):
    def setUp(self):
        self.state = _synthetic_card_state(tpc_defs={1: 0x03})
        self.card = snapshot_from_raw_state(**self.state)

    def test_full_navigation_compute_gpc_memory_pcie_nvlink_preview_save(self):
        # Scripted choices (all entries are 1-indexed):
        # 1 (toggle all healthy SM+HBM ON),
        # 2 (Compute submenu: [1]=All, [2]=Reset, [3]=GPC0..[8]=GPC5..[10]=GPC7)
        #   -> 8 (enter GPC5: [1]=TPC0..[8]=TPC7) -> 1 (disable TPC0 on GPC5) -> b (back to Compute)
        #   -> t10 (try toggling defective GPC7 -> refused!) -> b (back to root)
        # 3 (Memory submenu: [1]=All, [2]=Reset, [3]=FBP0..[14]=FBP11)
        #   -> 3 (lock FBP0 OFF) -> b (back to root)
        # 4 (PCIe submenu: [1]=keep, [2]=gen3, [3]=gen4, [4]=lock) -> 2 (Gen3) -> b (back to root)
        # 5 (toggle NVLink ON)
        # p (Preview) -> "" (continue)
        # s (Save)
        answers = iter([
            "1",
            "2", "8", "1", "b", "t10", "b",
            "3", "3", "b",
            "4", "2", "b",
            "5",
            "p", "",
            "s",
        ])
        shown: list[str] = []
        cfg = run_menuconfig(
            self.card,
            use_curses=False,
            input_fn=lambda _prompt: next(answers),
            show=shown.append,
        )
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertFalse(cfg.tpc_enabled[5][0])
        self.assertFalse(cfg.gpc_enabled[7])  # defective stayed disabled
        self.assertFalse(cfg.fbp_enabled[0])
        self.assertEqual(cfg.pcie_mode, "gen3")
        self.assertTrue(cfg.nvlink_enabled)
        output = "\n".join(shown)
        self.assertIn("defective", output.lower())
        self.assertIn("Preview Exact FF Records", output)

    def test_cancel_returns_none(self):
        cfg = run_menuconfig(
            self.card,
            use_curses=False,
            input_fn=lambda _prompt: "q",
            show=lambda _s: None,
        )
        self.assertIsNone(cfg)


class TestCLIConfigureMenuAndEndToEndSimulate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.work.chmod(0o700)
        self.state = _synthetic_card_state(tpc_defs={1: 0x03})
        self.cold_adapter = _build_sim_adapter(self.state, goldilocks=False)
        self.cold_snap = self.work / "cold_sim.json"
        write_private_json(self.cold_snap, self.cold_adapter.to_snapshot_dict())

    def tearDown(self):
        self.tmp.cleanup()

    def test_external_importers_removed_from_cli(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                parser.parse_args(["import-ffcalc"])
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                parser.parse_args(["import-ga100-ff-tools"])

    def test_cli_configure_noninteractive_and_end_to_end_simulate_burn(self):
        plan_path = self.work / "plan_sm_hbm_gen3_nvl.json"
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            rc = main([
                "configure",
                "--sim-pcie", str(self.cold_snap),
                "--out", str(plan_path),
                "--sm-hbm",
                "--pcie", "gen3",
                "--nvlink",
            ])
        self.assertEqual(rc, 0, err_buf.getvalue())
        self.assertTrue(plan_path.exists())
        self.assertEqual(plan_path.stat().st_mode & 0o777, 0o600)

        # Now burn the generated plan on a fresh Goldilocks SimulatedGA100Adapter!
        goldilocks_adapter = _build_sim_adapter(self.state, goldilocks=True)
        plan = Plan.load(plan_path)
        journal_path = self.work / "burn.jsonl"
        with Journal(journal_path) as j:
            next_row = process_records(
                plan,
                goldilocks_adapter,
                goldilocks_adapter,
                j,
                stage="all",
                confirm=lambda _p, _tok: True,
                show=lambda _msg: None,
            )
        self.assertEqual(next_row, plan.expected_next_row + len(plan.records))
        # Verify final replayed FF rows match every expected final mask
        final_decoded = decode_all_fields(replay_ff_rows(goldilocks_adapter.pcie.rows))
        self.assertEqual(final_decoded["gpc_disable"], 0x80)
        self.assertEqual(final_decoded["tpc_gpc1_disable"], 0x03)
        self.assertEqual(final_decoded["fbp_disable"], 0xC00)
        self.assertEqual(final_decoded["nvlink_disable"], 0x0)
        self.assertEqual(final_decoded["disable_gen3_speed_bit1"], 0)
        self.assertEqual(final_decoded["pcie_boot_gen3_dis"], 0)

    def test_execute_configure_interactive_plain(self):
        plan_path = self.work / "interactive_plan.json"
        answers = iter(["1", "s"])
        shown: list[str] = []
        rc = execute_configure(
            target=TargetSpec(sim_pcie=str(self.cold_snap)),
            out=str(plan_path),
            use_curses=False,
            io=CommandIO(show=shown.append, input_fn=lambda _p: next(answers)),
        )
        self.assertEqual(rc, 0)
        self.assertTrue(plan_path.exists())
        loaded = Plan.load(plan_path)
        self.assertEqual(loaded.base_sm_ceiling, 108)

    def test_burn_menu_setup_a_configure_and_review(self):
        plan_path = self.work / "menu_plan.json"
        # Menu flow:
        #   1 (Setup) -> a (Configure Unlocks) -> [menuconfig: 5 (NVLink ON), s (Save)]
        #   -> output path -> 0 (Back to Main Menu) -> 2 (Review Plan) -> q (Quit)
        answers = iter([
            "1",
            "a",
            "5", "s",
            str(plan_path),
            "2",
            "q",
        ])
        shown: list[str] = []
        menu = BurnMenu(
            MenuSession(fake_path=str(self.cold_snap)),
            input_fn=lambda _p="": next(answers),
            show=shown.append,
        )
        rc = menu.run()
        self.assertEqual(rc, 0)
        self.assertTrue(plan_path.exists())
        self.assertEqual(menu.s.plan_path, str(plan_path))
        output = "\n".join(shown)
        self.assertIn("Saved & loaded", output)
        self.assertIn("NOOP_TEST", output)

    def test_plan_review_flags_sku_and_die_mismatch_without_mixing_live_ff(self):
        from vfi_burn.console import format_plan_review
        from vfi_burn.sim_pcie import sim_pcie_from_file

        # Build an 8 GB (10de:20c2) plan from self.cold_snap
        plan_8gb_path = self.work / "plan_8gb.json"
        card_8gb = capture_card_snapshot(self.cold_adapter)
        cfg = UnlockConfig.from_card(card_8gb)
        cfg.unlock_all_healthy_sm_hbm()
        write_private_json(plan_8gb_path, build_plan_from_config(card_8gb, cfg))
        plan_8gb = Plan.load(plan_8gb_path)

        # Connect a 10 GB (10de:2082) target with different die rows & FF rows -> must NOT mix rows!
        snap_10gb = json.loads(self.cold_snap.read_text(encoding="utf-8"))
        snap_10gb["pci_id"] = "10de:2082"
        snap_10gb["rows"]["8"] = "0x11111111"
        snap_10gb_path = self.work / "sim_10gb.json"
        write_private_json(snap_10gb_path, snap_10gb)
        sim_10gb = sim_pcie_from_file(snap_10gb_path)

        lines = format_plan_review(plan_8gb, reader=sim_10gb)
        txt = "\n".join(lines)
        self.assertIn("TARGET MISMATCH (WRONG CARD CONNECTED — BURN REFUSED)", txt)
        self.assertIn("SKU MISMATCH", txt)
        self.assertIn("CMP 170HX 10 GB", txt)
        self.assertIn("OFFLINE — TARGET MISMATCH", txt)

        # Same SKU (10de:20c2) but DIFFERENT Die ID -> must also refuse live FF overlay!
        snap_other_8gb = json.loads(self.cold_snap.read_text(encoding="utf-8"))
        snap_other_8gb["rows"]["8"] = "0x22222222"
        snap_other_path = self.work / "sim_other_8gb.json"
        write_private_json(snap_other_path, snap_other_8gb)
        sim_other_8gb = sim_pcie_from_file(snap_other_path)

        lines2 = format_plan_review(plan_8gb, reader=sim_other_8gb)
        txt2 = "\n".join(lines2)
        self.assertIn("TARGET MISMATCH (WRONG CARD CONNECTED — BURN REFUSED)", txt2)
        self.assertIn("DIE ID MISMATCH", txt2)
        self.assertIn("OFFLINE — TARGET MISMATCH", txt2)

        # Default save path for a 10 GB card must NOT clobber an active 8 GB plan path
        card_10gb = capture_card_snapshot(sim_10gb)
        menu = BurnMenu(MenuSession(plan_path=str(plan_8gb_path)))
        self.assertEqual(menu._default_plan_path_for_card(card_10gb), "private/plan_2082.json")
        self.assertEqual(menu._default_plan_path_for_card(card_8gb), str(plan_8gb_path))

    def test_a100_devid_codec_validation_menuconfig_and_simulated_burn(self):
        from vfi_burn.console import format_plan_review
        from vfi_burn.engine import Overrides, check_plan, process_records
        from vfi_burn.ffrecord import (
            decode_a100_devid_ff_word,
            encode_a100_devid_ff_words,
            ff_word_safety_audit,
        )
        from vfi_burn.journal import Journal

        # 1. Exact bit encoding for all 4 A100 target IDs (device_id << 1, bit 0 == 0)
        expected_words = {
            0x20F1: (0x41E20942, 0x41E20982),
            0x20B5: (0x416A0942, 0x416A0982),
            0x20B0: (0x41600942, 0x41600982),
            0x20B2: (0x41640942, 0x41640982),
        }
        for devid, (exp_a, exp_b) in expected_words.items():
            wa, wb = encode_a100_devid_ff_words(devid)
            self.assertEqual((wa, wb), (exp_a, exp_b))
            self.assertEqual(decode_a100_devid_ff_word(wa), ("DEVIDA", devid))
            self.assertEqual(decode_a100_devid_ff_word(wb), ("DEVIDB", devid))
            # Default audit (allow_a100_devid=False) MUST still hard-deny c2/b37-38
            hard, _ = ff_word_safety_audit(wa)
            self.assertTrue(hard)
            hard_ok, over_ok = ff_word_safety_audit(wa, allow_a100_devid=True)
            self.assertEqual((hard_ok, over_ok), ([], []))

        # Reject invalid IDs or corrupted bit 0
        self.assertIsNone(decode_a100_devid_ff_word(encode_ff(2, 37, (0x20B5 << 1) | 1)))
        self.assertIsNone(decode_a100_devid_ff_word(encode_ff(2, 37, 0x1234 << 1)))

        # 2. Interactive menuconfig flow: open PCIe Device ID submenu [6], pick 0x20F1 [2],
        #    go back [b], open Preview [8] (must handle old=None FPF cleanly), back [b], save [s]
        card = capture_card_snapshot(self.cold_adapter)
        inputs = iter(["6", "2", "b", "8", "b", "s"])
        shown: list[str] = []
        cfg = run_menuconfig(
            card,
            use_curses=False,
            input_fn=lambda _p="": next(inputs),
            show=shown.append,
        )
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.a100_devid, 0x20F1)
        menu_out = "\n".join(shown)
        self.assertIn("WARNING — A100 PCIe DEVICE ID CONVERSION SELECTED", menu_out)
        self.assertIn("0x41E20942", menu_out)
        self.assertIn("0x41E20982", menu_out)
        self.assertIn("FPF   ->0x41E2", menu_out)

        # 3. Build plan, validate, and verify format_plan_review output
        plan_path = self.work / "plan_a100_20f1.json"
        plan_dict = build_plan_from_config(card, cfg)
        write_private_json(plan_path, plan_dict)
        plan = Plan.load(plan_path)
        check_plan(plan)
        self.assertEqual([r.kind for r in plan.records], ["test", "devid", "devid"])
        self.assertEqual([r.word for r in plan.records[1:]], [0x41E20942, 0x41E20982])

        review_txt = "\n".join(format_plan_review(plan, reader=self.cold_adapter))
        self.assertIn("10de:20f1", review_txt)
        self.assertIn("REQUIRES A100 VBIOS + 128 GB BAR1", review_txt)

        # 4. Split half-identity (DEVIDA without DEVIDB) MUST be refused by check_plan
        half_dict = json.loads(plan_path.read_text(encoding="utf-8"))
        half_dict["records"] = half_dict["records"][:2]
        half_path = self.work / "plan_half_devid.json"
        write_private_json(half_path, half_dict)
        with self.assertRaisesRegex(BurnError, "BOTH DEVIDA"):
            check_plan(Plan.load(half_path))

        # 5. Conflicting DEVIDA != DEVIDB MUST be refused by check_plan
        conflict_dict = json.loads(plan_path.read_text(encoding="utf-8"))
        conflict_dict["records"][2]["word"] = "0x416a0982"  # 0x20B5 vs 0x20F1
        conflict_path = self.work / "plan_conflict_devid.json"
        write_private_json(conflict_path, conflict_dict)
        with self.assertRaisesRegex(BurnError, "conflicting DEVIDA/DEVIDB"):
            check_plan(Plan.load(conflict_path))

        # 6. Raw HAL program_row without allow_a100_devid=True MUST still refuse valid A100 DEVID
        gold_ad = _build_sim_adapter(self.state, goldilocks=True)
        gold_ad.active_sense_readonly()
        with self.assertRaisesRegex(BurnError, "PCIe device-ID block"):
            gold_ad.controller.program_row(card.next_free_row, 0x41E20942)

        # 7. End-to-end simulated burn: declining CONFIRM A100 DEVID aborts before write;
        #    confirming burns test + DEVIDA + DEVIDB cleanly through GA100FuseController HAL!
        journal_path = self.work / "devid_burn.jsonl"
        burn_out: list[str] = []
        seen_tokens: list[str] = []

        def _decline_devid(_prompt: str, token: str) -> bool:
            seen_tokens.append(token)
            return not token.startswith("CONFIRM A100 DEVID")

        with Journal(journal_path) as j:
            with self.assertRaisesRegex(BurnError, "A100 Device ID conversion not confirmed"):
                process_records(
                    plan,
                    gold_ad,
                    gold_ad,
                    j,
                    stage="all",
                    dry_run=False,
                    overrides=Overrides(no_test_receipt=True),
                    confirm=_decline_devid,
                    show=burn_out.append,
                )
        self.assertIn("CONFIRM A100 DEVID 0x20F1", seen_tokens)
        self.assertIn("WILL NOT POST on its stock CMP 170HX VBIOS", "\n".join(burn_out))
        self.assertIn("128 GB 64-BIT BAR1 MMIO WINDOW", "\n".join(burn_out))

        # Now confirm all prompts and verify both DEVIDA & DEVIDB land in OTP!
        gold_ad2 = _build_sim_adapter(self.state, goldilocks=True)
        journal_path2 = self.work / "devid_burn_ok.jsonl"
        tokens_ok: list[str] = []

        def _accept_all(_prompt: str, token: str) -> bool:
            tokens_ok.append(token)
            return True

        with Journal(journal_path2) as j2:
            process_records(
                plan,
                gold_ad2,
                gold_ad2,
                j2,
                stage="all",
                dry_run=False,
                overrides=Overrides(no_test_receipt=True),
                confirm=_accept_all,
                show=lambda _s: None,
            )
        self.assertEqual(tokens_ok.count("CONFIRM A100 DEVID 0x20F1"), 2)
        self.assertEqual(gold_ad2.fuse_row(card.next_free_row + 1), 0x41E20942)
        self.assertEqual(gold_ad2.fuse_row(card.next_free_row + 2), 0x41E20982)

    def test_mig_sys_pipe_unlock_preserves_defects_and_nvlink(self):
        # Build a synthetic card with SYS_PIPE_DISABLE=0xFE (7 disabled), DEFECTIVE=0x80 (only bit 7 defective)
        st = _synthetic_card_state()
        # Update c1/b23: bits 0..2 = nvlink_disable (0x7), bit 3 = unmapped 1 (0x8), bits 4..11 = sys_pipe_disable (0xFE)
        # Keep top 4 bits (bits 12..15) = 0 -> b23 = 0x0FEF
        for r, w in list(st["ff_rows"].items()):
            if w != 0:
                ff = decode_ff(w)
                if (ff.chain, ff.block) == (1, 23):
                    st["ff_rows"][r] = encode_ff(1, 23, 0x0FEF)
        st["defect_masks"]["SYS_PIPE_DISABLE"] = 0x80
        st["shadow_masks"]["sys_pipe_disable"] = 0xFE

        ad = _build_sim_adapter(st, goldilocks=False)
        card = capture_card_snapshot(ad)
        self.assertEqual(card.current_sys_pipe_count(), 1)
        self.assertEqual(card.max_healthy_sys_pipe_count(), 7)
        self.assertEqual(card.sys_pipe_unlockable_mask(), 0x7E)

        cfg = UnlockConfig.from_card(card)
        cfg.set_sys_pipe_unlock(True)
        items, issues = derive_ff_records(card, cfg)
        self.assertEqual(len(items), 1)
        self.assertEqual(issues, [])
        # Only non-defective SYS_PIPE bits cleared: 0x0FEF -> 0x080F (bit 11 = bit7 defect preserved, low 0x0F preserved!)
        self.assertEqual(items[0]["chain"], 1)
        self.assertEqual(items[0]["block"], 23)
        self.assertEqual(items[0]["old"], 0x0FEF)
        self.assertEqual(items[0]["new"], 0x080F)

        plan_path = self.work / "plan_mig_sys_pipe.json"
        write_private_json(plan_path, build_plan_from_config(card, cfg))
        plan = Plan.load(plan_path)
        self.assertEqual(plan.records[1].to_disable, 0x80)
        self.assertEqual(plan.records[1].from_disable, 0xFE)


class TestPCIGPUDiscoveryAndPicker(unittest.TestCase):
    def setUp(self):
        import struct
        from vfi_burn.fuses import OPTION_REGS
        from vfi_burn.ga100 import BAR0_MAP_SIZE

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sysfs = self.root / "pci_devices"
        self.sysfs.mkdir()
        self.drivers = self.root / "drivers"
        self.drivers.mkdir()
        self.proc_modules = self.root / "proc_modules"
        self.proc_modules.write_text("pci_stub 16384 1 - Live 0x0\n")
        self.pci_ids = self.root / "pci.ids"
        self.pci_ids.write_text(
            "1a03  ASPEED Technology, Inc.\n"
            "\t2000  ASPEED Graphics Family\n"
        )

        def _make_dev(
            bdf: str,
            vendor: str,
            device: str,
            *,
            cls: str = "0x030200",
            sub_v: str = "0x10de",
            sub_d: str = "0x1532",
            driver: str | None = "pci-stub",
            bar0_size: int = 16 * 1024 * 1024,
            bar1_size: int = 16 * 1024 * 1024 * 1024,
            cur_spd: str = "2.5 GT/s PCIe",
            cur_wid: str = "4",
            max_spd: str = "5.0 GT/s PCIe",
            max_wid: str = "4",
            bar0_regs: dict[int, int] | None = None,
            mem_decode: bool = True,
        ) -> Path:
            d = self.sysfs / bdf
            d.mkdir()
            (d / "vendor").write_text(vendor + "\n")
            (d / "device").write_text(device + "\n")
            (d / "class").write_text(cls + "\n")
            (d / "subsystem_vendor").write_text(sub_v + "\n")
            (d / "subsystem_device").write_text(sub_d + "\n")
            b0_start = 0xF6000000
            b0_end = b0_start + bar0_size - 1
            b1_start = 0x1000000000
            b1_end = b1_start + bar1_size - 1 if bar1_size > 0 else 0
            (d / "resource").write_text(
                f"0x{b0_start:016x} 0x{b0_end:016x} 0x0000000000040200\n"
                f"0x{b1_start if bar1_size else 0:016x} 0x{b1_end:016x} 0x000000000014220c\n"
            )
            (d / "current_link_speed").write_text(cur_spd + "\n")
            (d / "current_link_width").write_text(cur_wid + "\n")
            (d / "max_link_speed").write_text(max_spd + "\n")
            (d / "max_link_width").write_text(max_wid + "\n")
            if driver:
                drv_dir = self.drivers / driver
                drv_dir.mkdir(exist_ok=True)
                (d / "driver").symlink_to(drv_dir)
            cfg = bytearray(256)
            struct.pack_into("<HHH", cfg, 0, int(vendor, 16), int(device, 16), 0x0002 if mem_decode else 0x0000)
            (d / "config").write_bytes(bytes(cfg))
            if bar0_regs is not None:
                res0 = d / "resource0"
                with res0.open("wb") as f:
                    f.truncate(BAR0_MAP_SIZE)
                    for off, val in bar0_regs.items():
                        f.seek(off)
                        f.write(struct.pack("<I", val & 0xFFFFFFFF))
            return d

        # 1) Cold CMP 170HX at 0000:05:00.0 (76/108 SMs, 8/10 FBPs, 16 GiB BAR1)
        cold_regs = {
            REG_FUSECTRL: 0xE0040000,
            REG_DSOV_STATUS: 1,
            REG_EFP_STATUS: 1,
            REG_CRC_STATUS: 0x55,
            OPTION_REGS["GPC_DISABLE"]: 0xE0,
            OPTION_REGS["FBP_DISABLE"]: 0xF00,
            FUSES_BY_KEY["gpc_disable"].defect_addr: 0x80,
            FUSES_BY_KEY["fbp_disable"].defect_addr: 0xC00,
        }
        for g in range(8):
            cold_regs[OPTION_REGS[f"TPC_GPC{g}_DISABLE"]] = 0xFF if g >= 5 else (0x03 if g == 1 else 0x00)
            cold_regs[FUSES_BY_KEY[f"tpc_gpc{g}_disable"].defect_addr] = 0xFF if g == 7 else (0x03 if g == 1 else 0x00)
        _make_dev("0000:05:00.0", "0x10de", "0x20c2", bar0_regs=cold_regs)

        # 2) Unsensed Goldilocks CMP 170HX at 0000:06:00.0 (DSOV=0, EFP=0 -> must NOT report bogus SMs)
        gold_regs = {
            REG_FUSECTRL: 0xE0040000,
            REG_DSOV_STATUS: 0,
            REG_EFP_STATUS: 0,
            REG_CRC_STATUS: 0x55,
        }
        _make_dev("0000:06:00.0", "0x10de", "0x20c2", bar0_regs=gold_regs, cur_spd="8.0 GT/s PCIe", max_spd="8.0 GT/s PCIe")

        # 3) Non-CMP170HX RTX 3090 at 0000:01:00.0 (32 GiB BAR1, nvidia driver)
        _make_dev(
            "0000:01:00.0",
            "0x10de",
            "0x2204",
            cls="0x030000",
            sub_d="0x1467",
            driver="nvidia",
            bar1_size=32 * 1024 * 1024 * 1024,
            cur_spd="16.0 GT/s PCIe",
            cur_wid="16",
            max_spd="16.0 GT/s PCIe",
            max_wid="16",
        )

        # 4) BMC VGA resolved via pci.ids
        _make_dev(
            "0000:03:00.0",
            "0x1a03",
            "0x2000",
            cls="0x030000",
            sub_v="0x1a03",
            sub_d="0x2000",
            driver="ast",
            bar1_size=16 * 1024 * 1024,
            cur_spd="2.5 GT/s PCIe",
            cur_wid="1",
            max_spd="2.5 GT/s PCIe",
            max_wid="1",
        )

        # 5) Non-GPU Ethernet controller (must be ignored)
        _make_dev("0000:04:00.0", "0x8086", "0x1533", cls="0x020000", driver="igb")

    def tearDown(self):
        self.tmp.cleanup()

    def test_discover_pci_gpus_sorting_bar1_and_readonly_bar0_probe(self):
        from vfi_burn.ga100 import discover_pci_gpus, format_gpu_entry

        gpus = discover_pci_gpus(
            sysfs_root=self.sysfs,
            probe_bar0=True,
            pci_ids_paths=[self.pci_ids],
            proc_modules_path=self.proc_modules,
        )
        # 4 GPUs (Ethernet ignored), sorted: CMP 170HX first, then other NVIDIA, then ASPEED
        self.assertEqual([g.bdf for g in gpus], ["0000:05:00.0", "0000:06:00.0", "0000:01:00.0", "0000:03:00.0"])

        g_cold = gpus[0]
        self.assertTrue(g_cold.is_cmp170hx)
        self.assertEqual(g_cold.bar1_bytes, 16 * 1024**3)
        self.assertEqual((g_cold.cur_sm, g_cold.max_sm), (76, 108))
        self.assertEqual((g_cold.cur_fbp, g_cold.max_fbp), (8, 10))
        self.assertIn("COLD", g_cold.fuse_state or "")
        lines = "\n".join(format_gpu_entry(g_cold, index=1))
        self.assertIn("10de:20c2", lines)
        self.assertIn("BAR1=16 GiB", lines)
        self.assertIn("HBM/FBPs=8/10", lines)
        self.assertIn("SMs=76/108", lines)
        self.assertIn("drv=pci-stub", lines)

        # Unsensed Goldilocks card must NOT report SMs/FBPs from zeroed shadows
        g_gold = gpus[1]
        self.assertIn("GOLDILOCKS (unsensed, EFP=0)", g_gold.fuse_state or "")
        self.assertIsNone(g_gold.cur_sm)
        self.assertIsNone(g_gold.cur_fbp)

        # RTX 3090 and ASPEED BMC
        self.assertFalse(gpus[2].is_cmp170hx)
        self.assertEqual(gpus[2].bar1_bytes, 32 * 1024**3)
        self.assertIn("ASPEED Graphics Family", gpus[3].name)

    def test_prompt_select_gpu_bdf_rejects_non_cmp170hx_and_accepts_pick_or_manual(self):
        from vfi_burn.ga100 import prompt_select_gpu_bdf

        # Try picking #3 (RTX 3090 -> refused!), then pick #1 (CMP 170HX -> accepted!)
        answers = iter(["3", "1"])
        shown: list[str] = []
        picked = prompt_select_gpu_bdf(
            input_fn=lambda _p: next(answers),
            show=shown.append,
            sysfs_root=self.sysfs,
            probe_bar0=False,
        )
        self.assertEqual(picked, "0000:05:00.0")
        out = "\n".join(shown)
        self.assertIn("REFUSED", out)
        self.assertIn("BAR1=16 GiB", out)
        self.assertIn("BAR1=32 GiB", out)

        # Manual short BDF entry normalizes cleanly
        picked2 = prompt_select_gpu_bdf(
            input_fn=lambda _p: "06:00.0",
            show=lambda _s: None,
            sysfs_root=self.sysfs,
            probe_bar0=False,
        )
        self.assertEqual(picked2, "0000:06:00.0")

    def test_menu_setup_c_and_cli_configure_use_gpu_picker(self):
        # Menu [1] Setup -> [c] Select LIVE GPU -> pick "1" -> Back -> Quit
        answers = iter(["1", "c", "1", "0", "q"])
        shown: list[str] = []
        menu = BurnMenu(
            MenuSession(),
            input_fn=lambda _p="": next(answers),
            show=shown.append,
            sysfs_root=self.sysfs,
        )
        rc = menu.run()
        self.assertEqual(rc, 0)
        self.assertEqual(menu.s.bdf, "0000:05:00.0")
        out = "\n".join(shown)
        self.assertIn("Detected PCI GPUs:", out)
        self.assertIn("10de:20c2", out)
        self.assertIn("BAR1=16 GiB", out)

        # CLI execute_configure with empty TargetSpec opens picker; blank cancels cleanly
        shown2: list[str] = []
        rc2 = execute_configure(
            target=TargetSpec(),
            out=str(self.root / "unused.json"),
            io=CommandIO(show=shown2.append, input_fn=lambda _p: ""),
            sysfs_root=str(self.sysfs),
        )
        self.assertEqual(rc2, 1)
        self.assertIn("Detected PCI GPUs:", "\n".join(shown2))

    def test_10gb_2082_supported_end_to_end_and_cross_sku_refused(self):
        from vfi_burn.ga100 import DiscoveredGPU, format_gpu_entry
        from vfi_burn.model import SKU_CMP170HX_10GB

        # 1) Formatted GPU entry shows 10GB CMP 170HX as supported (no [not CMP 170HX] tag) + LnkCap wording
        g10 = DiscoveredGPU(
            bdf="0000:03:00.0",
            pci_id=SKU_CMP170HX_10GB,
            subsystem_id="10de:1557",
            pci_class="0x030200",
            name="NVIDIA CMP 170HX 10GB (GA100)",
            driver="pci-stub",
            bar0_bytes=16 * 1024**2,
            bar1_bytes=32 * 1024**3,
            current_link_speed="16.0 GT/s PCIe",
            current_link_width="4",
            max_link_speed="2.5 GT/s PCIe",
            max_link_width="4",
            is_cmp170hx=True,
        )
        text = "\n".join(format_gpu_entry(g10, index=2))
        self.assertIn("CMP 170HX 10GB (GA100)", text)
        self.assertNotIn("[not CMP 170HX]", text)
        self.assertIn("PCIe x4 @ 16.0 GT/s (LnkCap 2.5 GT/s)", text)

        # 2) Build a 10GB (10de:2082) plan from card snapshot, load, verify, and simulate burn
        state_10gb = _synthetic_card_state(tpc_defs={1: 0x03})
        state_10gb["pci_id"] = SKU_CMP170HX_10GB
        cold_10gb = _build_sim_adapter(state_10gb, goldilocks=False)
        card_10gb = capture_card_snapshot(cold_10gb, label="10gb-cold")
        self.assertEqual(card_10gb.pci_id, SKU_CMP170HX_10GB)

        cfg = UnlockConfig.from_card(card_10gb)
        cfg.unlock_all_healthy_sm_hbm()
        cfg.set_pcie_mode("gen4")
        plan_dict = build_plan_from_config(card_10gb, cfg)
        self.assertEqual(plan_dict["card"]["pci_id"], SKU_CMP170HX_10GB)

        plan_path = self.root / "plan_10gb.json"
        self.root.chmod(0o700)
        write_private_json(plan_path, plan_dict)
        loaded = Plan.load(plan_path)
        self.assertEqual(loaded.pci_id, SKU_CMP170HX_10GB)
        check_plan(loaded, allow_unmapped_ff=False)

        # Burns cleanly on matching 10de:2082 card
        gold_10gb = _build_sim_adapter(state_10gb, goldilocks=True)
        with Journal(self.root / "j10.jsonl") as j10:
            next_row = process_records(
                loaded,
                gold_10gb,
                gold_10gb,
                j10,
                stage="all",
                confirm=lambda _p, _tok: True,
                show=lambda _s: None,
            )
        self.assertEqual(next_row, loaded.expected_next_row + len(loaded.records))

        # Cross-SKU safety: 10de:2082 plan MUST be refused on a 10de:20c2 card!
        state_8gb = _synthetic_card_state(tpc_defs={1: 0x03})
        gold_8gb = _build_sim_adapter(state_8gb, goldilocks=True)
        with Journal(self.root / "j_cross.jsonl") as jc:
            with self.assertRaisesRegex(BurnError, "PCI identity mismatch"):
                process_records(
                    loaded,
                    gold_8gb,
                    gold_8gb,
                    jc,
                    stage="all",
                    confirm=lambda _p, _tok: True,
                    show=lambda _s: None,
                )

    def test_unload_gpu_drivers_order_pcistub_bind_enable_and_busy_hint(self):
        import struct
        from vfi_burn.ga100 import ensure_gpu_detached_interactive, unload_gpu_drivers

        bus_root = self.root / "pci_bus"
        (bus_root / "drivers" / "pci-stub").mkdir(parents=True)
        drivers_probe = bus_root / "drivers_probe"
        drivers_probe.write_text("")

        # Prepare a synthetic CMP 170HX bound to nvidia with COMMAND bit 1 OFF (simulating post-unload)
        dev = self.sysfs / "0000:05:00.0"
        (dev / "driver").unlink()
        nv_drv = self.drivers / "nvidia"
        nv_drv.mkdir(exist_ok=True)
        (nv_drv / "unbind").write_text("")
        (dev / "driver").symlink_to(nv_drv)
        (dev / "driver_override").write_text("")
        (dev / "enable").write_text("0\n")
        cfg = bytearray(256)
        struct.pack_into("<HHH", cfg, 0, 0x10DE, 0x20C2, 0x0000)  # memory decode OFF
        (dev / "config").write_bytes(bytes(cfg))

        mods = self.root / "proc_mods_nv"
        mods.write_text(
            "nvidia_uvm 1234 0 - Live 0x0\n"
            "nvidia_drm 1234 0 - Live 0x0\n"
            "nvidia_modeset 1234 1 nvidia_drm, Live 0x0\n"
            "nvidia 1234 2 nvidia_modeset,nvidia_uvm, Live 0x0\n"
        )

        # 1) Declining interactive prompt leaves modules untouched and runs zero commands
        cmds_declined: list[list[str]] = []
        ok = ensure_gpu_detached_interactive(
            "0000:05:00.0",
            input_fn=lambda _p: "n",
            show=lambda _s: None,
            sysfs_root=self.sysfs,
            pci_bus_root=bus_root,
            proc_modules_path=mods,
            runner=lambda cmd: (cmds_declined.append(cmd), (0, "", ""))[1],
            check_root=False,
        )
        self.assertFalse(ok)
        self.assertEqual(cmds_declined, [])

        # 2) Confirming unloads in reverse-dependency order, binds pci-stub, and enables BAR0 decode
        cmds_run: list[list[str]] = []

        def _fake_runner(cmd: list[str]):
            cmds_run.append(cmd)
            if cmd[:2] == ["modprobe", "-r"]:
                # Remove that module line from synthetic /proc/modules
                remaining = [
                    ln for ln in mods.read_text().splitlines()
                    if not ln.startswith(cmd[2] + " ")
                ]
                mods.write_text("\n".join(remaining) + ("\n" if remaining else ""))
            return 0, "", ""

        ok2 = ensure_gpu_detached_interactive(
            "0000:05:00.0",
            input_fn=lambda _p: "y",
            show=lambda _s: None,
            sysfs_root=self.sysfs,
            pci_bus_root=bus_root,
            proc_modules_path=mods,
            runner=_fake_runner,
            check_root=False,
        )
        self.assertTrue(ok2)
        self.assertEqual(
            cmds_run[:4],
            [
                ["modprobe", "-r", "nvidia_drm"],
                ["modprobe", "-r", "nvidia_modeset"],
                ["modprobe", "-r", "nvidia_uvm"],
                ["modprobe", "-r", "nvidia"],
            ],
        )
        self.assertEqual((dev / "driver_override").read_text().strip(), "pci-stub")
        self.assertEqual(drivers_probe.read_text().strip(), "0000:05:00.0")
        self.assertEqual((dev / "enable").read_text().strip(), "1")

        # 3) When modprobe -r fails because nvidia-persistenced holds /dev/nvidiactl, report PID + hint
        mods.write_text("nvidia 1234 1 - Live 0x0\n")
        fake_proc = self.root / "fake_proc"
        p_fd = fake_proc / "4321" / "fd"
        p_fd.mkdir(parents=True)
        (fake_proc / "4321" / "comm").write_text("nvidia-persistenced\n")
        (p_fd / "3").symlink_to("/dev/nvidiactl")
        with self.assertRaisesRegex(BurnError, r"(?s)pid 4321 \(nvidia-persistenced\).*systemctl stop nvidia-persistenced"):
            unload_gpu_drivers(
                "0000:05:00.0",
                show=lambda _s: None,
                runner=lambda _cmd: (1, "", "modprobe: FATAL: Module nvidia is in use."),
                sysfs_root=self.sysfs,
                pci_bus_root=bus_root,
                proc_modules_path=mods,
                proc_root=fake_proc,
                check_root=False,
            )


if __name__ == "__main__":
    unittest.main()
