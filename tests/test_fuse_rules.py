"""Tests for declarative GA100 fuse combination groups and coherence rules."""
from __future__ import annotations

import unittest

from vfi_burn.fuse_rules import (
    COMBINATION_RULES,
    FUSE_GROUPS,
    FUSE_GROUPS_BY_NAME,
    FuseCombinationRule,
    evaluate_fuse_combinations,
    poweroff_domain_for_group,
    poweroff_domains_for_groups,
)
from vfi_burn.fuses import FUSES_BY_KEY


class TestFuseGroupsAndRules(unittest.TestCase):
    def test_fuse_groups_reference_known_overridable_fuses(self):
        self.assertGreaterEqual(len(FUSE_GROUPS), 5)
        for grp in FUSE_GROUPS:
            self.assertTrue(grp.member_keys, f"group {grp.name} has no members")
            self.assertTrue(grp.poweroff_domain, f"group {grp.name} has no poweroff_domain")
            for k in grp.member_keys:
                self.assertIn(k, FUSES_BY_KEY)
                self.assertTrue(FUSES_BY_KEY[k].ff_overridable)
        self.assertIn("gpc_disable", FUSE_GROUPS_BY_NAME["compute"].member_keys)
        self.assertIn("fbp_disable", FUSE_GROUPS_BY_NAME["memory"].member_keys)
        self.assertIn("pcie_magic_gen4_dis", FUSE_GROUPS_BY_NAME["pcie"].member_keys)

    def test_compute_and_memory_share_atomic_sm_hbm_poweroff_domain(self):
        self.assertEqual(poweroff_domain_for_group("compute"), "sm_hbm")
        self.assertEqual(poweroff_domain_for_group("memory"), "sm_hbm")
        self.assertEqual(poweroff_domain_for_group("pcie"), "pcie")
        self.assertEqual(
            poweroff_domains_for_groups(["compute", "memory", "noop"]),
            {"sm_hbm"},
        )

    def test_combination_rule_abc_cannot_be_instantiated_incomplete(self):
        class IncompleteRule(FuseCombinationRule):
            pass

        with self.assertRaises(TypeError):
            IncompleteRule()

    def test_gpc_tpc_and_pes_coherence(self):
        # Clean compute state: GPC0 enabled with active TPCs and PES; GPC1 disabled.
        clean = {
            "gpc_disable": 0x02,
            "pes_disable": 0x000038,  # GPC1's 3 PES bits disabled, GPC0 enabled
            **{f"tpc_gpc{i}_disable": 0x00 for i in range(8)},
        }
        self.assertEqual(evaluate_fuse_combinations(clean), [])

        # Enabled GPC0 with all 8 TPCs disabled -> flagged.
        bad_tpc = dict(clean, tpc_gpc0_disable=0xFF)
        issues = evaluate_fuse_combinations(bad_tpc)
        self.assertEqual([i.rule_id for i in issues], ["compute.gpc_tpc_coherence"])
        self.assertIn("tpc_gpc0_disable", issues[0].fields)

        # Enabled GPC0 with all 3 PES disabled -> flagged.
        bad_pes = dict(clean, pes_disable=0x000007)
        issues_pes = evaluate_fuse_combinations(bad_pes)
        self.assertEqual([i.rule_id for i in issues_pes], ["compute.gpc_pes_coherence"])

    def test_memory_slice_pairing_rules(self):
        # FBP0 enabled, FBP1 disabled; FBPA/FBIO/ROP_L2 disable bits 2,3 (for FBP1) only -> clean.
        clean = {
            "fbp_disable": 0x002,
            "fbpa_disable": 0x00000C,
            "fbio_disable": 0x00000C,
            "rop_l2_disable": 0x00000C,
        }
        self.assertEqual(evaluate_fuse_combinations(clean), [])

        # FBP0 enabled, but both paired FBPA slices (bits 0,1) disabled -> flagged.
        bad_pair = dict(clean, fbpa_disable=0x00000F, fbio_disable=0x00000F)
        issues = evaluate_fuse_combinations(bad_pair)
        rule_ids = [i.rule_id for i in issues]
        self.assertIn("memory.fbp_slice_pairing", rule_ids)

        # FBPA and FBIO disagree on enabled FBP0 -> flagged.
        mismatch = dict(clean, fbpa_disable=0x00000D, fbio_disable=0x00000C)
        issues_mm = evaluate_fuse_combinations(mismatch)
        self.assertEqual([i.rule_id for i in issues_mm], ["memory.fbpa_fbio_mismatch"])

    def test_pcie_speed_coherence_rules(self):
        # All locked (stock CMP 170HX) or all unlocked (Gen4 ready) -> clean.
        stock = {
            "disable_gen3_speed_bit1": 1,
            "disable_gen3_speed_bit9": 1,
            "pcie_magic_gen4_dis": 1,
            "pcie_boot_gen23_dis": 1,
            "pcie_boot_gen3_dis": 1,
        }
        unlocked = {k: 0 for k in stock}
        self.assertEqual(evaluate_fuse_combinations(stock), [])
        self.assertEqual(evaluate_fuse_combinations(unlocked), [])

        # Split Gen3 bits (bit1=0, bit9=1) -> flagged.
        split_gen3 = dict(stock, disable_gen3_speed_bit1=0)
        self.assertEqual(
            [i.rule_id for i in evaluate_fuse_combinations(split_gen3)],
            ["pcie.gen3_split_bits"],
        )

        # Gen4 enabled while Gen3 still disabled -> flagged.
        gen4_only = dict(stock, pcie_magic_gen4_dis=0)
        self.assertEqual(
            [i.rule_id for i in evaluate_fuse_combinations(gen4_only)],
            ["pcie.gen4_with_gen3_disabled"],
        )

        # Boot Gen3 cleared while Boot Gen23 still set -> flagged.
        boot_conflict = dict(stock, pcie_boot_gen3_dis=0)
        self.assertEqual(
            [i.rule_id for i in evaluate_fuse_combinations(boot_conflict)],
            ["pcie.boot_gen23_conflict"],
        )

    def test_partial_field_dict_skips_inapplicable_rules(self):
        # Passing only unrelated or empty fields must not produce spurious warnings.
        self.assertEqual(evaluate_fuse_combinations({}), [])
        self.assertEqual(evaluate_fuse_combinations({"gpc_disable": 0x00}), [])
        self.assertGreaterEqual(len(COMBINATION_RULES), 4)


if __name__ == "__main__":
    unittest.main()
