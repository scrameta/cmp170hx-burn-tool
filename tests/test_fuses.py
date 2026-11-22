"""Tests for the unified GA100 fuse & register definition catalog (vfi_burn.fuses)."""
from __future__ import annotations

import unittest

from vfi_burn.fuses import (
    DEVICE_ID_DENIED_BLOCKS,
    EXPERIMENTAL_INSPECT_REGS,
    FUSE_CATALOG,
    FUSES_BY_KEY,
    FuseType,
    OPTION_FIELDS,
    OPTION_REGS,
    REG_DEBUGCTRL,
    REG_FUSECTRL,
    register_name_for_address,
)
from vfi_burn.mmio import resolve_register_address


class TestFuseCatalog(unittest.TestCase):
    def test_unique_keys_and_nonempty_descriptions(self):
        keys = [f.key for f in FUSE_CATALOG]
        self.assertEqual(len(keys), len(set(keys)))
        for f in FUSE_CATALOG:
            self.assertTrue(f.key.strip(), f"empty key in {f}")
            self.assertTrue(f.name.strip(), f"empty name in {f}")
            self.assertTrue(f.description.strip(), f"empty description in {f}")
            self.assertTrue(f.functional_group.strip(), f"empty functional_group in {f}")

    def test_ff_overridable_and_burnable_invariants(self):
        for f in FUSE_CATALOG:
            if f.ff_overridable:
                self.assertEqual(f.fuse_type, FuseType.FF_OPTION)
                self.assertIsNotNone(f.ff_location, f"{f.key} is ff_overridable without ff_location")
            if f.burnable:
                self.assertTrue(f.ff_overridable, f"{f.key} is burnable without ff_overridable")
            if f.fuse_type is not FuseType.FF_OPTION:
                self.assertFalse(f.ff_overridable, f"non-FF fuse {f.key} must not be ff_overridable")
                self.assertFalse(f.burnable, f"non-FF fuse {f.key} must not be burnable")
                self.assertIsNone(f.ff_location)

        # Device-ID fuses are mapped in FF chain 2 but MUST never be overridable or burnable.
        for devid_key in ("pcie_devida", "pcie_devidb"):
            f = FUSES_BY_KEY[devid_key]
            self.assertFalse(f.ff_overridable)
            self.assertFalse(f.burnable)
            self.assertIsNotNone(f.ff_location)
        self.assertEqual(DEVICE_ID_DENIED_BLOCKS, frozenset({(2, 37), (2, 38)}))

    def test_all_bar0_addresses_are_4_byte_aligned(self):
        for f in FUSE_CATALOG:
            for attr in ("fused_addr", "defect_addr", "status_addr"):
                addr = getattr(f, attr)
                if addr is not None:
                    self.assertGreaterEqual(addr, 0)
                    self.assertEqual(addr & 3, 0, f"{f.key}.{attr}=0x{addr:x} not 4-byte aligned")

    def test_derived_projections_cover_catalog(self):
        ff_defs = [f for f in FUSE_CATALOG if f.ff_location is not None]
        self.assertEqual(len(OPTION_FIELDS), len(ff_defs))
        exp_defs = [f for f in FUSE_CATALOG if f.fuse_type is not FuseType.FF_OPTION]
        self.assertEqual(len(EXPERIMENTAL_INSPECT_REGS), len(exp_defs))

        # Spot-check key option, defect, status, and OTP registers resolve cleanly.
        self.assertEqual(OPTION_REGS["GPC_DISABLE"], 0x00820350)
        self.assertEqual(OPTION_REGS["GPC_DEFECTIVE"], 0x008205C4)
        self.assertEqual(OPTION_REGS["GPC_STATUS"], 0x00820C1C)
        self.assertEqual(OPTION_REGS["TPC_GPC7_DEFECTIVE"], 0x00820B80)
        self.assertEqual(OPTION_REGS["FUSE_ECC_EN"], 0x00820228)
        self.assertEqual(resolve_register_address("GPC_DISABLE"), 0x00820350)
        self.assertEqual(resolve_register_address("OPT_ECC_EN"), 0x00820228)
        self.assertEqual(register_name_for_address(REG_FUSECTRL), "FUSECTRL")
        self.assertEqual(register_name_for_address(REG_DEBUGCTRL), "DEBUGCTRL")


if __name__ == "__main__":
    unittest.main()
