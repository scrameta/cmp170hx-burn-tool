import mmap
import tempfile
import unittest

from vfi_burn.mmio import (
    AlignedMMIO,
    REG_DEBUGCTRL,
    REG_FUSECTRL,
    REG_PLM_F8,
    ReadOnlyMMIO,
    resolve_register_address,
)
from vfi_burn.model import BurnError


class TestMMIO(unittest.TestCase):
    def test_readonly_mmio_reads_and_cannot_write(self):
        buf = bytearray(16)
        buf[4:8] = (0xE0040000).to_bytes(4, "little")
        ro = ReadOnlyMMIO(buf)
        self.assertEqual(ro.read32(4), 0xE0040000)
        self.assertFalse(hasattr(ro, "write32"))

    def test_aligned_mmio_single_32bit_store_and_read(self):
        buf = bytearray(0x100)
        mmio = AlignedMMIO(buf)
        mmio.write32(0x10, 0x13572468)
        self.assertEqual(mmio.read32(0x10), 0x13572468)
        self.assertEqual(int.from_bytes(buf[0x10:0x14], "little"), 0x13572468)

    def test_alignment_bounds_and_value_checks(self):
        buf = bytearray(16)
        mmio = AlignedMMIO(buf)
        with self.assertRaisesRegex(BurnError, "4-byte aligned"):
            mmio.read32(2)
        with self.assertRaisesRegex(BurnError, "4-byte aligned"):
            mmio.write32(3, 1)
        with self.assertRaisesRegex(BurnError, "exceeds mapping size"):
            mmio.read32(16)
        with self.assertRaisesRegex(BurnError, "exceeds mapping size"):
            mmio.write32(16, 1)
        with self.assertRaisesRegex(BurnError, "32-bit range"):
            mmio.write32(0, 0x1_0000_0000)
        with self.assertRaisesRegex(BurnError, "32-bit range"):
            mmio.write32(0, -1)
        with self.assertRaisesRegex(BurnError, "32-bit range"):
            mmio.write32(0, True)

    def test_debugctrl_guard_blocks_implicit_write(self):
        buf = bytearray(REG_DEBUGCTRL + 16)
        mmio = AlignedMMIO(buf)
        with self.assertRaisesRegex(BurnError, "DEBUGCTRL"):
            mmio.write32(REG_DEBUGCTRL, 0x12345678)
        self.assertEqual(mmio.read32(REG_DEBUGCTRL), 0)
        mmio.write32(REG_DEBUGCTRL, 0x12345678, allow_debugctrl=True)
        self.assertEqual(mmio.read32(REG_DEBUGCTRL), 0x12345678)

    def test_mmap_closes_cleanly_after_mmio_access(self):
        with tempfile.TemporaryFile() as f:
            f.write(b"\x00" * 4096)
            f.flush()
            mm = mmap.mmap(f.fileno(), 4096)
            mmio = AlignedMMIO(mm)
            mmio.write32(0x20, 0xCAFEBABE)
            self.assertEqual(mmio.read32(0x20), 0xCAFEBABE)
            # If memoryview was leaked, mm.close() would raise BufferError.
            mm.close()

    def test_resolve_register_address(self):
        self.assertEqual(resolve_register_address("FUSECTRL"), REG_FUSECTRL)
        self.assertEqual(resolve_register_address("debugctrl"), REG_DEBUGCTRL)
        self.assertEqual(resolve_register_address("plm_f8"), REG_PLM_F8)
        self.assertEqual(resolve_register_address("FUSE_ECC_EN"), 0x00820228)
        self.assertEqual(resolve_register_address("opt_ecc_en"), 0x00820228)
        self.assertEqual(resolve_register_address("FUSE_SS_DP"), 0x00820224)
        self.assertEqual(resolve_register_address("FEAT_READOUT_0"), 0x00823814)
        self.assertEqual(resolve_register_address("FBPA_ECC_CTRL"), 0x009A0470)
        self.assertEqual(resolve_register_address("0x820000"), REG_FUSECTRL)
        self.assertEqual(resolve_register_address(0x8200F8), REG_PLM_F8)
        with self.assertRaisesRegex(BurnError, "aligned"):
            resolve_register_address("0x820001")
        with self.assertRaisesRegex(BurnError, "unknown register"):
            resolve_register_address("NOT_A_REG")


if __name__ == "__main__":
    unittest.main()
