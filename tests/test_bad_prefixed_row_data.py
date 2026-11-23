"""A fuse ROW that reads 0xBADxxxxx can be real OTP data, not a fuse-unit error.

FUSEDATA returns the raw 32-bit row, and an FF record's top 16 bits are its block data,
so a legitimate record with data 0xBAD0..0xBADF reads 0xBADxxxxx. Previously every row
read with that prefix was refused as poison (in read_row, the engine scans, Plan.load's
baseline check and the console), so a card carrying such a record could never be used.
GA100 read_row now decides: error only if the controller is unhealthy straight after the
read or a second FUSEDATA read disagrees; otherwise the value is data.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from vfi_burn.engine import Journal, process_records
from vfi_burn.ffrecord import FF_ROW_MAX, FF_ROW_MIN, encode_ff, make_noop_test_record
from vfi_burn.fuses import DIE_ID_ROWS, REG_FUSECTRL, REG_FUSEDATA, REG_PMC_BOOT_0
from vfi_burn.ga100 import CMD_READ, GA100FuseController
from vfi_burn.model import BurnError, Plan, canonical_ff_prefix_sha256
from vfi_burn.sim_pcie import SimulatedGA100Adapter, SimulatedGA100PCIe

BAD_DATA_WORD = encode_ff(1, 7, 0xBAD3)  # FBPA block 7 holding block data 0xBAD3
assert BAD_DATA_WORD >> 20 == 0xBAD


class FaultAfterRead(SimulatedGA100PCIe):
    """Inject a fault that shows up only after a CMD_READ, the way a real PRI error would."""

    def __init__(self, *, fault: str, **kw):
        super().__init__(**kw)
        self.fault = fault
        self.armed = False
        self.fusedata_reads = 0

    def write32(self, offset, value, *, allow_debugctrl=False):
        super().write32(offset, value, allow_debugctrl=allow_debugctrl)
        if offset == REG_FUSECTRL and value == CMD_READ:
            self.armed = True

    def read32(self, offset):
        v = super().read32(offset)
        if not self.armed:
            return v
        if offset == REG_FUSEDATA:
            self.fusedata_reads += 1
            if self.fault == "unstable" and self.fusedata_reads > 1:
                return 0xBAD00100
            return 0xBADF5040 if self.fault != "unstable" else v
        if self.fault == "fusectrl" and offset == REG_FUSECTRL and self.fusedata_reads:
            return 0xBADF5040  # poll_idle saw idle; the error shows only after FUSEDATA
        if self.fault == "pmc" and offset == REG_PMC_BOOT_0:
            return 0xFFFFFFFF
        return v


class TestReadRow(unittest.TestCase):
    def ctl(self, mmio):
        return GA100FuseController(mmio, read_only=True, sleep=lambda _: None)

    def test_bad_prefixed_row_is_returned_as_data_when_controller_healthy(self):
        mmio = SimulatedGA100PCIe(rows={222: BAD_DATA_WORD})
        self.assertEqual(self.ctl(mmio).read_row(222), BAD_DATA_WORD)

    def test_fusectrl_poisoned_after_read_is_an_error(self):
        mmio = FaultAfterRead(fault="fusectrl", rows={222: 0x12345678})
        with self.assertRaisesRegex(BurnError, "FUSECTRL poisoned"):
            self.ctl(mmio).read_row(222)

    def test_unstable_fusedata_is_an_error(self):
        mmio = FaultAfterRead(fault="unstable", rows={222: BAD_DATA_WORD})
        with self.assertRaisesRegex(BurnError, "FUSEDATA unstable"):
            self.ctl(mmio).read_row(222)

    def test_bar0_dead_after_read_is_an_error(self):
        mmio = FaultAfterRead(fault="pmc", rows={222: 0x12345678})
        with self.assertRaisesRegex(BurnError, "PMC_BOOT_0"):
            self.ctl(mmio).read_row(222)


class TestNoopPicker(unittest.TestCase):
    def test_noop_never_uses_a_bad_prefixed_block(self):
        rows = {222: BAD_DATA_WORD, 223: encode_ff(1, 19, 0x4290)}
        rec = make_noop_test_record(rows)
        self.assertEqual(rec["target"], "NOOP_CHAIN1_BLK19")
        with self.assertRaisesRegex(BurnError, "0xBAD0..0xBADF"):
            make_noop_test_record({222: BAD_DATA_WORD})


class TestSessionOnCardWithBadPrefixedRecord(unittest.TestCase):
    """Factory FF prefix contains a record whose block data is 0xBAD3."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work = Path(self._tmp.name)
        self.work.chmod(0o700)
        self.die = {i: 0x1000 + i for i in DIE_ID_ROWS}
        self.base = {222: BAD_DATA_WORD, 223: encode_ff(1, 19, 0x4290)}
        self.expected = {"FBPA_DISABLE": 0xBAD3 << 5, "FBP_DISABLE": 0x852, "ROP_L2_DISABLE": 0}
        self.defect = {"FBPA_DISABLE": 0, "FBP_DISABLE": 0x840, "ROP_L2_DISABLE": 0}

    def plan(self):
        noop, fbp = encode_ff(1, 19, 0x4290), encode_ff(1, 19, 0x4200)
        obj = {
            "schema_version": 1,
            "card": {"pci_id": "10de:20c2",
                     "die_rows": {str(k): f"0x{v:08x}" for k, v in self.die.items()},
                     "expected_masks": {k: f"0x{v:x}" for k, v in self.expected.items()},
                     "defect_masks": {k: f"0x{v:x}" for k, v in self.defect.items()}},
            "fpf": {"first_row": FF_ROW_MIN, "last_row": FF_ROW_MAX, "empty_word": "0x00000000",
                    "expected_next_row": 224,
                    "baseline_ff_rows": {str(r): f"0x{w:08x}" for r, w in self.base.items()},
                    "baseline_ff_sha256": canonical_ff_prefix_sha256(self.base)},
            "ceiling": {"base_sm": 122},
            "records": [
                {"name": "NOOP_TEST", "kind": "test", "target": "NOOP_C1_B19", "word": f"0x{noop:08x}",
                 "decode": "no-op", "defect_sensitive": False, "expected_cold": {"FBP_DISABLE": "0x852"}},
                {"name": "UNLOCK_FBP", "kind": "regular", "target": "FBP_DISABLE", "word": f"0x{fbp:08x}",
                 "decode": "FBP 0x852 -> 0x840", "defect_sensitive": True,
                 "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
                 "expected_cold": {"FBP_DISABLE": "0x840"}},
            ],
        }
        p = self.work / "plan.json"
        p.write_text(json.dumps(obj))
        p.chmod(0o600)
        return Plan.load(p)  # previously refused: baseline row "matches poison prefix"

    def test_full_session_burns_on_a_card_with_bad_prefixed_record(self):
        ad = SimulatedGA100Adapter(die=self.die, rows=dict(self.base), fields=dict(self.expected))
        with Journal(self.work / "j.jsonl") as j:
            process_records(self.plan(), ad, ad, j, stage="all", confirm=lambda *_: True,
                            show=lambda *_: None)
        self.assertEqual(ad.pcie.program_calls, 2)
        self.assertEqual(ad.pcie.rows[222], BAD_DATA_WORD, "factory record untouched")
        self.assertEqual(ad.pcie.rows[225], encode_ff(1, 19, 0x4200))


if __name__ == "__main__":
    unittest.main()
