import struct
import unittest

from vfi_burn.ffrecord import encode_ff
from vfi_burn.ga100 import (
    CMD_IDLE,
    CMD_PROGRAM,
    CMD_READ,
    CMD_SENSE_CTRL,
    GA100FuseController,
    REG_CRC_STATUS,
    REG_CTRL_WR_SECURE,
    REG_DFP_STATUS,
    REG_DSOV_STATUS,
    REG_EFP_STATUS,
    REG_EN_SW_OVERRIDE,
    REG_FUSEADDR,
    REG_FUSEDATA,
    REG_FUSEWDATA,
    REG_IFF_RECORD,
    REG_PLM_F4,
    REG_PLM_FC,
    REG_PMC_BOOT_0,
    REG_RESENSE_DISABLE,
    REG_SBFE_SHADOW,
    REG_WDATA_WR_SECURE,
    is_fusectrl_idle,
    normalize_bdf,
    parse_pcie_lnkcap2,
)
from vfi_burn.mmio import REG_DEBUGCTRL, REG_FUSECTRL, REG_PLM_F8
from vfi_burn.model import BurnError


class FakeGA100MMIO:
    """Realistic GA100 fuse-controller register & OTP state machine for offline testing."""

    def __init__(self, *, rows=None, goldilocks=True):
        self.rows = {i: 0 for i in range(512)}
        if rows:
            self.rows.update(rows)
        self.regs = {
            REG_PMC_BOOT_0: 0x170000A1,
            REG_FUSECTRL: 0xE0040000,
            REG_FUSEADDR: 0x00000000,
            REG_FUSEDATA: 0x00000000,
            REG_FUSEWDATA: 0x00000000,
            REG_EN_SW_OVERRIDE: 0x00000000,
            REG_IFF_RECORD: 0x00000000,
            REG_DEBUGCTRL: 0x00000000,
            REG_CRC_STATUS: 0x00000055,
            REG_EFP_STATUS: 0x00000000 if goldilocks else 0x00000001,
            REG_DFP_STATUS: 0x00000000,
            REG_DSOV_STATUS: 0x00000000 if goldilocks else 0x00000001,
            REG_PLM_F4: 0x00000000,  # locked until WDATA_WR_SECURE=0
            REG_PLM_F8: 0xFFFFFFFF if goldilocks else 0x00000000,
            REG_PLM_FC: 0xFFFFFFFF if goldilocks else 0x00000000,
            REG_WDATA_WR_SECURE: 0x00000001,
            REG_CTRL_WR_SECURE: 0x00000000,
            REG_RESENSE_DISABLE: 0x00000000 if goldilocks else 0x00000001,
            REG_SBFE_SHADOW: 0x00000000 if goldilocks else 0x00000001,
        }
        self.writes = []
        self.reads = []
        self.sense_calls = 0
        self.program_calls = 0
        self.poison_on_sense = False
        self.drop_plm_f8_on_sense = False
        self.lose_window_on_sense = False

    def read32(self, offset: int) -> int:
        self.reads.append(offset)
        return self.regs.get(offset, 0)

    def write32(self, offset: int, value: int, *, allow_debugctrl: bool = False) -> None:
        if offset == REG_DEBUGCTRL and not allow_debugctrl:
            raise BurnError("DEBUGCTRL blocked")
        self.writes.append((offset, value))
        if offset == REG_EN_SW_OVERRIDE:
            # Sticks only when DSOV == 0.
            if self.regs[REG_DSOV_STATUS] == 0:
                self.regs[offset] = value & 1
            return
        if offset == REG_WDATA_WR_SECURE:
            if self.regs[REG_DSOV_STATUS] == 0 and self.regs[REG_EN_SW_OVERRIDE] == 1:
                self.regs[offset] = value & 1
                self.regs[REG_PLM_F4] = 0xFFFFFFFF if (value & 1) == 0 else 0x00000000
            return
        if offset == REG_SBFE_SHADOW:
            if self.regs[REG_DSOV_STATUS] == 0 and self.regs[REG_EN_SW_OVERRIDE] == 1:
                self.regs[offset] = value & 1
            return
        if offset == REG_FUSEWDATA:
            # Only sticks when FUSEWDATA_PLM (PLM_f4) is open!
            if self.regs[REG_PLM_F4] == 0xFFFFFFFF:
                self.regs[offset] = value & 0xFFFFFFFF
            return
        if offset == REG_FUSECTRL:
            cmd = value & 0x1F
            if cmd == CMD_IDLE:
                return
            if cmd == CMD_READ:
                row = self.regs[REG_FUSEADDR]
                self.regs[REG_FUSEDATA] = self.rows[row]
                return
            if cmd == CMD_SENSE_CTRL:
                self.sense_calls += 1
                if self.poison_on_sense:
                    self.regs[REG_FUSECTRL] = 0xBADF510E
                    return
                if self.drop_plm_f8_on_sense:
                    self.regs[REG_PLM_F8] = 0xFFFFFFFE
                    return
                if self.lose_window_on_sense:
                    self.regs[REG_DSOV_STATUS] = 1
                    self.regs[REG_EFP_STATUS] = 1
                    self.regs[REG_SBFE_SHADOW] = 1
                    self.regs[REG_RESENSE_DISABLE] = 1
                    return
                # Real hardware behavior observed in ga100-ff-tools commit cb3a71d:
                # EFP 0->1, SBFE 0->1, RESENSE_DISABLE 0->1, DSOV stays 0, EN_SW/WSEC/PLMs preserved.
                self.regs[REG_EFP_STATUS] = 1
                self.regs[REG_SBFE_SHADOW] = 1
                self.regs[REG_RESENSE_DISABLE] = 1
                return
            if cmd == CMD_PROGRAM:
                self.program_calls += 1
                row = self.regs[REG_FUSEADDR]
                # Hardware enforces EFP=1, SBFE=0 (for FF rows >= 222), PLM_f4 open, DEBUGCTRL=0.
                if (
                    self.regs[REG_EFP_STATUS] == 1
                    and self.regs[REG_SBFE_SHADOW] == 0
                    and self.regs[REG_PLM_F4] == 0xFFFFFFFF
                    and self.regs[REG_DEBUGCTRL] == 0
                ):
                    # OTP fuses are 0->1 only (bitwise OR).
                    self.rows[row] |= self.regs[REG_FUSEWDATA]
                return
        self.regs[offset] = value & 0xFFFFFFFF


class TestGA100Controller(unittest.TestCase):
    def test_full_goldilocks_sense_arm_and_program_flow(self):
        mmio = FakeGA100MMIO(rows={8: 0x1008, 222: encode_ff(1, 5, 0x9867)})
        ctl = GA100FuseController(mmio, read_only=False, sleep=lambda _: None)

        # 1. Pre-sense health shows EFP=0, DSOV=0, PLM_f4=0 (locked).
        h0 = ctl.read_health_registers()
        self.assertEqual(h0.efp, 0)
        self.assertEqual(h0.dsov, 0)
        self.assertEqual(h0.plm_f4, 0)

        # 2. Cannot arm before SENSE_CTRL (EFP=0).
        with self.assertRaisesRegex(BurnError, "EFP_STATUS != 1"):
            ctl.arm()

        # 3. SENSE_CTRL transitions EFP 0->1, SBFE -> 1, RESENSE_DISABLE -> 1, keeps DSOV=0!
        ctl.sense_ctrl()
        self.assertEqual(mmio.sense_calls, 1)
        h1 = ctl.read_health_registers()
        self.assertEqual(h1.efp, 1)
        self.assertEqual(h1.dsov, 0)
        self.assertEqual(h1.sbfe, 1)
        self.assertEqual(h1.resense_disable, 1)
        self.assertEqual(h1.plm_f4, 0xFFFFFFFF)

        # 4. Second sense_ctrl() call in the same window is idempotent (does NOT re-issue cmd 3).
        ctl.sense_ctrl()
        self.assertEqual(mmio.sense_calls, 1)

        # 5. program_row arms (clears SBFE -> 0), programs blank row 223, and verifies readback!
        word = encode_ff(1, 19, 0x4200)
        got = ctl.program_row(223, word)
        self.assertEqual(got, word)
        self.assertEqual(mmio.rows[223], word)
        self.assertEqual(mmio.program_calls, 1)
        # DEBUGCTRL and FUSEDATA were NEVER written.
        written_regs = {addr for addr, _ in mmio.writes}
        self.assertNotIn(REG_DEBUGCTRL, written_regs)
        self.assertNotIn(REG_FUSEDATA, written_regs)

    def test_program_row_enforces_ff_bounds_iff_bit_blank_and_debugctrl(self):
        mmio = FakeGA100MMIO(rows={222: encode_ff(1, 5, 0x9867)})
        ctl = GA100FuseController(mmio, read_only=False, sleep=lambda _: None)
        ctl.sense_ctrl()
        good_word = encode_ff(1, 19, 0x4200)

        # Refuses rows outside 222..486 (control/FPF rows and IFF rows).
        for bad_row in (0, 1, 8, 221, 487, 511):
            with self.assertRaisesRegex(BurnError, "outside the FF region"):
                ctl.program_row(bad_row, good_word)

        # Refuses IFF bit 5 set.
        with self.assertRaisesRegex(BurnError, "IFF bit 5"):
            ctl.program_row(223, good_word | (1 << 5))

        # Refuses non-blank row 222.
        with self.assertRaisesRegex(BurnError, "not blank"):
            ctl.program_row(222, good_word)

        # Refuses when DEBUGCTRL (RWL) is nonzero.
        mmio.regs[REG_DEBUGCTRL] = 0x2
        with self.assertRaisesRegex(BurnError, "DEBUGCTRL"):
            ctl.program_row(223, good_word)
        self.assertEqual(mmio.program_calls, 0)

    def test_readonly_controller_blocks_arm_and_program(self):
        mmio = FakeGA100MMIO(rows={222: encode_ff(1, 5, 0x9867)})
        ctl = GA100FuseController(mmio, read_only=True, sleep=lambda _: None)
        self.assertEqual(ctl.read_row(222), encode_ff(1, 5, 0x9867))
        ctl.sense_ctrl()
        with self.assertRaisesRegex(BurnError, "read-only"):
            ctl.arm()
        with self.assertRaisesRegex(BurnError, "read-only"):
            ctl.program_row(223, encode_ff(1, 19, 0x4200))
        self.assertEqual(mmio.program_calls, 0)

    def test_sense_ctrl_detects_poison_plm_f8_drop_and_window_loss(self):
        for attr, regex in [
            ("poison_on_sense", "poisoned"),
            ("drop_plm_f8_on_sense", "too deep"),
            ("lose_window_on_sense", "WINDOW LOST"),
        ]:
            with self.subTest(case=attr):
                mmio = FakeGA100MMIO()
                setattr(mmio, attr, True)
                ctl = GA100FuseController(mmio, read_only=False, sleep=lambda _: None)
                with self.assertRaisesRegex(BurnError, regex):
                    ctl.sense_ctrl()

    def test_poison_stops_immediately_without_further_bar0_reads(self):
        mmio = FakeGA100MMIO()
        mmio.regs[REG_FUSECTRL] = 0xBADF510E
        ctl = GA100FuseController(mmio, read_only=True, sleep=lambda _: None)
        with self.assertRaisesRegex(BurnError, "poisoned"):
            ctl.read_health_registers()
        # Only FUSECTRL was read; zero other BAR0 registers touched!
        self.assertEqual(mmio.reads, [REG_FUSECTRL])

    def test_helpers_bdf_idle_and_pcie_lnkcap2(self):
        self.assertEqual(normalize_bdf("05:00.0"), "0000:05:00.0")
        self.assertEqual(normalize_bdf("0000:81:00.0"), "0000:81:00.0")
        with self.assertRaises(BurnError):
            normalize_bdf("not-a-bdf")
        self.assertTrue(is_fusectrl_idle(0xE0040000))
        self.assertFalse(is_fusectrl_idle(0xE00C0000))  # WRITE_PROGRAM state=3
        self.assertFalse(is_fusectrl_idle(0xBADF510E))

        # Build synthetic 256-byte PCIe config space with Cap ID 0x10 at 0x60 and LnkCap2 at 0x8C.
        cfg = bytearray(256)
        struct.pack_into("<I", cfg, 0x00, 0x20C210DE)
        cfg[0x34] = 0x60
        cfg[0x60] = 0x10  # PCIe capability ID
        cfg[0x61] = 0x00  # next cap = end
        struct.pack_into("<I", cfg, 0x60 + 0x2C, 0x0000001E)  # Gen1..4 supported
        self.assertEqual(parse_pcie_lnkcap2(bytes(cfg)), 0x0000001E)

    def test_end_to_end_process_records_with_ga100_controller(self):
        import json
        from pathlib import Path
        import tempfile
        from vfi_burn.backend import Host
        from vfi_burn.engine import Journal, process_records, verify_cold_snapshot
        from vfi_burn.model import Plan

        die = {i: 0x1000 + i for i in range(8, 16)}
        existing_ff = {222: encode_ff(1, 7, 0x0123), 223: encode_ff(1, 19, 0x4290)}
        noop_word = encode_ff(1, 7, 0x0123)
        fbp_word = encode_ff(1, 19, 0x4200)
        nvlink_word = encode_ff(1, 23, 0x0FE8)

        mmio = FakeGA100MMIO(rows={**die, **existing_ff}, goldilocks=True)
        ctl = GA100FuseController(mmio, read_only=False, sleep=lambda _: None)

        class MockLiveAdapter:
            simulation = False
            writer_verified = True
            def __init__(self, boot_id):
                self._boot = boot_id
            def pci_id(self): return "10de:20c2"
            def die_row(self, i): return ctl.read_row(i)
            def fuse_row(self, i): return ctl.read_row(i)
            def register(self, a): return ctl.rd(a)
            def field(self, name):
                return {"FBP_DISABLE": 0x852, "GPC_DISABLE": 0x0}.get(name, 0)
            def health(self): return ctl.read_health_registers(status_word=0, vout_mv=787.5)
            def host(self): return Host("pci-stub", False, False, "[none]", False, True, self._boot)
            def active_sense_readonly(self): ctl.sense_ctrl()
            def write_one_record(self, row, word): ctl.program_row(row, word)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            work.chmod(0o700)
            plan_path = work / "plan.json"
            plan_path.write_text(json.dumps({
                "schema_version": 1,
                "card": {
                    "pci_id": "10de:20c2",
                    "die_rows": {str(k): f"0x{v:08x}" for k, v in die.items()},
                    "expected_masks": {"FBP_DISABLE": "0x852"},
                    "defect_masks": {"FBP_DISABLE": "0x840"},
                },
                "fpf": {"first_row": 222, "last_row": 230, "empty_word": "0x00000000"},
                "environment": {"stock_mv": 787.5, "tolerance_mv": 25},
                "ceiling": {"base_sm": 124},
                "records": [
                    {
                        "name": "NOOP_TEST", "kind": "test", "target": "NOOP_C1_B7",
                        "word": f"0x{noop_word:08x}", "decode": "no-op test row",
                        "defect_sensitive": False, "expected_cold": {"FBP_DISABLE": "0x852"},
                    },
                    {
                        "name": "UNLOCK_FBP", "kind": "regular", "target": "FBP_DISABLE",
                        "word": f"0x{fbp_word:08x}", "decode": "FBP 0x852 -> 0x840",
                        "defect_sensitive": True,
                        "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
                        "expected_cold": {"FBP_DISABLE": "0x840"},
                    },
                    {
                        "name": "UNLOCK_NVLINK", "kind": "regular", "target": "NVLINK_DISABLE",
                        "word": f"0x{nvlink_word:08x}", "decode": "NVLINK clear",
                        "defect_sensitive": False,
                    },
                ],
            }))
            plan_path.chmod(0o600)
            plan = Plan.load(plan_path)
            jpath = work / "session.jsonl"

            # Stage 1: burn no-op test row with strict confirm + SENSE confirm
            tokens_1 = []
            def confirm_1(prompt, token):
                tokens_1.append(token)
                return True
            with Journal(jpath) as j:
                ptr = process_records(
                    plan, MockLiveAdapter("boot-1"), MockLiveAdapter("boot-1"), j,
                    stage="test", confirm=confirm_1, strict_confirm=True,
                    confirm_active_sense=True, show=lambda *_: None,
                )
            self.assertEqual(ptr, 225)
            self.assertEqual(mmio.rows[224], noop_word)
            self.assertEqual(mmio.sense_calls, 1)
            self.assertEqual(tokens_1, ["SENSE", f"YES 224 0x{noop_word:08x}"])

            # Cold boot verify Stage 1 -> receipt
            cold_1 = {
                "pci_id": "10de:20c2", "cold_boot": True,
                "boot_id": "boot-2", "previous_boot_id": "boot-1",
                "die_rows": {str(k): f"0x{v:08x}" for k, v in die.items()},
                "fpf_rows": {"224": f"0x{noop_word:08x}"},
                "fields": {"FBP_DISABLE": "0x852"},
            }
            with Journal(jpath) as j:
                receipt = verify_cold_snapshot(plan, cold_1, stage="test", journal=j)
            rpath = work / "test.receipt"
            rpath.write_text(json.dumps(receipt))
            rpath.chmod(0o600)

            # Stage 2: fresh Goldilocks window on boot-3; burn BOTH real records in ONE window!
            # SENSE_CTRL must fire ONCE (on record 1), and record 2 must succeed with RESENSE_DISABLE=1!
            mmio2 = FakeGA100MMIO(rows=dict(mmio.rows), goldilocks=True)
            ctl = GA100FuseController(mmio2, read_only=False, sleep=lambda _: None)
            with Journal(jpath) as j:
                ptr2 = process_records(
                    plan, MockLiveAdapter("boot-3"), MockLiveAdapter("boot-3"), j,
                    stage="real", test_receipt=rpath, confirm=lambda p, t: True,
                    strict_confirm=True, confirm_active_sense=True, show=lambda *_: None,
                )
            self.assertEqual(ptr2, 227)
            self.assertEqual(mmio2.rows[225], fbp_word)
            self.assertEqual(mmio2.rows[226], nvlink_word)
            self.assertEqual(mmio2.sense_calls, 1)
            self.assertEqual(mmio2.program_calls, 2)


if __name__ == "__main__":
    unittest.main()
