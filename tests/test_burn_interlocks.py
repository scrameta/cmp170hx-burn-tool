"""Regression coverage for burn safety interlocks; no real GPU access."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from vfi_burn.engine import (
    Overrides,
    _refuse_unresolved_attempt,
    _write_verify_or_halt,
    process_records,
)
from vfi_burn.ffrecord import FF_ROW_MAX, FF_ROW_MIN, encode_ff
from vfi_burn.fuses import DIE_ID_ROWS, FFCALC_KEY_TO_FIELD, FUSES_BY_KEY
from vfi_burn.ga100 import (
    CMD_PROGRAM,
    CMD_READ,
    GA100FuseController,
    GA100LiveAdapter,
    REG_FUSEADDR,
    REG_FUSECTRL,
    REG_FUSEDATA,
    REG_FUSEWDATA,
    REG_IFF_RECORD,
    read_ga100_card_state,
)
from vfi_burn.interfaces import Host
from vfi_burn.journal import Journal
from vfi_burn.model import (
    BurnError,
    Plan,
    PreTriggerRefusal,
    UnsafePreTriggerRefusal,
    canonical_ff_prefix_sha256,
)
from vfi_burn.sim_pcie import SimulatedGA100PCIe


class TestBurnInterlocks(unittest.TestCase):
    def make_controller(self):
        mmio = SimulatedGA100PCIe(rows={8: 0x1008, 222: encode_ff(1, 5, 0x9867)})
        ctl = GA100FuseController(mmio, read_only=False, sleep=lambda _: None)
        ctl.sense_ctrl()
        return mmio, ctl

    def test_ignored_clear_quarantines_without_restoring_address(self):
        mmio, ctl = self.make_controller()
        mmio.regs[REG_FUSEADDR] = 77
        write = mmio.write32

        def drop_clear(offset, value, **kwargs):
            if offset == REG_FUSEWDATA and value == 0 and mmio.program_calls:
                return
            return write(offset, value, **kwargs)

        mmio.write32 = drop_clear
        with self.assertRaisesRegex(BurnError, "zero latch did not stick"):
            ctl.program_row(223, encode_ff(1, 19, 0x4200))
        self.assertEqual(mmio.program_calls, 1)
        self.assertEqual(mmio.regs[REG_FUSEADDR], 223)
        before = (len(mmio.reads), len(mmio.writes))
        with self.assertRaisesRegex(BurnError, "quarantined"):
            ctl.read_row(223)
        self.assertEqual(before, (len(mmio.reads), len(mmio.writes)))

    def test_dropped_blank_read_cannot_return_stale_zero(self):
        mmio, ctl = self.make_controller()
        write = mmio.write32

        def drop_read(offset, value, **kwargs):
            if offset == REG_FUSECTRL and value == CMD_READ and mmio.regs[REG_FUSEADDR] == 223:
                return
            return write(offset, value, **kwargs)

        mmio.write32 = drop_read
        with self.assertRaisesRegex(PreTriggerRefusal, "not blank"):
            ctl.program_row(223, encode_ff(1, 19, 0x4200), blank_witness=(8, 0x1008))
        self.assertEqual(mmio.program_calls, 0)

    def test_delayed_read_completes_during_settle(self):
        mmio, ctl = self.make_controller()
        write = mmio.write32
        pending: list[int] = []

        def delay_read(offset, value, **kwargs):
            if offset == REG_FUSECTRL and value == CMD_READ:
                pending.append(mmio.regs[REG_FUSEADDR])
                return
            return write(offset, value, **kwargs)

        def settle(_):
            if pending:
                mmio.regs[REG_FUSEDATA] = mmio.rows[pending.pop(0)]

        mmio.write32 = delay_read
        ctl._sleep = settle
        self.assertEqual(ctl.read_row(8), 0x1008)

    def test_program_exception_does_not_touch_controller_again(self):
        mmio, ctl = self.make_controller()
        write = mmio.write32
        counts: list[tuple[int, int]] = []

        def fail_after_trigger(offset, value, **kwargs):
            write(offset, value, **kwargs)
            if offset == REG_FUSECTRL and value == CMD_PROGRAM:
                counts.append((len(mmio.reads), len(mmio.writes)))
                raise TypeError("injected transport exception after trigger")

        mmio.write32 = fail_after_trigger
        with self.assertRaises(TypeError):
            ctl.program_row(223, encode_ff(1, 19, 0x4200))
        self.assertEqual(mmio.program_calls, 1)
        self.assertEqual(counts[0], (len(mmio.reads), len(mmio.writes)))

    def test_unsafe_pre_trigger_refusal_preserves_root_cause_and_logs_halt_no_write(self):
        mmio, ctl = self.make_controller()
        orig_write = mmio.write32

        def poison_after_wdata(offset, value, **kwargs):
            orig_write(offset, value, **kwargs)
            if offset == REG_FUSEWDATA and value != 0:
                mmio.regs[REG_FUSECTRL] = 0xBADF510E

        mmio.write32 = poison_after_wdata
        with self.assertRaises(UnsafePreTriggerRefusal) as ctx:
            ctl.program_row(223, encode_ff(1, 19, 0x4200))
        msg = str(ctx.exception)
        self.assertIn("final pre-trigger controller gate failed", msg)
        self.assertIn("controller cleanup failed", msg)
        self.assertEqual(mmio.program_calls, 0)

        reader = Mock()
        writer = Mock()
        writer.write_one_record.side_effect = ctx.exception
        journal = Mock()
        record = SimpleNamespace(kind="regular", name="MEM_FBP", word=encode_ff(1, 19, 0x4200))
        with self.assertRaisesRegex(BurnError, "NO program command issued"):
            _write_verify_or_halt(
                None, reader, writer, journal, pointer=223,
                record=record, ov=Overrides(), manual_record_sha256=None,
            )
        self.assertEqual(journal.append.call_args.kwargs["event"], "HALT_UNVERIFIED")
        self.assertIs(journal.append.call_args.kwargs["write_issued"], False)

    def test_writer_typeerror_no_retry_or_diagnostic_reads(self):
        reader = Mock()
        writer = Mock()
        writer.write_one_record.side_effect = TypeError("after trigger")
        journal = Mock()
        record = SimpleNamespace(kind="devid", name="DEV", word=0x1234)
        with self.assertRaisesRegex(BurnError, "do NOT retry"):
            _write_verify_or_halt(
                None, reader, writer, journal, pointer=223,
                record=record, ov=Overrides(), manual_record_sha256=None,
            )
        writer.write_one_record.assert_called_once_with(223, 0x1234, allow_a100_devid=True)
        self.assertEqual(reader.mock_calls, [])
        self.assertEqual(journal.append.call_args.kwargs["event"], "HALT_UNVERIFIED")

    def test_journal_blocks_unresolved_or_uncertain_attempt(self):
        for outcome in (None, "HALT_UNVERIFIED", "aborted_pre_trigger", "verified_readback_only"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)
                p.chmod(0o700)
                with Journal(p / "burn.jsonl") as j:
                    j.append(
                        event="attempt", row=223, record="TEST",
                        intended="0x00001234", plan_sha256="hash",
                    )
                    if outcome == "aborted_pre_trigger":
                        j.append(event=outcome, row=223, record="TEST", write_issued=False)
                    elif outcome is not None:
                        j.append(
                            event=outcome, row=223, record="TEST",
                            readback="0x00001234", plan_sha256="hash",
                        )
                    if outcome in ("verified_readback_only", "aborted_pre_trigger"):
                        _refuse_unresolved_attempt(j)
                    else:
                        with self.assertRaises(BurnError):
                            _refuse_unresolved_attempt(j)

    def test_iff_read_exception_is_not_treated_as_zero(self):
        backend = Mock()
        backend.pci_id.return_value = "10de:20c2"
        backend.health.return_value = SimpleNamespace(fusectrl=0xE0040000, crc=0x55, efp=1)
        backend.register.side_effect = OSError("injected IFF read error")
        with self.assertRaisesRegex(BurnError, "cannot establish IFF boundary"):
            read_ga100_card_state(backend)
        backend.register.assert_called_once_with(REG_IFF_RECORD)
        backend.fuse_row.assert_not_called()

    def test_native_defect_mismatch_refuses_baseline(self):
        # Invoke the native validation with a mock object, without opening sysfs.
        adapter = Mock()
        adapter.health.return_value = SimpleNamespace(efp=1)
        adapter.register.return_value = 0
        masks = {
            field: 0
            for key, field in FFCALC_KEY_TO_FIELD.items()
            if FUSES_BY_KEY[key].defect_addr is not None
        }
        masks["FBP_DISABLE"] = 1
        plan = SimpleNamespace(
            first_row=222, last_row=486, empty_word=0,
            baseline_ff_rows={}, defect_masks=masks,
        )
        with self.assertRaisesRegex(BurnError, "live defect mask FBP_DISABLE"):
            GA100LiveAdapter.validate_burn_baseline(adapter, plan)
        self.assertIsNone(adapter._burn_bounds)
        adapter.die_row.assert_not_called()

    def _build_native_plan_and_adapter(self, tmp_dir: Path, records: list[dict], *, last_row: int = FF_ROW_MAX):
        die = {i: 0x1000 + i for i in DIE_ID_ROWS}
        base_rows = {
            222: encode_ff(1, 7, 0x0123),    # FBPA (no-op target)
            223: encode_ff(1, 19, 0x4290),   # FBP_DISABLE=0x852, ROP_L2[0]=0
        }
        defect_masks = {
            field: (0x840 if field == "FBP_DISABLE" else 0)
            for key, field in FFCALC_KEY_TO_FIELD.items()
            if FUSES_BY_KEY[key].defect_addr is not None
        }
        expected_masks = {"FBP_DISABLE": 0x852, "FBPA_DISABLE": 0x2460}
        plan_obj = {
            "schema_version": 1,
            "card": {
                "pci_id": "10de:20c2",
                "die_rows": {str(k): f"0x{v:08x}" for k, v in die.items()},
                "expected_masks": {k: f"0x{v:x}" for k, v in expected_masks.items()},
                "defect_masks": {k: f"0x{v:x}" for k, v in defect_masks.items()},
            },
            "fpf": {
                "first_row": FF_ROW_MIN,
                "last_row": last_row,
                "empty_word": "0x00000000",
                "expected_next_row": 224,
                "baseline_ff_rows": {str(r): f"0x{w:08x}" for r, w in base_rows.items()},
                "baseline_ff_sha256": canonical_ff_prefix_sha256(base_rows),
            },
            "ceiling": {"base_sm": 128},
            "records": records,
        }
        plan_path = tmp_dir / "plan.json"
        plan_path.write_text(json.dumps(plan_obj), encoding="utf-8")
        plan_path.chmod(0o600)
        plan = Plan.load(plan_path)

        regs = {REG_IFF_RECORD: 0}
        for key, field in FFCALC_KEY_TO_FIELD.items():
            addr = FUSES_BY_KEY[key].defect_addr
            if addr is not None:
                regs[addr] = defect_masks[field]
        mmio = SimulatedGA100PCIe(rows={**die, **base_rows})
        mmio.regs.update(regs)

        class NativeLikeAdapter(GA100LiveAdapter):
            """Exercises GA100LiveAdapter methods over SimulatedGA100PCIe without sysfs."""

            simulation = False

            def __init__(self, pcie: SimulatedGA100PCIe):
                self.bdf = "0000:05:00.0"
                self.writable = True
                self.write_enabled = True
                self.writer_verified = True
                self.assume_hold_active = False
                self._bdf_lock = None
                self._fd = None
                self._mm = None
                self.controller = GA100FuseController(pcie, read_only=False, sleep=lambda _: None)
                self._fields = dict(expected_masks)

            def pci_id(self) -> str:
                return "10de:20c2"

            def host(self) -> Host:
                return Host(
                    bound_driver=None,
                    nvidia_loaded=False,
                    nouveau_loaded=False,
                    lockdown="[none]",
                    hold_active=True,
                    boot_id="boot-native-test",
                )

            def field(self, name: str) -> int:
                return self._fields[name]

        return plan, mmio, NativeLikeAdapter(mmio)

    def test_native_baseline_happy_path_bounds_and_two_pass_scan(self):
        noop = encode_ff(1, 7, 0x0123)
        fbp = encode_ff(1, 19, 0x4200)
        records = [
            {
                "name": "NOOP_TEST", "kind": "test", "target": "NOOP_C1_B7",
                "word": f"0x{noop:08x}", "decode": "no-op", "defect_sensitive": False,
                "expected_cold": {"FBP_DISABLE": "0x852"},
            },
            {
                "name": "MEM_FBP", "kind": "regular", "target": "FBP_DISABLE",
                "word": f"0x{fbp:08x}", "decode": "FBP 0x852 -> 0x840",
                "defect_sensitive": True,
                "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            work.chmod(0o700)
            plan, mmio, adapter = self._build_native_plan_and_adapter(work, records, last_row=240)
            # Writing before validate_burn_baseline is refused pre-trigger.
            with self.assertRaisesRegex(PreTriggerRefusal, "baseline not validated"):
                adapter.write_one_record(224, noop)
            self.assertEqual(mmio.program_calls, 0)

            with Journal(work / "burn.jsonl") as j:
                ptr = process_records(
                    plan, adapter, adapter, j, stage="all",
                    confirm=lambda *_: True, show=lambda *_: None,
                )
            self.assertEqual(ptr, 226)
            self.assertEqual(mmio.program_calls, 2)
            # Live burn runs 2-pass FF scan: at least 2 * (240 - 222 + 1) = 38 CMD_READs.
            cmd_reads = sum(1 for off, val in mmio.writes if off == REG_FUSECTRL and val == CMD_READ)
            self.assertGreaterEqual(cmd_reads, 2 * (240 - FF_ROW_MIN + 1))

            # Row outside validated bounds is refused pre-trigger.
            with self.assertRaisesRegex(PreTriggerRefusal, "outside validated bounds"):
                adapter.write_one_record(241, noop)

    def test_native_pre_burn_preview_vetoes_split_group_with_zero_program_calls(self):
        noop = encode_ff(1, 7, 0x0123)
        fbp = encode_ff(1, 19, 0x4200)
        fbpa = encode_ff(1, 7, 0x0103)
        records = [
            {
                "name": "NOOP_TEST", "kind": "test", "target": "NOOP_C1_B7",
                "word": f"0x{noop:08x}", "decode": "no-op", "defect_sensitive": False,
                "expected_cold": {"FBP_DISABLE": "0x852"},
            },
            {
                "name": "MEM_FBP", "kind": "regular", "target": "FBP_DISABLE",
                "word": f"0x{fbp:08x}", "decode": "FBP 0x852 -> 0x840",
                "defect_sensitive": True,
                "mask_update": {"field": "FBP_DISABLE", "from": "0x852", "to": "0x840"},
            },
            {
                "name": "MEM_FBPA", "kind": "regular", "target": "FBPA_DISABLE",
                "word": f"0x{fbpa:08x}", "decode": "FBPA layout update",
                "defect_sensitive": False,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            work.chmod(0o700)
            plan, mmio, adapter = self._build_native_plan_and_adapter(work, records, last_row=240)
            shown: list[str] = []
            with Journal(work / "burn.jsonl") as j:
                with self.assertRaisesRegex(BurnError, "unsafe power-off boundary"):
                    process_records(
                        plan, adapter, adapter, j, stage="all", max_records=2,
                        confirm=lambda *_: True, show=shown.append,
                    )
            self.assertEqual(mmio.program_calls, 0)
            self.assertIn("not a clean power-off boundary", "\n".join(shown))

    def test_devid_writer_missing_authorization_kwarg_refused_before_burn(self):
        noop = encode_ff(1, 7, 0x0123)
        devida = encode_ff(2, 37, 0x20F1 << 1)
        devidb = encode_ff(2, 38, 0x20F1 << 1)
        records = [
            {
                "name": "NOOP_TEST", "kind": "test", "target": "NOOP_C1_B7",
                "word": f"0x{noop:08x}", "decode": "no-op", "defect_sensitive": False,
                "expected_cold": {"FBP_DISABLE": "0x852"},
            },
            {
                "name": "A100_DEVIDA", "kind": "devid", "target": "PCIE_DEVIDA",
                "word": f"0x{devida:08x}", "decode": "DEVIDA -> 0x20f1", "defect_sensitive": False,
            },
            {
                "name": "A100_DEVIDB", "kind": "devid", "target": "PCIE_DEVIDB",
                "word": f"0x{devidb:08x}", "decode": "DEVIDB -> 0x20f1", "defect_sensitive": False,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            work.chmod(0o700)
            plan, mmio, adapter = self._build_native_plan_and_adapter(work, records, last_row=240)

            class LegacyWriter:
                def write_one_record(self, row: int, word: int) -> None:
                    raise AssertionError("must not be called")

            with Journal(work / "burn.jsonl") as j:
                with self.assertRaisesRegex(BurnError, "explicit device-ID authorization"):
                    process_records(
                        plan, adapter, LegacyWriter(), j, stage="all",
                        confirm=lambda *_: True, show=lambda *_: None,
                    )
            self.assertEqual(mmio.program_calls, 0)


if __name__ == "__main__":
    unittest.main()
