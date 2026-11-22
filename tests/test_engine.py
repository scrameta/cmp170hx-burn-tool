import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from vfi_burn.backend import FakeController, Health, Host, MissingWriter
from vfi_burn.engine import (Journal, preflight, scan_slots, check_plan, check_health,
                             reconcile, process_records, verify_cold_snapshot)
from vfi_burn.model import BurnError, Plan, Record

ROOT = Path(__file__).resolve().parents[1]


class Harness(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        data = json.loads((ROOT / 'examples/plan.example.json').read_text())
        p = self.work / 'plan.json'
        p.write_text(json.dumps(data))
        p.chmod(0o600)
        self.plan = Plan.load(p)
        self.controller = FakeController(fields={'GPC_DISABLE':0x85},
                                         rows={222: 0xABCD00DE, **{i:0 for i in range(223,228)}})
        self.journal_path = self.work / 'journal.jsonl'
        self.confirm_yes = lambda _, token: True
        self.confirm_no = lambda _, token: False

    def journal(self):
        return Journal(self.journal_path)

    def run_test(self, *, confirm=None, writer=None):
        with self.journal() as j:
            return process_records(self.plan, self.controller, writer or self.controller, j,
                                   stage='test', confirm=confirm or self.confirm_yes, show=lambda *_:None)

    def logged(self):
        return [json.loads(line) for line in self.journal_path.read_text().splitlines()]

    def test_plan_valid_and_offline(self):
        self.assertEqual(check_plan(self.plan)['GPC_DISABLE'], 0x84)

    def test_missing_real_writer_stops_and_preserves_pointer(self):
        with self.assertRaisesRegex(BurnError, 'disabled|DISABLED'):
            self.run_test(writer=MissingWriter())
        self.assertEqual(self.controller.rows[223],0)
        self.assertTrue(any(e['event']=='HALT_UNVERIFIED' and not e['pointer_advanced']
                            for e in self.logged()))

    def test_test_record_one_at_a_time(self):
        ptr = self.run_test()
        self.assertEqual(ptr,224)
        self.assertEqual(self.controller.rows[223],0x13572468)
        self.assertEqual(self.controller.rows[224],0)
        self.assertEqual(len([e for e in self.controller.events if e[0]=='write']),1)
        self.assertEqual(len([e for e in self.logged() if e['event']=='verified_readback_only']),1)
        self.assertFalse(self.logged()[-2]['cold_boot_verified'])

    def test_no_confirmation_no_writes(self):
        with self.assertRaisesRegex(BurnError, 'declined'):
            self.run_test(confirm=self.confirm_no)
        self.assertFalse(any(e[0]=='write' for e in self.controller.events))

    def test_poison_health_preflight(self):
        for attr,value in [('fusectrl',0xBADF510E),('en_sw',0xBAD00001),
                           ('plm_f8',0xFFFFFFFE),('status_word',0x2042),
                           ('crc',0x44),('plm',0xFFFFFFFE),('dsov',1),
                           ('vout_mv',300.0)]:
            with self.subTest(attr=attr):
                c=copy.deepcopy(self.controller)
                setattr(c.state,attr,value)
                with self.assertRaises(BurnError):
                    preflight(self.plan,c,active_sense=True)
                self.assertFalse(any(e[0]=='write' for e in c.events))

    def test_sense_poison_refuses(self):
        self.controller.after_sense = lambda c: setattr(c.state,'en_sw',0xBADF510E)
        with self.assertRaisesRegex(BurnError, 'poison'):
            self.run_test()
        self.assertFalse(any(e[0]=='write' for e in self.controller.events))

    def test_sense_plm_collapse_refuses(self):
        self.controller.after_sense=lambda c: setattr(c.state,'plm_f8',0xFFFFFFFE)
        with self.assertRaisesRegex(BurnError, 'too deep'):
            self.run_test()

    def test_host_gates(self):
        variations = [ ('bound_driver','nvidia'), ('nvidia_loaded',True),
                       ('nouveau_loaded',True), ('lockdown','[integrity]'),
                       ('vrm_tainted',True), ('hold_active',False) ]
        for attr,v in variations:
            with self.subTest(attr=attr):
                c=copy.deepcopy(self.controller)
                setattr(c.machine, attr, v)
                with self.assertRaises(BurnError):
                    preflight(self.plan,c,active_sense=True)

    def test_die_identity_mask_and_sku(self):
        c=copy.deepcopy(self.controller)
        c.die[8] ^= 1
        with self.assertRaisesRegex(BurnError, 'DIE ID MISMATCH'):
            preflight(self.plan,c,active_sense=True)
        c=copy.deepcopy(self.controller)
        c.fields['GPC_DISABLE'] ^= 1
        with self.assertRaisesRegex(BurnError, 'IDENTITY MISMATCH'):
            preflight(self.plan,c,active_sense=True)
        c=copy.deepcopy(self.controller)
        c._pci_id='10de:2082'
        with self.assertRaisesRegex(BurnError, 'PCI identity'):
            preflight(self.plan,c,active_sense=True)

    def test_ambiguous_gap_and_no_free_slots(self):
        c=copy.deepcopy(self.controller)
        c.rows[225]=0x01234567
        with self.assertRaisesRegex(BurnError, 'ambiguous'):
            scan_slots(self.plan,c)
        c=copy.deepcopy(self.controller)
        for i in range(223,228): c.rows[i] = i
        with self.assertRaisesRegex(BurnError, 'full'):
            scan_slots(self.plan,c)

    def test_unstable_scan(self):
        c=copy.deepcopy(self.controller)
        prior = c.fuse_row
        count=[0]
        def unstable(i):
            count[0]+=1
            return prior(i) ^ 1 if count[0]>6 else prior(i)
        c.fuse_row=unstable
        with self.assertRaisesRegex(BurnError, 'changed between scans'):
            scan_slots(self.plan,c)

    def test_defective_bit_refused(self):
        r=replace(self.plan.records[1], from_disable=0x85, to_disable=0x80)
        with self.assertRaisesRegex(BurnError, 'genuinely defective'):
            reconcile(self.plan,r,self.plan.expected_masks.copy())

    def test_ceiling_undetermined_and_exceeded(self):
        r=replace(self.plan.records[1],resulting_sm_count=None)
        with self.assertRaisesRegex(BurnError, 'undetermined'):
            reconcile(self.plan,r,self.plan.expected_masks.copy())
        r=replace(self.plan.records[1],resulting_sm_count=123)
        with self.assertRaisesRegex(BurnError, 'beyond supplied ceiling'):
            reconcile(self.plan,r,self.plan.expected_masks.copy())

    def test_wrong_mask_chain(self):
        r=replace(self.plan.records[1],from_disable=0x89)
        with self.assertRaisesRegex(BurnError, 'inconsistent'):
            reconcile(self.plan,r,self.plan.expected_masks.copy())

    def test_no_double_write_on_readback_corruption(self):
        self.controller.corrupt_readback=True
        with self.assertRaisesRegex(BurnError, 'READBACK MISMATCH'):
            self.run_test()
        self.assertEqual(len([e for e in self.controller.events if e[0]=='write']),1)
        self.assertFalse(self.logged()[-1]['pointer_advanced'])

    def test_no_advance_postwrite_poison(self):
        for field, value in [('fusectrl',0xBADF5040),('status_word',0x2042),
                             ('plm_f8',0xFFFFFFFE),('vout_mv',41.0)]:
            with self.subTest(field=field):
                c=copy.deepcopy(self.controller)
                c.after_write=lambda obj, f=field, v=value: setattr(obj.state,f,v)
                with self.journal() as j:
                    with self.assertRaisesRegex(BurnError, 'HALT'):
                        process_records(self.plan,c,c,j,stage='test',confirm=self.confirm_yes,
                                        show=lambda *_:None)
                self.assertFalse(self.logged()[-1]['pointer_advanced'])
                self.assertEqual(len([e for e in c.events if e[0]=='write']),1)

    def test_no_retry_on_write_exception(self):
        self.controller.fail_write=True
        with self.assertRaisesRegex(BurnError, 'HALT'):
            self.run_test()
        self.assertEqual(len([e for e in self.controller.events if e[0]=='write']),1)

    def test_real_stage_requires_receipt(self):
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError, 'receipt required'):
                process_records(self.plan,self.controller,self.controller,j,
                                stage='real',confirm=self.confirm_yes,show=lambda *_:None)

    def test_journal_permissions_and_no_symlink(self):
        with self.journal() as j: j.append(event='test')
        self.assertEqual(self.journal_path.stat().st_mode & 0o777, 0o600)
        target = self.work / 'out'
        target.touch()
        symlink=self.work / 'lnk'
        symlink.symlink_to(target)
        with self.assertRaises(BurnError): Journal(symlink)

    def test_plan_mode_and_bad_die(self):
        p=self.work / 'bad.json'
        p.write_text((ROOT / 'examples/plan.example.json').read_text())
        with self.assertRaisesRegex(BurnError, '0600'):
            Plan.load(p)
        p.chmod(0o600)
        bad=json.loads(p.read_text())
        del bad['card']['die_rows']['8']
        p.write_text(json.dumps(bad))
        with self.assertRaisesRegex(BurnError, 'eight'):
            Plan.load(p)

    def test_cold_snapshot_comparison_and_fake_receipt(self):
        self.run_test()
        cold=dict(pci_id=self.plan.pci_id, fpf_rows={"223": "0x13572468"}, die_rows={str(k):v for k,v in self.plan.die_rows.items()},
                  fields={'SYNTHETIC_TEST_EFFECT':'0x1'}, cold_boot=True,
                  boot_id='synthetic-cold-boot-2',previous_boot_id='fake-boot-1')
        with self.journal() as j:
            receipt=verify_cold_snapshot(self.plan,cold,stage='test',journal=j)
        self.assertEqual(receipt['status'],'PASS')
        cold['fields']['SYNTHETIC_TEST_EFFECT']='0x0'
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError, 'differs'):
                verify_cold_snapshot(self.plan,cold,stage='test',journal=j)
        cold['boot_id']=cold['previous_boot_id']
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError, 'distinct'):
                verify_cold_snapshot(self.plan,cold,stage='test',journal=j)

    def test_real_requires_persistent_test_row_and_can_complete_simulated_record(self):
        self.run_test()
        cold=dict(pci_id=self.plan.pci_id, fpf_rows={"223":"0x13572468"},
                  die_rows={str(k):v for k,v in self.plan.die_rows.items()},
                  fields={"SYNTHETIC_TEST_EFFECT":"0x1"}, cold_boot=True,
                  boot_id="synthetic-boot-2", previous_boot_id="fake-boot-1")
        with self.journal() as j:
            receipt=verify_cold_snapshot(self.plan,cold,stage="test",journal=j)
        receipt_file=self.work / "test_receipt.json"
        receipt_file.write_text(json.dumps(receipt))
        receipt_file.chmod(0o600)
        with self.journal() as j:
            pointer=process_records(self.plan,self.controller,self.controller,j,
                                    stage="real",test_receipt=receipt_file,
                                    confirm=self.confirm_yes,show=lambda *_:None)
        self.assertEqual(pointer,225)
        self.assertEqual(self.controller.rows[224],self.plan.records[1].word)
        self.assertEqual(len([e for e in self.controller.events if e[0]=="write"]),2)
        # A stale but otherwise valid receipt is not enough if the test row changed.
        self.controller.rows[223]=0
        with self.journal() as j:
            with self.assertRaises(BurnError):
                process_records(self.plan,self.controller,self.controller,j,
                                stage="real",test_receipt=receipt_file,
                                confirm=self.confirm_yes,show=lambda *_:None)

    def test_pcie_requires_second_review_and_current_link_state(self):
        example = dict(name="SYNTHETIC_PCIE",target="BOOT_GEN3_DIS",word="0x11223344",
                       decode="SYNTHETIC",kind="pcie",defect_sensitive=False,
                       link_cap2_before="0x02",link_cap2_after="0x03")
        record = Record.parse(example)
        updated = replace(self.plan,records=(self.plan.records[0],record))
        self.run_test()
        self.controller.fields["LnkCap2"]=2
        receipt_file=self.work / "receipt"
        receipt_file.write_text(json.dumps({"status":"PASS","stage":"test",
             "plan_sha256":self.plan.content_hash,"test_word":"0x13572468","test_row":223}))
        receipt_file.chmod(0o600)
        prompts=[]
        def confirm(p,token):
            prompts.append(token)
            return token != "REVIEW PCIE"
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError,"special record not confirmed"):
                process_records(updated,self.controller,self.controller,j,
                    stage="real",confirm=confirm,test_receipt=receipt_file,show=lambda *_:None)
        self.assertEqual(prompts,["REVIEW PCIE"])
        self.assertEqual(self.controller.rows[224],0)
        self.controller.fields["LnkCap2"]=1
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError,"LnkCap2"):
                process_records(updated,self.controller,self.controller,j,
                    stage="real",confirm=self.confirm_yes,test_receipt=receipt_file,show=lambda *_:None)

    def test_automatic_debugctrl_forbidden_manual_second_confirmation(self):
        r=Record.parse(dict(name="SYNTHETIC_DEBUG",target="FUSE_DEBUGCTRL",word="0x10101010",
                            decode="SYNTHETIC",kind="regular",defect_sensitive=False))
        updated=replace(self.plan,records=(self.plan.records[0],r))
        self.run_test()
        receipt_file=self.work / "receipt"
        receipt_file.write_text(json.dumps({"status":"PASS","stage":"test",
             "plan_sha256":self.plan.content_hash,"test_word":"0x13572468","test_row":223}))
        receipt_file.chmod(0o600)
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError,"DEBUGCTRL"):
                process_records(updated,self.controller,self.controller,j,
                    stage="real",confirm=self.confirm_yes,test_receipt=receipt_file,
                    show=lambda *_:None)
        prompts=[]
        def deny_second(p,token):
            prompts.append(token)
            return token != "CONFIRM DEBUGCTRL"
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError,"declined"):
                process_records(updated,self.controller,self.controller,j,
                    stage="real",confirm=deny_second,test_receipt=receipt_file,
                    show=lambda *_:None,allow_manual_debugctrl=True)
        self.assertEqual(prompts,["REVIEW DEBUGCTRL","CONFIRM DEBUGCTRL"])
        self.assertEqual(self.controller.rows[224],0)

    def test_postwrite_driver_change_halts(self):
        self.controller.after_write=lambda c: setattr(c.machine,'nvidia_loaded',True)
        with self.assertRaisesRegex(BurnError,"HALT"):
            self.run_test()
        self.assertFalse(self.logged()[-1]['pointer_advanced'])

    def test_receipt_requires_private_permissions(self):
        self.run_test()
        r=self.work / "receipt"
        r.write_text(json.dumps({"status":"PASS","stage":"test",
            "plan_sha256":self.plan.content_hash,"test_word":"0x13572468","test_row":223}))
        r.chmod(0o644)
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError,"mode 0600"):
                process_records(self.plan,self.controller,self.controller,j,
                                stage="real",test_receipt=r,
                                confirm=self.confirm_yes,show=lambda *_:None)

    def test_cold_test_row_mismatch_refuses(self):
        self.run_test()
        cold=dict(pci_id=self.plan.pci_id, fpf_rows={"223":"0x13572469"},
                  die_rows={str(k):v for k,v in self.plan.die_rows.items()},
                  fields={"SYNTHETIC_TEST_EFFECT":"0x1"}, cold_boot=True,
                  boot_id="fake-boot-2",previous_boot_id="fake-boot-1")
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError,"does not persist"):
                verify_cold_snapshot(self.plan,cold,stage="test",journal=j)

    def test_reconfig_and_pcie_extra_conditions(self):
        r=replace(self.plan.records[1],kind='reconfig')
        with self.assertRaisesRegex(BurnError, 'RECONFIG'):
            reconcile(self.plan,r,self.plan.expected_masks.copy())
        r=replace(self.plan.records[1],kind='pcie')
        with self.assertRaisesRegex(BurnError, 'PCIe'):
            reconcile(self.plan,r,self.plan.expected_masks.copy())

    def test_pre_trigger_gate_aborts_if_frame_drops_during_confirmation(self):
        # Card looks healthy when prompt is shown, then drops PLM_f8 while operator types YES.
        def confirm_with_drop(prompt, token):
            self.controller.state.plm_f8 = 0xFFFFFFFE
            return True
        with self.assertRaisesRegex(BurnError, 'ABORTED before write'):
            self.run_test(confirm=confirm_with_drop)
        self.assertEqual(len([e for e in self.controller.events if e[0] == 'write']), 0)
        self.assertEqual(self.controller.rows[223], 0)
        self.assertEqual(self.logged()[-1]['event'], 'aborted_pre_trigger')
        self.assertFalse(self.logged()[-1]['write_issued'])

    def test_strict_confirm_requires_exact_row_and_word(self):
        seen = []
        def bad_confirm(prompt, token):
            seen.append(token)
            return token == 'YES'  # plain YES must NOT satisfy strict_confirm
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError, 'declined'):
                process_records(self.plan, self.controller, self.controller, j,
                                stage='test', confirm=bad_confirm, strict_confirm=True,
                                show=lambda *_: None)
        self.assertEqual(seen, ['YES 223 0x13572468'])
        self.assertEqual(self.controller.rows[223], 0)

    def test_multi_record_stepping_and_resume_across_sessions(self):
        r2 = Record.parse(dict(
            name='FAKE_COMPUTE_2', kind='regular', target='GPC_DISABLE',
            word='0x35792468', decode='second real record', defect_sensitive=True,
            mask_update=dict(field='GPC_DISABLE', from_='0x84', to='0x80') if False else {'field': 'GPC_DISABLE', 'from': '0x84', 'to': '0x80'},
            resulting_sm_count=90, expected_cold={'GPC_DISABLE': '0x80'},
        ))
        # Defect mask is 0x04; 0x84 -> 0x80 would touch 0x04! Use 0x84 -> 0x04 instead (clears bit 7).
        r2 = replace(r2, to_disable=0x04, expected_cold={'GPC_DISABLE': 0x04})
        multi_plan = replace(self.plan, records=(self.plan.records[0], self.plan.records[1], r2))
        with self.journal() as j:
            process_records(multi_plan, self.controller, self.controller, j,
                            stage='test', confirm=self.confirm_yes, show=lambda *_: None)
        receipt_file = self.work / 'test.receipt'
        receipt_file.write_text(json.dumps({
            'status': 'PASS', 'stage': 'test', 'plan_sha256': multi_plan.content_hash,
            'test_word': '0x13572468', 'test_row': 223,
        }))
        receipt_file.chmod(0o600)
        # Session 1: burn ONLY 1 real record (max_records=1).
        with self.journal() as j:
            ptr = process_records(multi_plan, self.controller, self.controller, j,
                                  stage='real', test_receipt=receipt_file,
                                  confirm=self.confirm_yes, max_records=1, show=lambda *_: None)
        self.assertEqual(ptr, 225)
        self.assertEqual(self.controller.rows[224], self.plan.records[1].word)
        self.assertEqual(self.controller.rows[225], 0)
        # Cold-boot effect: GPC_DISABLE is now 0x84 on the card!
        self.controller.fields['GPC_DISABLE'] = 0x84
        # Session 2: resumes cleanly at row 225 and burns record 2.
        with self.journal() as j:
            ptr2 = process_records(multi_plan, self.controller, self.controller, j,
                                   stage='real', test_receipt=receipt_file,
                                   confirm=self.confirm_yes, max_records=1, show=lambda *_: None)
        self.assertEqual(ptr2, 226)
        self.assertEqual(self.controller.rows[225], r2.word)

    def test_manual_override_slot_for_ambiguous_gap_still_refuses_non_empty(self):
        from vfi_burn.engine import Overrides
        self.controller.rows[225] = 0x55555555  # creates non-contiguous gap at 223, 224
        with self.assertRaisesRegex(BurnError, 'ambiguous'):
            scan_slots(self.plan, self.controller)
        # Refuses non-empty override slot even with Overrides!
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError, 'NOT empty'):
                process_records(self.plan, self.controller, self.controller, j,
                                stage='test', confirm=self.confirm_yes,
                                overrides=Overrides(slot=225), show=lambda *_: None)
        # Allows explicit empty slot 226 after typed OVERRIDE SLOT 226 confirmation.
        tokens = []
        def track_confirm(prompt, token):
            tokens.append(token)
            return True
        with self.journal() as j:
            ptr = process_records(self.plan, self.controller, self.controller, j,
                                  stage='test', confirm=track_confirm,
                                  overrides=Overrides(slot=226), show=lambda *_: None)
        self.assertEqual(ptr, 227)
        self.assertEqual(self.controller.rows[226], self.plan.records[0].word)
        self.assertEqual(tokens, ['OVERRIDE SLOT 226', 'YES'])

    def test_manual_overrides_skip_sense_no_receipt_vout_and_ceiling(self):
        from vfi_burn.engine import Overrides
        # 1. VOUT slightly outside tolerance (750 mV) allowed with override, but 300 mV floor NEVER allowed.
        self.controller.state.vout_mv = 750.0
        with self.assertRaisesRegex(BurnError, 'expected 787.5'):
            check_health(self.controller.state, self.plan)
        tokens = []
        def track(prompt, token):
            tokens.append(token)
            return True
        with self.journal() as j:
            process_records(
                self.plan, self.controller, self.controller, j,
                stage='real', confirm=track,
                overrides=Overrides(no_test_receipt=True, skip_active_sense=True, vout_range=True),
                show=lambda *_: None,
            )
        self.assertIn('OVERRIDE NO TEST RECEIPT', tokens)
        self.assertIn('OVERRIDE SKIP SENSE', tokens)
        self.assertIn('OVERRIDE VOUT 750.0', tokens)
        self.assertEqual([e for e in self.controller.events if e[0] == 'sense'], [])
        # Hard 400 mV rail floor is NEVER overrideable:
        self.controller.state.vout_mv = 300.0
        with self.assertRaisesRegex(BurnError, 'hard safety bounds'):
            check_health(self.controller.state, self.plan, allow_vout_override=True)
        # Genuine defect bits are NEVER overrideable even with undetermined_ceiling=True:
        bad_defect = replace(self.plan.records[1], from_disable=0x85, to_disable=0x80)
        with self.assertRaisesRegex(BurnError, 'genuinely defective'):
            reconcile(self.plan, bad_defect, self.plan.expected_masks.copy(),
                      allow_undetermined_ceiling=True)

    def test_pmbus_fault_bits_and_allowed_status_mask(self):
        for sw in (0x2042, 0x2842, 0x0040, 0x8000, 0x0800, 0x0004):
            with self.subTest(status_word=hex(sw)):
                c = copy.deepcopy(self.controller)
                c.state.status_word = sw
                with self.assertRaisesRegex(BurnError, 'STATUS_WORD'):
                    check_health(c.state, self.plan)
        # Benign non-critical bit allowed if explicitly declared in allowed_status_mask
        c = copy.deepcopy(self.controller)
        c.state.status_word = 0x0001
        relaxed = replace(self.plan, allowed_status_mask=0x0001)
        check_health(c.state, relaxed)
        # Critical OFF bit (0x0040) STILL refused even if in allowed_status_mask!
        c.state.status_word = 0x0040
        with self.assertRaisesRegex(BurnError, 'STATUS_WORD'):
            check_health(c.state, replace(self.plan, allowed_status_mask=0xFFFF))

    def test_plan_validation_rejects_empty_word_poison_word_dups_and_numeric_debugctrl(self):
        base = json.loads((ROOT / 'examples/plan.example.json').read_text())
        for label, mutate, regex in [
            ('empty_word', lambda d: d['records'][1].update(word='0x00000000'), 'empty_word'),
            ('poison_word', lambda d: d['records'][1].update(word='0xBADF1234'), 'poison prefix'),
            ('dup_name', lambda d: d['records'][1].update(name='FAKE_TEST'), 'duplicate record name'),
        ]:
            with self.subTest(case=label):
                d = copy.deepcopy(base)
                mutate(d)
                p = self.work / f'{label}.json'
                p.write_text(json.dumps(d))
                p.chmod(0o600)
                with self.assertRaisesRegex(BurnError, regex):
                    Plan.load(p)
        # Numeric DEBUGCTRL (0x820060 or decimal 8519776) rejected by check_plan immediately!
        for target in ('0x820060', '0x00820060', '8519776', 'FUSE_DEBUGCTRL'):
            r = replace(self.plan.records[1], target=target, defect_sensitive=False,
                        mask_field=None, from_disable=None, to_disable=None)
            with self.assertRaisesRegex(BurnError, 'DEBUGCTRL'):
                check_plan(replace(self.plan, records=(self.plan.records[0], r)))

    def test_reconcile_checks_sm_ceiling_even_without_mask_field(self):
        r = Record.parse(dict(
            name='NO_MASK_SM', kind='regular', target='SOME_FUSE',
            word='0x12345678', decode='exceeds base ceiling without mask_field',
            defect_sensitive=False, resulting_sm_count=125,
        ))
        with self.assertRaisesRegex(BurnError, 'beyond supplied ceiling'):
            reconcile(self.plan, r, self.plan.expected_masks.copy())

    def test_journal_exclusive_flock_prevents_concurrent_sessions(self):
        with self.journal() as j1:
            j1.append(event='first')
            with self.assertRaisesRegex(BurnError, 'locked by another'):
                Journal(self.journal_path)

    def test_verify_cold_final_expected_takes_precedence_and_checks_real_fpf_rows(self):
        self.run_test()
        cold_test = dict(
            pci_id=self.plan.pci_id, fpf_rows={'223': '0x13572468'},
            die_rows={str(k): v for k, v in self.plan.die_rows.items()},
            fields={'SYNTHETIC_TEST_EFFECT': '0x1'}, cold_boot=True,
            boot_id='boot-2', previous_boot_id='fake-boot-1',
        )
        with self.journal() as j:
            receipt = verify_cold_snapshot(self.plan, cold_test, stage='test', journal=j)
        rf = self.work / 'r.json'
        rf.write_text(json.dumps(receipt))
        rf.chmod(0o600)
        full_plan = replace(
            self.plan,
            final_expected_cold={
                'GPC_DISABLE': 0x84, 'FBP_DISABLE': 0, 'FBIO_LAYOUT': 1,
                'ROP_L2_LAYOUT': 1, 'FBPA_LAYOUT': 1, 'MAGIC_D': 1,
                'BOOT_GEN3_DIS': 0, 'DISABLE_GEN3_SPEED': 0, 'LnkCap2': 3, 'SM_COUNT': 75,
            },
        )
        with self.journal() as j:
            process_records(full_plan, self.controller, self.controller, j,
                            stage='real', test_receipt=rf, confirm=self.confirm_yes,
                            show=lambda *_: None)
        cold_real = dict(
            pci_id=full_plan.pci_id,
            fpf_rows={'223': '0x13572468', '224': '0x99999999'},  # wrong persisted real word!
            die_rows={str(k): v for k, v in full_plan.die_rows.items()},
            fields={**full_plan.final_expected_cold, 'SYNTHETIC_TEST_EFFECT': 1},
            cold_boot=True, boot_id='boot-3', previous_boot_id='fake-boot-1',
        )
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError, 'does not match written record'):
                verify_cold_snapshot(full_plan, cold_real, stage='real', journal=j)
        cold_real['fpf_rows']['224'] = '0x24681357'
        with self.journal() as j:
            res = verify_cold_snapshot(full_plan, cold_real, stage='real', journal=j)
        self.assertEqual(res['status'], 'PASS')

    def test_preflight_and_active_sense_failures_are_logged_with_health(self):
        self.controller.after_sense = lambda c: setattr(c.state, 'plm_f8', 0xFFFFFFFE)
        with self.assertRaisesRegex(BurnError, 'too deep'):
            self.run_test()
        entries = self.logged()
        self.assertEqual(entries[-1]['event'], 'preflight_failed')
        self.assertEqual(entries[-1]['phase'], 'initial_preflight')
        self.assertEqual(entries[-1]['health']['plm_f8'], 0xFFFFFFFE)

    def test_confirm_active_sense_gates_and_logs_sense(self):
        seen = []
        def deny_sense(prompt, token):
            seen.append(token)
            return False
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError, 'active sense declined'):
                process_records(
                    self.plan, self.controller, self.controller, j,
                    stage='test', confirm=deny_sense, confirm_active_sense=True,
                    show=lambda *_: None,
                )
        self.assertEqual(seen, ['SENSE'])
        self.assertEqual([e for e in self.controller.events if e[0] == 'sense'], [])
        self.assertEqual(self.logged()[-1]['event'], 'declined')
        self.assertEqual(self.logged()[-1]['reason'], 'SENSE')

        seen.clear()
        def accept_all(prompt, token):
            seen.append(token)
            return True
        with self.journal() as j:
            process_records(
                self.plan, self.controller, self.controller, j,
                stage='test', confirm=accept_all, confirm_active_sense=True,
                show=lambda *_: None,
            )
        self.assertEqual(seen, ['SENSE', 'YES'])
        self.assertTrue(any(e['event'] == 'active_sense_confirmed' for e in self.logged()))

    def test_verify_cold_early_rejection_is_logged(self):
        self.run_test()
        bad_cold = dict(
            pci_id=self.plan.pci_id, fpf_rows={'223': '0x13572469'},
            die_rows={str(k): v for k, v in self.plan.die_rows.items()},
            fields={'SYNTHETIC_TEST_EFFECT': '0x1'}, cold_boot=True,
            boot_id='boot-2', previous_boot_id='fake-boot-1',
        )
        with self.journal() as j:
            with self.assertRaisesRegex(BurnError, 'does not persist'):
                verify_cold_snapshot(self.plan, bad_cold, stage='test', journal=j)
        self.assertEqual(self.logged()[-1]['event'], 'cold_snapshot_rejected')
        self.assertEqual(self.logged()[-1]['status'], 'FAIL')


if __name__=='__main__': unittest.main()
