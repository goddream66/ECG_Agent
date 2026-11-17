import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from cardioagent.ecg_agent.benchmarks.event_evaluate import (
    aggregate, evaluate_record, intersection, paired_bootstrap, time_metrics, prepare_validity)


class EventEvaluationTests(unittest.TestCase):
    def ref(self, intervals):
        return {'start_s': 0., 'end_s': 100., 'intervals': intervals, 'tau': .5,
                'burden': sum(b-a for a,b in intervals)/100}

    def test_fragments_do_not_disappear_and_merge_is_penalized(self):
        row = evaluate_record('a', [[0,20],[30,70]], self.ref([[0,80]]))
        self.assertEqual(row['segments_per_reference'], [2])
        self.assertEqual(row['long_fragment_excess'], 1)
        self.assertEqual(row['fp'], 1)
        merged = evaluate_record('a', [[0,70]], self.ref([[0,20],[50,70]]))
        self.assertEqual(merged['merged_predictions'], 1)
        self.assertEqual(merged['merge_crossed_non_af_s'], 30)

    def test_unknown_is_not_silently_removed_from_error(self):
        row = time_metrics([[0,60]], [[0,40]], [[0,100]], [[40,60]])
        self.assertEqual(row['time_fp_s'], 20)
        self.assertEqual(row['unknown_coverage'], .2)
        self.assertEqual(row['conditional_known_time_error'], 0)
        self.assertEqual(row['unconditional_time_error'], .2)

    def test_unlabeled_time_is_not_normal_and_zero_denominators_are_null(self):
        row = time_metrics([[0,10]], [], [[20,100]])
        self.assertEqual(row['time_fp_s'], 0)
        self.assertIsNone(row['time_precision'])
        row = evaluate_record('a', [], self.ref([[0,80]]))
        self.assertEqual(row['long_missed'], 1)
        self.assertIsNone(aggregate([row])['merge_rate'])

    def test_censored_ends_are_not_zero_offset_error(self):
        event = {'right_censored': True, 'reported_intervals': [[0,100]], 'termination_reason': 'stream_end'}
        row = evaluate_record('a', [[0,100]], self.ref([[0,100]]), events=[event])
        self.assertEqual(row['offset_error_n'], 0)

    def test_merge_gap_excludes_outer_tails_and_overlapping_predictions_do_not_add_gaps(self):
        row=evaluate_record('a',[[0,100]],self.ref([[10,20],[50,70]]))
        self.assertEqual(row['merge_crossed_non_af_s'],30)
        row=evaluate_record('a',[[0,50],[40,80]],self.ref([[0,80]]))
        self.assertEqual(row['long_fragment_excess'],1)
        self.assertEqual(row['long_interruptions'],0)

    def test_one_confirmation_is_not_two_independent_alarms(self):
        updates=[{'recorded_at_s':30,'event_update':{'kind':'confirm','interval':[0,30],'recorded_at_s':30}}]
        row=evaluate_record('a',[[0,30]],self.ref([[0,10],[20,30]]),updates=updates)
        self.assertEqual(row['alarm_delay_n'],1)
        self.assertEqual(row['unalarmed_references'],1)

    def test_delay_uses_actual_confirmation_and_unalarmed_events_remain(self):
        updates = [{'recorded_at_s': 20, 'event_update': {'kind':'confirm','interval':[0,20],'recorded_at_s':20}}]
        row = evaluate_record('a', [[0,20]], self.ref([[0,70],[80,90]]), updates=updates)
        self.assertEqual(row['alarm_delay_sum_s'], 20)
        self.assertEqual(row['unalarmed_references'], 1)
        self.assertEqual(row['compatibility_delay_n'], 0)

    def test_bootstrap_is_paired_and_reproducible(self):
        a = evaluate_record('a', [[0,20]], self.ref([[0,20]]))
        b = evaluate_record('b', [], self.ref([[0,20]]))
        one = paired_bootstrap([a,b], [a,b], repetitions=20)
        self.assertEqual(one, paired_bootstrap([a,b], [a,b], repetitions=20))
        self.assertEqual(one['differences_new_minus_baseline']['micro_episode_f1']['ci95'], [0.,0.])

    def test_raw_annotation_validity_does_not_turn_unknown_labels_into_normal(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'00001.atr').write_bytes(b'fixture')
            task={'task_id':'t','record_id':'00001','start_s':0.,'end_s':100.}
            (root/'tasks.jsonl').write_text(json.dumps(task)+'\n',encoding='utf-8')
            (root/'00001.windows.jsonl').write_text(json.dumps({'channel':'ECG1','invalid_spans':[]})+'\n',encoding='utf-8')
            header=SimpleNamespace(fs=250,sig_len=25000)
            ann=SimpleNamespace(sample=[2500,5000,7500,10000],aux_note=['(AFIB','(NOISE','(N','(AFL'])
            with patch('cardioagent.ecg_agent.benchmarks.event_evaluate.wfdb.rdheader',return_value=header), \
                 patch('cardioagent.ecg_agent.benchmarks.event_evaluate.wfdb.rdann',return_value=ann):
                result=prepare_validity(root/'tasks.jsonl',root,root,root/'validity.json')['t']
            self.assertEqual(result['valid_intervals'],[[10.,20.],[30.,100.]])
            self.assertEqual(result['raw_annotation_af_intervals'],[[10.,20.],[40.,100.]])
            self.assertEqual(result['unlabeled_s'],20.)
