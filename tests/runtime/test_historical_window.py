import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from cardioagent.ecg_agent.experiments.original_window_review import action_plan, support_refs, QUALITY
from cardioagent.ecg_agent.harness.historical_window import HistoricalWindow
from cardioagent.ecg_agent.schemas import EvidenceContext


class HistoricalWindowTests(unittest.TestCase):
    def test_shifted_score_reads_correct_samples_and_rejects_future(self):
        task = {'record_id': '05121', 'start_s': 0, 'end_s': 300}
        captured = []
        reader = HistoricalWindow('.', task, lambda values: (captured.append(values) or .8, .1))
        context = EvidenceContext('test', 'task', '05121')

        def read_window(stem, *, sampfrom, sampto):
            signal = np.sin(np.arange(sampfrom, sampto) / 10)
            return SimpleNamespace(fs=250, sig_name=['ECG1', 'ECG2'], units=['mV', 'mV'],
                                   p_signal=np.column_stack([signal, signal]))

        with patch('cardioagent.ecg_agent.harness.historical_window.wfdb.rdrecord', side_effect=read_window) as read:
            with self.assertRaises(ValueError):
                reader.shifted_score(160, 170, 170, 'ECG1', 5, context)
            with self.assertRaises(ValueError):
                reader.shifted_score(160, 170, 180, 'ECG1', 3, context)
            read.assert_not_called()
            result = reader.shifted_score(160, 170, 180, 'ECG1', 5, context)
            read.assert_called_once_with(reader.stem, sampfrom=41250, sampto=45000)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['window_s'], [165, 175])
        self.assertEqual(result['available_at_s'], 180)
        self.assertEqual(result['result']['original_window_s'], [160, 170])
        self.assertEqual(result['result']['overlap_original_s'], [165, 170])
        self.assertEqual(captured[0].shape, (5000,))

    def test_historical_evidence_available_only_at_review_cursor(self):
        signal = np.column_stack([np.sin(np.arange(12500) / 10), np.zeros(12500)])
        task = {'record_id': '05121', 'start_s': 0, 'end_s': 300}
        reader = HistoricalWindow('.', task, lambda values: (.8, .1))
        context = EvidenceContext('test', 'task', '05121')
        with patch('cardioagent.ecg_agent.harness.historical_window.wfdb.rdrecord') as read:
            with self.assertRaises(ValueError):
                reader.session(160, 170, 165, context)
            read.assert_not_called()
            read.return_value = SimpleNamespace(fs=250, sig_name=['ECG1', 'ECG2'], units=['mV', 'mV'], p_signal=signal)
            session = reader.session(160, 170, 210, context)
            read.assert_called_once_with(reader.stem, sampfrom=40000, sampto=52500)
            args = {'lead': 'ECG1', 'start_s': 160, 'end_s': 170}
            result = session.call('score_ecgfounder', args)
            self.assertEqual((result['status'], result['window_s'], result['available_at_s']), ('ok', [160, 170], 210))
            self.assertEqual(session.call('score_ecgfounder', dict(args, lead='ECG2'))['status'], 'insufficient')
            self.assertEqual(session.call('assess_quality', dict(args, start_s=205, end_s=215))['status'], 'error')

    def test_missing_quality_can_add_support_but_warning_cannot(self):
        window = {'window_s': [160, 170], 'leads': {'ECG1': {
            'score': [.85, .1], 'score_id': 'e1', 'score_status': 'ok', 'quality': 'not_measured'}}}
        packet = {'windows': [window], 'gap_s': [160, 170], 'start_threshold': .6,
                  'evidence_index': {'e1': 'base-score'}}
        key, = action_plan([window], .6)
        self.assertEqual(key, (160, 170, 'ECG1', QUALITY))
        self.assertEqual(support_refs(packet, {}), [])
        for reasons, expected in ((['flat_signal'], []), ([], ['base-score', 'supplement'])):
            added = {key: dict(status='ok', result={'reasons': reasons}, evidence_id='supplement')}
            self.assertEqual(support_refs(packet, added), expected)
