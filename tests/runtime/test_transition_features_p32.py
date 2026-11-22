"""The exploratory feature probe reports unavailable evidence and temporal change."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ops' / 'server'))
from probe_transition_features_p32 import block_features, matched_peaks, probe_record


class ProbeTests(unittest.TestCase):
    def test_flat_signal_is_unusable(self):
        self.assertEqual(block_features(np.zeros(1000), 100)['reason'], 'near_flat')

    def test_rr_change_is_measurable_without_diagnosis(self):
        def pulse(interval):
            x = np.zeros(1000)
            for index in range(10, 1000, interval):
                x[index:index+3] = [0.2, 1.0, 0.2]
            return x
        faster = block_features(pulse(50), 100)
        slower = block_features(pulse(100), 100)
        self.assertTrue(faster['usable'])
        self.assertTrue(slower['usable'])
        self.assertLess(faster['median_rr_s'], slower['median_rr_s'])
        self.assertEqual(len(faster['raw_rr_s']), len(faster['rr_keep_mask']))
        self.assertEqual(len(faster['candidate_peak_offsets_s']), faster['candidate_peak_count'])

    def test_cross_lead_matching_is_one_to_one(self):
        result = matched_peaks([1.0, 1.04, 2.0], [1.01, 2.01])
        self.assertEqual(result['matched'], 2)
        self.assertAlmostEqual(result['agreement_fraction'], .8)

    def test_record_reader_stops_at_cursor(self):
        fs = 100
        signal = np.zeros((20 * fs, 2))
        for index in range(10, len(signal), 100):
            signal[index:index+3, :] = np.array([.2, 1., .2])[:, None]
        calls = []
        def read(path, sampfrom, sampto):
            calls.append((sampfrom, sampto))
            return SimpleNamespace(p_signal=signal, sig_name=['ECG1', 'ECG2'])
        with patch.dict(sys.modules, {'wfdb': SimpleNamespace(
                rdheader=lambda path: SimpleNamespace(fs=fs), rdrecord=read)}):
            result = probe_record(Path('record'), 40)
        self.assertEqual(calls, [(2000, 4000)])
        self.assertEqual(len(result), 2)
        self.assertIn('cross_lead_agreement', result[0]['after'])


if __name__ == '__main__':
    unittest.main()
