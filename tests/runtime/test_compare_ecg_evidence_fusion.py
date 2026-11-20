from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ops' / 'server'))
import compare_ecg_evidence_fusion as fusion


class EvidenceFusionTests(unittest.TestCase):
    def test_arms_share_exact_features_and_exclude_reference(self):
        case = {'record_id': 'r', 'reference': 1, 'cursor_s': 60,
                'scores': [{'window_s': [0, 60], 'features': {'a': .3}}],
                'traditional': [{'window_s': [50, 60], 'features': {'rr_cv_raw': .4, 'rr_cv_clean': .1}}]}
        for group in fusion.FEATURE_GROUPS:
            observed = fusion.view(case, group)
            self.assertNotIn('reference', observed)
            self.assertEqual(fusion.vector(case, group), [v for k in fusion.FEATURE_GROUPS[group]
                for row in observed[k] for _, v in sorted(row['features'].items())])
        self.assertEqual(fusion.vector(case, 'fusion'), [.3, .1, .4])

    def test_measurement_reads_only_arrived_samples_and_preserves_missing(self):
        raw = SimpleNamespace(p_signal=np.zeros((7500, 2)), fs=250,
                              sig_name=('ECG1', 'ECG2'), units=('mV', 'mV'))
        with patch.object(fusion.wfdb, 'rdrecord', return_value=raw) as read:
            result = fusion.measure_window('record', 250, 60)
        read.assert_called_once_with('record', sampfrom=7500, sampto=15000)
        for lead in result:
            self.assertEqual(lead['rr_status'], 'insufficient')
            self.assertIsNone(lead['rr_cv_raw'])
            self.assertEqual(lead['raw_rr_ms'], [])
