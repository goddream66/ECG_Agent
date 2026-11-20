import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ops' / 'server'))
import fixed_transition_p35 as p35


class EvidenceAblationTests(unittest.TestCase):
    def test_scores_stop_at_cursor_and_missing_is_not_negative(self):
        index = {(lead, t, t + 10): {'status': 'ok', 'waveform_row': 0}
                 for lead in ('ECG1', 'ECG2') for t in range(10, 70, 10)}
        index['ECG2', 60, 70] = {'status': 'insufficient', 'waveform_row': None}
        result = p35.score_rows({'cursor_s': 70}, index, np.array([[.9, .1]]))
        self.assertEqual(len(result), 12)
        self.assertEqual(result[-1]['window_s'], [60, 70])
        self.assertIsNone(result[-1]['result'])

    def test_arms_keep_scores_fixed_and_expose_only_requested_evidence(self):
        rr = {'id': 'R1', 'status': 'ok', 'lead': 'ECG1', 'window_s': [50, 60],
              'result': {'rr_intervals_ms': [800, 900, 1000], 'cleaning_keep_mask': [True] * 3}}
        case = {'cursor_s': 60, 'scores': [{'id': 'S1'}], 'xqrs_rr': [rr],
                'quality_hints': [{'id': 'Q1'}], 'reference': 'negative'}
        for arm, has_rr, has_quality, has_mask in zip(
                p35.ARMS, (0, 1, 1, 1, 1), (0, 0, 1, 0, 1), (0, 0, 0, 1, 1)):
            view = p35.prompt_view(case, arm)
            self.assertNotIn('reference', json.dumps(view))
            self.assertEqual(view['ecgfounder_scores'], case['scores'])
            self.assertEqual(len(view['rr_measurements']), has_rr)
            self.assertEqual(len(view['quality_hints']), has_quality)
            self.assertEqual(bool(view['rr_measurements'] and
                                  'rr_cleaning_keep_mask' in view['rr_measurements'][0]), has_mask)
