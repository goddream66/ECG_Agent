import importlib.util
import json
from pathlib import Path
import sys
import unittest


SERVER = Path(__file__).resolve().parents[2] / 'ops' / 'server'
sys.path.insert(0, str(SERVER))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SERVER / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p33 = load('evaluate_peak_detection_p33')
p34 = load('fixed_transition_p34')


class PeakAuditTests(unittest.TestCase):
    def test_one_to_one_and_rr_require_adjacent_matched_beats(self):
        reference = [1., 2., 3., 4.]
        predicted = [1.01, 2.01, 2.02, 4.01]
        result = p33.match_peaks(reference, predicted, tolerance=.05)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (3, 1, 1))
        self.assertEqual(len(result['rr_errors_ms']), 1)

    def test_fixed_views_change_rr_source_and_quality_only(self):
        base = {'id': 'R1', 'lead': 'ECG1', 'window_s': [10., 20.], 'status': 'ok'}
        case = {'cursor_s': 20., 'reference': 'negative', 'primary_scores': [
            {'id': 'E1', 'status': 'ok', 'lead': 'ECG1', 'window_s': [10., 20.],
             'result': {'af_score': .1, 'afl_score': .1}}],
            'original_rr': [dict(base, result={'rr_intervals_ms': [800.],
                                               'raw': {'mean_ms': 800., 'cv': None}})],
            'xqrs_rr': [dict(base, result={'rr_intervals_ms': [1000.],
                                           'raw': {'mean_ms': 1000., 'cv': None},
                                           'cleaning_keep_mask': [True],
                                           'removed_fraction': 0.})],
            'quality_hints': [{'id': 'Q1', 'status': 'ok', 'lead': 'ECG1',
                               'window_s': [10., 20.], 'result': {'score': .8}}]}
        views = [p34.prompt_view(case, arm) for arm in p34.ARMS]
        for view in views:
            self.assertNotIn('reference', json.dumps(view))
            self.assertEqual(view['primary_scores'], views[0]['primary_scores'])
        self.assertEqual([v['rr_measurements'][0]['mean_rr_ms'] for v in views],
                         [800., 1000., 1000.])
        self.assertEqual([len(v['quality_hints']) for v in views], [0, 0, 1])
        self.assertNotIn('rr_cleaning_keep_mask', views[1]['rr_measurements'][0])
        self.assertEqual(views[2]['rr_measurements'][0]['rr_cleaning_keep_mask'], [True])


if __name__ == '__main__':
    unittest.main()
