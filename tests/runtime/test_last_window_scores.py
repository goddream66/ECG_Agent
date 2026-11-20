from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ops' / 'server'))
import check_last_window_scores as check


def example():
    return {'cursor_s': 60, 'reference': 'negative', 'scores': [
        {'id': f'{end}-{lead}', 'window_s': [end - 10, end], 'lead': lead, 'status': 'ok',
         'result': {'af_score': .1 if lead == 'ECG1' and end == 60 else .8,
                    'afl_score': .05 if lead == 'ECG1' and end == 60 else .4}}
        for end in range(10, 61, 10) for lead in check.CHANNELS]}


class LastWindowTests(unittest.TestCase):
    def test_only_final_scores_change_and_input_is_immutable(self):
        case = example()
        before = deepcopy(case)
        for arm in check.ARMS:
            view = check.intervention(case, arm)
            scores = view['ecgfounder_scores']
            self.assertEqual(scores[:10], case['scores'][:10])
            self.assertNotIn('reference', view)
            self.assertEqual(view['rr_measurements'], [])
            for changed, original in zip(scores[-2:], case['scores'][-2:]):
                self.assertEqual({k: v for k, v in changed.items() if k != 'result'},
                                 {k: v for k, v in original.items() if k != 'result'})
        self.assertEqual(check.intervention(case, 'repeat_previous')['ecgfounder_scores'][-2]['result'],
                         case['scores'][-4]['result'])
        capped = check.intervention(case, 'cap_at_020')['ecgfounder_scores']
        self.assertEqual(capped[-2]['result'], case['scores'][-2]['result'])
        self.assertEqual(capped[-1]['result'], {'af_score': .2, 'afl_score': .1})
        self.assertEqual(case, before)

    def test_rejects_future_or_unmeasured_final_score(self):
        case = example()
        case['scores'][-1]['window_s'] = [60, 70]
        with self.assertRaises(ValueError):
            check.intervention(case, 'repeat_previous')
        case = example()
        case['scores'][-1].update(status='insufficient', result=None)
        with self.assertRaises(ValueError):
            check.intervention(case, 'cap_at_020')
