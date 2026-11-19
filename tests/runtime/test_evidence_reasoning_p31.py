"""P3-1 matched evidence and intervention checks."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ops' / 'server'))
import evidence_reasoning_p31 as probe


def score(evidence_id, start, value):
    return {'id': evidence_id, 'status': 'ok', 'lead': 'ECG1',
            'window_s': [start, start + 10],
            'result': {'af_score': value, 'afl_score': 0.0}}


class ProbeTests(unittest.TestCase):
    def test_alignment_preserves_measurements_and_intervention_is_isolated(self):
        original = [score('E4', 0, .2), score('E5', 10, .9), score('E6', 20, .2),
                    score('E12', 20, .9)]
        aligned = probe.view(original, 'aligned')
        recovered = [item for window in aligned['windows'] for item in window['measurements']]
        self.assertEqual(recovered, original)
        changed, change = probe.intervention(original, 'backup')
        self.assertEqual(change['changed_id'], 'E12')
        self.assertEqual(original[-1]['result']['af_score'], .9)
        self.assertEqual(changed[-1]['result']['af_score'], .1)
        changed, change = probe.intervention(original, 'recovery')
        self.assertEqual(change['changed_id'], 'E5')
        self.assertEqual(changed[1]['result']['af_score'], .1)

    def test_citation_validation_rejects_unseen_id(self):
        answer = {'decision': 'UNCERTAIN', 'support_transition_ids': [],
                  'against_transition_ids': ['E99'], 'unresolved': '', 'reason': 'conflict'}
        reply = type('Reply', (), {'message': {'tool_calls': [{
            'function': {'name': 'judge_transition', 'arguments': json.dumps(answer)}}]}})()
        with self.assertRaises(ValueError):
            probe.validate(reply, {'E1'})


if __name__ == '__main__':
    unittest.main()
