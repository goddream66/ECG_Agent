"""P3-0 evidence visibility and label isolation."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

server = Path(__file__).resolve().parents[2] / 'ops' / 'server'
sys.path.insert(0, str(server))
spec = importlib.util.spec_from_file_location('fixed_transition_p3_0', server / 'fixed_transition_p3_0.py')
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)


class Model:
    def __init__(self):
        self.messages = None
        self.tools = None

    def complete(self, messages, tools):
        self.messages, self.tools = messages, tools
        return type('Reply', (), {'message': {'tool_calls': [{
            'function': {'name': 'judge_transition', 'arguments': json.dumps({
                'decision': 'UNCERTAIN', 'evidence_ids': ['E1'], 'reason': 'insufficient'})}}]}})()


class P3Tests(unittest.TestCase):
    def setUp(self):
        self.case = {'record_id': 'x', 'cursor_s': 20, 'reference': 'negative',
                     'primary': [{'id': 'E1', 'status': 'ok', 'window_s': [10, 20],
                                  'source_evidence_id': 'real-primary'}],
                     'backup': [{'id': 'E7', 'status': 'ok', 'window_s': [10, 20]}],
                     'rr': [{'id': 'E19', 'status': 'insufficient'}],
                     'quality': [{'id': 'E13', 'status': 'ok'}],
                     'wave': [{'id': 'E21', 'status': 'error'}]}

    def test_arms_add_only_requested_evidence(self):
        expected = {'A': {'cursor_s', 'primary_scores'},
                    'B': {'cursor_s', 'primary_scores', 'backup_scores'},
                    'C': {'cursor_s', 'primary_scores', 'rr_measurements'},
                    'D': {'cursor_s', 'primary_scores', 'quality_hints', 'local_waveform_proxies'},
                    'E': {'cursor_s', 'primary_scores', 'backup_scores', 'rr_measurements',
                          'quality_hints', 'local_waveform_proxies'}}
        for arm, keys in expected.items():
            self.assertEqual(set(p3.payload(self.case, arm)), keys)
            self.assertNotIn('reference', json.dumps(p3.payload(self.case, arm)))

    def test_same_decision_tool_and_no_reference_leak(self):
        for arm in p3.ARMS:
            model = Model()
            result = p3.judge(model, self.case, arm)
            self.assertEqual(result['answer']['resolved_evidence_ids'], ['real-primary'])
            self.assertEqual([t['function']['name'] for t in model.tools], ['judge_transition'])
            self.assertNotIn('negative', model.messages[1]['content'])


if __name__ == '__main__':
    unittest.main()
