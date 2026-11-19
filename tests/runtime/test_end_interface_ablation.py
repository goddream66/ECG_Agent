"""Fixed-evidence interface ablation keeps future and unavailable evidence hidden."""
import importlib.util
import json
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / 'ops/server/ablate_end_decision.py'
SPEC = importlib.util.spec_from_file_location('ablate_end_decision', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeModel:
    def __init__(self, action):
        self.action = action
        self.last_raw = None

    def complete(self, messages, tools):
        self.input = json.loads(messages[-1]['content'])
        args = self.action
        return type('Reply', (), {'message': {'tool_calls': [{'function': {
            'name': 'submit_event_decision', 'arguments': json.dumps(args)}}]}, 'usage': {}})()


class EndInterfaceAblationTests(unittest.TestCase):
    def setUp(self):
        self.trace = [{'start_s': 0, 'end_s': 10, 'af_score': .2, 'afl_score': .1}]
        self.evidence = [dict(evidence_id='now', tool='score_ecgfounder', lead='ECG2',
                              window_s=[0, 10], available_at_s=10, status='ok',
                              result={'af_score': .2, 'afl_score': .1}),
                         dict(evidence_id='future', tool='score_ecgfounder', lead='ECG1',
                              window_s=[0, 10], available_at_s=20, status='ok',
                              result={'af_score': .1, 'afl_score': .1})]

    def test_limited_view_keeps_primary_and_hides_future(self):
        v = MODULE.make_view(10, self.trace, self.evidence, 'unified_limited')
        self.assertEqual([e['evidence_id'] for e in v['measured_evidence']], ['now'])

    def test_close_cannot_cite_future(self):
        model = FakeModel(dict(decision='CLOSE', event_id=1, end_s=10,
                               evidence_ids=['future'], reason='test'))
        row = MODULE.decide(model, 10, self.trace, self.evidence, 'unified_full')
        self.assertIn('unavailable_evidence_id', row['error'])

    def test_renamed_end_action_is_parsed_as_close(self):
        model = FakeModel(dict(event_id=1, end_s=10, evidence_ids=['E1'], reason='end'))
        original_complete = model.complete
        def renamed_complete(messages, tools):
            reply = original_complete(messages, tools)
            reply.message['tool_calls'][0]['function']['name'] = 'end_confirmed_event'
            return reply
        model.complete = renamed_complete
        row = MODULE.decide(model, 10, self.trace, self.evidence, 'renamed')
        self.assertEqual(row['action'], 'CLOSE')
        self.assertEqual(row['resolved_evidence_ids'], ['now'])


if __name__ == '__main__':
    unittest.main()
