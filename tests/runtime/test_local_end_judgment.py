"""Causal fixed-evidence end-judgment probe."""
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / 'ops/server/evaluate_local_end_judgment.py'
SPEC = importlib.util.spec_from_file_location('local_end_judgment', SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class FakeModel:
    def __init__(self, answer):
        self.answer = answer

    def complete(self, messages, tools):
        import json
        self.view = json.loads(messages[-1]['content'])
        return type('Reply', (), {'message': {'tool_calls': [{'function': {
            'name': 'judge_event_end', 'arguments': json.dumps(self.answer)}}]}, 'usage': {}})()


class LocalEndJudgmentTest(unittest.TestCase):
    def setUp(self):
        self.trace = [dict(start_s=0, end_s=10, decision='support', state='active',
                           af_score=.9, afl_score=.1)]
        self.evidence = [dict(evidence_id='arrived', tool='score_ecgfounder', lead='ECG1',
                              window_s=[0, 10], available_at_s=10, status='ok',
                              result={'af_score': .9, 'afl_score': .1}),
                         dict(evidence_id='future', tool='score_ecgfounder', lead='ECG2',
                              window_s=[0, 10], available_at_s=20, status='ok',
                              result={'af_score': .1, 'afl_score': .1})]

    def test_future_measurement_is_hidden(self):
        model = FakeModel(dict(action='continue', end_s=None,
                               evidence_ids=['arrived'], reason='support'))
        row = PROBE.evaluate_case(model, 10, self.trace, self.evidence)
        self.assertEqual(row['answer']['action'], 'continue')
        self.assertEqual([item['evidence_id'] for item in model.view['measured_evidence']],
                         ['arrived'])

    def test_close_cannot_cite_future_measurement(self):
        model = FakeModel(dict(action='close', end_s=10,
                               evidence_ids=['future'], reason='end'))
        row = PROBE.evaluate_case(model, 10, self.trace, self.evidence)
        self.assertEqual(row['error'], 'ValueError')


if __name__ == '__main__':
    unittest.main()
