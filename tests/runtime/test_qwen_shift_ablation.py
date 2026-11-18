import json
import unittest

from cardioagent.ecg_agent.experiments.qwen_shift_ablation import review, shifted_observations
from cardioagent.ecg_agent.harness.shifted_evidence_review import ShiftedEvidenceReviewer
from cardioagent.ecg_agent.llm.contracts import ModelReply


class RecordingModel:
    def complete(self, messages, tools):
        self.state = json.loads(messages[-1]['content'])
        return ModelReply({'role': 'assistant', 'tool_calls': [{'function': {
            'name': 'choose_measurement', 'arguments': {'action': 'A1'}}}]})


class ShiftedEvidenceTests(unittest.TestCase):
    def test_shift_is_context_and_does_not_fill_original_evidence(self):
        packet = dict(gap_s=[160, 170], cursor_s=180, start_threshold=.6, left={}, right={},
                      evidence_index={'e1': 'low'}, windows=[dict(window_s=[160, 170], decision='negative',
                          leads={'ECG1': dict(score=[.2, .1], score_status='ok', score_id='e1', quality='no_warning'),
                                 'ECG2': dict(score=None, score_status='not_measured', quality='not_measured')})])
        raw = dict(window_s=[160, 170], shifted_s=[165, 175], gap_s=[160, 170], cursor_s=180,
                   lead='ECG1', offset_s=5, original=[.2, .1], shifted=[.95, .1], status='ok', evidence_id='shift')
        shifts = shifted_observations(packet, [raw])
        model = RecordingModel()
        evidence = {(180, 160, 170, 'ECG2', 'score_ecgfounder'):
                    dict(window_s=[160, 170], lead='ECG2', tool='score_ecgfounder', status='ok',
                         result={'af_score': .1, 'afl_score': .1}, evidence_id='new-low')}
        result = review(packet, evidence, shifts, ShiftedEvidenceReviewer(model), 2)
        self.assertEqual(model.state['shifted_observations'][0]['outside_gap_s'], 5)
        self.assertEqual(result['calls'], 1)
        self.assertEqual(result['support_ids'], [])
        with self.assertRaises(ValueError):
            shifted_observations(dict(packet, cursor_s=172), [dict(raw, cursor_s=172)])
