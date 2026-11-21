"""Focused smoke tests for the optional agent-authored evidence investigation checkpoint.

Run from project root with: PYTHONPATH=src python -m unittest tests.runtime.test_evidence_focus_checkpoint
No ECG data, model weights, or GPU is required.
"""
import json
import unittest
from types import SimpleNamespace

from cardioagent.ecg_agent.harness.react import ReActReview
from cardioagent.ecg_agent.llm.contracts import ModelReply
from cardioagent.ecg_agent.schemas import CachedScoreArgs, WindowArgs


def call(name, args):
    return ModelReply({'role': 'assistant', 'tool_calls': [{
        'function': {'name': name, 'arguments': json.dumps(args)}}]},
        {'prompt_tokens': 1, 'completion_tokens': 1})


class Script:
    def __init__(self, actions):
        self.actions = list(actions)
        self.seen = []
        self.final_payload = None

    def complete(self, messages, tools):
        names = [t['function']['name'] for t in tools]
        name, args = self.actions.pop(0)
        self.seen.append((name, names))
        if names == ['submit_event_decision']:
            self.final_payload = json.loads(messages[-1]['content'])
        assert name in names, (name, names)
        return call(name, args)


class Session:
    def __init__(self):
        self.max_calls = 12
        self.calls = []
        self.context = SimpleNamespace(origin_s=1770., waveform=SimpleNamespace(channels=('ECG1', 'ECG2')))
        self.evidence_context = SimpleNamespace(run_id='run', task_id='task', record_id='04936')
        self.registry = SimpleNamespace(
            _tools={'score_ecgfounder': SimpleNamespace(arguments=CachedScoreArgs),
                    'detect_beats': SimpleNamespace(arguments=WindowArgs)},
            function_schemas=lambda **kw: [{'type': 'function', 'function': {
                'name': name, 'parameters': {'type': 'object'}, 'description': 'test'}}
                for name in ('score_ecgfounder', 'detect_beats')])

    def call_window(self, tool, args):
        n = len(self.calls) + 1
        row = {'evidence_id': f'evidence-{n}', 'tool': tool, 'lead': args['lead'],
               'window_s': [args['start_s'], args['end_s']], 'status': 'ok',
               'available_at_s': 1780.,
               'result': {'af_score': .91, 'afl_score': .03} if tool == 'score_ecgfounder'
               else {'r_peaks_s': [1771., 1772., 1773.]}, 'warnings': [], 'error': None}
        self.calls.append(row)
        return row


FOCUS = dict(issue='possible_onset', observation='Current model evidence supports possible onset',
             evidence_ids=['evidence-1'],
             unresolved_question='Are current beats consistent with the reported score?',
             next_tool='detect_beats', target_lead='ECG1',
             target_start_s=1770., target_end_s=1780.,
             expected_information='Beat-level timing may clarify rhythm interpretation')


class FocusTests(unittest.TestCase):
    def run_case(self, actions, enable=True):
        model = Script(actions)
        session = Session()
        logs = []
        reviewer = ReActReview(model, max_rounds=12, history_windows=6,
                               boundary_tools=True, event_control=True,
                               unified_event_decision=True, separate_final_decision=True,
                               llm_window_interpretation=True, investigation_checkpoint=enable)
        tracker = SimpleNamespace(state='IDLE', active=None, support_count=0,
                                  negative_count=0, unknown_start=None)
        decision = reviewer.review(None, None, 'ECG2', tracker, None, session,
                                   lambda name, row: logs.append((name, row)),
                                   call_window=session.call_window,
                                   window_s=(1770., 1780.), primary_lead='ECG1')
        return decision, reviewer, model, logs

    def test_checkpoint_gates_second_tool_then_resumes(self):
        actions = [
            ('score_ecgfounder', dict(lead='ECG1', start_s=1770., end_s=1780.)),
            ('record_evidence_focus', FOCUS),
            ('detect_beats', dict(lead='ECG1', start_s=1770., end_s=1780.)),
            ('finish_measurements', {}),
            ('submit_event_decision', dict(action='KEEP_OPEN', interpretation='support',
                                           event_id=None, end_s=None, evidence_ids=['E1', 'E2'],
                                           reason='Grounded current measurements')),
        ]
        decision, reviewer, model, logs = self.run_case(actions)
        self.assertEqual(decision[0], 'support')
        self.assertNotIn('record_evidence_focus', model.seen[0][1])
        self.assertEqual(model.final_payload['evidence_focus'], FOCUS)
        self.assertEqual(model.seen[1][1], ['record_evidence_focus'])
        self.assertEqual(reviewer.evidence_focus['next_tool'], 'detect_beats')
        self.assertTrue(any(name == 'llm_steps' and row.get('status') == 'focus_recorded'
                            for name, row in logs))

    def test_rejects_forged_evidence_and_allows_correction(self):
        forged = dict(FOCUS, evidence_ids=['fabricated-id'])
        actions = [
            ('score_ecgfounder', dict(lead='ECG1', start_s=1770., end_s=1780.)),
            ('record_evidence_focus', forged),
            ('record_evidence_focus', FOCUS),
            ('finish_measurements', {}),
            ('submit_event_decision', dict(action='KEEP_OPEN', interpretation='support',
                                           event_id=None, end_s=None, evidence_ids=['E1'],
                                           reason='Measured current AF/AFL evidence')),
        ]
        decision, reviewer, model, logs = self.run_case(actions)
        self.assertEqual(decision[0], 'support')
        self.assertEqual(model.seen[2][1], ['record_evidence_focus'])
        self.assertTrue(any(name == 'llm_steps' and row.get('status') == 'focus_rejected'
                            for name, row in logs))
        self.assertEqual(reviewer.evidence_focus['issue'], 'possible_onset')

    def test_off_preserves_original_measurement_flow(self):
        actions = [
            ('score_ecgfounder', dict(lead='ECG1', start_s=1770., end_s=1780.)),
            ('detect_beats', dict(lead='ECG1', start_s=1770., end_s=1780.)),
            ('finish_measurements', {}),
            ('submit_event_decision', dict(action='KEEP_OPEN', interpretation='support',
                                           event_id=None, end_s=None, evidence_ids=['E1'],
                                           reason='Current evidence supports AF/AFL')),
        ]
        decision, reviewer, model, logs = self.run_case(actions, enable=False)
        self.assertEqual(decision[0], 'support')
        self.assertNotIn('record_evidence_focus', model.seen[1][1])
        self.assertIsNone(reviewer.evidence_focus)
        self.assertNotIn('evidence_focus', model.final_payload)


if __name__ == '__main__':
    unittest.main()
