import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cardioagent.ecg_agent.experiments.runner import run_job
from cardioagent.ecg_agent.harness.event_decision import WindowEventDecision, validate_grounding
from cardioagent.ecg_agent.harness.react import ReActReview
from cardioagent.ecg_agent.policies.event_rules import EventConfig
from tests.runtime import test_event_checkpoints as fixtures
from tests.runtime.test_ecg_toolkit import measured
from tests.runtime.test_react import ScriptModel, reply


CONFIG = Path(__file__).resolve().parents[2] / 'configs/ecg/react_qwen_llm_decision_v2.json'


def submit(action, interpretation, ids, event_id=None, end_s=None):
    return reply('submit_event_decision', dict(action=action, interpretation=interpretation,
                 event_id=event_id, end_s=end_s, evidence_ids=ids, reason='Review of measured rhythm'))


def measure(tool, lead, start):
    return reply(tool, dict(lead=lead, start_s=start, end_s=start + 10))


class LLMWindowTests(unittest.TestCase):
    def run_pair(self, actions, *, exit_windows=2, **settings):
        with TemporaryDirectory() as tmp:
            job = fixtures.EventCheckpointTests().make_job(Path(tmp))
            job.update(policy='event-react', llm=json.loads(CONFIG.read_text()) | settings,
                       config=EventConfig(enter_windows=1, exit_windows=exit_windows).to_dict())
            model = ScriptModel(actions)
            wave = np.sin(np.arange(5000) / 8.)
            raw = SimpleNamespace(fs=250, sig_name=('ECG1','ECG2'), sig_len=5000,
                                  p_signal=np.column_stack([wave,wave]), units=('mV','mV'))
            with (
                patch('cardioagent.ecg_agent.llm.providers.build_chat_model', return_value=model),
                patch('cardioagent.ecg_agent.agent.wfdb.rdrecord', return_value=raw),
                patch('cardioagent.ecg_agent.tools.algorithms.measure', side_effect=measured),
                patch('cardioagent.ecg_agent.harness.react.decide_event') as rule,
            ):
                result, _ = run_job(job)
                rule.assert_not_called()
            def rows(name):
                return [json.loads(s) for s in (Path(job['target']) / (name+'.jsonl')).read_text().splitlines()]
            return result, rows('event_updates'), rows('window_trace'), rows('assessments'), rows('llm_steps')

    def test_fixed_measurements_follow_submitted_interpretation(self):
        finish = reply('finish_measurements', {})
        for action, interpretation, state, ids, boundary in (
                ('KEEP_OPEN', 'support', 'ACTIVE', ['E1'], None),
                ('DEFER', 'negative', 'PENDING_END', ['E1'], None),
                ('DEFER', 'conflict', 'UNKNOWN_HOLD', ['E1','E2'], None),
                ('DEFER', 'unavailable', 'UNKNOWN_HOLD', [], None),
                ('CLOSE', 'negative', 'IDLE', ['E1','E2'], 15.)):
            with self.subTest(action=action, interpretation=interpretation):
                actions = [measure('score_ecgfounder','ECG1',0.), finish,
                           submit('KEEP_OPEN','support',['E1']),
                           measure('score_ecgfounder','ECG1',10.), measure('detect_beats','ECG2',10.),
                           finish, submit(action,interpretation,ids,1,boundary)]
                result, updates, trace, assessments, steps = self.run_pair(actions)
                self.assertEqual(updates[-1]['state'], state)
                for row in (updates[-1], trace[-1], assessments[-1]):
                    self.assertEqual(row['decision_source'], 'llm')
                    self.assertEqual(row['interpretation'], interpretation)
                    self.assertEqual(row['action'], action)
                    self.assertEqual(len(row['evidence_ids']), len(ids))
                    self.assertEqual(len(row['acquired_evidence_ids']), 2)
                self.assertTrue(assessments[-1]['checked'])
                if boundary:
                    self.assertEqual(result['intervals'], [[0.,15.]])
                    self.assertEqual(result['events'][0]['end_decided_at_s'], 20.)

    def test_reserved_final_retry_after_eleven_interactions_and_cache_hits(self):
        actions = [measure('score_ecgfounder','ECG1',0.)] * 11
        actions += [submit('CLOSE','negative',['E1'],None,8.), submit('KEEP_OPEN','support',['E1']),
                    reply('finish_measurements',{}), submit('DEFER','negative',['forged'],1),
                    submit('CLOSE','support',[],1,15.)]
        result, updates, trace, assessments, steps = self.run_pair(actions)
        self.assertEqual(result['counts']['llm_calls'], 16)
        self.assertEqual(trace[0]['tool_calls'], 11)
        self.assertEqual(trace[1]['tool_calls'], 0)
        self.assertEqual(trace[1]['decision_source'], 'llm_action_error')
        self.assertEqual(trace[1]['action'], 'ACTION_ERROR')
        self.assertFalse(assessments[1]['checked'])
        self.assertEqual(updates[1]['state'], 'UNKNOWN_HOLD')
        self.assertEqual([s['phase'] for s in steps if s.get('cursor_s') == 10 and 'phase' in s],
                         ['acquisition']*11 + ['final','retry'])

    def test_tool_cap_is_independent_of_round_cap(self):
        for invalid in (0, -1, True, 1.2):
            with self.assertRaises(ValueError):
                ReActReview(None, llm_tool_max_calls=invalid)
        finish = reply('finish_measurements', {})
        actions = [measure('score_ecgfounder','ECG1',0.)]*6
        actions += [submit('KEEP_OPEN','support',['E1']), finish,
                    submit('DEFER','unavailable',[],1)]
        _, _, trace, _, steps = self.run_pair(actions, llm_tool_max_calls=5,
                                           memory_write=False, memory_retrieve=False)
        self.assertEqual(trace[0]['tool_calls'], 5)
        self.assertEqual(trace[0]['action'], 'KEEP_OPEN')
        self.assertTrue(any(s.get('phase')=='final' and s['status']=='error' for s in steps))

    def test_safety_closure_attribution(self):
        finish = reply('finish_measurements', {})
        actions = [measure('score_ecgfounder','ECG1',0.), finish, submit('KEEP_OPEN','support',['E1']),
                   measure('score_ecgfounder','ECG1',10.), finish, submit('DEFER','negative',['E1'],1)]
        _, updates, _, assessments, _ = self.run_pair(actions, rule_safety_fallback=True, exit_windows=1)
        self.assertEqual(assessments[-1]['decision_source'], 'llm')
        self.assertEqual(updates[-1]['action'], 'DEFER')
        self.assertEqual(updates[-1]['event_update']['decision_source'], 'rule_fallback')

    def test_invalid_grounding_and_action_pairs(self):
        def decision(action, interpretation, ids):
            return WindowEventDecision(action=action, interpretation=interpretation, evidence_ids=ids,
                                       event_id=1, end_s=15. if action=='CLOSE' else None, reason='measured')
        for action in ('KEEP_OPEN','DEFER','CLOSE'):
            for value in ('support','negative','conflict','unavailable'):
                legal = (action == 'DEFER' and value != 'support' or
                         (action,value) in {('KEEP_OPEN','support'),('CLOSE','negative')})
                if not legal:
                    with self.assertRaises(ValueError):
                        decision(action,value,[])
        base = dict(evidence_id='e', tool='detect_beats', lead='ECG2', status='ok',
                    available_at_s=20., window_s=[10.,20.], result={'r_peaks_s':[11.,12.]})
        for delta, action, value in (({},'CLOSE','negative'), ({},'DEFER','conflict'),
                                   ({'window_s':[0.,10.]},'KEEP_OPEN','support'),
                                   ({'status':'error'},'DEFER','negative'),
                                   ({'available_at_s':30.},'KEEP_OPEN','support')):
            with self.assertRaises(ValueError):
                validate_grounding(decision(action,value,['e']), {'e':base | delta},
                                   None,10.,20.,{'event_id':1},{'detect_beats'})
