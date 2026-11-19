"""Causal temporal summaries, history controls and audit execution."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

server = Path(__file__).resolve().parents[2] / 'ops' / 'server'
sys.path.insert(0, str(server))
spec = importlib.util.spec_from_file_location('fixed_transition_p2', server / 'fixed_transition_p2.py')
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)


def score(start, value, ident):
    return {'evidence_id': ident, 'tool': 'score_ecgfounder', 'lead': 'ECG1',
            'window_s': [start, start + 10], 'available_at_s': start + 10,
            'status': 'ok', 'result': {'af_score': value, 'afl_score': 0.0}}


class Model:
    def __init__(self):
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        name = tools[0]['function']['name']
        args = ({'decision': 'TRANSITION', 'evidence_ids': ['E1'],
                 'reason': 'candidate negative evidence'} if name == 'judge_transition' else
                {'decision': 'UNCERTAIN', 'counterevidence_ids': ['E1'],
                 'evidence_ids': ['E1'], 'reason': 'negative evidence is not conclusive'})
        return type('Reply', (), {'message': {'tool_calls': [
            {'function': {'name': name, 'arguments': json.dumps(args)}}]}})()


class P2Tests(unittest.TestCase):
    def setUp(self):
        self.trace = []
        self.evidence = []
        kinds = ['support', 'negative', 'support', 'negative', 'negative', 'negative']
        for i, kind in enumerate(kinds):
            self.trace.append({'start_s': i * 10, 'end_s': (i + 1) * 10,
                               'decision': kind, 'af_score': .9 if kind == 'support' else .1,
                               'afl_score': 0.0})
            self.evidence.append(score(i * 10, .9 if kind == 'support' else .1, f'Eraw{i}'))
        self.evidence.append(score(60, .9, 'future'))

    def test_current_and_prior_negative_runs_are_distinct(self):
        recent, visible, _, aliases = p2.p1.view(self.trace, self.evidence, 60)
        payload = p2.temporal_view(recent, visible, aliases)
        self.assertEqual(payload['current_consecutive_negative']['start_s'], 30)
        self.assertEqual(payload['current_consecutive_negative']['end_s'], 60)
        self.assertEqual(payload['earlier_negative_runs'][0]['start_s'], 10)
        self.assertEqual(payload['last_observed_support']['window_s'], [20, 30])
        self.assertNotIn('future', json.dumps(payload))

    def test_history_requires_past_recovered_dip_and_matched_control(self):
        older = [score(0, .8, 'a'), score(10, .2, 'b'), score(20, .9, 'c'),
                 score(30, .7, 'd'), score(40, .8, 'e')]
        variants = p2.history_variants(older, 'ECG1')
        self.assertEqual(variants['pairs'], [['b', 'c']])
        self.assertEqual(len(variants['relevant']), len(variants['control']))
        self.assertFalse(set(e['evidence_id'] for e in variants['relevant']) &
                         set(e['evidence_id'] for e in variants['control']))

    def test_audit_revises_candidate_and_cannot_see_future(self):
        model = Model()
        result = p2.judge(model, 'audit', self.trace, self.evidence, 60, 'ECG1')
        self.assertEqual(result['initial_answer']['decision'], 'TRANSITION')
        self.assertEqual(result['answer']['decision'], 'UNCERTAIN')
        self.assertEqual(result['model_calls'], 2)
        self.assertNotIn('"future"', model.calls[0][0][1]['content'])

    def test_all_history_arms_use_same_final_schema(self):
        prior = [score(-50, .8, 'a'), score(-40, .2, 'b'), score(-30, .9, 'c'),
                 score(-20, .7, 'd'), score(-10, .8, 'e')]
        for arm in ('history_none', 'history_relevant', 'history_control'):
            model = Model()
            result = p2.judge(model, arm, self.trace, prior + self.evidence, 60, 'ECG1')
            self.assertEqual(result['answer']['decision'], 'TRANSITION')
            self.assertEqual([t['function']['name'] for t in model.calls[0][1]],
                             ['judge_transition'])
            self.assertNotIn('"future"', model.calls[0][0][1]['content'])


if __name__ == '__main__':
    unittest.main()
