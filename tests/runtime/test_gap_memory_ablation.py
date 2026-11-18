import unittest

from cardioagent.ecg_agent.experiments.gap_memory_ablation import (
    automatic_support, coverage_blockers, memory_inputs,
)
from cardioagent.ecg_agent.harness.gap_relationship import GapRelationshipReviewer


class GapMemoryAblationTests(unittest.TestCase):
    def test_evidence_and_citation_bottlenecks_are_separable(self):
        state = {
            'gap_s': [10, 30],
            'threshold': 0.6,
            'observations': [
                dict(id='O1', window_s=[10, 20], score=[0.6, 0.1], quality='no_warning'),
                dict(id='O2', window_s=[20, 30], score=[0.8, 0.1], quality='not_measured'),
                dict(id='S1', window_s=[25, 35], score=[0.9, 0.1], quality='no_warning'),
            ],
        }
        refs = {'O1': ['score1', 'quality1'], 'O2': ['score2'], 'S1': ['shift']}
        result = automatic_support('continuous', state, refs)
        self.assertFalse(result['support_ids'])
        self.assertEqual(coverage_blockers(state, result['validated_support_s']), [
            {'interval_s': [20, 30], 'reasons': ['positive_quality_not_measured']},
        ])
        state['observations'][1]['quality'] = 'no_warning'
        self.assertTrue(automatic_support('continuous', state, refs)['support_ids'])
        self.assertFalse(automatic_support('uncertain', state, refs)['support_ids'])

    def test_memory_exposure_varies_only_the_requested_fields(self):
        inputs, refs = memory_inputs(dict(
            gap_s=[10, 20], cursor_s=40, start_threshold=0.6,
            left={'interval': [0, 10]}, right={'interval': [20, 40]},
            windows=[dict(window_s=[10, 20], threshold=0.4, decision='negative', leads={
                'ECG1': dict(score=[0.4, 0.1], score_status='ok', quality='no_warning',
                             score_id='e1', quality_id='e2'),
            })], evidence_index={'e1': 'hash1', 'e2': 'hash2'},
        ))
        self.assertEqual(inputs['summary']['observations'], [])
        self.assertEqual(inputs['windows']['events'], [])
        self.assertEqual(inputs['both']['events'], inputs['summary']['events'])
        self.assertEqual(inputs['both']['observations'], inputs['windows']['observations'])
        self.assertEqual(inputs['both']['observations'][0]['decision_threshold'], 0.4)
        self.assertEqual(refs['O1'], ['hash1', 'hash2'])

    def test_event_ids_are_valid_context_but_not_window_support(self):
        from types import SimpleNamespace

        args = dict(relation='separate', evidence=['E_LEFT'], support=[], reason='Test.')
        response = SimpleNamespace(usage={}, message={'role': 'assistant', 'tool_calls': [
            {'function': {'name': 'submit_gap_judgment', 'arguments': args}},
        ]})
        model = SimpleNamespace(complete=lambda *unused: response)
        reviewer = GapRelationshipReviewer(model)
        state = dict(events=[{'id': 'E_LEFT'}], observations=[])
        self.assertEqual(reviewer.judge(state)['status'], 'ok')
        args['support'] = ['E_LEFT']
        self.assertEqual(reviewer.judge(state)['status'], 'error')
