import unittest

from cardioagent.ecg_agent.experiments.qwen_gap_relationship import reference_relation, revisions


class GapRelationshipTests(unittest.TestCase):
    def test_shifted_internal_support_leaves_edges_unknown(self):
        state = dict(gap_s=[160, 180], threshold=.6, observations=[
            dict(id='S1', window_s=[155, 165], score=[.9, .1], quality='not_measured'),
            dict(id='S2', window_s=[165, 175], score=[.9, .1], quality='not_measured'),
        ])
        result = revisions(dict(relation='continuous', support=['S1', 'S2']), state, {'S1': ['a'], 'S2': ['b']})
        self.assertEqual(result['model_supported_s'], [[165, 175]])
        self.assertEqual(result['model_unknown_s'], [[160, 165], [175, 180]])
        self.assertEqual(result['support_ids'], [])

    def test_reference_requires_known_gap_and_same_episode(self):
        gap, left, right = [160, 180], [150, 160], [180, 190]
        self.assertEqual(reference_relation(gap, left, right, [[150, 190]], [[150, 190]]), 'continuous')
        self.assertEqual(reference_relation(gap, left, right, [[150, 162], [178, 190]], [[150, 190]]), 'separate')
        self.assertEqual(reference_relation(gap, left, right, [[150, 190]], [[150, 169], [170, 190]]), 'unscored')
