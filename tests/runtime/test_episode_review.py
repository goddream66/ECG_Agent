"""Regression checks for causal event snapshots and evidence validation."""

import unittest

from cardioagent.ecg_agent.harness.episode_review import EpisodeRecommendation, validate_recommendation
from cardioagent.ecg_agent.memory.episode_review import review_packets


class EpisodeReviewTests(unittest.TestCase):
    def test_packet_excludes_future_event_updates_and_evidence(self):
        traces, updates, evidence = [], [], []
        for end in range(10, 81, 10):
            start = end - 10
            ids = [f'{tool}-{end}' for tool in ('score_ecgfounder', 'assess_quality')]
            traces.append(dict(task_id='task', start_s=start, end_s=end, cursor_s=end,
                               decision='conflict' if end in (30, 40) else 'support',
                               reason='example', decision_threshold=.6, evidence_ids=ids))
            update = None
            if end in (20, 60, 70, 80):
                update = dict(kind='confirm' if end <= 60 else 'extend',
                              interval=[0 if end == 20 else 40, end], recorded_at_s=end)
            updates.append(dict(recorded_at_s=end, event_id=1 if end <= 40 else 2, event_update=update))
            for tool, key in zip(('score_ecgfounder', 'assess_quality'), ids):
                payload = dict(af_score=.9, afl_score=.1) if tool == 'score_ecgfounder' else dict(reasons=[])
                evidence.append(dict(task_id='task', window_s=[start, end], lead='ECG1', tool=tool,
                                     evidence_id=key, available_at_s=end, status='ok', result=payload))
        final = [dict(event_id=1, closed_at_s=40, termination_reason='unknown_timeout'),
                 dict(event_id=2, closed_at_s=80, termination_reason='stream_end')]
        packet, = review_packets(traces, updates, evidence, final, start_threshold=.6)
        self.assertEqual(packet['right']['interval'], [40, 60])
        self.assertEqual(packet['gap_s'], [20, 40])
        self.assertNotIn('score_ecgfounder-70', packet['evidence_index'].values())
        valid_ids = [key for choices in packet['gap_support'] for key in choices[0]]
        valid = EpisodeRecommendation(action='merge', reason='Gap supported', evidence_ids=valid_ids)
        validate_recommendation(valid, packet)
        with self.assertRaisesRegex(ValueError, 'merge_without_full_gap_support'):
            validate_recommendation(valid.model_copy(update={'evidence_ids': valid_ids[:2]}), packet)
        with self.assertRaisesRegex(ValueError, 'unknown_evidence_reference'):
            validate_recommendation(valid.model_copy(update={'evidence_ids': ['future']}), packet)
        packet['sustained_opposition'] = True
        with self.assertRaisesRegex(ValueError, 'link_despite_sustained_opposition'):
            validate_recommendation(valid.model_copy(update={'action': 'link_only'}), packet)
