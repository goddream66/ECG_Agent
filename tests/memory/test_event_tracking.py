import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from cardioagent.ecg_agent.event_state import EventConfig, EventTracker
from cardioagent.ecg_agent.evidence import EvidenceLedger
from cardioagent.ecg_agent.review_policy import decide_event
from cardioagent.ecg_agent.toolkit import ToolContext, ToolSession
from cardioagent.ecg_agent.toolkit.contracts import EvidenceContext
from cardioagent.recording import Waveform


class EventTrackingTests(unittest.TestCase):
    def tracker(self, decisions, **config):
        t = EventTracker(EventConfig(**config))
        trace = [t.observe(i * 10., (i + 1) * 10., decision) for i, decision in enumerate(decisions)]
        return t, trace

    def test_onset_confirmation_and_end_are_distinct(self):
        t, trace = self.tracker(['support', 'support', 'negative', 'negative'])
        e = t.events[0]
        self.assertEqual((e['estimated_start_s'], e['first_seen_at_s'], e['confirmed_at_s']), (0, 10, 20))
        self.assertEqual((e['estimated_end_s'], e['end_decided_at_s']), (20, 40))
        self.assertEqual(t.intervals, [[0, 20]])
        self.assertEqual(trace[2]['state'], 'PENDING_END')
        self.assertEqual(e['termination_reason'], 'confirmed_end')

    def test_llm_closure_commits_fractional_end_and_does_not_auto_end(self):
        tracker = EventTracker(EventConfig(), llm_closure=True)
        tracker.observe(0, 10, 'support')
        tracker.observe(10, 20, 'support')
        tracker.observe(20, 30, 'negative')
        tracker.observe(30, 40, 'negative')
        self.assertEqual(tracker.state, 'PENDING_END')
        self.assertIsNone(tracker.active['end_decided_at_s'])
        update = tracker.observe(40, 50, 'negative', close_request={
            'event_id': 1, 'end_s': 22.35, 'evidence_ids': ['beat-1'], 'reason': 'reviewed transition'})
        self.assertEqual(tracker.state, 'IDLE')
        self.assertEqual(tracker.intervals, [[0, 22.35]])
        self.assertEqual((tracker.events[0]['estimated_end_s'], tracker.events[0]['end_decided_at_s']),
                         (22.35, 50))
        self.assertEqual(tracker.events[0]['termination_reason'], 'llm_close')
        self.assertEqual(update['event_update']['previous_interval'], [0, 20])
        self.assertEqual(update['event_update']['evidence_ids'], ['beat-1'])
        tracker.observe(50, 60, 'support')
        self.assertEqual(tracker.events[1]['event_id'], 2)

    def test_llm_closure_keeps_unknown_open_until_stream_end(self):
        tracker = EventTracker(EventConfig(), llm_closure=True)
        tracker.observe(0, 10, 'support')
        tracker.observe(10, 20, 'support')
        tracker.observe(20, 30, 'unavailable')
        tracker.observe(30, 40, 'unavailable')
        self.assertEqual(tracker.state, 'UNKNOWN_HOLD')
        tracker.finish(40)
        self.assertTrue(tracker.events[0]['right_censored'])

    def test_llm_close_rejects_future_or_wrong_event_without_mutation(self):
        tracker = EventTracker(EventConfig(), llm_closure=True)
        tracker.observe(0, 10, 'support')
        tracker.observe(10, 20, 'support')
        for event_id, end_s in ((2, 18.), (1, 25.), (1, 0.)):
            with self.subTest(event_id=event_id, end_s=end_s):
                with self.assertRaises(ValueError):
                    tracker.close_event({'event_id': event_id, 'end_s': end_s,
                                         'evidence_ids': ['beat-1'], 'reason': 'test'}, 20)
                self.assertEqual(tracker.intervals, [[0, 20]])

    def test_one_negative_does_not_split_event_but_remains_counter_evidence(self):
        t, _ = self.tracker(['support', 'support', 'negative', 'support'])
        self.assertEqual(t.intervals, [[0, 40]])
        self.assertEqual(t.events[0]['counter_evidence_intervals'], [[20, 30]])
        self.assertEqual(t.events[0]['unknown_intervals'], [])

    def test_unknown_preserves_identity_and_is_not_confirmed_support(self):
        t, _ = self.tracker(['support', 'support', 'conflict', 'support'])
        self.assertEqual(len(t.events), 1)
        self.assertEqual(t.intervals, [[0, 40]])
        self.assertNotIn([20, 30], t.events[0]['confirmed_support_intervals'])
        self.assertEqual(t.events[0]['unknown_intervals'], [[20, 30]])

    def test_unknown_breaks_negative_consecutiveness(self):
        t, _ = self.tracker(['support', 'support', 'negative', 'conflict', 'negative'])
        self.assertEqual(t.state, 'PENDING_END')
        self.assertIsNone(t.events[0]['end_decided_at_s'])
        t.observe(50, 60, 'negative')
        self.assertEqual(t.events[0]['estimated_end_s'], 40)

    def test_unknown_timeout_never_fabricates_end_and_new_event_has_new_identity(self):
        t, _ = self.tracker(['support', 'support', 'unavailable', 'unavailable', 'support', 'support'])
        self.assertEqual(t.intervals, [[0, 20], [40, 60]])
        self.assertEqual(t.events[0]['termination_reason'], 'unknown_timeout')
        self.assertIsNone(t.events[0]['estimated_end_s'])
        self.assertEqual([e['event_id'] for e in t.events], [1, 2])

    def test_candidate_timeout_and_record_end(self):
        t, _ = self.tracker(['support', 'unavailable', 'unavailable'])
        self.assertEqual(t.events[0]['termination_reason'], 'candidate_timeout')
        self.assertEqual(t.intervals, [])
        t, _ = self.tracker(['support', 'support'])
        t.finish(20)
        self.assertTrue(t.events[0]['right_censored'])
        self.assertIsNone(t.events[0]['end_decided_at_s'])

    def test_future_observations_do_not_mutate_existing_writes(self):
        t, trace = self.tracker(['support', 'support'])
        saved = deepcopy(trace)
        t.observe(20, 30, 'support')
        self.assertEqual(trace, saved)
        self.assertEqual(trace[1]['event_update']['interval'], [0, 20])
        with self.assertRaises(ValueError):
            t.observe(40, 50, 'support')

    def test_validation(self):
        for kwargs in ({'unknown_grace_s': float('nan')}, {'exit_windows': 0}, {'review_budget': 0}):
            with self.assertRaises(ValueError): EventConfig(**kwargs)

    def test_namespaced_sessions_and_cache_sources(self):
        w = Waveform(np.sin(np.arange(5000) / 10.)[:, None], 500, ('ECG1',), ('normalized',))
        args = {'lead': 'ECG1', 'start_s': 0., 'end_s': 10.}
        ids = []
        for record in ('a', 'b', 'a'):
            c = EvidenceContext('run', 'task', record)
            session = ToolSession(ToolContext(w, 0., 10.), evidence_context=c)
            a = session.call('assess_quality', args)
            b = session.call('assess_quality', args)
            ids.extend([a['evidence_id'], b['evidence_id']])
            self.assertEqual(session.cache_sources[b['evidence_id']], a['evidence_id'])
            ledger = EvidenceLedger(c)
            ledger.append(a)
            a['result'].clear()
            self.assertTrue(ledger.get(a['evidence_id'])['result'])
            with self.assertRaises(ValueError): ledger.assessment('support', 'x', ['absent'], 10)
            with self.assertRaises(ValueError): ledger.assessment('support', 'x', [b['evidence_id']], 5)
        self.assertEqual(len(ids), len(set(ids)))

    def test_active_negative_requests_secondary_and_budget_failure_is_unknown(self):
        tracker, _ = self.tracker(['support', 'support'])
        primary = {'lead': 'ECG1', 'status': 'ok', 'result': {'af_score': .1, 'afl_score': .02}}
        calls = []
        def call(tool, lead):
            calls.append((tool, lead))
            return {'lead': lead, 'status': 'ok', 'result': {'reasons': []} if tool == 'assess_quality'
                    else {'af_score': .9, 'afl_score': .01}}
        decision, _, _ = decide_event(call, primary, 'ECG2', tracker, EventConfig())
        self.assertEqual(decision, 'conflict')
        self.assertIn(('score_ecgfounder', 'ECG2'), calls)
        decision, _, reason = decide_event(call, primary, 'ECG2', tracker, EventConfig(review_budget=1))
        self.assertEqual((decision, reason), ('unavailable', 'review_budget_exhausted'))

    def test_secondary_only_is_explicit_and_rr_is_not_requested(self):
        tracker = EventTracker()
        calls = []
        def call(tool, lead):
            calls.append(tool)
            return {'lead': lead, 'status': 'ok', 'result': {'reasons': []} if tool == 'assess_quality'
                    else {'af_score': .9, 'afl_score': .01}}
        result = decide_event(call, {'status': 'insufficient'}, 'ECG2', tracker, EventConfig())
        self.assertEqual(result, ('support', True, 'single_usable_secondary'))
        self.assertNotIn('get_rate_rhythm', calls)

    def test_borderline_negative_without_event_still_requests_review(self):
        calls=[]
        def call(tool,lead):
            calls.append((tool,lead))
            return {'lead':lead,'status':'ok','result':{'reasons':[]} if tool=='assess_quality'
                    else {'af_score':.8,'afl_score':0.}}
        primary={'lead':'ECG1','status':'ok','result':{'af_score':.49,'afl_score':0.}}
        result=decide_event(call,primary,'ECG2',EventTracker(),EventConfig())
        self.assertEqual(result[0],'conflict')
        self.assertIn(('score_ecgfounder','ECG2'),calls)
