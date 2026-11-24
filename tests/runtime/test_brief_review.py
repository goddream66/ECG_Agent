from copy import deepcopy
from unittest import TestCase

from cardioagent.ecg_agent.harness.brief_review import BriefCandidateHistory, review
from tests.runtime.test_react import reply


def history():
    state = BriefCandidateHistory()
    for i, value in enumerate((.15, .12, .8, .1)):
        packet = state.observe(10*i, 10*i+10, {'ECG1':value, 'ECG2':value})
    return state, packet


class Model:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(deepcopy((messages, tools)))
        return next(self.replies)


class BriefReviewTest(TestCase):
    def test_only_elapsed_complete_agreeing_windows_trigger(self):
        state, packet = history()
        self.assertEqual((packet['candidate_s'],packet['window_s']), ([20,30],[30,40]))
        self.assertEqual([r['available_at_s'] for r in packet['measurements']], [10,10,20,20,30,30,40,40])
        original = deepcopy(packet)
        state.observe(40, 50, {'ECG1':1, 'ECG2':1})
        self.assertEqual(packet, original)
        for invalid in ({'ECG1':.1}, {'ECG1':.1,'ECG2':.9}, {'ECG1':float('nan'),'ECG2':.3}):
            state = BriefCandidateHistory()
            self.assertIsNone(state.observe(0,10,{'ECG1':.1,'ECG2':.1}))
            self.assertIsNone(state.observe(10,20,invalid))
            self.assertIsNone(state.observe(20,30,{'ECG1':.9,'ECG2':.9}))
            self.assertIsNone(state.observe(30,40,{'ECG1':.1,'ECG2':.1}))
        with self.assertRaises(ValueError):
            state.observe(30,40,{'ECG1':.2,'ECG2':.2})

    def test_active_acquisition_releases_only_requested_data_and_writes_at_decision_time(self):
        _, packet = history()
        model = Model([reply('select_brief_measurement',dict(option='left_ECG2',reason='Check onset.')),
                       reply('select_brief_measurement',dict(option='finish',reason='Sufficient.')),
                       reply('review_brief_candidate',dict(decision='REPORT_PAST_EVENT',
                             evidence_ids=['E5','S1','E8'],reason='Earlier supported interval, then low.'))])
        calls = []
        def measure(lead, a, b, now, offset):
            calls.append((lead,a,b,now,offset))
            return dict(lead=lead, window_s=[a+offset,b+offset], available_at_s=now, status='ok',
                        result=dict(af_score=.81,afl_score=.1))
        result = review(packet, 'active', model, measure, lambda *args: None)
        self.assertEqual(calls, [('ECG2',20,30,40,-5)])
        self.assertEqual((result['write']['interval'], result['write']['t_recorded']), ([20,30],40))
        self.assertNotIn('0.81',model.requests[0][0][1]['content'])
        self.assertIn('0.81',model.requests[-1][0][1]['content'])

    def test_failed_measurement_is_uncertain_and_forged_citation_is_rejected(self):
        _, packet = history()
        invalid = reply('review_brief_candidate',dict(decision='REPORT_PAST_EVENT', evidence_ids=['S2'],
                                                       reason='Unavailable.'))
        model = Model([invalid,invalid])
        result = review(packet,'aligned',model,None,lambda *args: None)
        self.assertEqual(result['decision']['status'],'error')
        self.assertIsNone(result['write'])
        model = Model([reply('review_brief_candidate', dict(decision='UNCERTAIN', evidence_ids=['E6'],
                                                             reason='The passage is inconclusive.'))])
        def missing(lead,a,b,now,offset):
            return dict(lead=lead,window_s=[a+offset,b+offset],available_at_s=now,status='insufficient',result={})
        result = review(packet,'all',model,missing,lambda *args: None)
        self.assertEqual(result['decision']['decision'],'UNCERTAIN')
        self.assertIsNone(result['write'])
        self.assertEqual(len(result['requests']),4)
        packet['measurements'][0]['available_at_s'] = 50
        with self.assertRaisesRegex(ValueError,'arrived'):
            review(packet,'aligned',model,None,lambda *args: None)

    def test_offline_reference_fields_are_rejected_before_contacting_model(self):
        _, packet = history()
        model = Model([])
        with self.assertRaisesRegex(ValueError, 'input fields'):
            review(dict(packet, truth=True),'aligned',model,None,lambda *args: None)
        packet['measurements'][0]['reference'] = 'AF'
        with self.assertRaisesRegex(ValueError, 'measurement fields'):
            review(packet,'aligned',model,None,lambda *args: None)
        self.assertEqual(model.requests, [])
