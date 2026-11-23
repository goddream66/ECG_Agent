import unittest

from cardioagent.ecg_agent.harness import measured_event as contract
from tests.runtime import test_autonomous_events as fixtures
from tests.runtime.test_autonomous_events import FINISH, measure, reply


def decision(action, role, citations, **kwargs):
    return reply('submit_event_decision', dict(event_action=action, assessment=role,
                 reason='Paired protocol test', citations=citations, **kwargs))


def cite(key, role, **kwargs):
    return dict(evidence_id=key,role=role,**kwargs)


class MeasuredEventTests(unittest.TestCase):
    def test_two_fields_open_followup_and_boundary_recovery(self):
        positive, negative = contract.DIRECTIONS
        opened = decision('OPEN', positive, [cite('E1',positive,field='af_score'),cite('E1',positive,field='afl_score')],
                          pending=dict(question='Does support persist?',expected_evidence='Next pair'),
                          boundary=dict(kind='onset_unobserved',first_supported_interval_s=[0,10]), next_review_s=19)
        rejected = decision('CLOSE',negative,[cite('E1',negative)],pending_resolution='resolved',
                            boundary=dict(kind='bracketed',range_s=[10,20]))
        # Recovery retrieves the previously measured first interval.
        retrieve = reply('retrieve_memory',dict(scope='evidence',start_s=0.,end_s=10.))
        closed = decision('CLOSE',negative,[cite('H1',positive),cite('E1',negative)],pending_resolution='resolved',
                          boundary=dict(kind='bracketed',range_s=[0,20]))
        actions=[measure(0),FINISH,opened,measure(10),FINISH,rejected,retrieve,closed]
        result,trace,_,steps,requests=fixtures.EpisodeTests().replay('fixed',actions,2,split='v16')
        self.assertEqual([row['action'] for row in trace],['OPEN','CLOSE'])
        self.assertEqual(len(trace[0]['llm_action']['measured_facts']),2)
        self.assertEqual(len(trace[0]['llm_action']['evidence_ids']),1)
        self.assertEqual(trace[0]['next_review_s'],20)
        self.assertEqual(result['pending_questions'][0]['status'],'resolved')
        self.assertEqual(sum(x['status']=='evidence_recovery_requested' for x in steps),1)
        self.assertEqual(len(requests),len(actions))

    def test_context_cannot_be_the_only_directional_fact(self):
        positive = contract.DIRECTIONS[0]
        illegal = decision('RECORD_ONLY',positive,[cite('E1',positive)])
        repaired = decision('RECORD_ONLY','UNCERTAIN',[cite('E1','NEUTRAL')],uncertainty_cause='insufficient')
        _,trace,_,steps,_=fixtures.EpisodeTests().replay('fixed',[measure(0,'get_rate_rhythm'),FINISH,illegal,repaired],1,split='v16')
        self.assertEqual(trace[0]['llm_action']['assessment'],'UNCERTAIN')
        self.assertTrue(any('Only a usable Founder' in s.get('error_message','') for s in steps))
        self.assertFalse(any(s['status']=='evidence_recovery_requested' for s in steps))

    def test_boundary_cannot_be_a_single_score_or_peak(self):
        direction = {contract.DIRECTIONS[0]:[dict(window_s=[1,11])],contract.DIRECTIONS[1]:[]}
        with self.assertRaises(contract.BoundaryEvidenceNeeded):
            contract.validate_boundary(contract.legacy.Bracket(kind='bracketed',range_s=[1,11]),'OPEN',direction)
        with self.assertRaises(contract.BoundaryEvidenceNeeded):
            contract.validate_boundary(contract.legacy.Point(kind='localized',time_s=5),'OPEN',direction)

