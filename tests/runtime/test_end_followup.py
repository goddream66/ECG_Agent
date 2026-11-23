import unittest

from cardioagent.ecg_agent.harness.end_review import end_evidence
from tests.runtime import test_autonomous_events as fixture
from tests.runtime.test_react import reply
from tests.runtime.test_staged_review import act, boundary


class EndFollowupTests(unittest.TestCase):
    def test_overlaps_gaps_conflicts_and_future(self):
        def row(a,b,d):
            return dict(id=f'{a}-{b}-{d}',window_s=[a,b], reference=dict(direction=d))
        support=row(0,10,'support')
        neg=row(10,20,'negative')
        later=row(20,30,'negative')
        def eligible(seq):
            return end_evidence(seq,[20,30],dict(estimated_start_s=0))['can_confirm']
        self.assertTrue(eligible([later,support,neg]))
        for seq in ([support,neg,neg],[support,later],[support,row(15,25,'negative'),later],
                    [support,neg,row(20,30,'support'),later],[support,neg,row(30,40,'negative')]):
            self.assertFalse(eligible(seq))

    def test_full_runtime_preserves_identity_and_projects_earlier_boundary(self):
        def observe(result):
            return reply('assess_window', dict(finding=result,evidence_ids=['E1'],reason='current evidence'))
        cfg=fixture.ROOT/'configs/ecg/react_qwen_followup_close.json'
        for recovers in (True,False):
            actions=[fixture.measure(0),fixture.FINISH,observe('TARGET_PRESENT'),act('CONFIRM_EVENT',None),
                     boundary('unlocalized'),fixture.measure(10),fixture.FINISH,observe('TARGET_ABSENT'),
                     act('REQUEST_CLOSE',1),fixture.measure(20),fixture.FINISH]
            if recovers:
                actions.extend([observe('TARGET_PRESENT'),act('CONTINUE_EVENT',1)])
            else:
                actions.extend([observe('TARGET_ABSENT'),act('REQUEST_CLOSE',1),
                                reply('confirm_event_end',dict(decision='CONFIRM_END',reason='continued contrary measurements'))])
            result,trace,*_=fixture.EpisodeTests().replay('fixed',actions,3,config_file=cfg,
                              score_pairs=[[.99,.1],[.1,.02],[.97 if recovers else .12,.03]])
            event,=result['events']
            self.assertEqual(trace[1]['state'],'OPEN')
            self.assertEqual(event['final_status'],'RIGHT_CENSORED' if recovers else 'CLOSED')
            self.assertEqual(event['end_decided_at_s'],None if recovers else 30)
            if not recovers:
                self.assertEqual(event['estimated_end_s'],10)
                self.assertIn('confirmation_evidence_id',trace[2]['llm_action']['refinement'])
