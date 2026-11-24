from unittest import TestCase

from cardioagent.ecg_agent.experiments.shared_prefix_replay import replay, restore_prefix
from tests.runtime.test_brief_review import Model
from tests.runtime.test_react import reply


def reply_obs(direction, evidence, start, end):
    return dict(interpretation=direction, assessment='SUPPORTS_TARGET' if direction=='support' else 'AGAINST_TARGET',
                evidence_ids=[evidence], support_intervals=[[start,end]] if direction=='support' else [],
                opposing_intervals=[[start,end]] if direction=='negative' else [],reason='recorded measurement')


def intent(action, ident=None):
    return dict(status='accepted', action=action,event_id=ident,reason='recorded action')


def row(direction, evidence, start, end):
    return dict(id=evidence,evidence_id=evidence,lead='ECG1',window_s=[start,end],available_at_s=end,
                target_score=.9 if direction=='support' else .1,reference=dict(direction=direction))


class PrefixForkTests(TestCase):
    def test_common_ledger_fork_closes_and_can_reopen(self):
        support=reply_obs('support','M1',0,10)
        archive=[dict(start_s=0,cursor_s=10,decision='SUPPORTS_TARGET',
                      llm_action=dict(support,lifecycle=intent('CONFIRM_EVENT'),refinement={},pending=None,next_review_s=20)),
                 dict(start_s=10,cursor_s=20,decision='ACTION_ERROR'),
                 dict(start_s=20,cursor_s=30,decision='AGAINST_TARGET',
                      llm_action=dict(reply_obs('negative','M2',20,30),lifecycle=intent('REQUEST_CLOSE',1),
                                      refinement={},pending=None,next_review_s=40))]
        seed=restore_prefix(archive,30)
        timeline=[row('negative','M2',20,30),row('negative','M3',30,40)]
        packet=dict(warm=[dict(window_s=[0,10],current=[row('support','M1',0,10)]),
                          dict(window_s=[20,30],current=[timeline[0]])],
                    suffix=[dict(window_s=[30,40],observation=reply_obs('negative','M3',30,40),
                                 timeline=timeline,fork_intent=intent('REQUEST_CLOSE',1)),
                            dict(window_s=[40,50],observation=reply_obs('support','M4',40,50),
                                 timeline=[row('support','M4',40,50)])])
        conf=reply('confirm_event_end',dict(decision='CONFIRM_END',reason='Two measurements agree.'))
        reopen=reply('choose_event_action',dict(action='CONFIRM_EVENT',event_id=None,reason='Support resumed.'))
        hold=reply('choose_event_action',dict(action='CONTINUE_EVENT',event_id=1,reason='Support resumed.'))
        memory=replay(seed,packet,Model([conf,reopen]),lambda *args:None,True)
        rolling=replay(seed,packet,Model([hold]),lambda *args:None,False)
        self.assertEqual(memory['intervals'],[[0,15],[40,50]])
        self.assertEqual(rolling['intervals'],[[0,50]])
        self.assertEqual((seed.cursor,seed.active['event_id']), (30,1))
