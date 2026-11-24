from unittest import TestCase

from cardioagent.ecg_agent.harness.staged_review import settled_lifecycle


class SelectiveLifecycleTests(TestCase):
    def setUp(self):
        self.window=[140,150]
        self.active=dict(event_id=1,confirmed_at_s=10)
        self.transition=dict(pending_question=None,rhythm_by_time=[
            dict(lead=lead,window_s=self.window,reference=dict(direction='support')) for lead in ('ECG1','ECG2')])

    def test_accepted_routine_actions(self):
        no_open=settled_lifecycle(self.window,dict(interpretation='negative'),None,None,self.transition)
        self.assertEqual(no_open['action'],'NO_EVENT_CHANGE')
        action=settled_lifecycle(self.window,dict(interpretation='support'),self.active,None,self.transition)
        self.assertEqual((action['action'],action['event_id']),('CONTINUE_EVENT',1))

    def test_closures_openings_conflicts_and_candidate_context_keep_model(self):
        for direction,active,candidate,ctx in (
                ('support', None, None, self.transition),
                ('negative', self.active, None, self.transition),
                ('conflict',self.active,None,self.transition),
                ('negative',None,dict(event_id=1),self.transition),
                ('support',self.active,None,dict(self.transition,pending_question={'question':'review'})),
                ('support',dict(self.active,confirmed_at_s=130),None,self.transition),
                ('support',self.active,None,dict(self.transition,rhythm_by_time=self.transition['rhythm_by_time'][:1]))):
            self.assertIsNone(settled_lifecycle(self.window,dict(interpretation=direction),active,candidate,ctx))

    def test_returned_boundary_evidence_keeps_model(self):
        earlier=dict(self.transition['rhythm_by_time'][0],window_s=[0,10])
        ctx=dict(self.transition,rhythm_by_time=[earlier,*self.transition['rhythm_by_time']])
        self.assertIsNone(settled_lifecycle(self.window,dict(interpretation='support'),self.active,None,ctx))
