from copy import deepcopy
from unittest import TestCase

from cardioagent.ecg_agent.experiments.recorded_end_confirmation import confirm
from tests.runtime.test_brief_review import Model
from tests.runtime.test_react import reply


class RecordedConfirmationTests(TestCase):
    def packet(self):
        anchors={key:dict(id=ident,direction=direction,window_s=span) for key,ident,direction,span in (
            ('before','A','support',[0,10]),('first_against','B','negative',[20,30]),
            ('later_against','C','negative',[30,40]))}
        view=dict(window_s=[30,40],assessment=dict(interpretation='negative'),open_event=dict(event_id=1),
                  rhythm_by_time=[],pending_question=None,end_evidence=dict(anchors,can_confirm=True))
        return dict(key='00-40',cursor_s=40,context=view,aliases={'A':'h1','B':'h2','C':'h3'})

    def test_confirmation_preserves_raw_measurements_and_boundary(self):
        model=Model([reply('confirm_event_end',dict(decision='CONFIRM_END',reason='Later measurements agree.'))])
        result=confirm(self.packet(),model,lambda *args:None)['result']
        self.assertEqual(result['boundary']['range_s'],[0,30])
        self.assertEqual(result['confirmation_evidence_id'],'h3')

    def test_gate_is_causal_and_recovery_does_not_call_model(self):
        packet=self.packet()
        bad=deepcopy(packet)
        bad['context']['end_evidence']['later_against']['window_s']=[40,50]
        with self.assertRaises(ValueError):
            confirm(bad,None,None)
        packet['context']['end_evidence']['can_confirm']=False
        self.assertEqual(confirm(packet,None,None)['result']['status'],'unlocalized')
