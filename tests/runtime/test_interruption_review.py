from unittest import TestCase
import numpy as np
from cardioagent.ecg_agent.experiments.interruption_review import review
from cardioagent.ecg_agent.experiments.subwindow_change import measure_lead
from tests.runtime.test_brief_review import Model
from tests.runtime.test_react import reply


class InterruptionTest(TestCase):
    case=dict(record_id='12',cursor=30,scores=[[20,30,[.9,.8]]])

    def test_score_only_and_optional_acquisition(self):
        done=reply('submit_interruption_review',dict(action='NO_FLAG',cell='none',reason='stable'))
        for mode in ('aligned','active','all'):
            acquire=mode!='aligned'
            seen=[]
            model=Model(([reply('measure_beat_change',{})] if mode=='active' else [])+[done])
            res=review(self.case,model,lambda:seen.append(1) or {'ECG1':[None]*5,'ECG2':[None]*5},
                       lambda *args:None,mode)
            self.assertEqual(res['decision']['action'],'NO_FLAG')
            self.assertEqual(res['measurement_requests'],int(acquire))
            self.assertEqual(len(seen),int(acquire))

    def test_causal_and_missing(self):
        for data in (dict(self.case,cursor=20),dict(self.case,truth=1)):
            with self.assertRaises(ValueError):
                review(data,None,None,None,'aligned')
        self.assertEqual(measure_lead(np.zeros(3840),128),[None]*5)
        with self.assertRaises(ValueError):
            measure_lead(np.zeros(3841),128)
