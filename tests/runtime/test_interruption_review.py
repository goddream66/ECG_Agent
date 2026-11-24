from unittest import TestCase
from tempfile import TemporaryDirectory
from pathlib import Path
import gzip
import json
import numpy as np
from cardioagent.ecg_agent.llm.recording import RecordingChatModel
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

    def test_real_call_logger_records_cursor(self):
        model=Model([reply('submit_interruption_review',dict(action='NO_FLAG',cell='none',reason='stable'))])
        with TemporaryDirectory() as tmp:
            log=Path(tmp)/'calls.jsonl.gz'
            res=review(self.case,RecordingChatModel(model,log),None,lambda *args:None,'aligned')
            self.assertEqual(res['decision']['status'],'accepted')
            with gzip.open(log,'rt') as stream:
                self.assertEqual(json.load(stream)['cursor_s'],30)

    def test_finding_resolution_tracks_available_measurements(self):
        def submit(action):
            return reply('submit_interruption_review',dict(action=action,reason='Review the measured interval.'))
        model=Model([submit('FLAG_CELL_2'),submit('FLAG_WINDOW')])
        result=review(self.case,model,None,lambda *args:None,'aligned',grounded=True)
        self.assertEqual(result['decision']['submitted_action'],'FLAG_WINDOW')
        self.assertEqual(result['counts']['llm_calls'],2)
        measured=lambda:{'ECG1':[None,None,[1.,.1,.1,1.,2,.1],None,None],'ECG2':[None]*5}
        model=Model([reply('measure_beat_change',{}),submit('FLAG_CELL_2')])
        result=review(self.case,model,measured,lambda *args:None,'active',grounded=True)
        opts=lambda i:model.requests[i][1][0]['function']['parameters']['properties']['action']['enum']
        self.assertNotIn('FLAG_CELL_2',opts(0))
        self.assertIn('FLAG_CELL_2',opts(1))
        self.assertNotIn('FLAG_CELL_1',opts(1))
        self.assertEqual(result['decision']['cell'],'2')
