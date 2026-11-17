from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cardioagent.ecg_agent.benchmarks.event_replay import fixed_evidence_record, replay_event_record
from cardioagent.ecg_agent.benchmarks.review_replay import EventMemory, decide
from cardioagent.ecg_agent.event_state import EventConfig
from cardioagent.ecg_agent.toolkit.contracts import EvidenceContext


class Logs:
    def __init__(self): self.rows = {}
    def emit(self, name, row): self.rows.setdefault(name,[]).append(deepcopy(row))


class EventReplayTests(unittest.TestCase):
    def old_rows(self):
        rows=[]
        shadow=EventMemory()
        for i, score in enumerate((.8,.9,.9,.1,.1)):
            calls=[]
            start,end=float(i*10),float((i+1)*10)
            def call(tool,lead):
                result={'tool':tool,'tool_version':'1','status':'ok','lead':lead,'window_s':[start,end],
                        'available_at_s':end,'evidence_id':f'old-{i}-{len(calls)}',
                        'result':{'reasons':[]} if tool=='assess_quality' else
                                  {'af_score':score,'afl_score':0.},
                        'cached':False,'duration_ms':1.,'warnings':[],'error':None,'schema_version':1}
                calls.append(result)
                return result
            primary=call('score_ecgfounder','ECG1')
            decision,checked,reason=decide(call,primary,'ECG2',.5,shadow,'adaptive',4)
            shadow.observe(start,end,decision,checked,[r['evidence_id'] for r in calls])
            rows.append({'task_id':'t','channel':'ECG1','start_s':start,'end_s':end,'cursor_s':end,
                         'af_score':score,'afl_score':0.,'decision':decision,'reason':reason,'tool_calls':calls})
        return rows

    def test_fixed_evidence_reconstructs_and_preserves_every_call(self):
        rows=self.old_rows()
        logs=Logs()
        result=fixed_evidence_record(rows,{'task_id':'t','start_s':0,'end_s':50},EventConfig(),logs,EvidenceContext('run','t','r'))
        self.assertTrue(result['integrity']['identical'])
        self.assertEqual(result['integrity']['source_calls'],sum(len(r['tool_calls']) for r in rows))
        self.assertEqual(result['intervals'],[[0,30]])
        self.assertEqual(result['events'][0]['end_decided_at_s'],50)
        self.assertTrue(all(r['source_evidence_id'] for r in logs.rows['evidence']))
        rows[0]['decision']='negative'
        with self.assertRaisesRegex(ValueError,'reconstruction'):
            fixed_evidence_record(rows,{'task_id':'t','start_s':0,'end_s':50},EventConfig(),Logs(),EvidenceContext('run','t','r'))

    def test_full_replay_exposes_no_future_waveform_or_score(self):
        with TemporaryDirectory() as tmp:
            cache=Path(tmp)
            record={'record_id':'00001','source_rate_hz':250,'source_samples':10000,'source_record_samples':10000}
            rows=[]
            waves=[]
            for step in range(4):
                for lead in ('ECG1','ECG2'):
                    rows.append({'record_id':'00001','channel':lead,'start_sample':step*2500,
                                 'end_sample':(step+1)*2500,'status':'ok','waveform_row':len(waves),'invalid_spans':[]})
                    waves.append(np.full(5000,.9 if step<2 else np.nan,dtype=np.float32))
            np.save(cache/'00001.npy',np.stack(waves))
            (cache/'00001.windows.jsonl').write_text('\n'.join(map(json.dumps,rows)),encoding='utf-8')
            raw=np.column_stack([np.sin(np.arange(10000)/10.)]*2)
            def run(future):
                raw[5000:]=future
                obj=SimpleNamespace(fs=250,sig_name=['ECG1','ECG2'],sig_len=10000,p_signal=raw,units=['mV','mV'])
                logs=Logs()
                scores=[]
                def score(lead,values,location):
                    self.assertLess(location,4)
                    self.assertTrue(np.isfinite(values).all())
                    scores.append((lead,location))
                    return .9,0.
                with patch('cardioagent.ecg_agent.benchmarks.event_replay.wfdb.rdrecord',return_value=obj):
                    result=replay_event_record(cache,record,{'task_id':'t','start_s':0,'end_s':20},'ECG1',
                                                EventConfig(),score,logs,EvidenceContext('run','t','r'),raw_root=cache)
                for group in logs.rows.values():
                    for r in group:
                        r.pop('evidence_id',None);r.pop('evidence_ids',None);r.pop('duration_ms',None)
                result.pop('elapsed_ms')
                result['counts'].pop('tool_ms',None);result['counts'].pop('other_tool_ms',None)
                return result,scores,logs.rows['window_trace']
            self.assertEqual(run(0),run(np.nan))
