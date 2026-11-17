import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import json
import numpy as np
import wfdb

from cardioagent.ecg_agent.data.label_mcmed_af import classify, intervals, run, timestamp


class AFOnlyLabels(unittest.TestCase):
    def test_binary_target(self):
        self.assertTrue(classify(.1,.81,.8))
        self.assertFalse(classify(.8,.1,.8))
        self.assertTrue(classify(.5,.1,.5,'ge'))
        self.assertFalse(classify(.499,.1,.5,'ge'))
        self.assertTrue(classify(.1,.7,.5,'ge'))
        with self.assertRaises(ValueError):classify(float('nan'),.1,.8)

    def test_interval_unknown_and_negative_gaps(self):
        rows=[]
        for i,state in enumerate([True,True,None,True,False,True]):
            rows.append(dict(patient_id='p',visit_id='v',segment_id='s',window_index=i,
                             af_afl_candidate=state,start_s=i*10,end_s=(i+1)*10,
                             start_datetime=None,end_datetime=None))
        self.assertEqual([(r['start_s'],r['end_s']) for r in intervals(rows)],[(0,20),(30,40),(50,60)])
        self.assertEqual(timestamp('2200-01-01 00:00:00',10),'2200-01-01T00:00:10')

    def test_fixture_binary_export_and_partial_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'source';folder=source/'waveforms'/'123'/'123'/'II';folder.mkdir(parents=True)
            (source/'visits.csv').write_text('MRN,CSN\np,123\n')
            x=np.sin(np.arange(12500)*2*np.pi/500)[:,None]
            wfdb.wrsamp('segment',fs=500,units=['mV'],sig_name=['II'],p_signal=x,fmt=['16'],write_dir=str(folder))
            model=base/'checkpoint';model.write_bytes(b'fixture')
            args=SimpleNamespace(source_root=source,output=base/'output',models=base,patients=['p'],device='cpu',threshold=.8)
            calls=[]
            def factory(*args):
                def predict(values):
                    calls.append(values);return (.9,.1) if len(calls)==1 else (.1,.1)
                return predict,{'checkpoint':str(model)}
            run(args,factory=factory)
            rows=[json.loads(x) for x in (args.output/'windows.jsonl').read_text().splitlines()]
            self.assertEqual([x['af_afl_candidate'] for x in rows],[True,False,None])
            self.assertEqual(len(calls),2)
            self.assertEqual(json.loads((args.output/'manifest.json').read_text())['status'],'complete')
            self.assertEqual(len((args.output/'intervals.jsonl').read_text().splitlines()),1)
            self.assertTrue(all('ranked_candidates' not in row for row in rows))
            args.output=base/'binary_output';args.threshold=.5;args.comparison='ge';args.binary=True;args.cpu_threads=1
            calls.clear()
            run(args,factory=factory)
            rows=[json.loads(x) for x in (args.output/'windows.jsonl').read_text().splitlines()]
            self.assertEqual([x['binary_label'] for x in rows],[1,0,None])
            self.assertEqual([x['label'] for x in rows],['AF/AFL','no_AF/AFL','unknown'])
            manifest=json.loads((args.output/'manifest.json').read_text())
            self.assertEqual(manifest['rule'],'max(AF,AFL) >= threshold')
            self.assertEqual(manifest['cpu_threads'],1)


if __name__=='__main__':unittest.main()
