import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from cardioagent.ecg_agent.benchmarks.intervals import interval_metrics,burden_score,validate_intervals,union_seconds
from cardioagent.ecg_agent.benchmarks.evaluate import score_files
from cardioagent.ecg_agent.benchmarks.tasks import write_jsonl


class IntervalBenchmarkTests(unittest.TestCase):
    def test_exact_threshold_and_empty_semantics(self):
        self.assertEqual(interval_metrics([[0,5]],[[0,10]])['f1'],1.)
        self.assertEqual(interval_metrics([[0,4.9]],[[0,10]])['f1'],0.)
        self.assertEqual(interval_metrics([],[])['f1'],1.)
        self.assertEqual(interval_metrics([],[[0,10]])['fn'],1)

    def test_duplicate_predictions_cannot_match_same_event_twice(self):
        r=interval_metrics([[0,10],[0,10]],[[0,10]])
        self.assertEqual((r['tp'],r['fp'],r['fn']),(1,1,0))
        self.assertAlmostEqual(r['f1'],2/3)

    def test_union_and_burden_floor(self):
        self.assertEqual(union_seconds([[0,10],[5,15]]),15)
        self.assertAlmostEqual(burden_score(.25,.20),.75)
        self.assertEqual(burden_score(.02,0),0.)
        self.assertEqual(burden_score(0,0),1.)

    def test_invalid_and_future_intervals_rejected(self):
        for intervals in ([[0,11]],[[5,3]],[[0,float('nan')]],[[True,5]]):
            with self.assertRaises(ValueError):validate_intervals(intervals,0,10)

    def test_file_scoring_perfect_and_missing_task_validation(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            refs=[{'task_id':'a','start_s':0.,'end_s':100.,'intervals':[[10.,30.]],'burden':.2,'tau':.5,'burden_floor':.02},
                  {'task_id':'b','start_s':0.,'end_s':100.,'intervals':[],'burden':0.,'tau':.5,'burden_floor':.02}]
            write_jsonl(root/'refs.jsonl',refs)
            write_jsonl(root/'pred.jsonl',[{'task_id':r['task_id'],'intervals':r['intervals']} for r in refs])
            result=score_files(root/'refs.jsonl',root/'pred.jsonl',root/'perfect')
            self.assertEqual(result['macro_episode_f1'],1.)
            self.assertEqual(result['mean_burden_absolute_error_pp'],0.)
            write_jsonl(root/'pred.jsonl',[{'task_id':'a','intervals':[]}])
            with self.assertRaises(ValueError):score_files(root/'refs.jsonl',root/'pred.jsonl',root/'missing')

    def test_task_import_keeps_answers_out_of_agent_manifest(self):
        import numpy as np
        import wfdb
        from cardioagent.ecg_agent.benchmarks.tasks import prepare,read_jsonl
        with TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw'/'afdb_local';raw.mkdir(parents=True)
            wfdb.wrsamp('demo',fs=100,units=['mV'],sig_name=['II'],
                        p_signal=np.sin(np.arange(4000)/10)[:,None],fmt=['16'],write_dir=str(raw))
            row={'id':'demo-ep','task':'episode_detection','pn_dir':'afdb','record':'demo',
                 't0':0.,'t1':40.,'answer':{'intervals':[[5.,15.]]},
                 'verifier':{'type':'temporal_f1','params':{'tau':.5}}}
            write_jsonl(root/'source.jsonl',[row])
            prepare(root/'source.jsonl',root/'raw',root/'a',duration_s=30.)
            before=(root/'a'/'agent_tasks.jsonl').read_text()
            public=next(iter(read_jsonl(root/'a'/'agent_tasks.jsonl','task_id').values()))
            self.assertNotIn('answer',public);self.assertNotIn('intervals',public);self.assertNotIn('n_episodes',public)
            row['answer']['intervals']=[[12.,32.]]
            write_jsonl(root/'source.jsonl',[row])
            prepare(root/'source.jsonl',root/'raw',root/'b',duration_s=30.)
            self.assertEqual(before,(root/'b'/'agent_tasks.jsonl').read_text())
            reference=next(iter(read_jsonl(root/'b'/'evaluation_references.jsonl','task_id').values()))
            self.assertEqual(reference['intervals'],[[12.,30.]])
            self.assertAlmostEqual(reference['burden'],.6)
