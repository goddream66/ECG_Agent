import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import wfdb
from cardioagent.ecg_agent.data.build_mcmed import build,candidates,select_segments,windows,LABEL_MAPPING,automatic_labels


class MCMedBuildTests(unittest.TestCase):
    def setup_source(self,root):
        root.mkdir()
        (root/'visits.csv').write_text('MRN,CSN,Dx_name\np1,99000001,SECRET\np1,99000002,SECRET\np2,99000003,SECRET\n')
        (root/'split_random_train.csv').write_text('99000001\n99000002\n99000003\n')
        (root/'split_random_test.csv').write_text('99000003\n')
        for name in ('99000001','99000002','99000003'):
            folder=root/'waveforms'/name[-3:]/name/'II';folder.mkdir(parents=True)
            t=np.arange(5000)/250.;x=sum(np.exp(-((t-p)/.025)**2) for p in np.arange(.5,20,1))
            wfdb.wrsamp(name+'_0',fs=250,units=['mV'],sig_name=['II'],p_signal=x[:,None],fmt=['16'],write_dir=str(folder))

    def args(self,root,out,mode='index'):
        return argparse.Namespace(source_root=root,output_root=out,split='train',patients=2,duration_s=20,
            seed=17,max_patient_scan=10,mode=mode,device='cpu',models=out/'models')

    def test_selection_patient_isolation_and_window_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';self.setup_source(root)
            a,_=select_segments(root,'train',2,20,17,10);b,_=select_segments(root,'train',2,20,17,10)
            self.assertEqual(a,b);self.assertEqual(len({r['patient_id'] for r in a}),2)
            w=list(windows(a[0]));self.assertEqual(w[1]['start_sample'],2500);self.assertEqual(w[1]['end_s'],20.)

    def test_index_never_executes_model_or_contains_diagnoses(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';self.setup_source(root)
            out=build(self.args(root,Path(d)/'out'),lambda *a,**k:self.fail('model loaded'))
            self.assertFalse((out/'model_scores').exists())
            self.assertNotIn('SECRET',(out/'records.json').read_text()+(out/'windows.jsonl').read_text())
            self.assertTrue(json.loads((out/'manifest.json').read_text())['selection_complete'])

    def test_missing_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';self.setup_source(root)
            for p in root.rglob('*.dat'):p.unlink()
            records,stats=select_segments(root,'train',2,20,17,10)
            self.assertFalse(records);self.assertTrue(stats['source_rejections'])

    def test_prelabel_artifacts_and_human_labels_remain_empty(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';self.setup_source(root);calls=[]
            labels=list(dict.fromkeys(n for ns in LABEL_MAPPING.values() for n in ns))
            labels += ['other_'+str(i) for i in range(150-len(labels))]
            def factory(*a,**kw):
                def predict(x):
                    calls.append(x.copy())
                    row=np.full(150,.1,dtype=np.float32);row[labels.index('other_42')]=.9;return row
                return predict,labels,{'device':'cpu','test_double':True}
            args=self.args(root,Path(d)/'out','prelabel');args.patients=1
            out=build(args,factory)
            self.assertEqual(len(calls),2);self.assertEqual(calls[0].shape,(5000,))
            rows=[json.loads(x) for x in (out/'windows.jsonl').read_text().splitlines()]
            self.assertFalse((out/'model_scores').exists())
            auto=[json.loads(x) for x in (out/'auto_labels.jsonl').read_text().splitlines()]
            self.assertTrue(all(x['labels'][0]=='other_42' and len(x['display_candidates'])==5 and len(x['ranked_candidates'])==150 for x in auto))
            self.assertTrue(all(x['label_status']=='assigned' and not x['human_verified'] for x in auto))
            review=[json.loads(x) for x in (out/'review/annotations.jsonl').read_text().splitlines()]
            self.assertTrue(all(x['human_labels'] is None for x in review))
            evidence=[json.loads(x) for x in (out/'tool_evidence.jsonl').read_text().splitlines()]
            self.assertTrue(all(x['window_s'][1]<=x['available_at_s'] for x in evidence))

    def test_no_target_is_not_normal_and_conflicts_are_retained(self):
        labels=list(dict.fromkeys(n for ns in LABEL_MAPPING.values() for n in ns))
        x=np.zeros(len(labels));r=candidates(x,labels,.5)
        self.assertEqual(r['review_state'],'no_target_candidate');self.assertFalse(r['ground_truth'])
        x[labels.index('ATRIAL FIBRILLATION')]=.9;x[labels.index('NORMAL ECG')]=.9
        self.assertEqual(candidates(x,labels,.5)['review_state'],'conflicting_candidates')

    def test_outputs_never_overwrite_and_cannot_live_in_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';self.setup_source(root);args=self.args(root,Path(d)/'out')
            a=build(args);b=build(args);self.assertNotEqual(a,b)
            args.output_root=root/'output'
            with self.assertRaises(ValueError):build(args)


    def test_full_head_retains_ranked_candidates_and_first_score(self):
        labels=['NORMAL ECG','ATRIAL FIBRILLATION']+['label_'+str(i) for i in range(148)]
        x=np.zeros(150);x[0]=.9
        self.assertEqual(automatic_labels(x,labels)['top1_label'],'NORMAL ECG')
        x[1]=.8;x[149]=.7
        r=automatic_labels(x,labels)
        self.assertEqual(r['label_ids'][:3],[0,1,149]);self.assertAlmostEqual(r['top1_score'],.9)

    def test_top1_has_no_threshold_and_ties_use_vocabulary_order(self):
        r=automatic_labels(np.full(150,.001),['label_'+str(i) for i in range(150)])
        self.assertEqual(r['labels'],['label_'+str(i) for i in range(5)]);self.assertEqual(r['label_status'],'assigned')

    def test_full_scores_can_be_retained_explicitly(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';self.setup_source(root)
            labels=['NORMAL ECG']+['label_'+str(i) for i in range(149)]
            def factory(*args,**kw):return lambda x:np.zeros(150,dtype=np.float32),labels,{'test_double':True}
            args=self.args(root,Path(d)/'out','prelabel');args.patients=1;args.save_scores=True
            out=build(args,factory)
            self.assertEqual(np.load(next((out/'model_scores').glob('*.npy'))).shape,(2,150))


    def test_missing_or_flat_windows_never_load_founder(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';self.setup_source(root)
            args=self.args(root,Path(d)/'out','prelabel');args.patients=1
            original=wfdb.rdrecord
            def flat(*a,**kw):
                record=original(*a,**kw);record.p_signal[:]=0.;return record
            with patch('wfdb.rdrecord',side_effect=flat):
                out=build(args,lambda *a,**kw:self.fail('Founder loaded on flat window'))
            auto=[json.loads(x) for x in (out/'auto_labels.jsonl').read_text().splitlines()]
            self.assertTrue(all(x['label'] is None for x in auto))
            self.assertTrue(all(x['quality_status']=='unusable_for_auto_label' for x in auto))
            summary=json.loads((out/'label_summary.json').read_text())
            self.assertEqual(summary['counts']['model_skipped'],2)

    def test_preprocessing_failure_is_retained_without_model_label(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';self.setup_source(root)
            args=self.args(root,Path(d)/'out','prelabel');args.patients=1
            with patch('cardioagent.ecg_agent.data.build_mcmed.preprocess_window',side_effect=ValueError('degenerate')):
                out=build(args,lambda *a,**kw:self.fail('Founder loaded on failed preprocessing'))
            auto=[json.loads(x) for x in (out/'auto_labels.jsonl').read_text().splitlines()]
            self.assertTrue(all(x['label'] is None and x['quality_status']=='preprocessing_failed' for x in auto))
            self.assertEqual(json.loads((out/'manifest.json').read_text())['status'],'complete')


    def test_broad_summary_does_not_hide_specific_candidates(self):
        labels=['ABNORMAL ECG','ATRIAL FIBRILLATION','NORMAL ECG','BORDERLINE ECG']+['other_'+str(i) for i in range(146)]
        x=np.zeros(150);x[:4]=[.99,.8,.6,.9]
        r=automatic_labels(x,labels)
        self.assertEqual(r['top1_label'],'ABNORMAL ECG');self.assertAlmostEqual(r['top1_score'],.99)
        self.assertEqual(r['display_candidates'][0]['label'],'ATRIAL FIBRILLATION')
        self.assertEqual(r['display_candidates'][0]['rank'],3)
        self.assertNotIn('ABNORMAL ECG',r['labels']);self.assertIn('NORMAL ECG',r['labels'])
        self.assertEqual(len(r['ranked_candidates']),150);self.assertIsNone(r['threshold'])
        self.assertEqual(r['summary_candidates'][0]['label'],'ABNORMAL ECG')
        with self.assertRaises(ValueError):automatic_labels(np.full(150,np.nan),labels)
