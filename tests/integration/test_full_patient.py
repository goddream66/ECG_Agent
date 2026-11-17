from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np
import wfdb
from cardioagent.ecg_agent.data.build_mcmed import windows,build
from cardioagent.ecg_agent.data.select_full_patient import select_full_patient


class CompletePatientTests(unittest.TestCase):
    def test_all_segments_and_visits_exclusion_and_tail(self):
        from tests.integration.test_build_mcmed import MCMedBuildTests
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';helper=MCMedBuildTests();helper.setup_source(root)
            folder=root/'waveforms/001/99000001/II'
            wfdb.wrsamp('extra',fs=250,units=['mV'],sig_name=['II'],p_signal=np.sin(np.arange(2813)/30)[:,None],fmt=['16'],write_dir=str(folder))
            records,stats=select_full_patient(root,'train',{'p2'},17)
            self.assertEqual(len(records),3);self.assertEqual({r['visit_id'] for r in records},{'99000001','99000002'})
            extra=next(r for r in records if r['segment_id']=='extra')
            spans=list(windows(extra));self.assertEqual(spans[-1]['end_sample'],2813)
            args=helper.args(root,Path(d)/'out','prelabel');args.patients=1
            factory=lambda *a,**kw:(lambda x:np.zeros(150,dtype=np.float32),['label_'+str(i) for i in range(150)],{})
            out=build(args,factory,selected_records=[extra],selection_info=stats)
            import json
            rows=[json.loads(line) for line in (out/'auto_labels.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2);self.assertIsNone(rows[-1]['label'])
            self.assertEqual(rows[-1]['quality_status'],'incomplete_window')
