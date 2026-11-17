import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from cardioagent.ecg_agent.event_state import EventConfig


def runner():
    from cardioagent.ecg_agent.experiments import runner as module
    return module


class EventCheckpointTests(unittest.TestCase):
    def test_live_device_change_does_not_reuse_cpu_checkpoint(self):
        module = runner()
        with TemporaryDirectory() as tmp:
            job = dict(self.make_job(Path(tmp)), mode='live', device='cpu')
            with patch.object(module, '_MODEL', return_value=(.9, .0)) as model, \
                 patch.object(module, '_MODEL_INFO', {'device': 'cpu'}):
                result, _ = module.run_job(job)
                self.assertEqual(result['execution']['device'], 'cpu')
                count = model.call_count
                self.assertGreater(count, 0)
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    module.run_job(dict(job, device='cuda:0'))
                self.assertEqual(model.call_count, count)

    def make_job(self, root):
        cache,scores=root/'cache',root/'scores'
        cache.mkdir();scores.mkdir()
        np.save(cache/'00001.npy',np.ones((4,5000),dtype=np.float32))
        np.save(scores/'00001.npy',np.tile(np.array([[.9,.0]],dtype=np.float32),(4,1)))
        rows=[]
        for step in range(2):
            for channel in ('ECG1','ECG2'):
                rows.append({'record_id':'00001','channel':channel,'start_sample':step*2500,
                             'end_sample':(step+1)*2500,'status':'ok','waveform_row':len(rows),'invalid_spans':[]})
        (cache/'00001.windows.jsonl').write_text('\n'.join(map(json.dumps,rows)),encoding='utf-8')
        return {'target':str(root/'result'),'run_id':'run','manifest_id':'manifest','cache':str(cache),
                'task':{'task_id':'task','start_s':0,'end_s':20},'record':{'record_id':'00001',
                'source_rate_hz':250,'source_samples':5000,'source_record_samples':5000},
                'channel':'ECG1','config':EventConfig().to_dict(),'mode':'cached','policy':'legacy-dual',
                'raw_root':str(cache),'threads':1,'scores':str(scores),'source_trace':'unused'}

    def test_verified_resume_and_corruption_rejection(self):
        module=runner()
        with TemporaryDirectory() as tmp:
            job=self.make_job(Path(tmp))
            first,resumed=module.run_job(job)
            self.assertFalse(resumed)
            second,resumed=module.run_job(job)
            self.assertTrue(resumed)
            self.assertEqual(first,second)
            self.assertEqual(first['intervals'],[[0.,20.]])
            self.assertEqual(first['execution']['actual_model_calls'],0)
            path=Path(job['target'])/'evidence.jsonl'
            path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Corrupt'):
                module.run_job(job)

    def test_changed_configuration_and_incomplete_checkpoint_are_not_reused(self):
        module=runner()
        with TemporaryDirectory() as tmp:
            job=self.make_job(Path(tmp))
            module.run_job(job)
            changed=dict(job,config=EventConfig(exit_windows=1).to_dict())
            with self.assertRaisesRegex(ValueError,'provenance'):
                module.run_job(changed)
            (Path(job['target'])/'result.json').unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                module.run_job(job)
