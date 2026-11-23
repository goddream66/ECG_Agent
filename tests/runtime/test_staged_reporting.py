import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from tempfile import TemporaryDirectory
import json

spec=spec_from_file_location('report_staged_events',Path(__file__).resolve().parents[2]/'ops/server/report_staged_events.py')
report=module_from_spec(spec)
spec.loader.exec_module(report)


class ReportingTests(unittest.TestCase):
    def test_early_report_and_unlocalized_confirmation_are_distinct(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            def lines(name,items):
                (root/name).write_text(''.join(json.dumps(row)+'\n' for row in items))
            obs=dict(interpretation='support',assessment='SUPPORTS_TARGET',evidence_ids=['e1'],reason='score')
            lines('evidence.jsonl',[dict(evidence_id='e1',lead='ECG1',window_s=[0,10],tool='score_ecgfounder',status='ok',available_at_s=10)])
            lines('llm_steps.jsonl',[dict(run_id='r',record_id='p',task_id='t',status='stage_committed')])
            win=[]
            raw=[]
            for t,action in [(10,'CREATE_CANDIDATE'),(20,'CONFIRM_EVENT')]:
                win.append(dict(start_s=t-10,cursor_s=t,window_s=[t-10,t],llm_action=dict(window_assessment=obs,
                                lifecycle=dict(status='accepted',action=action),refinement=dict(status='unlocalized'))))
                raw.append(dict(stage='window',cursor_s=t,window_s=[t-10,t],**obs))
            lines('window_trace.jsonl',win)
            lines('assessments.jsonl',raw)
            (root/'result.json').write_text(json.dumps(dict(counts=dict(review_wakes=2),candidates=[
                dict(event_id=1,detected_at_s=10,status='confirmed')],events=[dict(event_id=1,
                detected_at_s=10,confirmed_at_s=20,final_status='RIGHT_CENSORED',right_censored=True,
                onset_status='unlocalized',estimated_start_s=0)])))
            metrics=report.analyze(root,dict(intervals=[[14,25]]))
            self.assertEqual(metrics['detection']['numerator'],0)
            self.assertEqual(metrics['confirmation']['numerator'],1)
            self.assertEqual(metrics['unlocalized_rate']['rate'],1)
            self.assertIsNone(metrics['onset_mae_s'])
            self.assertEqual(metrics['episode_records'][0]['window_finding_ids'],['r/t/p/w/10','r/t/p/w/20'])

