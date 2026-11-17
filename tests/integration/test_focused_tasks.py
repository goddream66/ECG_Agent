import unittest
from cardioagent.ecg_agent.data.label_mcmed_tasks import task_states,event_candidates,TASKS

class FocusedTaskTests(unittest.TestCase):
    def test_tasks_are_independent(self):
        r=task_states(.8,120,'no_quality_flag')
        self.assertEqual(r['af']['state'],'candidate_positive')
        self.assertEqual(r['high_rate']['state'],'candidate_positive')
        self.assertEqual(r['low_rate']['state'],'candidate_negative')
        self.assertEqual(set(r),set(TASKS))
    def test_boundaries_and_uncertainty(self):
        self.assertEqual(task_states(.4,60,'no_quality_flag')['af']['state'],'uncertain')
        self.assertEqual(task_states(.6,100,'no_quality_flag')['af']['state'],'candidate_positive')
        for rate in (60,100):
            r=task_states(.1,rate,'no_quality_flag')
            self.assertEqual(r['high_rate']['state'],'candidate_negative')
            self.assertEqual(r['low_rate']['state'],'candidate_negative')
        self.assertEqual(task_states(.1,59,'no_quality_flag')['low_rate']['state'],'candidate_positive')
    def test_quality_review_retains_proposal_and_unknown_is_not_negative(self):
        r=task_states(.8,120,'review_required')
        self.assertEqual(r['af']['state'],'uncertain')
        self.assertEqual(r['af']['proposed_state'],'candidate_positive')
        for q in ('no_quality_flag','incomplete_window','source_unavailable'):
            r=task_states(None,None,q)
            self.assertTrue(all(x['state']=='unknown' for x in r.values()))
        self.assertTrue(all(x['state']=='unknown' for x in task_states(.9,150,'incomplete_window').values()))
    def test_events_do_not_bridge_unknown_or_segments(self):
        rows=[]
        for i in range(7):
            row={'patient_id':'p','visit_id':'v','segment_id':'a' if i<6 else 'b',
                 'start_s':i*10.,'end_s':(i+1)*10.,'window_id':str(i),
                 'tasks':task_states(.8 if i!=3 else None,80,'no_quality_flag')}
            rows.append(row)
        es=event_candidates(rows)
        self.assertEqual([e['duration_s'] for e in es],[30.,20.,10.])
        self.assertEqual([e['meets_duration_rule'] for e in es],[True,False,False])
        self.assertTrue(all(not e['human_verified'] for e in es))
