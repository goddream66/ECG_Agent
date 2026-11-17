import unittest
from cardioagent.ecg_agent.data.quality_reassessment import reassess_window_quality
from cardioagent.ecg_agent.tools.physio import template_correlation_passes

class GroupGateRemovalTests(unittest.TestCase):
    def evidence(self,score,groups,flags=None):
        return {'assess_template_quality':{'tool':'assess_template_quality','status':'ok','tool_version':'1.0.0',
          'evidence_id':'quality-old','duration_ms':10.,'warnings':[],'result':{
          'template_correlation':score,'coefficient_group_count':groups,'template_consistent':False,
          'prescreen_flags':flags or {}}}}
    def test_groups_do_not_gate_anymore_and_source_is_immutable(self):
        for group in (1,2,3,5,10):
            source=self.evidence(.9,group)
            status,reasons,new=reassess_window_quality({'quality_status':'review_required'},source)
            self.assertEqual(status,'no_quality_flag')
            self.assertFalse(source['assess_template_quality']['result']['template_consistent'])
            q=new['assess_template_quality']
            self.assertEqual(q['source_evidence_id'],'quality-old')
            self.assertNotEqual(q['evidence_id'],'quality-old')
            self.assertEqual(q['result']['coefficient_group_count'],group)
            self.assertFalse(q['result']['heuristic_parameters']['group_count_gate_enabled'])
    def test_correlation_boundary_other_flags_and_unknown_are_preserved(self):
        self.assertFalse(template_correlation_passes(.8));self.assertFalse(template_correlation_passes(None))
        self.assertFalse(template_correlation_passes(float('nan')))
        for score,groups,flags in ((.7,1,{}),(.9,3,{'peak_to_peak_over_5mv':True})):
            status,_,_=reassess_window_quality({'quality_status':'review_required'},self.evidence(score,groups,flags))
            self.assertEqual(status,'review_required')
        status,_,_=reassess_window_quality({'quality_status':'incomplete_window'},self.evidence(.99,5))
        self.assertEqual(status,'incomplete_window')
        status,_,_=reassess_window_quality({'quality_status':'review_required'},{})
        self.assertEqual(status,'review_required')
