import unittest
from collections import Counter
import json

from cardioagent.ecg_agent.harness.end_review import end_evidence
from tests.runtime import test_autonomous_events as fixture
from tests.runtime.test_react import reply
from tests.runtime.test_react import ScriptModel
from cardioagent.ecg_agent.harness.staged_review import assess
from tests.runtime.test_staged_review import act, boundary


class EndFollowupTests(unittest.TestCase):
    def test_direction_only_view_preserves_score_timing_direction_and_archive(self):
        from cardioagent.ecg_agent.harness.window_semantics import direction_only_view, REFERENCE_DESCRIPTION
        row = dict(id='H1', lead='ECG2', window_s=[10,20],fields={'target_score':.855},
                   empirical_reference=dict(direction='support',target_fraction=.9829))
        data = dict(vote_on_this_interval=[],earlier_measurements_for_comparison_only=[row])
        view, prompt = direction_only_view(data, REFERENCE_DESCRIPTION)
        actual, = view['earlier_measurements_for_comparison_only']
        self.assertEqual(actual,dict(row,empirical_reference=dict(evidence_for_this_window='FAVORS_AF_OR_AFL')))
        self.assertIn('target_fraction',row['empirical_reference'])
        self.assertIn('same lead',prompt)
    def test_indeterminate_measurement_never_offers_an_unexecutable_direction(self):
        from cardioagent.ecg_agent.harness.window_semantics import available_findings
        profile=json.loads((fixture.ROOT/'configs/ecg/react_qwen_followup_close.json').read_text())['window_band_profile']
        row=dict(id='E1',tool='score_ecgfounder',lead='ECG2',window_s=[10,20],direction_allowed=True,
                 status='ok',fields=dict(af_score=.667,afl_score=.61))
        self.assertEqual(available_findings({'E1':row},[10,20],profile),['CANNOT_TELL'])
        row['fields']['af_score']=.99
        self.assertIn('TARGET_PRESENT',available_findings({'E1':row},[10,20],profile))
        self.assertEqual(available_findings({'E1':row},[20,30],profile),['CANNOT_TELL'])
    def test_disagreeing_measured_leads_cannot_be_hidden_by_selective_citation(self):
        cfg=json.loads((fixture.ROOT/'configs/ecg/react_qwen_followup_leads.json').read_text())
        rows=[dict(id=k,tool='score_ecgfounder',lead=lead,window_s=[10,20],direction_allowed=True,
                   status='ok',fields=dict(af_score=af,afl_score=.05))
              for k,lead,af in [('E1','ECG1',.98),('E2','ECG2',.10)]]
        data=dict(window_s=[10,20],cursor_s=20,primary_lead='ECG1',measurements=rows)
        model=ScriptModel([reply('assess_window',dict(finding='TARGET_PRESENT',evidence_ids=['E1'],reason='one lead high')),
                           reply('assess_window',dict(finding='CONFLICTING_MEASUREMENTS',evidence_ids=['E1','E2'],reason='opposing directions'))])
        steps=[]
        out=assess(model,Counter(),lambda _,s:steps.append(s),{},data,{'E1':'hash1','E2':'hash2'},
                   lambda x:True,lambda x:True,'union_reference',cfg['window_band_profile'],True)
        self.assertEqual([s['status'] for s in steps],['stage_rejected','stage_committed'])
        self.assertEqual(out['interpretation'],'conflict')
    def test_overlaps_gaps_conflicts_and_future(self):
        def row(a,b,d):
            return dict(id=f'{a}-{b}-{d}',window_s=[a,b], reference=dict(direction=d))
        support=row(0,10,'support')
        neg=row(10,20,'negative')
        later=row(20,30,'negative')
        def eligible(seq):
            return end_evidence(seq,[20,30],dict(estimated_start_s=0))['can_confirm']
        self.assertTrue(eligible([later,support,neg]))
        for seq in ([support,neg,neg],[support,later],[support,row(15,25,'negative'),later],
                    [support,neg,row(20,30,'support'),later],[support,neg,row(30,40,'negative')]):
            self.assertFalse(eligible(seq))

    def test_full_runtime_preserves_identity_and_projects_earlier_boundary(self):
        def observe(result):
            return reply('assess_window', dict(finding=result,evidence_ids=['E1'],reason='current evidence'))
        cfg=fixture.ROOT/'configs/ecg/react_qwen_followup_close.json'
        for recovers in (True,False):
            actions=[fixture.measure(0),fixture.FINISH,observe('TARGET_PRESENT'),act('CONFIRM_EVENT',None),
                     boundary('unlocalized'),fixture.measure(10),fixture.FINISH,observe('TARGET_ABSENT'),
                     act('REQUEST_CLOSE',1),fixture.measure(20),fixture.FINISH]
            if recovers:
                actions.extend([observe('TARGET_PRESENT'),act('CONTINUE_EVENT',1)])
            else:
                actions.extend([observe('TARGET_ABSENT'),act('REQUEST_CLOSE',1),
                                reply('confirm_event_end',dict(decision='CONFIRM_END',reason='continued contrary measurements'))])
            result,trace,*_=fixture.EpisodeTests().replay('fixed',actions,3,config_file=cfg,
                              score_pairs=[[.99,.1],[.1,.02],[.97 if recovers else .12,.03]])
            event,=result['events']
            self.assertEqual(trace[1]['state'],'OPEN')
            self.assertEqual(event['final_status'],'RIGHT_CENSORED' if recovers else 'CLOSED')
            self.assertEqual(event['end_decided_at_s'],None if recovers else 30)
            if not recovers:
                self.assertEqual(event['estimated_end_s'],10)
                refinement=trace[2]['llm_action']['refinement']
                anchors=event['end_evidence_by_role']
                self.assertEqual(anchors,dict(last_support=refinement['before_id'],
                                             first_against=refinement['after_id'],
                                             confirming_against=refinement['confirmation_evidence_id']))
                self.assertEqual(len(set(anchors.values())),3)
                self.assertTrue(set(anchors.values()).issubset(event['end_evidence_ids']))
