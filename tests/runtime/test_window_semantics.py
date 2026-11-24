import unittest
from cardioagent.ecg_agent.harness.window_semantics import canonical_arguments, window_request


class WindowSemanticsTests(unittest.TestCase):
    def test_mask_preserves_only_observed_data(self):
        data = dict(window_s=[0,10], cursor_s=10, primary_lead='ECG2', measurements=[{'id':'E1'}],
                    recent_decisions=['support'], recent_window_votes={'hash':'support'}, candidate={'event_id':1},
                    pending_question='continue?', open_event={'event_id':1})
        masked, _, _ = window_request(data, {}, '', 'masked')
        self.assertEqual(set(masked), {'window_s','cursor_s','primary_lead','measurements'})
        self.assertEqual(masked['measurements'], data['measurements'])
        self.assertIn('candidate', data)

    def test_structured_direction_conflict_rejected(self):
        args = dict(finding='EVIDENCE_WEIGHS_AGAINST_BOTH', measurement_direction='TOWARD_TARGET',
                    evidence_ids=['E1'], reason='Measured both outputs')
        with self.assertRaises(ValueError):
            canonical_arguments(args, 'explicit_masked')
        args['measurement_direction']='AWAY_FROM_TARGET'
        self.assertEqual(canonical_arguments(args,'explicit_masked')['interpretation'],'negative')

    def test_verified_uses_only_current_finite_founder_measurements(self):
        from cardioagent.ecg_agent.harness.window_semantics import verify_band
        profile = {'edges':[.2,.8], 'bins':{'ECG1':[{'direction':v} for v in ('negative','indeterminate','support')]}}
        def row(af,afl,span):
            return dict(tool='score_ecgfounder',status='ok',direction_allowed=True,lead='ECG1',
                        fields=dict(af_score=af,afl_score=afl),window_s=span)
        rows={'current':row(.12,.96,[10,20]),'earlier':row(.02,.01,[0,10])}
        args=dict(interpretation='support',evidence_ids=['current','earlier'])
        verify_band(args,rows,[10,20],profile)
        rows['current']=row(.16,.01,[10,20])
        with self.assertRaises(ValueError):
            verify_band(args,rows,[10,20],profile)
        rows['current']=row(.16,.01,[20,30])
        with self.assertRaises(ValueError):
            verify_band(args,rows,[10,20],profile)
        verify_band(dict(interpretation='unavailable',evidence_ids=[]),rows,[10,20],profile)

    def test_union_input_invariant_to_subtype_names_and_sorted_by_time(self):
        from copy import deepcopy
        tool={'function':{'parameters':{'properties':{'evidence_ids':{},'reason':{}}}}}
        def row(start,af,afl):
            return dict(id=str(start),lead='ECG1',window_s=[start,start+10],status='ok',
                        tool='score_ecgfounder',fields={'af_score':af,'afl_score':afl})
        data=dict(window_s=[10,20],cursor_s=20,primary_lead='ECG1',measurements=[row(10,.1,.95),row(0,.05,.08)])
        flip=deepcopy(data)
        for r in flip['measurements']:
            f=r['fields']
            f['af_score'],f['afl_score']=f['afl_score'],f['af_score']
        a=window_request(data,tool,'','union_max')
        self.assertEqual(a,window_request(flip,tool,'','union_max'))
        self.assertEqual([r['fields']['target_score'] for r in a[0]['rhythm_measurements_by_time']],[.08,.95])
        self.assertNotEqual(window_request(data,tool,'','union_raw'),window_request(flip,tool,'','union_raw'))

    def test_mixed_citation_retry_identifies_each_current_lead_without_using_history(self):
        from cardioagent.ecg_agent.harness.window_semantics import verify_band
        bins = [dict(direction=v) for v in ('negative', 'indeterminate', 'support')]
        profile = dict(edges=[0.2, 0.8], bins=dict(ECG1=bins, ECG2=bins))
        def score(lead, value, window):
            return dict(tool='score_ecgfounder', status='ok', direction_allowed=True,
                        lead=lead, fields=dict(af_score=value, afl_score=0.1), window_s=window)
        now = [10, 20]
        rows = {'E1': score('ECG2', 0.89, now), 'E2': score('ECG1', 0.63, now),
                'H1': score('ECG1', 0.12, [0, 10])}
        with self.assertRaises(ValueError) as rejected:
            verify_band(dict(interpretation='support', evidence_ids=['E1', 'E2', 'H1']), rows, now, profile)
        self.assertIn('E1 (ECG2): support, E2 (ECG1): indeterminate', str(rejected.exception))
        self.assertNotIn('H1', str(rejected.exception))
        verify_band(dict(interpretation='support', evidence_ids=['E1']), rows, now, profile)
        with self.assertRaisesRegex(ValueError, 'requires negative'):
            verify_band(dict(interpretation='negative', evidence_ids=['E1']), rows, now, profile)

    def test_uncertain_window_can_investigate_but_only_grounded_pair_closes(self):
        from tests.runtime import test_autonomous_events as fixture
        from tests.runtime.test_react import reply
        from tests.runtime.test_staged_review import act, boundary

        def observe(label):
            return reply('assess_window', dict(finding=label,evidence_ids=['E1'],reason='Current measurement'))
        cfg = fixture.ROOT/'configs/ecg/react_qwen_window_reference_pilot.json'
        for outcome, pair, closes in [('unlocalized',[.1,.04],False),
                                       ('located',[.95,.1],False),('located',[.1,.04],True)]:
            end = boundary(outcome, **({'before_id':'H1','after_id':'E1'} if outcome=='located' else {}))
            actions = [fixture.measure(0),fixture.FINISH,observe('TARGET_PRESENT'),
                       act('CONFIRM_EVENT',None),boundary('unlocalized'),fixture.measure(10),
                       fixture.FINISH,observe('CANNOT_TELL'),act('REQUEST_CLOSE',1),end]
            if outcome=='located' and not closes:
                actions += [end]
            result,trace,*_ = fixture.EpisodeTests().replay('fixed',actions,2,config_file=cfg,
                                                           score_pairs=[[.1,.95],pair])
            self.assertEqual(trace[-1]['action'],'REQUEST_CLOSE')
            self.assertEqual(result['events'][0]['final_status']=='CLOSED',closes)
            self.assertEqual(bool(result['pending_questions']),not closes)

        expanded_cfg=fixture.ROOT/'configs/ecg/react_qwen_end_range_pilot.json'
        for config_file, expanded in [(cfg,False),(expanded_cfg,True)]:
            actions=[fixture.measure(0),fixture.FINISH,observe('TARGET_PRESENT'),act('CONFIRM_EVENT',None),
                     boundary('unlocalized'),fixture.measure(10),fixture.FINISH,observe('CANNOT_TELL'),
                     act('REQUEST_CLOSE',1),boundary('unlocalized')]
            *_, requests=fixture.EpisodeTests().replay('fixed',actions,2,config_file=config_file,
                                                       score_pairs=[[.1,.95],[.12,.06]])
            prompts=[r['messages'][0]['content'] for r in requests if r['tools'][0]['function']['name']=='refine_boundary']
            self.assertEqual(['a TIME RANGE' in p for p in prompts],[False,expanded])
