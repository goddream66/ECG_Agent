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
