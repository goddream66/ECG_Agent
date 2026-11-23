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
