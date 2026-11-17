import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cardioagent.common.paths import code_root
from cardioagent.ecg_agent.data.label_mcmed import load_config, check_ready, configure_device
from cardioagent.ecg_agent.data.label_quality import summarize_quality


class MCMedLabelPreparationTests(unittest.TestCase):
    def evidence(self, consistent=True, status='ok'):
        return [{'tool': 'assess_template_quality', 'evidence_id': 'quality:1', 'status': status,
                 'result': {'template_consistent': consistent,
                            'prescreen_flags': {'fewer_than_six_peaks': not consistent}}}]

    def test_quality_flags_request_review_without_dropping_possible_rhythm(self):
        for responses in ([], self.evidence(False), self.evidence(status='error')):
            result = summarize_quality([], responses)
            self.assertEqual(result['status'], 'review_required')
            self.assertFalse(result['skip_founder'])
        self.assertEqual(summarize_quality([], self.evidence())['status'], 'no_quality_flag')

    def test_missing_spans_veto_founder_even_with_consistent_template(self):
        result = summarize_quality([(0, 500, 'suspected_flatline')], self.evidence())
        self.assertTrue(result['skip_founder'])
        self.assertEqual(result['quality_evidence_id'], 'quality:1')
        self.assertEqual(result['reasons'], ['suspected_flatline'])

    def test_config_validates_and_is_independent_of_original_profile(self):
        source = code_root()/'configs/ecg/mcmed_labeling.json'
        args = load_config(source, patients=1, duration_s=20, device='cpu')
        self.assertEqual((args.patients, args.duration_s, args.device), (1, 20, 'cpu'))
        with self.assertRaises(ValueError):
            load_config(source, duration_s=11)
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1'}):
            with self.assertRaises(ValueError):
                configure_device('cuda:0')

    def test_preflight_uses_synthetic_signal_without_reading_patient_waveforms(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'waveforms').mkdir()
            (root/'visits.csv').write_text('MRN,CSN\n')
            (root/'split_random_train.csv').write_text('1\n')
            (root/'1_lead_ECGFounder.pth').write_bytes(b'fixture')
            args = load_config(code_root()/'configs/ecg/mcmed_labeling.json', device='cpu')
            args.source_root = args.models = root
            labels = ['label_'+str(i) for i in range(150)]
            def predict(values):
                self.assertEqual(values.shape, (5000,))
                return np.full(150, .01)
            with patch('cardioagent.ecg_agent.data.label_mcmed.create_ecgfounder_label_scorer',
                       return_value=(predict, labels, {'device': 'cpu'})), \
                 patch('wfdb.rdrecord', side_effect=AssertionError('Patient waveform read')):
                result = check_ready(args)
            self.assertEqual(result['status'], 'ready')
            self.assertFalse(result['patient_inference'])
            self.assertFalse(result['dataset_package_created'])
