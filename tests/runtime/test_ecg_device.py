from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import importlib.util
import unittest

import numpy as np
import torch

from cardioagent.ecg_agent.models.ecgfounder import (
    cached_ecgfounder_scorer, ecgfounder_execution, normalize_ecg_device,
)


def runner():
    from cardioagent.ecg_agent.experiments import runner as module
    return module


class TinyECG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(2))
        self.seen = []

    def forward(self, x):
        self.seen.append((tuple(x.shape), str(x.device)))
        mean = x.mean(dim=(1, 2))
        return torch.stack((mean, mean + 1), dim=1) + self.offset, mean[:, None]


class ECGDeviceTests(unittest.TestCase):
    def test_single_device_names_only(self):
        self.assertEqual(normalize_ecg_device('cuda'), 'cuda:0')
        self.assertEqual(normalize_ecg_device('cuda:03'), 'cuda:3')
        self.assertEqual(normalize_ecg_device('cpu'), 'cpu')
        for name in ('cuda:0,1', 'cuda:-1', 'cuda:', 'mps', 'auto', 'cpu:1'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                normalize_ecg_device(name)

    def test_cuda_unavailable_fails_before_loading_weights(self):
        with patch('torch.cuda.is_available', return_value=False), patch('cardioagent.ecg_agent.models.ecgfounder.load_ecgfounder') as load:
            with self.assertRaisesRegex(RuntimeError, 'CUDA was requested'):
                cached_ecgfounder_scorer(Path('.'), device='cuda:0')
            load.assert_not_called()

    def test_device_outside_visibility_is_rejected(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1):
            with self.assertRaisesRegex(ValueError, 'visible CUDA'):
                ecgfounder_execution('cuda:1')

    def test_cpu_scorer_keeps_window_shape_scores_and_readonly_inputs(self):
        model = TinyECG()
        labels = ['ATRIAL FIBRILLATION', 'ATRIAL FLUTTER']
        values = np.zeros(5000, dtype=np.float32)
        values.flags.writeable = False
        with patch('cardioagent.ecg_agent.models.ecgfounder.load_ecgfounder', return_value=(model, labels, Path('weights.pth'))):
            predict, info = cached_ecgfounder_scorer(Path('.'))
            scores = predict(values)
            self.assertEqual(info['device'], 'cpu')
            np.testing.assert_allclose(scores, [.5, torch.sigmoid(torch.tensor(1.)).item()], rtol=0, atol=1e-7)
            self.assertEqual(model.seen, [((1, 1, 5000), 'cpu')])
            self.assertFalse(model.training)
            with self.assertRaises(ValueError):
                predict(np.zeros(2500, dtype=np.float32))
            with self.assertRaises(ValueError):
                predict(np.zeros(5000, dtype=np.float64))
            self.assertEqual(len(model.seen), 1)

    def test_cuda_selects_one_process(self):
        module = runner()
        with patch('cardioagent.ecg_agent.harness.execution.ecgfounder_execution', return_value={'device': 'cuda:0'}):
            info, workers = module.execution_settings('live', 'cuda', 8)
        self.assertEqual(workers, 1)
        self.assertEqual(info['requested_workers'], 8)
        self.assertEqual(info['requested_device'], 'cuda:0')

    def test_replays_never_probe_cuda(self):
        module = runner()
        with patch('cardioagent.ecg_agent.harness.execution.ecgfounder_execution', side_effect=AssertionError('must not probe CUDA')):
            for mode in ('cached', 'fixed'):
                info, workers = module.execution_settings(mode, 'cuda:99', 4)
                self.assertEqual(info['device'], 'none')
                self.assertEqual(workers, 4)

    def test_cuda_jobs_reuse_one_model_without_process_pool(self):
        module = runner()
        with TemporaryDirectory() as folder:
            jobs = [{'target': str(Path(folder) / str(i))} for i in range(3)]
            with patch.object(module, 'initialize') as initialize, \
                 patch.object(module, 'run_job', side_effect=lambda job: ({'target': job['target']}, False)), \
                 patch.object(module, 'ProcessPoolExecutor', side_effect=AssertionError('CUDA must not fork')):
                results = list(module.execute_jobs(jobs, 'live', 'models', 2, 'cuda:0', 1))
            initialize.assert_called_once_with('live', 'models', 2, 'cuda:0')
            self.assertEqual([x[0] for x in results], jobs)

    def test_fully_resumed_gpu_jobs_do_not_load_model(self):
        module = runner()
        with TemporaryDirectory() as folder:
            jobs = [{'target': folder}]
            with patch.object(module, 'initialize') as initialize, \
                 patch.object(module, 'run_job', return_value=({'saved': True}, True)):
                result = list(module.execute_jobs(jobs, 'live', 'models', 2, 'cuda:0', 1))
            initialize.assert_not_called()
            self.assertTrue(result[0][2])


if __name__ == '__main__':
    unittest.main()
