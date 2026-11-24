from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import wfdb

from cardioagent.ecg_agent.experiments.gap_external import SOURCE_FS, live_repair
from cardioagent.ecg_agent.experiments.ltafdb_adaptive_pilot import select_clips
from cardioagent.ecg_agent.experiments.runner import run_job, validate_dataset
from cardioagent.ecg_agent.data.offline import cache_manifest, write_record
from cardioagent.ecg_agent.harness.historical_window import HistoricalWindow
from cardioagent.ecg_agent.memory.events import EventConfig
from cardioagent.ecg_agent.schemas import EvidenceContext


class GapExternalTest(TestCase):
    def test_external_job_releases_completed_128_hz_windows(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            t = np.arange(40 * SOURCE_FS) / SOURCE_FS
            wave = np.sin(2 * np.pi * 8 * t)
            wfdb.wrsamp('00', SOURCE_FS, ['mV', 'mV'], ['ECG1', 'ECG2'],
                        p_signal=np.column_stack((wave, wave)), fmt=['16', '16'], write_dir=work)
            record = write_record(root / '00', root, 40, source_fs=SOURCE_FS)
            tasks = {'external': dict(task_id='external', record_id='00', source_database='ltafdb',
                                      start_s=10, end_s=30)}
            manifest = cache_manifest(root, [record], 'LTAFDB')
            self.assertEqual(validate_dataset(manifest, tasks, 'ltafdb'), ({'00': record}, SOURCE_FS))
            with self.assertRaisesRegex(ValueError, 'AFDB cache'):
                validate_dataset(manifest, tasks, 'afdb')
            job = dict(target=str(root / 'results' / '00'), cache=work, raw_root=work,
                       source_fs=SOURCE_FS, task=tasks['external'], record=record, channel='ECG1',
                       run_id='test', manifest_id='test', config=EventConfig().to_dict(),
                       mode='live', policy='event-adaptive', device='cpu', threads=2)
            with patch('cardioagent.ecg_agent.experiments.runner._MODEL', return_value=(.95, .01)) as model:
                result, resumed = run_job(job)
            self.assertFalse(resumed)
            self.assertGreater(model.call_count, 0)
            self.assertTrue(all(call.args[0].shape == (5000,) for call in model.call_args_list))
            self.assertEqual(result['intervals'], [[10., 30.]])

    def test_external_selection_keeps_reference_and_stratum_out_of_inference(self):
        tasks = {name: dict(record_id=name, start_s=0, end_s=500) for name in ('00', '01', '02')}
        refs = {'00': {'intervals': [[100, 118]]}, '01': {'intervals': [[100, 220]]}, '02': {'intervals': []}}
        valid = {name: {'valid_intervals': [[0., 500.]]} for name in tasks}
        selected = select_clips(tasks, refs, valid, per_stratum=1)
        self.assertEqual([task['record_id'] for task, _, _, _ in selected], ['00', '01', '02'])
        self.assertEqual([kind for _, _, kind, _ in selected], ['short_event', 'event_end', 'background'])
        for task, reference, _, _ in selected:
            self.assertEqual(task['end_s'] - task['start_s'], 120)
            self.assertEqual(set(task), {'task_id', 'record_id', 'source_database', 'start_s', 'end_s'})
            self.assertEqual(task['task_id'], reference['task_id'])

    def test_real_128_hz_tool_only_uses_selected_arrived_window(self):
        def measured(score):
            return dict(score=[score, .01], score_status='ok', quality='no_warning',
                        score_id='existing-score', quality_id='existing-quality')

        missing_quality = dict(score=[.9, .01], score_status='ok', score_id='high', quality='not_measured')
        missing = dict(score=None, score_status='not_measured', quality='not_measured')
        packet = {'gap_s': [10, 40], 'cursor_s': 50, 'start_threshold': .6, 'windows': [
            {'window_s': [10, 20], 'leads': {'ECG1': measured(.9), 'ECG2': measured(.7)}},
            {'window_s': [20, 30], 'leads': {'ECG1': missing_quality, 'ECG2': missing}},
            {'window_s': [30, 40], 'leads': {'ECG1': measured(.2), 'ECG2': measured(.1)}},
        ]}
        task = dict(record_id='00', start_s=0, end_s=60)
        t = np.arange(60 * SOURCE_FS) / SOURCE_FS
        wave = np.sin(2 * np.pi * 8 * t)
        with TemporaryDirectory() as work:
            wfdb.wrsamp('00', SOURCE_FS, ['mV', 'mV'], ['ECG1', 'ECG2'],
                         p_signal=np.column_stack((wave, wave)), fmt=['16', '16'], write_dir=work)

            def predict(_):
                self.fail('A known high scoring lead needs only a quality check')

            reader = HistoricalWindow(Path(work), task, predict, source_fs=SOURCE_FS)
            identity = EvidenceContext('run', 'task', '00')
            selected = live_repair(packet, reader, identity, 'feasible_adaptive')
            self.assertTrue(selected['link'])
            self.assertEqual(len(selected['actions']), 1)
            action = selected['actions'][0]
            self.assertEqual((action['tool'], action['lead'], action['window_s'], action['available_at_s']),
                             ('assess_quality', 'ECG1', [20, 30], 50))
            with self.assertRaisesRegex(ValueError, 'arrived'):
                reader.session(50, 60, 50, identity)
        self.assertNotIn('quality_id', packet['windows'][1]['leads']['ECG1'])
