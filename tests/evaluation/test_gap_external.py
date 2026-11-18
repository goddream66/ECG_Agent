from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import wfdb

from cardioagent.ecg_agent.experiments.gap_external import SOURCE_FS, live_repair
from cardioagent.ecg_agent.harness.historical_window import HistoricalWindow
from cardioagent.ecg_agent.schemas import EvidenceContext


class GapExternalTest(TestCase):
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
