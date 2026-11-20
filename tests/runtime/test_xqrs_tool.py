import unittest
from unittest.mock import patch

import numpy as np

from cardioagent.common.recording import Waveform
from cardioagent.ecg_agent.harness import ToolContext, ToolSession, build_registry


NAME = 'detect_beats_xqrs_proxy'
ARGS = {'lead': 'ECG1', 'start_s': 100., 'end_s': 110.}


def session(*, future=0., enabled=True):
    fs = 250.
    x = np.sin(np.arange(5000) / fs * 2 * np.pi)
    x[2500:] += future
    wave = Waveform(x[:, None], fs, ('ECG1',), ('mV',))
    return ToolSession(ToolContext(wave, 100., 110.), allow_experimental=enabled)


class XQRSEvidenceTests(unittest.TestCase):
    def test_experimental_gate_and_schema(self):
        self.assertEqual(session(enabled=False).call(NAME, ARGS)['error']['code'], 'experimental_disabled')
        self.assertNotIn(NAME, [x['name'] for x in build_registry().schemas()])
        self.assertIn(NAME, [x['name'] for x in build_registry().schemas(include_experimental=True)])

    def test_uses_only_elapsed_window_and_keeps_raw_intervals(self):
        seen = []

        class Detector:
            def __init__(self, sig, fs):
                seen.append(sig.copy())
                self.qrs_inds = np.array([125, 375, 625, 875, 1125])

            def detect(self, verbose=False):
                self.verbose = verbose

        with patch('cardioagent.ecg_agent.tools.xqrs.processing.XQRS', Detector):
            first = session().call(NAME, ARGS)
            second = session(future=99.).call(NAME, ARGS)
        self.assertEqual(first['status'], 'ok')
        self.assertEqual(first['result'], second['result'])
        self.assertEqual(len(seen), 2)
        np.testing.assert_array_equal(seen[0], seen[1])
        self.assertEqual(first['result']['r_peaks_s'], [100.5, 101.5, 102.5, 103.5, 104.5])
        self.assertEqual(first['result']['rr_intervals_ms'], [1000.] * 4)
        self.assertEqual(first['result']['cleaning_keep_mask'], [True] * 4)

    def test_missing_flat_failure_and_cache(self):
        with patch('cardioagent.ecg_agent.tools.xqrs.processing.XQRS') as detector:
            flat = session()
            flat.context.waveform.values[:] = 0
            self.assertEqual(flat.call(NAME, ARGS)['status'], 'insufficient')
            detector.assert_not_called()
        class Detector:
            def __init__(self, **kwargs):
                self.qrs_inds = np.array([125, 375, 625, 875])
            def detect(self, verbose=False):
                pass
        with patch('cardioagent.ecg_agent.tools.xqrs.processing.XQRS', Detector):
            s = session()
            first = s.call(NAME, ARGS)
            first['result']['r_peaks_s'].clear()
            second = s.call(NAME, ARGS)
            self.assertTrue(second['cached'])
            self.assertEqual(len(second['result']['r_peaks_s']), 4)
            self.assertEqual(s.call(NAME, {**ARGS, 'end_s': 111.})['error']['code'], 'unavailable_window')


if __name__ == '__main__':
    unittest.main()
