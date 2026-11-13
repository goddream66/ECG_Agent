import json
import unittest

import numpy as np

from cardioagent.benchmark import rate_metrics, select_records
from cardioagent.dataset import DATASET_ROOT, load_manifest
from cardioagent.recording import Recording, Waveform
from cardioagent.signal.synchronization import TimeAxis, inspect_axes
from cardioagent.v2 import inspect_recording


class TimelineTests(unittest.TestCase):
    def test_fractional_window_keeps_actual_sample_time(self):
        wave = Waveform(np.arange(20).reshape(-1, 1), 1000, ("I",), ("mV",))
        axis = TimeAxis(1000, 20, start_s=0.1, offset_correction_s=0.02)
        values, first = axis.select(wave, "I", 0.1235, 0.1275)
        self.assertEqual(first, 4)
        np.testing.assert_array_equal(values, [4, 5, 6, 7])
        self.assertAlmostEqual(axis.time_at(first), 0.124)
        np.testing.assert_array_equal(wave.window("I", 0.0035, 0.0075), values)

    def test_cross_modality_timing_requires_documentation_and_matching_grid(self):
        ecg = Waveform(np.ones((6000, 1)), 1000, ("I",), ("mV",))
        ppg = Waveform(np.ones((6001, 1)), 1000, ("p",), ("au",))
        mismatched = Recording("x", {"ecg": ecg, "ppg": ppg}, {})
        matched = Recording("x", {"ecg": ecg, "ppg": ecg}, {})
        self.assertFalse(inspect_axes(mismatched, True)[1]["cross_modality_timing_allowed"])
        self.assertFalse(inspect_axes(matched, False)[1]["cross_modality_timing_allowed"])
        self.assertTrue(inspect_axes(matched, True)[1]["sample_grid_timing_allowed"])
        self.assertFalse(inspect_axes(matched, True)[1]["cross_modality_timing_allowed"])
        calibration = {"record_id": "x", "method": "shared_hardware_marker", "reference": "test fixture",
                       "modalities": {"ecg": {"latency_s": 0, "uncertainty_s": 0.001},
                                      "ppg": {"latency_s": 0.018, "uncertainty_s": 0.002}}}
        axes, timing = inspect_axes(matched, True, calibration)
        self.assertTrue(timing["cross_modality_timing_allowed"])
        self.assertAlmostEqual(axes["ppg"].time_at(18), 0)


class BenchmarkTests(unittest.TestCase):
    def test_low_error_on_accepted_records_does_not_hide_refusals(self):
        rows = [{"reference_bpm": 100, "rate_bpm": x} for x in (None, 100, 50, None)]
        metrics = rate_metrics(rows, "rate_bpm")
        self.assertEqual(metrics["mae_bpm"], 25)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["within_5_bpm_of_all"], 0.25)
        self.assertEqual(metrics["possible_octave_errors"], 1)


@unittest.skipUnless((DATASET_ROOT / "manifest.json").is_file(), "local SensSmartTech release not imported")
class ReleaseIntegrationTests(unittest.TestCase):
    def test_manifest_has_no_subject_leakage(self):
        manifest = load_manifest(DATASET_ROOT)
        groups = [set(ids) for ids in manifest["subject_ids"].values()]
        self.assertEqual([len(g) for g in groups], [20, 6, 6])
        self.assertEqual(len(set.union(*groups)), 32)
        self.assertIn(1, groups[0])
        self.assertEqual(len(select_records(manifest, "all", None)), 338)
        for row in manifest["records"]:
            self.assertIn(row["subject_id"], manifest["subject_ids"][row["split"]])

    def test_real_fractional_window_and_missing_ppg(self):
        record = Recording.read(DATASET_ROOT / "WFDB", "1_10-09-54", {"ppg"})
        start, end = 0.0005, 10.0005
        report = inspect_recording(record, documented_common_grid=True, start=start, end=end)
        json.dumps(report, allow_nan=False)
        self.assertEqual({row["modality"] for row in report["observations"]}, {"ecg", "pcg"})
        ecg, pcg = report["observations"]
        self.assertGreater(len(ecg["events"]), 5)
        for event in ecg["events"]:
            self.assertTrue(start <= event["time_s"] < end)
            self.assertAlmostEqual(event["time_s"], event["sample_index"] / 1000)
        self.assertIn("S1", pcg["unavailable_events"])
        self.assertIn("S2", pcg["unavailable_events"])
        self.assertEqual(pcg["events"], [])

    def test_post_activity_ecg_has_event_rate(self):
        record = Recording.read(DATASET_ROOT / "WFDB", "1_10-25-13", {"ppg", "pcg"})
        ecg = inspect_recording(record, documented_common_grid=True)["observations"][0]
        self.assertGreater(len(ecg["events"]), 50)
        self.assertAlmostEqual(ecg["rate_bpm"], 129.9, delta=5)


if __name__ == "__main__":
    unittest.main()
