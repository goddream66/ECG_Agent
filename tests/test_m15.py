import unittest

import numpy as np

from cardioagent.event_audit import compute_inter_rater_metrics, event_errors
from cardioagent.paired_metrics import paired_metrics
from cardioagent.rate_policy import ecg_first
from cardioagent.recording import Recording, Waveform
from cardioagent.workflow import rate_summary


class M15Tests(unittest.TestCase):
    def test_policy_does_not_measure_auxiliaries_when_ecg_is_usable(self):
        wave = Waveform(np.zeros((10000, 1)), 1000, ("I",), ("mV",))
        record = Recording("case", {"ecg": wave, "ppg": wave, "pcg": wave}, {})
        calls = []

        def observe(modality, channel):
            calls.append(modality)
            return {"modality": modality, "channel": channel, "status": "ok", "quality": "usable",
                    "rate_bpm": 75, "window_s": [0, 10], "evidence_id": modality}

        result = ecg_first(record, 6, observe)
        self.assertEqual(calls, ["ecg"])
        self.assertEqual(result["rate_bpm"], 75)

    def test_missing_ecg_fallback_respects_one_action_cap(self):
        wave = Waveform(np.zeros((10000, 1)), 1000, ("p",), ("au",))
        record = Recording("case", {"ppg": wave, "pcg": wave}, {"ecg": "missing"})
        calls = []

        def observe(modality, channel):
            calls.append(modality)
            return {"modality": modality, "channel": channel, "status": "insufficient", "quality": "review",
                    "rate_bpm": None, "window_s": [0, 10], "evidence_id": modality}

        result = ecg_first(record, 1, observe)
        self.assertEqual(calls, ["ppg"])
        self.assertIsNone(result["rate_bpm"])

    def test_paired_coverage_and_error_are_calculated_on_correct_records(self):
        rows = [
            {"record_id": "a", "subject_id": 1, "reference_bpm": 80, "ecg": None, "ecg_first": 82},
            {"record_id": "b", "subject_id": 1, "reference_bpm": 80, "ecg": 81, "ecg_first": 82},
            {"record_id": "c", "subject_id": 2, "reference_bpm": 80, "ecg": 80, "ecg_first": None},
        ]
        result = paired_metrics(rows, "ecg", "ecg_first")
        self.assertEqual(result["subjects"], 2)
        self.assertEqual(result["shared_scored"], 1)
        self.assertEqual(result["candidate_minus_anchor_mae_bpm"], 1)
        self.assertEqual(result["coverage_delta"], 0)
        self.assertEqual(result["only_candidate_scored"], 1)

    def test_event_matching_cannot_use_a_detection_twice(self):
        errors = event_errors([1.0, 1.04, 2.0], [1.01, 2.07], 0.05)
        self.assertEqual((errors["matched"], errors["missed"], errors["extra"]), (1, 2, 1))
        self.assertAlmostEqual(errors["matched_errors_ms"][0], 10)

    def test_ppg_corroborates_one_ecg_lead_without_overwriting_the_other(self):
        rows = [("ecg", "I", 80), ("ecg", "II", 92), ("ppg", "p", 92)]
        evidence = [
            {"modality": m, "channel": c, "rate_bpm": rate, "quality": "usable", "status": "ok",
             "window_s": [0, 10], "evidence_id": str(i)}
            for i, (m, c, rate) in enumerate(rows)
        ]
        self.assertIsNone(rate_summary(evidence[:2], 0, 10)["rate_bpm"])
        summary = rate_summary(list(reversed(evidence)), 0, 10)
        self.assertEqual(summary["rate_bpm"], 92)
        self.assertEqual(summary["observed"]["ecg"]["channels"]["I"]["rate_bpm"], 80)
        self.assertEqual(summary["observed"]["ecg"]["resolution"], "cross_modality_support")
        self.assertEqual(summary["evidence_ids"], ["1", "2"])

    def test_agreeing_ecg_leads_count_as_one_vote(self):
        evidence = [
            {"modality": m, "channel": c, "rate_bpm": rate, "quality": "usable", "status": "ok",
             "window_s": [0, 10], "evidence_id": c}
            for m, c, rate in (("ecg", "I", 80), ("ecg", "II", 80), ("ecg", "V3", 80),
                               ("ppg", "p", 92), ("pcg", "s", 92))
        ]
        self.assertEqual(rate_summary(evidence, 0, 10)["evidence_ids"], ["p", "s"])

    def test_alternate_lead_is_acquired_before_ppg(self):
        wave = Waveform(np.zeros((10000, 2)), 1000, ("I", "II"), ("mV", "mV"))
        record = Recording("case", {"ecg": wave, "ppg": wave}, {})

        def observe(modality, channel):
            good = modality == "ppg" or channel == "II"
            return {"modality": modality, "channel": channel, "status": "ok",
                    "quality": "usable" if good else "review", "rate_bpm": 90 if good else 80,
                    "window_s": [0, 10], "evidence_id": f"{modality}/{channel}"}

        result = ecg_first(record, 2, observe, strategy="alternate_lead_first")
        self.assertEqual([a["action"] for a in result["actions"]], ["ACQUIRE_ECG", "VERIFY_ECG:II", "STOP"])
        self.assertEqual(result["evidence_ids"], ["ecg/II"])
        self.assertEqual(ecg_first(record, 2, observe)["evidence_ids"], ["ppg/I"])

    def test_inter_rater_queue_captures_timing_and_unmatched_points(self):
        key = ("window_01", "pulse_foot")
        reviewers = {f"reviewer_{name}": {
            "completed": {key}, "reviews": {key: {"reviewer_id": name}}, "events": {key: times},
        } for name, times in (("a", [1.0, 2.0]), ("b", [1.03, 3.0]))}
        result = compute_inter_rater_metrics(reviewers)
        self.assertEqual(result["agreement"]["pulse_foot"]["metrics"]["f1"], 0.5)
        self.assertEqual([r["reason"] for r in result["adjudication_queue"][0]["events"]],
                         ["timing_difference", "only_a", "only_b"])
