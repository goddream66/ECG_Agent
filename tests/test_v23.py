import unittest

from cardioagent.paired_metrics import risk_coverage
from cardioagent.policy.state import SLOTS, EvidenceState
from cardioagent.policy.stopping import RiskCalibrator, fit_calibrator
from cardioagent.policy.value import ABSTAIN, OUTPUT_SLOTS, Profile, eig_scores, rule_action, run_policy


class V23PolicyTests(unittest.TestCase):
    def test_channel_qc_preserves_a_clean_alternate_lead(self):
        options = {"ACQUIRE_ECG": ("ecg", "I", "primary_ecg"),
                   "VERIFY_ECG:II": ("ecg", "II", "alternate_ecg_1"), "ACQUIRE_PPG": ("ppg", "IR", "ppg")}
        choices = ["VERIFY_ECG:II", "ACQUIRE_PPG"]
        state = EvidenceState("heart_rate", 10, options,
                              {"ACQUIRE_ECG": {"modality": "ecg", "rate_bpm": None, "quality": "review"}},
                              {choices[0]: {"valid": 1, "score": .90}, choices[1]: {"valid": 1, "score": .25}}, 2)
        costs = dict.fromkeys(SLOTS, 1.)
        fixed = eig_scores(state, choices, costs, channel_qc=False, cost_aware=False)
        channel = eig_scores(state, choices, costs, channel_qc=True, cost_aware=False)
        self.assertGreater(fixed[choices[1]], fixed[choices[0]])
        self.assertGreater(channel[choices[0]], channel[choices[1]])
        self.assertEqual(rule_action("channel_quality", state, choices + [ABSTAIN], costs), choices[0])
        costs["alternate_ecg_1"] = 1000.
        priced = eig_scores(state, choices, costs, channel_qc=True, cost_aware=True)
        self.assertGreater(priced[choices[1]], priced[choices[0]])

    def test_calibrated_stop_can_decline_a_legal_estimate(self):
        action = "ACQUIRE_ECG"
        profile = Profile("heart_rate", 10, {action: ("ecg", "I", "primary_ecg")}, action,
                          {action: {"evidence_id": "ecg/I", "modality": "ecg", "channel": "I", "window_s": [0, 10],
                                    "status": "ok", "rate_bpm": 80, "quality": "usable", "duration_ms": 10.,
                                    "signal_quality": {}, "event_quality": {}}},
                          {action: {"valid": 1, "score": .8, "duration_ms": 2.}})
        costs = dict.fromkeys(SLOTS, 1.)

        def model(_):
            return [.3] * len(OUTPUT_SLOTS)

        raw = run_policy(profile, 1, "learned_voi", costs, model=model)
        gated = run_policy(profile, 1, "learned_voi_calibrated", costs, model=model,
                           calibrator=RiskCalibrator(0., 1.), max_risk=.20)
        self.assertEqual(raw["rate_bpm"], 80)
        self.assertIsNone(gated["rate_bpm"])
        self.assertEqual(gated["decision"], ABSTAIN)
        self.assertEqual(gated["cost"]["analysis_ms"], 12.)

        next_action = "ACQUIRE_PPG"
        profile.options[next_action] = ("ppg", "IR", "ppg")
        profile.measurements[next_action] = profile.measurements[action] | {
            "evidence_id": "ppg/IR", "modality": "ppg", "channel": "IR"}
        profile.previews[next_action] = profile.previews[action]

        def low_risk_model(_):
            scores = [.10] * len(OUTPUT_SLOTS)
            scores[OUTPUT_SLOTS.index("ppg")] = .13
            return scores

        allowed = run_policy(profile, 2, "learned_voi_calibrated", costs, model=low_risk_model,
                             calibrator=RiskCalibrator(.65, 1.), max_risk=.20)
        self.assertEqual(allowed["decision"], "STOP")
        self.assertEqual(allowed["cost"]["waveform_requests"], 1)

    def test_thresholds_keep_ties_and_count_independent_subjects(self):
        rows = [{"subject_id": "a", "reference_bpm": 80, "rate": 81, "rank": .6},
                {"subject_id": "a", "reference_bpm": 80, "rate": 85, "rank": .6},
                {"subject_id": "b", "reference_bpm": 80, "rate": None, "rank": .4}]
        empty, accepted = risk_coverage(rows, "rate", "rank", (.7, .6))
        self.assertIsNone(empty["selective_error_rate"])
        self.assertEqual((accepted["n"], accepted["subjects"], accepted["total_subjects"]), (2, 1, 2))
        self.assertTrue(accepted["low_support"])
        self.assertIsNone(accepted["selective_error_rate_ci95"])
        self.assertAlmostEqual(accepted["unconditional_failure"], 2 / 3)

    def test_calibration_requires_outcomes_and_two_subjects(self):
        trials = [{"subject_id": "a", "raw_risk": .1, "failed": 0},
                  {"subject_id": "b", "raw_risk": .8, "failed": 1}]
        with self.assertRaises(ValueError):
            fit_calibrator(trials[:1])
        fitted = fit_calibrator(trials)
        self.assertLess(fitted.predict(.1), fitted.predict(.8))
