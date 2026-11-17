import unittest

import numpy as np

from cardioagent.paired_metrics import matched_operating_points
from cardioagent.policy.conditional_value import Variant
from cardioagent.policy.features import numeric
from cardioagent.policy.objectives import Objectives
from cardioagent.policy.selection import run
from cardioagent.policy.stopping import trial_weights
from cardioagent.policy.targets import counterfactual
from cardioagent.policy.value import ABSTAIN, OUTPUT_SLOTS, Profile, output_index


def observation(modality, channel, rate):
    return {"modality": modality, "channel": channel, "rate_bpm": rate, "evidence_id": f"{modality}/{channel}",
            "status": "ok", "quality": "usable", "window_s": [0, 10], "signal_quality": {},
            "event_quality": {}, "duration_ms": 2.}


class V24Tests(unittest.TestCase):
    def test_counterfactual_distinguishes_two_acceptable_errors(self):
        lead, pulse = "VERIFY_ECG:II", "ACQUIRE_PPG"
        options = {lead: ("ecg", "II", "alternate_ecg_1"), pulse: ("ppg", "IR", "ppg")}
        profile = Profile("heart_rate", 10, options, None,
                          {lead: observation("ecg", "II", 80.1), pulse: observation("ppg", "IR", 81.9)},
                          {a: {"valid": 1., "score": .8, "duration_ms": .5} for a in options})
        choose, examples = counterfactual(profile, 1, 80., Objectives(), .1)
        self.assertEqual(choose((), [lead, pulse, ABSTAIN]), lead)
        _, targets, _, _ = next(examples())
        one, two = (targets[output_index(a, options)] for a in (lead, pulse))
        self.assertLess(one[0], two[0])
        self.assertEqual((one[1], two[1]), (0., 0.))

    def test_qc_is_charged_and_unseen_measurements_are_not_read(self):
        pulse = "ACQUIRE_PPG"
        options = {pulse: ("ppg", "IR", "ppg")}
        preview = {"valid": 1., "score": .9, "duration_ms": .5}
        profile = Profile("heart_rate", 10, options, None, {pulse: observation("ppg", "IR", 80.)}, {pulse: preview})
        costs = {"waveform": {"ppg": 2.}, "qc": {"ppg": .5}, "decision": .1}

        class Value:
            variant = Variant()

            def predict(self, batch, task):
                result = np.zeros((len(batch), len(OUTPUT_SLOTS), 5))
                i = output_index(pulse, options)
                for features, rows in zip(batch, result):
                    rows[i, 0] = .02 if features[6 + 3 * 12 + 4] else .5
                    rows[i, 2] = 1.
                return result

        priors = {"heart_rate/ppg": [preview]}
        used = run(profile, 1, costs, priors, model=Value())
        refused = run(profile, 1, costs | {"qc": {"ppg": 10000.}}, priors, model=Value())
        self.assertEqual([step["action"] for step in used["actions"]], [f"QC:{pulse}", pulse, "STOP"])
        self.assertEqual(used["cost"]["qc_ms"], .5)
        self.assertEqual(refused["decision"], ABSTAIN)
        self.assertEqual(refused["cost"]["waveform_requests"], 0)

    def test_missing_interval_indicator_differs_from_observed_zero(self):
        option = {"ACQUIRE_ECG": ("ecg", "I", "primary_ecg")}
        item = observation("ecg", "I", 80.)
        profile = Profile("heart_rate", 10, option, "ACQUIRE_ECG", {}, {})
        missing = profile.state({"ACQUIRE_ECG": item}, {}, 1)
        zero = profile.state({"ACQUIRE_ECG": item | {"event_quality": {"interval_mad_ratio": 0.}}}, {}, 1)
        self.assertEqual(np.count_nonzero(numeric(missing, missing.summary()) != numeric(zero, zero.summary())), 1)

    def test_record_weighting_and_fixed_coverage(self):
        cases = [{"subject_id": str(i % 3), "record_id": str(i), "case_id": str(i), "reference_bpm": 80.,
                  "rank": float(i), "estimate": 80.} for i in range(10)]
        weights = trial_weights([cases[0], cases[0], cases[1]], "subject_record")
        self.assertEqual(list(weights), [.5, .5, 1.])
        points = matched_operating_points(cases, cases[:6], "estimate", "rank", (.7,), (.1,))
        self.assertEqual(points["fixed_coverage"][0]["n"], 7)
        self.assertEqual(points["validation_fixed_risk"][0]["validation_n"], 6)
