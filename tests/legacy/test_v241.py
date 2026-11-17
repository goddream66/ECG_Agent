import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import wfdb

from cardioagent.policy.mimic import _record, local_segments
from cardioagent.policy.objectives import DELTA, Objectives
from cardioagent.policy.qc_prior import distribution, fit_conditionals
from cardioagent.policy.targets import counterfactual
from cardioagent.policy.value import ABSTAIN, STOP, Profile, output_index


class V241Tests(unittest.TestCase):
    def test_advantage_matches_full_costed_continuation(self):
        primary, pulse = "ACQUIRE_ECG", "ACQUIRE_PPG"
        options = {primary: ("ecg", "I", "primary_ecg"), pulse: ("ppg", "IR", "ppg")}
        measurements = {a: {"modality": modality, "channel": channel, "rate_bpm": rate, "evidence_id": a,
                            "window_s": [0, 10], "quality": "usable", "status": "ok", "duration_ms": 10.}
                        for (a, (modality, channel, _)), rate in zip(options.items(), (80., 81.))}
        preview = {"valid": 1., "score": .8, "duration_ms": 1.}
        profile = Profile("heart_rate", 10, options, primary, measurements, {a: preview for a in options})
        objective = Objectives()
        choose, examples = counterfactual(profile, 2, 80., objective, .2, advantage=True)
        _, targets, mask, _ = next(examples())
        legal = [pulse, STOP, ABSTAIN]
        self.assertEqual(choose((primary,), legal), STOP)
        indices = [output_index(a, options) for a in legal]
        losses = [objective.loss(targets[i]) for i in indices]
        for i, cost in zip(indices, losses):
            self.assertEqual(mask[i, DELTA], 1.)
            self.assertAlmostEqual(targets[i, DELTA], min(losses) - cost)

    def test_qc_distribution_depends_on_observed_anchor(self):
        primary, pulse = "ACQUIRE_ECG", "ACQUIRE_PPG"
        options = {primary: ("ecg", "I", "primary_ecg"), pulse: ("ppg", "IR", "ppg")}
        cases = []
        for quality, score in (("usable", .85), ("review", .20)):
            observation = {"quality": quality, "rate_bpm": 80., "modality": "ecg"}
            profile = Profile("heart_rate", 10, options, primary, {primary: observation}, {pulse: {"score": score}})
            cases.append(({"split": "train", "subject_id": quality}, profile))
        priors = {"conditions": fit_conditionals(cases, 2, min_subjects=1), "heart_rate/ppg": [{"score": .5}]}
        for (_, profile), expected in zip(cases, (.85, .20)):
            values, level = distribution(priors, profile, profile.measurements, "ppg")
            self.assertEqual(level, "joint")
            self.assertEqual({v["score"] for v in values}, {expected})
        self.assertEqual(distribution(priors, cases[0][1], {}, "ppg")[1], "marginal")

    def test_mimic_loader_uses_local_paired_waveforms_and_release_fold(self):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            target = root / "p01" / "p010001"
            target.mkdir(parents=True)
            time = np.arange(3750) / 125
            signal = np.sin(2 * np.pi * time)
            stem = "1234567_0001_0_0"
            wfdb.wrsamp(stem, fs=125, units=["mV", "NU"], sig_name=["II", "PLETH"],
                       p_signal=np.column_stack([signal, signal]), fmt=["16", "16"], write_dir=str(target))
            with (root / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["patient", "signal_file_name", "median_30s_hr", "strat_fold"])
                writer.writerow(["p010001", f"{stem}.hea", "60", "9"])
            [(fold, _, path, hr)] = list(local_segments(root, 8, 1))
            self.assertEqual((fold, hr), ("test", 60.))
            self.assertEqual(set(_record(path).waves), {"ecg", "ppg"})
