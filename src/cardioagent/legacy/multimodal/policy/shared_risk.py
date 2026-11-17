"""One policy-independent risk scorer for comparisons at matched coverage."""

from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from cardioagent.legacy.multimodal.policy.stopping import trial_weights
from cardioagent.legacy.multimodal.policy.targets import subsets


def features(task, rate, evidence):
    detected = [row for row in evidence if row.get("rate_bpm") is not None]
    accepted = [row for row in detected if row["quality"] == "usable"]
    gaps = [abs(a["rate_bpm"] - b["rate_bpm"]) for a, b in combinations(detected, 2)]
    values = [1., task == "peripheral_pulse_rate", (rate - 80) / 80, len(detected) / 5,
              len(accepted) / 5, min(max(gaps, default=0.) / 30, 3.)]
    for modality in ("ecg", "ppg", "pcg"):
        items = [row for row in detected if row["modality"] == modality]
        values.extend([bool(items), any(row["quality"] == "usable" for row in items),
                       max((row["signal_quality"].get("physiology_band_power_fraction") or 0. for row in items), default=0.)])
    return values


def fit(cases, budget):
    trials, predictors, targets = [], [], []
    for row, profile in cases:
        if row["split"] != "train":
            raise ValueError("Shared risk requires training subjects only")
        initial = (profile.initial,) if profile.initial else ()
        for extra in subsets(tuple(a for a in profile.options if a not in initial)):
            actions = initial + extra
            if len(actions) > budget:
                continue
            evidence = {a: profile.measurements[a] for a in actions}
            rate = profile.state(evidence, {}, budget - len(actions)).summary()["rate_bpm"]
            if rate is not None:
                trials.append(row)
                predictors.append(features(profile.task, rate, evidence.values()))
                targets.append(float(abs(rate - row["reference_bpm"]) > 2))
    if not trials:
        raise ValueError("Shared risk has no eligible training estimates")
    x, y = np.asarray(predictors), np.asarray(targets)
    w = trial_weights(trials, "subject_record")
    w /= w.sum()

    def loss(beta):
        z = x @ beta
        penalty = beta.copy()
        penalty[0] *= .1
        return (float(w @ (np.logaddexp(0., z) - y * z) + .005 * (penalty @ beta)),
                x.T @ (w * (expit(z) - y)) + .01 * penalty)

    result = minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise ValueError(f"Shared risk failed to converge: {result.message}")
    return {"coefficients": result.x.tolist(), "training_estimates": len(trials),
            "training_subjects": len({row["subject_id"] for row in trials})}


def score(scorer, task, result):
    rate = result["rate_bpm"]
    if rate is None:
        return 0.
    return float(expit(-np.dot(scorer["coefficients"], features(task, rate, result["evidence"]))))
