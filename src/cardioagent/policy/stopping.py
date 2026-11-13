"""Fit a monotone probability calibration on validation subjects."""

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


@dataclass(frozen=True)
class RiskCalibrator:
    intercept: float
    slope: float

    def predict(self, risk: float) -> float:
        return float(expit(self.intercept + self.slope * logit(np.clip(risk, 1e-5, 1 - 1e-5))))

    def describe(self) -> dict:
        return asdict(self)


def trial_weights(trials, weighting):
    subjects = Counter(r["subject_id"] for r in trials)
    if weighting == "subject":
        return np.array([1 / subjects[r["subject_id"]] for r in trials])
    records = Counter((r["subject_id"], r["case_id"]) for r in trials)
    per_subject = Counter(subject for subject, _ in records)
    if weighting == "record":
        return np.array([1 / records[r["subject_id"], r["case_id"]] for r in trials])
    if weighting == "subject_record":
        return np.array([1 / (records[r["subject_id"], r["case_id"]] * per_subject[r["subject_id"]]) for r in trials])
    raise ValueError(f"Unknown calibration weighting: {weighting}")


def fit_calibrator(trials: list[dict], weighting="subject") -> RiskCalibrator:
    subjects = Counter(r["subject_id"] for r in trials)
    if len(subjects) < 2 or len({r["failed"] for r in trials}) < 2:
        raise ValueError("Calibration needs at least two validation subjects and both success and failure outcomes")
    x = logit(np.clip([r["raw_risk"] for r in trials], 1e-5, 1 - 1e-5))
    y = np.array([r["failed"] for r in trials], dtype=float)
    w = trial_weights(trials, weighting)
    w /= w.sum()

    def objective(params):
        b, a = params
        z = b + a * x
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)) + 0.002 * (b * b + (a - 1) ** 2))

    result = minimize(objective, [0., 1.], bounds=[(-12., 12.), (0., 12.)], method="L-BFGS-B")
    if not result.success:
        raise ValueError(f"Validation calibration did not converge: {result.message}")
    return RiskCalibrator(*map(float, result.x))


def calibration_metrics(trials: list[dict], calibrator: RiskCalibrator | None, weighting="subject") -> dict:
    if not trials:
        return {"trials": 0, "subjects": 0}
    subjects = Counter(r["subject_id"] for r in trials)
    weights = trial_weights(trials, weighting)
    return {"trials": len(trials), "subjects": len(subjects),
            "brier_raw": float(np.average([(r["raw_risk"] - r["failed"]) ** 2 for r in trials], weights=weights)),
            "brier_calibrated": float(np.average([((calibrator.predict(r["raw_risk"]) if calibrator else r["risk"]) - r["failed"]) ** 2
                                                 for r in trials], weights=weights))}


def leave_one_subject_out(trials, weighting="subject_record"):
    folds, held_out = [], []
    for subject in sorted({r["subject_id"] for r in trials}):
        development = [r for r in trials if r["subject_id"] != subject]
        validation = [r for r in trials if r["subject_id"] == subject]
        try:
            calibrator = fit_calibrator(development, weighting)
        except ValueError as exc:
            folds.append({"subject_id": subject, "error": str(exc)})
            continue
        held_out.extend(row | {"risk": calibrator.predict(row["raw_risk"])} for row in validation)
        folds.append({"subject_id": subject, **calibration_metrics(validation, calibrator, weighting)})
    return {"folds": folds, "cross_fitted": calibration_metrics(held_out, None, weighting)}
