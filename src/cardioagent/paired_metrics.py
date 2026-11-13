"""Record-paired metrics with subject-level bootstrap resampling."""

from collections import defaultdict
from hashlib import sha256
from math import ceil

import numpy as np

from .benchmark import rate_metrics


def summarize_rates(rows: list[dict], key: str) -> dict:
    result = rate_metrics(rows, key)
    result["rejection_rate"] = round(1 - result["coverage"], 6) if rows else None
    result["subjects"] = len({row["subject_id"] for row in rows})
    return result


def per_subject_rates(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["subject_id"]].append(row)
    return {subject: summarize_rates(group, key) for subject, group in groups.items()}


def confidence_rank(status: str) -> int:
    # Heuristic rank for selective evaluation, with ties kept together.
    return {"agreement": 3, "single_source": 2, "partial_agreement": 1}.get(status, 0)


def risk_coverage(rows: list[dict], key: str, rank_key: str, thresholds=None) -> list[dict]:
    """Fixed-threshold selective risk and percentile subject-cluster intervals."""
    points = []
    if not rows:
        return points
    subjects = sorted({row["subject_id"] for row in rows})
    index = {subject: i for i, subject in enumerate(subjects)}
    population = np.bincount([index[row["subject_id"]] for row in rows], minlength=len(subjects))
    draws = (np.random.default_rng(20261112).integers(0, len(subjects), size=(4000, len(subjects)))
             if len(subjects) >= 2 else None)
    if thresholds is None:
        thresholds = sorted({row[rank_key] for row in rows if row.get(key) is not None}, reverse=True)
    for threshold in thresholds:
        selected = [row for row in rows if row.get(key) is not None and row[rank_key] >= threshold]
        result = rate_metrics(selected, key)
        totals = np.zeros((len(subjects), 3))
        for row in selected:
            error = abs(row[key] - row["reference_bpm"])
            totals[index[row["subject_id"]]] += (1, error, error > 2)
        accepted, _, failures = totals.sum(axis=0)
        contributing = int(np.count_nonzero(totals[:, 0]))
        intervals = {}
        if draws is not None:
            n, error, failure = totals[draws].sum(axis=1).T
            denominator = population[draws].sum(axis=1)
            def ci(values):
                return np.percentile(values, [2.5, 97.5]).round(6).tolist() if len(values) else None
            keep = n > 0
            intervals = {"coverage_ci95": ci(n / denominator),
                         "selective_error_rate_ci95": ci(failure[keep] / n[keep]) if contributing >= 2 else None,
                         "mae_bpm_ci95": ci(error[keep] / n[keep]) if contributing >= 2 else None,
                         "unconditional_failure_ci95": ci((denominator - n + failure) / denominator)}
        points.append({"minimum_rank": float(threshold), "coverage": len(selected) / len(rows),
                       "mae_bpm": result["mae_bpm"], "n": len(selected), "subjects": contributing,
                       "total_subjects": len(subjects), "low_support": len(selected) < 5 or contributing < 3,
                       "selective_error_rate": float(failures / accepted) if accepted else None,
                       "unconditional_failure": float((len(rows) - accepted + failures) / len(rows)), **intervals})
    return points


def pareto_frontier(points: dict) -> list[str]:
    """Minimize waveform requests and unconditional refusal-or-error risk."""
    return [key for key, p in points.items() if p["risk"] is not None and not any(
        q["risk"] is not None and q["calls"] <= p["calls"] and q["risk"] <= p["risk"]
        and (q["calls"] < p["calls"] or q["risk"] < p["risk"]) for q in points.values()
    )]


def matched_operating_points(test, validation, key, rank_key, coverage_targets=(.50, .70, .85), risk_targets=(.05, .10, .20)):
    """Rank without test labels, then apply validation-chosen risk cutoffs unchanged."""
    def order(row):
        stable = sha256(f"{row['subject_id']}:{row.get('case_id', row['record_id'])}".encode()).digest()
        return -row[rank_key], stable

    if not test:
        return {"fixed_coverage": [], "validation_fixed_risk": []}
    available = sorted((r for r in test if r.get(key) is not None), key=order)
    ranks = {id(row): len(available) - index for index, row in enumerate(available)}
    ranked = [row | {"_position": ranks.get(id(row), 0)} for row in test]
    fixed_coverage = []
    for target in coverage_targets:
        n = ceil(target * len(test))
        if len(available) < n:
            fixed_coverage.append({"target_coverage": target, "attainable": False, "maximum_n": len(available)})
            continue
        point = risk_coverage(ranked, key, "_position", [len(available) - n + 1])[0]
        fixed_coverage.append({"target_coverage": target, "attainable": True, **point})

    val = sorted((r for r in validation if r.get(key) is not None), key=order)
    thresholds = []
    failures, subjects = 0, set()
    for i, row in enumerate(val, 1):
        failures += abs(row[key] - row["reference_bpm"]) > 2
        subjects.add(row["subject_id"])
        if i >= 5 and len(subjects) >= 3 and (i == len(val) or val[i][rank_key] != row[rank_key]):
            thresholds.append((row[rank_key], i, failures / i))
    fixed_risk = []
    for target in risk_targets:
        choice = next((p for p in reversed(thresholds) if p[2] <= target), None)
        if choice is None:
            fixed_risk.append({"target_risk": target, "attainable_on_validation": False})
            continue
        threshold, accepted, risk = choice
        point = risk_coverage(test, key, rank_key, [threshold])[0]
        fixed_risk.append({"target_risk": target, "attainable_on_validation": True, "validation_n": accepted,
                           "validation_selective_error_rate": risk, **point})
    return {"fixed_coverage": fixed_coverage, "validation_fixed_risk": fixed_risk}


def paired_metrics(rows: list[dict], anchor: str, candidate: str, seed: int = 20261111) -> dict:
    by_subject = defaultdict(lambda: np.zeros(6))
    details = []
    for row in rows:
        target = row["reference_bpm"]
        first = abs(row[anchor] - target) if row[anchor] is not None else None
        second = abs(row[candidate] - target) if row[candidate] is not None else None
        shared = first is not None and second is not None
        delta = second - first if shared else None
        coverage_delta = int(second is not None) - int(first is not None)
        success_delta = int(second is not None and second <= 5) - int(first is not None and first <= 5)
        success_2_delta = int(second is not None and second <= 2) - int(first is not None and first <= 2)
        by_subject[row["subject_id"]] += [delta or 0, int(shared), coverage_delta, success_delta, 1, success_2_delta]
        details.append({"record_id": row["record_id"], "case_id": row.get("case_id", row["record_id"]), "subject_id": row["subject_id"],
                        "anchor_error_bpm": first, "candidate_error_bpm": second, "delta_bpm": delta})

    def ratio(numerator: float, denominator: float) -> float | None:
        return round(float(numerator / denominator), 6) if denominator else None

    n_subjects = len(by_subject)
    totals = np.stack(list(by_subject.values())) if by_subject else np.empty((0, 6))
    sample_sums = None
    if n_subjects >= 2:
        rng = np.random.default_rng(seed)
        sampled = rng.integers(0, n_subjects, size=(4000, n_subjects))
        sample_sums = totals[sampled].sum(axis=1)

    def confidence_interval(numerator: int, denominator: int) -> list[float] | None:
        if sample_sums is None:
            return None
        valid = sample_sums[:, denominator] > 0
        if not valid.any():
            return None
        values = sample_sums[valid, numerator] / sample_sums[valid, denominator]
        return np.round(np.percentile(values, [2.5, 97.5]), 6).tolist()

    delta, shared, coverage, success, total, success_2 = totals.sum(axis=0)
    return {
        "subjects": n_subjects, "shared_scored": int(shared),
        "candidate_minus_anchor_mae_bpm": ratio(delta, shared),
        "mae_delta_ci95": confidence_interval(0, 1),
        "coverage_delta": ratio(coverage, total), "coverage_delta_ci95": confidence_interval(2, 4),
        "within_5_delta": ratio(success, total), "within_5_delta_ci95": confidence_interval(3, 4),
        "risk_reduction_at_2_bpm": ratio(success_2, total),
        "risk_reduction_at_2_bpm_ci95": confidence_interval(5, 4),
        "only_anchor_scored": sum(r["anchor_error_bpm"] is not None and r["candidate_error_bpm"] is None for r in details),
        "only_candidate_scored": sum(r["candidate_error_bpm"] is not None and r["anchor_error_bpm"] is None for r in details),
        "records": details,
    }
