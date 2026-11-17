"""V2.3 matched experiments for channel choice, VOI, and selective stopping."""

from collections import Counter
from pathlib import Path
from statistics import mean

from cardioagent.common.io import digest, save_json
from cardioagent.legacy.multimodal.dataset import load_manifest
from cardioagent.legacy.multimodal.evaluation_reference import evaluation_cases, load_case_record
from cardioagent.legacy.multimodal.paired_metrics import paired_metrics, pareto_frontier, per_subject_rates, risk_coverage, summarize_rates
from cardioagent.legacy.multimodal.provenance import save_provenance
from cardioagent.legacy.multimodal.rate_audit import perturb_recording
from cardioagent.legacy.multimodal.workflow import rate_summary
from cardioagent.legacy.multimodal.policy.stopping import RiskCalibrator, calibration_metrics, fit_calibrator
from cardioagent.legacy.multimodal.policy.value import BASELINES, ERROR_LIMIT, Profile, capture_profile, channel_observers, estimate_costs, run_policy
from cardioagent.legacy.multimodal.policy.state import TASKS, actions_for


SCENARIOS = ("original", "lead_i_low_snr", "low_snr", "missing_ecg", "ecg_ppg_low_snr", "ecg_pcg_low_snr", "joint_low_snr")
METHODS = (*BASELINES, "learned_voi", "learned_voi_calibrated", "oracle")
RISK_LIMIT = 0.20
CURVE_THRESHOLDS = (0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.25, 0.)


def cohort_profiles(dataset, rows):
    profiles = {scenario: [] for scenario in SCENARIOS}
    for i, row in enumerate(rows, 1):
        record = load_case_record(dataset, row)
        original = capture_profile(record)
        for scenario in SCENARIOS:
            profile = original if scenario == "original" else capture_profile(
                perturb_recording(record, scenario), original=record, original_profile=original)
            profiles[scenario].append((row, profile))
        if i == len(rows) or i % 30 == 0:
            print(f"{row['split']}: measured {i}/{len(rows)} records", flush=True)
    return profiles


def stop_trials(row, result):
    return [trial | {"subject_id": row["subject_id"],
                     "failed": int(abs(trial["rate_bpm"] - row["reference_bpm"]) > ERROR_LIMIT)}
            for trial in result["stop_trials"]]


def test_cases(profiles, budget, costs, net, calibration):
    from cardioagent.legacy.multimodal.policy.trainer import counterfactual

    scored, calibration_trials = [], []
    for row, profile in profiles:
        case = row.copy()
        for method in METHODS:
            pick = counterfactual(profile, budget, row["reference_bpm"], costs)[0] if method == "oracle" else None
            result = run_policy(profile, budget, method, costs, model=net.predict, oracle=pick,
                                calibrator=calibration if method.endswith("_calibrated") else None,
                                max_risk=RISK_LIMIT if method.endswith("_calibrated") else None)
            case[method] = result["rate_bpm"]
            case[f"{method}_rank"] = result["quality_rank"]
            case[f"{method}_trace"] = {key: result[key] for key in ("actions", "decision", "cost", "stop_trials")}
            if method == "learned_voi":
                calibration_trials.extend(stop_trials(row, result))
                case["calibrated_rank_for_returned_estimate"] = (
                    1 - calibration.predict(result["stop_trials"][-1]["raw_risk"])
                    if result["rate_bpm"] is not None and result["stop_trials"] else 0.
                )
        primary = profile.measurements.get(profile.initial)
        case["primary_quality"] = primary["quality"] if primary else "missing"
        for modality in ("ppg", "pcg"):
            key = f"ACQUIRE_{modality.upper()}"
            case[f"ecg+{modality}"] = rate_summary(
                [m for m in (primary, profile.measurements.get(key)) if m], 0, profile.duration)["rate_bpm"]
        scored.append(case)
    return scored, calibration_trials


def summarize(cases, trials, calibration):
    rows = {}
    for method in METHODS:
        info = summarize_rates(cases, method)
        cost = [row[f"{method}_trace"]["cost"] for row in cases]
        paths = Counter(" → ".join(step["action"] for step in row[f"{method}_trace"]["actions"]) for row in cases)
        rank = f"{method}_rank"
        rows[method] = info | {
            "unconditional_failure": 1 - info["within_2_bpm_of_all"],
            "selective_error_rate": sum(abs(r[method] - r["reference_bpm"]) > ERROR_LIMIT
                                       for r in cases if r[method] is not None) / info["scored"] if info["scored"] else None,
            "mean_replayed_analysis_ms": mean(c["analysis_ms"] for c in cost),
            "mean_waveform_requests": mean(c["waveform_requests"] for c in cost),
            "mean_qc_requests": mean(c["qc_requests"] for c in cost),
            "mean_policy_loop_ms": mean(1000 * c["wall_time_s"] for c in cost),
            "action_paths": dict(paths),
            "by_subject": per_subject_rates(cases, method),
            "heuristic_selective_risk_coverage": risk_coverage(cases, method, rank, CURVE_THRESHOLDS),
        }
    points = {name: {"calls": metric["mean_replayed_analysis_ms"], "risk": metric["unconditional_failure"]}
              for name, metric in rows.items() if name != "oracle"}
    gated_trials = [trial for case in cases for trial in stop_trials(case, case["learned_voi_calibrated_trace"])]
    return {
        "metrics": rows, "replayed_time_risk_pareto": pareto_frontier(points),
        "calibration_on_test_stop_candidates": calibration_metrics(trials, calibration),
        "calibration_on_closed_loop_test_stop_candidates": calibration_metrics(gated_trials, calibration),
        "paired_rule_and_voi_comparisons": {
            f"{anchor}_vs_{candidate}": {key: value for key, value in paired_metrics(cases, anchor, candidate).items()
                                         if key != "records"}
            for anchor, candidate in (
                ("fixed_gaussian_eig", "channel_eig"), ("channel_eig", "cost_aware_channel_eig"),
                ("alternate_lead_first", "learned_voi"), ("cost_aware_channel_eig", "learned_voi"),
                ("learned_voi", "learned_voi_calibrated"), ("learned_voi", "oracle"),
            )
        },
        "calibrated_risk_coverage_on_ungated_voi_outputs": risk_coverage(
            cases, "learned_voi", "calibrated_rank_for_returned_estimate", CURVE_THRESHOLDS),
        "auxiliary_value_by_primary_quality": {
            stratum: {method: paired_metrics([row for row in cases if row["primary_quality"] == stratum],
                                             "ecg_only", method)
                      for method in ("ecg+ppg", "ecg+pcg")}
            for stratum in ("usable", "review", "missing")
        },
        "records": cases,
    }


def run_audit(dataset: Path, output: Path, *, pilot: bool, development=None, test=None,
              budget: int = 6, epochs: int = 16) -> dict:
    import torch
    from cardioagent.legacy.multimodal.policy.trainer import fit_value_network

    if pilot and (development or test):
        raise ValueError("Choose the published-reference pilot or two adjudicated packets")
    if not pilot and (development is None or test is None):
        raise ValueError("Provide adjudicated development and test packets, or explicitly opt into the published-reference pilot")
    manifest = load_manifest(dataset)
    groups, references = {}, {}
    for split in ("train", "validation", "test"):
        packet = test if split == "test" else development
        groups[split], references[split] = evaluation_cases(dataset, manifest, split, packet)
    ids = {split: sorted({r["subject_id"] for r in cases}) for split, cases in groups.items()}
    if sum(map(len, ids.values())) != len(set().union(*map(set, ids.values()))):
        raise ValueError("Training, calibration and test subjects must be disjoint")
    output.mkdir(parents=True, exist_ok=False)
    config = {"experiment": "V2.3 task-conditioned active tool selection", "task": "heart_rate",
              "published_reference_pilot": pilot, "budget": budget, "epochs": epochs,
              "scenarios": SCENARIOS, "subject_ids": ids, "references": references,
              "dataset_manifest_sha256": digest(dataset / "manifest.json")}
    provenance = save_provenance(output, config)
    train = [case for cases in cohort_profiles(dataset, groups["train"]).values() for case in cases]
    costs = estimate_costs([profile for _, profile in train])
    net, training = fit_value_network(train, budget, costs, epochs=epochs)
    del train
    validation = cohort_profiles(dataset, groups["validation"])
    trials = [trial for cases in validation.values() for row, profile in cases
              for trial in stop_trials(row, run_policy(profile, budget, "learned_voi", costs, model=net.predict))]
    calibration = fit_calibrator(trials)
    torch.save({"state_dict": net.state_dict(), "width": net.width, "costs_ms": costs,
                "calibrator": calibration.describe(), "max_risk": RISK_LIMIT, "task": "heart_rate",
                "budget": budget, "published_reference_pilot": pilot}, output / "value_network.pt")
    report = config | {
        "provenance": provenance, "training": training, "estimated_action_costs_ms_train_only": costs,
        "cost_semantics": "waveform and channel-QC times measured offline; online model/decision loop reported separately",
        "validation_calibration": calibration.describe() | calibration_metrics(trials, calibration),
        "reliable_stopping_max_predicted_error_risk": RISK_LIMIT,
        "stopping_calibration_semantics": "calibration masks STOP; eligible action order uses the original value scores",
        "oracle_semantics": "uses the reference and all tool outcomes; privileged upper bound for the penalized objective",
        "risk_semantics": "selective error = |error| > 2 bpm among accepted estimates; unconditional failure includes abstentions",
        "results": {},
    }
    for scenario, profiles in cohort_profiles(dataset, groups["test"]).items():
        cases, test_trials = test_cases(profiles, budget, costs, net, calibration)
        report["results"][scenario] = summarize(cases, test_trials, calibration)
    save_json(output / "v23_audit.json", report)
    return report


def analyze(record, task: str, budget: int, method: str, checkpoint=None) -> dict:
    if task not in TASKS or method not in (*BASELINES, "learned_voi_calibrated"):
        raise ValueError("Unsupported task or policy")
    options, initial = actions_for(record, task)
    profile = Profile(task, record.shared_duration, options, initial, {}, {})
    kwargs = {}
    model_reference = None
    costs = {slot: 1. for _, _, slot in options.values()}
    if method in ("learned_voi_calibrated", "cost_aware_channel_eig"):
        if checkpoint is None:
            raise ValueError("This policy requires --checkpoint for training-derived action costs")
        import torch
        from cardioagent.legacy.multimodal.policy.network import ValueNetwork
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data["task"] != task:
            raise ValueError("Checkpoint task differs from requested task")
        model_reference = {"checkpoint": str(checkpoint.resolve()),
                           "published_reference_pilot": data["published_reference_pilot"]}
        costs = data["costs_ms"]
        if method == "learned_voi_calibrated":
            if data["budget"] != budget:
                raise ValueError("Use the action budget recorded in this checkpoint")
            net = ValueNetwork(data["width"])
            net.load_state_dict(data["state_dict"])
            kwargs = {"model": net.eval().predict, "calibrator": RiskCalibrator(**data["calibrator"]),
                      "max_risk": data["max_risk"]}
    return run_policy(profile, budget, method, costs, live=channel_observers(record), **kwargs) | {
        "record": record.describe(), "budget": budget, "checkpoint_provenance": model_reference}
