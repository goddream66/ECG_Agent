"""V2.4 train, calibration, held-out comparisons, and reproducible export."""

from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, median
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from ..dataset import digest, save_json
from ..paired_metrics import matched_operating_points, paired_metrics, pareto_frontier, per_subject_rates, risk_coverage, summarize_rates
from ..provenance import save_provenance
from .conditional_value import ConditionalNetwork, VARIANTS, Variant, decision_estimate, examples, preferences, train
from .features import numeric
from .network import ValueNetwork
from .objectives import DELTA, Objectives, cost_table
from .replay import read_replay
from .selection import fit_qc_priors, run
from . import shared_risk
from .qc_prior import fit_conditionals, validate_priors
from .state import TASKS, actions_for
from .stopping import RiskCalibrator, calibration_metrics, fit_calibrator, leave_one_subject_out
from .targets import counterfactual
from .value import BASELINES, Profile, channel_observers, output_index, run_policy


STOP_RISK = .20
CALIBRATED = "conditional_voi_calibrated"
TARGET_CALIBRATED = "conditional_voi_target_calibrated"
EXPORTS = ("v24_audit.json", "models.pt", "experiment_config.json", "source_manifest.json", "dataset_provenance.json")


def _fit(cases, budget, epochs, *, corrected=False):
    torch.set_num_threads(3)
    profiles = [profile for _, profile in cases]
    estimate = decision_estimate(profiles[0], budget)
    costs = cost_table(profiles, estimate)
    priors = fit_qc_priors(cases)
    if corrected:
        priors["conditions"] = fit_conditionals(cases, budget)
    torch.manual_seed(20261124)
    state = profiles[0].state({}, {}, budget)
    initial_variant = replace(VARIANTS["conditional_voi"], advantage=corrected)
    initial_model = ConditionalNetwork(len(numeric(state, state.summary())), initial_variant).eval()
    replayed = [run(p, budget, costs, priors, model=initial_model) for p in profiles[::max(1, len(profiles) // 16)]]
    costs["decision"] = median(row["cost"]["decision_ms"] / len(row["actions"]) for row in replayed)
    main = examples(cases, budget, costs["decision"], Objectives(), advantage=corrected)
    print(f"Generated {len(main[0])} counterfactual training states", flush=True)
    models, history = {}, {}
    for name, variant in VARIANTS.items():
        if name == "full_qc":
            continue
        variant = replace(variant, advantage=corrected)
        data = examples(cases, budget, costs["decision"], preferences(variant), advantage=corrected) if name == "no_cost" else main
        tasks = tuple(TASKS) if name == "single_task_voi" else (None,)
        for task in tasks:
            key = f"{name}/{task}" if task else name
            models[key], history[key] = train(data, variant, epochs, only_task=task)
            print(f"Trained {key}", flush=True)
    return models, history, costs, priors


def _old_checkpoint(path):
    if path is None:
        return None
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["task"] != "heart_rate":
        raise ValueError("The V2.3 comparison checkpoint must be a heart-rate checkpoint")
    net = ValueNetwork(data["width"])
    net.load_state_dict(data["state_dict"])
    return net.eval(), data["costs_ms"], data["budget"]


def _trials(row, result):
    return [trial | {"subject_id": row["subject_id"], "case_id": row["case_id"], "scenario": row["scenario"],
                     "failed": int(abs(trial["rate_bpm"] - row["reference_bpm"]) > 2)} for trial in result["stop_trials"]]


def _calibrate(cases, model, budget, costs, priors):
    results, metadata = {}, {}
    for task in TASKS:
        trials = [trial for row, profile in cases if profile.task == task
                  for trial in _trials(row, run(profile, budget, costs, priors, model=model))]
        methods = {}
        for weighting in ("subject", "record", "subject_record"):
            try:
                fit = fit_calibrator(trials, weighting)
            except ValueError as exc:
                methods[weighting] = {"error": str(exc)}
                continue
            methods[weighting] = fit.describe() | calibration_metrics(trials, fit, weighting)
            if weighting == "subject_record":
                results[task] = fit
        metadata[task] = {"fits": methods, "leave_one_subject_out": leave_one_subject_out(trials),
                          "validation_scenarios": {s: calibration_metrics([r for r in trials if r["scenario"] == s], fit, "subject_record")
                                                   for s in sorted({r["scenario"] for r in trials}) if (fit := results.get(task))}}
    return results, metadata


def _group(cases):
    groups = defaultdict(list)
    for row, profile in cases:
        groups[(row["dataset"], profile.task, row["scenario"])].append((row, profile))
    return groups


def _compact(report, *, legacy):
    cost = report["cost"]
    if legacy:
        decision = cost["wall_time_s"] * 1000
        cost = cost | {"decision_ms": decision, "analysis_ms": cost["analysis_ms"] + decision}
    return {k: report[k] for k in ("actions", "decision", "stop_trials")} | {"cost": cost}


def _delta_check(profile, budget, reference, models, decision_ms):
    if len(profile.options) <= 1:
        return []
    _, generate = counterfactual(profile, budget, reference, Objectives(), decision_ms,
                                 advantage=models["conditional_voi"].variant.advantage)
    # The first block enumerates preview subsets at the common initial evidence state.
    candidates = len(profile.options) - int(profile.initial is not None) if budget > int(profile.initial is not None) else 0
    if not candidates:
        return []
    from itertools import islice
    x, y, mask, _ = next(islice(generate(), (1 << candidates) - 1, None))
    names = ("multitask_voi", "conditional_voi")
    estimates = {name: models[name].predict([x], profile.task)[0] for name in names}
    rows = []
    for action in profile.options:
        i = output_index(action, profile.options)
        if mask[i, DELTA]:
            rows.append({"action": action, "target_delta": float(y[i, DELTA]),
                         **{name: float(estimates[name][i, DELTA]) for name in names}})
    return rows


def _evaluate(cases, models, budget, costs, priors, calibrators, old, *, testing, scorer=None, target_fit=None):
    rows = []
    for row, profile in cases:
        task = profile.task
        item = row.copy()
        results = {}
        rules = BASELINES if task == "heart_rate" else ("ecg_only",)
        for name in rules:
            results[name if task == "heart_rate" else "ppg_only"] = (
                run_policy(profile, budget, name, costs["waveform"]), True)
        if old and task == "heart_rate":
            net, old_cost, old_budget = old
            if old_budget == budget:
                results["v23_binary_voi"] = (run_policy(profile, budget, "learned_voi", old_cost, model=net.predict), True)
        for name in VARIANTS:
            key = f"{name}/{task}" if name == "single_task_voi" else ("conditional_voi" if name == "full_qc" else name)
            results[name] = (run(profile, budget, costs, priors, model=models[key], qc_mode=VARIANTS[name].qc), False)
        if task in calibrators:
            results[CALIBRATED] = (run(profile, budget, costs, priors, model=models["conditional_voi"],
                                       calibrator=calibrators[task], limit=STOP_RISK), False)
        if target_fit is not None:
            results[TARGET_CALIBRATED] = (run(profile, budget, costs, priors, model=models["conditional_voi"],
                                              calibrator=target_fit, limit=STOP_RISK), False)
        if testing:
            oracle, _ = counterfactual(profile, budget, row["reference_bpm"], Objectives(), costs["decision"])
            results["oracle"] = (run(profile, budget, costs, priors, oracle=oracle), False)
            item["conditional_delta"] = _delta_check(profile, budget, row["reference_bpm"], models, costs["decision"])
        for name, (result, legacy) in results.items():
            item[name] = result["rate_bpm"]
            trials = result["stop_trials"]
            item[f"{name}_rank"] = (1 - trials[-1]["risk"] if trials and result["rate_bpm"] is not None else result["quality_rank"])
            if scorer is not None:
                item[f"{name}_shared_rank"] = shared_risk.score(scorer, task, result)
            item[f"{name}_trace"] = _compact(result, legacy=legacy)
            if "qc_distribution_queries" in result:
                item[f"{name}_trace"]["qc_distribution_queries"] = result["qc_distribution_queries"]
        rows.append(item)
    return rows


def _summarize(rows, validation, calibrator, *, shared=False):
    names = [key.removesuffix("_trace") for key in rows[0] if key.endswith("_trace")]
    metrics, pareto = {}, {}
    for name in names:
        info = summarize_rates(rows, name)
        costs = [row[f"{name}_trace"]["cost"] for row in rows]
        rank = f"{name}_rank"
        operating = risk_coverage(rows, name, rank, [0.])[0]
        first_actions = Counter()
        for row in rows:
            actions = [step["action"] for step in row[f"{name}_trace"]["actions"] if not step["action"].startswith("QC:")]
            first_actions[actions[1] if actions[0] == "ACQUIRE_ECG" and len(actions) > 1 else actions[0]] += 1
        metrics[name] = info | {"operating_point": operating, "by_subject": per_subject_rates(rows, name),
                                "mean_cost": {key: mean(c[key] for c in costs) for key in (
                                    "waveform_requests", "qc_requests", "waveform_ms", "qc_ms", "decision_ms", "analysis_ms")},
                                "first_after_ecg_i_or_start": dict(first_actions),
                                "matched_operating_points": matched_operating_points(rows, validation, name, rank)}
        if shared:
            metrics[name]["shared_risk_operating_points"] = matched_operating_points(rows, validation, name, f"{name}_shared_rank")
        if name != "oracle":
            pareto[name] = {"calls": metrics[name]["mean_cost"]["analysis_ms"], "risk": operating["unconditional_failure"]}
    delta = defaultdict(list)
    for row in rows:
        for candidate in row["conditional_delta"]:
            delta[candidate["action"]].append(candidate)
    return {"metrics": metrics, "cost_failure_pareto": pareto_frontier(pareto), "records": rows,
            "calibration_on_test_candidates": calibration_metrics(
                [trial for row in rows for trial in _trials(row, row["conditional_voi_trace"])], calibrator, "subject_record"),
            "delta_with_all_qc_visible_at_initial_state": {action: {
                "n": len(cases), "mean_target_delta": mean(c["target_delta"] for c in cases),
                **{f"{name}_delta_mae": mean(abs(c[name] - c["target_delta"]) for c in cases)
                   for name in ("multitask_voi", "conditional_voi")}} for action, cases in delta.items()},
            "paired_vs_conditional": {name: {k: v for k, v in paired_metrics(rows, name, "conditional_voi").items() if k != "records"}
                                      for name in names if name != "conditional_voi"}}


def audit(sens, bidmc, output, *, budget, epochs, v23_checkpoint=None, corrected=False, paired_bidmc=False):
    source_bidmc, bidmc_rows = read_replay(bidmc)
    datasets = {"bidmc": source_bidmc | {"replay_sha256": digest(bidmc)}}
    if paired_bidmc and not corrected:
        raise ValueError("The paired BIDMC experiment is available in V2.4.1")
    if paired_bidmc:
        matched = defaultdict(set)
        for row, profile in bidmc_rows:
            if row["scenario"] == "original":
                matched[row["case_id"]].add(profile.task)
        rows = [(r, p) for r, p in bidmc_rows if r["scenario"] == "original" and matched[r["case_id"]] == set(TASKS)]
    else:
        source_sens, sens_rows = read_replay(sens)
        datasets["senssmarttech"] = source_sens | {"replay_sha256": digest(sens)}
        rows = sens_rows + bidmc_rows
    split = {name: [(r, p) for r, p in rows if r["split"] == name] for name in ("train", "validation", "test")}
    subjects = {name: sorted({r["subject_id"] for r, _ in cases}) for name, cases in split.items()}
    if sum(map(len, subjects.values())) != len(set().union(*map(set, subjects.values()))):
        raise ValueError("Train, validation and test subject sets overlap")
    if any({p.task for _, p in cases} != set(TASKS) for cases in split.values()):
        raise ValueError("All splits must contain both HR and PR")
    if not corrected and any(r["dataset"] == "bidmc" and p.task == "heart_rate" for name in ("train", "validation") for r, p in split[name]):
        raise ValueError("BIDMC HR is reserved for external testing")
    external_val = [(r, p) for r, p in split["validation"] if r["dataset"] == "bidmc" and p.task == "heart_rate"] if corrected and not paired_bidmc else []
    if corrected and not paired_bidmc:
        split["train"] = [(r, p) for r, p in split["train"] if not (r["dataset"] == "bidmc" and p.task == "heart_rate")]
    fit_val = [(r, p) for r, p in split["validation"] if not external_val or not (r["dataset"] == "bidmc" and p.task == "heart_rate")]
    release = "v241" if corrected else "v24"
    config = {"experiment": release, "budget": budget, "epochs": epochs, "paired_bidmc": paired_bidmc,
              "stop_risk_limit": STOP_RISK, "objectives": asdict(Objectives()), "subjects": subjects,
              "variants": {name: asdict(replace(variant, advantage=corrected)) for name, variant in VARIANTS.items()},
              "replay_sha256": {name: item["replay_sha256"] for name, item in datasets.items()},
              "v23_checkpoint_sha256": digest(v23_checkpoint) if v23_checkpoint else None}
    output.mkdir(parents=True, exist_ok=False)
    provenance = save_provenance(output, config)
    save_json(output / "experiment_config.json", config)
    save_json(output / "dataset_provenance.json", datasets)
    old = _old_checkpoint(v23_checkpoint)
    models, history, costs, priors = _fit(split["train"], budget, epochs, corrected=corrected)
    fits, calibration = _calibrate(fit_val, models["conditional_voi"], budget, costs, priors)
    domain_fits, domain_metadata = _calibrate(external_val, models["conditional_voi"], budget, costs, priors) if external_val else ({}, {})
    scorer = shared_risk.fit(split["train"], budget) if corrected else None
    torch.save({"schema": 1, "budget": budget, "models": {name: {"width": model.width, "variant": asdict(model.variant),
                "state_dict": model.state_dict()} for name, model in models.items()}, "costs": costs, "priors": priors,
                "calibrators": {task: fit.describe() for task, fit in fits.items()}, "stop_risk": STOP_RISK,
                "provenance": provenance, "shared_risk": scorer,
                "target_calibrators": {task: fit.describe() for task, fit in domain_fits.items()}}, output / "models.pt")
    validation = {key: _evaluate(cases, models, budget, costs, priors, fits, old, testing=False, scorer=scorer,
                                 target_fit=domain_fits.get(key[1]) if key[0] == "bidmc" else None)
                  for key, cases in _group(split["validation"]).items()}
    results = {}
    for key, cases in _group(split["test"]).items():
        dataset, task, scenario = key
        val_key = ("senssmarttech", task, scenario) if dataset == "bidmc" and task == "heart_rate" and key not in validation else key
        target_fit = domain_fits.get(task) if dataset == "bidmc" else None
        test = _evaluate(cases, models, budget, costs, priors, fits, old, testing=True, scorer=scorer, target_fit=target_fit)
        result = _summarize(test, validation.get(val_key, []), fits.get(task), shared=corrected) | {
            "risk_threshold_validation_group": "/".join(val_key)}
        if target_fit:
            trials = [trial for row in test for trial in _trials(row, row["conditional_voi_trace"])]
            result["calibration_on_test_candidates_target"] = calibration_metrics(trials, target_fit, "subject_record")
        results["/".join(key)] = result
        print(f"Scored {'/'.join(key)}: {len(cases)} records/windows", flush=True)
    report = config | {"provenance": provenance, "training": history, "training_action_costs_ms": costs,
                       "calibration": calibration, "target_calibration": domain_metadata, "shared_risk": scorer,
                       "qc_prior_validation": validate_priors(fit_val, priors) if corrected else None, "results": results}
    save_json(output / f"{release}_audit.json", report)
    archive = output / f"{release}_experiment.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for name in (f"{release}_audit.json", *EXPORTS[1:]):
            bundle.write(output / name, name)
        if v23_checkpoint:
            bundle.write(v23_checkpoint, "baselines/v23_value_network.pt")
        for name in ("src", "tests", "scripts"):
            root = Path(__file__).resolve().parents[3]
            for path in sorted((root / name).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    bundle.write(path, path.relative_to(root).as_posix())
        bundle.write(root / "pyproject.toml", "pyproject.toml")
    return report


def analyze(record, task, method, checkpoint):
    if method not in (*VARIANTS, CALIBRATED):
        raise ValueError(f"Unsupported V2.4 policy: {method}")
    torch.set_num_threads(1)
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if data["schema"] != 1:
        raise ValueError("Unsupported V2.4 checkpoint")
    options, initial = actions_for(record, task)
    profile = Profile(task, record.shared_duration, options, initial, {}, {})
    key = {CALIBRATED: "conditional_voi", "full_qc": "conditional_voi", "single_task_voi": f"single_task_voi/{task}"}.get(method, method)
    stored = data["models"][key]
    model = ConditionalNetwork(stored["width"], Variant(**stored["variant"]))
    model.load_state_dict(stored["state_dict"])
    calibration = data["calibrators"].get(task)
    if method == CALIBRATED and calibration is None:
        raise ValueError(f"This checkpoint has no fitted risk calibrator for {task}")
    fit = RiskCalibrator(**calibration) if method == CALIBRATED else None
    return run(profile, data["budget"], data["costs"], data["priors"], model=model.eval(),
               calibrator=fit, limit=data["stop_risk"], live=channel_observers(record),
               qc_mode=VARIANTS[method].qc if method in VARIANTS else None) | {
        "record": record.describe(), "task": task, "method": method, "checkpoint_sha256": digest(checkpoint)}
