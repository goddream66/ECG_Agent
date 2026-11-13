"""Paired audit of rules and LLM policies under explicit resource limits."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from statistics import mean

from .acquisition import STRATEGIES
from .active_agent import active_agent
from .dataset import digest, load_manifest, save_json
from .evaluation_reference import evaluation_cases, load_case_record
from .paired_metrics import confidence_rank, paired_metrics, pareto_frontier, per_subject_rates, risk_coverage, summarize_rates
from .provenance import experiment_provenance
from .rate_audit import perturb_recording
from .rate_policy import ecg_first, measure_for_audit, observe_rate
from .workflow import Analysis


LEGACY_MODES = ("ppg_fallback", "alternate_lead_first", "single", "multi")
ACTIVE_MODES = (*STRATEGIES, "single_active", "multi_active")
STRUCTURED_MODES = (*STRATEGIES, "single_active", "two_stage_llm_coordinator")
LLM_BUDGET = 12
LABELS = {
    "rule": "ECG → PPG fallback", "lead_rule": "ECG → Alternate ECG lead",
    "ecg_only": "ECG only", "ppg_fallback": "ECG → PPG fallback",
    "alternate_lead_first": "ECG → Alternate ECG lead", "quality_aware": "Quality aware",
    "cost_aware_eig": "Fixed-Gaussian EIG", "single_active": "Single active Agent",
    "multi_active": "Two-stage LLM Coordinator", "two_stage_llm_coordinator": "Two-stage LLM Coordinator",
    "single": "Single Agent (full modality)", "multi": "Multi-Agent (full modality)",
}


def run_case(ref, scenario, budget, modes, args, output, client) -> dict:
    record = perturb_recording(load_case_record(args.dataset_root, ref), scenario)
    case = ref.copy()
    for mode in modes:
        dest = output / "reports" / scenario / str(budget) / ref["case_id"] / f"{mode}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        report = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else None
        if report is None or report["agent_errors"]:
            def observe(modality, channel):
                return observe_rate(record, modality, channel)

            if mode in STRATEGIES:
                result = ecg_first(record, budget, observe, strategy=mode)
                report = {"rate_summary": {k: v for k, v in result.items() if k not in ("evidence", "actions", "cost")},
                          "evidence": result["evidence"], "actions": result["actions"], "cost": result["cost"],
                          "agent_errors": []}
            elif mode in ("single_active", "multi_active", "two_stage_llm_coordinator"):
                report = active_agent(record, budget, observe, client, mode, LLM_BUDGET,
                                      structured=args.protocol == "structured")
            else:
                report = Analysis(record, 0, record.shared_duration, None, dest.parent, client,
                                  action_budget=budget, measure=measure_for_audit).run(mode)
            temporary = dest.with_suffix(".tmp")
            save_json(temporary, report)
            temporary.replace(dest)
        if report["cost"]["action_budget_units"] > budget:
            raise RuntimeError(f"{mode} exceeded the action budget on {record.record_id}")
        if args.protocol != "legacy" and report["cost"]["llm_invocations"] > LLM_BUDGET:
            raise RuntimeError(f"{mode} exceeded the LLM budget on {record.record_id}")
        summary = report["rate_summary"]
        case[mode] = summary["rate_bpm"] if not report["agent_errors"] else None
        case[f"{mode}_rank"] = confidence_rank(summary["status"])
        case[f"{mode}_cost"] = report["cost"]
        case[f"{mode}_errors"] = report["agent_errors"]
        case[f"{mode}_path"] = [step["action"] for step in report["actions"]]
        case[f"{mode}_invalid_llm_actions"] = sum(step.get("result") == "invalid_action"
                                                   for step in report.get("agent_trace", []))
    return case


def summarize_agent_cases(cases: list[dict], modes: tuple[str, ...]) -> dict:
    metrics = {}
    for mode in modes:
        costs = [row[f"{mode}_cost"] for row in cases]
        averages = {key: mean(cost[key] for cost in costs) for key in costs[0]}
        metrics[mode] = summarize_rates(cases, mode) | {
            "mean_cost": averages,
            "mean_total_tokens": (averages["input_tokens"] + averages["output_tokens"]
                                  if not averages["llm_usage_missing"] else None),
            "failed_runs": sum(bool(row[f"{mode}_errors"]) for row in cases),
            "invalid_llm_actions": sum(row[f"{mode}_invalid_llm_actions"] for row in cases),
            "action_paths": dict(Counter(" → ".join(row[f"{mode}_path"]) for row in cases)),
            "by_subject": per_subject_rates(cases, mode),
            "risk_coverage": risk_coverage(cases, mode, f"{mode}_rank"),
        }
    first, second = modes[-2:]
    pareto_points = {mode: {"calls": m["mean_cost"]["waveform_tool_calls"],
                           "risk": 1 - m["within_2_bpm_of_all"]} for mode, m in metrics.items()}
    return {
        "metrics": metrics,
        "paired_vs_ppg_fallback": {mode: paired_metrics(cases, "ppg_fallback", mode)
                                   for mode in modes if mode != "ppg_fallback"},
        "paired_multi_vs_single": paired_metrics(cases, first, second),
        "rule_paths_equal_records": sum(row["ppg_fallback_path"] == row["alternate_lead_first_path"] for row in cases),
        "waveform_cost_pareto": pareto_frontier(pareto_points), "records": cases,
    }


def run_agent_audit(args, output: Path, client) -> dict:
    manifest = load_manifest(args.dataset_root)
    rows, reference = evaluation_cases(args.dataset_root, manifest, "test", args.adjudicated_packet)
    if args.per_subject is not None:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["subject_id"]].append(row)
        rows = []
        for group in grouped.values():
            before = [r for r in group if r["activity"] == "B"]
            after = [r for r in group if r["activity"] == "A"]
            ordered = [r for pair in zip(before, after) for r in pair]
            ordered += [r for r in group if r not in ordered]
            rows.extend(ordered[:args.per_subject])
    budgets = args.budgets or [6]
    scenarios = args.scenarios or ["original"]
    if len(set(budgets)) != len(budgets) or len(set(scenarios)) != len(scenarios):
        raise ValueError("Provide each budget and scenario at most once")
    modes = {"active": ACTIVE_MODES, "structured": STRUCTURED_MODES, "legacy": LEGACY_MODES}[args.protocol]
    config = {
        "dataset_manifest_sha256": digest(args.dataset_root / "manifest.json"), "model": args.model,
        "budgets": budgets, "record_ids": [r["record_id"] for r in rows], "case_ids": [r["case_id"] for r in rows],
        "measurement": "same rate detectors, quality gate, and same-window waveform for each method",
        "reference_provenance": reference, "protocol": args.protocol, "scenarios": scenarios, "modes": list(modes),
        "budget_unit": ("one waveform request (including mandatory initial ECG when available)"
                        if args.protocol != "legacy" else "one waveform or coordinator recheck request"),
        "llm_invocation_cap": LLM_BUDGET if args.protocol != "legacy" else None,
        "bootstrap": "4000 percentile replicates resampled by subject; small subject-held-out pilot",
        "workers": args.workers, "model_max_tokens": 64 if args.protocol != "legacy" else 220,
    }
    provenance = experiment_provenance(config)
    config = config | {key: provenance[key] for key in ("source_tree_sha256", "configuration_sha256", "git_commit")}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "agent_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Resume requires the original code, reference, model, budgets and case selection")
    save_json(config_path, config)
    save_json(output / "source_manifest.json", provenance)
    comparisons = {}
    for scenario in scenarios:
        for budget in budgets:
            completed = {}
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                pending = [executor.submit(run_case, ref, scenario, budget, modes, args, output, client) for ref in rows]
                for index, future in enumerate(as_completed(pending), 1):
                    case = future.result()
                    completed[case["case_id"]] = case
                    if index == 1 or index % 5 == 0 or index == len(rows):
                        print(f"{scenario}, budget {budget}: {index}/{len(rows)}", flush=True)
            cases = [completed[row["case_id"]] for row in rows]
            comparisons[f"{scenario}/{budget}"] = summarize_agent_cases(cases, modes)
            save_json(output / "agent_comparison.json", config | {"results": comparisons})
    return comparisons


def plot_actual_cost(report_path: Path, output: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report["results"]
    fig = Figure(figsize=(13.5, 3.9 * len(results) + 0.75), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(results), 3, squeeze=False)
    old_schema = "protocol" not in report
    for axis_row, (condition, result) in zip(axes, results.items()):
        plotted_costs = [[], [], []]
        for index, (mode, metric) in enumerate(result["metrics"].items()):
            costs = metric.get("mean_cost", {})
            points = (costs.get("waveform_tool_calls"), metric.get("mean_total_tokens"),
                      costs.get("wall_time_s", metric.get("mean_seconds")))
            for ax, point, plotted in zip(axis_row, points, plotted_costs):
                if point is not None and metric["mae_bpm"] is not None:
                    plotted.append(point)
                    ax.scatter(point, metric["mae_bpm"], label=LABELS.get(mode, mode), color=f"C{index}",
                               marker=("o", "x", "s", "^", "v", "D", "P")[index], s=66, alpha=0.8)
        for ax, xlabel, plotted in zip(axis_row, ("Executed waveform tools per record", "LLM tokens per record",
                                                 "Wall time per record (s)"), plotted_costs):
            ax.set(xlabel=xlabel, ylabel="MAE on returned estimates (bpm)")
            ax.grid(alpha=0.2)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            if plotted:
                high = max(plotted) or 1
                ax.set_xlim(-0.035 * high if min(plotted) == 0 else 0, 1.08 * high)
        axis_row[0].set_title(f"{condition}  |  shared cohort")
        if old_schema:
            for ax in axis_row[:2]:
                ax.text(0.5, 0.5, "Unavailable in historical report", transform=ax.transAxes, ha="center")
        axis_row[-1].legend(loc="best", fontsize=7)
    fig.suptitle(f"SensSmartTech pilot · {len(report['record_ids'])} paired cases")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "risk_actual_cost.png", dpi=180)

    from matplotlib.ticker import PercentFormatter
    fig = Figure(figsize=(12, 4 * len(results)), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(results), 2, squeeze=False)
    for (left, right), (condition, result) in zip(axes, results.items()):
        for i, (mode, metric) in enumerate(result["metrics"].items()):
            seconds = metric.get("mean_cost", {}).get("wall_time_s", metric.get("mean_seconds"))
            if seconds is None:
                continue
            style = {"color": f"C{i}", "label": LABELS.get(mode, mode),
                     "marker": ("o", "x", "s", "^", "v", "D", "P")[i], "s": 65}
            left.scatter(seconds, metric["coverage"], **style)
            if metric.get("within_2_bpm_of_all") is not None:
                right.scatter(seconds, 1 - metric["within_2_bpm_of_all"], **style)
        for ax, label in ((left, "Coverage"), (right, "Unconditional refusal or >2 bpm error")):
            ax.set(xlabel="Wall time per record (s)", ylabel=label, ylim=(-0.03, 1.03), title=condition)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=0.2)
        left.legend(fontsize=7)
    fig.savefig(output / "coverage_unconditional_risk.png", dpi=180)
