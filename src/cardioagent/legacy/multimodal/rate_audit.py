"""Measure incremental modality value on original and ECG-degraded recordings."""

from dataclasses import replace
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

import numpy as np

from cardioagent.legacy.multimodal.acquisition import STRATEGIES
from cardioagent.common.io import digest, save_json
from cardioagent.legacy.multimodal.dataset import load_manifest
from cardioagent.legacy.multimodal.evaluation_reference import evaluation_cases, load_case_record
from cardioagent.legacy.multimodal.paired_metrics import confidence_rank, paired_metrics, pareto_frontier, per_subject_rates, risk_coverage, summarize_rates
from cardioagent.legacy.multimodal.rate_policy import ecg_first, observe_rate
from cardioagent.legacy.multimodal.provenance import save_provenance
from cardioagent.common.recording import MODALITIES, Recording
from cardioagent.common.signal_tools import bandpass
from cardioagent.legacy.multimodal.workflow import preferred_channel, rate_summary


SCENARIOS = {
    "original": {"kind": "observed", "method": "published WFDB"},
    "low_snr": {"kind": "synthetic", "method": "1–45 Hz noise; -6 dB relative to 5–40 Hz ECG RMS"},
    "baseline_drift": {"kind": "synthetic", "method": "0.25 Hz sinusoid; amplitude 4 times 5–40 Hz ECG RMS"},
    "emg": {"kind": "synthetic", "method": "25–150 Hz filtered white noise; 0 dB relative to 5–40 Hz ECG RMS"},
    "local_dropout": {"kind": "synthetic", "method": "flat fill across 25% of each ECG lead at a seeded location"},
    "missing_ecg": {"kind": "simulated_missingness", "method": "ECG absent; PPG and PCG unchanged"},
    "lead_i_low_snr": {"kind": "synthetic", "method": "-6 dB ECG noise on lead I only"},
    "lead_i_dropout": {"kind": "synthetic", "method": "25% flat interval on ECG lead I only"},
    "ppg_low_snr": {"kind": "synthetic", "method": "-6 dB noise in 0.45–5 Hz on all PPG channels"},
    "pcg_low_snr": {"kind": "synthetic", "method": "-6 dB noise in 25–180 Hz on PCG"},
    "aux_dropout": {"kind": "synthetic", "method": "25% flat interval on PPG and PCG"},
    "joint_low_snr": {"kind": "synthetic", "method": "all modalities receive separate -6 dB band-limited noise"},
    "ecg_ppg_low_snr": {"kind": "synthetic", "method": "all ECG and PPG channels receive noise; PCG original"},
    "ecg_pcg_low_snr": {"kind": "synthetic", "method": "all ECG and PCG channels receive noise; PPG original"},
}
COMBINATIONS = tuple(combination for n in (1, 2, 3) for combination in combinations(MODALITIES, n))
METHODS = tuple("+".join(combination) for combination in COMBINATIONS) + STRATEGIES
CONDITIONAL_SCENARIOS = ("original", "low_snr", "missing_ecg", "ppg_low_snr", "pcg_low_snr",
                         "ecg_ppg_low_snr", "ecg_pcg_low_snr", "joint_low_snr")
PERTURBATIONS = {
    "low_snr": ("low_snr", ("ecg",)), "baseline_drift": ("baseline_drift", ("ecg",)),
    "emg": ("emg", ("ecg",)), "local_dropout": ("dropout", ("ecg",)),
    "lead_i_low_snr": ("low_snr", ("ecg",)), "lead_i_dropout": ("dropout", ("ecg",)),
    "ppg_low_snr": ("low_snr", ("ppg",)), "pcg_low_snr": ("low_snr", ("pcg",)),
    "aux_dropout": ("dropout", ("ppg", "pcg")), "joint_low_snr": ("low_snr", MODALITIES),
    "ecg_ppg_low_snr": ("low_snr", ("ecg", "ppg")), "ecg_pcg_low_snr": ("low_snr", ("ecg", "pcg")),
}


def perturb_recording(record: Recording, scenario: str) -> Recording:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if scenario == "original":
        return record
    if scenario == "missing_ecg":
        return replace(record, waves={m: w for m, w in record.waves.items() if m != "ecg"},
                       unavailable=record.unavailable | {"ecg": "removed in M1.5 missingness experiment"})

    seed = int.from_bytes(sha256(f"20261111:{record.record_id}:{scenario}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    effect, modalities = PERTURBATIONS[scenario]
    updated = record.waves.copy()
    for modality in modalities:
        if modality not in updated:
            continue
        wave = updated[modality]
        if scenario.startswith("lead_i_") and "I" not in wave.channels:
            raise ValueError(f"{record.record_id} has no lead I for {scenario}")
        samples = wave.values.copy()
        width = len(samples) // 4
        start = int(rng.integers(len(samples) // 10, len(samples) - width - len(samples) // 10))
        t = np.arange(len(samples)) / wave.fs
        physiological_band = {"ecg": (5, 40), "ppg": (0.45, 5), "pcg": (25, 180)}[modality]
        for name, channel in zip(wave.channels, samples.T):
            if scenario.startswith("lead_i_") and name != "I":
                continue
            if effect == "dropout":
                channel[start:start + width] = np.median(channel)
                continue
            rms = np.std(bandpass(channel, wave.fs, *physiological_band))
            if effect == "baseline_drift":
                phase = rng.uniform(0, 2 * np.pi)
                channel += 4 * rms * np.sin(2 * np.pi * 0.25 * t + phase)
                continue
            noise_band = (25, 150) if effect == "emg" else (1, 45) if modality == "ecg" else physiological_band
            snr_db = 0 if effect == "emg" else -6
            noise = bandpass(rng.standard_normal(len(channel)), wave.fs, *noise_band)
            channel += noise / np.std(noise) * rms * 10 ** (-snr_db / 20)
        updated[modality] = replace(wave, values=samples)
    return replace(record, waves=updated)


def evidence_rank(result: dict, evidence: list[dict]) -> int:
    """Rank accepted estimates by fusion status and observed signal/event quality."""
    if result["rate_bpm"] is None:
        return 0
    selected = set(result["evidence_ids"])
    scores = {}
    for item in evidence:
        if item["evidence_id"] not in selected:
            continue
        band_power = item["signal_quality"].get("physiology_band_power_fraction") or 0
        event = item["event_quality"]
        if item["modality"] == "pcg":
            stability = float(np.clip(event.get("autocorrelation") or 0, 0, 1))
        else:
            mad = event.get("interval_mad_ratio") or 0
            stability = (event.get("plausible_interval_fraction") or 0) * (1 - min(1, mad / 0.12))
        score = (band_power + stability) / 2
        scores[item["modality"]] = max(scores.get(item["modality"], 0), score)
    quality = round(20 * mean(scores.values())) if scores else 0
    return 21 * confidence_rank(result["status"]) + quality


def evaluate_case(record: Recording, cached: dict[tuple[str, str], dict]) -> dict:
    computed = cached.copy()

    def replay(modality: str, channel: str) -> dict:
        key = (modality, channel)
        if key not in computed:
            computed[key] = observe_rate(record, modality, channel)
        # Measurement computation is shared across methods; policy cost here is an evidence request.
        return computed[key] | {"cache_hit": True}

    adaptive = {name: ecg_first(record, 6, replay, strategy=name) for name in STRATEGIES}
    preferred = {m: replay(m, preferred_channel(m, wave.channels)) for m, wave in record.waves.items()}
    outcomes = {}
    requests = {}
    ranks = {}
    for modalities in COMBINATIONS:
        evidence = [preferred[m] for m in modalities if m in preferred]
        method = "+".join(modalities)
        outcomes[method] = rate_summary(evidence, 0, record.shared_duration)
        requests[method] = len(evidence)
        ranks[method] = evidence_rank(outcomes[method], evidence)
    for name, result in adaptive.items():
        outcomes[name] = {k: v for k, v in result.items() if k not in ("evidence", "observed")}
        requests[name] = result["cost"]["waveform_requests"]
        ranks[name] = evidence_rank(result, result["evidence"])
    fixed = {method: {k: v for k, v in result.items() if k != "observed"} for method, result in outcomes.items()}
    return {
        "outcomes": fixed, "preferred_measurements": preferred,
        "waveform_requests": requests, "new_measurements_for_shared_cache": len(computed) - len(cached),
        "ecg_quality_stratum": preferred.get("ecg", {}).get("quality", "missing"),
        **{method: outcomes[method]["rate_bpm"] for method in METHODS},
        **{f"{method}_rank": rank for method, rank in ranks.items()},
    }


def summarize_group(rows: list[dict]) -> dict:
    metrics = {method: summarize_rates(rows, method) for method in METHODS}
    calls = {method: mean(r["waveform_requests"][method] for r in rows) if rows else None for method in METHODS}
    paired = {}
    for method in ("ecg+ppg", "ecg+pcg", "ecg+ppg+pcg", *STRATEGIES):
        paired[method] = paired_metrics(rows, "ecg", method) | {
            "additional_mean_waveform_requests": calls[method] - calls["ecg"] if rows else None,
        }
    points = {method: {"calls": calls[method], "risk": 1 - item["within_2_bpm_of_all"] if rows else None}
              for method, item in metrics.items()}
    return {
        "metrics": metrics,
        "ungated_single_modalities": {
            modality: summarize_rates([
                row | {"ungated": row["preferred_measurements"].get(modality, {}).get("rate_bpm")} for row in rows
            ], "ungated") for modality in MODALITIES
        },
        "paired_vs_ecg": paired,
        "paired_alternate_vs_ppg_fallback": paired_metrics(rows, "ppg_fallback", "alternate_lead_first"),
        "mean_waveform_requests": calls,
        "rule_paths_equal_records": sum(r["outcomes"]["ppg_fallback"]["actions"] ==
                                        r["outcomes"]["alternate_lead_first"]["actions"] for r in rows),
        "risk_coverage": {method: risk_coverage(rows, method, f"{method}_rank") for method in METHODS},
        "risk_cost_points": points, "pareto_frontier": pareto_frontier(points),
        "by_subject": {method: per_subject_rates(rows, method) for method in METHODS},
    }


def run_rate_audit(dataset: Path, output: Path, *, split: str = "all", packet=None, matrix: str = "conditional") -> dict:
    manifest = load_manifest(dataset)
    cases, reference = evaluation_cases(dataset, manifest, split, packet)
    scenarios = CONDITIONAL_SCENARIOS if matrix == "conditional" else tuple(SCENARIOS)
    output.mkdir(parents=True, exist_ok=False)
    provenance = save_provenance(output, {"experiment": "V2.2 conditional modality value", "seed": 20261111,
                                          "scenarios": scenarios, "strategies": STRATEGIES, "tool_cap": 6,
                                          "split": split, "reference": reference,
                                          "dataset_manifest_sha256": digest(dataset / "manifest.json")})
    all_cases = {scenario: [] for scenario in scenarios}
    for i, ref in enumerate(cases, 1):
        record = load_case_record(dataset, ref)
        originals = {(m, ch): observe_rate(record, m, ch) for m, wave in record.waves.items()
                     for ch in (wave.channels if m == "ecg" else (preferred_channel(m, wave.channels),))}
        for scenario, rows in all_cases.items():
            case = perturb_recording(record, scenario)
            cached = {
                (m, ch): obs for (m, ch), obs in originals.items()
                if m in case.waves and (case.waves[m] is record.waves[m] or
                                        (m == "ecg" and scenario.startswith("lead_i_") and ch != "I"))
            }
            rows.append(ref | evaluate_case(case, cached))
        if i % 50 == 0 or i == len(cases):
            print(f"Modality audit: {i}/{len(cases)} recordings", flush=True)

    results = {}
    for scenario, rows in all_cases.items():
        results[scenario] = {
            split: summarize_group([r for r in rows if split == "all" or r["split"] == split])
            for split in ("all", "train", "validation", "test")
        }
        test = [row for row in rows if row["split"] == "test"]
        results[scenario]["test_by_ecg_quality"] = {
            stratum: summarize_group([row for row in test if row["ecg_quality_stratum"] == stratum])
            for stratum in ("usable", "review", "missing")
        }
    original_test = [row for row in all_cases["original"] if row["split"] == "test"]
    natural = {
        group: summarize_group([
            row for row in original_test if row["preferred_measurements"].get("ecg", {}).get("quality") == group
        ]) for group in ("usable", "review")
    }
    natural["before_activity"] = summarize_group([row for row in original_test if row["activity"] == "B"])
    natural["after_activity"] = summarize_group([row for row in original_test if row["activity"] == "A"])
    report = {
        "dataset": manifest["dataset"], "dataset_manifest_sha256": digest(dataset / "manifest.json"),
        "reference_provenance": reference,
        "scenarios": {s: SCENARIOS[s] for s in scenarios}, "bootstrap": "4000 percentile replicates resampled by subject",
        "methods": "shared detectors and gate; channel-first aggregation for all seven fixed combinations",
        "cost_semantics": "offline replay with a shared measurement cache; compare waveform evidence requests",
        "risk_semantics": "unconditional refusal or absolute error >2 bpm; risk-coverage MAE on returned estimates",
        "coverage_rank_semantics": "ordinal fusion status, then band-power and event-stability heuristic; no fitted confidence",
        "natural_strata": "observed ECG quality gate is a proxy; activity labels are independent of gate",
        "results": results, "natural_test_strata": natural, "records": all_cases, "provenance": provenance,
    }
    save_json(output / "incremental_value.json", report)
    return report


def plot_conditional(report_path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    report = json.loads(report_path.read_text(encoding="utf-8"))
    conditions = [name for name in CONDITIONAL_SCENARIOS if name in report["results"]]
    methods = STRATEGIES
    fig = Figure(figsize=(14, 3.3 * ((len(conditions) + 1) // 2)), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots((len(conditions) + 1) // 2, 2, squeeze=False)
    for ax, condition in zip(axes.flat, conditions):
        group = report["results"][condition]["test"]
        for index, method in enumerate(methods):
            point = group["risk_cost_points"][method]
            if point["risk"] is not None:
                ax.scatter(point["calls"], point["risk"], label=method, marker=("o", "s", "D", "x", "^")[index])
        ax.set(title=condition, xlabel="Mean waveform requests", ylabel="Refusal or error >2 bpm", xlim=(-0.1, 5.2), ylim=(-0.04, 1.04))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(conditions):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(report_path.parent / "conditional_risk_cost.png", dpi=170)

    fig = Figure(figsize=(12, 4.6), layout="constrained")
    FigureCanvasAgg(fig)
    fig.suptitle("Heuristic selective risk–coverage")
    subset = [name for name in ("original", "joint_low_snr") if name in conditions]
    axes = fig.subplots(1, len(subset), squeeze=False)
    for ax, condition in zip(axes.flat, subset):
        for index, method in enumerate(methods):
            points = report["results"][condition]["test"]["risk_coverage"][method]
            if points:
                ax.plot([p["coverage"] for p in points], [p["mae_bpm"] for p in points],
                        label=method, marker=("o", "s", "D", "x", "^")[index])
        ax.set(title=condition, xlabel="Coverage", ylabel="MAE on returned estimates (bpm)", xlim=(0, 1.03))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(report_path.parent / "conditional_risk_coverage.png", dpi=170)
