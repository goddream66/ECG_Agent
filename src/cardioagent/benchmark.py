"""Reproducible V1 and V2 measurements on the same full-record references."""

from collections import Counter
from importlib.metadata import PackageNotFoundError, version
import csv
from pathlib import Path
import platform
from statistics import mean, median

import numpy as np

from .dataset import digest, load_manifest, save_json
from .provenance import experiment_provenance, save_provenance
from .recording import MODALITIES, Recording
from .v2 import inspect_recording
from .workflow import Analysis


MODES = ("v1_fixed", "v1_single", "v1_multi", "v2_events")
SCENARIOS = {"clean": set()} | {f"drop_{m}": {m} for m in MODALITIES}
SOURCE_FILES = ("cli.py", "recording.py", "workflow.py", "signal_tools.py", "model_adapters.py")
UPSTREAM_FILES = (
    "upstream/ecgfounder/net1d.py", "upstream/ecgfounder/dataset.py",
    "upstream/ecgfounder/tasks.txt", "upstream/ecgfounder/util.py",
    "upstream/anyppg/resnet1d.py",
    "upstream/cardiophon-release/heart_sound_classification/byol_a2/models.py",
    "upstream/cardiophon-release/heart_sound_classification/byol_a2/dataset.py",
    "upstream/cardiophon-release/heart_sound_classification/config/config_4k.yaml",
)
PREPROCESSING = {
    "ecg": {"target_hz": 500, "input_s": 10, "lead": "I", "notch_hz": 50,
            "bandpass_hz": [0.67, 40], "normalization": "window z-score",
            "output": "1024 features and 150 uncalibrated class scores (top 5 reported)"},
    "ppg": {"target_hz": 125, "input_s": 10, "normalization": "window z-score",
            "output": "512 features"},
    "pcg": {"target_hz": 4000, "segment_s": 0.95, "n_fft": 512, "win_length": 512,
            "hop_length": 160, "mel_bins": 64, "mel_range_hz": [60, 2000],
            "normalization": "per-segment log-mel z-score", "aggregation": "mean of complete segments",
            "output": "3072 features"},
}


def freeze_assets(root: Path) -> dict:
    assets = []
    with (root / "SHA256SUMS.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            path = root / row["Path"]
            if path.stat().st_size != int(row["Bytes"]) or digest(path) != row["SHA256"].lower():
                raise ValueError(f"Model/runtime checksum mismatch: {row['Path']}")
            assets.append({"path": row["Path"], "bytes": int(row["Bytes"]), "sha256": row["SHA256"].lower()})
    if len(assets) != 5 or len({item["path"] for item in assets}) != 5:
        raise ValueError("Expected the original three weights, Qwen weights, and runtime archive")
    package = Path(__file__).parent
    versions = {}
    for name in ("numpy", "scipy", "wfdb", "torch", "torchaudio", "langchain", "langgraph", "langchain-openai"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(), "packages": versions, "verified_assets": assets,
        "upstream_source_sha256": {path: digest(root / path) for path in UPSTREAM_FILES},
        "v1_source_sha256": {name: digest(package / name) for name in SOURCE_FILES},
        "experiment_source": experiment_provenance({"experiment": "model asset freeze", "preprocessing": PREPROCESSING}),
        "preprocessing_v1": PREPROCESSING,
    }


def select_records(manifest: dict, split: str, limit: int | None) -> list[dict]:
    rows = [row for row in manifest["records"] if split == "all" or row["split"] == split]
    if limit is None:
        return rows
    if not 0 < limit <= len(rows):
        raise ValueError(f"--limit must be between 1 and {len(rows)} for {split}")
    # Deterministic coverage of the selected split across its published row order.
    return [rows[(2 * i + 1) * len(rows) // (2 * limit)] for i in range(limit)]


def rate_metrics(rows: list[dict], key: str) -> dict:
    pairs = [(r[key], r["reference_bpm"]) for r in rows if r.get(key) is not None]
    errors = [abs(x - ref) for x, ref in pairs]
    return {
        "records": len(rows), "scored": len(errors),
        "coverage": round(len(errors) / len(rows), 6) if rows else None,
        "mae_bpm": round(mean(errors), 3) if errors else None,
        "median_absolute_error_bpm": round(median(errors), 3) if errors else None,
        "rmse_bpm": round(mean(e * e for e in errors) ** 0.5, 3) if errors else None,
        "p95_absolute_error_bpm": round(float(np.percentile(errors, 95)), 3) if errors else None,
        **{f"within_{limit}_bpm_of_all": round(sum(e <= limit for e in errors) / len(rows), 6) if rows else None
           for limit in (1, 2, 5)},
        "possible_octave_errors": sum(0.45 <= x / ref <= 0.55 or 1.8 <= x / ref <= 2.2 for x, ref in pairs),
    }


def summarize(rows: list[dict]) -> dict:
    fused = rows[0]["mode"] != "v2_events"
    return {
        "fusion": rate_metrics(rows, "rate_bpm") if fused else None,
        "by_modality": {m: rate_metrics(rows, f"{m}_bpm") for m in MODALITIES},
        "by_activity": {activity: rate_metrics([r for r in rows if r["activity"] == activity], "rate_bpm")
                        for activity in ("A", "B")} if fused else None,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "tool_calls_mean": round(mean(r["tool_calls"] for r in rows), 3),
        "duration_s_mean": round(mean(r["duration_s"] for r in rows), 3),
        "records_with_agent_errors": sum(bool(r["agent_errors"]) for r in rows),
    }


def paired_delta(rows: list[dict], baseline: str, mode: str) -> dict:
    left = {r["record_id"]: r for r in rows if r["mode"] == baseline}
    right = {r["record_id"]: r for r in rows if r["mode"] == mode}
    shared = [(l, right[k]) for k, l in left.items() if l["rate_bpm"] is not None and right[k]["rate_bpm"] is not None]
    return {
        "shared_scored": len(shared),
        "candidate_minus_fixed_mae_bpm": round(mean(
            abs(r["rate_bpm"] - r["reference_bpm"]) - abs(l["rate_bpm"] - l["reference_bpm"])
            for l, r in shared
        ), 3) if shared else None,
        "only_fixed_scored": sum(l["rate_bpm"] is not None and right[k]["rate_bpm"] is None for k, l in left.items()),
        "only_candidate_scored": sum(r["rate_bpm"] is not None and left[k]["rate_bpm"] is None for k, r in right.items()),
    }


def run_benchmark(args, output: Path, models, client) -> dict:
    from time import perf_counter

    manifest = load_manifest(args.dataset_root)
    selected = select_records(manifest, args.split, args.limit)
    modes = args.modes or ["v1_fixed", "v2_events"]
    scenarios = args.scenarios or ["clean"]
    if len(modes) != len(set(modes)) or len(scenarios) != len(set(scenarios)):
        raise ValueError("Select each mode and scenario only once")
    rows = []
    for scenario in scenarios:
        for i, ref in enumerate(selected, 1):
            record = Recording.read(args.dataset_root / "WFDB", ref["record_id"], SCENARIOS[scenario])
            for mode in modes:
                dest = output / "reports" / scenario / mode / record.record_id
                dest.mkdir(parents=True, exist_ok=False)
                if mode == "v2_events":
                    began = perf_counter()
                    report = inspect_recording(record, documented_common_grid=True)
                    rates = {item["modality"]: item["rate_bpm"] for item in report["observations"]}
                    summary = {"status": "observation_only", "rate_bpm": None}
                    tool_calls, duration, errors = len(report["observations"]), perf_counter() - began, []
                else:
                    analysis = Analysis(record, 0, record.shared_duration, models, dest, client, args.with_models)
                    report = analysis.run(mode.removeprefix("v1_"))
                    summary = report["rate_summary"]
                    rates = {m: item["rate_bpm"] for m, item in summary["observed"].items()}
                    tool_calls, duration, errors = report["tool_calls"], report["duration_s"], report["agent_errors"]
                save_json(dest / "report.json", report)
                rows.append(ref | {
                    "mode": mode, "scenario": scenario, "rate_bpm": summary["rate_bpm"],
                    "status": summary["status"], "tool_calls": tool_calls, "duration_s": round(duration, 4),
                    "agent_errors": errors,
                    "report": (dest / "report.json").relative_to(output).as_posix(),
                    **{f"{m}_bpm": rates.get(m) for m in MODALITIES},
                })
            if i == 1 or i % 25 == 0 or i == len(selected):
                print(f"{scenario}: {i}/{len(selected)} records", flush=True)
    metrics = {}
    for scenario in scenarios:
        selected_rows = [row for row in rows if row["scenario"] == scenario]
        metrics[scenario] = {mode: summarize([row for row in selected_rows if row["mode"] == mode]) for mode in modes}
        if "v1_fixed" in modes:
            metrics[scenario]["paired_vs_v1_fixed"] = {
                mode: paired_delta(selected_rows, "v1_fixed", mode) for mode in ("v1_single", "v1_multi") if mode in modes
            }
    payload = {
        "dataset": manifest["dataset"], "dataset_manifest_sha256": digest(args.dataset_root / "manifest.json"),
        "reference": manifest["reference"], "split": args.split, "limit": args.limit, "with_models_v1": args.with_models,
        "llm": args.model if client else None, "split_seed": manifest["split_seed"],
        "v1_weights_manifest_sha256": digest(args.models / "SHA256SUMS.csv") if args.with_models else None,
        "v2_observation_only": True,
        "metrics": metrics, "records": rows,
    }
    payload["provenance"] = save_provenance(output, {
        "experiment": "baseline benchmark", "dataset_manifest_sha256": payload["dataset_manifest_sha256"],
        "split": args.split, "modes": modes, "scenarios": scenarios,
        "with_models": args.with_models, "llm": payload["llm"], "record_ids": [row["record_id"] for row in selected],
    })
    save_json(output / "benchmark.json", payload)
    return payload
