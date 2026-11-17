import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
from statistics import mean
from uuid import uuid4

from langchain_openai import ChatOpenAI

from cardioagent.legacy.multimodal.model_adapters import PretrainedModels
from cardioagent.ecg_agent.models.ecgfounder import normalize_ecg_device
from cardioagent.common.recording import MODALITIES, Recording
from cardioagent.legacy.multimodal.workflow import Analysis


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze synchronized SensSmartTech WFDB recordings.")
    parser.add_argument("--data", type=Path, default=Path("data/senssmarttech/WFDB"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("runs"))
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("analyze", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--mode", choices=("fixed", "single", "multi"), default="fixed")
        sub.add_argument("--with-models", action="store_true", help="Enable pretrained model tools")
        sub.add_argument("--omit", choices=MODALITIES, action="append", default=[])
        sub.add_argument("--base-url", default=os.environ.get("CARDIOAGENT_BASE_URL", "http://127.0.0.1:8089/v1"))
        sub.add_argument("--model", default=os.environ.get("CARDIOAGENT_MODEL", "qwen3-4b"))
        if command == "analyze":
            sub.add_argument("record")
            sub.add_argument("--start", type=float, default=0)
            sub.add_argument("--end", type=float)
        else:
            sub.add_argument("records", nargs="*")
            sub.add_argument("--reference", type=Path)

    from cardioagent.legacy.multimodal.benchmark import MODES, SCENARIOS
    from cardioagent.legacy.multimodal.acquisition import LEGACY_NAMES, STRATEGIES
    from cardioagent.legacy.multimodal.dataset import DATASET_ROOT
    from cardioagent.legacy.multimodal.rate_audit import SCENARIOS as STRESS_SCENARIOS
    from cardioagent.legacy.multimodal.policy.state import TASKS
    from cardioagent.legacy.multimodal.policy.value import BASELINES as V23_BASELINES

    sub = commands.add_parser("prepare-ecg-scroll", help="Import waveform-backed AF tasks, separating answers from inference")
    sub.add_argument("--source", type=Path, default=Path("D:/ECGagent/ECG/data/tasks/ecgscroll_lh.jsonl"))
    sub.add_argument("--raw-root", type=Path, default=Path("D:/ECGagent/ECG/data/raw"))
    sub.add_argument("--dataset", choices=("afdb", "ltafdb"), default="afdb")
    sub.add_argument("--max-records", type=int, default=6)
    sub.add_argument("--duration-s", type=float, default=1800., help="Fixed prefix duration; 0 imports full records")
    sub = commands.add_parser("score-ecg-intervals", help="Score final AF intervals without accessing a model")
    sub.add_argument("--references", type=Path, required=True)
    sub.add_argument("--predictions", type=Path, required=True)
    sub = commands.add_parser('track-afdb', help='Run evidence-grounded ECG event tracking and record checkpoints')
    sub.add_argument('--mode', choices=('live', 'cached', 'fixed'), required=True)
    sub.add_argument('--cache', type=Path, default=Path('runs/afdb-preprocessed-full-20261113'))
    sub.add_argument('--tasks', type=Path, default=Path('runs/afdb-full-review/20261113-233212-64035fa8/agent_tasks.jsonl'))
    sub.add_argument('--scores', type=Path, default=Path('runs/afdb-full-review/full-cpu/model_scores'))
    sub.add_argument('--baseline', type=Path, default=Path('runs/afdb-full-review/full-cpu'))
    sub.add_argument('--config', type=Path, default=Path('configs/ecg_event_v1.json'))
    sub.add_argument('--run-dir', type=Path, required=True, help='Stable directory; reuse to resume verified checkpoints')
    sub.add_argument('--workers', type=int, default=4)
    sub.add_argument('--device', type=normalize_ecg_device,
                     default=os.environ.get('CARDIOAGENT_ECG_DEVICE', 'cpu'))
    sub.add_argument('--limit-records', type=int, default=0)
    sub.add_argument('--duration-s', type=float, default=0)
    sub.add_argument('--threads', type=int, default=2)
    sub.add_argument('--channels', choices=('ECG1', 'ECG2'), nargs='+', default=['ECG1', 'ECG2'])
    sub = commands.add_parser("replay-afdb", help="Run ECGFounder on offline cache windows in arrival order")
    sub.add_argument("--cache", type=Path, required=True)
    sub.add_argument("--tasks", type=Path, required=True)
    sub.add_argument("--channel", choices=("ECG1", "ECG2"), default="ECG1")
    sub.add_argument("--threshold", type=float, required=True, help="Fixed, uncalibrated AF/AFL candidate threshold")
    sub.add_argument('--policy', choices=('threshold', 'two-window', 'dual', 'adaptive'), default='threshold')
    sub.add_argument('--review-budget', type=int, choices=range(1, 5), default=4,
                     help='Maximum extra calls per window for adaptive review')

    sub = commands.add_parser("list-ecg-tools", help="Export versioned tool catalog and JSON schemas")
    sub.add_argument("--experimental", action="store_true")
    sub = commands.add_parser("call-ecg-tool", help="Execute one registered ECG tool locally")
    sub.add_argument("tool")
    sub.add_argument("--record", type=Path, required=True)
    sub.add_argument("--lead", required=True)
    sub.add_argument("--start", type=float, default=0.)
    sub.add_argument("--end", type=float, required=True)
    sub.add_argument("--experimental", action="store_true")

    for command in ("ecg-agent", "audit-ecg-agent"):
        sub = commands.add_parser(command, help="ECG-only streaming measurement agent")
        sub.add_argument("--chunk-s", type=float, default=30.)
        sub.add_argument("--measurement-budget", type=int, default=2)
        if command == "ecg-agent":
            sub.add_argument("--record", type=Path, required=True, help="ECG-only WFDB record stem without .hea")
            sub.add_argument("--policy", choices=("primary", "quality", "verify"), default="verify")
        else:
            sub.add_argument("--source", type=Path, default=Path("data/external-events/mitdb"))
            sub.add_argument("--records", nargs="+", default=["100", "101", "103", "105", "108", "112"])

    sub = commands.add_parser("prepare-v24-sens", help="Cache SensSmartTech tool observations for V2.4")
    sub.add_argument("replay", type=Path)
    sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    sub.add_argument("--allow-published-pilot", action="store_true")
    sub.add_argument("--development-packet", type=Path)
    sub.add_argument("--test-packet", type=Path)
    sub = commands.add_parser("prepare-v24-bidmc", help="Cache BIDMC pulse-rate and external heart-rate windows")
    sub.add_argument("replay", type=Path)
    sub.add_argument("--source", type=Path, default=Path("data/bidmc-1.0.0"))
    sub.add_argument("--download", action="store_true", help="Download the open BIDMC WFDB files into --source")
    sub.add_argument("--paired-hr", action="store_true", help="Include HR references for all BIDMC splits")
    for command in ("audit-v24", "audit-v241"):
        sub = commands.add_parser(command, help="Train and compare conditional evidence value policies")
        sub.add_argument("--sens-replay", type=Path, default=Path("runs/v24-replay/senssmarttech.jsonl"))
        bidmc_name = "bidmc-paired.jsonl" if command == "audit-v241" else "bidmc.jsonl"
        sub.add_argument("--bidmc-replay", type=Path, default=Path("runs/v24-replay") / bidmc_name)
        sub.add_argument("--v23-checkpoint", type=Path)
        sub.add_argument("--epochs", type=int, choices=range(1, 101), default=16)
        sub.add_argument("--budget", type=int, choices=range(1, 7), default=6)
        if command == "audit-v241":
            sub.add_argument("--paired-bidmc", action="store_true", help="Train and test both tasks on matched BIDMC windows")
    sub = commands.add_parser("plot-v24", help="Plot V2.4 selective error at matched coverage")
    sub.add_argument("report", type=Path)
    sub.add_argument("--shared-risk", action="store_true", help="Use the common risk scorer from a V2.4.1 audit")
    sub = commands.add_parser("audit-v241-mimic", help="Evaluate V2.4.1 on local authorized MIMIC-III-Ext-PPG WFDB files")
    sub.add_argument("--source", type=Path, required=True)
    sub.add_argument("--checkpoint", type=Path, required=True)
    sub.add_argument("--max-subjects", type=int, default=100)
    sub.add_argument("--segments-per-subject", type=int, default=2)
    sub = commands.add_parser("audit-v24-pcg", help="Check PCG cycle estimates against CirCor S1 intervals")
    sub.add_argument("--source", type=Path, default=Path("data/external-events/circor"))
    sub = commands.add_parser("analyze-v24", help="Run a V2.4 policy on one synchronized record")
    sub.add_argument("record")
    sub.add_argument("--checkpoint", type=Path, required=True)
    sub.add_argument("--task", choices=TASKS, default="heart_rate")
    sub.add_argument("--method", default="conditional_voi")
    sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)

    sub = commands.add_parser("audit-v23", help="Train and evaluate the conditional value policy")
    sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    sub.add_argument("--allow-published-pilot", action="store_true", help="Explicitly use the ECG-derived surrogate reference")
    sub.add_argument("--development-packet", type=Path, help="Adjudicated training and validation windows")
    sub.add_argument("--test-packet", type=Path, help="Separate adjudicated test windows")
    sub.add_argument("--budget", type=int, choices=range(1, 7), default=6)
    sub.add_argument("--epochs", type=int, choices=range(1, 101), default=16)
    sub = commands.add_parser("plot-v23", help="Plot risk–coverage and replayed risk–cost tradeoffs")
    sub.add_argument("report", type=Path)
    sub = commands.add_parser("analyze-v23", help="Run task-conditioned evidence selection")
    sub.add_argument("record")
    sub.add_argument("--task", choices=TASKS, default="heart_rate")
    sub.add_argument("--method", choices=(*V23_BASELINES, "learned_voi_calibrated"), default="channel_quality")
    sub.add_argument("--budget", type=int, choices=range(1, 7), default=6)
    sub.add_argument("--checkpoint", type=Path)
    sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)

    sub = commands.add_parser("prepare-data", help="Import the full local SensSmartTech ZIP")
    sub.add_argument("archive", type=Path)
    sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    sub = commands.add_parser("events", help="Inspect synchronized waveforms and candidate events")
    sub.add_argument("record")
    sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    sub.add_argument("--start", type=float)
    sub.add_argument("--end", type=float)
    sub.add_argument("--all-channels", action="store_true")
    sub.add_argument("--calibration", type=Path, help="Optional independently measured latency JSON")
    sub.add_argument("--omit", choices=MODALITIES, action="append", default=[])

    sub = commands.add_parser("audit-incremental", help="Run the conditional modality value experiment")
    sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    sub.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    sub.add_argument("--matrix", choices=("conditional", "all"), default="conditional")
    sub.add_argument("--adjudicated-packet", type=Path, help="Use the manually reviewed R peaks and their exact windows")
    sub = commands.add_parser("plot-conditional", help="Plot risk, coverage and replay cost")
    sub.add_argument("report", type=Path)
    sub = commands.add_parser("analyze-ecg-first", help="Analyze a recording using conditional ECG-first acquisition")
    sub.add_argument("record")
    sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    sub.add_argument("--budget", type=int, default=6)
    sub.add_argument("--strategy", choices=(*STRATEGIES, *LEGACY_NAMES), default="ppg_fallback")
    sub.add_argument("--omit", choices=MODALITIES, action="append", default=[])
    sub = commands.add_parser("audit-agents", help="Compare conditional policies with shared observations and budgets")
    sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    sub.add_argument("--budget", dest="budgets", choices=(6, 8, 10), type=int, action="append")
    sub.add_argument("--protocol", choices=("active", "structured", "legacy"), default="active")
    sub.add_argument("--scenario", dest="scenarios", choices=STRESS_SCENARIOS, action="append")
    sub.add_argument("--adjudicated-packet", type=Path, help="Evaluate on reviewed test-window R peaks")
    sub.add_argument("--per-subject", type=int, help="Maximum records per test subject, alternating activity states")
    sub.add_argument("--workers", type=int, choices=range(1, 9), default=1,
                     help="Parallel records; match the number of local server slots")
    sub.add_argument("--resume", type=Path, help="Continue an earlier Agent audit run")
    sub.add_argument("--base-url", default=os.environ.get("CARDIOAGENT_BASE_URL", "http://127.0.0.1:8089/v1"))
    sub.add_argument("--model", default=os.environ.get("CARDIOAGENT_MODEL", "qwen3-4b"))
    sub = commands.add_parser("plot-agent-cost", help="Plot risk against actual calls and latency from a finished audit")
    sub.add_argument("report", type=Path)

    sub = commands.add_parser("prepare-event-review", help="Export blinded SensSmartTech annotation windows")
    sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    sub.add_argument("--cohort", choices=("development", "test"), default="development")
    sub.add_argument("--expanded", action="store_true", help="Include more activity records per subject")
    sub = commands.add_parser("review-event-annotations", help="Compare the two blinded reviews and list disagreements")
    sub.add_argument("packet", type=Path)
    sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    sub = commands.add_parser("score-event-review", help="Score candidates against adjudicated human labels")
    sub.add_argument("packet", type=Path)
    sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    sub = commands.add_parser("audit-ecg-events", help="Validate ECG timestamps against MIT-BIH beat annotations")
    sub.add_argument("--source", type=Path, default=Path("data/external-events/mitdb"))
    sub = commands.add_parser("audit-aux-events", help="Audit external PPG onset and PCG segmentation references")
    sub.add_argument("--source", type=Path, default=Path("data/external-events"))

    for command in ("benchmark", "audit-models"):
        sub = commands.add_parser(command)
        sub.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
        sub.add_argument("--split", choices=("train", "validation", "test", "all"), default="validation")
        sub.add_argument("--limit", type=int, help="Deterministic subset for smoke tests")
        if command == "benchmark":
            sub.add_argument("--mode", dest="modes", choices=MODES, action="append")
            sub.add_argument("--scenario", dest="scenarios", choices=SCENARIOS, action="append")
            sub.add_argument("--with-models", action="store_true", help="Enable pretrained tools for V1 modes")
            sub.add_argument("--base-url", default=os.environ.get("CARDIOAGENT_BASE_URL", "http://127.0.0.1:8089/v1"))
            sub.add_argument("--model", default=os.environ.get("CARDIOAGENT_MODEL", "qwen3-4b"))
    return parser.parse_args()


def model_client(args: argparse.Namespace, *, max_tokens: int = 768):
    if args.mode == "fixed":
        return None
    key = os.environ.get("CARDIOAGENT_API_KEY")
    from urllib.parse import urlparse

    if not key and urlparse(args.base_url).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Set CARDIOAGENT_API_KEY to use a non-local model endpoint")
    return ChatOpenAI(
        model=args.model, base_url=args.base_url, api_key=key or "local",
        temperature=0, max_tokens=max_tokens, timeout=90, max_retries=0,
        model_kwargs={"parallel_tool_calls": False},
    )


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_reference(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        next(stream, None)
        rows = csv.DictReader(stream)
        return {
            row["ECG"].removesuffix("_ecg"): float(row["Median heart rate (bpm)"])
            for row in rows if row.get("ECG") and row.get("Median heart rate (bpm)")
        }


def analyze(args, record_id, output, models, client, start=0, end=None) -> dict:
    recording = Recording.read(args.data, record_id, set(args.omit))
    analysis = Analysis(
        recording, start, recording.shared_duration if end is None else end,
        models, output, client, args.with_models,
    )
    output.mkdir(parents=True, exist_ok=False)
    report = analysis.run(args.mode)
    report["llm"] = args.model if client is not None else None
    save_json(output / "report.json", report)
    result = report["rate_summary"]
    print(f"{record_id}: {result['status']} | {result['rate_bpm']} bpm | {report['tool_calls']} tool calls")
    print(f"Report: {output.resolve() / 'report.json'}")
    return report


def main() -> None:
    args = arguments()
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    output = args.output / stamp
    try:
        if args.command == 'track-afdb':
            import subprocess
            import sys
            script = code_root() / 'scripts/run_ecg_events.py'
            subprocess.run([sys.executable, str(script), '--mode', args.mode, '--cache', str(args.cache),
                            '--tasks', str(args.tasks), '--scores', str(args.scores),
                            '--baseline', str(args.baseline), '--models', str(args.models),
                            '--config', str(args.config), '--output', str(args.run_dir),
                            '--workers', str(args.workers), '--threads', str(args.threads),
                            '--device', args.device, '--limit-records', str(args.limit_records),
                            '--duration-s', str(args.duration_s),
                            '--channels', *args.channels], check=True, cwd=script.parents[1])
            return
        if args.command == "prepare-ecg-scroll":
            from cardioagent.ecg_agent.evaluation.references import prepare
            prepare(args.source, args.raw_root, output, max_records=args.max_records,
                    duration_s=args.duration_s, dataset=args.dataset)
            print(f"ECG-Scroll task bundle: {output.resolve()}")
            return
        if args.command == "score-ecg-intervals":
            from cardioagent.ecg_agent.evaluation.legacy_intervals import score_files
            score_files(args.references, args.predictions, output)
            print(f"Interval scores: {output.resolve() / 'interval_scores.json'}")
            return
        if args.command == "replay-afdb":
            from cardioagent.legacy.ecg_measurement.replay import run_replay
            from cardioagent.ecg_agent.models.ecgfounder import cached_ecgfounder_scorer
            predict, info = cached_ecgfounder_scorer(args.models)
            report = run_replay(args.cache, args.tasks, output, args.channel, args.threshold, predict, info,
                                policy=args.policy, review_budget=args.review_budget)
            print(f"AFDB replay: {output.resolve()} | coverage={report['coverage']:.2%}")
            return
        if args.command in ("list-ecg-tools", "call-ecg-tool"):
            from cardioagent.ecg_agent.harness import build_registry, ToolContext, ToolSession
            registry = build_registry()
            if args.command == "list-ecg-tools":
                result = {"tools":registry.schemas(args.experimental),
                          "function_tools":registry.function_schemas(args.experimental)}
                name = 'ecg_tool_catalog.json'
            else:
                import math
                from cardioagent.legacy.ecg_measurement.evaluation import read_ecg
                from cardioagent.common.recording import Waveform
                if not (math.isfinite(args.start) and math.isfinite(args.end) and 0 <= args.start < args.end):
                    raise ValueError('Require finite 0 <= start < end')
                wave = read_ecg(args.record)
                if args.end > wave.duration:
                    raise ValueError('end exceeds ECG duration')
                # Provide a copied elapsed block, not the rest of the record.
                first = max(0, math.ceil(args.start*wave.fs-1e-8))
                last = min(len(wave.values), math.ceil(args.end*wave.fs-1e-8))
                block = Waveform(wave.values[first:last].copy(), wave.fs, wave.channels, wave.units)
                origin = first/wave.fs
                result = ToolSession(ToolContext(block,origin,origin+block.duration),
                                     allow_experimental=args.experimental).call(
                    args.tool, {"lead":args.lead,"start_s":origin,"end_s":origin+block.duration})
                name = 'ecg_tool_result.json'
            output.mkdir(parents=True, exist_ok=False)
            save_json(output / name, result)
            print(f"ECG tools: {output.resolve() / name}")
            if args.command == "call-ecg-tool" and result['status'] == 'error':
                raise SystemExit(2)
            return
        if args.command in ("ecg-agent", "audit-ecg-agent"):
            from cardioagent.ecg_agent import Config, ECGAgent
            from cardioagent.legacy.ecg_measurement.evaluation import audit, read_ecg
            config = Config(chunk_s=args.chunk_s, measurement_budget=args.measurement_budget,
                            policy=getattr(args, "policy", "verify"))
            if args.command == "audit-ecg-agent":
                audit(args.source, output, config, args.records)
                print(f"ECG Agent audit: {output.resolve() / 'ecg_agent_audit.json'}")
            else:
                result = ECGAgent(config).run(read_ecg(args.record))
                output.mkdir(parents=True, exist_ok=False)
                save_json(output / "ecg_agent_trace.json", result)
                print(f"ECG Agent trace: {output.resolve() / 'ecg_agent_trace.json'}")
            return
        if args.command == "prepare-v24-sens":
            from cardioagent.legacy.multimodal.policy.replay import capture_senssmarttech
            capture_senssmarttech(args.dataset_root, args.replay, pilot=args.allow_published_pilot,
                                  development=args.development_packet, test=args.test_packet)
            print(f"SensSmartTech replay: {args.replay.resolve()}")
            return
        if args.command == "prepare-v24-bidmc":
            from cardioagent.legacy.multimodal.policy.bidmc import capture, download
            if args.download:
                download(args.source)
            capture(args.source, args.replay, paired_hr=args.paired_hr)
            print(f"BIDMC replay: {args.replay.resolve()}")
            return
        if args.command in ("audit-v24", "audit-v241"):
            from cardioagent.legacy.multimodal.policy.audit_v24 import audit
            corrected = args.command == "audit-v241"
            audit(args.sens_replay, args.bidmc_replay, output, budget=args.budget, epochs=args.epochs,
                  v23_checkpoint=args.v23_checkpoint, corrected=corrected, paired_bidmc=getattr(args, "paired_bidmc", False))
            print(f"Policy audit: {output.resolve() / ('v241_audit.json' if corrected else 'v24_audit.json')}")
            return
        if args.command == "audit-v241-mimic":
            from cardioagent.legacy.multimodal.policy.mimic import audit
            if min(args.max_subjects, args.segments_per_subject) < 1:
                raise ValueError("Subject and segment limits must be positive")
            audit(args.source, args.checkpoint, output, args.max_subjects, args.segments_per_subject)
            print(f"MIMIC HR aggregate: {output.resolve() / 'mimic_hr_aggregate.json'}")
            return
        if args.command == "plot-v24":
            from cardioagent.legacy.multimodal.policy.validation_v24 import plot_matched
            plot_matched(args.report, shared=args.shared_risk)
            print(f"Matched-coverage figure saved under {args.report.parent.resolve()}")
            return
        if args.command == "audit-v24-pcg":
            from cardioagent.legacy.multimodal.policy.validation_v24 import audit_pcg
            audit_pcg(args.source, output)
            print(f"CirCor PCG audit: {output.resolve() / 'circor_pcg_period.json'}")
            return
        if args.command == "analyze-v24":
            from cardioagent.legacy.multimodal.policy.audit_v24 import analyze
            record = Recording.read(args.dataset_root / "WFDB", args.record)
            result = analyze(record, args.task, args.method, args.checkpoint)
            output.mkdir(parents=True, exist_ok=False)
            save_json(output / "v24_analysis.json", result)
            print(f"V2.4 analysis: {output.resolve() / 'v24_analysis.json'}")
            return
        if args.command == "audit-v23":
            from cardioagent.legacy.multimodal.policy.audit import run_audit
            run_audit(args.dataset_root, output, pilot=args.allow_published_pilot, development=args.development_packet,
                      test=args.test_packet, budget=args.budget, epochs=args.epochs)
            print(f"V2.3 audit: {output.resolve() / 'v23_audit.json'}")
            return
        if args.command == "analyze-v23":
            from cardioagent.legacy.multimodal.policy.audit import analyze as analyze_v23
            record = Recording.read(args.dataset_root / "WFDB", args.record)
            result = analyze_v23(record, args.task, args.budget, args.method, args.checkpoint)
            output.mkdir(parents=True, exist_ok=False)
            save_json(output / "v23_analysis.json", result)
            print(f"V2.3 analysis: {output.resolve() / 'v23_analysis.json'}")
            return
        if args.command == "plot-v23":
            from cardioagent.legacy.multimodal.policy.plots import plot_audit
            plot_audit(args.report)
            print(f"V2.3 figures: {args.report.parent.resolve()}")
            return
        if args.command == "audit-incremental":
            from cardioagent.legacy.multimodal.rate_audit import run_rate_audit
            run_rate_audit(args.dataset_root, output, split=args.split, packet=args.adjudicated_packet, matrix=args.matrix)
            print(f"Modality audit: {output.resolve() / 'incremental_value.json'}")
            return
        if args.command == "plot-conditional":
            from cardioagent.legacy.multimodal.rate_audit import plot_conditional
            plot_conditional(args.report)
            print(f"Conditional experiment figures: {args.report.parent.resolve()}")
            return
        if args.command == "analyze-ecg-first":
            from cardioagent.legacy.multimodal.rate_policy import ecg_first, observe_rate
            if args.budget < 1:
                raise ValueError("--budget must be positive")
            record = Recording.read(args.dataset_root / "WFDB", args.record, set(args.omit))
            result = ecg_first(record, args.budget, lambda m, ch: observe_rate(record, m, ch), strategy=args.strategy)
            report = {"record": record.describe(), "budget": args.budget, **result}
            output.mkdir(parents=True, exist_ok=False)
            save_json(output / "ecg_first.json", report)
            print(f"{args.record}: {result['status']} | {result['rate_bpm']} bpm | {result['tool_calls']} tool calls")
            print(f"ECG-first analysis: {output.resolve() / 'ecg_first.json'}")
            return
        if args.command == "audit-agents":
            from cardioagent.legacy.multimodal.agent_audit import run_agent_audit
            if args.per_subject is not None and args.per_subject < 1:
                raise ValueError("--per-subject must be positive")
            args.mode = "agent"
            output = args.resume if args.resume is not None else output
            results = run_agent_audit(args, output, model_client(args, max_tokens=64 if args.protocol != "legacy" else 220))
            print(f"Agent audit: {output.resolve() / 'agent_comparison.json'}")
            if any(m["failed_runs"] for result in results.values() for m in result["metrics"].values()):
                raise SystemExit("Some Agent runs were incomplete. Use --resume to retry failed records.")
            return
        if args.command == "plot-agent-cost":
            from cardioagent.legacy.multimodal.agent_audit import plot_actual_cost
            plot_actual_cost(args.report, args.report.parent)
            print(f"Risk and actual cost: {args.report.parent.resolve() / 'risk_actual_cost.png'}")
            return
        if args.command in ("prepare-event-review", "review-event-annotations", "score-event-review",
                            "audit-ecg-events", "audit-aux-events"):
            from cardioagent.legacy.multimodal.event_audit import audit_external_aux, audit_external_ecg, export_blind_packet, review_packet, score_packet
            if args.command == "prepare-event-review":
                export_blind_packet(args.dataset_root, output, args.cohort, expanded=args.expanded)
            elif args.command == "review-event-annotations":
                review_packet(args.dataset_root, args.packet, output)
            elif args.command == "score-event-review":
                score_packet(args.dataset_root, args.packet, output)
            elif args.command == "audit-ecg-events":
                audit_external_ecg(args.source, output)
            else:
                audit_external_aux(args.source, output)
            print(f"Event audit: {output.resolve()}")
            return
        if args.command in ("prepare-data", "events", "benchmark", "audit-models"):
            return run_v2_command(args, output)
        models = PretrainedModels(args.models)
        client = model_client(args)
        if args.command == "analyze":
            report = analyze(args, args.record, output, models, client, args.start, args.end)
            if report["agent_errors"]:
                raise SystemExit("Agent execution was incomplete. See agent_errors in the report.")
            return
        records = args.records or sorted(
            {p.stem.rsplit("_", 1)[0] for p in args.data.glob("*.hea")
             if p.stem.rsplit("_", 1)[-1] in MODALITIES}
        )
        if not records:
            raise ValueError("No WFDB recordings found")
        if len(records) != len(set(records)):
            raise ValueError("Provide each recording at most once")
        reference = load_reference(args.reference) if args.reference else {}
        rows = []
        for index, record_id in enumerate(records, start=1):
            report = analyze(args, record_id, output / f"record-{index:03d}", models, client)
            rate = report["rate_summary"]["rate_bpm"]
            ref = reference.get(record_id)
            rows.append({
                "record_id": record_id, "rate_bpm": rate, "reference_bpm": ref,
                "absolute_error_bpm": abs(rate - ref) if rate is not None and ref is not None else None,
                "status": report["rate_summary"]["status"],
                "tool_calls": report["tool_calls"], "duration_s": report["duration_s"],
                "agent_errors": report["agent_errors"],
            })
        errors = [row["absolute_error_bpm"] for row in rows if row["absolute_error_bpm"] is not None]
        summary = {
            "mode": args.mode, "num_records": len(rows), "num_scored": len(errors),
            "coverage": sum(row["rate_bpm"] is not None for row in rows) / len(rows),
            "mae_bpm": round(mean(errors), 3) if errors else None,
            "reference_source": str(args.reference) if args.reference else None,
            "records": rows,
        }
        save_json(output / "evaluation.json", summary)
        print(f"Evaluation: {output.resolve() / 'evaluation.json'}")
        if any(row["agent_errors"] for row in rows):
            raise SystemExit("Some Agent executions were incomplete. See agent_errors in the evaluation.")
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def run_v2_command(args, output: Path) -> None:
    from cardioagent.legacy.multimodal.dataset import load_manifest, prepare_dataset

    if args.command == "prepare-data":
        from zipfile import BadZipFile
        try:
            manifest = prepare_dataset(args.archive, args.dataset_root)
        except BadZipFile as exc:
            raise ValueError(f"Invalid dataset ZIP: {exc}") from exc
        print(f"Dataset: {args.dataset_root.resolve()} | subject split records: {manifest['record_counts']}")
        return
    if args.command == "events":
        from cardioagent.legacy.multimodal.v2 import inspect_recording
        manifest = load_manifest(args.dataset_root)
        if not any(r["record_id"] == args.record for r in manifest["records"]):
            raise ValueError(f"Record is not in the imported release: {args.record}")
        record = Recording.read(args.dataset_root / "WFDB", args.record, set(args.omit))
        calibration = json.loads(args.calibration.read_text(encoding="utf-8-sig")) if args.calibration else None
        if args.calibration and not isinstance(calibration, dict):
            raise ValueError("Calibration file must contain a JSON object")
        report = inspect_recording(record, documented_common_grid=True, start=args.start,
                                   end=args.end, all_channels=args.all_channels, calibration=calibration)
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / "events.json", report)
        print(f"Events: {output.resolve() / 'events.json'}")
        return
    models = PretrainedModels(args.models)
    if args.command == "audit-models":
        from cardioagent.legacy.multimodal.model_audit import run_audit
        run_audit(args, output, models)
        print(f"Model audit: {output.resolve() / 'model_audit.json'}")
        return
    from cardioagent.legacy.multimodal.benchmark import run_benchmark
    args.mode = "agent" if {"v1_single", "v1_multi"}.intersection(args.modes or []) else "fixed"
    payload = run_benchmark(args, output, models, model_client(args))
    print(f"Benchmark: {output.resolve() / 'benchmark.json'}")
    if any(row["agent_errors"] for row in payload["records"]):
        raise SystemExit("Some Agent runs were incomplete. See agent_errors in the benchmark.")


if __name__ == "__main__":
    main()

from cardioagent.common.paths import code_root
