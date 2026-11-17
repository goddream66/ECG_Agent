"""Replay offline AFDB windows through a single-window prediction interface."""

from collections import Counter
from collections.abc import Callable, Iterator
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.data.preprocessing import PROFILE, TARGET_FS, WINDOW_SECONDS
from cardioagent.common.intervals import validate_intervals
from cardioagent.common.io import read_jsonl, write_jsonl


CHANNELS = ("ECG1", "ECG2")
SOURCE_FS = 250
Predict = Callable[[np.ndarray], tuple[float, float]]






def replay_task(cache: Path, record: dict, task: dict, channel: str, threshold: float,
                predict: Predict, emit: Callable[[dict], None]) -> dict:
    start_s, end_s = float(task["start_s"]), float(task["end_s"])
    validate_intervals([], start_s, end_s)
    intervals = []
    counts = Counter()
    scored_seconds = 0.0
    for row, values in arrivals(cache, record, channel, start_s, end_s):
        start = row["start_sample"] / SOURCE_FS
        cursor = row["end_sample"] / SOURCE_FS
        status = row["status"]
        counts[status] += 1
        evidence = {
            "task_id": task["task_id"], "channel": channel, "start_s": start,
            "end_s": cursor, "cursor_s": cursor, "status": status,
            "af_score": None, "afl_score": None, "candidate": None,
            "invalid_spans": row["invalid_spans"], "event_update": None,
        }
        if values is not None:
            if not np.isfinite(values).all():
                raise ValueError("Cached model input contains non-finite samples")
            before = perf_counter()
            af, afl = map(float, predict(values))
            if not (0 <= af <= 1 and 0 <= afl <= 1):
                raise ValueError("Predictor must return finite AF/AFL scores in [0,1]")
            candidate = max(af, afl) >= threshold
            evidence.update(af_score=af, afl_score=afl, candidate=candidate,
                            model_ms=(perf_counter() - before) * 1000)
            scored_seconds += cursor - start
            if candidate:
                if intervals and intervals[-1][1] == start:
                    intervals[-1][1] = cursor
                else:
                    intervals.append([start, cursor])
                evidence["event_update"] = {"interval": intervals[-1].copy(), "recorded_at_s": cursor}
        emit(evidence)
    return {"task_id": task["task_id"], "intervals": validate_intervals(intervals, start_s, end_s),
            "scored_seconds": scored_seconds, "unscored_seconds": end_s - start_s - scored_seconds,
            "window_status_counts": dict(counts)}


def run_replay(cache: Path, tasks: Path, output: Path, channel: str, threshold: float,
               predict: Predict, model_info: dict, *, policy: str = 'threshold', review_budget: int = 4) -> dict:
    from cardioagent.ecg_agent.policies.baselines import POLICIES
    from cardioagent.legacy.ecg_measurement.review_replay import replay_review

    if channel not in CHANNELS or not 0 < threshold <= 1:
        raise ValueError("Expected ECG1 or ECG2 and a threshold in (0, 1]")
    if policy not in ('threshold', *POLICIES) or type(review_budget) is not int or not 1 <= review_budget <= 4:
        raise ValueError('Invalid replay policy or review budget (1–4)')
    cache, tasks, output = map(Path, (cache, tasks, output))
    if output.exists():
        raise FileExistsError(output)
    manifest_file = cache / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("profile") != PROFILE
            or manifest.get("dataset") != "AFDB" or tuple(manifest["channels"]) != CHANNELS
            or manifest["window_s"] != WINDOW_SECONDS or manifest["stride_s"] != WINDOW_SECONDS
            or manifest["output_rate_hz"] != TARGET_FS):
        raise ValueError("Expected the offline AFDB 10-second cache")
    records = {row["record_id"]: row for row in manifest["records"]}
    public = read_jsonl(tasks, "task_id")
    for task in public.values():
        if (task.get("source_database") != "afdb" or task.get("record_id") not in records
                or {"answer", "intervals", "burden"} & task.keys()):
            raise ValueError("Replay requires an answer-free AFDB agent_tasks.jsonl and matching cache")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".afdb-replay-", dir=output.parent) as temporary:
        staging = Path(temporary) / "results"
        staging.mkdir()
        results = []
        with (staging / "window_trace.jsonl").open("w", encoding="utf-8", newline="\n") as trace:
            def emit(evidence):
                trace.write(json.dumps(evidence, allow_nan=False) + "\n")

            for task in public.values():
                args = (cache, records[task['record_id']], task, channel, threshold, predict, emit)
                if policy == 'threshold':
                    results.append(replay_task(*args))
                else:
                    results.append(replay_review(*args, Path(manifest['raw_root']), policy, review_budget))
        write_jsonl(staging / "predictions.jsonl", [
            {"task_id": row["task_id"], "intervals": row["intervals"]} for row in results
        ])
        scored = sum(row["scored_seconds"] for row in results)
        unscored = sum(row["unscored_seconds"] for row in results)
        report = {"cache_manifest_sha256": digest(manifest_file), "tasks_sha256": digest(tasks),
                  "profile": PROFILE, "channel": channel, "anatomical_lead": "unknown",
                  "threshold": threshold, "candidate_rule": "max(AF, AFL) >= threshold",
                  "clock": "simulated_record_seconds", "scored_seconds": scored,
                  "unscored_seconds": unscored, "coverage": scored / (scored + unscored),
                  "model": model_info, "policy": policy, "tasks": results}
        if policy != 'threshold':
            for key in ('tool_calls', 'model_calls', 'review_calls', 'tool_ms', 'review_seconds', 'elapsed_ms'):
                report[key] = sum(row[key] for row in results)
            report['review_budget_per_window'] = review_budget
            write_jsonl(staging / 'event_memory.jsonl', [
                {'task_id': row['task_id'], 'events': row['events']} for row in results
            ])
        save_json(staging / "replay_report.json", report)
        staging.rename(output)
    return report

from cardioagent.ecg_agent.data.afdb import CHANNELS, SOURCE_FS, TARGET_FS, _arrivals, arrivals
