"""Run the frozen AFDB policies with independent, memoized ECGFounder scores."""

import argparse
from collections import Counter
from hashlib import blake2b
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
import torch

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.evaluation.legacy_intervals import score_files
from cardioagent.ecg_agent.evaluation.intervals import iou
from cardioagent.ecg_agent.data.afdb import CHANNELS
from cardioagent.legacy.ecg_measurement.replay import run_replay
from cardioagent.common.io import read_jsonl
from cardioagent.legacy.multimodal.model_adapters import PretrainedModels


POLICIES = ('threshold', 'two-window', 'dual', 'adaptive')
THRESHOLD = 0.5


def score_cache(cache, models, output, device, batch_size):
    manifest_file = cache / 'manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    checkpoint = models / '1_lead_ECGFounder.pth'
    provenance = {'cache_manifest_sha256': digest(manifest_file), 'checkpoint_sha256': digest(checkpoint),
                  'network_sha256': digest(models / 'upstream/ecgfounder/net1d.py'),
                  'labels_sha256': digest(models / 'upstream/ecgfounder/tasks.txt'),
                  'torch_version': str(torch.__version__), 'device': device, 'batch_size': batch_size}
    output.mkdir(parents=True, exist_ok=True)
    provenance_file = output / 'provenance.json'
    if provenance_file.exists():
        if json.loads(provenance_file.read_text(encoding='utf-8')) != provenance:
            raise ValueError('Score cache provenance changed; use a new output directory')
    else:
        save_json(provenance_file, provenance)

    model = None
    for index, record in enumerate(manifest['records'], 1):
        name = record['record_id']
        path = output / f'{name}.npy'
        if path.exists():
            continue
        if model is None:
            model, labels, _ = PretrainedModels(models).load('ecg')
            if device == 'cuda':
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
            else:
                torch.set_num_threads(min(12, torch.get_num_threads() * 3))
            model = model.to(device)
            classes = [labels.index(x) for x in ('ATRIAL FIBRILLATION', 'ATRIAL FLUTTER')]
        data = np.load(cache / record['waveforms'], mmap_mode='r', allow_pickle=False)
        scores = np.empty((len(data), 2), dtype=np.float32)
        began = perf_counter()
        with torch.inference_mode():
            for start in range(0, len(data), batch_size):
                values = torch.from_numpy(data[start:start + batch_size].copy()).to(device)
                logits, _ = model(values[:, None, :])
                scores[start:start + len(values)] = logits[:, classes].sigmoid().cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError(f'{name}: non-finite model scores')
        temporary = path.with_suffix('.partial.npy')
        np.save(temporary, scores, allow_pickle=False)
        temporary.replace(path)
        print(f'Scored {index}/{len(manifest["records"])}: {name}, {len(scores)} windows, '
              f'{perf_counter() - began:.1f} s', flush=True)
    return manifest, provenance


def cached_predictor(cache, score_dir, records):
    """A callback keyed only by the copied waveform delivered to the tool."""
    scores = {}
    for record in records:
        name = record['record_id']
        data = np.load(cache / record['waveforms'], mmap_mode='r', allow_pickle=False)
        values = np.load(score_dir / f'{name}.npy', allow_pickle=False)
        if values.shape != (len(data), 2) or not np.isfinite(values).all():
            raise ValueError(f'{name}: score cache does not match waveform cache')
        for window, pair in zip(data, values, strict=True):
            key = blake2b(window, digest_size=16).digest()
            result = tuple(map(float, pair))
            previous = scores.setdefault(key, result)
            if not np.allclose(previous, result, rtol=0, atol=1e-5):
                raise ValueError('Identical input produced inconsistent scores')
            if any((a >= t) != (b >= t) for a, b in zip(previous, result) for t in (0.5, 0.6)):
                raise ValueError('Identical input straddles a policy threshold')

    def predict(window):
        return scores[blake2b(window, digest_size=16).digest()]

    return predict


def summarize(path, references, report, score):
    refs = read_jsonl(references, 'task_id')
    writes = {key: [] for key in refs}
    with (path / 'window_trace.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            cursor = row['cursor_s']
            if any(call['window_s'] != [row['start_s'], cursor] or call['available_at_s'] != cursor
                   for call in row.get('tool_calls', [])):
                raise ValueError('Tool trace violated the current-window boundary')
            update = row.get('event_update') or (row.get('memory') or {}).get('event_update')
            if update:
                if update['recorded_at_s'] != cursor or update['interval'][1] > cursor:
                    raise ValueError('Event write violated the current cursor')
                writes[row['task_id']].append(update)
    latencies = []
    durations = Counter()
    for task in report['tasks']:
        key = task['task_id']
        ref = refs[key]
        available = set(range(len(ref['intervals'])))
        for prediction in task['intervals']:
            candidates = [(iou(prediction, ref['intervals'][j]), j) for j in available]
            overlap, match = max(candidates, default=(0, -1))
            if overlap >= ref['tau']:
                available.remove(match)
        for j, event in enumerate(ref['intervals']):
            length = event[1] - event[0]
            group = '<30s' if length < 30 else '30–60s' if length < 60 else '>=60s'
            durations[group] += 1
            durations[f'{group}_matched'] += j not in available
            times = [w['recorded_at_s'] for w in writes[key]
                     if w['recorded_at_s'] >= event[0] and iou(w['interval'], event) >= ref['tau']]
            if times:
                latencies.append(min(times) - event[0])
    assert sum(value for key, value in durations.items() if key.endswith('_matched')) == score['tp']
    metrics = {key: score[key] for key in ('tp', 'fp', 'fn', 'micro_episode_f1', 'macro_episode_f1',
                                          'mean_burden_absolute_error_pp', 'false_events_per_recording_hour')}
    return metrics | {'channel': report['channel'], 'policy': report['policy'],
                      'coverage': report['coverage'], 'review_seconds': report.get('review_seconds'),
                      'model_calls': report.get('model_calls', round(report['scored_seconds'] / 10)),
                      'extra_tool_calls': report.get('review_calls', 0),
                      'reference_events_by_duration': dict(durations),
                      'latency_matched_writes': len(latencies),
                      'latency_mean_s': mean(latencies) if latencies else None,
                      'latency_median_s': median(latencies) if latencies else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--scores-only', action='store_true')
    parser.add_argument('--channel', choices=CHANNELS, action='append')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    score_dir = args.output / 'model_scores'
    manifest, model = score_cache(args.cache, args.models, score_dir, args.device, args.batch_size)
    if args.scores_only:
        return
    predict = cached_predictor(args.cache, score_dir, manifest['records'])
    results = []
    for channel in args.channel or CHANNELS:
        for policy in POLICIES:
            path = args.output / f'{channel.lower()}-{policy}'
            if path.exists():
                report = json.loads((path / 'replay_report.json').read_text(encoding='utf-8'))
                if (report['tasks_sha256'] != digest(args.tasks) or report['model'] != model
                        or report['channel'] != channel or report['policy'] != policy
                        or report['threshold'] != THRESHOLD):
                    raise ValueError(f'Replay provenance mismatch: {path}')
            else:
                report = run_replay(args.cache, args.tasks, path, channel, THRESHOLD, predict, model, policy=policy)
            score_path = path / 'evaluation' / 'interval_scores.json'
            if score_path.exists():
                score = json.loads(score_path.read_text(encoding='utf-8'))
            else:
                score = score_files(args.references, path / 'predictions.jsonl', path / 'evaluation')
            result = summarize(path, args.references, report, score)
            save_json(path / 'metrics.json', result)
            results.append(result)
            print(json.dumps(result), flush=True)
    save_json(args.output / 'comparison.json', {'records': len(manifest['records']), 'threshold': THRESHOLD,
              'compute': 'Independent batched scoring; replay exposes scores only on current-window calls. '
                         'Call counts are logical. Cached replay timings do not include neural model execution.',
              'latency': 'First cursor-timestamped write that overlaps the reference at the task IoU threshold.',
              'references_sha256': digest(args.references), 'runs': results})


if __name__ == '__main__':
    main()
