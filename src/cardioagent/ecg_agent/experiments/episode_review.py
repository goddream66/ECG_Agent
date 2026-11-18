"""Evaluate causal event-pair review on a frozen quality-as-hint replay."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
from time import perf_counter

from cardioagent.common.intervals import union_seconds
from cardioagent.common.io import digest, read_jsonl, save_json, write_jsonl
from cardioagent.ecg_agent.evaluation.metrics import aggregate, evaluate_record, intersection, paired_bootstrap
from cardioagent.ecg_agent.experiments.analysis import jsonlines, trace_data
from cardioagent.ecg_agent.harness.episode_review import EpisodeReviewer, PROMPT
from cardioagent.ecg_agent.llm.providers import build_chat_model
from cardioagent.ecg_agent.memory.episode_review import review_packets


METHODS = ('original', 'fixed_gap', 'evidence_rule', 'qwen')


def collect(source, tasks, output, channels):
    for channel in channels:
        for task in tasks.values():
            record = task['record_id']
            folder = source / 'records' / f'{channel.lower()}-quality-as-hint' / record
            result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
            if (result['task_id'] != task['task_id'] or result['channel'] != channel
                    or result['experiment_method'] != 'quality-as-hint'):
                raise ValueError(f'Wrong source replay: {folder}')
            packets = list(review_packets(jsonlines(folder / 'window_trace.jsonl'),
                           jsonlines(folder / 'event_updates.jsonl'), jsonlines(folder / 'evidence.jsonl'),
                           result['events'], start_threshold=result['config']['threshold']))
            path = output / 'packets' / f'{channel}-{record}.jsonl'
            write_jsonl(path, packets)
            yield (channel, task, folder, path, len(packets), sum(all(p['gap_support']) for p in packets))


def infer_batch(device, jobs, model_path, output):
    if not any(job[4] for job in jobs):
        return
    reviewer = EpisodeReviewer(build_chat_model({'provider': 'transformers', 'model_path': str(model_path),
                               'device': device, 'max_new_tokens': 256}))
    for channel, task, _, path, _, _ in jobs:
        decisions = []
        for packet in jsonlines(path):
            result = reviewer.review(packet)
            decisions.append(dict(result, gap_s=packet['gap_s'], cursor_s=packet['cursor_s']))
        write_jsonl(output / 'decisions' / path.name, decisions)
        print(f'{device} {channel} {task["record_id"]}: {len(decisions)} reviews', flush=True)


def merge_gaps(intervals, selected_gaps):
    """Preserve original interval order and revisions of only adjacent pairs."""
    if not intervals:
        return []
    predicted = [list(intervals[0])]
    used = set()
    for start, end in intervals[1:]:
        gap = (predicted[-1][1], start)
        if gap in selected_gaps:
            used.add(gap)
            predicted[-1][1] = end
        else:
            predicted.append([start, end])
    if used != selected_gaps:
        raise ValueError('Selected gap does not join adjacent original intervals')
    return predicted


def evaluate_jobs(jobs, output, references, validity):
    rows = {}
    diagnostics = Counter()
    for channel, task, folder, path, count, eligible in jobs:
        key = task['task_id']
        decisions = list(jsonlines(output / 'decisions' / path.name)) if count else []
        packets = list(jsonlines(path))
        if len(decisions) != count:
            raise ValueError('Review count differs from the frozen candidate count')
        gaps = {name: set() for name in METHODS}
        for packet, decision in zip(packets, decisions):
            gap = tuple(packet['gap_s'])
            if (decision['gap_s'] != packet['gap_s'] or decision['cursor_s'] != packet['cursor_s']):
                raise ValueError('Review does not match the frozen evidence packet')
            gaps['fixed_gap'].add(gap)
            if all(packet['gap_support']):
                gaps['evidence_rule'].add(gap)
            if decision['action'] == 'merge' and decision['status'] == 'ok':
                gaps['qwen'].add(gap)
            diagnostics['calls'] += 1
            diagnostics['model_wall_s'] += decision['wall_s']
            diagnostics[decision['status']] += 1
            diagnostics['action_' + decision['action']] += 1
            diagnostics['missing_evidence_requests'] += bool(decision.get('proposal', {}).get('missing_evidence'))
            diagnostics.update(decision['usage'])
        diagnostics['evidence_eligible'] += eligible

        result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        _, unknown, _ = trace_data(jsonlines(folder / 'window_trace.jsonl'))
        updates = list(jsonlines(folder / 'event_updates.jsonl'))
        for name, selected in gaps.items():
            prediction = merge_gaps(result['intervals'], selected)
            valid = validity[key]['valid_intervals']
            row = evaluate_record(key, prediction, references[key], valid=valid, unknown=unknown,
                                  updates=updates, events=result['events'])
            added = intersection(selected, valid)
            row.update(record_id=task['record_id'], reviewed_pairs=count, accepted_merges=len(selected),
                       added_prediction_s=union_seconds(added), added_non_af_s=union_seconds(added) -
                       union_seconds(intersection(added, references[key]['intervals'])))
            rows.setdefault((channel, name), []).append(row)
    summaries = []
    for (channel, method), records in rows.items():
        metrics = aggregate(records)
        for key in ('accepted_merges', 'added_prediction_s', 'added_non_af_s', 'merged_predictions'):
            metrics[key] = sum(row[key] for row in records)
        summaries.append({'channel': channel, 'method': method, 'metrics': metrics, 'rows': records})
    paired = {channel: paired_bootstrap(rows[channel, 'qwen'], rows[channel, 'evidence_rule'])
              for channel, method in rows if method == 'qwen'}
    signal_hours = sum(row['duration_s'] for (channel, method), records in rows.items()
                       if method == 'original' for row in records) / 3600
    diagnostics['calls_per_channel_hour'] = diagnostics['calls'] / signal_hours
    return summaries, paired, dict(diagnostics)


def report(output, summaries):
    lines = ['# AFDB event review comparison', '',
             'Exploratory fixed-evidence replay. Original alarm times are preserved.', '',
             '| Lead | Method | Event F1 | Time recall | Non-AF min | Excess fragments | Merged predictions | Accepted gaps | Added non-AF s |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        m = row['metrics']
        recall = 'NA' if m['time_recall'] is None else f"{m['time_recall']:.4f}"
        lines.append(f"| {row['channel']} | {row['method']} | {m['micro_episode_f1']:.4f} | {recall} | "
                     f"{m['non_af_alarm_minutes']:.3f} | {m['long_fragment_excess']} | {m['merged_predictions']} | "
                     f"{m['accepted_merges']} | {m['added_non_af_s']:.2f} |")
    (output / 'COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--records', nargs='+')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if len(set(args.devices)) != len(args.devices):
        parser.error('Assign each worker a separate device')
    if not args.prepare_only and (args.references is None or args.model_path is None):
        parser.error('--references and --model-path are required for evaluation')
    tasks = read_jsonl(args.source / 'tasks.jsonl', 'task_id')
    if args.records:
        tasks = {key: row for key, row in tasks.items() if row['record_id'] in args.records}
        if {row['record_id'] for row in tasks.values()} != set(args.records):
            raise ValueError('Requested records are absent from the source replay')
    if any({'answer', 'intervals', 'burden'} & task.keys() for task in tasks.values()):
        raise ValueError('The review task file must be unlabeled')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'packets').mkdir()
    started = perf_counter()
    jobs = list(collect(args.source, tasks, args.output, ('ECG1', 'ECG2')))
    counts = {'pairs': sum(job[4] for job in jobs), 'evidence_eligible': sum(job[5] for job in jobs)}
    save_json(args.output / 'candidate_counts.json', counts)
    print(counts, flush=True)
    if args.prepare_only:
        return
    model_files = sorted(file for file in args.model_path.iterdir()
                         if file.is_file() and file.suffix in ('.json', '.safetensors'))
    save_json(args.output / 'inputs.json', {
        'source': str(args.source.resolve()), 'source_comparison_sha256': digest(args.source / 'comparison.json'),
        'packets_sha256': {job[3].name: digest(job[3]) for job in jobs}, 'prompt_sha256': digest(PROMPT),
        'model_path': str(args.model_path.resolve()), 'model_sha256': {p.name: digest(p) for p in model_files},
        'devices': args.devices, 'max_new_tokens': 256})
    (args.output / 'decisions').mkdir()
    batches = [jobs[index::len(args.devices)] for index in range(len(args.devices))]
    with ProcessPoolExecutor(max_workers=len(batches)) as pool:
        list(pool.map(partial(infer_batch, model_path=args.model_path, output=args.output), args.devices, batches))
    references = read_jsonl(args.references, 'task_id')
    validity = json.loads((args.source / 'evaluation_validity.json').read_text(encoding='utf-8'))['records']
    summaries, paired, diagnostics = evaluate_jobs(jobs, args.output, references, validity)
    save_json(args.output / 'comparison.json', dict(results=summaries, paired_vs_evidence_rule=paired,
              diagnostics=diagnostics, references_sha256=digest(args.references), wall_s=perf_counter() - started))
    report(args.output, summaries)


if __name__ == '__main__':
    main()
