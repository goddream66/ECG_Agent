"""Paired Qwen replay with and without previously measured shifted evidence."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter

from cardioagent.common.intervals import union_seconds
from cardioagent.common.io import digest, read_jsonl, save_json, write_jsonl
from cardioagent.ecg_agent.evaluation.metrics import intersection, paired_bootstrap
from cardioagent.ecg_agent.experiments.analysis import jsonlines
from cardioagent.ecg_agent.experiments.original_window_review import (
    QUALITY, SCORE, action_plan, evaluate, gap_windows, support_refs,
)
from cardioagent.ecg_agent.harness.shifted_evidence_review import PROMPT, ShiftedEvidenceReviewer
from cardioagent.ecg_agent.llm.providers import build_chat_model
from cardioagent.ecg_agent.memory.evidence import canonical_hash


METHODS = ('rule', 'qwen_original', 'qwen_shift')


def measurement_key(result):
    return (*result['window_s'], result['lead'], result['tool'])


def index_evidence(path):
    items = {}
    for result in jsonlines(path):
        key = (result['available_at_s'], *measurement_key(result))
        if key in items:
            raise ValueError(f'Duplicate original-window measurement: {key}')
        items[key] = result
    return items


def shifted_observations(packet, rows):
    windows = {(tuple(w['window_s']), lead): data['score']
               for w in gap_windows(packet) for lead, data in w['leads'].items()}
    items = []
    for row in rows:
        a, b = row['window_s']
        offset = row['offset_s']
        shifted = row['shifted_s']
        if (row['cursor_s'] != packet['cursor_s'] or row['gap_s'] != packet['gap_s']
                or offset not in (-5, 5) or shifted != [a + offset, b + offset]
                or shifted[1] > packet['cursor_s']
                or windows.get(((a, b), row['lead'])) != row['original']):
            raise ValueError('Shifted measurement does not match the arrived packet')
        items.append({key: row[key] for key in (
            'window_s', 'shifted_s', 'lead', 'offset_s', 'original', 'shifted', 'status', 'evidence_id'
        )})
        start, end = packet['gap_s']
        items[-1]['outside_gap_s'] = 10 - max(0, min(end, shifted[1]) - max(start, shifted[0]))
    return items


def supported_windows(packet, extra):
    return [w['window_s'] for w in gap_windows(packet) if support_refs(
        dict(packet, gap_s=w['window_s'], windows=[w]), extra)]


def available_actions(packet, extra, remaining):
    supported = {tuple(span) for span in supported_windows(packet, extra)}
    windows = {tuple(w['window_s']): w for w in gap_windows(packet)}
    valid = []
    for key in action_plan(windows.values(), packet['start_threshold']):
        a, b, lead, _ = key
        if (a, b) in supported or key in extra:
            continue
        item = windows[a, b]['leads'][lead]
        score = extra.get((a, b, lead, SCORE))
        quality = extra.get((a, b, lead, QUALITY))
        if score is not None:
            if score['status'] != 'ok' or max(score['result'][name] for name in ('af_score', 'afl_score')) < packet['start_threshold']:
                continue
        elif item['score_status'] != 'not_measured' and (
                item['score'] is None or max(item['score']) < packet['start_threshold']):
            continue
        if quality is not None:
            if quality['status'] != 'ok' or quality['result'].get('reasons') != []:
                continue
        elif item['quality'] not in ('not_measured', 'no_warning'):
            continue
        missing = int(score is None and 'score_id' not in item) + int(quality is None and 'quality_id' not in item)
        if missing <= remaining:
            valid.append((missing, a, b, lead, key[3] != SCORE, key))
    return [row[-1] for row in sorted(valid)]


def compact_window(window):
    return dict(window_s=window['window_s'], decision=window['decision'],
                leads={lead: {key: state.get(key) for key in ('score', 'score_status', 'quality', 'quality_reasons')}
                       for lead, state in window['leads'].items()})


def review(packet, evidence, shifts, reviewer, budget):
    extra = {}
    trace = []
    initial = supported_windows(packet, extra)
    stop = 'budget'
    began = perf_counter()
    for step in range(budget):
        if support_refs(packet, extra):
            stop = 'all_windows_supported'
            break
        actions = available_actions(packet, extra, budget - step)
        if not actions:
            stop = 'no_useful_action'
            break
        if reviewer is None:
            key = actions[0]
        else:
            state = dict(gap_s=packet['gap_s'], cursor_s=packet['cursor_s'],
                         threshold=packet['start_threshold'], remaining_budget=budget-step,
                         left=packet['left'], right=packet['right'],
                         windows=[compact_window(w) for w in packet['windows']],
                         supported_windows=supported_windows(packet, extra),
                         new_results=[dict(window_s=result['window_s'], lead=result['lead'],
                                           tool=result['tool'], status=result['status'],
                                           values={k: v for k, v in result['result'].items() if k in (
                                               'af_score', 'afl_score', 'reasons')}) for result in extra.values()])
            if shifts is not None:
                state['shifted_observations'] = [{key: value for key, value in item.items() if key != 'evidence_id'}
                                                 for item in shifts]
            key, attempt = reviewer.choose(state, actions)
            trace.append(attempt)
            if key is None:
                stop = 'model_stop' if attempt['status'] == 'ok' else 'model_error'
                break
        result = evidence[packet['cursor_s'], *key]
        extra[key] = result
        if reviewer is not None:
            trace[-1]['measured_evidence_id'] = result['evidence_id']
    support = support_refs(packet, extra)
    if support:
        stop = 'all_windows_supported'
    new = [span for span in supported_windows(packet, extra) if span not in initial]
    return dict(support_ids=support, newly_supported=new, calls=len(extra), forwards=0,
                stop=stop, steps=list(extra), evidence_ids=[item['evidence_id'] for item in extra.values()],
                llm=trace, wall_s=perf_counter()-began)


def infer_file(job, args, reviewer):
    _, _, _, path, count, _ = job
    decisions = []
    if count:
        measurements = index_evidence(args.original / 'evidence' / ('D-' + path.name))
        shifts_by_gap = {}
        for row in jsonlines(args.shifted / 'sensitivity' / path.name):
            shifts_by_gap.setdefault((tuple(row['gap_s']), row['cursor_s']), []).append(row)
        for packet in jsonlines(path):
            key = tuple(packet['gap_s']), packet['cursor_s']
            shifts = shifted_observations(packet, shifts_by_gap.get(key, []))
            base = support_refs(packet, {})
            if bool(base) != all(packet['gap_support']):
                raise ValueError('B0 differs from the frozen packet')
            row = dict(gap_s=packet['gap_s'], cursor_s=packet['cursor_s'], available_shifted=len(shifts),
                       shifted_evidence_ids=[x['evidence_id'] for x in shifts], B0={'support_ids': base})
            for method in METHODS:
                row[method] = review(packet, measurements, shifts if method == 'qwen_shift' else None,
                                     None if method == 'rule' else reviewer, args.budget)
            decisions.append(row)
    write_jsonl(args.output / 'decisions' / path.name, decisions)
    print(f'{path.stem}: {len(decisions)} paired candidates', flush=True)


def infer_worker(device, batch, args):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    reviewer = ShiftedEvidenceReviewer(build_chat_model(dict(
        provider='transformers', model_path=str(args.model_path), device='cuda:0', max_new_tokens=64)))
    for job in batch:
        infer_file(job, args, reviewer)


def analyze(jobs, args):
    refs = read_jsonl(args.references, 'task_id')
    validity = json.loads((args.source / 'evaluation_validity.json').read_text(encoding='utf-8'))['records']
    summaries, categories = evaluate(jobs, args.output, refs, validity, ('A', 'B0', *METHODS))
    diagnostics = Counter()
    per_record = {(row['channel'], row['method'], item['task_id']): item
                  for row in summaries for item in row['rows']}
    for channel, task, _, path, _, _ in jobs:
        key = task['task_id']
        for item in jsonlines(args.output / 'decisions' / path.name):
            plain, shifted = (item[name] for name in ('qwen_original', 'qwen_shift'))
            diagnostics[channel, 'paired_first_action_changes'] += bool(
                plain['llm'] and shifted['llm'] and plain['llm'][0]['action'] != shifted['llm'][0]['action'])
            diagnostics[channel, 'paired_first_measurement_changes'] += (
                plain['steps'][:1] != shifted['steps'][:1])
            diagnostics[channel, 'first_requests_with_shifted_evidence'] += bool(
                shifted['llm'] and shifted['llm'][0]['request']['shifted_observations'])
            for method in METHODS:
                result = item[method]
                record = per_record[channel, method, key]
                spans = intersection(result['newly_supported'], validity[key]['valid_intervals'])
                true = union_seconds(intersection(spans, refs[key]['intervals']))
                record['new_support_true_s'] = record.get('new_support_true_s', 0) + true
                record['new_support_non_af_s'] = record.get('new_support_non_af_s', 0) + union_seconds(spans) - true
                diagnostics[channel, method + '_calls'] += len(result['llm'])
                diagnostics[channel, method + '_errors'] += sum(x['status'] != 'ok' for x in result['llm'])
                diagnostics[channel, method + '_model_s'] += sum(x['wall_s'] for x in result['llm'])
                diagnostics[channel, method + '_prompt_tokens'] += sum(x['usage'].get('prompt_tokens', 0) for x in result['llm'])
    for row in summaries:
        for metric in ('new_support_true_s', 'new_support_non_af_s'):
            row['metrics'][metric] = sum(item.get(metric, 0) for item in row['rows'])
    comparison = {(row['channel'], row['method']): row for row in summaries}
    paired = {ch: paired_bootstrap(comparison[ch, 'qwen_shift']['rows'], comparison[ch, 'qwen_original']['rows'])
              for ch in ('ECG1', 'ECG2')}
    return summaries, categories, paired, {ch: {k: v for (lead, k), v in diagnostics.items() if lead == ch}
                                           for ch in ('ECG1', 'ECG2')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'original', 'shifted', 'references', 'model-path', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument('--records', nargs='+')
    parser.add_argument('--budget', type=int, choices=(2, 6), default=2)
    args = parser.parse_args()
    if len(set(args.devices)) != len(args.devices) or any(device < 0 for device in args.devices):
        parser.error('GPU IDs must be unique nonnegative integers')
    tasks = read_jsonl(args.source / 'tasks.jsonl', 'task_id')
    if args.records:
        tasks = {key: task for key, task in tasks.items() if task['record_id'] in args.records}
        if {task['record_id'] for task in tasks.values()} != set(args.records):
            parser.error('Requested records are absent')
    if any({'answer', 'intervals', 'burden'} & task.keys() or task['source_database'] != 'afdb'
           for task in tasks.values()):
        parser.error('Expected unlabeled AFDB tasks')
    began = perf_counter()
    source_hash = digest(args.source / 'comparison.json')
    for root in (args.original, args.shifted):
        manifest = json.loads((root / 'inputs.json').read_text(encoding='utf-8'))
        if manifest['source_sha256'] != source_hash:
            raise ValueError(f'Supplementary evidence comes from another baseline: {root}')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'decisions').mkdir()
    jobs = []
    inputs = {}
    for channel in ('ECG1', 'ECG2'):
        for task in tasks.values():
            name = f'{channel}-{task["record_id"]}.jsonl'
            path = args.original / 'packets' / name
            shifted_path = args.shifted / 'packets' / name
            inputs[name] = {'packet_sha256': digest(path)}
            if digest(shifted_path) != inputs[name]['packet_sha256']:
                raise ValueError(f'Frozen candidates differ: {name}')
            packets = list(jsonlines(path))
            jobs.append((channel, task, args.source / 'records' / f'{channel.lower()}-quality-as-hint' /
                         task['record_id'], path, len(packets), sum(all(p['gap_support']) for p in packets)))
            if packets:
                inputs[name]['original_evidence_sha256'] = digest(args.original / 'evidence' / ('D-' + name))
                inputs[name]['shifted_evidence_sha256'] = digest(args.shifted / 'sensitivity' / name)
    model = sorted(path for path in args.model_path.iterdir()
                   if path.is_file() and path.suffix in ('.safetensors', '.json'))
    save_json(args.output / 'inputs.json', dict(
        inputs=inputs, source_sha256=source_hash,
        experiment_sha256=digest(Path(__file__)), prompt_sha256=digest(PROMPT),
        original_manifest_sha256=digest(args.original / 'inputs.json'),
        shifted_manifest_sha256=digest(args.shifted / 'inputs.json'),
        budget=args.budget, devices=args.devices, model_sha256={p.name: digest(p) for p in model},
        acquisition='fixed causal measurement replay; shifted observations are preprovided and their cost reported separately'))
    batches = [[] for _ in args.devices]
    for job in sorted(jobs, key=lambda item: -item[4]):
        min(batches, key=lambda batch: sum(item[4] for item in batch)).append(job)
    with ProcessPoolExecutor(max_workers=len(batches)) as pool:
        work = [pool.submit(infer_worker, device, batch, args) for device, batch in zip(args.devices, batches)]
        for future in work:
            future.result()
    summaries, categories, paired, diagnostics = analyze(jobs, args)
    save_json(args.output / 'comparison.json', dict(results=summaries, categories=categories,
              paired=paired, diagnostics=diagnostics, budget=args.budget,
              references_sha256=digest(args.references), wall_s=perf_counter()-began))
    lines = ['# Qwen shifted evidence ablation', '',
             f'{sum(job[4] for job in jobs)} frozen candidates. {args.budget} original measurements per pair.',
             'Previously measured shifted scores are provided only to qwen_shift; no new ECGFounder forwards.', '',
             '| Lead | Method | F1 | Gaps merged | Newly supported true s* | Newly supported non-AF s* | Calls |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        m = row['metrics']
        lines.append(f"| {row['channel']} | {row['method']} | {m['micro_episode_f1']:.4f} | {m['merges']} | "
                     f"{m['new_support_true_s']:.2f} | {m['new_support_non_af_s']:.2f} | {m['extra_tool_calls']} |")
    lines.extend(['', '* Intermediate original-window support, independent of final merge.', '',
                  'Shifted evidence availability and runtime are reported separately from the original-measurement budget.'])
    (args.output / 'COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(diagnostics, flush=True)


if __name__ == '__main__':
    main()
