"""Diagnose gap repair and compare three Qwen memory inputs on frozen AFDB evidence."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from cardioagent.common.io import digest, read_jsonl, save_json, write_jsonl
from cardioagent.ecg_agent.evaluation.metrics import ratio
from cardioagent.ecg_agent.experiments.analysis import jsonlines
from cardioagent.ecg_agent.experiments.original_window_review import evaluate
from cardioagent.ecg_agent.experiments.qwen_gap_relationship import (
    inputs, reference_relation, revisions,
)
from cardioagent.ecg_agent.harness.gap_relationship import GapRelationshipReviewer
from cardioagent.ecg_agent.llm.providers import build_chat_model


ARMS = ('summary', 'windows', 'both')
METHODS = (*ARMS, 'rule')
ORDERINGS = tuple(permutations(ARMS))
PROMPT = Path(__file__).resolve().parents[1] / 'prompts/gap_memory_v1.txt'


def case_key(row):
    return tuple(row['gap_s']), row['cursor_s']


def automatic_support(relation, state, refs):
    chosen = {'relation': relation, 'support': [item['id'] for item in state['observations']]}
    return revisions(chosen, state, refs)


def coverage_blockers(state, cover):
    """Explain each uncovered subinterval under the published full-gap acceptance rule."""
    a, b = state['gap_s']
    inside = [item for item in state['observations'] if a <= item['window_s'][0]
              and item['window_s'][1] <= b]
    edges = sorted({a, b, *(edge for item in inside for edge in item['window_s'])})
    rows = []
    for left, right in zip(edges, edges[1:]):
        if any(x <= left and right <= y for x, y in cover):
            continue
        containing = [item for item in inside if item['window_s'][0] <= left
                      and right <= item['window_s'][1]]
        scored = [item for item in containing if item['score'] is not None]
        positive = [item for item in scored if max(item['score']) >= state['threshold']]
        if positive:
            reasons = sorted({'positive_quality_' + item['quality'] for item in positive})
        elif scored:
            reasons = ['scores_below_threshold']
        elif containing:
            reasons = ['no_score']
        else:
            reasons = ['no_window']
        rows.append({'interval_s': [left, right], 'reasons': reasons})
    return rows


def memory_inputs(packet):
    state, _, refs = inputs(packet, [])
    events = [dict(state[name], id=f'E_{name.upper()}') for name in ('left', 'right')]
    thresholds = {tuple(win['window_s']): win['threshold'] for win in packet['windows']}
    observations = [dict(item, decision_threshold=thresholds[tuple(item['window_s'])])
                    for item in state['observations']]
    common = {key: state[key] for key in ('cursor_s', 'gap_s', 'threshold')}
    return {
        'summary': dict(common, events=events, observations=[]),
        'windows': dict(common, events=[], observations=observations),
        'both': dict(common, events=events, observations=observations),
    }, refs


def diagnostic(jobs, previous, refs, validity, out):
    cases = []
    totals = {}
    for lead, task, _, packet_path, _, _ in jobs:
        key = task['task_id']
        logged = previous / 'decisions' / packet_path.name
        for packet, old in zip(jsonlines(packet_path), jsonlines(logged), strict=True):
            if case_key(packet) != case_key(old):
                raise ValueError(f'Committed packet and Qwen log mismatch: {packet_path}')
            reference = reference_relation(
                packet['gap_s'], packet['left']['interval'], packet['right']['interval'],
                refs[key]['intervals'], validity[key]['valid_intervals'],
            )
            row = dict(lead=lead, record=task['record_id'], gap_s=packet['gap_s'],
                       cursor_s=packet['cursor_s'], reference=reference)
            for arm in ('original', 'shifted'):
                reply = old[arm]
                state, aliases = reply['request'], reply['evidence_index']
                if case_key(state) != case_key(packet):
                    raise ValueError('Logged Qwen request does not match the candidate')
                ceiling = automatic_support('continuous', state, aliases)
                bypass_refs = automatic_support(reply['relation'], state, aliases)
                actual = revisions(reply, state, aliases)
                if actual['support_ids'] != reply['support_ids']:
                    raise ValueError('Previously reported Qwen evidence cannot be reproduced')
                enough = bool(ceiling['support_ids'])
                if arm == 'original' and enough != all(packet['gap_support']):
                    raise ValueError('Automatic original coverage differs from frozen rule')
                blockers = coverage_blockers(state, ceiling['validated_support_s'])
                counts = totals.setdefault((lead, arm), Counter())
                counts['candidates'] += 1
                counts['reference_' + reference] += 1
                if enough:
                    counts['automatically_eligible_' + reference] += 1
                if bypass_refs['support_ids']:
                    counts['qwen_with_auto_refs_' + reference] += 1
                if actual['support_ids']:
                    counts['qwen_with_own_refs_' + reference] += 1
                if reference == 'continuous':
                    if not enough:
                        counts['blocked_by_evidence'] += 1
                        for reason in {r for block in blockers for r in block['reasons']}:
                            counts['blocker_' + reason] += 1
                    elif reply['relation'] != 'continuous':
                        counts['blocked_by_qwen_judgment'] += 1
                    elif not actual['support_ids']:
                        counts['blocked_by_qwen_citations'] += 1
                    else:
                        counts['successful_qwen_merge'] += 1
                row[arm] = dict(eligible=enough, qwen_relation=reply['relation'],
                                auto_refs_merge=bool(bypass_refs['support_ids']),
                                qwen_refs_merge=bool(actual['support_ids']),
                                eligible_support_s=ceiling['validated_support_s'], blockers=blockers)
            cases.append(row)
    write_jsonl(out / 'diagnostic_cases.jsonl', cases)
    result = {lead: {arm: dict(totals[lead, arm]) for arm in ('original', 'shifted')}
              for lead in ('ECG1', 'ECG2')}
    save_json(out / 'diagnostic.json', result)
    return cases, result


def worker(device, jobs, model_path, out):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    reviewer = GapRelationshipReviewer(build_chat_model(dict(
        provider='transformers', model_path=str(model_path), device='cuda:0',
        max_new_tokens=384,
    )), prompt=PROMPT)
    for _, _, _, path, count, offset in jobs:
        decisions = []
        for index, packet in enumerate(jsonlines(path)):
            states, refs = memory_inputs(packet)
            row = dict(gap_s=packet['gap_s'], cursor_s=packet['cursor_s'])
            for arm in ORDERINGS[(offset + index) % len(ORDERINGS)]:
                state = states[arm]
                result = reviewer.judge(state)
                row[arm] = dict(result, request=state, **revisions(result, state, refs))
            rule = ('continuous' if all(packet['gap_support']) else
                    'separate' if packet['sustained_opposition'] else 'uncertain')
            row['rule'] = dict(relation=rule, status='ok', usage={}, wall_s=0,
                               **automatic_support(rule, states['both'], refs))
            decisions.append(row)
        if count != len(decisions):
            raise ValueError(f'Wrong candidate count for {path}')
        write_jsonl(out / 'decisions' / path.name, decisions)
        print(f'{path.stem}: {count} three-arm decisions', flush=True)


def relation_summary(jobs, out, cases):
    labels = {(row['lead'], row['record'], *case_key(row)): row['reference'] for row in cases}
    records = {}
    paired_rows = []
    for lead, task, _, path, _, _ in jobs:
        record = task['record_id']
        counts = records.setdefault((lead, record), Counter())
        for row in jsonlines(out / 'decisions' / path.name):
            ref = labels.pop((lead, record, *case_key(row)))
            paired_rows.append(dict(lead=lead, record=record, reference=ref,
                                    gap_s=row['gap_s'], cursor_s=row['cursor_s'],
                                    **{m: row[m]['relation'] for m in METHODS}))
            for method in METHODS:
                d = row[method]
                counts[method, ref, d['relation']] += 1
                counts[method, 'errors'] += d['status'] != 'ok'
                counts[method, 'prompt_tokens'] += d['usage'].get('prompt_tokens', 0)
                counts[method, 'model_seconds'] += d['wall_s']
    if labels:
        raise ValueError('Missing model judgments for frozen candidates')
    write_jsonl(out / 'memory_pair_labels.jsonl', paired_rows)

    def stats(c, method):
        positive = sum(c[method, 'continuous', clinical] for clinical in ('continuous', 'separate', 'uncertain'))
        negative = sum(c[method, 'separate', clinical] for clinical in ('continuous', 'separate', 'uncertain'))
        tp, fp = c[method, 'continuous', 'continuous'], c[method, 'separate', 'continuous']
        tn, fn = c[method, 'separate', 'separate'], c[method, 'continuous', 'separate']
        total = positive + negative
        return dict(positive=positive, negative=negative, tp=tp, fp=fp, tn=tn, fn=fn,
                    uncertain=total - (tp + fp + tn + fn),
                    accuracy=ratio(tp + tn, total), coverage=ratio(tp + fp + tn + fn, total),
                    continuity_recall=ratio(tp, positive), false_link_rate=ratio(fp, negative),
                    errors=c[method, 'errors'], prompt_tokens=c[method, 'prompt_tokens'],
                    model_seconds=c[method, 'model_seconds'])

    report = {}
    for lead in ('ECG1', 'ECG2'):
        subset = [c for (ch, _), c in records.items() if ch == lead]
        report[lead] = {method: stats(sum(subset, Counter()), method) for method in METHODS}
        rng = np.random.default_rng(20261114)
        deltas = {arm: {metric: [] for metric in ('accuracy', 'coverage', 'continuity_recall', 'false_link_rate')}
                  for arm in ('summary', 'windows', 'rule')}
        for sample in rng.integers(0, len(subset), size=(2000, len(subset))):
            c = sum((subset[i] for i in sample), Counter())
            full = stats(c, 'both')
            for arm, metrics in deltas.items():
                other = stats(c, arm)
                for metric, samples in metrics.items():
                    if full[metric] is not None and other[metric] is not None:
                        samples.append(full[metric] - other[metric])
        report[lead]['both_minus_ci95'] = {
            arm: {metric: np.quantile(samples, [.025, .975]).tolist() if samples else None
                  for metric, samples in metrics.items()}
            for arm, metrics in deltas.items()
        }
    return report


def write_report(out, diagnostics, relation=None, intervals=None):
    lines = ['# AFDB gap repair diagnostics and memory ablation', '',
             '| Lead | Evidence | True continuity | Evidence can cover | Blocked by evidence | '
             'Blocked by judgment | Blocked by citations | Qwen merges on true continuity |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for lead, report in diagnostics.items():
        for arm, d in report.items():
            counts = [d.get(k, 0) for k in (
                'reference_continuous', 'automatically_eligible_continuous', 'blocked_by_evidence',
                'blocked_by_qwen_judgment', 'blocked_by_qwen_citations', 'successful_qwen_merge',
            )]
            lines.append(f'| {lead} | {arm} | ' + ' | '.join(map(str, counts)) + ' |')
    if relation:
        lines += ['', '| Lead | Memory | True continuity | False continuity | True separation | '
                  'False separation | Uncertain | Errors |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for lead, report in relation.items():
            for arm in METHODS:
                d = report[arm]
                counts = [d[k] for k in ('tp', 'fp', 'tn', 'fn', 'uncertain', 'errors')]
                lines.append(f'| {lead} | {arm} | ' + ' | '.join(map(str, counts)) + ' |')
        lines += ['', '| Lead | Method | Merges | Added AF/AFL s | Added non-AF/AFL s | '
                  'Event F1 | Burden MAE (pp) |', '|---|---|---:|---:|---:|---:|---:|']
        for row in intervals:
            m = row['metrics']
            lines.append(f"| {row['channel']} | {row['method']} | {m['merges']} | {m['added_true_s']:.3f} | "
                         f"{m['added_non_af_s']:.3f} | {m['micro_episode_f1']:.6f} | "
                         f"{m['valid_time_burden_mae_pp']:.6f} |")
    (out / 'COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'shifted', 'previous', 'references', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument('--diagnostic-only', action='store_true')
    args = parser.parse_args()
    if not args.diagnostic_only and not args.model_path:
        parser.error('--model-path is required for Qwen inference')
    if len(set(args.devices)) != len(args.devices) or min(args.devices) < 0:
        parser.error('GPU indices must be distinct and nonnegative')

    now = perf_counter()
    path = args.shifted / 'packets'
    previous = json.loads((args.previous / 'inputs.json').read_text(encoding='utf-8'))
    source_hash = digest(args.source / 'comparison.json')
    previous_summary = json.loads((args.previous / 'comparison.json').read_text(encoding='utf-8'))
    ref_hash = digest(args.references)
    if previous['source_sha256'] != source_hash or previous_summary['references_sha256'] != ref_hash:
        raise ValueError('Source or independent reference differs from previous Qwen run')
    tasks = read_jsonl(args.source / 'tasks.jsonl', 'task_id')
    jobs, hashes = [], {}
    offset = 0
    for lead in ('ECG1', 'ECG2'):
        for task in tasks.values():
            record = task['record_id']
            p = path / f'{lead}-{record}.jsonl'
            hashes[p.name] = dict(packet=digest(p),
                                  previous_decisions=digest(args.previous / 'decisions' / p.name))
            if hashes[p.name]['packet'] != previous['inputs'][p.name]['packet']:
                raise ValueError(f'Frozen evidence was modified: {p}')
            folder = args.source / 'records' / f'{lead.lower()}-quality-as-hint' / record
            count = sum(1 for _ in jsonlines(p))
            jobs.append((lead, task, folder, p, count, offset))
            offset += count

    args.output.mkdir(parents=True, exist_ok=False)
    validity_path = args.source / 'evaluation_validity.json'
    provenance = dict(source_sha256=source_hash, references_sha256=ref_hash, inputs=hashes,
                      validity_sha256=digest(validity_path), runner_sha256=digest(Path(__file__)),
                      prompt_sha256=digest(PROMPT), devices=args.devices, candidates=offset,
                      previous_comparison_sha256=digest(args.previous / 'comparison.json'))
    save_json(args.output / 'inputs.json', provenance)
    refs = read_jsonl(args.references, 'task_id')
    validity = json.loads(validity_path.read_text(encoding='utf-8'))['records']
    cases, results = diagnostic(jobs, args.previous, refs, validity, args.output)
    write_report(args.output, results)
    print(f'Offline diagnostics completed for {len(cases)} candidates', flush=True)
    if args.diagnostic_only:
        return

    model_hash = {p.name: digest(p) for p in sorted(args.model_path.iterdir())
                  if p.suffix in ('.json', '.safetensors')}
    if model_hash != previous['model_sha256']:
        raise ValueError('Qwen checkpoint differs from the previous run')
    save_json(args.output / 'inputs.json', dict(provenance, model_sha256=model_hash))
    (args.output / 'decisions').mkdir()
    batches = [[] for _ in args.devices]
    for job in sorted(jobs, key=lambda item: -item[4]):
        min(batches, key=lambda batch: sum(item[4] for item in batch)).append(job)
    with ProcessPoolExecutor(max_workers=len(batches)) as pool:
        futures = [pool.submit(worker, device, batch, args.model_path, args.output)
                   for device, batch in zip(args.devices, batches) if batch]
        for future in futures:
            future.result()

    relation = relation_summary(jobs, args.output, cases)
    intervals, _ = evaluate(jobs, args.output, refs, validity, ('A', *METHODS))
    for group in intervals:
        errors = [abs(row['predicted_af_s'] - row['reference_af_s']) / row['valid_s']
                  for row in group['rows'] if row['valid_s']]
        group['metrics']['valid_time_burden_mae_pp'] = 100 * float(np.mean(errors)) if errors else None
    save_json(args.output / 'comparison.json', dict(
        diagnostics=results, relation=relation, intervals=intervals, wall_s=perf_counter() - now,
    ))
    write_report(args.output, results, relation, intervals)


if __name__ == '__main__':
    main()
