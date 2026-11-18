"""Paired AFDB test of Qwen event judgments with and without shifted scores."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from cardioagent.common.intervals import union_seconds
from cardioagent.common.io import digest, read_jsonl, save_json, write_jsonl
from cardioagent.ecg_agent.evaluation.metrics import intersection, ratio, union
from cardioagent.ecg_agent.experiments.analysis import jsonlines
from cardioagent.ecg_agent.experiments.original_window_review import evaluate
from cardioagent.ecg_agent.experiments.qwen_shift_ablation import shifted_observations
from cardioagent.ecg_agent.harness.gap_relationship import GapRelationshipReviewer, PROMPT, RELATIONS
from cardioagent.ecg_agent.llm.providers import build_chat_model


METHODS = ('original', 'shifted')


def inputs(packet, shifts):
    observations = []
    refs = {}
    a, b = packet['gap_s']
    for window in packet['windows']:
        for lead, item in window['leads'].items():
            key = f'O{len(observations)+1}'
            observations.append(dict(id=key, window_s=window['window_s'], lead=lead,
                                     score=item['score'], score_status=item['score_status'],
                                     quality=item['quality'], quality_reasons=item.get('quality_reasons'),
                                     window_decision=window['decision']))
            refs[key] = [packet['evidence_index'][item[field]] for field in ('score_id', 'quality_id')
                         if field in item]
    state = dict(gap_s=[a,b], cursor_s=packet['cursor_s'], threshold=packet['start_threshold'],
                 left=packet['left'], right=packet['right'], observations=observations)
    extended = list(observations)
    for i, item in enumerate(shifts, 1):
        key = f'S{i}'
        extended.append(dict(id=key, window_s=item['shifted_s'], original_window_s=item['window_s'],
                             lead=item['lead'], score=item['shifted'], score_status=item['status'],
                             quality='not_measured', outside_gap_s=item['outside_gap_s']))
        refs[key] = [item['evidence_id']]
    return state, dict(state, observations=extended), refs


def revisions(judgment, state, refs):
    a, b = state['gap_s']
    spans, admissible, checked_spans = [], [], []
    for item in state['observations']:
        c, d = item['window_s']
        if (item['id'] not in judgment['support'] or not a <= c < d <= b or item['score'] is None
                or max(item['score']) < state['threshold']):
            continue
        spans.append([c,d])
        if item['quality'] == 'no_warning':
            admissible.extend(refs[item['id']])
            checked_spans.append([c,d])
    supported = union(spans)
    unknown = []
    previous = a
    for left, right in supported:
        if left > previous:
            unknown.append([previous, left])
        previous = right
    if previous < b:
        unknown.append([previous, b])
    validated = union(checked_spans)
    return dict(model_supported_s=supported, model_unknown_s=unknown, validated_support_s=validated,
                support_ids=admissible if judgment['relation'] == 'continuous' and validated == [[a,b]] else [])


def worker(device, jobs, shifts_root, model_path, output):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    reviewer = GapRelationshipReviewer(build_chat_model(dict(
        provider='transformers', model_path=str(model_path), device='cuda:0', max_new_tokens=384)))
    for _, _, _, path, count, _ in jobs:
        shifts = {}
        if count:
            for row in jsonlines(shifts_root / 'sensitivity' / path.name):
                shifts.setdefault((tuple(row['gap_s']), row['cursor_s']), []).append(row)
        decisions = []
        for packet in jsonlines(path):
            key = tuple(packet['gap_s']), packet['cursor_s']
            observed = shifted_observations(packet, shifts.get(key, []))
            states = inputs(packet, observed)
            refs = states[-1]
            row = dict(gap_s=packet['gap_s'], cursor_s=packet['cursor_s'], shifted_count=len(observed))
            ordering = METHODS if len(decisions) % 2 == 0 else METHODS[::-1]
            for method in ordering:
                state = states[METHODS.index(method)]
                result = reviewer.judge(state)
                row[method] = dict(result, request=state, evidence_index={item['id']: refs[item['id']]
                                   for item in state['observations']}, **revisions(result, state, refs))
            decisions.append(row)
        write_jsonl(output / 'decisions' / path.name, decisions)
        print(f'{path.stem}: {len(decisions)} paired judgments', flush=True)


def reference_relation(gap, left, right, regions, valid):
    a,b = gap
    if union_seconds(intersection([gap], valid)) < b-a-1e-8:
        return 'unscored'
    if any(c <= a < b <= d and max(c,left[0]) < a and b < min(d,right[1]) for c,d in regions):
        return 'continuous'
    return 'separate' if union_seconds(intersection([gap], regions)) < b-a-1e-8 else 'unscored'


def summarize(jobs, out, refs, validity):
    base = {}
    transitions = {}
    rows = []
    localized = {}
    for lead, task, _, path, count, _ in jobs:
        key = task['task_id']
        counts = Counter()
        for packet, row in zip(jsonlines(path), jsonlines(out/'decisions'/path.name), strict=True):
            if any(packet[field] != row[field] for field in ('gap_s', 'cursor_s')):
                raise ValueError('Decision does not match frozen candidate')
            ref = reference_relation(row['gap_s'], packet['left']['interval'], packet['right']['interval'],
                                     refs[key]['intervals'], validity[key]['valid_intervals'])
            rows.append(dict(lead=lead, record=task['record_id'], gap_s=row['gap_s'], cursor_s=row['cursor_s'],
                             reference=ref, **{name: row[name]['relation'] for name in METHODS}))
            transitions.setdefault(lead, Counter())[ref,row['original']['relation'],row['shifted']['relation']] += 1
            for method in METHODS:
                reply = row[method]
                counts[method, ref, reply['relation']] += 1
                counts[method, 'errors', 'count'] += reply['status'] != 'ok'
                counts[method, 'model_seconds', 'sum'] += reply['wall_s']
                counts[method, 'tokens', 'sum'] += reply['usage'].get('prompt_tokens', 0)
                coverage = intersection(reply['model_supported_s'], validity[key]['valid_intervals'])
                positive = intersection(coverage, refs[key]['intervals'])
                localized.setdefault((lead, method, 'supported'), {}).setdefault(key, []).extend(coverage)
                localized.setdefault((lead, method, 'true'), {}).setdefault(key, []).extend(positive)
        base[lead,task['record_id']] = counts
    write_jsonl(out/'relation_labels.jsonl', rows)

    def stats(c, method):
        positives = sum(c[method,'continuous',p] for p in RELATIONS)
        negatives = sum(c[method,'separate',p] for p in RELATIONS)
        tp, fp = c[method,'continuous','continuous'], c[method,'separate','continuous']
        tn = c[method,'separate','separate']
        abstain = c[method,'continuous','uncertain'] + c[method,'separate','uncertain']
        return dict(tp=tp, fp=fp, tn=tn, positives=positives, negatives=negatives, uncertain=abstain,
                    unscored=sum(c[method,'unscored',p] for p in RELATIONS), recall=ratio(tp,positives),
                    false_link_rate=ratio(fp,negatives), precision=ratio(tp,tp+fp),
                    accuracy=ratio(tp+tn,positives+negatives), coverage=ratio(positives+negatives-abstain,positives+negatives),
                    errors=c[method,'errors','count'], model_seconds=c[method,'model_seconds','sum'],
                    prompt_tokens=c[method,'tokens','sum'])

    reports = {}
    for lead in ('ECG1','ECG2'):
        records = [c for (ch,_),c in base.items() if ch == lead]
        total = sum(records, Counter())
        report = {method: stats(total, method) for method in METHODS}
        for method in METHODS:
            covered = sum(union_seconds(spans) for spans in localized.get((lead, method, 'supported'), {}).values())
            true = sum(union_seconds(spans) for spans in localized.get((lead, method, 'true'), {}).values())
            report[method].update(localized_true_s=true, localized_non_af_s=max(0., covered - true))
        rng = np.random.default_rng(20261114)
        deltas = {key: [] for key in ('recall', 'false_link_rate', 'accuracy', 'coverage')}
        for indices in rng.integers(0,len(records),size=(2000,len(records))):
            sample = sum((records[i] for i in indices), Counter())
            plain, shifted = (stats(sample, method) for method in METHODS)
            for key, values in deltas.items():
                if plain[key] is not None and shifted[key] is not None:
                    values.append(shifted[key]-plain[key])
        report['paired_ci95'] = {key: np.quantile(values,[.025,.975]).tolist() if values else None
                                 for key, values in deltas.items()}
        report['transitions'] = [{'reference':ref,'original':a,'shifted':b,'count':n}
                                 for (ref,a,b),n in sorted(transitions.get(lead,{}).items())]
        reports[lead] = report
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','shifted','references','model-path','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--devices',nargs='+',type=int,default=[0])
    parser.add_argument('--records',nargs='+')
    args = parser.parse_args()
    if min(args.devices) < 0 or len(set(args.devices)) != len(args.devices):
        parser.error('GPU IDs must be distinct and nonnegative')
    began = perf_counter()
    tasks = read_jsonl(args.source/'tasks.jsonl','task_id')
    if args.records:
        tasks = {key:t for key,t in tasks.items() if t['record_id'] in args.records}
        if {t['record_id'] for t in tasks.values()} != set(args.records):
            parser.error('Selected records not found')
    if any({'answer', 'intervals', 'burden'} & task.keys() or task['source_database'] != 'afdb'
           for task in tasks.values()):
        parser.error('Expected unlabeled AFDB tasks')
    source_hash = digest(args.source/'comparison.json')
    manifest = json.loads((args.shifted/'inputs.json').read_text(encoding='utf-8'))
    if manifest['source_sha256'] != source_hash:
        raise ValueError('Shifted scores came from a different baseline')
    jobs, hashes = [], {}
    for lead in ('ECG1','ECG2'):
        for task in tasks.values():
            record = task['record_id']
            path = args.shifted/'packets'/f'{lead}-{record}.jsonl'
            hashes[path.name] = dict(packet=digest(path), shifts=digest(args.shifted/'sensitivity'/path.name))
            if hashes[path.name]['packet'] != manifest['packets_sha256'][path.name]:
                raise ValueError('Frozen packet was modified')
            folder = args.source/'records'/f'{lead.lower()}-quality-as-hint'/record
            jobs.append((lead,task,folder,path,sum(1 for _ in jsonlines(path)),0))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'decisions').mkdir()
    model_files = sorted(p for p in args.model_path.iterdir() if p.suffix in ('.safetensors','.json'))
    save_json(args.output/'inputs.json',dict(source_sha256=source_hash, inputs=hashes,
              prompt_sha256=digest(PROMPT), runner_sha256=digest(Path(__file__)), devices=args.devices,
              model_sha256={p.name:digest(p) for p in model_files}))
    batches = [[] for _ in args.devices]
    for job in sorted(jobs,key=lambda item:-item[4]):
        min(batches,key=lambda items:sum(item[4] for item in items)).append(job)
    with ProcessPoolExecutor(max_workers=len(batches)) as pool:
        futures = [pool.submit(worker,device,batch,args.shifted,args.model_path,args.output)
                   for device,batch in zip(args.devices,batches) if batch]
        for future in futures:
            future.result()
    refs = read_jsonl(args.references,'task_id')
    validity = json.loads((args.source/'evaluation_validity.json').read_text(encoding='utf-8'))['records']
    relation = summarize(jobs,args.output,refs,validity)
    intervals,_ = evaluate(jobs,args.output,refs,validity,('A',*METHODS))
    for group in intervals:
        for row in group['rows']:
            row['valid_time_burden'] = ratio(row['predicted_af_s'], row['valid_s'])
            reference = ratio(row['reference_af_s'], row['valid_s'])
            row['valid_time_burden_error'] = (abs(row['valid_time_burden'] - reference)
                                               if reference is not None else None)
        fractions = [row['valid_time_burden'] for row in group['rows'] if row['valid_time_burden'] is not None]
        group['metrics']['mean_valid_time_burden'] = float(np.mean(fractions)) if fractions else None
        errors = [row['valid_time_burden_error'] for row in group['rows'] if row['valid_time_burden_error'] is not None]
        group['metrics']['valid_time_burden_mae_pp'] = 100 * float(np.mean(errors)) if errors else None
    save_json(args.output/'comparison.json',dict(relation=relation,intervals=intervals,
              references_sha256=digest(args.references),wall_s=perf_counter()-began))
    lines = ['# AFDB event relationship with shifted evidence', '',
             '| Lead | Input | Correct continuity | False continuity | Correct separation | Uncertain | Errors |',
             '|---|---|---:|---:|---:|---:|---:|']
    for lead, report in relation.items():
        for method in METHODS:
            m = report[method]
            lines.append(f"| {lead} | {method} | {m['tp']}/{m['positives']} | {m['fp']}/{m['negatives']} | "
                         f"{m['tn']} | {m['uncertain']} | {m['errors']} |")
    (args.output/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
