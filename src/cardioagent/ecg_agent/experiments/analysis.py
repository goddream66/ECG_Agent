from cardioagent.common.paths import code_root
"""Freeze M0, evaluate completed M1 runs, and compare live/cached evidence."""
import argparse
from collections import Counter
from dataclasses import replace
from itertools import groupby, zip_longest
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.memory.events import EventConfig, EventTracker
from cardioagent.ecg_agent.evaluation.metrics import aggregate, evaluate_record, intersection, paired_bootstrap, prepare_validity, union
from cardioagent.common.io import read_jsonl, write_jsonl


def jsonlines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def old_decision(row):
    if 'decision' in row:
        return row['decision']
    return 'support' if row.get('candidate') is True else 'negative' if row.get('candidate') is False else 'unavailable'


def trace_data(rows):
    updates, unknown = [], []
    previous_start = None
    states = Counter()
    for row in rows:
        decision = old_decision(row)
        memory = row.get('memory') or {}
        state = row.get('state', memory.get('state', 'untracked'))
        start, end = row['start_s'], row['cursor_s']
        states[state] += end-start
        if decision in ('unavailable', 'conflict') or state in ('CANDIDATE', 'PENDING_END', 'UNKNOWN_HOLD', 'pending', 'needs_review'):
            unknown.append([start,end])
        update = row.get('event_update') or memory.get('event_update')
        if update:
            update = dict(update)
            if 'kind' not in update:
                update['kind'] = 'confirm' if update['interval'][0] != previous_start else 'extend'
            previous_start = update['interval'][0]
            updates.append({'recorded_at_s': row['cursor_s'], 'event_update': update})
    return updates, union(unknown), dict(states)


def freeze_baseline(args, tasks, references, validity):
    files = [args.tasks, args.references, args.baseline / 'comparison.json', args.cache / 'manifest.json',
             args.models / '1_lead_ECGFounder.pth']
    files += [p for folder in args.baseline.iterdir() if folder.is_dir()
              for p in (folder / 'predictions.jsonl', folder / 'replay_report.json', folder / 'window_trace.jsonl',
                        folder / 'evaluation/interval_scores.json') if p.exists()]
    fingerprints = {str(p.resolve()): digest(p) for p in files}
    output = args.output / 'baseline_frozen.json'
    manifest = {'files': fingerprints, 'records': len(tasks),
                'reference_events': sum(len(r['intervals']) for r in references.values()),
                'scope': 'exploratory AFDB; not unseen test data',
                'record_folds': {t['record_id']: i % 5 for i, t in enumerate(sorted(tasks.values(), key=lambda t:t['record_id']))},
                'fold_note': 'record-level development grouping only; no subject mapping is available',
                'matching': 'existing prediction-order greedy IoU>=0.5; both-empty=1'}
    if output.exists():
        previous = json.loads(output.read_text(encoding='utf-8'))
        if previous != manifest:
            raise ValueError('Frozen baseline changed')
    else:
        save_json(output, manifest)
    summaries = []
    for lead in ('ECG1','ECG2'):
        for policy in ('threshold','two-window','dual','adaptive'):
            folder = args.baseline / f'{lead.lower()}-{policy}'
            report = json.loads((folder / 'replay_report.json').read_text(encoding='utf-8'))
            if (report['tasks_sha256'] != digest(args.tasks)
                    or report['cache_manifest_sha256'] != digest(args.cache/'manifest.json')
                    or report['model']['checkpoint_sha256'] != digest(args.models/'1_lead_ECGFounder.pth')):
                raise ValueError('Baseline inputs differ from recorded provenance')
            predicted = read_jsonl(folder / 'predictions.jsonl', 'task_id')
            task_results = {r['task_id']:r for r in report['tasks']}
            rows = []
            for task_id, group in groupby(jsonlines(folder/'window_trace.jsonl'), key=lambda r:r['task_id']):
                updates, unknown, states = trace_data(group)
                row = evaluate_record(task_id, predicted[task_id]['intervals'], references[task_id],
                                      valid=validity[task_id]['valid_intervals'], unknown=unknown, updates=updates)
                old = task_results[task_id]
                row.update(model_calls=old.get('model_calls', round(old['scored_seconds']/10)),
                           extra_tool_calls=old.get('review_calls',0), state_seconds=states)
                rows.append(row)
            name = f'{lead.lower()}-{policy}'
            result = {'name':name, 'phase':'baseline', 'metrics':aggregate(rows), 'rows':rows}
            original = json.loads((folder/'evaluation/interval_scores.json').read_text(encoding='utf-8'))
            if any(result['metrics'][k] != original[k] for k in ('tp','fp','fn')):
                raise ValueError('Original matching regression')
            save_json(args.output / f'{name}-metrics.json', result)
            summaries.append(result)
    return summaries


def simple_baselines(args, tasks, references, validity):
    summaries = []
    for lead in ('ECG1','ECG2'):
        trace = args.baseline / f'{lead.lower()}-threshold/window_trace.jsonl'
        source = {task: list(rows) for task, rows in groupby(jsonlines(trace), key=lambda r:r['task_id'])}
        for name in ('causal-median3', 'hysteresis', 'gap10'):
            outputs, predictions, all_updates = [], [], []
            for task_id, rows in source.items():
                config = EventConfig(enter_windows=2 if name=='hysteresis' else 1,
                                     exit_windows=2 if name in ('hysteresis','gap10') else 1,
                                     unknown_grace_s=10 if name=='hysteresis' else 0)
                tracker, history, transformed = EventTracker(config), [], []
                scores = 0
                for row in rows:
                    if row['af_score'] is None:
                        decision = 'unavailable'
                        history.clear()
                    else:
                        score = max(row['af_score'],row['afl_score'])
                        scores += 1
                        history.append(score)
                        if name == 'causal-median3':
                            label = float(np.median(history[-3:])) >= .5
                        elif name == 'hysteresis':
                            # Once confirmed, the lower exit threshold maintains ACTIVE.
                            label = score >= (.4 if tracker.active and tracker.active['confirmed_at_s'] is not None else .6)
                        else:
                            label = score >= .5
                        decision = 'support' if label else 'negative'
                    observation = tracker.observe(row['start_s'],row['cursor_s'],decision)
                    all_updates.append(dict(observation,task_id=task_id))
                    transformed.append(dict(row,decision=decision,state=tracker.state,event_update=observation['event_update']))
                events = tracker.finish(tasks[task_id]['end_s'])
                updates, unknown, states = trace_data(transformed)
                evaluated = evaluate_record(task_id, tracker.intervals, references[task_id],
                                             valid=validity[task_id]['valid_intervals'], unknown=unknown,
                                             updates=updates, events=events)
                evaluated.update(model_calls=scores, extra_tool_calls=0, state_seconds=states)
                outputs.append(evaluated)
                predictions.append({'task_id':task_id,'intervals':tracker.intervals})
            folder = args.output / f'{lead.lower()}-{name}'
            folder.mkdir(exist_ok=True)
            write_jsonl(folder/'predictions.jsonl',predictions)
            write_jsonl(folder/'event_updates.jsonl',all_updates)
            result = {'name':f'{lead.lower()}-{name}','phase':'simple-baseline','metrics':aggregate(outputs),'rows':outputs}
            save_json(folder/'metrics.json',result)
            summaries.append(result)
    return summaries


def evaluate_new(args, tasks, references, validity):
    summaries = []
    for phase in ('fixed','cached','live'):
        root = args.runs / phase
        if not (root/'completion.json').exists():
            continue
        for folder in sorted(root.glob('ecg*-*')):
            if not (folder/'summary.json').exists():
                continue
            rows = []
            for task_id, task in tasks.items():
                path = folder / f'records/{task["record_id"]}'
                result = json.loads((path/'result.json').read_text(encoding='utf-8'))
                if result.get('integrity') and not result['integrity']['identical']:
                    raise ValueError('Fixed evidence integrity failed')
                # Read small per-window rows and separate append-only event updates.
                _, unknown, states = trace_data(jsonlines(path/'window_trace.jsonl'))
                updates = list(jsonlines(path/'event_updates.jsonl'))
                evaluated = evaluate_record(task_id,result['intervals'],references[task_id],
                                              valid=validity[task_id]['valid_intervals'],unknown=unknown,
                                              updates=updates,events=result['events'])
                evaluated.update(result.get('counts',{}))
                evaluated.update(result['execution'])
                evaluated['state_seconds'] = states
                # Keep evaluator coverage, not the runtime's model-score coverage.
                rows.append(evaluated)
            result = {'name':folder.name,'phase':phase,'metrics':aggregate(rows),'rows':rows}
            save_json(folder/'metrics.json',result)
            summaries.append(result)
    return summaries


def compare_scores(args, tasks):
    results = []
    root = args.runs/'live'
    if not (root/'completion.json').exists():
        return {'status':'pending', 'reason':'all live jobs must finish first'}
    for folder in sorted(root.glob('ecg*-*')):
        compared = flips = 0
        maximum = 0.
        state_diffs = []
        for task in tasks.values():
            name = task['record_id']
            locations = {(r['channel'],r['start_s'],r['end_s']):r['waveform_row']
                         for r in jsonlines(args.cache/f'{name}.windows.jsonl')}
            cached = np.load(args.scores/f'{name}.npy',mmap_mode='r')
            for row in jsonlines(folder/f'records/{name}/evidence.jsonl'):
                if row['tool'] != 'score_ecgfounder' or row['status'] != 'ok':
                    continue
                location = locations[(row['lead'],*row['window_s'])]
                expected = cached[location]
                actual = np.array([row['result']['af_score'],row['result']['afl_score']])
                maximum = max(maximum,float(np.max(np.abs(expected-actual))))
                flips += any((float(max(expected)) >= t) != (float(max(actual)) >= t) for t in (.4,.5,.6))
                compared += 1
            if folder.name.endswith('event-adaptive'):
                other = args.runs/'cached'/folder.name/f'records/{name}/window_trace.jsonl'
                if other.exists():
                    mismatch = 0
                    for a,b in zip_longest(jsonlines(folder/f'records/{name}/window_trace.jsonl'),jsonlines(other)):
                        mismatch += a is None or b is None or any(a.get(k)!=b.get(k) for k in ('decision','reason','state'))
                    if mismatch:
                        state_diffs.append({'record':name,'different_windows':mismatch})
            else:
                old = read_jsonl(args.baseline/f'{folder.name[:4]}-dual/predictions.jsonl','task_id')
                new = json.loads((folder/f'records/{name}/result.json').read_text(encoding='utf-8'))
                if new['intervals'] != old[task['task_id']]['intervals']:
                    state_diffs.append({'record':name,'legacy_final_intervals_differ':True})
        results.append({'method':folder.name,'compared_model_results':compared,'max_absolute_score_difference':maximum,
                        'threshold_flip_calls':flips,'decision_or_legacy_regression_differences':state_diffs})
    return {'status':'complete','results':results}


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    tasks, references = read_jsonl(args.tasks,'task_id'), read_jsonl(args.references,'task_id')
    cache_manifest = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    validity = prepare_validity(args.tasks,cache_manifest['raw_root'],args.cache,args.output/'evaluation_validity.json')
    annotation_audit=[]
    for task_id,ref in references.items():
        expected=union(ref['intervals'])
        original=validity[task_id]['raw_annotation_af_intervals']
        same_count=len(expected)==len(original)
        delta=max((abs(a-c) for first,second in zip(expected,original) for a,c in zip(first,second)),default=0.) if same_count else None
        annotation_audit.append({'task_id':task_id,'published_events':len(expected),'raw_events':len(original),
                                 'same_merged_event_count':same_count,'max_boundary_delta_s':delta,
                                 'within_rounding_tolerance':same_count and delta <= .01})
    save_json(args.output/'annotation_audit.json',{
        'original_targets_unchanged':True,'rounding_tolerance_s':.01,'records':annotation_audit,
        'all_within_tolerance':all(r['within_rounding_tolerance'] for r in annotation_audit)})
    summaries = freeze_baseline(args,tasks,references,validity)
    summaries += simple_baselines(args,tasks,references,validity)
    summaries += evaluate_new(args,tasks,references,validity)
    by_name = {r['name']:r for r in summaries if r['phase'] in ('baseline','simple-baseline')}
    for result in summaries:
        if result['phase'] in ('fixed','cached','live'):
            baseline = by_name[f'{result["name"][:4]}-adaptive']
            result['paired_bootstrap_vs_old_adaptive'] = paired_bootstrap(result['rows'],baseline['rows'],
                                                                         repetitions=args.bootstrap)
            if result['name'].endswith(('fixed-evidence','event-adaptive','legacy-dual')):
                result['paired_bootstrap_vs_strong_baselines'] = {
                    name: paired_bootstrap(result['rows'],by_name[f'{result["name"][:4]}-{name}']['rows'],
                                           repetitions=args.bootstrap)
                    for name in ('two-window','dual','hysteresis')}
    parity = compare_scores(args,tasks)
    save_json(args.output/'score_parity.json',parity)
    save_json(args.output/'comparison.json',{'records':len(tasks),'bootstrap_repetitions':args.bootstrap,
              'evaluation_source_sha256': digest(code_root()/'src/cardioagent/ecg_agent/benchmarks/event_evaluate.py'),
              'analysis_script_sha256': digest(Path(__file__)),
              'scope':'AFDB exploratory development; no LTAFDB used','results':summaries})
    live_root=args.runs/'live'
    if (live_root/'completion.json').exists():
        loads={}
        actual_calls=0
        for path in live_root.glob('*/records/*/result.json'):
            execution=json.loads(path.read_text(encoding='utf-8'))['execution']
            identity=f"{execution['worker_pid']}:{execution['worker_model_load_ms']:.6f}"
            loads[identity]=execution['worker_model_load_ms']
            actual_calls+=execution['actual_model_calls']
        save_json(args.output/'live_runtime.json',{
            'pool_completion':json.loads((live_root/'completion.json').read_text(encoding='utf-8')),
            'actual_model_calls':actual_calls,'unique_model_loads':len(loads),
            'model_load_seconds_sum':sum(loads.values())/1000,
            'timing_scope':'concurrent CPU service times, not isolated request latency',
            'software':{package:version(package)
                        for package in ('torch','numpy','scipy','wfdb','pydantic')}})
    lines = ['# ECG event tracking M0–M1 comparison','',
             'AFDB exploratory development; all 23 records. Cached timings exclude neural execution.','',
             '| Phase | Method | Event F1 | Time precision | Time recall | FP min | Short recall | Long fragments | Model calls |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x: 'NA' if x is None else f'{x:.3f}'
    for result in summaries:
        m=result['metrics']
        lines.append(f"| {result['phase']} | {result['name']} | {fmt(m['micro_episode_f1'])} | {fmt(m['time_precision'])} | {fmt(m['time_recall'])} | {fmt(m['non_af_alarm_minutes'])} | {fmt(m['short_event_recall'])} | {m['long_fragment_excess']} | {m['model_calls']} |")
    lines += ['', 'Model-score coverage, decision coverage and event correctness are distinct.',
              'Paired record bootstrap intervals and per-record failures are in comparison.json.',
              'Live parity status: '+parity['status']]
    (args.output/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'methods':len(summaries),'live_parity':parity['status'],'output':str(args.output)},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs/ecg-events-v1-r1'))
    parser.add_argument('--output',type=Path,default=Path('runs/ecg-events-v1-r1/analysis'))
    parser.add_argument('--cache',type=Path,default=Path('runs/afdb-preprocessed-full-20261113'))
    parser.add_argument('--baseline',type=Path,default=Path('runs/afdb-full-review/full-cpu'))
    parser.add_argument('--scores',type=Path,default=Path('runs/afdb-full-review/full-cpu/model_scores'))
    parser.add_argument('--models',type=Path,default=Path('models'))
    parser.add_argument('--tasks',type=Path,default=Path('runs/afdb-full-review/20261113-233212-64035fa8/agent_tasks.jsonl'))
    parser.add_argument('--references',type=Path,default=Path('runs/afdb-full-review/20261113-233212-64035fa8/evaluation_references.jsonl'))
    parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap < 1: parser.error('Bootstrap repetitions must be positive')
    run(args)


if __name__=='__main__': main()
