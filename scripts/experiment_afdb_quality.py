"""Compare three uses of ECG quality on AFDB using cached ECGFounder scores."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from cardioagent.common.intervals import union_seconds
from cardioagent.common.io import digest, read_jsonl, save_json, write_jsonl
from cardioagent.common.paths import code_root, default_inputs
from cardioagent.ecg_agent.agent import replay_event_record
from cardioagent.ecg_agent.data.afdb import CHANNELS
from cardioagent.ecg_agent.evaluation.metrics import aggregate, evaluate_record, intersection, ratio
from cardioagent.ecg_agent.evaluation.references import prepare_validity
from cardioagent.ecg_agent.experiments.analysis import jsonlines, trace_data
from cardioagent.ecg_agent.harness.logging import JsonLogs
from cardioagent.ecg_agent.memory.evidence import canonical_hash
from cardioagent.ecg_agent.policies.event_rules import EventConfig
from cardioagent.ecg_agent.policies.labels import score_label
from cardioagent.ecg_agent.schemas import EvidenceContext


def classify(label):
    return 'support' if label else 'negative'


def score_only_review(call, primary, secondary, tracker, config):
    threshold = config.threshold_for(tracker)
    label = score_label(primary, threshold)
    if label is not None:
        return classify(label), False, 'primary_score_only'
    other = score_label(call('score_ecgfounder', secondary), threshold)
    if other is None:
        return 'unavailable', False, 'no_score_on_either_lead'
    return classify(other), True, 'secondary_score_only'


def quality_flag(result):
    if result['status'] != 'ok' or result['result'].get('reasons') is None:
        return 'quality_measurement_unavailable'
    return 'quality_warning' if result['result']['reasons'] else 'no_quality_warning'


def quality_as_hint(call, primary, secondary, tracker, config):
    threshold = config.threshold_for(tracker)
    label = score_label(primary, threshold)
    if label is None:
        return score_only_review(call, primary, secondary, tracker, config)

    score = max(primary['result']['af_score'], primary['result']['afl_score'])
    if not label and tracker.active is None and score <= threshold - config.borderline_margin:
        return 'negative', False, 'primary_negative_without_event'

    primary_quality = quality_flag(call('assess_quality', primary['lead']))
    consequential = (
        abs(score - threshold) < config.borderline_margin
        or tracker.state in ('CANDIDATE', 'UNKNOWN_HOLD', 'PENDING_END')
        or (tracker.active is not None and not label)
        or (tracker.state == 'ACTIVE' and tracker.since_check + 1 >= config.recheck_windows)
        or (tracker.state == 'IDLE' and label and primary_quality == 'quality_warning')
    )
    if not consequential or config.review_budget < 3:
        return classify(label), False, 'retain_primary_' + primary_quality

    other = score_label(call('score_ecgfounder', secondary), threshold)
    if other is None:
        return classify(label), False, 'retain_primary_secondary_score_unavailable'
    secondary_quality = quality_flag(call('assess_quality', secondary))
    if label == other:
        return classify(label), True, 'lead_scores_agree'
    if primary_quality == 'quality_warning' and secondary_quality == 'no_quality_warning':
        return classify(other), True, 'prefer_secondary_on_quality_disagreement'
    if primary_quality == 'no_quality_warning' and secondary_quality == 'quality_warning':
        return classify(label), True, 'prefer_primary_on_quality_disagreement'
    return 'conflict', False, 'lead_scores_conflict_without_quality_preference'


METHODS = {
    'score-only': score_only_review,
    'current-quality-gate': None,
    'quality-as-hint': quality_as_hint,
}


def load_inputs(args):
    manifest_file = args.cache / 'manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    raw_root = args.raw_root or Path(manifest['raw_root'])
    checkpoint = args.models / '1_lead_ECGFounder.pth'
    expected = {
        'cache_manifest_sha256': digest(manifest_file),
        'checkpoint_sha256': digest(checkpoint),
        'network_sha256': digest(code_root() / 'third_party/ecgfounder/net1d.py'),
        'labels_sha256': digest(code_root() / 'third_party/ecgfounder/tasks.txt'),
    }
    scores_manifest = args.scores / 'provenance.json'
    saved = json.loads(scores_manifest.read_text(encoding='utf-8'))
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError('Cached ECGFounder score provenance does not match the cache and model')

    tasks = list(read_jsonl(args.tasks, 'task_id').values())
    if args.records:
        requested = set(args.records)
        tasks = [task for task in tasks if task['record_id'] in requested]
        if {task['record_id'] for task in tasks} != requested:
            raise ValueError('Some requested record IDs are absent from --tasks')
    if not tasks or len({task['record_id'] for task in tasks}) != len(tasks):
        raise ValueError('Expected exactly one task per selected record')
    records = {record['record_id']: record for record in manifest['records']}
    for task in tasks:
        name = task['record_id']
        if (task.get('source_database') != 'afdb' or name not in records
                or len(name) != 5 or not name.isascii() or not name.isdecimal()
                or {'answer', 'intervals', 'burden'} & task.keys()):
            raise ValueError(f'Expected an unlabeled, cached AFDB task for {name!r}')
        if args.duration_s:
            end = task['start_s'] + args.duration_s
            task['end_s'] = min(task['end_s'], end)
    provenance = dict(expected, tasks_sha256=digest(args.tasks), config_sha256=digest(args.config),
                      scores_manifest_sha256=digest(scores_manifest), script_sha256=digest(Path(__file__)),
                      raw_root=str(raw_root.resolve()), channels=args.channels,
                      records={task['record_id']: {
                          'scores_sha256': digest(args.scores / f"{task['record_id']}.npy"),
                          'index_sha256': digest(args.cache / f"{task['record_id']}.windows.jsonl"),
                          'header_sha256': digest(raw_root / f"{task['record_id']}.hea"),
                          'signal_sha256': digest(raw_root / f"{task['record_id']}.dat"),
                          'start_s': task['start_s'], 'end_s': task['end_s'],
                      } for task in tasks})
    return tasks, records, raw_root, provenance


def run_replays(args, tasks, records, raw_root, config, provenance):
    run_id = uuid4().hex
    manifest_id = canonical_hash(provenance)
    for task in tasks:
        name = task['record_id']
        scores = np.load(args.scores / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        samples = np.load(args.cache / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        if scores.shape != (len(samples), 2):
            raise ValueError(f'{name}: cached score rows do not align with waveform rows')

        def score(lead, values, location):
            return tuple(map(float, scores[location]))

        for method, review in METHODS.items():
            for channel in args.channels:
                folder = result_folder(args.output, channel, method, name)
                logs = JsonLogs(folder)
                context = EvidenceContext(f'{run_id}-{method}-{channel}', task['task_id'], name, manifest_id)
                try:
                    result = replay_event_record(args.cache, records[name], task, channel, config,
                                                 score, logs, context, raw_root=raw_root, rule_review=review)
                finally:
                    logs.close()
                result['experiment_method'] = method
                result['actual_ecgfounder_forwards'] = 0
                save_json(folder / 'result.json', result)
                print(f'{name} {channel} {method}: {result["elapsed_ms"] / 1000:.2f} s', flush=True)


def result_folder(output, channel, method, record_id):
    return output / 'records' / f'{channel.lower()}-{method}' / record_id


def evaluate_job(folder, task, reference, valid):
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    trace = list(jsonlines(folder / 'window_trace.jsonl'))
    _, unknown, _ = trace_data(trace)
    updates = list(jsonlines(folder / 'event_updates.jsonl'))
    row = evaluate_record(task['task_id'], result['intervals'], reference, valid=valid,
                          unknown=unknown, updates=updates, events=result['events'])
    positive = intersection(reference['intervals'], valid)
    direct_unknown = [[win['start_s'], win['end_s']] for win in trace
                      if win['decision'] in ('unavailable', 'conflict')]
    quality_results = [item for item in jsonlines(folder / 'evidence.jsonl')
                       if item['tool'] == 'assess_quality']
    warnings = Counter(reason for item in quality_results if quality_flag(item) == 'quality_warning'
                       for reason in item['result']['reasons'])
    row.update(result['counts'])
    row.update(record_id=task['record_id'], replay_seconds=result['elapsed_ms'] / 1000,
               decision_unknown_af_s=union_seconds(intersection(positive, direct_unknown)),
               tracking_unknown_af_s=union_seconds(intersection(positive, unknown)),
               warning_counts=dict(warnings),
               quality_failures=sum(quality_flag(item) == 'quality_measurement_unavailable' for item in quality_results),
               decision_counts=dict(Counter(win['decision'] for win in trace)),
               reason_counts=dict(Counter(win['reason'] for win in trace)))
    return row


def evaluate(args, tasks, raw_root):
    references = read_jsonl(args.references, 'task_id')
    validities = prepare_validity(args.output / 'tasks.jsonl', raw_root, args.cache,
                                  args.output / 'evaluation_validity.json')
    results = []
    for channel in args.channels:
        for method in METHODS:
            rows = []
            for task in tasks:
                key, name = task['task_id'], task['record_id']
                ref = references[key]
                start, end = task['start_s'], task['end_s']
                if not ref['start_s'] <= start < end <= ref['end_s']:
                    raise ValueError(f'{name}: reference does not cover the requested task span')
                if (start, end) != (ref['start_s'], ref['end_s']):
                    cropped = [[max(start, a), min(end, b)] for a, b in ref['intervals']
                               if min(end, b) > max(start, a)]
                    ref = dict(ref, start_s=start, end_s=end, intervals=cropped,
                               burden=union_seconds(cropped) / (end - start))
                rows.append(evaluate_job(result_folder(args.output, channel, method, name), task, ref,
                                         validities[key]['valid_intervals']))
            metrics = aggregate(rows)
            for field in ('decision_unknown_af_s', 'tracking_unknown_af_s', 'calls_assess_quality',
                          'calls_score_ecgfounder', 'tool_calls', 'quality_failures', 'replay_seconds'):
                metrics[field] = sum(row.get(field, 0) for row in rows)
            af_s = sum(row['reference_af_s'] for row in rows)
            metrics['decision_unknown_af_fraction'] = ratio(metrics['decision_unknown_af_s'], af_s)
            warnings = Counter()
            for row in rows:
                warnings.update(row['warning_counts'])
            metrics['warning_counts'] = dict(warnings)
            results.append(dict(channel=channel, method=method, metrics=metrics, rows=rows))
    return results


def write_report(output, results):
    fields = {
        'micro_episode_f1': 'Event F1', 'time_recall': 'Time recall',
        'non_af_alarm_minutes': 'Non-AF alarm min', 'decision_unknown_af_s': 'AF/AFL seconds with unknown decision',
        'long_fragment_excess': 'Excess fragments', 'merge_rate': 'Merge rate',
        'calls_assess_quality': 'QC calls', 'calls_score_ecgfounder': 'Score reads',
        'extra_tool_calls': 'Extra calls',
    }
    lines = ['# AFDB quality strategy comparison', '',
             'Exploratory cached replay. Neural model forwards and LLM calls are both zero.',
             'Upstream preprocessing already removed some windows from the score cache.',
             'Rhythm truth measures downstream event errors. It is not a signal quality annotation.', '',
             '| Channel | Strategy | ' + ' | '.join(fields.values()) + ' |',
             '|---|---|' + '---:|' * len(fields)]
    for result in results:
        cells = [result['channel'], result['method']]
        for key in fields:
            value = result['metrics'].get(key)
            cells.append('NA' if value is None else f'{value:.4g}')
        lines.append('| ' + ' | '.join(cells) + ' |')
    (output / 'COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = default_inputs()
    for name in ('cache', 'tasks', 'scores', 'models'):
        parser.add_argument('--' + name, type=Path, default=code_root() / defaults[name])
    parser.add_argument('--references', type=Path)
    parser.add_argument('--raw-root', type=Path, help='Override the waveform path in the cache manifest')
    parser.add_argument('--config', type=Path, default=code_root() / 'configs/ecg/policy_hysteresis060040.json')
    parser.add_argument('--output', required=True, type=Path, help='New directory for logs and evaluation')
    parser.add_argument('--records', nargs='+', help='Optional subset of AFDB record IDs')
    parser.add_argument('--channels', nargs='+', choices=CHANNELS, default=list(CHANNELS))
    parser.add_argument('--duration-s', type=float, default=0, help='Crop to a multiple of 10 s; 0 uses the full task')
    args = parser.parse_args()
    if not math.isfinite(args.duration_s) or args.duration_s < 0 or args.duration_s % 10:
        parser.error('--duration-s must be a nonnegative multiple of 10')
    if len(args.channels) != len(set(args.channels)):
        parser.error('--channels must not contain duplicates')
    args.references = args.references or args.tasks.with_name('evaluation_references.jsonl')
    if not args.references.is_file():
        parser.error(f'Reference file not found: {args.references}')
    config = EventConfig(**json.loads(args.config.read_text(encoding='utf-8')))
    tasks, records, raw_root, provenance = load_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / 'tasks.jsonl', tasks)
    began = perf_counter()
    run_replays(args, tasks, records, raw_root, config, provenance)
    results = evaluate(args, tasks, raw_root)
    save_json(args.output / 'comparison.json', dict(
        scope='AFDB exploratory development; cached scores shared across strategies',
        references_sha256=digest(args.references), provenance=provenance, config=config.to_dict(),
        wall_seconds=perf_counter() - began, results=results))
    write_report(args.output, results)
    print(args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
