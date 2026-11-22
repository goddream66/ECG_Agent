"""Independent labels, interval/time metrics and paired record bootstrap for M0/M1."""
from collections import Counter
from math import isfinite
from pathlib import Path
import json

import numpy as np
import wfdb

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.evaluation.intervals import burden_metrics, interval_metrics, iou, mean_burdens
from cardioagent.common.intervals import union_seconds, validate_intervals
from cardioagent.common.io import read_jsonl


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([float(a), float(b)])
    return out


def intersection(left, right):
    a, b = union(left), union(right)
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append([lo, hi])
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def ratio(a, b):
    return a / b if b else None


def greedy_pairs(predicted, reference, tau=.5):
    used, pairs = set(), []
    for i, prediction in enumerate(predicted):
        best, overlap = None, tau
        for j, ref in enumerate(reference):
            if j not in used and iou(prediction, ref) >= overlap:
                best, overlap = j, iou(prediction, ref)
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


def time_metrics(predicted, reference, valid, unknown=()):
    p, r = intersection(predicted, valid), intersection(reference, valid)
    tp = union_seconds(intersection(p, r))
    p_s, r_s, valid_s = union_seconds(p), union_seconds(r), union_seconds(valid)
    unknown_s = union_seconds(intersection(unknown, valid))
    # Unknown retains its error contribution to unconditional time metrics.
    known_valid_s = max(0., valid_s - unknown_s)
    error_intervals_s = p_s + r_s - 2 * tp
    p_unknown = union_seconds(intersection(p, intersection(unknown, valid)))
    r_unknown = union_seconds(intersection(r, intersection(unknown, valid)))
    tp_unknown = union_seconds(intersection(intersection(p, r), intersection(unknown, valid)))
    known_error = max(0., error_intervals_s - (p_unknown + r_unknown - 2 * tp_unknown))
    return {'time_tp_s': tp, 'time_fp_s': p_s - tp, 'time_fn_s': r_s - tp,
            'reference_af_s': r_s, 'predicted_af_s': p_s, 'valid_s': valid_s,
            'unknown_s': unknown_s, 'time_precision': ratio(tp, p_s), 'time_recall': ratio(tp, r_s),
            'unknown_coverage': ratio(unknown_s, valid_s), 'decision_coverage': ratio(known_valid_s, valid_s),
            'unconditional_time_error': ratio(error_intervals_s, valid_s),
            'conditional_known_time_error': ratio(known_error, known_valid_s),
            'known_error_s': max(0., known_error)}




def evaluate_record(task_id, predicted, reference, *, valid=None, unknown=(), updates=(), events=(), duration=None,
                    burden_intervals=None):
    start, end = reference['start_s'], reference['end_s']
    expected = validate_intervals(reference['intervals'], start, end)
    actual = validate_intervals(predicted, start, end)
    supported = actual if burden_intervals is None else validate_intervals(burden_intervals, start, end)
    if valid is None:
        valid = [[start, end]]
    pairs = greedy_pairs(actual, expected, reference.get('tau', .5))
    matched = {j for _, j in pairs}
    unmatched_indices = set(range(len(actual))) - {i for i, _ in pairs}
    counts = [sum(min(b, d) > max(a, c) for a, b in actual) for c, d in expected]
    intersections = [sum(min(b, d) > max(a, c) for c, d in expected) for a, b in actual]
    durations = Counter()
    for j, (a, b) in enumerate(expected):
        group = 'short' if b-a < 30 else 'medium' if b-a < 60 else 'long'
        durations[f'{group}_events'] += 1
        durations[f'{group}_matched'] += j in matched
    confirmations = [u for u in updates if (u.get('event_update') or {}).get('kind') == 'confirm']
    writes = [u['event_update'] for u in updates if u.get('event_update')]
    alarms, compatibility, endings = [], [], []
    alarm_times = {}
    for update in confirmations:
        interval = update['event_update']['interval']
        candidates = [(iou(interval, target), -j, j) for j, target in enumerate(expected)
                      if update['recorded_at_s'] >= target[0]
                      and min(interval[1], target[1]) > max(interval[0], target[0])]
        if candidates:
            _, _, j = max(candidates)
            delay = update['recorded_at_s'] - expected[j][0]
            alarm_times[j] = min(alarm_times.get(j, delay), delay)
    for j, target in enumerate(expected):
        if j in alarm_times:
            alarms.append(alarm_times[j])
        eligible_writes = [u['recorded_at_s'] for u in writes
                           if u['recorded_at_s'] >= target[0] and iou(u['interval'], target) >= reference.get('tau', .5)]
        if eligible_writes:
            compatibility.append(min(eligible_writes) - target[0])
    censored_pred = {tuple(p) for event in events
                     if event.get('right_censored') or event.get('termination_reason') in ('unknown_timeout', 'stream_end')
                     for p in event.get('reported_intervals', [])}
    offset_errors = [abs(actual[i][1] - expected[j][1]) for i, j in pairs
                     if tuple(actual[i]) not in censored_pred and expected[j][1] < end]
    for i, j in pairs:
        for event in events:
            if (event.get('termination_reason') == 'confirmed_end' or
                    (burden_intervals is not None and event.get('termination_reason') == 'llm_close')) and actual[i] in event.get('reported_intervals', []):
                endings.append(event['end_decided_at_s'] - expected[j][1])
                break
    n_long = sum(b-a >= 60 for a, b in expected)
    long_fragments = sum(max(0, n-1) for n, (a, b) in zip(counts, expected) if b-a >= 60)
    long_interruptions = sum(max(0, len(intersection(actual, [[a,b]]))-1)
                             for a,b in expected if b-a >= 60)
    long30 = [(n, b-a) for n, (a,b) in zip(counts, expected) if b-a >= 1800]
    row = {'task_id': task_id, 'duration_s': end-start, **interval_metrics(actual, expected, reference.get('tau', .5)),
           **time_metrics(supported, expected, valid, unknown), **dict(durations),
           'prediction_count': len(actual), 'reference_count': len(expected),
           'segments_per_reference': counts, 'long_fragment_excess': long_fragments, 'long_reference_count': n_long,
           'long_interruptions': long_interruptions, 'long_reference_s': sum(b-a for a,b in expected if b-a >= 60),
           'long_missed': sum(n == 0 for n, (a,b) in zip(counts, expected) if b-a >= 60),
           'long30_fragment_excess': sum(max(0,n-1) for n, _ in long30), 'long30_reference_count': len(long30),
           'merged_predictions': sum(n >= 2 for n in intersections),
           'disjoint_false_predictions': sum(intersections[i] == 0 for i in unmatched_indices),
           'overlapping_unmatched_predictions': sum(intersections[i] > 0 for i in unmatched_indices),
           'onset_error_sum_s': sum(abs(actual[i][0]-expected[j][0]) for i,j in pairs), 'onset_error_n': len(pairs),
           'offset_error_sum_s': sum(offset_errors), 'offset_error_n': len(offset_errors),
           'alarm_delay_sum_s': sum(alarms), 'alarm_delay_n': len(alarms),
           'unalarmed_references': len(expected)-len(alarms),
           'compatibility_delay_sum_s': sum(compatibility), 'compatibility_delay_n': len(compatibility),
           'end_delay_sum_s': sum(endings), 'end_delay_n': len(endings),
           'unknown_timeout_events': sum(e.get('termination_reason') == 'unknown_timeout' for e in events),
           'candidate_timeout_events': sum(e.get('termination_reason') == 'candidate_timeout' for e in events),
           'right_censored_events': sum(e.get('right_censored', False) for e in events),
           **burden_metrics(actual, supported if burden_intervals is not None else None, reference)}
    # Negative time crossed by a merged prediction, not merely the count of merged events.
    crossed = 0.
    for prediction, n in zip(actual, intersections):
        if n >= 2:
            touched = intersection([prediction], expected)
            crossed += sum(max(0., right[0]-left[1]) for left,right in zip(touched,touched[1:]))
    row['merge_crossed_non_af_s'] = crossed
    return row


def aggregate(rows):
    sums = Counter()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
                sums[key] += value
    tp, fp, fn = (sums[k] for k in ('tp', 'fp', 'fn'))
    result = {'records': len(rows), 'tp': tp, 'fp': fp, 'fn': fn,
              'micro_episode_f1': ratio(2*tp, 2*tp+fp+fn) if 2*tp+fp+fn else 1.,
              'macro_episode_f1': sum(r['f1'] for r in rows)/len(rows) if rows else None,
              'time_precision': ratio(sums['time_tp_s'], sums['time_tp_s']+sums['time_fp_s']),
              'time_recall': ratio(sums['time_tp_s'], sums['time_tp_s']+sums['time_fn_s']),
              'non_af_alarm_minutes': sums['time_fp_s']/60,
              'unknown_coverage': ratio(sums['unknown_s'], sums['valid_s']),
              'decision_coverage': ratio(sums['valid_s']-sums['unknown_s'], sums['valid_s']),
              'unconditional_time_error': ratio(sums['time_fp_s']+sums['time_fn_s'], sums['valid_s']),
              'conditional_known_time_error': ratio(sums['known_error_s'], sums['valid_s']-sums['unknown_s']),
              'short_event_recall': ratio(sums['short_matched'], sums['short_events']),
              'long_fragment_excess': sums['long_fragment_excess'],
              'long_fragments_per_reference': ratio(sums['long_fragment_excess'], sums['long_reference_count']),
              'long_interruptions_per_af_hour': ratio(sums['long_interruptions'], sums['long_reference_s']/3600),
              'long_missed': sums['long_missed'],
              'long30_fragment_excess': sums['long30_fragment_excess'],
              'merge_rate': ratio(sums['merged_predictions'], sums['prediction_count']),
              'merge_crossed_non_af_s': sums['merge_crossed_non_af_s'],
              **mean_burdens(rows),
              'extra_calls_per_hour': ratio(sums['extra_tool_calls'], sums['duration_s']/3600),
              'model_calls': sums['model_calls'], 'extra_tool_calls': sums['extra_tool_calls'],
              'actual_model_calls': sums['actual_model_calls'], 'other_tool_seconds': sums['other_tool_ms']/1000,
              'neural_seconds': sums['neural_ms']/1000, 'record_wall_seconds_sum': sums['record_wall_ms']/1000}
    for name in ('onset_error', 'offset_error', 'alarm_delay', 'compatibility_delay', 'end_delay'):
        result[f'{name}_mean_s'] = ratio(sums[f'{name}_sum_s'], sums[f'{name}_n'])
        result[f'{name}_n'] = sums[f'{name}_n']
    for name in ('disjoint_false_predictions', 'overlapping_unmatched_predictions', 'unknown_timeout_events',
                 'candidate_timeout_events', 'right_censored_events', 'unalarmed_references'):
        result[name] = sums[name]
    return result


def paired_bootstrap(rows, baseline, *, repetitions=2000, seed=20261114):
    by_id = {r['task_id']: r for r in baseline}
    if set(by_id) != {r['task_id'] for r in rows}:
        raise ValueError('Paired records differ')
    baseline = [by_id[r['task_id']] for r in rows]
    metrics = ('micro_episode_f1', 'time_recall', 'non_af_alarm_minutes', 'short_event_recall',
               'long_fragments_per_reference', 'mean_burden_absolute_error_pp', 'mean_burden_score',
               'mean_supported_burden_score', 'extra_calls_per_hour')
    generator = np.random.default_rng(seed)
    differences = {key: [] for key in metrics}
    for _ in range(repetitions):
        indices = generator.integers(0, len(rows), len(rows))
        new = aggregate([rows[i] for i in indices])
        old = aggregate([baseline[i] for i in indices])
        for key in metrics:
            if key == 'mean_supported_burden_score':
                paired = [(rows[i]['supported_burden_score'], baseline[i]['supported_burden_score'])
                          for i in indices if rows[i].get('supported_burden_score') is not None
                          and baseline[i].get('supported_burden_score') is not None]
                if paired:
                    differences[key].append(sum(a-b for a,b in paired)/len(paired))
                continue
            if new[key] is not None and old[key] is not None:
                differences[key].append(new[key]-old[key])
    return {'unit': 'record; subject mapping unavailable, residual identity leakage not excluded',
            'seed': seed, 'repetitions': repetitions,
            'differences_new_minus_baseline': {key: {'ci95': np.quantile(values, [.025, .975]).tolist(),
                                                     'valid_resamples': len(values)} if values else None
                                              for key, values in differences.items()}}

from .references import prepare_validity
