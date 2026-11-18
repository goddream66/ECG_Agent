"""Evaluate repair of frozen AFDB candidate gaps without new model or tool calls."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from cardioagent.common.intervals import union_seconds
from cardioagent.common.io import digest, read_jsonl, save_json, write_jsonl
from cardioagent.ecg_agent.evaluation.metrics import aggregate, evaluate_record, greedy_pairs, intersection
from cardioagent.ecg_agent.experiments.analysis import jsonlines
from cardioagent.ecg_agent.experiments.episode_review import merge_gaps
from cardioagent.ecg_agent.experiments.qwen_gap_relationship import reference_relation


LEADS = ('ECG1', 'ECG2')
LENGTH_LIMITS = (10, 20, 30, 40, 50, 60)
MIN_SUPPORT = (0.50, 2 / 3, 0.80)
MAX_LOW_S = 10
STAGE_ONE = ('no_repair', 'strict', 'ideal_reference')


@dataclass(frozen=True)
class Record:
    lead: str
    record_id: str
    intervals: list
    reference: dict
    valid: list
    packets: list


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_records(source):
    runs = source / 'runs'
    quality = runs / 'afdb-quality-full-20261118T034433Z'
    review = runs / 'afdb-event-review-b8d284c'
    labels = runs / 'afdb-qwen-memory-ablation-6ee53d5' / 'memory_pair_labels.jsonl'
    refs_path = source / 'data/afdb_tasks_server/evaluation_references.jsonl'
    tasks_path = quality / 'tasks.jsonl'
    comparison = read_json(review / 'comparison.json')
    inputs = read_json(review / 'inputs.json')
    validity = read_json(quality / 'evaluation_validity.json')
    if (comparison['references_sha256'] != digest(refs_path)
            or inputs['source_comparison_sha256'] != digest(quality / 'comparison.json')
            or validity['tasks_sha256'] != digest(tasks_path)):
        raise ValueError('Frozen tasks, references, and replay comparison disagree')
    tasks = read_jsonl(tasks_path, 'task_id')
    references = read_jsonl(refs_path, 'task_id')
    if len(tasks) != 23 or tasks.keys() != references.keys() or tasks.keys() != validity['records'].keys():
        raise ValueError('Expected the same 23 AFDB tasks for predictions, references, and validity')
    expected_labels = {(row['lead'], row['record'], tuple(row['gap_s']), row['cursor_s']): row['reference']
                       for row in jsonlines(labels)}
    if len(expected_labels) != 263:
        raise ValueError('Expected 263 distinct labeled candidate gaps')
    records = []
    seen = set()
    for lead in LEADS:
        for task_id, task in tasks.items():
            record_id = task['record_id']
            path = review / 'packets' / f'{lead}-{record_id}.jsonl'
            if digest(path) != inputs['packets_sha256'][path.name]:
                raise ValueError(f'Frozen packet fingerprint mismatch: {path}')
            result = read_json(quality / 'records' / f'{lead.lower()}-quality-as-hint' / record_id / 'result.json')
            if (result['task_id'], result['channel'], result['experiment_method']) != (task_id, lead, 'quality-as-hint'):
                raise ValueError(f'Wrong cached prediction: {task_id} / {lead}')
            record = Record(lead, record_id, result['intervals'], references[task_id],
                            validity['records'][task_id]['valid_intervals'], list(jsonlines(path)))
            for packet in record.packets:
                gap = tuple(packet['gap_s'])
                key = (lead, record_id, gap, packet['cursor_s'])
                if key in seen or relation(record, packet) != expected_labels[key]:
                    raise ValueError(f'Candidate label or identity mismatch: {key}')
                if packet['right']['confirmed_at_s'] != packet['cursor_s'] or not 0 < gap[1] - gap[0] <= 60:
                    raise ValueError(f'Candidate review cursor mismatch: {key}')
                seen.add(key)
            merge_gaps(record.intervals, {tuple(packet['gap_s']) for packet in record.packets})
            records.append(record)
    if seen != expected_labels.keys():
        raise ValueError('Frozen candidate set changed')
    return records, comparison


def relation(record, packet):
    return reference_relation(packet['gap_s'], packet['left']['interval'], packet['right']['interval'],
                              record.reference['intervals'], record.valid)


def subtype(record, packet, label):
    if label != 'separate':
        return label
    hits = []
    expected = record.reference['intervals']
    for side in ('left', 'right'):
        span = intersection([packet[side]['interval']], record.valid)
        hits.append([i for i, ref in enumerate(expected) if intersection(span, [ref])])
    left, right = hits
    if len(left) > 1 or len(right) > 1:
        return 'multiple_reference_overlaps'
    if left and right:
        return 'two_reference_events' if left != right else 'ambiguous_same_reference'
    return 'left_only' if left else 'right_only' if right else 'neither_side'


def temporal_features(packet):
    """Use original, quality-measured windows available at the frozen decision time."""
    start, end = packet['gap_s']
    duration = end - start
    threshold = packet['start_threshold']
    features = dict(duration_s=duration, supported_s=0., unknown_s=0., low_only_s=0.,
                    longest_low_only_s=0., longest_unsupported_s=0., disagreement_s=0., dual_usable_s=0.)
    lead_states = {lead: Counter() for lead in LEADS}
    low_run = unsupported_run = 0.
    position = start
    for window in packet['windows']:
        a, b = window['window_s']
        if b > packet['cursor_s']:
            raise ValueError('Future window exposed to repair policy')
        if b <= start or a >= end:
            continue
        if a != position or b > end or b <= a:
            raise ValueError('Expected a disjoint original-window partition of the gap')
        position = b
        length = b - a
        scores = []
        for lead, item in window['leads'].items():
            score = max(item['score']) if item['score_status'] == 'ok' and item['score'] is not None else None
            if score is None:
                state = item['score_status'] if item['score_status'] != 'ok' else 'missing_value'
            elif item['quality'] != 'no_warning':
                state = 'quality_' + item['quality']
            else:
                scores.append(score)
                state = 'support' if score >= threshold else 'low' if score < 0.4 else 'borderline'
            lead_states[lead][state] += length
        support = any(score >= threshold for score in scores)
        low_only = bool(scores) and all(score < 0.4 for score in scores)
        features['supported_s'] += length * support
        features['unknown_s'] += length * (not scores)
        features['low_only_s'] += length * low_only
        features['dual_usable_s'] += length * (len(scores) == 2)
        if len(scores) == 2 and min(scores) < 0.4 and max(scores) >= threshold:
            features['disagreement_s'] += length
        low_run = low_run + length if low_only else 0.
        unsupported_run = unsupported_run + length if not support else 0.
        features['longest_low_only_s'] = max(features['longest_low_only_s'], low_run)
        features['longest_unsupported_s'] = max(features['longest_unsupported_s'], unsupported_run)
    if position != end:
        raise ValueError('Candidate evidence does not cover the gap grid')
    features['support_fraction'] = features['supported_s'] / duration
    features['lead_state_seconds'] = lead_states
    return features


def selections(packet, features, stage):
    methods = ['strict'] if all(packet['gap_support']) else []
    if stage == 2:
        methods.extend(f'length_{limit}s' for limit in LENGTH_LIMITS if features['duration_s'] <= limit)
        if features['unknown_s'] == 0 and features['longest_low_only_s'] <= MAX_LOW_S:
            methods.extend(f'temporal_{cutoff:.0%}' for cutoff in MIN_SUPPORT
                           if features['support_fraction'] >= cutoff)
    return methods


def evaluate(records, methods, stage):
    rows = {(lead, method): [] for lead in LEADS for method in methods}
    audited = []
    for record in records:
        chosen = {method: {} for method in methods}
        for packet in record.packets:
            gap = tuple(packet['gap_s'])
            features = temporal_features(packet)
            selected = selections(packet, features, stage)
            label = relation(record, packet)
            if stage == 1 and label == 'continuous':
                selected.append('ideal_reference')
            for method in selected:
                chosen[method][gap] = label
            audited.append(dict(lead=record.lead, record_id=record.record_id, gap_s=gap, cursor_s=packet['cursor_s'],
                                reference_relation=label, subtype=subtype(record, packet, label),
                                sustained_rule_opposition=packet['sustained_opposition'], choices=selected, **features))
        task_id = record.reference['task_id']
        expected = record.reference['intervals']
        tau = record.reference['tau']
        before = {j for _, j in greedy_pairs(record.intervals, expected, tau)}
        for method, chosen_labels in chosen.items():
            predicted = merge_gaps(record.intervals, set(chosen_labels))
            after = {j for _, j in greedy_pairs(predicted, expected, tau)}
            row = evaluate_record(task_id, predicted, record.reference, valid=record.valid)
            added = intersection(chosen_labels, record.valid)
            true_s = union_seconds(intersection(added, expected))
            counts = Counter(chosen_labels.values())
            row.update(record_id=record.record_id, correct_links=counts['continuous'], false_links=counts['separate'],
                       unscored_links=counts['unscored'], recovered_af_s=true_s,
                       added_non_af_s=max(0., union_seconds(added) - true_s),
                       rescued_event_ids=sorted(after - before), lost_event_ids=sorted(before - after))
            rows[record.lead, method].append(row)
    return rows, audited


def digest_audit(source):
    return {path.relative_to(source).as_posix(): digest(path) for path in sorted(source.rglob('*')) if path.is_file()}


def paired_delta(rows, baseline):
    """Each resample preserves pairing and all candidates from each sampled record."""
    def vector(row):
        return [row[name] for name in ('tp', 'fp', 'fn', 'correct_links', 'false_links', 'recovered_af_s', 'added_non_af_s')]

    if [r['record_id'] for r in rows] != [r['record_id'] for r in baseline]:
        raise ValueError('Record pairing changed')
    idx = np.random.default_rng(20261114).integers(len(rows), size=(2000, len(rows)))
    x, y = (np.array([vector(row) for row in source])[idx].sum(axis=1) for source in (rows, baseline))
    def f1(z):
        denominator = 2 * z[:, 0] + z[:, 1] + z[:, 2]
        return np.divide(2 * z[:, 0], denominator, out=np.ones_like(denominator, dtype=float), where=denominator != 0)
    delta = np.column_stack((f1(x) - f1(y), x[:, 3:] - y[:, 3:]))
    names = ('event_f1', 'correct_links', 'false_links', 'recovered_af_s', 'added_non_af_s')
    return {key: np.quantile(delta[:, k], [0.025, 0.975]).tolist() for k, key in enumerate(names)}


def summarize(rows):
    output = []
    for (lead, method), values in rows.items():
        metrics = aggregate(values)
        metrics.update({key: sum(row[key] for row in values) for key in
                        ('correct_links', 'false_links', 'unscored_links', 'recovered_af_s', 'added_non_af_s')})
        metrics['records_with_correct_link'] = sum(row['correct_links'] > 0 for row in values)
        metrics['rescued_events'] = sum(len(row['rescued_event_ids']) for row in values)
        metrics['lost_events'] = sum(len(row['lost_event_ids']) for row in values)
        output.append(dict(lead=lead, method=method, metrics=metrics, per_record=values,
                           paired_vs_strict=paired_delta(values, rows[lead, 'strict'])))
    return output


def verify_baselines(summary, published):
    prior = {(row['channel'], row['method']): row['metrics'] for row in published['results']}
    for row in summary:
        name = {'no_repair': 'original', 'strict': 'evidence_rule', 'length_60s': 'fixed_gap'}.get(row['method'])
        if name is None:
            continue
        expected = prior[row['lead'], name]
        metrics = row['metrics']
        fields = ('micro_episode_f1', 'long_fragment_excess', 'mean_burden_absolute_error_pp', 'non_af_alarm_minutes')
        if any(not np.isclose(metrics[field], expected[field], rtol=0., atol=1e-9) for field in fields):
            raise ValueError(f'Published baseline failed to reproduce: {row["lead"]} {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    records, published = load_records(args.source)
    methods = STAGE_ONE if args.stage == 1 else (
        'no_repair', 'strict', *(f'length_{n}s' for n in LENGTH_LIMITS), *(f'temporal_{n:.0%}' for n in MIN_SUPPORT))
    rows, cases = evaluate(records, methods, args.stage)
    summary = summarize(rows)
    verify_baselines(summary, published)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'comparison.json', dict(stage=args.stage, scope='exploratory AFDB paired by record and lead',
              results=summary, fingerprints=digest_audit(args.source), code_sha256=digest(Path(__file__)),
              temporal_config=dict(min_support=MIN_SUPPORT, max_low_s=MAX_LOW_S, require_usable_every_window=True),
              length_limits_s=LENGTH_LIMITS, model_calls=0, tool_calls=0))
    write_jsonl(args.output / 'candidate_audit.jsonl', cases)
    for row in summary:
        m = row['metrics']
        print(row['lead'], row['method'], f"F1={m['micro_episode_f1']:.4f}",
              f"links={m['correct_links']}+{m['false_links']} false", f"+AF={m['recovered_af_s']:.2f}s",
              f"+non-AF={m['added_non_af_s']:.2f}s", f"fragments={m['long_fragment_excess']}",
              f"rescued/lost={m['rescued_events']}/{m['lost_events']}")


if __name__ == '__main__':
    main()
