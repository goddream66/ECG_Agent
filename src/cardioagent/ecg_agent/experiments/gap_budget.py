"""Audit two-call acquisition opportunities with recorded AFDB tool outcomes."""

import argparse
from collections import Counter
from copy import deepcopy
from itertools import combinations
from pathlib import Path

from cardioagent.common.io import digest, save_json, write_jsonl
from cardioagent.ecg_agent.experiments.analysis import jsonlines
from cardioagent.ecg_agent.experiments.gap_repair import load_records, read_json, relation, temporal_features
from cardioagent.ecg_agent.experiments.original_window_review import QUALITY, SCORE, action_plan, gap_windows


BUDGET = 2
METHODS = ('zero', 'fixed', 'adaptive', 'feasible_adaptive', 'ideal_outcomes')


def apply_observation(packet, item):
    if item['status'] != 'ok':
        return
    window = next(w for w in gap_windows(packet) if w['window_s'] == item['window_s'])
    observation = window['leads'][item['lead']]
    if item['tool'] == SCORE:
        observation['score'] = [item['result']['af_score'], item['result']['afl_score']]
        observation['score_status'] = 'ok'
        observation['score_id'] = item['evidence_id']
    elif item['tool'] == QUALITY:
        observation['quality'] = 'warning' if item['result']['reasons'] else 'no_warning'
        observation['quality_id'] = item['evidence_id']
    else:
        raise ValueError('Unexpected archived tool')


def accepts(packet, mode):
    f = temporal_features(packet)
    if mode == 'strict':
        return f['support_fraction'] == 1.
    return f['support_fraction'] >= 2/3 and f['unknown_s'] == 0 and f['longest_low_only_s'] <= 10.


def prioritise(packet, remaining):
    windows = {tuple(w['window_s']): w for w in gap_windows(packet)}
    threshold = packet['start_threshold']

    def order(item):
        window = windows[tuple(item['window_s'])]
        data = window['leads'][item['lead']]
        leads = tuple(window['leads'].values())
        if any(x['score'] is not None and max(x['score']) >= threshold and x['quality'] == 'no_warning' for x in leads):
            return 4
        if item['tool'] == QUALITY and data['score'] is not None and max(data['score']) >= threshold:
            return 0
        if item['tool'] == SCORE and data['quality'] == 'no_warning':
            return 0
        if item['tool'] == SCORE and data['quality'] == 'not_measured':
            return 1
        return 3

    return min(remaining, key=order)


def shortest_plan(packet, outcomes, mode, budget, *, optimistic=False):
    for count in range(1, budget+1):
        for sequence in combinations(outcomes, count):
            state = deepcopy(packet)
            for item in sequence:
                if optimistic:
                    result = {'reasons': []} if item['tool'] == QUALITY else {'af_score': 1., 'afl_score': 0.}
                    item = {**item, 'status': 'ok', 'result': result}
                apply_observation(state, item)
            if accepts(state, mode):
                return list(sequence)
    return None


def next_action(state, remaining, mode, method, budget):
    if not remaining:
        return None
    if method == 'fixed':
        return remaining[0]
    options = remaining
    if method == 'feasible_adaptive':
        options = shortest_plan(state, remaining, mode, budget, optimistic=True)
        if not options:
            return None
    return prioritise(state, options)


def replay(packet, outcomes, mode, method):
    if accepts(packet, mode):
        return []
    if method == 'zero':
        return None
    if method == 'ideal_outcomes':
        return shortest_plan(packet, outcomes, mode, BUDGET)
    state = deepcopy(packet)
    remaining, used = list(outcomes), []
    while remaining and len(used) < BUDGET:
        chosen = next_action(state, remaining, mode, method, BUDGET-len(used))
        if chosen is None:
            break
        remaining.remove(chosen)
        apply_observation(state, chosen)
        used.append(chosen)
        if accepts(state, mode):
            break
    return used


def source_outcomes(archive, record):
    filename = f'{record.lead}-{record.record_id}.jsonl'
    by_id = {row['evidence_id']: row for row in jsonlines(archive / 'evidence' / f'D-{filename}')}
    decisions = {(tuple(row['gap_s']), row['cursor_s']): row
                 for row in jsonlines(archive / 'decisions' / filename)}
    if len(decisions) != len(record.packets):
        raise ValueError('Archived acquisition decisions differ from the candidate cohort')
    for packet in record.packets:
        gap = tuple(packet['gap_s'])
        cursor = packet['cursor_s']
        row = decisions[gap, cursor]
        outcomes = [by_id[evidence_id] for evidence_id in row['D']['evidence_ids']]
        for item in outcomes:
            a, b = item['window_s']
            if not gap[0] <= a < b <= gap[1] or item['available_at_s'] != cursor or b > cursor:
                raise ValueError('Historical tool outcome violates candidate time envelope')
            if item['record_id'] != record.record_id or item['task_id'] != record.reference['task_id']:
                raise ValueError('Historical tool outcome has wrong identity')
        actual = [(a, b, item['lead'], item['tool']) for item in outcomes for a, b in (item['window_s'],)]
        if outcomes and actual != action_plan(gap_windows(packet), packet['start_threshold']):
            raise ValueError('Incomplete or reordered full acquisition outcomes')
        if len(outcomes) != row['D']['calls'] or (not all(packet['gap_support']) and
                                                 not outcomes and action_plan(gap_windows(packet), packet['start_threshold'])):
            raise ValueError('Missing original acquisition outputs')
        yield packet, outcomes


def run(records, archive):
    output = []
    for record in records:
        for packet, outcomes in source_outcomes(archive, record):
            for mode in ('strict', 'temporal_67%'):
                base = accepts(packet, mode)
                if mode == 'strict' and base != all(packet['gap_support']):
                    raise ValueError('Frozen strict rule diverges')
                entry = dict(lead=record.lead, record_id=record.record_id, gap_s=packet['gap_s'], cursor_s=packet['cursor_s'],
                             relation=relation(record, packet), gate=mode, measurements_available=len(outcomes), methods={})
                for method in METHODS:
                    sequence = replay(packet, outcomes, mode, method)
                    state = deepcopy(packet)
                    for item in sequence or ():
                        apply_observation(state, item)
                    admitted = accepts(state, mode)
                    entry['methods'][method] = dict(link=admitted, added_link=admitted and not base,
                                                     actions=[{key: item[key] for key in ('evidence_id', 'tool', 'lead', 'window_s', 'status')}
                                                              for item in sequence or ()],
                                                     calls=len(sequence or ()), archived_tool_ms=sum(item['duration_ms'] for item in sequence or ()))
                output.append(entry)
    return output


def summarize(cases):
    counts = {}
    for row in cases:
        for method, info in row['methods'].items():
            key = f'{row["gate"]}:{row["lead"]}:{method}'
            counted = counts.setdefault(key, Counter())
            counted['calls'] += info['calls']
            counted['model_actions'] += sum(action['tool'] == SCORE for action in info['actions'])
            counted['archived_tool_ms'] += info['archived_tool_ms']
            counted['new_correct'] += info['added_link'] and row['relation'] == 'continuous'
            counted['new_false'] += info['added_link'] and row['relation'] == 'separate'
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, _ = load_records(args.source)
    archive = args.source / 'runs/afdb-original-window-full-20261118T055927Z'
    inputs = read_json(archive / 'inputs.json')
    published = args.source / 'runs/afdb-event-review-b8d284c/inputs.json'
    if inputs['packets_sha256'] != read_json(published)['packets_sha256'] or inputs['budget'] != BUDGET:
        raise ValueError('Archived original-window replay mismatch')
    cases = run(records, archive)
    counts = summarize(cases)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'comparison.json', dict(budget=BUDGET, methods=counts, model_calls=0,
              independent_tool_calls=0, archived_inputs_sha256=digest(archive/'inputs.json'), code_sha256=digest(Path(__file__))))
    write_jsonl(args.output / 'candidate_actions.jsonl', cases)
    for name, summary in counts.items():
        print(name, dict(summary))


if __name__ == '__main__':
    main()
