"""Summarize paired judgments and export independent decidability review queue."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from evidence_reasoning_p31 import ARMS


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def key(row):
    return row['group'], row['record_id'], row['lead'], row['cursor_s']


def summarize(directory, cases, output):
    baseline = {key({**x, 'group': group}): x for group in ('main', 'hard')
                for x in read(cases / group / 'cases.jsonl')}
    arms = {arm: {} for arm in ARMS}
    variants = {}
    for path in directory.glob('*.jsonl'):
        if not path.name.startswith(('main-', 'hard-')):
            continue
        for row in read(path):
            if row['arm'] in arms:
                arms[row['arm']][key(row)] = row
            else:
                variants.setdefault(key(row), []).append(row)
    if any(set(rows) != set(baseline) for rows in arms.values()):
        raise ValueError('Unpaired or missing evidence-reading cases')
    metrics = {}
    for arm, rows in arms.items():
        metrics[arm] = {}
        for group in ('main', 'hard'):
            selected = [r for r in rows.values() if r['group'] == group]
            positive = [r for r in selected if r['reference'] == 'positive']
            negative = [r for r in selected if r['reference'] == 'negative']
            decision = lambda r: (r.get('answer') or {}).get('decision', 'ERROR')
            metrics[arm][group] = {'n': len(selected),
                'valid': sum(decision(r) != 'ERROR' for r in selected),
                'transition_tp': sum(decision(r) == 'AF_AFL_ENDED' for r in positive),
                'positive_n': len(positive),
                'false_transition': sum(decision(r) == 'AF_AFL_ENDED' for r in negative),
                'negative_n': len(negative),
                'uncertain': sum(decision(r) == 'UNCERTAIN' for r in selected),
                'decisions': dict(Counter(decision(r) for r in selected))}
            numeric_direction_flags = 0
            overlapping_ids = 0
            for row in selected:
                answer = row.get('answer') or {}
                case = baseline[key(row)]
                values = {item['id']: max(item['result'].values())
                          for item in case['primary'] + case['backup']
                          if item['status'] == 'ok'}
                support = set(answer.get('support_transition_ids', []))
                against = set(answer.get('against_transition_ids', []))
                overlapping_ids += len(support & against)
                numeric_direction_flags += sum(values.get(identifier, .5) >= .6 for identifier in support)
                numeric_direction_flags += sum(values.get(identifier, .5) < .4 for identifier in against)
            # A flagged score may be a valid historical boundary reference; this
            # proxy needs manual context review before it can be called an error.
            metrics[arm][group]['numeric_direction_flags'] = numeric_direction_flags
            metrics[arm][group]['ids_on_both_sides'] = overlapping_ids
    paired = []
    for identity, case in baseline.items():
        answers = {arm: (arms[arm][identity].get('answer') or {}) for arm in ARMS}
        primary, backup = case['primary'][-1], case['backup'][-1]
        p = max(primary['result'].values()) if primary['status'] == 'ok' else None
        b = max(backup['result'].values()) if backup['status'] == 'ok' else None
        conflict = p is not None and b is not None and p < .4 and b >= .6
        previous = case['primary'][-2]
        recovery = previous['status'] == 'ok' and previous['result'] and \
            max(previous['result'].values()) >= .6 and p is not None and p < .4
        row = {'group': identity[0], 'record_id': identity[1], 'lead': identity[2],
               'cursor_s': identity[3], 'reference': case['reference'],
               'lead_conflict': conflict, 'prior_support_current_low': recovery}
        for arm, answer in answers.items():
            row[f'{arm}_decision'] = answer.get('decision', 'ERROR')
            row[f'{arm}_against_ids'] = answer.get('against_transition_ids', [])
            row[f'{arm}_support_ids'] = answer.get('support_transition_ids', [])
        row['reasoned_cites_conflicting_backup'] = (backup['id'] in
            answers['reasoned'].get('against_transition_ids', [])) if conflict else None
        paired.append(row)
    counters = {}
    for kind in ('backup', 'recovery', 'order'):
        comparisons = [(arms['reasoned'][identity], variant) for identity, items in variants.items()
                       for variant in items if variant['intervention']['kind'] == kind]
        counters[kind] = {'n': len(comparisons), 'valid': sum('answer' in b for _, b in comparisons),
            'decision_changed': sum((a.get('answer') or {}).get('decision') !=
                                    (b.get('answer') or {}).get('decision') for a, b in comparisons),
            'changed_id_cited': sum(b['intervention'].get('changed_id') in
                ((b.get('answer') or {}).get('support_transition_ids', []) +
                 (b.get('answer') or {}).get('against_transition_ids', []))
                for _, b in comparisons if b['intervention'].get('changed_id'))}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps({'arms': metrics, 'counterfactual': counters,
        'lead_conflict_n': sum(x['lead_conflict'] for x in paired),
        'reasoned_conflict_recognized_n': sum(x['reasoned_cites_conflicting_backup'] is True
                                            for x in paired)}, indent=2), encoding='utf-8')
    (output / 'paired.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in paired), encoding='utf-8')
    fields = ['group', 'record_id', 'lead', 'cursor_s', 'reference', 'review_status',
              'decidability', 'waveform_basis', 'reviewer', 'notes']
    with (output / 'decidability_review.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in paired:
            writer.writerow({**{k: row[k] for k in fields[:5]}, 'review_status': 'pending',
                'decidability': '', 'waveform_basis': '', 'reviewer': '', 'notes': ''})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.results, args.cases, args.output)


if __name__ == '__main__':
    main()
