"""Paired summary of fixed-evidence transition, audit and history-content probes."""
import argparse
from collections import Counter
import json
from pathlib import Path

ARMS = ('temporal', 'audit', 'history_none', 'history_relevant', 'history_control')


def read(directory, arm):
    return [json.loads(line) for path in sorted(directory.glob(f'{arm}-*.jsonl'))
            for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def key(row):
    return row['record_id'], row['lead'], row['cursor_s'], row['reference']


def summarize(rows):
    positive = [r for r in rows if r['reference'] == 'positive']
    negative = [r for r in rows if r['reference'] != 'positive']
    decision = lambda r: (r.get('answer') or {}).get('decision', 'ERROR')
    tp = sum(decision(r) == 'TRANSITION' for r in positive)
    fp = sum(decision(r) == 'TRANSITION' for r in negative)
    return {'cases': len(rows), 'positive': len(positive), 'negative': len(negative),
            'valid': sum(decision(r) in ('CONTINUE', 'TRANSITION', 'UNCERTAIN') for r in rows),
            'transition_tp': tp, 'transition_recall': tp / len(positive) if positive else None,
            'transition_fp': fp,
            'false_transition_rate': fp / len(negative) if negative else None,
            'decisions': dict(Counter(decision(r) for r in rows)),
            'audit_calls': sum(r.get('audit_answer') is not None for r in rows),
            'audit_changed_decision': sum(r.get('audit_answer') is not None and
                                          decision(r) != r['initial_answer']['decision'] for r in rows),
            'model_calls': sum(r.get('model_calls', 0) for r in rows),
            'elapsed_s': round(sum(r.get('elapsed_s', 0) for r in rows), 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--main', type=Path, required=True)
    p.add_argument('--hard', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = {}
    for name, directory in (('selected_endings', args.main),
                            ('transient_low_af', args.hard)):
        by_arm = {arm: {key(row): row for row in read(directory, arm)} for arm in ARMS}
        base = set(by_arm['temporal'])
        if not base or set(by_arm['audit']) != base:
            raise ValueError(f'Unpaired temporal/audit cases: {name}')
        history_keys = set.intersection(*(set(by_arm[arm]) for arm in ARMS[2:]))
        if any(set(by_arm[arm]) != base for arm in ARMS[2:]):
            raise ValueError(f'Missing history arm cases: {name}')
        eligible = {k for k in history_keys if all(
            not by_arm[arm][k].get('skipped') for arm in ARMS[2:])}
        result[name] = {
            'temporal': summarize(list(by_arm['temporal'].values())),
            'audit': summarize(list(by_arm['audit'].values())),
            'history_eligible': len(eligible),
            'history': {arm: summarize([by_arm[arm][k] for k in sorted(eligible)])
                        for arm in ARMS[2:]},
            'history_item_counts': {arm: sum(by_arm[arm][k].get('history_items', 0)
                                             for k in eligible) for arm in ARMS[2:]},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
