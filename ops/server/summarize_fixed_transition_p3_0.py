"""Paired P3-0 content-ablation summary; labels are read only here."""
import argparse
from collections import Counter
import json
from pathlib import Path

ARMS = 'ABCDE'


def rows(path, arm):
    return [json.loads(line) for file in path.glob(f'{arm}-*.jsonl')
            for line in file.read_text(encoding='utf-8').splitlines() if line.strip()]


def key(row):
    return row['record_id'], row['lead'], row['cursor_s'], row['reference']


def summary(items):
    positives = [x for x in items if x['reference'] == 'positive']
    negatives = [x for x in items if x['reference'] != 'positive']
    decision = lambda x: (x.get('answer') or {}).get('decision', 'ERROR')
    tp = sum(decision(x) == 'TRANSITION' for x in positives)
    fp = sum(decision(x) == 'TRANSITION' for x in negatives)
    return {'cases': len(items), 'positive': len(positives), 'negative': len(negatives),
            'valid': sum(decision(x) in ('CONTINUE', 'TRANSITION', 'UNCERTAIN') for x in items),
            'transition_tp': tp, 'transition_recall': tp / len(positives) if positives else None,
            'transition_fp': fp,
            'false_transition_rate': fp / len(negatives) if negatives else None,
            'decisions': dict(Counter(decision(x) for x in items)),
            'elapsed_s': round(sum(x.get('elapsed_s', 0) for x in items), 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--main', type=Path, required=True)
    p.add_argument('--hard', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = {}
    for name, path in (('selected_endings', args.main), ('transient_low_af', args.hard)):
        data = {arm: rows(path, arm) for arm in ARMS}
        keys = [set(map(key, items)) for items in data.values()]
        if not keys[0] or any(x != keys[0] for x in keys[1:]) or any(len(x) != len(keys[0]) for x in data.values()):
            raise ValueError(f'Unpaired P3-0 cases in {name}')
        result[name] = {arm: summary(items) for arm, items in data.items()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
