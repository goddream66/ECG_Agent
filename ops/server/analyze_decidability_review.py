"""Unblind completed P3-2 waveform reviews and stratify fixed-evidence errors."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


CATEGORIES = {'existing_sufficient', 'need_measurement', 'need_wait',
              'tools_insufficient', 'indeterminate'}
JUDGMENTS = {'AF_AFL_ONGOING', 'AF_AFL_ENDED', 'UNCERTAIN'}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def analyze(review_dir, results_dir):
    with (review_dir / 'review.csv').open(encoding='utf-8', newline='') as file:
        reviews = list(csv.DictReader(file))
    keys = read_jsonl(review_dir.parent / 'decidability_answer_key.jsonl')
    paired = read_jsonl(results_dir / 'paired.jsonl')
    by_id = {row['case_id']: row for row in keys}
    if len(by_id) != len(keys) or len(reviews) != len(keys) or \
            {row['case_id'] for row in reviews} != set(by_id):
        raise ValueError('Review and answer-key cases differ')
    by_case = {(row['group'], row['record_id'], row['lead'], row['cursor_s']): row
               for row in paired}
    counts = Counter()
    model = defaultdict(Counter)
    rows = []
    for review in reviews:
        if review['review_status'] == 'pending':
            counts['pending'] += 1
            continue
        if review['review_status'] != 'reviewed':
            raise ValueError(f"Invalid review status: {review['review_status']}")
        category = review['decidability']
        judgment = review['waveform_judgment']
        if category not in CATEGORIES or judgment not in JUDGMENTS or \
                not review['reviewer'].strip() or not review['evidence_basis'].strip():
            raise ValueError(f"Incomplete blind review: {review['case_id']}")
        key = by_id[review['case_id']]
        case_key = (key['group'], key['record_id'], key['lead'], key['cursor_s'])
        case = by_case[case_key]
        reference = ('AF_AFL_ENDED' if key['reference'] == 'positive'
                     else 'AF_AFL_ONGOING')
        counts['reviewed'] += 1
        counts[category] += 1
        counts['review_reference_agree'] += judgment == reference
        for arm in ('raw', 'aligned', 'reasoned'):
            answer = case[f'{arm}_decision']
            model[arm][f'{category}_n'] += 1
            model[arm][f'{category}_reference_errors'] += answer != reference
            model[arm][f'{category}_review_disagreement'] += answer != judgment
        rows.append({'case_id': review['case_id'], 'decidability': category,
                     'review_judgment': judgment, 'reference': reference,
                     'raw': case['raw_decision'], 'aligned': case['aligned_decision'],
                     'reasoned': case['reasoned_decision']})
    return {'n_total': len(keys), 'counts': dict(counts),
            'by_arm': {arm: dict(value) for arm, value in model.items()},
            'cases': rows,
            'interpretation_limit': ('Review judgments are independent assessments, not a '
                'replacement for the reference. Do not select a training or acquisition '
                'route until review coverage and disagreements are examined.')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.review_dir, args.results_dir)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'n_total': report['n_total'], 'counts': report['counts']}))


if __name__ == '__main__':
    main()
