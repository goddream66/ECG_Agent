"""Compare sealed score-only and score-plus-waveform blind reviews after release."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from prepare_decidability_review_v2 import (DECIDABILITY, JUDGMENTS, read_csv,
                                             validate_review)


NEXT_EVIDENCE = {'none', 'more_measurement', 'later_window',
                 'unavailable_tool', 'cannot_tell'}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def analyze(output, results):
    stage_a = output / 'stage_a'
    stage_b = output / 'stage_b'
    a = read_csv(stage_a / 'review.csv')
    key = validate_review(a, stage_a, output)
    sealed = (stage_b / 'LOCKED_STAGE_A_SHA256.txt').read_text(
        encoding='utf-8').strip()
    actual = hashlib.sha256((stage_a / 'review.csv').read_bytes()).hexdigest()
    if sealed != actual:
        raise ValueError('Stage A review changed after Stage B was released')
    b = read_csv(stage_b / 'review.csv')
    if len(b) != len(a) or [row['case_id'] for row in b] != \
            [row['case_id'] for row in a]:
        raise ValueError('Stage B cases or order differ from Stage A')
    paired = read_jsonl(results / 'paired.jsonl')
    by_identity = {(row['group'], row['record_id'], row['lead'], row['cursor_s']): row
                   for row in paired}
    counts = Counter()
    by_arm = defaultdict(Counter)
    by_cluster = defaultdict(set)
    case_rows = []
    records = set()
    for ar, br, answer in zip(a, b, key):
        if br['review_status'] != 'reviewed' or br['decidability'] not in DECIDABILITY \
                or br['judgment'] not in JUDGMENTS or not br['reviewer'].strip() \
                or not br['waveform_basis'].strip() or \
                br['next_evidence_needed'] not in NEXT_EVIDENCE:
            raise ValueError(f"Incomplete Stage B review: {br['case_id']}")
        try:
            confidence = int(br['confidence_1to5'])
        except ValueError as exc:
            raise ValueError(f"Missing Stage B confidence: {br['case_id']}") from exc
        if confidence not in range(1, 6):
            raise ValueError(f"Stage B confidence outside 1-5: {br['case_id']}")
        if br['decidability'] == 'sufficient' and br['judgment'] == 'UNCERTAIN':
            raise ValueError(f"Stage B sufficient needs binary judgment: {br['case_id']}")
        if int(br['order']) != int(ar['order']) or \
                br['record_alias'] != ar['record_alias'] or \
                float(br['cursor_s']) != float(ar['cursor_s']):
            raise ValueError('Stage B review metadata was modified')
        identity = (answer['group'], answer['record_id'], answer['lead'],
                    answer['cursor_s'])
        result = by_identity[identity]
        reference = ('AF_AFL_ENDED' if answer['reference'] == 'positive'
                     else 'AF_AFL_ONGOING')
        cluster = (answer['record_id'], answer['cursor_s'])
        records.add(answer['record_id'])
        score_sufficient = ar['decidability'] == 'sufficient'
        wave_sufficient = br['decidability'] == 'sufficient'
        if score_sufficient:
            category = 'score_sufficient'
        elif wave_sufficient:
            category = 'waveform_adds_information'
        else:
            category = 'neither_sufficient'
        counts[category] += 1
        by_cluster[category].add(cluster)
        counts['stage_a_reference_agree'] += ar['judgment'] == reference
        counts['stage_b_reference_agree'] += br['judgment'] == reference
        counts['stage_a_binary'] += ar['judgment'] != 'UNCERTAIN'
        counts['stage_b_binary'] += br['judgment'] != 'UNCERTAIN'
        if category == 'score_sufficient' and ar['judgment'] == reference:
            counts['score_sufficient_reference_concordant'] += 1
        if category == 'waveform_adds_information' and br['judgment'] == reference:
            counts['waveform_adds_reference_concordant'] += 1
        if category == 'neither_sufficient':
            counts['still_need_' + br['next_evidence_needed']] += 1
        for arm in ('raw', 'aligned', 'reasoned'):
            decision = result[f'{arm}_decision']
            by_arm[arm][category + '_n'] += 1
            by_arm[arm][category + '_reference_errors'] += decision != reference
            if score_sufficient and ar['judgment'] == reference and decision != reference:
                by_arm[arm]['score_sufficient_qwen_error_candidates'] += 1
            if category == 'waveform_adds_information' and \
                    br['judgment'] == reference and decision != reference:
                by_arm[arm]['waveform_information_error_candidates'] += 1
        case_rows.append({'case_id': ar['case_id'], 'record_alias': ar['record_alias'],
                          'cursor_s': ar['cursor_s'], 'category': category,
                          'stage_a_judgment': ar['judgment'],
                          'stage_a_decidability': ar['decidability'],
                          'stage_a_confidence': int(ar['confidence_1to5']),
                          'stage_b_judgment': br['judgment'],
                          'stage_b_decidability': br['decidability'],
                          'stage_b_confidence': confidence,
                          'next_evidence_needed': br['next_evidence_needed'],
                          'reference': reference,
                          **{arm: result[f'{arm}_decision'] for arm in
                             ('raw', 'aligned', 'reasoned')}})
    return {'n_cases': len(a), 'n_unique_record_cursor': len({
                (row['record_alias'], row['cursor_s']) for row in case_rows}),
            'n_records': len(records), 'counts': dict(counts),
            'unique_timepoints_by_category': {name: len(clusters) for name, clusters
                                              in by_cluster.items()},
            'by_arm': {arm: dict(value) for arm, value in by_arm.items()},
            'cases': case_rows,
            'interpretation_limit': ('The 35 cases overlap within only a few records. '
                'Reviewer judgments are independent assessments, not clinical ground '
                'truth. Counts are descriptive; review reference disagreements before '
                'attributing a Qwen error to reasoning or missing information.')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.review_root, args.results)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding='utf-8')
    print(json.dumps({'n_cases': report['n_cases'], 'counts': report['counts']}))


if __name__ == '__main__':
    main()
