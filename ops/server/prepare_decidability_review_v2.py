"""Prepare a score-only review, then release waveforms after score review is sealed.

Only stage_a/ or stage_b/ should be handed to a blinded reviewer. The answer key
and source record identities stay in the private parent directory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


DECIDABILITY = {'sufficient', 'insufficient', 'uncertain'}
JUDGMENTS = {'AF_AFL_ONGOING', 'AF_AFL_ENDED', 'UNCERTAIN'}


def read_cases(source):
    cases = []
    for group in ('main', 'hard'):
        path = source / group / 'cases.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                cases.append((group, json.loads(line)))
    cases.sort(key=lambda pair: (pair[1]['record_id'], pair[1]['cursor_s'],
                                 pair[1]['lead'], pair[0]))
    identities = [(group, case['record_id'], case['lead'], case['cursor_s'])
                  for group, case in cases]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate source case identity')
    return cases


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def source_items(case):
    items = case['primary'] + case['backup']
    cursor = float(case['cursor_s'])
    for item in items:
        span = item.get('window_s')
        if span is not None and (len(span) != 2 or span[0] >= span[1]
                                 or span[1] > cursor + 1e-8):
            raise ValueError('Future or invalid ECGFounder measurement')
    return items


def build_stage_a(source, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Review directory already contains data; do not overwrite reviews')
    stage = output / 'stage_a'
    stage.mkdir(parents=True)
    (stage / 'cases').mkdir()
    private = output / 'private_score_packets'
    private.mkdir()
    cases = read_cases(source)
    aliases = {record: f'record-{index:02d}' for index, record in
               enumerate(sorted({case['record_id'] for _, case in cases}), 1)}
    key = []
    review = []
    packet_hashes = {}
    for index, (group, case) in enumerate(cases, 1):
        case_id = f'case-{index:03d}'
        filename = f'cases/{case_id}.json'
        payload = {'case_id': case_id, 'cursor_s': case['cursor_s'],
                   'task': 'Has the previously confirmed AF/AFL episode ended by this cursor?',
                   'measurements': source_items(case)}
        packet = (private / f'{case_id}.json')
        packet.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                          encoding='utf-8')
        packet_hashes[case_id] = hashlib.sha256(packet.read_bytes()).hexdigest()
        key.append({'case_id': case_id, 'group': group,
                    'record_id': case['record_id'], 'lead': case['lead'],
                    'cursor_s': case['cursor_s'], 'reference': case['reference']})
        review.append({'order': index, 'case_id': case_id,
                       'record_alias': aliases[case['record_id']],
                       'cursor_s': case['cursor_s'], 'evidence_file': filename,
                       'review_status': 'pending', 'decidability': '', 'judgment': '',
                       'confidence_1to5': '', 'evidence_ids': '', 'rationale': '',
                       'reviewer': ''})
    (output / 'answer_key.jsonl').write_text(''.join(json.dumps(row) + '\n'
        for row in key), encoding='utf-8')
    (output / 'private_packet_hashes.json').write_text(json.dumps(packet_hashes,
        indent=2), encoding='utf-8')
    write_csv(stage / 'review.csv', list(review[0]), review)
    (stage / 'README.txt').write_text(
        'Stage A: review cases in CSV order, without opening later cases first. '
        'Each JSON contains the same ECGFounder measurement items available to Qwen, '
        'without reference labels or raw waveforms. Complete judgment '
        '(AF_AFL_ONGOING/AF_AFL_ENDED/UNCERTAIN), decidability '
        '(sufficient/insufficient/uncertain), confidence 1-5, cited evidence IDs, '
        'rationale and reviewer. A sufficient decision requires a binary judgment. '
        'Only the current case is released. Return each completed row to the study '
        'operator to release the next case. Stage B is unavailable until all Stage A '
        'rows are reviewed. '
        'Cases from the same anonymous record are ordered by increasing cursor time; '
        'some ECG1/ECG2 cases share a cursor and are correlated. Do not consult later '
        'cases while reviewing an earlier one.\n', encoding='utf-8')
    release_next(output, 'stage-a')
    return len(review)


def release_next(output, stage_name):
    stage = output / ('stage_a' if stage_name == 'stage-a' else 'stage_b')
    rows = read_csv(stage / 'review.csv')
    subfolder = 'cases' if stage_name == 'stage-a' else 'waveforms'
    private = output / ('private_score_packets' if stage_name == 'stage-a'
                        else 'private_waveforms')
    suffix = '.json' if stage_name == 'stage-a' else '.png'
    released_indices = [index for index, row in enumerate(rows)
        if (stage / subfolder / f"{row['case_id']}{suffix}").exists()]
    if released_indices != list(range(len(released_indices))):
        raise ValueError('A future case was released early')
    released = 0
    for index, row in enumerate(rows):
        case_id = row['case_id']
        target = stage / subfolder / f'{case_id}{suffix}'
        if target.exists():
            released += 1
            if row['review_status'] == 'pending':
                return {'released_case': None, 'released_count': released,
                        'total': len(rows), 'waiting_for_review': case_id}
            if row['review_status'] != 'reviewed':
                raise ValueError('Invalid review status')
            continue
        if row['review_status'] != 'pending':
            raise ValueError('Review rows must be completed in release order')
        if any((stage / subfolder / f"{later['case_id']}{suffix}").exists()
               for later in rows[index + 1:]):
            raise ValueError('A future case was released early')
        shutil.copyfile(private / f'{case_id}{suffix}', target)
        return {'released_case': case_id, 'released_count': released + 1,
                'total': len(rows)}
    if any(row['review_status'] != 'reviewed' for row in rows):
        return {'released_case': None, 'released_count': released, 'total': len(rows),
                'waiting_for_review': True}
    return {'released_case': None, 'released_count': released, 'total': len(rows),
            'complete': True}


def validate_review(rows, stage, output):
    key = [json.loads(line) for line in (output / 'answer_key.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != len(key) or [row['case_id'] for row in rows] != \
            [row['case_id'] for row in key]:
        raise ValueError('Review cases or order differ from sealed source')
    packet_hashes = json.loads((output / 'private_packet_hashes.json').read_text(
        encoding='utf-8'))
    aliases = {}
    previous = {}
    for row, answer in zip(rows, key):
        if row['review_status'] != 'reviewed' or row['decidability'] not in DECIDABILITY \
                or row['judgment'] not in JUDGMENTS or not row['reviewer'].strip() \
                or not row['rationale'].strip():
            raise ValueError(f"Incomplete Stage A review: {row['case_id']}")
        if row['decidability'] == 'sufficient' and row['judgment'] == 'UNCERTAIN':
            raise ValueError(f"Sufficient evidence needs a binary judgment: {row['case_id']}")
        try:
            confidence = int(row['confidence_1to5'])
        except ValueError as exc:
            raise ValueError(f"Missing confidence: {row['case_id']}") from exc
        if confidence not in range(1, 6):
            raise ValueError(f"Confidence outside 1-5: {row['case_id']}")
        if int(row['order']) != len(previous.get('_all', [])) + 1:
            raise ValueError('Review order was modified')
        previous.setdefault('_all', []).append(row['case_id'])
        record = answer['record_id']
        if record in aliases and aliases[record] != row['record_alias']:
            raise ValueError('Record alias was modified')
        aliases[record] = row['record_alias']
        if record in previous and float(row['cursor_s']) < previous[record]:
            raise ValueError('Review order leaks a later cursor into an earlier case')
        previous[record] = float(row['cursor_s'])
        if float(row['cursor_s']) != float(answer['cursor_s']):
            raise ValueError('Review cursor was modified')
        if row['evidence_file'] != f"cases/{row['case_id']}.json":
            raise ValueError('Review evidence path was modified')
        packet_file = stage / row['evidence_file']
        if hashlib.sha256(packet_file.read_bytes()).hexdigest() != \
                packet_hashes[row['case_id']]:
            raise ValueError('Score evidence changed after preparation')
        packet = json.loads(packet_file.read_text(encoding='utf-8'))
        available = {item['id'] for item in packet['measurements']
                     if item['status'] == 'ok'}
        cited = {value.strip() for value in row['evidence_ids'].split(';') if value.strip()}
        if not cited <= available:
            raise ValueError(f"Unknown evidence ID: {row['case_id']}")
        if row['decidability'] == 'sufficient' and not cited:
            raise ValueError(f"Sufficient judgment needs cited evidence: {row['case_id']}")
    return key


def render_waveform(raw_root, record, cursor, output, case_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import wfdb

    start = max(0.0, float(cursor) - 60.0)
    sample_start = int(round(start * 250))
    sample_end = int(round(float(cursor) * 250))
    raw = wfdb.rdrecord(str(raw_root / record), sampfrom=sample_start,
                        sampto=sample_end)
    if raw.fs != 250 or raw.p_signal.shape[1] != 2 or len(raw.p_signal) != sample_end - sample_start:
        raise ValueError('Unexpected or incomplete AFDB waveform')
    figure, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=False)
    for block in range(3):
        a = start + 20 * block
        b = min(float(cursor), a + 20)
        first = int(round((a - start) * 250))
        last = int(round((b - start) * 250))
        for lead in range(2):
            axis = axes[2 * block + lead]
            axis.plot([start + index / 250 for index in range(first, last)],
                      raw.p_signal[first:last, lead], linewidth=.55)
            axis.set_xlim(a, b)
            axis.set_ylabel(raw.sig_name[lead])
            axis.grid(alpha=.3)
    axes[-1].set_xlabel('Elapsed seconds; no signal after the decision cursor')
    figure.suptitle(f'{case_id} | cursor {cursor:g} s')
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    plt.close(figure)


def build_stage_b(output, raw_root, render=render_waveform):
    stage_a = output / 'stage_a'
    stage_b = output / 'stage_b'
    if stage_b.exists() and any(stage_b.iterdir()):
        raise FileExistsError('Stage B already exists; do not overwrite reviews')
    rows = read_csv(stage_a / 'review.csv')
    key = validate_review(rows, stage_a, output)
    digest = hashlib.sha256((stage_a / 'review.csv').read_bytes()).hexdigest()
    private = output / 'private_waveforms'
    private.mkdir(exist_ok=True)
    review = []
    for row, answer in zip(rows, key):
        name = f"waveforms/{row['case_id']}.png"
        render(raw_root, answer['record_id'], float(answer['cursor_s']),
               private / f"{row['case_id']}.png", row['case_id'])
        review.append({'order': row['order'], 'case_id': row['case_id'],
                       'record_alias': row['record_alias'], 'cursor_s': row['cursor_s'],
                       'waveform_file': name, 'review_status': 'pending',
                       'decidability': '', 'judgment': '', 'confidence_1to5': '',
                       'waveform_basis': '', 'next_evidence_needed': '', 'reviewer': ''})
    stage_b.mkdir()
    (stage_b / 'waveforms').mkdir()
    write_csv(stage_b / 'review.csv', list(review[0]), review)
    (stage_b / 'LOCKED_STAGE_A_SHA256.txt').write_text(digest + '\n', encoding='utf-8')
    (stage_b / 'README.txt').write_text(
        'Stage B is released only after all Stage A reviews are complete and sealed. '
        'Only one new plot is released after each completed review. Read in CSV order. '
        'Each plot contains both ECG leads from at most 60 seconds '
        'before the case cursor; no later signal is shown. Record judgment, '
        'decidability, confidence 1-5 and waveform basis. If evidence remains '
        'insufficient, choose next_evidence_needed from more_measurement, '
        'later_window, unavailable_tool, or cannot_tell; otherwise use none. '
        'An independent second reviewer is preferable to limit anchoring.\n',
        encoding='utf-8')
    release_next(output, 'stage-b')
    return len(review)


def audit_stage_a(output):
    stage = output / 'stage_a'
    key = [json.loads(line) for line in (output / 'answer_key.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    rows = read_csv(stage / 'review.csv')
    if len(rows) != len(key) or [row['case_id'] for row in rows] != \
            [row['case_id'] for row in key]:
        raise ValueError('Audit case mismatch')
    released = sorted((stage / 'cases').glob('case-*.json'))
    if [path.stem for path in released] != [row['case_id'] for row in rows[:len(released)]]:
        raise ValueError('Released cases are not a chronological prefix')
    packets = []
    for path in released:
        packet = json.loads(path.read_text(encoding='utf-8'))
        if set(packet) != {'case_id', 'cursor_s', 'task', 'measurements'}:
            raise ValueError('Stage A packet contains unexpected fields')
        if any(item.get('window_s') and item['window_s'][1] > packet['cursor_s']
               for item in packet['measurements']):
            raise ValueError('Stage A packet contains future measurements')
        packets.append(packet)
    if list(stage.rglob('*.png')):
        raise ValueError('Waveform leaked into Stage A')
    return {'cases': len(key), 'records': len({item['record_id'] for item in key}),
            'unique_record_cursor': len({(item['record_id'], item['cursor_s'])
                                         for item in key}),
            'released_stage_a_cases': len(released),
            'completed_stage_a_reviews': sum(row['review_status'] == 'reviewed'
                                             for row in rows),
            'stage_b_released': (output / 'stage_b').exists(),
            'released_cases_causal_and_label_blind': True}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='stage', required=True)
    a = sub.add_parser('stage-a')
    a.add_argument('--cases', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('stage-b')
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--raw-root', type=Path, required=True)
    next_case = sub.add_parser('release-next')
    next_case.add_argument('--output', type=Path, required=True)
    next_case.add_argument('--review-stage', choices=('stage-a', 'stage-b'), required=True)
    audit = sub.add_parser('audit-stage-a')
    audit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'release-next':
        print(json.dumps(release_next(args.output, args.review_stage)))
    elif args.stage == 'audit-stage-a':
        print(json.dumps(audit_stage_a(args.output)))
    else:
        count = (build_stage_a(args.cases, args.output) if args.stage == 'stage-a'
                 else build_stage_b(args.output, args.raw_root))
        print(json.dumps({'stage': args.stage, 'cases': count}))


if __name__ == '__main__':
    main()
