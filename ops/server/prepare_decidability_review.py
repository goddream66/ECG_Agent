"""Blind waveform review sheets for causal AF episode end decidability."""
import argparse
import csv
import json
from pathlib import Path


def build(cases, raw_root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import wfdb

    output.mkdir(parents=True, exist_ok=True)
    fields = ('case_id', 'cursor_s', 'waveform_png', 'evidence_jsonl', 'review_status', 'decidability',
              'waveform_judgment',
              'evidence_basis', 'reviewer', 'notes')
    with (output / 'review.csv').open('w', encoding='utf-8', newline='') as file, \
         (output / 'blinded_evidence.jsonl').open('w', encoding='utf-8') as evidence_file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        index = 0
        for group in ('main', 'hard'):
            for line in (cases / group / 'cases.jsonl').read_text(encoding='utf-8').splitlines():
                case = json.loads(line)
                index += 1
                cursor = case['cursor_s']
                start = max(0, cursor - 60)
                raw = wfdb.rdrecord(str(raw_root / case['record_id']),
                                    sampfrom=int(round(start * 250)),
                                    sampto=int(round(cursor * 250)))
                if raw.fs != 250 or raw.p_signal.shape[1] != 2:
                    raise ValueError('Unexpected AFDB waveform metadata')
                figure, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=False)
                for j in range(3):
                    a = max(start, cursor - (3-j)*20)
                    b = cursor - (2-j)*20
                    for lead in range(2):
                        axis = axes[j*2 + lead]
                        first = int(round((a-start)*250))
                        last = int(round((b-start)*250))
                        times = [start + k/250 for k in range(first, last)]
                        axis.plot(times, raw.p_signal[first:last, lead], linewidth=.55)
                        axis.set_ylabel(raw.sig_name[lead])
                        axis.set_xlim(a, b)
                        axis.grid(alpha=.3)
                axes[-1].set_xlabel('Elapsed seconds, all signal already arrived at review cursor')
                figure.suptitle(f'Case {index:03d} | cursor {cursor:g} s')
                figure.tight_layout()
                name = f'case-{index:03d}.png'
                figure.savefig(output / name, dpi=120)
                plt.close(figure)
                evidence_file.write(json.dumps({
                    'case_id': f'case-{index:03d}', 'cursor_s': cursor,
                    'measurements': [{key: item[key] for key in
                        ('id', 'status', 'lead', 'window_s', 'result') if key in item}
                        for item in case['primary'] + case['backup']],
                }, ensure_ascii=False) + '\n')
                writer.writerow({'case_id': f'case-{index:03d}', 'cursor_s': cursor,
                                 'waveform_png': name, 'evidence_jsonl': 'blinded_evidence.jsonl',
                                 'review_status': 'pending',
                                 'decidability': '', 'waveform_judgment': '',
                                 'evidence_basis': '', 'reviewer': '', 'notes': ''})
    # This linkage contains the reference labels for analysis after blind review.
    with (output.parent / 'decidability_answer_key.jsonl').open('w', encoding='utf-8') as file:
        index = 0
        for group in ('main', 'hard'):
            for line in (cases / group / 'cases.jsonl').read_text(encoding='utf-8').splitlines():
                case = json.loads(line)
                index += 1
                file.write(json.dumps({'case_id': f'case-{index:03d}', 'group': group,
                    'record_id': case['record_id'], 'lead': case['lead'],
                    'cursor_s': case['cursor_s'], 'reference': case['reference']}) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.cases, args.raw_root, args.output)


if __name__ == '__main__':
    main()
