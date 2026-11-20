"""Paired AF/AFL end judgments with fixed scores and experimental RR evidence."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from cardioagent.ecg_agent.llm.providers import build_chat_model
from fixed_transition_p34 import SYSTEM, judge_view, rr_view, write_jsonl
from evidence_reasoning_p31 import SCHEMA


ARMS = ('A', 'B', 'C', 'D', 'E')
ENDED, ONGOING, UNCERTAIN = 'AF_AFL_ENDED', 'AF_AFL_ONGOING', 'UNCERTAIN'
CONTRASTS = (('A', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'E'), ('B', 'E'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def score_rows(case, index, scores):
    result = []
    cursor = case['cursor_s']
    if cursor < 60 or cursor % 10 != 0:
        raise ValueError('Expected six aligned arrived score windows')
    for i in range(6):
        start = cursor - (6 - i) * 10
        end = start + 10
        for j, lead in enumerate(('ECG1', 'ECG2'), 1):
            window = index[lead, start, end]
            loc = window['waveform_row']
            score = None
            if window['status'] == 'ok':
                af, afl = map(float, scores[loc])
                if not (0 <= af <= 1 and 0 <= afl <= 1):
                    raise ValueError('Invalid ECGFounder score')
                score = {'af_score': af, 'afl_score': afl}
            result.append({'id': f'S{2*i+j}', 'lead': lead, 'window_s': [start, end],
                           'status': 'ok' if score is not None else 'insufficient', 'result': score})
    return result


def prepare(source, cache, score_dir, output):
    frozen = rows(source)
    prepared = []
    manifest = {'p34_inputs_sha256': digest(source), 'score_files': {}}
    differences = []
    for record in sorted({c['record_id'] for c in frozen}):
        index_file = cache / f'{record}.windows.jsonl'
        score_file = score_dir / f'{record}.npy'
        index = {(r['channel'], r['start_s'], r['end_s']): r for r in rows(index_file)}
        scores = np.load(score_file, allow_pickle=False)
        manifest['score_files'][record] = {'index_sha256': digest(index_file),
                                           'scores_sha256': digest(score_file)}
        if scores.shape != (sum(r['status'] == 'ok' for r in index.values()), 2):
            raise ValueError(f'Invalid score cache shape for {record}')
        for case in (c for c in frozen if c['record_id'] == record):
            if any(r['window_s'][1] > case['cursor_s'] for r in case['xqrs_rr']):
                raise ValueError('RR exceeds decision cursor')
            if any(r['window_s'] and r['window_s'][1] > case['cursor_s'] for r in case['quality_hints']):
                raise ValueError('Quality exceeds decision cursor')
            scored = score_rows(case, index, scores)
            lookup = {(r['lead'], *r['window_s']): r['result'] for r in scored}
            for old in case['primary_scores']:
                if old['status'] == 'ok':
                    measured = lookup[old['lead'], *old['window_s']]
                    if measured is None:
                        raise ValueError('Score cache lacks a measured primary window')
                    differences.extend(abs(v - measured[k]) for k, v in old['result'].items())
            prepared.append({k: case[k] for k in ('case_id', 'group', 'record_id', 'lead', 'cursor_s',
                                                   'xqrs_rr', 'quality_hints')} | {'scores': scored})
    if len({r['case_id'] for r in prepared}) != len(frozen):
        raise ValueError('Invalid case IDs')
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'inputs.jsonl', prepared)
    manifest.update({'cases': len(prepared), 'compared_score_values': len(differences),
                     'max_abs_score_difference': max(differences, default=None),
                     'score_statuses': dict(Counter(r['status'] for c in prepared for r in c['scores']))})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)


def prompt_view(case, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return {'cursor_s': case['cursor_s'], 'ecgfounder_scores': case['scores'],
            'rr_measurements': [rr_view(row, quality=arm in ('D', 'E'))
                                for row in case['xqrs_rr']] if arm != 'A' else [],
            'quality_hints': case['quality_hints'] if arm in ('C', 'E') else []}


def infer(source, output, model_path, lead, shard, shards):
    cases = [c for c in rows(source) if c['lead'].lower() == lead][shard::shards]
    identity = {'inputs': digest(source), 'prompt': sha256(SYSTEM.encode()).hexdigest(),
                'schema': sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest(),
                'script': digest(Path(__file__)), 'model': str(model_path.resolve()),
                'model_files': {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in model_path.iterdir()
                                if p.suffix in ('.json', '.safetensors', '.txt')}, 'max_new_tokens': 512}
    dest = output / f'{lead}-{shard}-of-{shards}.jsonl'
    previous = rows(dest) if dest.exists() else []
    if any(row['identity'] != identity for row in previous):
        raise ValueError('Cannot resume predictions with different inputs or model')
    done = {(row['case_id'], row['arm']) for row in previous}
    expected = {(c['case_id'], a) for c in cases for a in ARMS}
    if len(done) != len(previous) or not done <= expected:
        raise ValueError('Invalid saved predictions')
    if done == expected:
        return
    model = build_chat_model({'provider': 'transformers', 'model_path': str(model_path),
                              'device': 'cuda:0', 'max_new_tokens': 512})
    with dest.open('a', encoding='utf-8') as stream:
        for case in cases:
            for arm in ARMS:
                if (case['case_id'], arm) in done:
                    continue
                result = judge_view(model, prompt_view(case, arm))
                row = {'case_id': case['case_id'], 'arm': arm, 'identity': identity, **result}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                print(case['case_id'], arm, (result.get('answer') or {}).get('decision'), flush=True)


def summarize(source, key, output):
    cases = {r['case_id']: r for r in rows(source)}
    answers = {r['case_id']: r['reference'] for r in rows(key)}
    predictions = [r for path in sorted(output.glob('ecg*-*-of-*.jsonl')) for r in rows(path)]
    by_id = {(r['case_id'], r['arm']): r for r in predictions}
    if (set(cases) != set(answers) or len(by_id) != len(predictions)
            or set(by_id) != {(c, a) for c in cases for a in ARMS}):
        raise ValueError('Predictions must have one result in each arm for every fixed case')
    context = ('record_id', 'lead', 'cursor_s', 'group')
    audit = []
    for (case_id, arm), prediction in by_id.items():
        case = cases[case_id]
        answer = prediction.get('answer') or {}
        support = set(answer.get('support_transition_ids', []))
        against = set(answer.get('against_transition_ids', []))
        audit.append({k: case[k] for k in context} | {
            'case_id': case_id, 'arm': arm, 'reference': answers[case_id],
            'decision': answer.get('decision', 'ACTION_ERROR'),
            'has_nonconflicting_support_ids': bool(support) and not support & against,
            'citation_conflict': bool(support & against), 'answer': answer})

    def counts(group):
        return {arm: dict(Counter(f"{row['reference']}:{row['decision']}" for row in group
                                  if row['arm'] == arm)) for arm in ARMS}

    negative = [c for c, label in answers.items() if label == 'negative']
    decision = {(row['case_id'], row['arm']): row['decision'] for row in audit}
    comparisons = {}
    for before, after in CONTRASTS:
        comparisons[f'{before}->{after}'] = dict(Counter(
            f'{decision[c, before]}->{decision[c, after]}' for c in negative))
    errors = [r for r in audit if r['reference'] == 'negative' and r['decision'] == ENDED]
    summary = {'cases': len(cases), 'records': len({c['record_id'] for c in cases.values()}),
               'unique_record_cursors': len({(c['record_id'], c['cursor_s']) for c in cases.values()}),
               'overall': counts(audit), 'by_record': {record: counts(
                   [r for r in audit if r['record_id'] == record])
                   for record in sorted({c['record_id'] for c in cases.values()})},
               'hard_cases': counts([r for r in audit if r['group'] == 'hard']),
               'negative_paired_changes': comparisons,
               'false_end_with_nonconflicting_support_ids': dict(Counter(
                   r['arm'] for r in errors if r['has_nonconflicting_support_ids'])),
               'citation_conflicts': dict(Counter(r['arm'] for r in audit if r['citation_conflict']))}
    write_jsonl(output / 'judgment_audit.jsonl', audit)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    modes = parser.add_subparsers(dest='mode', required=True)
    prepare_args = modes.add_parser('prepare')
    for field in ('source', 'cache', 'scores'):
        prepare_args.add_argument('--' + field, type=Path, required=True)
    infer_args = modes.add_parser('infer')
    infer_args.add_argument('--model-path', type=Path, required=True)
    infer_args.add_argument('--lead', choices=('ecg1', 'ecg2'), required=True)
    infer_args.add_argument('--shard', type=int, default=0)
    infer_args.add_argument('--shards', type=int, default=1)
    summarize_args = modes.add_parser('summarize')
    summarize_args.add_argument('--answer-key', type=Path, required=True)
    args = parser.parse_args()
    inputs = args.output / 'inputs.jsonl'
    if args.mode == 'prepare':
        prepare(args.source, args.cache, args.scores, args.output)
    elif args.mode == 'infer':
        if not 0 <= args.shard < args.shards:
            parser.error('shard must be between zero and shards minus one')
        infer(inputs, args.output, args.model_path, args.lead, args.shard, args.shards)
    else:
        summarize(inputs, args.answer_key, args.output)


if __name__ == '__main__':
    main()
