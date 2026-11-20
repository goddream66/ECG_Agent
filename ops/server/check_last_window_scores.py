"""Re-run P3-5 scores-only judgments with a controlled last-window intervention."""
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from cardioagent.ecg_agent.llm.providers import build_chat_model
from fixed_transition_p34 import SYSTEM, judge_view, write_jsonl
from fixed_transition_p35 import digest, prompt_view, rows
from evidence_reasoning_p31 import SCHEMA


ARMS = ('observed', 'repeat_previous', 'cap_at_020')
CHANNELS = ('ECG1', 'ECG2')
FIELDS = {'af_score', 'afl_score'}
CAP = .20


def last_windows(case):
    cursor = case['cursor_s']
    evidence = {(row['lead'], tuple(row['window_s'])): row for row in case['scores']}
    expected = {(lead, (cursor - 10*i, cursor - 10*(i-1)))
                for lead in CHANNELS for i in range(1, 7)}
    if len(case['scores']) != 12 or set(evidence) != expected:
        raise ValueError('Expected six arrived score windows from both leads')
    out = {}
    for lead in CHANNELS:
        out[lead] = [evidence[lead, (cursor - 20, cursor - 10)],
                     evidence[lead, (cursor - 10, cursor)]]
        for row in out[lead]:
            values = row['result']
            if row['status'] != 'ok' or set(values or ()) != FIELDS or not all(
                    0 <= value <= 1 for value in values.values()):
                raise ValueError('Intervention needs measured final and preceding scores')
    return out


def intervention(case, arm):
    view = deepcopy(prompt_view(case, 'A'))
    spans = last_windows({'cursor_s': case['cursor_s'], 'scores': view['ecgfounder_scores']})
    if arm == 'observed':
        return view
    for previous, current in spans.values():
        if arm == 'repeat_previous':
            current['result'] = previous['result'].copy()
        elif arm == 'cap_at_020':
            peak = max(current['result'].values())
            if peak > CAP:
                current['result'] = {key: score / peak * CAP for key, score in current['result'].items()}
        else:
            raise ValueError(arm)
    return view


def unique_cases(source):
    grouped = {}
    for row in rows(source):
        last_windows(row)
        key = (row['record_id'], row['cursor_s'])
        view = prompt_view(row, 'A')
        case, ids = grouped.setdefault(key, (row, []))
        if prompt_view(case, 'A') != view or row['case_id'] in ids:
            raise ValueError(f'Nonidentical or duplicate scores-only view at {key}')
        ids.append(row['case_id'])
    return [(case, ids) for _, (case, ids) in sorted(grouped.items())]


def infer(args):
    identity = {'inputs': digest(args.inputs), 'code': digest(Path(__file__)),
                'prompt': sha256(SYSTEM.encode()).hexdigest(),
                'schema': sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest(),
                'model': str(args.model_path.resolve()), 'max_new_tokens': 512,
                'model_files': {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                                for p in args.model_path.iterdir()
                                if p.suffix in ('.json', '.safetensors', '.txt')}}
    cases = unique_cases(args.inputs)[args.part::args.parts]
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / f'part-{args.part}-of-{args.parts}.jsonl'
    prior = rows(dest) if dest.exists() else []
    done = {(r['case_id'], r['arm']) for r in prior}
    expected = {(case['case_id'], arm) for case, _ in cases for arm in ARMS}
    if len(done) != len(prior) or not done <= expected or any(r['identity'] != identity for r in prior):
        raise ValueError('Existing shard does not match this experiment')
    if done == expected:
        return
    model = build_chat_model({'provider': 'transformers', 'model_path': str(args.model_path),
                              'device': 'cuda:0', 'max_new_tokens': 512})
    with dest.open('a', encoding='utf-8') as stream:
        for case, _ in cases:
            for arm in ARMS:
                if (case['case_id'], arm) in done:
                    continue
                result = judge_view(model, intervention(case, arm))
                row = {'case_id': case['case_id'], 'arm': arm, 'identity': identity, **result}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                print(case['record_id'], case['cursor_s'], arm,
                      (row.get('answer') or {}).get('decision'), flush=True)


def decision(row):
    return (row.get('answer') or {}).get('decision', 'ACTION_ERROR')


def summarize(args):
    cases = unique_cases(args.inputs)
    answers = {r['case_id']: r['reference'] for r in rows(args.answer_key)}
    if set(answers) != {case_id for _, ids in cases for case_id in ids}:
        raise ValueError('Answer key does not match the frozen cases')
    predictions = [r for file in args.output.glob('part-*-of-*.jsonl') for r in rows(file)]
    by_key = {(r['case_id'], r['arm']): r for r in predictions}
    if len(by_key) != len(predictions) or set(by_key) != {
            (case['case_id'], arm) for case, _ in cases for arm in ARMS}:
        raise ValueError('Need one prediction per unique view and condition')
    if len({json.dumps(r['identity'], sort_keys=True) for r in predictions}) != 1:
        raise ValueError('Prediction shards have different experiment identities')
    old = [r for file in args.p35.glob('ecg*-of-*.jsonl') for r in rows(file) if r['arm'] == 'A']
    old_by_id = {r['case_id']: r for r in old}
    if set(old_by_id) != set(answers) or len(old_by_id) != len(old):
        raise ValueError('Need the complete original P3-5 A run')
    compared_fields = ('inputs', 'prompt', 'schema', 'model', 'model_files', 'max_new_tokens')
    if any(any(row['identity'][field] != predictions[0]['identity'][field]
               for field in compared_fields) for row in old):
        raise ValueError('P3-5 A must use the same inputs, prompt, schema and model')
    audit = []
    for case, ids in cases:
        reference, = {answers[c] for c in ids}
        variants = {}
        for arm in ARMS:
            view = intervention(case, arm)
            spans = last_windows({'cursor_s': case['cursor_s'], 'scores': view['ecgfounder_scores']})
            run = by_key[case['case_id'], arm]
            variants[arm] = {'decision': decision(run), 'answer': run.get('answer'),
                             'scores': {lead: [r['result'] for r in span] for lead, span in spans.items()}}
        audit.append({'record_id': case['record_id'], 'cursor_s': case['cursor_s'],
                      'case_ids': ids, 'reference_observed_only': reference,
                      'p35_a': [decision(old_by_id[c]) for c in ids], 'variants': variants})
    result = {'unique_record_cursors': len(audit), 'original_lead_cases': len(old),
              'inference_calls': sum(r['attempts'] for r in predictions),
              'observed_reproduced_original': sum(d == r['variants']['observed']['decision']
                                                  for r in audit for d in r['p35_a']),
              'decisions': {}, 'paired_changes_by_record': {}}
    for arm in ARMS:
        result['decisions'][arm] = dict(Counter(r['variants'][arm]['decision'] for r in audit))
    for scope in ('all', *sorted({r['record_id'] for r in audit})):
        group = [r for r in audit if scope == 'all' or r['record_id'] == scope]
        result['paired_changes_by_record'][scope] = {arm: dict(Counter(
            r['variants']['observed']['decision'] + '->' + r['variants'][arm]['decision'] for r in group))
            for arm in ARMS[1:]}
    write_jsonl(args.output / 'judgment_audit.jsonl', audit)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    modes = parser.add_subparsers(dest='mode', required=True)
    run = modes.add_parser('infer')
    run.add_argument('--model-path', type=Path, required=True)
    run.add_argument('--part', type=int, default=0)
    run.add_argument('--parts', type=int, default=1)
    report = modes.add_parser('summarize')
    report.add_argument('--answer-key', type=Path, required=True)
    report.add_argument('--p35', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'infer':
        if not 0 <= args.part < args.parts:
            parser.error('part must be between zero and parts minus one')
        infer(args)
    else:
        summarize(args)


if __name__ == '__main__':
    main()
