"""Frozen-evidence transition probe with explicit counterevidence and controlled history.

Reference labels select/evaluate cases only. They never enter the model prompt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from cardioagent.ecg_agent.llm.providers import build_chat_model
import fixed_transition_p1 as p1


AUDIT = {'type': 'function', 'function': {'name': 'audit_transition',
    'description': 'Check a proposed AF/AFL transition against contrary evidence.',
    'parameters': {'type': 'object', 'properties': {
        'decision': {'type': 'string', 'enum': ['CONTINUE', 'TRANSITION', 'UNCERTAIN']},
        'counterevidence_ids': {'type': 'array', 'items': {'type': 'string'}},
        'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
        'reason': {'type': 'string'}},
        'required': ['decision', 'counterevidence_ids', 'evidence_ids', 'reason'],
        'additionalProperties': False}}}

SYSTEM = (p1.SYSTEM + ' A score decrease is a measurement, not direct proof of rhythm '
          'conversion. Distinguish a current consecutive negative run from earlier isolated '
          'negative windows. A prior AF score does not prove AF is still present now. '
          'If the arrived evidence cannot distinguish a brief score dip from a genuine '
          'transition, choose UNCERTAIN.')

AUDIT_SYSTEM = ('A transition was proposed for an active AF/AFL event. Search the supplied '
    'already-arrived evidence for contradictions: current or recently recovered AF support, '
    'lead disagreement, missing measurements, and quality warnings. Check whether negative '
    'evidence persists across windows and usable leads. A past high score alone cannot rule '
    'out a later transition; a current low score alone cannot prove one. Revise to CONTINUE '
    'only with present support, TRANSITION only with adequate current opposing evidence, '
    'otherwise UNCERTAIN. Cite actual IDs. Do not infer unseen waveform or future evidence.')


def _ids(row):
    return sorted({item['id'] for collection in (row['scores'], row['quality'])
                   for item in collection.values() if isinstance(item, dict) and 'id' in item})


def temporal_view(recent, visible, aliases):
    """Keep time-local evidence and distinguish current from earlier negative runs."""
    timeline = p1.sequence(recent, visible, aliases)
    runs = []
    for row in timeline:
        kind = row['interpretation']
        if runs and runs[-1]['interpretation'] == kind and runs[-1]['end_s'] == row['window_s'][0]:
            runs[-1]['end_s'] = row['window_s'][1]
            runs[-1]['window_count'] += 1
            runs[-1]['evidence_ids'] = sorted(set(runs[-1]['evidence_ids']) | set(_ids(row)))
        else:
            runs.append({'start_s': row['window_s'][0], 'end_s': row['window_s'][1],
                         'interpretation': kind, 'window_count': 1,
                         'evidence_ids': _ids(row)})
    current_negative = runs[-1] if runs and runs[-1]['interpretation'] == 'negative' else None
    prior_negative = [run for run in runs[:-1] if run['interpretation'] == 'negative']
    last_support = next((row for row in reversed(timeline)
                         if row['interpretation'] == 'support'), None)
    return {'timeline': timeline,
            'measured_evidence': [p1.compact(e, a) for a, e in zip(aliases, visible)],
            'current_consecutive_negative': current_negative,
            'earlier_negative_runs': prior_negative,
            'last_observed_support': ({'window_s': last_support['window_s'],
                                       'evidence_ids': _ids(last_support)}
                                      if last_support else None),
            'conflict_or_unavailable_runs': [run for run in runs
                                               if run['interpretation'] in ('conflict', 'unavailable')],
            'quality_is_hint_not_diagnosis': True}


def history_pairs(older, primary_lead):
    """Earlier low-score windows followed by high-score recovery, fully before the cursor."""
    scores = sorted((e for e in older if e['tool'] == 'score_ecgfounder' and
                     e['lead'] == primary_lead and e['status'] == 'ok'),
                    key=lambda e: e['window_s'])
    pairs = []
    for i, low in enumerate(scores):
        if max(low['result']['af_score'], low['result']['afl_score']) >= .4:
            continue
        for high in scores[i + 1:i + 3]:
            if high['window_s'][0] - low['window_s'][1] > 10:
                break
            if max(high['result']['af_score'], high['result']['afl_score']) >= .6:
                pairs.append((low, high))
                break
    return pairs


def history_variants(older, primary_lead, limit_pairs=2):
    pairs = history_pairs(older, primary_lead)[-limit_pairs:]
    relevant = list(dict((e['evidence_id'], e) for pair in pairs for e in pair).values())
    used = {e['evidence_id'] for e in relevant}
    # Same patient and lead, same item count; distant older scores supply a content control.
    control = [e for e in older if e['tool'] == 'score_ecgfounder' and
               e['lead'] == primary_lead and e['status'] == 'ok' and
               e['evidence_id'] not in used][:len(relevant)]
    if not relevant or len(control) != len(relevant):
        return None
    return {'none': [], 'relevant': relevant, 'control': control,
            'pairs': [[a['evidence_id'], b['evidence_id']] for a, b in pairs]}


def _call(model, messages, schema, aliases):
    reply = model.complete(messages, [schema])
    if schema is p1.SCHEMA:
        return p1.parse(reply, aliases)
    calls = reply.message.get('tool_calls', [])
    if len(calls) != 1 or calls[0]['function']['name'] != 'audit_transition':
        raise ValueError('Expected one audit_transition call')
    raw = calls[0]['function']['arguments']
    answer = json.loads(raw) if isinstance(raw, str) else raw
    if set(answer) != {'decision', 'counterevidence_ids', 'evidence_ids', 'reason'} or \
            answer['decision'] not in ('CONTINUE', 'TRANSITION', 'UNCERTAIN') or \
            not isinstance(answer['reason'], str) or not answer['reason']:
        raise ValueError('Invalid audit fields')
    for key in ('counterevidence_ids', 'evidence_ids'):
        ids = answer[key]
        if not isinstance(ids, list) or len(set(ids)) != len(ids) or not set(ids) <= aliases.keys():
            raise ValueError('Invalid audit evidence IDs')
        answer['resolved_' + key] = [aliases[x] for x in ids]
    return answer


def judge(model, arm, trace, evidence, cursor, primary_lead):
    recent, visible, older, aliases = p1.view(trace, evidence, cursor)
    payload = temporal_view(recent, visible, aliases)
    history = history_variants(older, primary_lead)
    if arm.startswith('history_'):
        if history is None:
            return {'skipped': 'no_prior_recovered_dip_or_control'}
        condition = arm.removeprefix('history_')
        chosen = history[condition]
        history_aliases = {f'H{i}': e['evidence_id'] for i, e in enumerate(chosen, 1)}
        aliases.update(history_aliases)
        payload['retrieved_history'] = [p1.compact(e, alias) for alias, e in
                                        zip(history_aliases, chosen)]
        payload['history_note'] = ('These are older measurements, not ground truth; '
                                   'they cannot alone establish the current rhythm.')
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': json.dumps({'cursor_s': cursor, 'view': payload},
                                                       allow_nan=False)}]
    started = time.perf_counter()
    try:
        initial = _call(model, messages, p1.SCHEMA, aliases)
        final = initial
        audit = None
        if arm == 'audit' and initial['decision'] == 'TRANSITION':
            audit_messages = messages + [
                {'role': 'assistant', 'content': json.dumps({'proposed_transition': initial},
                                                            allow_nan=False)},
                {'role': 'user', 'content': AUDIT_SYSTEM}]
            audit = _call(model, audit_messages, AUDIT, aliases)
            final = audit
        return {'answer': final, 'initial_answer': initial, 'audit_answer': audit,
                'history_items': len(payload.get('retrieved_history', [])),
                'visible_count': len(visible), 'model_calls': 1 + int(audit is not None),
                'elapsed_s': round(time.perf_counter() - started, 3)}
    except Exception as exc:
        return {'error': type(exc).__name__, 'error_detail': str(exc)[:250],
                'history_items': len(payload.get('retrieved_history', [])),
                'visible_count': len(visible),
                'elapsed_s': round(time.perf_counter() - started, 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--lead', choices=('ecg1', 'ecg2'), required=True)
    p.add_argument('--arms', nargs='+', choices=(
        'temporal', 'audit', 'history_none', 'history_relevant', 'history_control'),
        default=['temporal', 'audit', 'history_none', 'history_relevant', 'history_control'])
    p.add_argument('--hard-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_chat_model({'provider': 'transformers', 'model_path': str(args.model_path),
                              'device': 'cuda:0', 'max_new_tokens': 512})
    selections = ([(rec, 'hard_negative', cursors) for rec, cursors in
                   p1.HARD_NEGATIVE[args.lead].items()] if args.hard_only else
                  [(rec, label, cursors) for rec in p1.CASES
                   for label, cursors in p1.CASES[rec].items()])
    for rec, label, cursors in selections:
        source = (p1.FULL_SOURCE / f'live-{args.lead}' / f'{args.lead}-event-adaptive' /
                  'records' / rec if args.hard_only else p1.SOURCES / args.lead / 'records' / rec)
        if not args.hard_only and rec == '04126' and args.lead == 'ecg2':
            source = source.parent / '04126-clean'
        trace = p1.read_jsonl(source / 'window_trace.jsonl')
        evidence = p1.read_jsonl(source / 'evidence.jsonl')
        hashes = {n: hashlib.sha256((source / n).read_bytes()).hexdigest()
                  for n in ('window_trace.jsonl', 'evidence.jsonl')}
        for cursor in cursors:
            if not any(w['end_s'] == cursor for w in trace):
                continue
            if not args.hard_only and rec == '06453' and args.lead == 'ecg1':
                continue
            item = {'record_id': rec, 'lead': args.lead, 'cursor_s': cursor,
                    'reference': 'negative' if args.hard_only else label,
                    'source': str(source), 'source_sha256': hashes,
                    'rule': p1.rule(trace, cursor)}
            for arm in args.arms:
                row = {**item, 'arm': arm,
                       **judge(model, arm, trace, evidence, cursor, args.lead.upper())}
                with (args.output / f'{arm}-{args.lead}.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(json.dumps({'record': rec, 'lead': args.lead, 'cursor': cursor,
                                  'arm': arm, 'decision': (row.get('answer') or {}).get('decision'),
                                  'error': row.get('error'), 'skipped': row.get('skipped')}), flush=True)


if __name__ == '__main__':
    main()
