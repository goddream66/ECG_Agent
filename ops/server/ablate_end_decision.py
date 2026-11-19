"""Paired, fixed-evidence comparison of AF end-decision interfaces.

This is an independent-cursor diagnostic, not an online event-F1 experiment.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from cardioagent.ecg_agent.llm.providers import build_chat_model
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_local_end_judgment import read_rows, prompt_for

FINAL_FIELDS = {
    'event_id': {'type': 'integer'}, 'end_s': {'type': 'number'},
    'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
    'reason': {'type': 'string'}}


def schema(name, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': name.replace('_', ' '),
            'parameters': {'type': 'object', 'properties': properties,
                           'required': required, 'additionalProperties': False}}}


OLD_TOOLS = [
    schema('finish_review', {'reason': {'type': 'string'},
                             'evidence_ids': FINAL_FIELDS['evidence_ids']},
           ['reason', 'evidence_ids']),
    schema('close_event', FINAL_FIELDS, list(FINAL_FIELDS))]
UNIFIED = [schema('submit_event_decision', {
    'decision': {'type': 'string', 'enum': ['KEEP_OPEN', 'CLOSE', 'DEFER']},
    'event_id': {'type': 'integer'}, 'end_s': {'type': ['number', 'null']},
    'evidence_ids': FINAL_FIELDS['evidence_ids'], 'reason': {'type': 'string'}},
    ['decision', 'event_id', 'end_s', 'evidence_ids', 'reason'])]


def make_view(cursor, trace, evidence, arm):
    view = prompt_for(cursor, trace, evidence)
    view['active_event'] = {'event_id': 1, 'status': 'confirmed_active'}
    if arm == 'unified_limited':
        current = view['recent_windows'][-1]
        primary_lead = evidence[0]['lead']
        view['recent_windows'] = view['recent_windows'][-6:]
        view['measured_evidence'] = [e for e in view['measured_evidence']
                                     if e['window_s'] == [current['start_s'], current['end_s']]
                                     and e['lead'] == primary_lead]
    return view


def decide(model, cursor, trace, evidence, arm):
    view = make_view(cursor, trace, evidence, arm)
    system = ('A confirmed AF/AFL event is active. Signal arrives online. Use only supplied '
              'measurements; low model scores are not definitive rhythm diagnoses. '
              'Decision time and estimated boundary are different. Cite real evidence IDs. ')
    tools = OLD_TOOLS if arm in ('original', 'priority') else UNIFIED
    if arm == 'original':
        system += ('Use finish_review when current review evidence is enough or no further '
                   'measurement is useful. Use close_event only when end is confirmed.')
    elif arm == 'priority':
        system += ('If evidence supports the event ending, call close_event, never '
                   'finish_review. Otherwise call finish_review and explain whether AF '
                   'continues or evidence is insufficient.')
    else:
        system += ('Call submit_event_decision exactly once: KEEP_OPEN if supported, '
                   'CLOSE if the end is supported, DEFER if insufficient or conflicting. '
                   'For non-CLOSE end_s must be null.')
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': json.dumps(view, allow_nan=False)}]
    began = time.perf_counter()
    row = {'cursor_s': cursor, 'arm': arm, 'input_evidence_ids':
           [e['evidence_id'] for e in view['measured_evidence']]}
    try:
        model.last_raw = None
        reply = model.complete(messages, tools)
        calls = reply.message['tool_calls']
        if len(calls) != 1:
            raise ValueError('expected_one_call')
        fn = calls[0]['function']
        args = json.loads(fn['arguments']) if isinstance(fn['arguments'], str) else fn['arguments']
        row.update(tool=fn['name'], arguments=args, usage=reply.usage)
        if fn['name'] not in {t['function']['name'] for t in tools}:
            raise ValueError('unknown_action')
        if not isinstance(args, dict) or not isinstance(args.get('evidence_ids'), list):
            raise ValueError('invalid_arguments')
        if not set(args['evidence_ids']) <= set(row['input_evidence_ids']):
            raise ValueError('unavailable_evidence_id')
        if fn['name'] == 'submit_event_decision':
            if args.get('decision') not in ('KEEP_OPEN', 'CLOSE', 'DEFER'):
                raise ValueError('invalid_decision')
            action = args['decision']
        else:
            action = 'CLOSE' if fn['name'] == 'close_event' else 'FINISH'
        end_s = args.get('end_s')
        if action == 'CLOSE':
            if args.get('event_id') != 1 or not isinstance(end_s, (int, float)) or not view['recent_windows'][0]['start_s'] < end_s <= cursor or not args['evidence_ids']:
                raise ValueError('invalid_close_fields')
        elif end_s is not None:
            raise ValueError('nonclose_end_s')
        row['action'] = action
    except Exception as exc:
        row['error'] = type(exc).__name__ + ':' + str(exc)[:120]
    row['raw_output'] = getattr(model, 'last_raw', None)
    row['elapsed_s'] = round(time.perf_counter()-began, 3)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--llm-config', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True, help='JSON list of cursor seconds')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    trace, evidence = (read_rows(args.source/name) for name in ('window_trace.jsonl', 'evidence.jsonl'))
    cursors = json.loads(args.cases.read_text())
    model = build_chat_model(json.loads(args.llm_config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as out:
        for cursor in cursors:
            for arm in ('original', 'priority', 'unified_limited', 'unified_full'):
                row = decide(model, cursor, trace, evidence, arm)
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                out.flush()
                print(cursor, arm, row.get('action'), row.get('error'), flush=True)
    args.output.with_suffix('.manifest.json').write_text(json.dumps({
        'source': str(args.source), 'source_hashes': {n: hashlib.sha256((args.source/n).read_bytes()).hexdigest()
        for n in ('window_trace.jsonl', 'evidence.jsonl')}, 'cursors': cursors,
        'config': json.loads(args.llm_config.read_text())}, indent=2))


if __name__ == '__main__':
    main()
