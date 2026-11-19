"""Probe Qwen's local AF ending judgment on frozen, causal evidence."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from cardioagent.ecg_agent.llm.providers import build_chat_model


CHOICES = {'type': 'function', 'function': {
    'name': 'judge_event_end',
    'description': 'Choose the next action for the already active AF/AFL event.',
    'parameters': {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['continue', 'recheck', 'close']},
        'end_s': {'type': ['number', 'null']},
        'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
        'reason': {'type': 'string'}},
        'required': ['action', 'end_s', 'evidence_ids', 'reason'],
        'additionalProperties': False}}}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def evidence_view(row):
    result = row['result']
    if row['tool'] == 'score_ecgfounder':
        result = {key: result.get(key) for key in ('af_score', 'afl_score')}
    elif row['tool'] == 'assess_quality':
        result = {key: result.get(key) for key in ('score', 'reasons')}
    return {'evidence_id': row['evidence_id'], 'tool': row['tool'], 'lead': row['lead'],
            'window_s': row['window_s'], 'status': row['status'], 'result': result}


def prompt_for(cursor, trace, evidence):
    history = [row for row in trace if row['end_s'] <= cursor][-6:]
    if not history:
        raise ValueError('No arrived windows at cursor')
    first = history[0]['start_s']
    selected = [evidence_view(row) for row in evidence
                if first <= row['window_s'][0] and row['available_at_s'] <= cursor]
    return {'cursor_s': cursor, 'recent_windows': [
        {key: row.get(key) for key in ('start_s', 'end_s', 'af_score', 'afl_score')}
        for row in history], 'measured_evidence': selected}


def evaluate_case(model, cursor, trace, evidence):
    view = prompt_for(cursor, trace, evidence)
    messages = [{'role': 'system', 'content': (
        'An AF/AFL event is currently confirmed and active. The recording arrives online. '
        'Choose exactly one action. CLOSE only if already arrived evidence supports a rhythm '
        'transition, give the estimated end time and cite actual evidence IDs. CONTINUE if '
        'AF/AFL remains supported. RECHECK if the evidence conflicts or is insufficient. '
        'Cursor time is the decision time, not automatically the event end time. '
        'Use only supplied evidence. Do not invent measurements or IDs.')},
        {'role': 'user', 'content': json.dumps(view, allow_nan=False)}]
    started = time.perf_counter()
    try:
        reply = model.complete(messages, [CHOICES])
        calls = reply.message.get('tool_calls', [])
        if len(calls) != 1 or calls[0]['function']['name'] != 'judge_event_end':
            raise ValueError('Expected one judgment tool call')
        raw = calls[0]['function']['arguments']
        answer = json.loads(raw) if isinstance(raw, str) else raw
        if set(answer) != {'action', 'end_s', 'evidence_ids', 'reason'}:
            raise ValueError('Invalid judgment fields')
        if answer['action'] not in ('continue', 'recheck', 'close'):
            raise ValueError('Unknown action')
        available = {item['evidence_id'] for item in view['measured_evidence']}
        if not isinstance(answer['evidence_ids'], list) or not all(
                isinstance(item, str) for item in answer['evidence_ids']):
            raise ValueError('Invalid evidence IDs')
        if not set(answer['evidence_ids']) <= available:
            raise ValueError('Evidence ID absent at decision time')
        if answer['action'] == 'close':
            if (not isinstance(answer['end_s'], (int, float)) or
                    not math.isfinite(answer['end_s']) or
                    not view['recent_windows'][0]['start_s'] < answer['end_s'] <= cursor):
                raise ValueError('End outside observed history')
            if not answer['evidence_ids']:
                raise ValueError('Close needs evidence')
        elif answer['end_s'] is not None:
            raise ValueError('Non-close action must have null end_s')
        return {'cursor_s': cursor, 'answer': answer, 'usage': reply.usage,
                'elapsed_s': round(time.perf_counter() - started, 3),
                'evidence_count': len(view['measured_evidence'])}
    except Exception as exc:
        return {'cursor_s': cursor, 'error': type(exc).__name__,
                'error_detail': str(exc)[:160], 'elapsed_s': round(time.perf_counter() - started, 3),
                'evidence_count': len(view['measured_evidence'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='One frozen rule-run record directory')
    parser.add_argument('--llm-config', type=Path, required=True)
    parser.add_argument('--cursors', type=float, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    trace = read_rows(args.source/'window_trace.jsonl')
    evidence = read_rows(args.source/'evidence.jsonl')
    model = build_chat_model(json.loads(args.llm_config.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open('w', encoding='utf-8') as out:
        for cursor in args.cursors:
            row = evaluate_case(model, cursor, trace, evidence)
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False)+'\n')
            out.flush()
            print(json.dumps({'cursor_s': cursor, 'action': row.get('answer', {}).get('action'),
                              'error': row.get('error'), 'elapsed_s': row['elapsed_s']}), flush=True)
    manifest = {'source': str(args.source), 'source_sha256': {
        name: hashlib.sha256((args.source/name).read_bytes()).hexdigest()
        for name in ('window_trace.jsonl', 'evidence.jsonl')},
        'llm_config': json.loads(args.llm_config.read_text(encoding='utf-8')),
        'cursors': args.cursors}
    args.output.with_suffix('.manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'cases': len(rows),
                      'actions': [row.get('answer', {}).get('action') for row in rows],
                      'errors': [row.get('error') for row in rows if 'error' in row]}))


if __name__ == '__main__':
    main()
