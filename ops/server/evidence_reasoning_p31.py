"""P3-1 paired evidence reading and controlled score interventions."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from cardioagent.ecg_agent.llm.providers import build_chat_model

ARMS = ('raw', 'aligned', 'reasoned')
SCHEMA = {'type': 'function', 'function': {'name': 'judge_transition',
    'description': 'Judge whether the already active AF/AFL episode has ended.',
    'parameters': {'type': 'object', 'properties': {
        'decision': {'type': 'string', 'enum': ['AF_AFL_ONGOING', 'AF_AFL_ENDED', 'UNCERTAIN']},
        'support_transition_ids': {'type': 'array', 'items': {'type': 'string'}},
        'against_transition_ids': {'type': 'array', 'items': {'type': 'string'}},
        'unresolved': {'type': 'string'},
        'reason': {'type': 'string'}},
        'required': ['decision', 'support_transition_ids', 'against_transition_ids',
                     'unresolved', 'reason'], 'additionalProperties': False}}}
SYSTEM = ('An AF/AFL episode was previously confirmed and is currently open. The sole task '
          'is to decide whether this existing AF/AFL episode has ENDED by the current cursor. '
          'AF_AFL_ONGOING means current evidence supports continuing AF or AFL. '
          'AF_AFL_ENDED means current evidence supports termination of AF/AFL. '
          'A rise in AF or AFL scores supports ONGOING, never ENDED. '
          'UNCERTAIN means current evidence does not settle whether it ended. '
          'Inspect only supplied, already arrived ECGFounder measurements. Scores are '
          'uncalibrated and correlated across leads. A low score alone is not proof of '
          'termination. List IDs supporting and opposing ENDING. Past high scores alone '
          'cannot prove current AF. Never infer missing measurements.')


def measurements(case):
    return [copy.deepcopy(item) for item in case['primary'] + case['backup']]


def view(items, arm):
    if arm == 'raw':
        return {'measurements': items}
    if arm in ('aligned', 'reasoned'):
        starts = sorted({tuple(item['window_s']) for item in items if item.get('window_s')})
        return {'windows': [{'window_s': list(span), 'measurements': [item for item in items
                if item.get('window_s') == list(span)]} for span in starts],
                'unmeasured': [item for item in items if not item.get('window_s')]}
    raise ValueError(arm)


def validate(reply, available):
    calls = reply.message.get('tool_calls', [])
    if len(calls) != 1 or calls[0]['function']['name'] != 'judge_transition':
        raise ValueError('Expected one judgment')
    raw = calls[0]['function']['arguments']
    answer = json.loads(raw) if isinstance(raw, str) else raw
    if set(answer) != {'decision', 'support_transition_ids', 'against_transition_ids',
                       'unresolved', 'reason'} or answer['decision'] not in (
                       'AF_AFL_ONGOING', 'AF_AFL_ENDED', 'UNCERTAIN'):
        raise ValueError('Invalid answer')
    for field in ('support_transition_ids', 'against_transition_ids'):
        ids = answer[field]
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or not set(ids) <= available:
            raise ValueError('Invalid evidence IDs')
    if any(not isinstance(answer[field], str) for field in ('unresolved', 'reason')):
        raise ValueError('Invalid explanation')
    return answer


def judge(model, items, arm, cursor):
    available = {item['id'] for item in items if item['status'] == 'ok'}
    instruction = (' First identify support for and against a transition, then judge. '
                   'Check score recovery and simultaneous lead disagreement.' if arm == 'reasoned' else '')
    messages = [{'role': 'system', 'content': SYSTEM + instruction},
                {'role': 'user', 'content': json.dumps({'cursor_s': cursor,
                    'view': view(items, arm)}, allow_nan=False) +
                    '\nValid evidence IDs: ' + ', '.join(sorted(available))}]
    started = time.perf_counter()
    for attempt in range(2):
        try:
            answer = validate(model.complete(messages, [SCHEMA]), available)
            return {'answer': answer, 'attempts': attempt + 1,
                    'elapsed_s': round(time.perf_counter() - started, 3)}
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)[:140]
            messages.append({'role': 'user', 'content': 'Invalid output. ' + error +
                             '. Submit one valid judgment using only listed IDs.'})
    return {'error': error, 'attempts': 2,
            'elapsed_s': round(time.perf_counter() - started, 3)}


def intervention(items, kind):
    altered = copy.deepcopy(items)
    if kind == 'order':
        altered.reverse()
        return altered, {'kind': kind, 'changed_id': None}
    target = ('E12' if kind == 'backup' else 'E5')
    item = next((x for x in altered if x['id'] == target and x['status'] == 'ok'), None)
    if item is None:
        return None, {'kind': kind, 'skipped': 'target_unavailable'}
    if kind == 'recovery':
        previous = next((x for x in altered if x['id'] == 'E4'), None)
        current = next((x for x in altered if x['id'] == 'E6'), None)
        if not previous or not current or previous['status'] != 'ok' or current['status'] != 'ok':
            return None, {'kind': kind, 'skipped': 'neighbor_unavailable'}
        if max(previous['result'].values()) >= .5 or max(current['result'].values()) >= .5:
            return None, {'kind': kind, 'skipped': 'not_a_low_high_low_pattern'}
    before = copy.deepcopy(item['result'])
    old = max(before['af_score'], before['afl_score'])
    new = .9 if old < .5 else .1
    item['result']['af_score'] = new
    item['result']['afl_score'] = min(new, .1)
    return altered, {'kind': kind, 'changed_id': target, 'before': before,
                     'after': item['result'], 'direction': 'higher' if new > old else 'lower'}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_chat_model({'provider': 'transformers', 'model_path': str(args.model_path),
                              'device': 'cuda:0', 'max_new_tokens': 512})
    for group in ('main', 'hard'):
        case_file = args.cases / group / 'cases.jsonl'
        source_sha = hashlib.sha256(case_file.read_bytes()).hexdigest()
        for line in case_file.read_text(encoding='utf-8').splitlines():
            case = json.loads(line)
            if case['lead'] != args.lead:
                continue
            items = measurements(case)
            base = {'record_id': case['record_id'], 'lead': args.lead,
                    'cursor_s': case['cursor_s'], 'reference': case['reference'],
                    'source_sha256': source_sha, 'group': group}
            for arm in ARMS:
                result = {**base, 'arm': arm, **judge(model, items, arm, case['cursor_s'])}
                with (args.output / f'{group}-{arm}-{args.lead}.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            for kind in ('backup', 'recovery', 'order'):
                changed, change = intervention(items, kind)
                if changed is None:
                    continue
                result = {**base, 'arm': 'counterfactual', 'intervention': change,
                          **judge(model, changed, 'reasoned', case['cursor_s'])}
                with (args.output / f'{group}-counterfactual-{args.lead}.jsonl').open(
                        'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(json.dumps({'group': group, 'record': case['record_id'],
                              'lead': args.lead, 'cursor': case['cursor_s']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--lead', choices=('ecg1', 'ecg2'), required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
