"""Paired first-response test of a window-only reference display on frozen arrived measurements."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path

from cardioagent.ecg_agent.harness.window_semantics import direction_only_view
from cardioagent.ecg_agent.llm.providers import build_chat_model
from cardioagent.ecg_agent.llm.recording import RecordingChatModel


POINTS = [('M-04936-7900', t) for t in (7920, 7930, 7940, 7950)]
POINTS += [('M-07162-4300', 4330), ('M-04936-1770', 1810)]


def run(root, config, destination):
    destination.mkdir(exist_ok=False)
    model = build_chat_model(json.loads(config.read_text(encoding='utf-8')))
    with (destination/'results.jsonl').open('x', encoding='utf-8') as output:
        for i, (name, cursor) in enumerate(POINTS):
            path, = (root/name).glob('output/*/records/*/qwen_calls.jsonl.gz')
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                archived = next(r for r in map(json.loads, stream) if r['cursor_s'] == cursor
                                and len(r['messages']) == 2 and r['tools'][0]['function']['name'] == 'assess_window')
            for arm in (('original','direction') if i%2==0 else ('direction','original')):
                messages = deepcopy(archived['messages'])
                if arm == 'direction':
                    data, prompt = direction_only_view(json.loads(messages[1]['content']), messages[0]['content'])
                    messages[0]['content'] = prompt
                    messages[1]['content'] = json.dumps(data, allow_nan=False)
                rec = RecordingChatModel(model, destination/(arm+'-qwen_calls.jsonl.gz'))
                record = dict(clip=name, cursor_s=cursor, arm=arm)
                try:
                    reply = rec.complete(messages, archived['tools'])
                    call, = reply.message['tool_calls']
                    args = json.loads(call['function']['arguments'])
                    data = json.loads(messages[1]['content'])
                    choices = archived['tools'][0]['function']['parameters']['properties']['finding']['enum']
                    visible = {r['id'] for k in ('vote_on_this_interval','earlier_measurements_for_comparison_only',
                                                  'other_measurements') for r in data[k]}
                    record.update(args, raw_reproduced=rec.last_raw==archived['raw'],
                                  valid_fields=call['function']['name']=='assess_window' and args.get('finding') in choices
                                  and isinstance(args.get('reason'),str) and bool(args['reason'].strip())
                                  and isinstance(args.get('evidence_ids'),list) and set(args['evidence_ids']) <= visible)
                except (ValueError, KeyError, TypeError) as exc:
                    record.update(valid_fields=False, error=str(exc))
                output.write(json.dumps(record, allow_nan=False)+'\n')
                output.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('root','config','destination'):
        parser.add_argument('--'+arg, required=True, type=Path)
    run(**vars(parser.parse_args()))
