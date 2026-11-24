"""Use the same arrived measurements to isolate the order of the two review actions."""
import argparse
import gzip
import json
from pathlib import Path

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.llm.providers import LocalQwenModel
from cardioagent.ecg_agent.llm.recording import RecordingChatModel


ARMS = ('archived', 'tool_order', 'instruction_order', 'both_orders')
NAMES = ['assess_window', 'request_more_evidence']
MARKER = ' Submit assess_window when you can judge the current segment'


def review_prompt(messages, tools, arm):
    if [tool['function']['name'] for tool in tools] != NAMES:
        raise ValueError('Expected the archived dual-action review schema')
    if len(messages) != 2 or [message['role'] for message in messages] != ['system','user']:
        raise ValueError('Only the first review prompt can enter the comparison')
    if arm in ('instruction_order','both_orders'):
        earlier, marker, acquisition = messages[0]['content'].partition(MARKER)
        if not marker:
            raise ValueError('Cannot isolate the acquisition instruction')
        messages = [dict(messages[0],content=(marker+acquisition).strip()+'\n\n'+earlier),messages[1]]
    if arm in ('tool_order','both_orders'):
        tools = tools[::-1]
    return messages, tools


def classify(reply, tools):
    calls = reply.message.get('tool_calls',[])
    if len(calls)!=1:
        raise ValueError('Expected one review action')
    fn = calls[0]['function']
    args = json.loads(fn['arguments'])
    name = fn['name']
    schema = next((tool['function']['parameters'] for tool in tools if tool['function']['name']==name),None)
    if (not isinstance(args,dict) or schema is None or set(args)!=set(schema['required']) or
            not isinstance(args['reason'],str) or not args['reason'].strip()):
        raise ValueError('Invalid review action or missing reason')
    fields = schema['properties']
    if name == 'request_more_evidence':
        if args['option'] not in fields['option']['enum']:
            raise ValueError('Requested an unavailable measurement')
    elif (args['finding'] not in fields['finding']['enum'] or not isinstance(args['evidence_ids'],list) or
          any(key not in fields['evidence_ids']['items']['enum'] for key in args['evidence_ids'])):
        raise ValueError('Invalid assessment label or unshown evidence ID')
    return dict(action=name,arguments=args)


def replay(args):
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data = json.loads(args.input.read_bytes())
    cases = data[args.shard::args.shards]
    model = LocalQwenModel(dict(json.loads(args.config.read_bytes()),device='cuda:0'))
    args.output.mkdir(parents=True,exist_ok=True)
    warm = cases[0]
    try:
        model.complete(warm['messages'],warm['tools'])
    except ValueError:
        pass
    fingerprint = dict(input_sha256=digest(args.input),config_sha256=digest(args.config),
                       shard=args.shard,shards=args.shards,gpu=torch.cuda.get_device_name(0))
    for index, case in enumerate(cases):
        key = case['id']
        if not key.replace('_','').isalnum() or set(case)!={'id','messages','tools'}:
            raise ValueError('Unexpected fields or case name in the frozen input')
        receipt = args.output/(key+'.json')
        raw = args.output/(key+'.jsonl.gz')
        if receipt.is_file():
            saved = json.loads(receipt.read_bytes())
            if saved['job']!=fingerprint or saved['log_sha256']!=digest(raw):
                raise ValueError('Cannot reuse a previous prompt replay')
            continue
        log = raw.with_suffix('.partial')
        log.unlink(missing_ok=True)
        recorder = RecordingChatModel(model,log)
        pivot = (index+args.shard)%len(ARMS)
        order = ARMS[pivot:]+ARMS[:pivot]
        outcomes = {}
        for arm in order:
            messages, tools = review_prompt(case['messages'],case['tools'],arm)
            try:
                reply = recorder.complete(messages,tools)
                outcomes[arm] = classify(reply,tools)
            except ValueError as error:
                outcomes[arm] = {'error':str(error)}
        with gzip.open(log,'rt',encoding='utf-8') as stream:
            records = [json.loads(row) for row in stream]
        for arm,record in zip(order,records,strict=True):
            outcomes[arm].update(elapsed_s=record['elapsed_s'],usage=record.get('usage'))
        log.replace(raw)
        tmp = receipt.with_suffix('.partial')
        save_json(tmp,dict(id=key,job=fingerprint,order=order,arms=outcomes,log_sha256=digest(raw)))
        tmp.replace(receipt)
        print(key,{arm: outcome.get('action',outcome.get('error')) for arm,outcome in outcomes.items()},flush=True)
    save_json(args.output/'complete.json',dict(job=fingerprint,cases=len(cases)))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    options = p.parse_args()
    if not 0<=options.shard<options.shards:
        p.error('Invalid shard')
    replay(options)
