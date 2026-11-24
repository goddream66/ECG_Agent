"""Replay the same recorded Qwen prompts with and without a bounded prompt prefix."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.llm.providers import LocalQwenModel


def run(args):
    import torch
    import transformers
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    items=json.loads(args.input.read_bytes())[args.shard::args.shards]
    if not items:
        raise ValueError('Empty benchmark shard')
    model=LocalQwenModel(dict(json.loads(args.config.read_bytes()),device='cuda:0'))
    identity=dict(input_sha256=digest(args.input),config_sha256=digest(args.config),
                  torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),
                  limit=args.limit,shard=args.shard,shards=args.shards)
    args.output.mkdir(parents=True,exist_ok=True)
    warm=items[0]['turns'][0]
    try:
        model.complete(warm['messages'],warm['tools'])
    except ValueError:
        pass
    for i,item in enumerate(items):
        key=item['case']
        if not all(c.isalnum() or c=='_' for c in key):
            raise ValueError('Unexpected fixed case name')
        dest=args.output/(key+'.json')
        if dest.exists():
            if json.loads(dest.read_bytes())['job']!=identity:
                raise ValueError('Incompatible prior result')
            continue
        results={}
        order=('baseline','cached') if i%2==0 else ('cached','baseline')
        for label in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            calls=[]
            cap=args.limit if label=='cached' else 0
            with model.prefix_scope(limit=cap):
                for turn in item['turns']:
                    model.last_raw=None
                    start=perf_counter()
                    try:
                        out=model.complete(turn['messages'],turn['tools'])
                        action=out.message['tool_calls'][0]['function']
                        result=dict(function=action['name'],arguments=json.loads(action['arguments']),
                                    usage=out.usage)
                    except ValueError as error:
                        result=dict(error=str(error))
                    torch.cuda.synchronize()
                    result.update(elapsed_s=perf_counter()-start,
                                  raw_sha256=sha256((model.last_raw or '').encode()).hexdigest())
                    calls.append(result)
            results[label]=dict(calls=calls,max_allocated_bytes=torch.cuda.max_memory_allocated())
        tmp=dest.with_suffix('.partial')
        save_json(tmp,dict(job=identity,case=key,order=order,arms=results))
        tmp.replace(dest)
        print(key,len(item['turns']),flush=True)
    save_json(args.output/'complete.json',dict(job=identity,cases=len(items)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--limit',type=int,default=3072)
    a=parser.parse_args()
    if not 0<=a.shard<a.shards or not 128<=a.limit<=4096:
        parser.error('Invalid shard or cache budget')
    run(a)
