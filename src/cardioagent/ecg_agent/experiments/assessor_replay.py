"""Replay previously observed assessor prompts, isolating historical evidence labels."""
import argparse
import json
from pathlib import Path

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.llm.providers import LocalQwenModel
from cardioagent.ecg_agent.llm.recording import RecordingChatModel
from cardioagent.ecg_agent.memory.evidence import canonical_hash


def run(args):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model=LocalQwenModel(dict(json.loads(args.config.read_bytes()),device='cuda:0'))
    cases=json.loads(args.input.read_bytes())
    args.output.mkdir(parents=True,exist_ok=True)
    result={}
    for entry in cases:
        if set(entry)!={'id','messages','tools'} or not entry['id'].replace('_','').isalnum():
            raise ValueError('Invalid recorded case')
        if [t['function']['name'] for t in entry['tools']]!=['assess_window']:
            raise ValueError('Accept only an ECG window assessment')
        log=args.output/(entry['id']+'.jsonl.gz')
        if log.exists(): raise ValueError('Use a new replay directory')
        recorder=RecordingChatModel(model,log)
        try:
            reply=recorder.complete(entry['messages'],entry['tools'])
            tool=reply.message['tool_calls'][0]['function']
            final=dict(assessment=json.loads(tool['arguments']),usage=reply.usage)
        except ValueError as err:
            final=dict(error=str(err))
        result[entry['id']]=dict(prompt_sha256=canonical_hash([entry['messages'],entry['tools']]),
                                 log_sha256=digest(log),**final)
        print(entry['id'],final.get('assessment',{}).get('finding'),flush=True)
    save_json(args.output/'complete.json',dict(input_sha256=digest(args.input),config_sha256=digest(args.config),
                                               cases=result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','config','output'):
        p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
