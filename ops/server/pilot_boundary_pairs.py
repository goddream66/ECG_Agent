"""Pair the first boundary responses on identical archived tool evidence and one GPU."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from cardioagent.ecg_agent.llm.providers import build_chat_model
from cardioagent.ecg_agent.llm.recording import RecordingChatModel


def boundary_calls(root):
    path, = root.glob('output/*/records/*/qwen_calls.jsonl.gz')
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        for row in map(json.loads, stream):
            if len(row['messages']) == 2 and row['tools'][0]['function']['name'] == 'refine_boundary':
                if json.loads(row['messages'][1]['content'])['event_action'] == 'REQUEST_CLOSE':
                    yield row


def run(root, config):
    dest=root/'paired-boundary'
    dest.mkdir(exist_ok=False)
    model=build_chat_model(json.loads(config.read_text()))
    revised=next(boundary_calls(root/'bracket-live-04936-1770'))['messages'][0]['content']
    points=[('04936-1770',1800),('04936-1770',1810),('04015-640',680),
            ('06426-4930',4960),('04936-7900',7930)]
    with (dest/'results.jsonl').open('x') as out:
        for clip,cursor in points:
            frozen=next(c for c in boundary_calls(root/('ref-live-'+clip)) if c['cursor_s']==cursor)
            for arm in ('original','range'):
                messages=deepcopy(frozen['messages'])
                if arm=='range':
                    assert revised.startswith(messages[0]['content']+' ')
                    messages[0]['content']=revised
                digest=hashlib.sha256(json.dumps([messages[1:],frozen['tools']],sort_keys=True).encode()).hexdigest()
                recorder=RecordingChatModel(model,dest/(arm+'-calls.jsonl.gz'))
                reply=recorder.complete(messages,frozen['tools'])
                calls=reply.message.get('tool_calls',[])
                item=dict(clip=clip,cursor_s=cursor,arm=arm,input_digest=digest,
                          identical_to_archive=recorder.last_raw==frozen['raw'],message=reply.message)
                if len(calls)==1 and calls[0]['function']['name']=='refine_boundary':
                    args=calls[0]['function']['arguments']
                    item['boundary_proposal']=json.loads(args) if isinstance(args,str) else args
                out.write(json.dumps(item)+'\n')
                out.flush()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','config'):
        parser.add_argument('--'+name,required=True,type=Path)
    run(**vars(parser.parse_args()))
