"""Audit exact LTAFDB reference-boundary transfer and event memory across all records."""
import argparse
from collections import Counter
import json
from math import ceil, floor, isfinite
from pathlib import Path
from time import perf_counter

from cardioagent.common.io import digest, save_json, write_jsonl
from cardioagent.ecg_agent.harness.staged_review import schema
from cardioagent.ecg_agent.llm.providers import LocalQwenModel, parse_qwen_tool_reply
from cardioagent.ecg_agent.memory.autonomous_events import AutonomousEvents
from cardioagent.ecg_agent.memory.evidence import EvidenceLedger, canonical_hash
from cardioagent.ecg_agent.schemas import EvidenceContext


TOOLS=[schema('store_reference_boundary','Submit the exact annotated event boundary.',
              dict(time_s=dict(type='number'),reference_id=dict(type='string')),['time_s','reference_id'])]
PROMPT=('Record an exact annotated LTAFDB boundary for an AF/AFL event. '
        'Copy reference_evidence.time_s into time_s and reference_evidence.id into reference_id. '
        'The currently available time is the clock, and must never replace the annotated time. '
        'Do not round. Submit the offered tool with only these two fields.')


def atomic_json(path,data):
    temp=path.with_suffix('.partial')
    save_json(temp,data)
    temp.replace(path)


def boundary_targets(case):
    previous=case['start_s']
    points=[]
    for ordinal,interval in enumerate(case['intervals'],1):
        a,b=interval
        if not previous<=a<b<=case['end_s'] or not all(isfinite(t) for t in interval):
            raise ValueError('Nonfinite, overlapping, unordered or clipped reference coordinates')
        before=min((floor(a/10)+1)*10,case['end_s'])
        after=min(ceil(b/10)*10,case['end_s'])
        for boundary,observed,value in (('onset',before,a),('offset',after,b)):
            evidence=dict(id='REF_START' if boundary=='onset' else 'REF_END',time_s=value,
                          available_interval_s=[a,None] if boundary=='onset' else [a,b])
            points.append(dict(ordinal=ordinal,boundary=boundary,available_through_s=observed,
                               reference_evidence=evidence))
        previous=b
    return points


class BatchQwen:
    def __init__(self,config,batch_size):
        self.adapter=LocalQwenModel(dict(config,device='cuda:0'))
        self.adapter.tokenizer.padding_side='left'
        self.limit=batch_size

    def generate(self, targets, retry=False):
        import torch

        if not targets:
            return []
        if len(targets)>self.limit:
            return [r for i in range(0,len(targets),self.limit) for r in self.generate(targets[i:i+self.limit],retry)]
        system=PROMPT+(' Your previous proposal failed validation. Read the displayed reference fields again.' if retry else '')
        tokenizer=self.adapter.tokenizer
        rendered=[tokenizer.apply_chat_template(
            [dict(role='system',content=system),dict(role='user',content=json.dumps(t,allow_nan=False))],
            tools=TOOLS,tokenize=False,add_generation_prompt=True,enable_thinking=False) for t in targets]
        try:
            return self.forward(rendered)
        except torch.cuda.OutOfMemoryError:
            if len(targets)==1:
                raise
        self.limit=max(1,min(self.limit,len(targets)//2))
        torch.cuda.empty_cache()
        return self.generate(targets,retry)

    def forward(self, rendered):
        import torch

        t=self.adapter.tokenizer
        inputs=t(rendered,padding=True,add_special_tokens=False,return_tensors='pt').to(self.adapter.device)
        with torch.inference_mode():
            generated=self.adapter.model.generate(**inputs,do_sample=False,max_new_tokens=96,
                                                   pad_token_id=t.pad_token_id)
        return t.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)


def validate(raw,target):
    try:
        message=parse_qwen_tool_reply(raw)
        fn=message['tool_calls'][0]['function']
        args=json.loads(fn['arguments'])
        expected=target['reference_evidence']
        if fn['name']!='store_reference_boundary' or set(args)!={'time_s','reference_id'}:
            raise ValueError('Unexpected function or arguments')
        if args['reference_id']!=expected['id']:
            raise ValueError('Incorrect reference ID')
        time=args['time_s']
        if type(time) not in (int,float) or not isfinite(time) or time>target['available_through_s']:
            raise ValueError('Invalid or unarrived time')
        if time!=expected['time_s']:
            raise ValueError('Time differs from reference')
        return dict(raw=raw,validated_time_s=time)
    except (ValueError,TypeError,KeyError) as exc:
        return dict(raw=raw,error=str(exc))


def review(model,points):
    rows=[dict(target=p,attempts=[validate(raw,p)]) for p,raw in zip(points,model.generate(points),strict=True)]
    failed=[row for row in rows if 'error' in row['attempts'][0]]
    for row,raw in zip(failed,model.generate([row['target'] for row in failed],retry=True),strict=True):
        row['attempts'].append(validate(raw,row['target']))
    return rows


def write_events(case,outcomes,reference_sha):
    clock=AutonomousEvents('fixed',case['start_s'])
    evidence=EvidenceLedger(EvidenceContext('ltafdb-full-reference-diagnostic',case['task_id'],
                                            case['record_id'],reference_sha))
    counts=Counter(events=len(case['intervals']))
    journal, events=[],[]
    for i in range(0,len(outcomes),2):
        pair=outcomes[i:i+2]
        for row in pair:
            key=row['target']['boundary']
            counts[key+'_first_correct']+=int('validated_time_s' in row['attempts'][0])
            counts[key+'_eventual_correct']+=int('validated_time_s' in row['attempts'][-1])
            counts['model_calls']+=len(row['attempts'])
        if any('validated_time_s' not in row['attempts'][-1] for row in pair):
            continue
        onset,offset=(row['attempts'][-1]['validated_time_s'] for row in pair)
        for idx,row in enumerate(pair):
            target=row['target']
            time=row['attempts'][-1]['validated_time_s']
            received=target['available_through_s']
            raw=dict(result=target['reference_evidence'],available_at_s=received,
                     window_s=None,tool='diagnostic_reference',lead=None)
            key='annotation:'+canonical_hash([case['record_id'],target['ordinal'],target['boundary'],raw])
            evidence.append(dict(raw,evidence_id=key),input_fingerprint=reference_sha)
            evidence.assessment('reference_boundary','Validated model point submission',[key],received,checked=True)
            # Apply the existing event writer on the known episode timeline. The recorded-at field
            # below keeps the diagnostic availability clock separate from those historical coordinates.
            tick=(onset+offset)/2 if idx==0 else offset
            update=clock.apply(clock.cursor,tick,dict(event_action='OPEN' if idx==0 else 'CLOSE',
                boundary=dict(kind='localized',time_s=time,evidence_id=key,source='LTAFDB_reference_diagnostic'),
                pending=None,next_review_s=None,support_intervals=[],opposing_intervals=[],evidence_ids=[key],
                reason='Qwen submitted the validated reference boundary'))['event_update']
            if idx==0:
                clock.active.update(confirmed_at_s=received,first_seen_at_s=received)
                update['interval'][1]=received
            else:
                clock.events[-1]['end_decided_at_s']=received
            journal.append(dict(update,recorded_at_s=received,reference_ordinal=target['ordinal']))
        event,=clock.retrieve(onset,offset,available_through_s=case['end_s'],
                              limit=1,event_id=journal[-1]['event_id'])
        if [event['estimated_start_s'],event['estimated_end_s']]!=case['intervals'][i//2]:
            raise ValueError('Persisted event did not read back exactly')
        if any(evidence.get(event[k]['evidence_id'])['result']['time_s']!=event[k]['time_s']
               for k in ('onset_boundary','end_boundary')):
            raise ValueError('Persisted reference link did not resolve')
        events.append(dict(event,reference_ordinal=i//2+1))
    counts['events_read_back']=len(events)
    return dict(counts),events,journal


def run(args):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    packet=json.loads(args.input.read_bytes())
    config=json.loads(args.config.read_bytes())
    job=dict(input_sha256=digest(args.input),config_sha256=digest(args.config),
             reference_sha256=packet['reference_sha256'],batch_size=args.batch_size,protocol='paired-reference-point-v1')
    args.output.mkdir(parents=True,exist_ok=True)
    model=None
    for case in packet['cases']:
        record=case['record_id']
        if not record.isdecimal():
            raise ValueError('Invalid record')
        dest=args.output/record
        dest.mkdir(exist_ok=True)
        end=dest/'result.json'
        if end.exists():
            result=json.loads(end.read_bytes())
            if result['job']!=job or any(digest(dest/name)!=sha for name,sha in result['artifacts'].items()):
                raise ValueError('Record outputs are not reusable')
            print(record,result['counts'],'verified',flush=True)
            continue
        points=boundary_targets(case)
        results=[]
        for index in range(0,len(points),args.batch_size):
            batch=points[index:index+args.batch_size]
            sealed=dest/f'batch-{index:05d}.json'
            if sealed.exists():
                ready=json.loads(sealed.read_bytes())
                if ready['job']!=job or [r['target'] for r in ready['points']]!=batch:
                    raise ValueError('Wrong cached stage')
            else:
                if model is None:
                    model=BatchQwen(config,args.batch_size)
                start=perf_counter()
                ready=dict(job=job,points=review(model,batch),elapsed_s=perf_counter()-start)
                atomic_json(sealed,ready)
            results.extend(ready['points'])
            print(record,f'{len(results)}/{len(points)}',round(ready['elapsed_s'],2),flush=True)
        counts,events,journal=write_events(case,results,job['reference_sha256'])
        write_jsonl(dest/'events.jsonl',events)
        write_jsonl(dest/'event_updates.jsonl',journal)
        write_jsonl(dest/'qwen_boundary_calls.jsonl',results)
        artifacts={file:digest(dest/file) for file in ('events.jsonl','event_updates.jsonl','qwen_boundary_calls.jsonl')}
        atomic_json(end,dict(job=job,record_id=record,counts=counts,artifacts=artifacts))
        print(record,counts,flush=True)
    atomic_json(args.output/'complete.json',dict(job=job,records=[case['record_id'] for case in packet['cases']]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=24)
    a=parser.parse_args()
    if not 2<=a.batch_size<=40 or a.batch_size%2:
        parser.error('Batch size must be even and between 2 and 40')
    run(a)
