"""Fork an observed event ledger and replay the real Qwen lifecycle on identical frozen observations."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.harness.end_review import advance_end_memory, end_evidence
from cardioagent.ecg_agent.harness.staged_review import choose_action, confirm_end_with_evidence
from cardioagent.ecg_agent.llm.providers import LocalQwenModel
from cardioagent.ecg_agent.llm.recording import RecordingChatModel
from cardioagent.ecg_agent.memory.evidence import canonical_hash
from cardioagent.ecg_agent.memory.staged_events import StagedEvents


def restore_prefix(rows, cursor):
    tracker=StagedEvents('fixed',0.)
    for row in rows:
        a,b=row['start_s'],row['cursor_s']
        if b>cursor:
            break
        if tracker.cursor!=a:
            raise ValueError('Incomplete shared prefix')
        if row['decision'] in ('ACTION_ERROR','UNREVIEWED'):
            tracker.skip(a,b,retry=row['decision']=='ACTION_ERROR')
        else:
            tracker.apply(a,b,deepcopy(row['llm_action']))
    if tracker.cursor!=cursor:
        raise ValueError('The recorded journal misses the fork cursor')
    return tracker


def active_view(tracker, state):
    view=state['open_event']
    if view:
        for field in ('confirmed_support_intervals','opposing_intervals','unknown_intervals'):
            view[field]=deepcopy(tracker.active[field][-3:])
    return view


def memory_proof(memory):
    proof={'can_confirm':memory.get('later_against') is not None}
    aliases={}
    for role,key in (('before','support'),('first_against','first_against'),('later_against','later_against')):
        row=memory.get(key)
        proof[role]=None
        if row:
            alias='recorded_'+role
            proof[role]=dict(id=alias,window_s=row['window_s'],lead=row['lead'],
                             direction=row['direction'],target_score=row['target_score'])
            aliases[alias]=row['evidence_id']
    return proof,aliases


def arrived(row, span):
    if (row['window_s']!=span or row['available_at_s']>span[1] or
            (row['reference']['direction'] not in ('support','negative','indeterminate'))):
        raise ValueError('Unavailable or invalid frozen measurement')
    return dict(evidence_id=row['evidence_id'],window_s=span,lead=row['lead'],
                direction=row['reference']['direction'],target_score=row['target_score'])


def replay(seed, case, model, emit, persist):
    tracker=deepcopy(seed)
    counts=Counter()
    memory={}
    for row in case['warm']:
        span=row['window_s']
        if span[1]>tracker.cursor or not tracker.active or span[0]<tracker.active['estimated_start_s']:
            raise ValueError('Warm observations must precede the fork and belong to the open event')
        memory=advance_end_memory(memory,span,[arrived(r,span) for r in row['current']])
    writes=[]
    for index,row in enumerate(case['suffix']):
        span=row['window_s']
        a,b=span
        if a!=tracker.cursor or b-a!=10:
            raise ValueError('The continuation must be contiguous')
        state=tracker.snapshot(a,b,index==len(case['suffix'])-1)
        obs=deepcopy(row['observation'])
        active=active_view(tracker,state)
        timeline=row['timeline']
        if any(r['available_at_s']>b or r['window_s'][1]>b for r in timeline):
            raise ValueError('Future measurement in recorded continuation')
        aliases={r['id']:r['evidence_id'] for r in timeline}
        visible=[{k:v for k,v in r.items() if k not in ('available_at_s','evidence_id')} for r in timeline]
        current=[arrived(r,span) for r in timeline if r['window_s']==span]
        if not active:
            memory={}
        memory=advance_end_memory(memory,span,current)
        transition=dict(rhythm_by_time=visible,pending_question=state['pending_question'])
        if active:
            if persist:
                proof,ids=memory_proof(memory)
                aliases.update(ids)
            else:
                proof=end_evidence(visible,span,active)
            transition['end_evidence']=proof
        audit=dict(cursor_s=b)
        if index==0:
            # Both arms retain the real archived request so any verdict change is attributable to the gate.
            intent=row['fork_intent']
            if not active or intent['event_id']!=active['event_id'] or intent['action']!='REQUEST_CLOSE':
                raise ValueError('The common fork must begin at the existing close request')
        else:
            intent=choose_action(model,counts,emit,audit,span,obs,active,state['candidate'],transition,True)
        action=intent.get('action')
        refine=dict(status='not_requested')
        if action=='REQUEST_CLOSE':
            view=dict(window_s=span,open_event=active,assessment=obs,**transition)
            refine=confirm_end_with_evidence(model,counts,emit,audit,view,aliases)
        pending=None
        if action=='REQUEST_CLOSE' and refine['status']!='located':
            pending=dict(question='Did the combined target end? Identify a measured before/after pair.',
                         needed_evidence='Earlier target support and later contrary evidence on arrived ECG')
        elif state['pending_question'] and action not in ('REQUEST_CLOSE','CONTINUE_EVENT'):
            pending={k:state['pending_question'][k] for k in ('question','needed_evidence')}
        payload=dict(obs,lifecycle=intent,refinement=refine,pending=pending,
                     next_review_s=state['scheduled_review_s'])
        if state['pending_question']:
            payload['pending_resolution']='still_pending' if pending else ('resolved' if action=='REQUEST_CLOSE' else 'withdrawn')
        if persist:
            payload['end_review_memory']=memory
        update=tracker.apply(a,b,payload)
        writes.append(dict(cursor_s=b,action=action,refinement=refine,update=update))
    events=tracker.finish(tracker.cursor)
    return dict(events=events,intervals=tracker.intervals,counts=dict(counts),writes=writes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','config','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=('rolling','event_memory'),required=True)
    args=p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data=json.loads(args.input.read_bytes())
    args.output.mkdir(parents=True,exist_ok=True)
    model=LocalQwenModel(dict(json.loads(args.config.read_bytes()),device='cuda:0'))
    job=dict(input_sha256=digest(args.input),config_sha256=digest(args.config),mode=args.mode)
    for case in data['cases']:
        fork=case['fork_s']
        seed=restore_prefix(data['prefix'],fork)
        active=active_view(seed,seed.snapshot(fork,fork+10,False))
        if active!=case['expected_open_event']:
            raise ValueError('Replayed state differs from the original active-event snapshot')
        key=str(int(fork))
        path=args.output/(key+'.json')
        log=args.output/(key+'.jsonl.gz')
        if path.exists():
            saved=json.loads(path.read_bytes())
            if saved['job']!=job or (saved['log_sha256'] and digest(log)!=saved['log_sha256']):
                raise ValueError('Saved replay does not match inputs')
            continue
        steps=[]
        result=replay(seed,case,RecordingChatModel(model,log),lambda tag,row:steps.append([tag,row]),
                      args.mode=='event_memory')
        tmp=path.with_suffix('.partial')
        save_json(tmp,dict(job=job,fork_s=fork,seed_sha256=canonical_hash(seed.__dict__),
                           log_sha256=digest(log) if log.exists() else None,steps=steps,**result))
        tmp.replace(path)
        print(key,'complete',flush=True)
    save_json(args.output/'complete.json',dict(job=job,cases=len(data['cases'])))


if __name__=='__main__':
    main()
