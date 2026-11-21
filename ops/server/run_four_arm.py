"""Run the fixed-evidence or continuous 04936 arm. Select the frozen A checkout via PYTHONPATH."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import torch

from run_full_focus import ContinuousLogs, Recorder, save
from cardioagent.ecg_agent.agent import replay_event_record
from cardioagent.ecg_agent.harness import react
from cardioagent.ecg_agent.harness.event_decision import WindowEventDecision, validate_grounding
from cardioagent.ecg_agent.harness.execution import build_model
from cardioagent.ecg_agent.llm.providers import build_chat_model
from cardioagent.ecg_agent.policies.event_rules import EventConfig
from cardioagent.ecg_agent.schemas import EvidenceContext


def fixed(output, model, args, prompt):
    cases = list(map(json.loads, args.evidence.read_text().splitlines()))
    assert [x['window_s'][1] for x in cases] == [1560,1670,1690,1760,1780,1810]
    history = {}
    with gzip.open(args.archived_calls,'rt') as stream:
        for line in stream:
            r = json.loads(line)
            t = r['cursor_s']
            if t > 1810:
                break
            if t in history or t not in {x['window_s'][1] for x in cases}:
                continue
            state = json.loads(r['messages'][1]['content'])
            history[t] = [h for h in state['recent_windows'] if h['window_s'][0] >= 1550]
    schema = {'type':'function','function':{'name':'submit_event_decision',
              'description':'Submit one current-window assessment.',
              'parameters':WindowEventDecision.model_json_schema()}}
    results = []
    for case in cases:
        a,b = case['window_s']
        hist = history[b]
        if args.arm == 'D':
            hist = react.factual_history(hist)
        aliases = {f'E{i}':r for i,r in enumerate(case['evidence'],1)}
        assert all(r['window_s']==[a,b] and r['available_at_s']<=b for r in aliases.values())
        state = dict(window_s=[a,b], state='IDLE', active_event=None, recent_windows=hist,
                     current_evidence=[dict(r,id=k) for k,r in aliases.items()])
        messages = [{'role':'system','content':prompt+'\nFixed evidence review. Use the supplied measurements only. Submit one decision with current evidence IDs or aliases.'},
                    {'role':'user','content':json.dumps(state,allow_nan=False)}]
        answer = {'window_s':[a,b]}
        try:
            reply = model.complete(messages,[schema])
            fn=reply.message['tool_calls'][0]['function']
            if fn['name']!='submit_event_decision':
                raise ValueError('Unexpected tool')
            proposal=WindowEventDecision.model_validate_json(fn['arguments'])
            answer['proposal']=proposal.model_dump()
            known={r['evidence_id']:r for r in aliases.values()}
            proposal.evidence_ids=[aliases[k]['evidence_id'] if k in aliases else k for k in proposal.evidence_ids]
            if len(set(proposal.evidence_ids))!=len(proposal.evidence_ids) or not set(proposal.evidence_ids)<=known.keys():
                raise ValueError('Invalid or duplicate current evidence citation')
            validate_grounding(proposal,known,None,a,b,None,set())
            if proposal.end_s is not None:
                raise ValueError('Unexpected end time in IDLE')
            answer['accepted']=True
        except (KeyError,ValueError) as e:
            answer.update(accepted=False,error=str(e))
        results.append(answer)
    save(output/'fixed.json',results)
    assert len(results)==6


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('output','storage','config','evidence','archived_calls'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',choices=list('ABCD'),required=True)
    p.add_argument('--phase',choices=['fixed','online'],required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert len(os.sched_getaffinity(0))==2 and torch.cuda.device_count()==1
    cfg=json.loads(args.config.read_text())
    cfg['investigation_checkpoint']=args.arm=='A'
    if args.arm=='D':
        cfg['fact_history']=True
    if args.arm in 'AB':
        react.WINDOW_PROMPT=Path(__file__).with_name('ablation_original_prompt.txt')
    prompt=react.WINDOW_PROMPT.read_text()
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    task=dict(task_id='full-04936',start_s=1550,end_s=1850)
    save(out/'experiment.json',dict(arm=args.arm,phase=args.phase,task=task,llm=cfg,
         cpu_affinity=sorted(os.sched_getaffinity(0)),gpu=os.environ['CUDA_VISIBLE_DEVICES'],
         harness_sha256=hashlib.sha256(Path(react.__file__).read_bytes()).hexdigest(),
         fixed_evidence_sha256=hashlib.sha256(args.evidence.read_bytes()).hexdigest()))
    (out/'prompt.txt').write_text(prompt)
    model=Recorder(build_chat_model(cfg),out/'qwen_calls.jsonl.gz')
    if args.phase=='fixed':
        fixed(out,model,args,prompt)
        return
    cache=args.storage/'data/afdb_preprocessed_server'
    record=next(r for r in json.loads((cache/'manifest.json').read_text())['records'] if r['record_id']=='04936')
    scorer,info,load_ms=build_model('live',args.storage/'models',2,'cpu')
    opts={k:v for k,v in cfg.items() if k not in {'provider','model_path','device','max_new_tokens'}}
    reviewer=react.ReActReview(model,live_scoring=True,**opts)
    logs=ContinuousLogs(out)
    try:
        result=replay_event_record(cache,record,task,'ECG1',EventConfig(),lambda lead,x,loc:scorer(x),
                logs,EvidenceContext('four-arm','full-04936','04936'),policy='event-react',
                raw_root=args.storage/'data/afdb_local',reviewer=reviewer)
        assert logs.windows==30
        result.update(founder=info,founder_load_ms=load_ms)
        save(out/'result.json',result)
    finally:
        logs.close()


if __name__=='__main__':
    main()
