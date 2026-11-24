"""Review possible brief interruptions in an open AF/AFL episode at a fixed arrived cursor."""
import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path
from time import time_ns, perf_counter
import wfdb
from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.harness.staged_review import ask, schema, choice, TEXT
from cardioagent.ecg_agent.llm.recording import RecordingChatModel
from cardioagent.ecg_agent.llm.providers import LocalQwenModel
from .subwindow_change import FEATURES, measure_lead


def review(item, model, measure, emit, mode):
    now=item['cursor']
    if set(item)!={'record_id','cursor','scores'} or mode not in ('aligned','active','all'):
        raise ValueError('Unexpected review inputs')
    if not item['scores'] or any(not now-30<=a<b<=now or b-a!=10 or len(v)!=2
                                 or any(z is not None and (not isinstance(z,(int,float)) or not isfinite(z) or not 0<=z<=1)
                                        for z in v) for a,b,v in item['scores']):
        raise ValueError('Unexpected evidence time or lead pair')
    actions=('FLAG_INTERVAL','NO_FLAG','UNCERTAIN')
    positions=['none']+[str(i) for i in range(5)]
    submit=schema('submit_interruption_review','Record a finding for this ten-second window.',
                  dict(action=choice(actions),cell=choice(positions),reason=TEXT),['action','cell','reason'])
    tool=schema('measure_beat_change','Measure local changes in detected heartbeats on both leads.',{},[])
    seen=[]
    def inspect(name,args):
        if name!='measure_beat_change' or args:
            raise ValueError('This measurement takes no arguments')
        result=dict(evidence_id='B1',columns=FEATURES,cells=measure())
        seen.append(True)
        return result
    def validate(args):
        if args['action'] not in actions or args['cell'] not in positions or (
                (args['action']=='FLAG_INTERVAL') != (args['cell']!='none')):
            raise ValueError('Name one cell only when flagging a possible interruption')
        return dict(status='accepted',**args)
    context=dict(cursor_s=now, founder_windows=item['scores'],cells_s=[[now-10+2*i,now-8+2*i] for i in range(5)])
    if mode=='all':
        context['measured_beats']=inspect('measure_beat_change',{})
    counts=Counter()
    result=ask(model,counts,emit,dict(cursor_s=now),context,submit,
               'An AF/AFL event was open before these last ten seconds. Decide whether a two-second cell '
               '(numbered 0 to 4) merits review for an actual interruption. Founder rows are start, end, '
               'and max(AF,AFL) on each lead. A low score could also be artifact. If flagging, identify the '
               'single best cell. NO_FLAG means no convincing interruption, UNCERTAIN means ambiguous. '
               'Every optional measurement describes the same five cells per lead, relative to the prior '
               '20 seconds. RR and amplitude ratios near one resemble the local reference. Distances '
               'measure differences in beat shape; missing and irregular beats alone do not diagnose a transition.',
               validate,extras=[tool] if mode=='active' else [],measure=inspect)
    return dict(decision=result,measurement_requests=len(seen),counts=dict(counts))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','raw','output','config'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=('aligned','active','all'),required=True)
    p.add_argument('--revision',required=True)
    a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows=json.loads(a.input.read_bytes())
    a.output.mkdir(parents=True,exist_ok=True)
    job=dict(revision=a.revision,input_sha256=digest(a.input),config_sha256=digest(a.config),mode=a.mode)
    model=LocalQwenModel(dict(json.loads(a.config.read_bytes()),device='cuda:0'))
    for item in rows:
        name=item['record_id']; now=item['cursor']
        if not name.isdecimal() or type(now) is not int or now<30:
            raise ValueError('Invalid case')
        key=f'{name}-{now}'; path=a.output/(key+'.json')
        if path.exists():
            old=json.loads(path.read_bytes())
            if old['job']!=job or old['key']!=key or digest(a.output/old['log'])!=old['log_sha256']:
                raise ValueError('Invalid saved review')
            continue
        def measure():
            w=wfdb.rdrecord(str(a.raw/name),sampfrom=(now-30)*128,sampto=now*128)
            if w.fs!=128 or w.sig_name!=['ECG1','ECG2']:
                raise ValueError('Unexpected leads')
            return {lead:measure_lead(x,128) for lead,x in zip(w.sig_name,w.p_signal.T)}
        log=a.output/f'{key}.{time_ns()}.jsonl.gz'
        start=perf_counter(); steps=[]
        result=review(item,RecordingChatModel(model,log),measure,lambda tag,row:steps.append([tag,row]),a.mode)
        tmp=path.with_suffix('.partial')
        save_json(tmp,dict(key=key,job=job,result=result,steps=steps,log=log.name,log_sha256=digest(log),wall_s=perf_counter()-start))
        tmp.replace(path)
        print(key,result['decision'].get('action','ACTION_ERROR'),flush=True)
    save_json(a.output/'complete.json',dict(job=job,cases=len(rows)))


if __name__=='__main__':
    main()
