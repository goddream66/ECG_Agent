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


def review(item, model, measure, emit, mode, *, grounded=False):
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
    broad=['NO_FLAG','UNCERTAIN','FLAG_WINDOW']
    localized=[f'FLAG_CELL_{i}' for i in range(5)]
    if grounded:
        submit=schema('submit_interruption_review','Submit one finding at the available time resolution.',
                      dict(action=choice(broad),reason=TEXT),['action','reason'])
    tool=schema('measure_beat_change','Measure local changes in detected heartbeats on both leads.',{},[])
    seen=[]
    def inspect(name,args):
        if name!='measure_beat_change' or args:
            raise ValueError('This measurement takes no arguments')
        result=dict(evidence_id='B1',columns=FEATURES,cells=measure())
        seen.append(True)
        if grounded:
            usable=[code for i,code in enumerate(localized)
                    if any(lead[i] is not None for lead in result['cells'].values())]
            submit['function']['parameters']['properties']['action']=choice(broad+usable)
        return result
    def validate(args):
        if grounded:
            action=args['action']
            if action not in submit['function']['parameters']['properties']['action']['enum']:
                raise ValueError('Choose one of the available findings at its measured time resolution.')
            flagged=action=='FLAG_WINDOW' or action in localized
            return dict(status='accepted',action='FLAG_INTERVAL' if flagged else action,
                        cell=str(localized.index(action)) if action in localized else 'none',
                        submitted_action=action,reason=args['reason'])
        if args['action'] not in actions or args['cell'] not in positions or (
                (args['action']=='FLAG_INTERVAL') != (args['cell']!='none')):
            raise ValueError('Name one cell only when flagging a possible interruption')
        return dict(status='accepted',**args)
    context=dict(window_s=[now-10,now],cursor_s=now, founder_windows=item['scores'],cells_s=[[now-10+2*i,now-8+2*i] for i in range(5)])
    if mode=='all':
        context['measured_beats']=inspect('measure_beat_change',{})
    counts=Counter()
    instruction=(
               'An AF/AFL event was open before these last ten seconds. Decide whether a two-second cell '
               '(numbered 0 to 4) merits review for an actual interruption. Founder rows are start, end, '
               'and max(AF,AFL) on each lead. A low score could also be artifact. If flagging, identify the '
               'single best cell. NO_FLAG means no convincing interruption, UNCERTAIN means ambiguous. '
               'Every optional measurement describes the same five cells per lead, relative to the prior '
               '20 seconds. RR and amplitude ratios near one resemble the local reference. Distances '
               'measure differences in beat shape; missing and irregular beats alone do not diagnose a transition.')
    if grounded:
        instruction=(
            'Review the last ten seconds of an open AF/AFL event for a possible brief interruption. '
            'Each Founder pair compares two leads over one entire ten-second window. All five two-second cells '
            'share exactly the same Founder measurement. A previous whole-window score can show a trend. '
            'FLAG_WINDOW reports a suspected interruption somewhere in the window. NO_FLAG and UNCERTAIN are valid. '
            'After local beat measurements arrive you may instead flag a specific cell if its measured change '
            'justifies that location. In B1, ECG1 and ECG2 each list five cells in time order, indexed 0 to 4. '
            'RR and amplitude ratios are relative to the preceding twenty seconds. Morphology distance describes '
            'a change in beat shape. Missing peaks and regular rhythm alone cannot establish an AF/AFL interruption. '
            'Give the actual time range and measurement that support your finding.')
    result=ask(model,counts,emit,dict(cursor_s=now),context,submit,instruction,
               validate,extras=[tool] if mode=='active' else [],measure=inspect)
    return dict(decision=result,measurement_requests=len(seen),counts=dict(counts))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','raw','output','config'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=('aligned','active','all'),required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--grounded',action='store_true')
    a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows=json.loads(a.input.read_bytes())
    a.output.mkdir(parents=True,exist_ok=True)
    job=dict(revision=a.revision,input_sha256=digest(a.input),config_sha256=digest(a.config),mode=a.mode,grounded=a.grounded)
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
        result=review(item,RecordingChatModel(model,log),measure,lambda tag,row:steps.append([tag,row]),a.mode,
                      grounded=a.grounded)
        tmp=path.with_suffix('.partial')
        save_json(tmp,dict(key=key,job=job,result=result,steps=steps,log=log.name,log_sha256=digest(log),wall_s=perf_counter()-start))
        tmp.replace(path)
        print(key,result['decision'].get('action','ACTION_ERROR'),flush=True)
    save_json(a.output/'complete.json',dict(job=job,cases=len(rows)))


if __name__=='__main__':
    main()
