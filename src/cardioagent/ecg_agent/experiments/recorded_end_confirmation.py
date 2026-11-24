"""Run the real end stage on arrived memory reconstructed from fixed recorded evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.harness.staged_review import confirm_end_with_evidence
from cardioagent.ecg_agent.llm.providers import LocalQwenModel
from cardioagent.ecg_agent.llm.recording import RecordingChatModel


def confirm(packet, model, emit):
    if set(packet)!={'cursor_s','context','key','aliases'}:
        raise ValueError('Expected recorded measurements only')
    view=packet['context']
    allowed={'window_s','assessment','open_event','rhythm_by_time','pending_question','end_evidence'}
    if set(view)!=allowed:
        raise ValueError('Unexpected context fields')
    now=packet['cursor_s']
    proof=view['end_evidence']
    if (view['window_s'][1]!=now or any(row['window_s'][1]>now for row in view['rhythm_by_time'])):
        raise ValueError('Measurements after the review cursor')
    if proof['can_confirm']:
        left,first,last=(proof[key] for key in ('before','first_against','later_against'))
        a,b=left['window_s']; c,d=first['window_s']; e,f=last['window_s']
        if (left['direction']!='support' or first['direction']!='negative' or last['direction']!='negative'
                or not a<b<=c<d<=e<f==now or any(x['id'] not in packet['aliases'] for x in (left,first,last))):
            raise ValueError('Invalid or untraceable confirmation proof')
    counts=Counter()
    result=confirm_end_with_evidence(model,counts,emit,dict(cursor_s=now),view,packet['aliases'])
    return dict(result=result,counts=dict(counts))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','config','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    packets=json.loads(a.input.read_bytes())
    a.output.mkdir(parents=True,exist_ok=True)
    job=dict(input_sha256=digest(a.input),config_sha256=digest(a.config))
    model=LocalQwenModel(dict(json.loads(a.config.read_bytes()),device='cuda:0'))
    for item in packets:
        key=item['key']
        if not key.replace('-','',1).isdecimal():
            raise ValueError('Invalid case key')
        path=a.output/(key+'.json')
        logfile=a.output/(key+'.jsonl.gz')
        if path.exists():
            saved=json.loads(path.read_bytes())
            if saved['job']!=job or (saved['log_sha256'] and digest(logfile)!=saved['log_sha256']):
                raise ValueError('Invalid stored end confirmation')
            continue
        t=perf_counter()
        steps=[]
        result=confirm(item,RecordingChatModel(model,logfile),lambda tag,row:steps.append([tag,row]))
        tmp=path.with_suffix('.partial')
        save_json(tmp,dict(job=job,key=key,cursor_s=item['cursor_s'],wall_s=perf_counter()-t,
                           log_sha256=digest(logfile) if logfile.exists() else None,steps=steps,**result))
        tmp.replace(path)
        print(key,result['result']['status'],flush=True)
    save_json(a.output/'complete.json',dict(job=job,cases=len(packets)))


if __name__=='__main__':
    main()
