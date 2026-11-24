"""Score an independent continuous waveform span. References are only consumed by an offline evaluator."""
import argparse
from collections import deque
import json
from pathlib import Path
from time import perf_counter
import wfdb

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.data.preprocessing import invalid_spans, preprocess_window
from cardioagent.ecg_agent.models.ecgfounder import create_ecgfounder_scorer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('input','raw','models','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--revision',required=True)
    a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data=json.loads(a.input.read_bytes())
    predict,info=create_ecgfounder_scorer(a.models,a.device)
    if digest(Path(info['checkpoint']))!=data['checkpoint_sha256']:
        raise ValueError('Unexpected model weights')
    job=dict(input_sha256=digest(a.input),revision=a.revision,device=a.device)
    a.output.mkdir(parents=True,exist_ok=True)
    for task in data['tasks']:
        name=task['record_id']; stem=a.raw/name
        if not name.isdecimal() or any(digest(stem.with_suffix('.'+ext))!=task[ext] for ext in ('hea','dat')):
            raise ValueError('Invalid record or waveform changed')
        output=a.output/(name+'.json')
        if output.exists():
            saved=json.loads(output.read_bytes())
            if saved['job']!=job or len(saved['rows'])*10!=task['end_s']-task['start_s']:
                raise ValueError('Incomplete or foreign saved record')
            continue
        t=perf_counter()
        fs=128
        def pair(start,end):
            wave=wfdb.rdrecord(str(stem),sampfrom=start*fs,sampto=end*fs)
            if wave.fs!=fs or tuple(wave.sig_name)!=('ECG1','ECG2') or wave.p_signal.shape!=(10*fs,2):
                raise ValueError('Unexpected waveform coordinates')
            return [None if invalid_spans(x,fs) else max(predict(preprocess_window(x,fs))) for x in wave.p_signal.T]
        rows=[]; shifts=[]
        active=False; highs=lows=0
        history=deque(maxlen=4)
        for start in range(task['start_s'],task['end_s'],10):
            cursor=start+10
            vals=pair(start,cursor)
            rows.append([start,cursor,vals])
            history.append(vals)
            high=None not in vals and min(vals)>=.5
            low=None not in vals and max(vals)<.5
            highs=highs+1 if high else 0
            if highs>=2:
                active=True
            lows=lows+1 if active and low else 0
            if active and lows==2:
                grid=start-10
                if len(history)==4 and all(None not in h for h in history):
                    shifts.append(dict(grid_end_s=grid,cursor_s=cursor,pair=pair(grid-5,grid+5)))
                active=False; lows=0
        tmp=output.with_suffix('.partial')
        save_json(tmp,dict(job=job,task=task,rows=rows,shifts=shifts,wall_s=perf_counter()-t))
        tmp.replace(output)
        print(name,len(rows),len(shifts),flush=True)
    save_json(a.output/'complete.json',dict(job=job,records=len(data['tasks'])))


if __name__=='__main__':
    main()
