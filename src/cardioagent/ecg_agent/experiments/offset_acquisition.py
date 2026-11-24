"""Replay Qwen's optional boundary measurement on frozen, reference-blind closure opportunities."""
import argparse
from pathlib import Path
import json
from hashlib import sha256
from time import perf_counter, time_ns

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.harness.historical_window import HistoricalWindow
from cardioagent.ecg_agent.harness.offset_localization import review_offset
from cardioagent.ecg_agent.llm.providers import LocalQwenModel
from cardioagent.ecg_agent.llm.recording import RecordingChatModel
from cardioagent.ecg_agent.models.ecgfounder import create_ecgfounder_scorer
from cardioagent.ecg_agent.schemas import EvidenceContext


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('input','raw','models','config','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data=json.loads(a.input.read_bytes())
    predict,info=create_ecgfounder_scorer(a.models,a.device)
    if digest(Path(info['checkpoint']))!=data['checkpoint_sha256']:
        raise ValueError('Founder checkpoint mismatch')
    model=LocalQwenModel(dict(json.loads(a.config.read_bytes()),device=a.device))
    a.output.mkdir(parents=True,exist_ok=True)
    job=dict(revision=a.revision,config_sha256=digest(a.config),input_sha256=digest(a.input),device=a.device)
    path=a.output/'job.json'
    if path.exists() and json.loads(path.read_bytes())!=job:
        raise ValueError('Changed boundary replay inputs')
    save_json(path,job)
    for rec in data['records']:
        name=rec['record_id']
        if any(digest(a.raw/f'{name}.{ext}')!=rec[ext] for ext in ('dat','hea')):
            raise ValueError('Changed waveform or header')
        reader=HistoricalWindow(a.raw,dict(record_id=name,start_s=0,end_s=rec['seconds']),predict)
        for row in rec['opportunities']:
            onset,grid,now=row['onset'],row['grid_end'],row['cursor']
            key=f'{name}-{now:g}'
            case=a.output/(key+'.json')
            if case.exists():
                old=json.loads(case.read_bytes())
                expected=old.pop('receipt_sha256')
                if (sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()!=expected or old['job']!=job
                        or old['key']!=key or (old['log_sha256'] and
                        digest(a.output/old['log_name'])!=old['log_sha256'])):
                    raise ValueError('Invalid saved boundary decision')
                continue
            ctx=EvidenceContext(f'offset-{a.revision}',key,name)
            logs=[]
            def measure(start,end,cursor):
                if (start,end,cursor)!=(grid-5,grid+5,now):
                    raise ValueError('Unexpected request')
                pair=[]
                for lead in ('ECG1','ECG2'):
                    receipt=reader.shifted_score(grid,grid+10,cursor,lead,-5,ctx)
                    logs.append(dict(category='measurement',receipt=receipt))
                    scores=receipt['result']
                    pair.append(max(scores['af_score'],scores['afl_score']) if receipt['status']=='ok' else None)
                return pair
            t=perf_counter()
            call_file=a.output/f'{key}.{time_ns()}.jsonl.gz'
            result=review_offset(onset,grid,now,row['aligned'],data['calibration'],
                                 RecordingChatModel(model,call_file),
                                 measure,lambda category,obj:logs.append(dict(category=category,**obj)))
            saved=dict(key=key,result=result,logs=logs,wall_s=perf_counter()-t,job=job,
                       log_name=call_file.name,log_sha256=digest(call_file) if call_file.exists() else None)
            saved['receipt_sha256']=sha256(json.dumps(saved,sort_keys=True).encode()).hexdigest()
            tmp=case.with_suffix('.partial')
            save_json(tmp,saved)
            tmp.replace(case)
            print(key,result['choice'],flush=True)
    save_json(a.output/'complete.json',dict(cases=sum(len(r['opportunities']) for r in data['records'])))


if __name__=='__main__':
    main()
