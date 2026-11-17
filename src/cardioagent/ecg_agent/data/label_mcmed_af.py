"""Offline AF/AFL-only weak labels for explicitly selected MC-MED patients."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import wfdb

from cardioagent.common.paths import code_root
from cardioagent.ecg_agent.data.ecgfounder_preprocessing import preprocess_window, load_profile
from cardioagent.ecg_agent.data.preprocessing import invalid_spans
from cardioagent.ecg_agent.models.ecgfounder import create_ecgfounder_scorer


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def timestamp(origin, seconds):
    return (datetime.fromisoformat(origin)+timedelta(seconds=seconds)).isoformat() if origin else None


def classify(af, afl, threshold, comparison='gt'):
    if not np.isfinite([af, afl]).all() or not 0 <= min(af, afl) <= max(af, afl) <= 1:
        raise ValueError('Invalid AF/AFL scores')
    if comparison not in ('gt', 'ge'):raise ValueError('Comparison must be gt or ge')
    return bool(max(af, afl) >= threshold if comparison=='ge' else max(af, afl) > threshold)


def intervals(rows):
    """Merge adjacent positive windows; negatives and unknowns split events."""
    result=[]
    for row in rows:
        if row['af_afl_candidate'] is not True:
            continue
        if result and result[-1]['end_s']==row['start_s'] and result[-1]['last_window_index']==row['window_index']-1:
            result[-1].update(end_s=row['end_s'], last_window_index=row['window_index'], end_datetime=row['end_datetime'])
        else:
            result.append({k:row[k] for k in ('patient_id','visit_id','segment_id','start_s','end_s','start_datetime','end_datetime')} |
                          {'first_window_index':row['window_index'],'last_window_index':row['window_index'],
                           'label':'AF/AFL','human_verified':False})
    for row in result:
        row['duration_s']=row['end_s']-row['start_s']
    return result


def select(root, patient_ids):
    selected=[];issues=[];seen=set();visits=[]
    with (root/'visits.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['MRN'] not in patient_ids:continue
            seen.add(row['MRN']);visit=row['CSN'];folder=root/'waveforms'/visit[-3:]/visit/'II'
            headers=sorted(folder.glob('*.hea'));visits.append({'patient_id':row['MRN'],'visit_id':visit,'headers':len(headers)})
            for header in headers:
                try:
                    h=wfdb.rdheader(str(header.with_suffix('')))
                    if h.n_sig!=1 or h.sig_name!=['II'] or h.fs<=80 or not float(h.fs).is_integer() or h.sig_len<=0:
                        raise ValueError('Unsupported signal header')
                    if h.units[0].lower().replace('μ','u').replace('µ','u') not in ('mv','uv','v'):
                        raise ValueError('Unsupported physical unit')
                    files=sorted({(header.parent/name).resolve() for name in h.file_name})
                    if not all(p.is_relative_to(root) and p.is_file() and p.stat().st_size for p in files):
                        raise ValueError('Missing or invalid data file')
                    selected.append({'patient_id':row['MRN'],'visit_id':visit,'segment_id':header.stem,
                                     'header':str(header.resolve()),'fs':float(h.fs),'samples':int(h.sig_len),
                                     'origin':str(h.base_datetime) if h.base_datetime else None,
                                     'source_sha256':{str(p):sha(p) for p in [header,*files]}})
                except (OSError, ValueError, IndexError) as exc:
                    issues.append({'header':str(header),'reason':str(exc)})
    if seen!=set(patient_ids):raise ValueError('Requested patient absent from visits table')
    return sorted(selected,key=lambda x:(x['patient_id'],x['visit_id'],x['origin'] or '',x['header'])),visits,issues


def run(args, factory=create_ecgfounder_scorer):
    if not 0 < args.threshold < 1:raise ValueError('Threshold must be in (0,1)')
    comparison=getattr(args,'comparison','gt');binary=getattr(args,'binary',False)
    cpu_threads=getattr(args,'cpu_threads',4)
    if comparison not in ('gt','ge') or cpu_threads<1:raise ValueError('Invalid comparison or CPU threads')
    positive_label,negative_label=('AF/AFL','no_AF/AFL') if binary else ('AF/AFL_candidate','below_threshold')
    root=args.source_root.resolve();output=args.output.resolve()
    if output.is_relative_to(root):raise ValueError('Output must be outside source data')
    output.mkdir(parents=True,exist_ok=False)
    started=perf_counter();records,visits,issues=select(root,args.patients)
    if not records:raise ValueError('No available ECG segments')
    dump(output/'records.json',records);dump(output/'visit_inventory.json',visits);dump(output/'source_issues.json',issues)
    git=lambda *a:subprocess.check_output(['git','-C',str(code_root()),*a],text=True).strip()
    manifest={'status':'running','patients':args.patients,'scope':'all available II segments across all visits',
              'window_s':10,'threshold':args.threshold,'rule':f"max(AF,AFL) {'>=' if comparison=='ge' else '>'} threshold",
              'binary':binary,'cpu_threads':cpu_threads,'positive_label':positive_label,'negative_label':negative_label,
              'preprocessing':load_profile(),'created_utc':datetime.now(timezone.utc).isoformat(),
              'git_commit':git('rev-parse','HEAD'),'git_dirty':bool(git('status','--porcelain')),
              'entry_sha256':sha(Path(__file__)),'visits_sha256':sha(root/'visits.csv'),
              'human_verified':False,'event_boundaries':'10-second grid, segment-relative; no cross-segment merging',
              'unknown_policy':'nonfinite, suspected flatline >=1s, incomplete tail, read/preprocessing failure',
              'not_a_streaming_agent_evaluation':True}
    dump(output/'manifest.json',manifest)
    load_start=perf_counter();predict,info=factory(args.models,args.device)
    import torch
    torch.set_num_threads(cpu_threads)
    manifest.update(model=info,checkpoint_sha256=sha(info['checkpoint']),model_load_s=perf_counter()-load_start)
    dump(output/'manifest.json',manifest)
    counts=Counter();patients={};model_s=0.
    try:
        with (output/'windows.jsonl').open('w') as windows_out, (output/'intervals.jsonl').open('w') as events_out:
            for number,r in enumerate(records,1):
                rows=[];fs=int(r['fs']);step=10*fs
                for i,a in enumerate(range(0,r['samples'],step)):
                    b=min(a+step,r['samples']);start=a/fs;end=b/fs
                    row={k:r[k] for k in ('patient_id','visit_id','segment_id')}
                    row.update(window_index=i,start_s=start,end_s=end,start_datetime=timestamp(r['origin'],start),
                               end_datetime=timestamp(r['origin'],end),af_afl_candidate=None,label='unknown',reason=None,human_verified=False,af_score=None,afl_score=None)
                    if binary:row['binary_label']=None
                    evidence={'window_index':i,'af_score':None,'afl_score':None}
                    try:
                        if b-a!=step:raise ValueError('incomplete_tail')
                        raw=wfdb.rdrecord(str(Path(r['header']).with_suffix('')),sampfrom=a,sampto=b)
                        if raw.fs!=fs or raw.sig_name!=['II'] or raw.p_signal.shape!=(step,1):raise ValueError('decoded_header_mismatch')
                        values=raw.p_signal[:,0];spans=invalid_spans(values,fs)
                        if spans:raise ValueError('invalid_signal:'+','.join(sorted({s[2] for s in spans})))
                        prepared=preprocess_window(values,fs)
                    except (OSError, ValueError, IndexError) as exc:
                        row['reason']=str(exc)
                    else:
                        t=perf_counter();af,afl=predict(prepared);model_s+=perf_counter()-t
                        row['af_afl_candidate']=classify(af,afl,args.threshold,comparison)
                        row['label']=positive_label if row['af_afl_candidate'] else negative_label
                        if binary:row['binary_label']=int(row['af_afl_candidate'])
                        row.update(af_score=af,afl_score=afl);evidence.update(af_score=af,afl_score=afl);counts['model_calls']+=1
                    rows.append(row);counts[row['label']]+=1
                    windows_out.write(json.dumps(row,allow_nan=False)+'\n')
                    # Only the two target scores are retained for audit, separately from labels.
                    with (output/'target_scores.jsonl').open('a') as scores_out:
                        scores_out.write(json.dumps({**{k:row[k] for k in ('patient_id','visit_id','segment_id')},**evidence},allow_nan=False)+'\n')
                events=intervals(rows)
                for event in events:events_out.write(json.dumps(event,allow_nan=False)+'\n')
                windows_out.flush();events_out.flush()
                p=patients.setdefault(r['patient_id'],Counter())
                p.update({label:sum(x['label']==label for x in rows) for label in (positive_label,negative_label,'unknown')})
                p.update(segments=1,intervals=len(events),positive_seconds=sum(x['duration_s'] for x in events),recorded_seconds=r['samples']/fs)
                dump(output/'progress.json',{'completed_segments':number,'total_segments':len(records),'counts':dict(counts),'patients':patients})
                print(json.dumps({'segment':number,'total':len(records),'patient':r['patient_id'],'windows':len(rows),'intervals':len(events)}),flush=True)
        manifest.update(status='complete',total_s=perf_counter()-started,model_execution_s=model_s)
        dump(output/'summary.json',{'counts':dict(counts),'patients':patients,'segments':len(records),'source_issues':len(issues)})
        with (output/'patient_summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['patient_id',positive_label,negative_label,'unknown','segments','intervals','positive_seconds','recorded_seconds']);writer.writeheader()
            for patient,stats in patients.items():writer.writerow({'patient_id':patient,**stats})
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:dump(output/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patients',nargs='+',required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threshold',type=float,default=0.8)
    parser.add_argument('--comparison',choices=['gt','ge'],default='gt')
    parser.add_argument('--binary',action='store_true',help='Export 1/0 labels; invalid windows remain null')
    parser.add_argument('--cpu-threads',type=int,default=4,help='CPU threads per worker, applied after model loading')
    run(parser.parse_args())


if __name__=='__main__':main()
