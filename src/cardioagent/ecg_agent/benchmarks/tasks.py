"""Convert ECG-Scroll AF tasks to separate inference manifests and evaluation references."""
from hashlib import sha256
import json
import math
from pathlib import Path
import wfdb
from ...dataset import digest,save_json
from .intervals import validate_intervals,union_seconds


def read_jsonl(path,key):
    rows={}
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            value=row.get(key)
            if not isinstance(value,str) or not value or value in rows:
                raise ValueError(f'Missing/duplicate {key} in {path}')
            rows[value]=row
    if not rows:raise ValueError(f'Empty JSONL: {path}')
    return rows


def write_jsonl(path,rows):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')


def prepare(source,raw_root,output,*,max_records=6,duration_s=1800.,dataset='afdb'):
    if max_records<1 or not math.isfinite(duration_s) or duration_s<0:
        raise ValueError('Positive record limit and nonnegative finite duration required; 0 means full record')
    source,raw_root,output=map(Path,(source,raw_root,output))
    rows=list(read_jsonl(source,'id').values())
    episodes=[r for r in rows if r['task']=='episode_detection' and r['pn_dir']==dataset]
    burdens={(r['pn_dir'],r['record'],r['t0'],r['t1']):r for r in rows
             if r['task']=='burden_quantification' and r.get('answer',{}).get('kind')=='af_burden'}
    episodes.sort(key=lambda r:sha256(f"ecg-pilot-v1:{r['pn_dir']}:{r['record']}".encode()).digest())
    public,truth,seen=[],[],set();input_hashes={};skipped=[]
    for row in episodes:
        record=row['record'];db=row['pn_dir']
        if Path(record).name!=record or record in ('.','..') or db not in ('afdb','ltafdb'):
            raise ValueError('Unsafe record/database identifier')
        identity=(db,record)
        if identity in seen:continue
        seen.add(identity)
        stem=raw_root/f'{db}_local'/record
        if not stem.with_suffix('.hea').is_file() or not stem.with_suffix('.dat').is_file():
            skipped.append(f'{db}/{record}');continue
        header=wfdb.rdheader(str(stem))
        start=float(row['t0']);end=float(row['t1'])
        if duration_s:end=min(end,start+duration_s)
        if not 0<=start<end<=header.sig_len/header.fs+1e-6:raise ValueError('Task/header bounds disagree')
        original=validate_intervals(row['answer']['intervals'],float(row['t0']),float(row['t1']))
        intervals=[[max(a,start),min(b,end)] for a,b in original if min(b,end)>max(a,start)]
        paired=burdens.get((db,record,row['t0'],row['t1']))
        full=start==row['t0'] and end==row['t1']
        reference=float(paired['answer']['value']) if paired and full else union_seconds(intervals)/(end-start)
        task_id=f"{row['id']}:window:{start:g}:{end:g}"
        public.append({'task_id':task_id,'task':'af_localization_and_burden',
            'record_stem':str(stem.resolve()),'source_database':db,'record_id':record,
            'start_s':start,'end_s':end,'time_coordinates':'absolute_record_seconds',
            'target':'AF/AFL (upstream ECG-Scroll label policy)',
            'instruction':'Return event intervals within the observed task span; derive burden from their union.'})
        truth.append({'task_id':task_id,'intervals':intervals,'burden':reference,
            'tau':row['verifier'].get('params',{}).get('tau',.5),
            'burden_floor':paired['verifier'].get('params',{}).get('floor',.02) if paired else .02,
            'start_s':start,'end_s':end,'source_task_ids':[row['id']]+([paired['id']] if paired else []),
            'truncated':not full,'label_policy':'AFIB+AFL; preserves upstream targets'})
        input_hashes[f'{db}/{record}.hea']=digest(stem.with_suffix('.hea'))
        if len(public)>=max_records:break
    if not public:raise ValueError('No local waveform-backed AF tasks found')
    output.mkdir(parents=True,exist_ok=False)
    write_jsonl(output/'agent_tasks.jsonl',public)
    write_jsonl(output/'evaluation_references.jsonl',truth)
    write_jsonl(output/'empty_predictions.jsonl',[{'task_id':r['task_id'],'intervals':[]} for r in public])
    save_json(output/'task_manifest.json',{'schema':1,'source_sha256':digest(source),'headers_sha256':input_hashes,
        'selection':'deterministic record hash; fixed prefix, no label-based window selection',
        'tasks':len(public),'records':len(public),'hours':sum(r['end_s']-r['start_s'] for r in public)/3600,
        'positive_tasks':sum(bool(r['intervals']) for r in truth),
        'negative_tasks':sum(not r['intervals'] for r in truth),'skipped_missing_waveforms':skipped,
        'scope':'engineering pilot; not a train/validation/test split or full ECG-Scroll reproduction',
        'label_policy':'AFIB+AFL; do not label this AFIB-only',
        'reference_precision':'Full tasks preserve published rounded burden; cropped tasks recompute burden from clipped intervals',
        'inference_manifest_sha256':digest(output/'agent_tasks.jsonl'),
        'references_sha256':digest(output/'evaluation_references.jsonl')})
    return output
