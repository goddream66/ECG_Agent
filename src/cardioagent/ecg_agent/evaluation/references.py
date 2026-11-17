"""Label loading is evaluation-only; inference tasks and answers stay separate."""
from hashlib import sha256
import json
import math
from pathlib import Path
import wfdb
from cardioagent.common.io import digest, save_json, read_jsonl, write_jsonl
from cardioagent.common.intervals import validate_intervals, union_seconds

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


def prepare_validity(tasks_file, raw_root, cache, output):
    """Read independent .atr only in evaluation; do not modify upstream targets."""
    from .metrics import union
    tasks = read_jsonl(tasks_file, 'task_id')
    rows = {}
    for task_id, task in tasks.items():
        record = task['record_id']
        stem = Path(raw_root) / record
        header = wfdb.rdheader(str(stem))
        ann = wfdb.rdann(str(stem), 'atr')
        boundaries = [(int(sample) / header.fs, note) for sample, note in zip(ann.sample, ann.aux_note)
                      if note.strip().startswith('(')]
        valid, raw_af = [], []
        for i, (a, note) in enumerate(boundaries):
            b = boundaries[i+1][0] if i+1 < len(boundaries) else header.sig_len/header.fs
            a, b = max(a, task['start_s']), min(b, task['end_s'])
            label = note.strip().lstrip('(').rstrip(')')
            if b > a and label in ('AFIB','AFL','J','N'):
                valid.append([a,b])
                if label in ('AFIB','AFL'):
                    raw_af.append([a,b])
        valid, raw_af = union(valid), union(raw_af)
        missing = {'ECG1': [], 'ECG2': []}
        with (Path(cache) / f'{record}.windows.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                for span in row['invalid_spans']:
                    missing[row['channel']].append([span['start_sample'] / header.fs,
                                                     span['end_sample'] / header.fs])
        rows[task_id] = {'record_id': record, 'valid_intervals': valid,
                        'annotation_sha256': digest(stem.with_suffix('.atr')),
                        'raw_annotation_af_intervals': raw_af,
                        'observed_rhythm_labels': sorted({note.strip() for _,note in boundaries}),
                        'missing_intervals': {k: union(v) for k, v in missing.items()},
                        'label_rule': 'upstream AFIB+AFL event targets unchanged; time metrics start at first rhythm annotation',
                        'unlabeled_s': task['end_s'] - task['start_s'] - union_seconds(valid)}
    save_json(Path(output), {'tasks_sha256': digest(Path(tasks_file)), 'records': rows,
                            'inference_access': False, 'scope': 'AFDB exploratory development'})
    return rows
