"""Export only AF, high-rate and low-rate weak annotations from verified evidence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib,json,math,os,shutil
from pathlib import Path
from uuid import uuid4

from cardioagent.ecg_agent.data.quality_reassessment import reassess_window_quality
from cardioagent.ecg_agent.data.label_quality import POLICY as QUALITY_POLICY
TASKS=('af','high_rate','low_rate')
VERSION='v0.5.1'
PROTOCOL={'af_negative_below':0.4,'af_positive_at_least':0.6,
          'high_rate_above_bpm':100.,'low_rate_below_bpm':60.,
          'minimum_candidate_duration_s':30.,
          'quality_review_policy':'retain proposal but set state uncertain',
          'group_count_gate_enabled':False,
          'clinical_calibration':'not validated; research weak labels only'}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rows(p):
    with Path(p).open() as f:
        for line in f:yield json.loads(line)
def task_states(af_score,rate,quality):
    proposed={'af':'unknown','high_rate':'unknown','low_rate':'unknown'}
    if af_score is not None and math.isfinite(af_score):
        proposed['af']='candidate_positive' if af_score>=PROTOCOL['af_positive_at_least'] else (
            'candidate_negative' if af_score<PROTOCOL['af_negative_below'] else 'uncertain')
    if rate is not None and math.isfinite(rate) and rate>0:
        proposed['high_rate']='candidate_positive' if rate>PROTOCOL['high_rate_above_bpm'] else 'candidate_negative'
        proposed['low_rate']='candidate_positive' if rate<PROTOCOL['low_rate_below_bpm'] else 'candidate_negative'
    result={}
    for task,p in proposed.items():
        state=p
        if quality not in ('no_quality_flag','review_required'):state='unknown'
        elif quality=='review_required' and p!='unknown':state='uncertain'
        result[task]={'state':state,'proposed_state':p,'human_state':None,'human_verified':False}
    return result

def event_candidates(windows):
    events=[]
    for task in TASKS:
        active=None
        for w in windows:
            positive=w['tasks'][task]['state']=='candidate_positive'
            same=(active is not None and active['segment_id']==w['segment_id']
                  and active['visit_id']==w['visit_id'] and active['patient_id']==w['patient_id']
                  and abs(active['end_s']-w['start_s'])<1e-6)
            if active is not None and (not positive or not same):
                events.append(active);active=None
            if positive:
                if active is None:
                    active={k:w[k] for k in ('patient_id','visit_id','segment_id','start_s','end_s')}
                    active.update(task=task,window_ids=[])
                active['end_s']=w['end_s'];active['window_ids'].append(w['window_id'])
        if active:events.append(active)
    for i,e in enumerate(events,1):
        e.update(event_id=f'candidate-{i:06d}',duration_s=e['end_s']-e['start_s'],
                 meets_duration_rule=e['end_s']-e['start_s']>=PROTOCOL['minimum_candidate_duration_s'],
                 human_verified=False,boundary_resolution_s=10.)
    return events

def export_package(source,parent,group):
    source=Path(source).resolve();m=json.loads((source/'manifest.json').read_text())
    if m['status']!='complete' or m['version']!='v0.4.0':raise ValueError('Expected complete v0.4.0 evidence package')
    for rel,digest in m['artifacts'].items():
        if sha(source/rel)!=digest:raise ValueError('Source artifact mismatch: '+rel)
    name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+group+'-'+uuid4().hex[:8]
    out=Path(parent)/VERSION/name;out.mkdir(parents=True,exist_ok=False)
    manifest={'version':VERSION,'status':'building','source_package':str(source),'source_manifest_sha256':sha(source/'manifest.json'),
              'tasks':list(TASKS),'protocol':PROTOCOL,'no_new_model_inference':True,'human_ground_truth':False,
              'code_sha256':sha(Path(__file__)),'quality_policy':QUALITY_POLICY,
              'quality_reassessment_sha256':sha(Path(__file__).with_name('quality_reassessment.py')),
              'quality_tool_sha256':sha(Path(__file__).parent.parent/'tools/physio.py'),'scope':m['config']['selection_scope']}
    dump(out/'manifest.json',manifest)
    try:
        auto={r['window_id']:r for r in rows(source/'auto_labels.jsonl')}
        evidence={};counts={t:Counter() for t in TASKS};quality_counts=Counter();result=[]
        for e in rows(source/'tool_evidence.jsonl'):
            if e['tool'] in ('detect_beats_featuredb','assess_template_quality'):
                evidence.setdefault(e['window_id'],{})[e['tool']]=e
        (out/'review').mkdir()
        with (out/'task_annotations.jsonl').open('w') as f,(out/'task_evidence.jsonl').open('w') as ef,(out/'review/annotations.jsonl').open('w') as review:
            for w in rows(source/'windows.jsonl'):
                a=auto[w['window_id']];es=evidence.get(w['window_id'],{})
                beat=es.get('detect_beats_featuredb');score=None;rate=None
                if a['ranked_candidates'] is not None:
                    score=next(c['score'] for c in a['ranked_candidates'] if c['label']=='ATRIAL FIBRILLATION')
                if beat and beat['status']=='ok':rate=beat['result'].get('rate_from_median_rr_bpm')
                quality,quality_reasons,es=reassess_window_quality(a,es);quality_counts[quality]+=1
                row={k:w[k] for k in ('window_id','patient_id','visit_id','segment_id','channel','start_s','end_s','start_sample','end_sample','segment_start_datetime','time_origin')}
                row.update(tasks=task_states(score,rate,quality),af_score=score,rate_bpm=rate,
                           rate_definition='60000 / median raw RR in milliseconds',quality_status=quality,
                           quality_reasons=quality_reasons,previous_quality_status=a['quality_status'],quality_policy=QUALITY_POLICY,evidence_ids=[e['evidence_id'] for e in es.values()])
                for task in TASKS:counts[task][row['tasks'][task]['state']]+=1
                f.write(json.dumps(row,allow_nan=False)+'\n');result.append(row)
                for e in es.values():ef.write(json.dumps(e,allow_nan=False)+'\n')
                review.write(json.dumps({'window_id':row['window_id'],'review_status':'pending',
                    'human_tasks':{t:None for t in TASKS},'candidate_tasks':row['tasks'],
                    'human_quality':None,'notes':''})+'\n')
        events=event_candidates(result)
        with (out/'event_candidates.jsonl').open('w') as f:
            for e in events:f.write(json.dumps(e)+'\n')
        shutil.copyfile(source/'records.json',out/'records.json')
        dump(out/'task_schema.json',{'tasks':list(TASKS),'states':['candidate_positive','candidate_negative','uncertain','unknown'],
             'note':'Targets are independent. Negative AF is not a normal ECG label. High/low rate does not imply sinus origin.'})
        dump(out/'protocol.json',PROTOCOL)
        summary={'windows':len(result),'tasks':{t:dict(c) for t,c in counts.items()},'quality':dict(quality_counts),
                 'events':{t:{'all_candidates':sum(e['task']==t for e in events),
                             'meeting_duration_rule':sum(e['task']==t and e['meets_duration_rule'] for e in events)} for t in TASKS}}
        dump(out/'summary.json',summary)
        (out/'README.md').write_text('# 三任务研究预标注\n\n仅 AF、心率过快、心率过慢，质量独立。task_annotations.jsonl 为逐窗结果；event_candidates.jsonl 为连续候选区间（保留短区间并标明是否达到30秒）。review/annotations.jsonl 为人工复核。protocol.json 为固定研究规则。AF分数未校准；心率来自原始RR中位数。质量提示使状态不确定，但保留 proposed_state。缺口和未知打断事件，不跨片段合并。未产生人工真值。未重新推理模型，仅从核验后的既有证据提取目标任务。其他临床标签不再导出。\n',encoding='utf8')
        manifest.update(status='complete',summary=summary,artifacts={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json' and 'review' not in p.parts})
        dump(out/'manifest.json',manifest);return out
    except Exception as exc:
        manifest.update(status='failed',error=str(exc));dump(out/'manifest.json',manifest);raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-batch',type=Path,default=Path('/data/1shared/gaojiajun/ECGagent/runs/mcmed-labeling-batch/20261116T140713Z-40ab8281'))
    p.add_argument('--run',action='store_true',help='Generate focused annotation packages; default checks configuration only')
    args=p.parse_args()
    if not args.run:
        print(json.dumps({'status':'ready','tasks':TASKS,'protocol':PROTOCOL,'source_batch':str(args.source_batch)},indent=2));return
    state=json.loads((args.source_batch/'status.json').read_text())
    if state['status']!='complete':raise ValueError('Incomplete source batch')
    storage=Path(os.environ.get('ECG_AGENT_STORAGE','/data/1shared/gaojiajun/ECGagent'))
    parent=storage/'datasets/mcmed-ecg-agent'
    report=storage/'runs/mcmed-three-task'/ (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    report.mkdir(parents=True,exist_ok=False);progress={'status':'running','outputs':{}}
    dump(report/'status.json',progress)
    try:
        for group in ('short','full'):
            out=export_package(Path(state[group+'_output']),parent,group)
            progress['outputs'][group]=str(out);dump(report/'status.json',progress)
            print('OUTPUT='+str(out),flush=True)
        progress['status']='complete';dump(report/'status.json',progress);print('REPORT='+str(report),flush=True)
    except Exception as exc:
        progress.update(status='failed',error=str(exc));dump(report/'status.json',progress);raise

if __name__=='__main__':main()
