"""Build versioned MC-MED ECG indexes or offline weak-label review packages.

python -m cardioagent.ecg_agent.data.build_mcmed --help
Never changes the source dataset; no human labels are inferred from diagnoses.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from time import perf_counter
from uuid import uuid4

import numpy as np
import wfdb
from cardioagent.common.paths import code_root
from cardioagent.common.recording import Waveform
from cardioagent.ecg_agent.data.preprocessing import invalid_spans
from cardioagent.ecg_agent.data.ecgfounder_preprocessing import preprocess_window, PROFILE, load_profile, profile_path
from cardioagent.ecg_agent.data.label_quality import POLICY as QUALITY_POLICY, summarize_quality
from cardioagent.ecg_agent.harness import ToolContext, ToolSession
from cardioagent.ecg_agent.schemas import EvidenceContext

DATASET = 'mcmed-ecg-agent'
VERSION = 'v0.4.0'
DISPLAY_CANDIDATES = 5
SUMMARY_LABELS = {'ABNORMAL ECG', 'BORDERLINE ECG', 'OTHERWISE NORMAL ECG'}
TOOLS = ('detect_beats_featuredb', 'extract_average_beat_featuredb', 'assess_template_quality')
LABEL_MAPPING = {'af': ['ATRIAL FIBRILLATION'], 'afl': ['ATRIAL FLUTTER'],
 'pvc_window': ['PREMATURE VENTRICULAR COMPLEXES'],
 'sinus_rhythm': ['SINUS RHYTHM', 'NORMAL SINUS RHYTHM'], 'normal_ecg': ['NORMAL ECG']}


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write_line(stream,obj):
    stream.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n')


def patient_index(root, split):
    with (root/f'split_random_{split}.csv').open(newline='') as f:
        allowed={row[0].strip() for row in csv.reader(f) if row and row[0].strip()!='CSN'}
    patients={}
    with (root/'visits.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            # No diagnoses, outcomes, reports or laboratory values enter this index.
            if row['CSN'] in allowed:
                patients.setdefault(row['MRN'],[]).append(row['CSN'])
    return patients


def select_segments(root, split, limit, duration, seed, max_scan):
    patients=patient_index(root,split)
    order=sorted(patients,key=lambda p:hashlib.sha256(f'{seed}:{p}'.encode()).hexdigest())
    chosen=[];rejected=Counter();scanned=0
    for patient in order[:max_scan]:
        scanned+=1;eligible=[]
        for visit in sorted(patients[patient]):
            folder=root/'waveforms'/visit[-3:]/visit/'II'
            if not folder.is_dir():rejected['no_II_directory']+=1;continue
            for header in sorted(folder.glob('*.hea')):
                try:
                    h=wfdb.rdheader(str(header.with_suffix('')))
                    assert h.n_sig==1 and h.sig_name==['II'], 'expected one II channel'
                    assert h.fs>80 and float(h.fs).is_integer(), 'unsupported sampling rate'
                    assert h.sig_len>=round(duration*h.fs), 'segment_too_short'
                    unit=h.units[0].strip().lower().replace('μ','u').replace('µ','u')
                    assert unit in ('mv','uv','v'), 'unknown_physical_units'
                    files=[]
                    for name in set(h.file_name):
                        p=(header.parent/name).resolve();assert p.is_relative_to(root.resolve()),'data_path_outside_source'
                        assert p.is_file() and p.stat().st_size>0, 'missing_or_empty_dat'
                        files.append(p)
                    eligible.append({'patient_id':patient,'visit_id':visit,'segment_id':header.stem,
                        'header':str(header.resolve()),'data_files':[str(p) for p in files],
                        'fs':float(h.fs),'signal_samples':int(h.sig_len),'unit':h.units[0],
                        'segment_start_datetime':str(h.base_datetime) if h.base_datetime else None,
                        'start_s':0.,'end_s':duration,'split':split})
                except (OSError,ValueError,AssertionError,IndexError) as e:
                    rejected[str(e)[:120]]+=1
        if eligible:
            # Selection is label/model independent; keep one continuous source segment per patient.
            chosen.append(min(eligible,key=lambda r:(r['segment_start_datetime'] or '',r['header'])))
        if len(chosen)==limit:break
    return chosen,{'patients_scanned':scanned,'requested_patients':limit,'selected_patients':len(chosen),
                   'source_rejections':dict(rejected),'max_scan':max_scan}


def windows(record):
    fs=int(record['fs']);count=math.ceil(round(record['end_s']*fs)/(10*fs))
    for i in range(count):
        yield {'window_id':f"{record['visit_id']}:{record['segment_id']}:II:{i*10:09d}",
          'patient_id':record['patient_id'],'visit_id':record['visit_id'],'segment_id':record['segment_id'],
          'channel':'II','start_s':float(i*10),'end_s':min(float((i+1)*10),record['end_s']),
          'start_sample':i*10*fs,'end_sample':min((i+1)*10*fs,round(record['end_s']*fs)),
          'time_origin':'source_segment_start','segment_start_datetime':record['segment_start_datetime'],
          'split':record['split']}


def automatic_labels(scores, labels):
    """Export full unthresholded rankings, with broad descriptions shown separately."""
    scores=np.asarray(scores)
    if (scores.shape!=(150,) or len(labels)!=150 or len(set(labels))!=150
            or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1))):
        raise ValueError('Expected 150 unique labels and finite class scores in [0,1]')
    order=np.argsort(-scores,kind='stable')
    ranked=[{'rank':rank,'label_id':int(index),'label':labels[index],
             'score':float(scores[index]),'summary_label':labels[index].upper() in SUMMARY_LABELS}
            for rank,index in enumerate(order,1)]
    display=[r for r in ranked if not r['summary_label']][:DISPLAY_CANDIDATES]
    first=ranked[0]
    return {'label_id':first['label_id'],'label':first['label'],
            'top1_label':first['label'],'top1_score':first['score'],
            'label_ids':[r['label_id'] for r in display],'labels':[r['label'] for r in display],
            'display_candidates':display,'summary_candidates':[r for r in ranked if r['summary_label']],
            'ranked_candidates':ranked,'label_status':'assigned','selection_rule':'ranked_candidates',
            'score_semantics':'uncalibrated_sigmoid','threshold':None,
            'annotation_source':'ECGFounder-150','human_verified':False}


def candidates(scores, labels, threshold):
    result={key:max(float(scores[labels.index(name)]) for name in names) for key,names in LABEL_MAPPING.items()}
    abnormal=[key for key in ('af','afl','pvc_window') if result[key]>=threshold]
    # Absence of target positives is NOT a normal-ECG label.
    state='target_candidate' if abnormal else 'no_target_candidate'
    if result['normal_ecg']>=threshold and abnormal:state='conflicting_candidates'
    return {'mapped_scores':result,'positive_target_candidates':abnormal,'review_state':state,
            'ground_truth':False,'threshold':threshold}


def build(args, predictor_factory=None, *, selected_records=None, selection_info=None):
    root=args.source_root.resolve();parent=args.output_root.resolve()
    if parent.is_relative_to(root):raise ValueError('Output must not be inside the read-only source dataset')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=parent/VERSION/f'{stamp}-{args.mode}-{uuid4().hex[:8]}'
    output.mkdir(parents=True,exist_ok=False)
    manifest={'dataset':DATASET,'version':VERSION,'status':'building','run_id':output.name,
      'mode':args.mode,'source_root':str(root),'output':str(output),'created_utc':stamp,
      'config':{'split':args.split,'seed':args.seed,'patients':args.patients,'duration_s':args.duration_s,
                'selection_scope':getattr(args,'selection_scope','one_segment_per_patient'),
                'window_s':10,'label_selection':'ranked_candidates','display_candidate_count':DISPLAY_CANDIDATES,'summary_labels':sorted(SUMMARY_LABELS),'quality_policy':QUALITY_POLICY,'save_scores':getattr(args,'save_scores',False),'max_patient_scan':args.max_patient_scan,
                'profile':PROFILE,'preprocessing_config':load_profile(),'preprocessing_config_sha256':sha(profile_path())},
      'claims':{'weak_labels_only':True,'human_ground_truth_created':False,'clinical_accuracy_validated':False,
                'no_full_dataset_integrity_scan':True,'not_a_streaming_agent_evaluation':True},
      'input_tables':{name:sha(root/name) for name in ('visits.csv',f'split_random_{args.split}.csv')},
      'runtime':{'python':platform.python_version(),'device_requested':args.device}}
    source_paths=[Path(__file__),code_root()/'src/cardioagent/ecg_agent/data/preprocessing.py',
                  code_root()/'src/cardioagent/ecg_agent/data/ecgfounder_preprocessing.py',
                  code_root()/'src/cardioagent/ecg_agent/data/label_quality.py',
                  code_root()/'src/cardioagent/ecg_agent/models/ecgfounder.py',
                  code_root()/'src/cardioagent/ecg_agent/tools/physio.py']
    manifest['code_hashes']={p.relative_to(code_root()).as_posix():sha(p) for p in source_paths}
    dump(output/'manifest.json',manifest)
    try:
        if selected_records is None:
            chosen,selection=select_segments(root,args.split,args.patients,args.duration_s,args.seed,args.max_patient_scan)
        else:
            chosen,selection=selected_records,selection_info
            if not selection:raise ValueError('Selected records require selection provenance')
        manifest['selection']=selection
        if not chosen:raise ValueError('No eligible continuous II segment found in the bounded patient selection')
        for r in chosen:
            r['source_sha256']={p:sha(p) for p in [r['header'],*r['data_files']]}
        dump(output/'records.json',chosen)
        mapping={'version':'full-150-ranked-v1','class_count':150,'selection':'stable_descending_scores',
                 'threshold':None,'display_candidates':DISPLAY_CANDIDATES,'summary_labels':sorted(SUMMARY_LABELS),
                 'tie_break':'checkpoint vocabulary order','all_150_scores_retained_in_json':True,
                 'note':'Ranked candidates are not confirmed positive diagnoses. Original first rank and score retained.'}
        dump(output/'label_mapping.json',mapping)
        model=None;labels=None;info=None;counts=Counter();label_counts=Counter();model_ms=load_ms=0.
        if args.mode=='prelabel':(output/'model_inputs').mkdir()
        if args.mode=='prelabel' and getattr(args,'save_scores',False):(output/'model_scores').mkdir()
        if args.mode=='prelabel':(output/'review').mkdir()
        from contextlib import ExitStack
        with ExitStack() as stack:
            index=stack.enter_context((output/'windows.jsonl').open('w',encoding='utf-8'))
            evidence=stack.enter_context((output/'tool_evidence.jsonl').open('w',encoding='utf-8')) if args.mode=='prelabel' else None
            auto_stream=stack.enter_context((output/'auto_labels.jsonl').open('w',encoding='utf-8')) if args.mode=='prelabel' else None
            review=stack.enter_context((output/'review/annotations.jsonl').open('w',encoding='utf-8')) if args.mode=='prelabel' else None
            for number,r in enumerate(chosen,1):
                n=math.ceil(round(r['end_s']*r['fs'])/(10*r['fs']))
                inputs=np.lib.format.open_memmap(output/f"model_inputs/{r['segment_id']}.npy",mode='w+',dtype=np.float32,shape=(n,5000)) if evidence else None
                scores=np.lib.format.open_memmap(output/f"model_scores/{r['segment_id']}.npy",mode='w+',dtype=np.float32,shape=(n,150)) if evidence and getattr(args,'save_scores',False) else None
                if inputs is not None:inputs[:]=np.nan
                if scores is not None:scores[:]=np.nan
                for i,w in enumerate(windows(r)):
                    counts['windows']+=1
                    auto={'window_id':w['window_id'],'start_s':w['start_s'],'end_s':w['end_s'],
                          'label':None,'label_id':None,'labels':None,'label_ids':None,'label_status':'not_evaluated',
                          'top1_label':None,'top1_score':None,'display_candidates':None,'summary_candidates':None,'ranked_candidates':None,
                          'annotation_source':'ECGFounder-150','human_verified':False}
                    if not evidence:
                        w['status']='indexed_not_processed';write_line(index,w);continue
                    try:
                        raw=wfdb.rdrecord(str(Path(r['header']).with_suffix('')),sampfrom=w['start_sample'],sampto=w['end_sample'])
                        assert raw.fs==r['fs'] and raw.sig_name==['II'] and raw.p_signal.shape==(w['end_sample']-w['start_sample'],1),'decoded waveform coordinates disagree'
                        values=raw.p_signal[:,0];spans=invalid_spans(values,int(r['fs']))
                    except (OSError,ValueError,AssertionError,IndexError) as e:
                        w.update(status='unavailable',reason=str(e));counts['unavailable']+=1;counts['model_skipped']+=1
                        auto.update(quality_status='source_unavailable',quality_reasons=[str(e)],quality_evidence_id=None)
                        counts['quality_source_unavailable']+=1
                        write_line(index,w)
                        write_line(auto_stream,auto|{'reason':w['reason']})
                        write_line(review,{'window_id':w['window_id'],'review_status':'pending','human_labels':None,'model_labels':None,'source_status':w['status'],
                            'quality_status':auto['quality_status'],'quality_reasons':auto['quality_reasons']});continue
                    if w['end_sample']-w['start_sample'] < 10*int(r['fs']):
                        w.update(status='insufficient',reason='short_final_window_no_padding')
                        auto.update(reason=w['reason'],quality_status='incomplete_window',quality_reasons=[w['reason']],quality_evidence_id=None)
                        counts['insufficient']+=1;counts['model_skipped']+=1;counts['quality_incomplete_window']+=1
                        write_line(index,w);write_line(auto_stream,auto)
                        write_line(review,{'window_id':w['window_id'],'review_status':'pending','human_labels':None,
                            'model_labels':None,'source_status':w['status'],'reason':w['reason']})
                        continue
                    w['invalid_spans']=[{'start_sample':a,'end_sample':b,'reason':why} for a,b,why in spans]
                    context=ToolContext(Waveform(raw.p_signal,r['fs'],('II',),tuple(raw.units)),w['start_s'],w['end_s'])
                    session=ToolSession(context,allow_experimental=True,evidence_context=EvidenceContext(output.name,w['window_id'],r['segment_id']))
                    responses=[session.call(t,{'lead':'II','start_s':w['start_s'],'end_s':w['end_s']}) for t in TOOLS]
                    for response in responses:write_line(evidence,{'window_id':w['window_id'],**response});counts['tool_'+response['status']]+=1
                    w['evidence_ids']=[response['evidence_id'] for response in responses]
                    quality=summarize_quality(spans,responses)
                    w['quality']=quality
                    auto.update(quality_status=quality['status'],quality_reasons=quality['reasons'],
                                quality_evidence_id=quality['quality_evidence_id'])
                    counts['quality_'+quality['status']]+=1
                    if quality['skip_founder']:
                        w.update(status='insufficient',reason='missing_or_suspected_flatline');counts['insufficient']+=1
                        counts['model_skipped']+=1
                        auto['reason']=w['reason']
                    else:
                        unit=r['unit'].lower().replace('μ','u').replace('µ','u');factor={'mv':1.,'uv':.001,'v':1000.}[unit]
                        try:
                            processed=preprocess_window(values*factor,int(r['fs']))
                        except ValueError as exc:
                            w.update(status='insufficient',reason='preprocessing_failed: '+str(exc))
                            counts['insufficient']+=1;counts['model_skipped']+=1
                            counts['quality_'+quality['status']]-=1
                            counts['quality_preprocessing_failed']+=1
                            quality.update(status='preprocessing_failed',skip_founder=True,
                                           reasons=quality['reasons']+[w['reason']])
                            auto.update(reason=w['reason'],quality_status=quality['status'],quality_reasons=quality['reasons'])
                            write_line(index,w);write_line(auto_stream,auto)
                            write_line(review,{'window_id':w['window_id'],'review_status':'pending',
                                'human_labels':None,'model_labels':None,'source_status':w['status'],
                                'quality_status':auto['quality_status'],'quality_reasons':auto['quality_reasons'],'reason':w['reason']})
                            continue
                        inputs[i]=processed
                        if model is None:
                            started=perf_counter()
                            if predictor_factory is None:
                                from cardioagent.ecg_agent.models.ecgfounder import create_ecgfounder_label_scorer
                                predictor_factory=create_ecgfounder_label_scorer
                            model,labels,info=predictor_factory(args.models,device=args.device)
                            if len(labels)!=150 or len(set(labels))!=150:raise ValueError('Expected 150 unique checkpoint labels')
                            if 'checkpoint' in info:info['checkpoint_sha256']=sha(info['checkpoint'])
                            info['labels_sha256']=sha(code_root()/'third_party/ecgfounder/tasks.txt')
                            info['network_sha256']=sha(code_root()/'third_party/ecgfounder/net1d.py')
                            dump(output/'labels.json',labels);load_ms=(perf_counter()-started)*1000
                        started=perf_counter();row=np.asarray(model(processed),dtype=np.float32);model_ms+=(perf_counter()-started)*1000
                        if row.shape!=(150,) or not np.isfinite(row).all() or np.any((row<0)|(row>1)):raise ValueError('Invalid 150-label prediction')
                        if scores is not None:
                            scores[i]=row;w.update(score_row=i,model_file=f"model_scores/{r['segment_id']}.npy")
                        counts['model_calls']+=1;w.update(status='labeled',input_file=f"model_inputs/{r['segment_id']}.npy")
                        auto.update(automatic_labels(row,labels))
                        counts['labels_'+auto['label_status']]+=1;label_counts[auto['label']]+=1
                    write_line(index,w)
                    write_line(auto_stream,auto)
                    write_line(review,{'window_id':w['window_id'],'review_status':'pending','human_labels':None,'model_labels':auto['labels'],'model_candidates':auto['display_candidates'],
                        'top1_label':auto['top1_label'],'top1_score':auto['top1_score'],'source_status':w['status'],
                        'quality_status':auto['quality_status'],'quality_reasons':auto['quality_reasons']})
                if inputs is not None:inputs.flush();del inputs
                if scores is not None:scores.flush();del scores
                print(json.dumps({'selected_record':number,'records':len(chosen),'counts':dict(counts)}),flush=True)
        (output/'README.md').write_text('# MC-MED ECG Agent development package\n\nAutomatic ECGFounder class labels; not human-verified gold labels. See manifest.json for selection, preprocessing and provenance.\nWindows use source-segment-relative time. Do not concatenate segments. Unavailable windows have no model label.\nIndex mode does not execute tools/models; prelabel mode writes auto_labels.jsonl with original class names and review templates. Scores are optional (--save-scores).\nThe independent ECGFounder filter+z-score profile is recorded in the manifest. All 150 uncalibrated scores and original first rank are retained; five non-summary candidates are displayed separately from broad summary labels. Candidate ranking is not a confirmed positive-label decision.\n',encoding='utf-8')
        dump(output/'label_summary.json',{'counts':dict(counts),'top1_label_counts':dict(label_counts),
            'quality_policy':QUALITY_POLICY,'all_labels_require_human_review':True})
        manifest.update(status='complete',counts=dict(counts),model=info,
            timing={'model_load_ms':load_ms,'neural_execution_ms':model_ms},
            selection_complete=len({r['patient_id'] for r in chosen})==args.patients,
            artifacts={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file() and p.name!='manifest.json' and 'review' not in p.parts})
        dump(output/'manifest.json',manifest)
        return output
    except Exception as e:
        manifest.update(status='failed',error={'type':type(e).__name__,'message':str(e)})
        dump(output/'manifest.json',manifest)
        raise


def main():
    storage=Path(os.environ.get('ECG_AGENT_STORAGE','/data/1shared/gaojiajun/ECGagent'))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path('/data/1shared/xiaoyujie/mc-med'))
    p.add_argument('--output-root',type=Path,default=storage/'datasets'/DATASET)
    p.add_argument('--mode',choices=('index','prelabel'),default='index')
    p.add_argument('--split',choices=('train','val','test'),default='train')
    p.add_argument('--patients',type=int,default=10)
    p.add_argument('--duration-s',type=int,default=600)
    p.add_argument('--max-patient-scan',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261116)
    p.add_argument('--save-scores',action='store_true',help='Optionally retain full score arrays alongside direct labels')
    p.add_argument('--models',type=Path,default=storage/'models')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.patients<1 or args.max_patient_scan<args.patients or args.duration_s<10 or args.duration_s%10:p.error('Invalid patient/window configuration')
    from cardioagent.ecg_agent.models.ecgfounder import normalize_ecg_device
    args.device=normalize_ecg_device(args.device)
    if args.mode=='prelabel' and args.device.startswith('cuda:'):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
        if ',' in os.environ['CUDA_VISIBLE_DEVICES']:p.error('Expose one GPU only using CUDA_VISIBLE_DEVICES')
    print('DATASET_OUTPUT='+str(build(args)),flush=True)


if __name__=='__main__':main()
