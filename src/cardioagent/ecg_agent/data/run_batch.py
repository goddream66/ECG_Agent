"""Run the requested ten short selections plus a distinct complete ECG patient."""
import json
import os
from pathlib import Path
from datetime import datetime,timezone
from uuid import uuid4
from cardioagent.common.paths import code_root
from cardioagent.ecg_agent.data.label_mcmed import load_config,configure_device
from cardioagent.ecg_agent.data.build_mcmed import build,select_segments,dump,sha
from cardioagent.ecg_agent.data.select_full_patient import select_full_patient

def main():
    args=load_config(code_root()/'configs/ecg/mcmed_labeling.json');configure_device(args.device)
    storage=Path(os.environ['ECG_AGENT_STORAGE'])
    batch=storage/'runs/mcmed-labeling-batch'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    batch.mkdir(parents=True,exist_ok=False)
    state={'status':'selecting','short_output':None,'full_output':None}
    dump(batch/'status.json',state);print('BATCH='+str(batch),flush=True)
    try:
        short,stats=select_segments(args.source_root,args.split,args.patients,args.duration_s,args.seed,args.max_patient_scan)
        if len(short)!=10:raise ValueError('Expected exactly ten selected patients')
        full,full_stats=select_full_patient(args.source_root,args.split,{r['patient_id'] for r in short},args.seed)
        args.selection_scope='ten_patients_first_600_seconds'
        dump(batch/'selection.json',{'short':short,'short_selection':stats,'full':full,'full_selection':full_stats,
             'selector_sha256':sha(Path(__file__).with_name('select_full_patient.py'))})
        state.update(status='labeling_short',full_selection=full_stats)
        dump(batch/'status.json',state)
        print('SELECTION='+json.dumps({'short_patients':len(short),'full_patient':full_stats}),flush=True)
        state['short_output']=str(build(args,selected_records=short,selection_info=stats))
        state['status']='labeling_full';dump(batch/'status.json',state)
        args.patients=1;args.duration_s=None;args.selection_scope=full_stats['scope']
        state['full_output']=str(build(args,selected_records=full,selection_info=full_stats))
        state['status']='complete';dump(batch/'status.json',state)
        print('BATCH_COMPLETE='+str(batch),flush=True)
    except Exception as exc:
        state.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
        dump(batch/'status.json',state);raise

if __name__=='__main__':main()
