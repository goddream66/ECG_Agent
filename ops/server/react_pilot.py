"""Four independent GPUs, two physical cores each; bounded AFDB ReAct pilot."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time


STORAGE = Path('/data/1shared/gaojiajun/ECGagent')
CODE = Path('/home/gaojiajun/ECGagent')


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs').mkdir()
    tasks = [json.loads(x) for x in (STORAGE/'data/afdb_tasks_server/agent_tasks.jsonl').read_text().splitlines()][:4]
    workers=[]
    for gpu, original in enumerate(tasks):
        task=dict(original, end_s=min(original['end_s'], original['start_s']+1800))
        path=root/'inputs'/f'gpu{gpu}.jsonl'
        path.write_text(json.dumps(task)+'\n')
        cfg=json.loads((CODE/'configs/ecg/react_qwen_local.json').read_text())
        cfg_path=root/'inputs'/f'gpu{gpu}-llm.json';save(cfg_path,cfg)
        workers.append({'gpu':gpu,'cpus':[gpu*2,gpu*2+1], 'record_id':task['record_id'],
                        'tasks':str(path),'llm_config':str(cfg_path)})
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    save(root/'experiment.json',{'status':'prepared','commit':commit,'workers':workers,
        'scope':'4 records x first 1800 seconds x ECG1/ECG2, exploratory pilot',
        'policies':['event-adaptive','event-react'],'created_unix':time.time()})
    return workers


def launch(root, w, phase):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(w['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    output=root/phase/f'gpu{w["gpu"]}'
    args=['taskset','-c',','.join(map(str,w['cpus'])),sys.executable,'-m',
        'cardioagent.ecg_agent.experiments.runner','--mode','live','--device','cuda:0',
        '--workers','1','--threads','2','--tasks',w['tasks'],
        '--config',str(CODE/'configs/ecg/policy_hysteresis060040.json'),
        '--llm-config',w['llm_config'],'--policies','event-adaptive','event-react',
        '--channels','ECG1','ECG2','--duration-s','20' if phase=='smoke' else '1800',
        '--output',str(output)]
    log=(root/f'{phase}-gpu{w["gpu"]}.log').open('w')
    proc=subprocess.Popen(args,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    log.close()
    return {'process':proc,'output':output,'worker':w,'command':args}


def validate(root, entries, phase):
    from cardioagent.common.io import digest
    from cardioagent.ecg_agent.memory.evidence import canonical_hash
    audits=[]
    for entry in entries:
        output=entry['output']; w=entry['worker']
        completion=json.loads((output/'completion.json').read_text())
        assert completion['completed']==4
        for folder in output.glob('ecg*-event-*/records/*'):
            manifest=json.loads((folder/'job_manifest.json').read_text())
            assert manifest['job_hash']==canonical_hash(manifest['job'])
            for name, expected in manifest['artifacts'].items():
                assert digest(folder/name)==expected,(folder,name)
            result=json.loads((folder/'result.json').read_text())
            assert result['execution']['device']=='cuda:0'
            assert result['execution']['model_info']['cuda_visible_devices']==str(w['gpu'])
            assert result['execution']['actual_model_calls']==result['counts']['model_calls']
            steps=[]
            if result['policy']=='event-react':
                steps=[json.loads(x) for x in (folder/'llm_steps.jsonl').read_text().splitlines()]
                assert result['counts'].get('llm_calls',0)>0
                errors=[x for x in steps if x['status']=='error']
                # Infrastructure/protocol failures must be fixed in smoke, not hidden in a long run.
                if phase=='smoke' and errors:
                    raise RuntimeError(f'ReAct smoke failures: {folder}: {errors}')
            audits.append({'record':result['record_id'],'channel':result['channel'],
                           'policy':result['policy'],'counts':result['counts'],
                           'llm_errors':sum(x['status']=='error' for x in steps)})
    save(root/f'{phase}-audit.json',audits)


def analyze(root, workers):
    # Ground truth is loaded here only, after inference; it never enters a worker.
    from cardioagent.common.io import read_jsonl, digest
    from cardioagent.common.intervals import union_seconds
    from cardioagent.ecg_agent.evaluation.metrics import prepare_validity, evaluate_record, aggregate, paired_bootstrap
    from cardioagent.ecg_agent.evaluation.intervals import burden_score
    from cardioagent.ecg_agent.experiments.analysis import trace_data
    out=root/'analysis';out.mkdir(exist_ok=True)
    selected=[json.loads(Path(w['tasks']).read_text()) for w in workers]
    tasks=out/'tasks.jsonl';tasks.write_text(''.join(json.dumps(t)+'\n' for t in selected))
    cache=STORAGE/'data/afdb_preprocessed_server'
    raw=json.loads((cache/'manifest.json').read_text())['raw_root']
    valid=prepare_validity(tasks,raw,cache,out/'validity.json')
    refs=read_jsonl(STORAGE/'data/afdb_tasks_server/evaluation_references.jsonl','task_id')
    results=[]
    for channel in ('ECG1','ECG2'):
        for policy in ('event-adaptive','event-react'):
            rows=[]
            for worker,task in zip(workers,selected):
                folder=root/'live'/f'gpu{worker["gpu"]}'/f'{channel.lower()}-{policy}'/'records'/worker['record_id']
                result=json.loads((folder/'result.json').read_text())
                ref=dict(refs[task['task_id']])
                a,b=task['start_s'],task['end_s']
                ref.update(start_s=a,end_s=b,intervals=[[max(a,c),min(b,d)] for c,d in ref['intervals'] if min(b,d)>max(a,c)])
                ref['burden']=union_seconds(ref['intervals'])/(b-a)
                trace=[json.loads(x) for x in (folder/'window_trace.jsonl').read_text().splitlines()]
                _,unknown,_=trace_data(trace)
                updates=[json.loads(x) for x in (folder/'event_updates.jsonl').read_text().splitlines()]
                row=evaluate_record(task['task_id'],result['intervals'],ref,
                    valid=valid[task['task_id']]['valid_intervals'],unknown=unknown,updates=updates,events=result['events'])
                row.update(result['counts']); row['record_id']=worker['record_id']
                row['burden_score']=burden_score(union_seconds(result['intervals'])/(b-a),ref['burden'],ref.get('burden_floor',.02))
                rows.append(row)
            metrics=aggregate(rows)
            for key in ('llm_calls','llm_ms','llm_prompt_tokens','llm_completion_tokens','tool_calls','model_calls'):
                metrics[key]=sum(r.get(key,0) for r in rows)
            metrics['mean_burden_score']=sum(r['burden_score'] for r in rows)/len(rows)
            results.append({'channel':channel,'policy':policy,'metrics':metrics,'rows':rows})
    paired=[]
    for ch in ('ECG1','ECG2'):
        r=[x for x in results if x['channel']==ch]
        paired.append(dict(channel=ch,**paired_bootstrap(r[1]['rows'],r[0]['rows'],repetitions=2000,seed=20261114)))
    save(out/'comparison.json',{'results':results,'paired':paired,'scope':'4-record cropped exploratory pilot',
         'references_sha256':digest(STORAGE/'data/afdb_tasks_server/evaluation_references.jsonl')})
    keys=['micro_episode_f1','tp','fp','fn','time_precision','time_recall','non_af_alarm_minutes',
          'unknown_coverage','long_fragment_excess','model_calls','llm_calls','llm_ms']
    lines=['# Qwen ReAct pilot comparison','','4 records, first 30 minutes, both main channels. Exploratory only.',
           '','| Channel / policy | '+' | '.join(keys)+' |','|---|'+'---|'*len(keys)]
    for row in results:
        lines.append('| '+row['channel']+' / '+row['policy']+' | '+' | '.join(str(row['metrics'].get(k)) for k in keys)+' |')
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,title in zip(axes,['micro_episode_f1','non_af_alarm_minutes','unknown_coverage'],['Event F1','Non-AF alarm minutes','Unknown coverage']):
        x=list(range(4)); ax.bar(x,[r['metrics'].get(key) or 0 for r in results],color=['#64748b','#0d9488']*2)
        ax.set_xticks(x,['ECG1 rule','ECG1 ReAct','ECG2 rule','ECG2 ReAct'],rotation=25,ha='right');ax.set_title(title)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)


def run(root):
    workers=prepare(root)
    manifest=json.loads((root/'experiment.json').read_text())
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(root/'environment.txt').open('w'),check=True)
    try:
        for phase in ('smoke','live'):
            manifest['status']=phase;save(root/'experiment.json',manifest)
            entries=[launch(root,w,phase) for w in workers]
            save(root/f'{phase}-processes.json',[{'pid':e['process'].pid,'worker':e['worker'],'command':e['command']} for e in entries])
            while any(e['process'].poll() is None for e in entries):
                failed=[e for e in entries if e['process'].poll() not in (None,0)]
                if failed:
                    for e in entries:
                        if e['process'].poll() is None:e['process'].terminate()
                    raise RuntimeError(f'Worker failed: {failed[0]["worker"]}')
                time.sleep(5)
            if any(e['process'].returncode for e in entries):raise RuntimeError('Worker failed')
            validate(root,entries,phase)
        analyze(root,workers)
        manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:
        manifest['updated_unix']=time.time();save(root/'experiment.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
