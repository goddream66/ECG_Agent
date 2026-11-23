"""Run a cold fixed-evidence gate, then the four live AFDB arms."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import run_04936_attribution_pair as base


def main(*, split_event=False, measured_event=False, staged_event=False, replicate=1):
    split_event = split_event or measured_event or staged_event
    old_root = base.OUTPUT
    checkout = Path(__file__).resolve().parents[2]
    experiment = '04936-event-v15' if split_event else '04936-grounding-v14'
    if measured_event:
        experiment = '04936-event-v16'
    if staged_event:
        experiment = '04936-event-v17'
    root = base.STORAGE/'runs'/f'{experiment}-1550-1850-r{replicate}'
    root.mkdir(exist_ok=True,parents=True)
    if (root/'status.json').exists():
        raise FileExistsError(root/'status.json')
    base.OUTPUT = root
    base.CHECKOUTS = checkout.parent
    original = next(r for r in base.entries(base.STORAGE/'data/afdb_tasks_server/agent_tasks.jsonl')
                    if r['record_id']==base.RECORD and r['start_s']<=base.START and r['end_s']>=base.END)
    task = json.loads((old_root/'task.jsonl').read_text())
    if (task['record_id'],task['start_s'],task['end_s'])!=('04936',1550,1850):
        raise ValueError('Unexpected archived task')
    (root/'task.jsonl').write_text(json.dumps(task)+'\n')
    cpus = sorted(os.sched_getaffinity(0))
    free = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    gpus = [int(line.split(',')[0]) for line in free.splitlines() if int(line.split(',')[1])>=21000][:4]
    if len(cpus)<8 or len(gpus)<4:
        raise RuntimeError('Four free GPUs and eight CPUs required')
    def launch(i, name, command, code):
        gpu = gpus[i]
        pair = cpus[2*i:2*i+2]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(code/'src'),
                 ECG_AGENT_CODE=str(code),ECG_AGENT_STORAGE=str(base.STORAGE),HF_HUB_OFFLINE='1',
                 TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HOME=str(base.STORAGE/'cache/huggingface'),
                 TMPDIR=str(base.STORAGE/'tmp'),PYTHONDONTWRITEBYTECODE='1',
                 **{key:'2' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        with (root/(name+'.log')).open('x') as log:
            proc = subprocess.Popen(['taskset','-c',','.join(map(str,pair)),sys.executable]+list(map(str,command)),
                                    cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
        return proc,dict(name=name,gpu=gpu,cpus=pair,pid=proc.pid),time.monotonic()
    def finish(group, stage):
        while True:
            for proc,job,start in group:
                if 'exit_code' not in job and proc.poll() is not None:
                    job.update(exit_code=proc.returncode,wall_s=round(time.monotonic()-start,2))
            jobs=[job for _,job,_ in group]
            base.save('status.json',dict(stage=stage,jobs=jobs))
            if all('exit_code' in j for j in jobs):
                if any(j['exit_code'] for j in jobs):
                    raise RuntimeError('Worker failed: '+str(jobs))
                return jobs
            time.sleep(5)
    if not split_event:
        fixed=[]
        old_code=Path.home()/'ECGagent-experiments/04936-attribution-v13/new'
        for version,code in [('v13',old_code),('v14',checkout)]:
            for lead in ('ecg1','ecg2'):
                name=f'frozen-{version}-{lead}'
                source=old_root/f'new-autonomous-{lead}'/f'{lead}-event-react/records/04936/evidence.jsonl'
                fixed.append(launch(len(fixed),name,[Path(__file__).with_name('replay_04936_frozen.py'),
                                    '--source',source,'--config',code/'configs/ecg/react_qwen_episode_fixed.json',
                                    '--output',root/name],code))
        frozen_jobs=finish(fixed,'frozen')
        base.save('frozen_results.json',{j['name']:json.loads((root/j['name']/'summary.json').read_text()) for j in frozen_jobs})
    live=[]
    for mode in ('fixed','autonomous'):
        for lead in ('ECG1','ECG2'):
            name=f'new-{mode}-{lead.lower()}'
            live.append(launch(len(live),name,['-m','cardioagent.ecg_agent.experiments.runner',
                      '--mode','live','--policies','event-react','--workers','1','--threads','2','--device','cuda:0',
                      '--tasks',root/'task.jsonl','--channels',lead,'--llm-config',
                      checkout/'configs/ecg'/('react_qwen_episode_'+mode+('_v17' if staged_event else '_v16' if measured_event else '_v15' if split_event else '')+'.json'),'--output',root/name],checkout))
    jobs=finish(live,'live')
    base.summarize(jobs,task,original['task_id'])
    base.save('status.json',dict(stage='complete',jobs=jobs))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--split-event',action='store_true')
    parser.add_argument('--measured-event',action='store_true')
    parser.add_argument('--replicate',type=int,choices=(1,2),default=1)
    parser.add_argument('--staged-event',action='store_true')
    args = parser.parse_args()
    main(split_event=args.split_event, measured_event=args.measured_event, replicate=args.replicate, staged_event=args.staged_event)
