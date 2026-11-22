"""Paired 1550–1850 s live ECGFounder/Qwen replay. Run from the isolated server deployment."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter, deque


CHECKOUTS = Path.home() / 'ECGagent-experiments/04936-attribution-v13'
STORAGE = Path('/data/1shared/gaojiajun/ECGagent')
OUTPUT = STORAGE / 'runs/04936-attribution-v13-1550-1850-r1'
RECORD = '04936'
START, END = 1550, 1850
VERSIONS = {'old':'03971d6177e01e06888e30d7f43b3c1ae3571c39',
            'new':'45330e89d7a94260b0df16fed2cf0b05fd3ee46d'}


def save(name, data):
    path = OUTPUT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def entries(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def summarize(jobs, task, original_id):
    sys.path.insert(0,str(CHECKOUTS/'new/src'))
    from cardioagent.common.intervals import union_seconds
    from cardioagent.ecg_agent.evaluation.metrics import evaluate_record
    refs = STORAGE/'data/afdb_tasks_server/evaluation_references.jsonl'
    full = next(r for r in entries(refs) if r['task_id'] == original_id)
    truth = [[max(START,a),min(END,b)] for a,b in full['intervals'] if a < END and b > START]
    reference = dict(start_s=START,end_s=END,intervals=truth,burden=union_seconds(truth)/(END-START),
                     tau=full.get('tau',.5),burden_floor=full.get('burden_floor',.02))
    rows = []
    for job in jobs:
        if job.get('exit_code') != 0:
            continue
        folder = next((OUTPUT/job['name']).glob('*/records/04936/result.json')).parent
        result = json.loads((folder/'result.json').read_text())
        decisions = list(entries(folder/'window_trace.jsonl'))
        metrics = evaluate_record(task['task_id'], result['intervals'], reference,
                                  unknown=result['unknown_intervals'],burden_intervals=result['burden_intervals'],
                                  updates=list(entries(folder/'event_updates.jsonl')),events=result['events'])
        rows.append(dict(job=job['name'],gpu=job['gpu'],wall_s=job['wall_s'],metrics=metrics,
                         event_intervals=result['intervals'],support_intervals=result['burden_intervals'],
                         conflict_intervals=result.get('conflict_intervals'),
                         actions=dict(Counter(d.get('action') or d['decision'] for d in decisions)),
                         counts=result['counts'],execution=result['execution'],logs=str(folder)))
    save('paired_results.json',dict(record=RECORD,span=[START,END],reference=reference,arms=rows,jobs=jobs))


def main():
    OUTPUT.mkdir(exist_ok=True,parents=True)
    for name,commit in VERSIONS.items():
        got = subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUTS/name,text=True).strip()
        if got != commit:
            raise ValueError(f'Unexpected {name} version: {got}')
    source = STORAGE/'data/afdb_tasks_server/agent_tasks.jsonl'
    all_tasks = [r for r in entries(source) if r['record_id']==RECORD and r['start_s']<=START and END<=r['end_s']]
    if len(all_tasks)!=1:
        raise ValueError('Expected one full AFDB 04936 inference task')
    original = all_tasks[0]
    task = {**original,'task_id':f'afdb-04936-window-{START}-{END}','start_s':START,'end_s':END}
    (OUTPUT/'task.jsonl').write_text(json.dumps(task)+'\n')
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus)<12:
        raise RuntimeError('Need 12 available CPUs')
    gpus = list(range(6))
    free = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    avail = dict(tuple(map(int,line.split(','))) for line in free.splitlines())
    if any(avail.get(g,0) < 21000 for g in gpus):
        raise RuntimeError('Six selected GPUs need at least 21000 MiB free each')
    jobs = []
    for mode in ('fixed','autonomous'):
        for channel in ('ECG1','ECG2'):
            for version in VERSIONS:
                jobs.append(dict(name=f'{version}-{mode}-{channel.lower()}',version=version,mode=mode,channel=channel))
    queue, active = deque(jobs), {}
    while queue or active:
        for gpu in gpus:
            if gpu in active:
                process, stream, job, started = active[gpu]
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                job.update(exit_code=code,wall_s=round(time.monotonic()-started,3))
                del active[gpu]
                summarize(jobs,task,original['task_id'])
            if not queue:
                continue
            job = queue.popleft()
            name = job['name']
            checkout = CHECKOUTS/job['version']
            pair = cpus[2*gpu:2*gpu+2]
            job.update(gpu=gpu,cpus=pair)
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(checkout/'src'),
                       ECG_AGENT_CODE=str(checkout),ECG_AGENT_STORAGE=str(STORAGE),PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',
                       TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                       HF_HOME=str(STORAGE/'cache/huggingface'),TMPDIR=str(STORAGE/'tmp'))
            cmd = ['taskset','-c',','.join(map(str,pair)),sys.executable,'-m','cardioagent.ecg_agent.experiments.runner',
                   '--mode','live','--policies','event-react','--workers','1','--threads','2','--device','cuda:0',
                   '--tasks',str(OUTPUT/'task.jsonl'),'--channels',job['channel'],'--llm-config',
                   str(checkout/'configs/ecg'/f"react_qwen_episode_{job['mode']}.json"),'--output',str(OUTPUT/name)]
            stream = (OUTPUT/(name+'.log')).open('a',buffering=1)
            child = subprocess.Popen(cmd,env=env,cwd=checkout,stdout=stream,stderr=subprocess.STDOUT)
            job['pid'] = child.pid
            active[gpu]=(child,stream,job,time.monotonic())
        save('status.json',dict(versions=VERSIONS,jobs=jobs))
        if active:
            time.sleep(10)
    return int(any(j.get('exit_code')!=0 for j in jobs))


if __name__ == '__main__':
    raise SystemExit(main())
