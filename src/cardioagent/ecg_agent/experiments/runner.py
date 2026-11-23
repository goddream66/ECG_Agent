"""Run M1 event experiments with record-level checkpoints and causal model execution."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from cardioagent.common.io import digest, save_json
from cardioagent.ecg_agent.memory.evidence import canonical_hash
from cardioagent.ecg_agent.memory.events import EventConfig
from cardioagent.ecg_agent.schemas import EvidenceContext
from cardioagent.ecg_agent.agent import replay_event_record
from cardioagent.ecg_agent.experiments.replay import fixed_evidence_record
from cardioagent.ecg_agent.harness.logging import JsonLogs
from cardioagent.common.io import read_jsonl, write_jsonl
from cardioagent.ecg_agent.models.ecgfounder import normalize_ecg_device, ecgfounder_execution


_MODEL = None
_LOAD_MS = 0.
_MODEL_INFO = None






def run_job(job):
    if (job.get('llm') or {}).get('event_review_mode'):
        job = {**job, 'episode_decision_protocol': 'attributed-evidence-v1'}
    target = Path(job['target'])
    if target.exists():
        result_file = target / 'result.json'
        manifest_file = target / 'job_manifest.json'
        if not result_file.exists() or not manifest_file.exists():
            raise ValueError(f'Incomplete checkpoint: {target}')
        saved = json.loads(manifest_file.read_text(encoding='utf-8'))
        if saved['job_hash'] != canonical_hash(job):
            raise ValueError(f'Checkpoint provenance mismatch: {target}')
        for name, expected in saved['artifacts'].items():
            if digest(target / name) != expected:
                raise ValueError(f'Corrupt checkpoint artifact: {target / name}')
        return json.loads(result_file.read_text(encoding='utf-8')), True
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if (job.get('llm') or {}).get('event_review_mode'):
        from .episode_checkpoint import EpisodeCheckpoint
        checkpoint = EpisodeCheckpoint(target, job)
    with (checkpoint or TemporaryDirectory(prefix=f'.{target.name}-', dir=target.parent)) as tmp:
        staging = Path(tmp) / 'result'
        logs = JsonLogs(staging, append=bool(checkpoint))
        context = EvidenceContext(job['run_id'], job['task']['task_id'], job['record']['record_id'],
                                  job['manifest_id'])
        config = EventConfig(**job['config'])
        prior_execution = checkpoint.saved.get('execution', {}) if checkpoint else {}
        neural_ms = prior_execution.get('neural_ms', 0.)
        actual_calls = prior_execution.get('actual_calls', 0)
        def live(lead, values, location):
            nonlocal neural_ms, actual_calls
            began = perf_counter()
            actual_calls += 1
            try:
                return _MODEL(values)
            finally:
                neural_ms += (perf_counter() - began) * 1000
                if checkpoint:
                    checkpoint.execution.update(neural_ms=neural_ms, actual_calls=actual_calls)
        if checkpoint:
            checkpoint.execution.update(prior_execution)
        began = perf_counter()
        reviewer = None
        llm_load_ms = 0.
        try:
            if job.get('llm') is not None:
                from cardioagent.ecg_agent.llm.providers import build_chat_model
                from cardioagent.ecg_agent.harness.react import ReActReview
                load_began = perf_counter()
                model = build_chat_model(job['llm'])
                if job['llm'].get('event_review_mode'):
                    from cardioagent.ecg_agent.llm.recording import RecordingChatModel
                    model = RecordingChatModel(model, staging / 'qwen_calls.jsonl.gz')
                reviewer = ReActReview(model,
                    max_rounds=job['llm'].get('max_rounds', 5),
                    llm_tool_max_calls=job['llm'].get('llm_tool_max_calls'),
                    llm_window_interpretation=job['llm'].get('llm_window_interpretation', False),
                    investigation_checkpoint=job['llm'].get('investigation_checkpoint', False),
                    history_windows=job['llm'].get('history_windows', 6),
                    fact_history=job['llm'].get('fact_history', False),
                    event_review_mode=job['llm'].get('event_review_mode'),
                    split_event_review=job['llm'].get('split_event_review',False),
                    measured_event_review=job['llm'].get('measured_event_review',False),
                    staged_event_review=job['llm'].get('staged_event_review',False),
                    window_assessment_mode=job['llm'].get('window_assessment_mode','legacy'),
                    window_band_profile=job['llm'].get('window_band_profile'),
                    end_range_prompt=job['llm'].get('end_range_prompt',False),
                    confirm_end_with_followup=job['llm'].get('confirm_end_with_followup',False),
                    compare_rhythm_leads=job['llm'].get('compare_rhythm_leads',False),
                    bound_window_choices=job['llm'].get('bound_window_choices',False),
                    direction_only_reference=job['llm'].get('direction_only_reference',False),
                    compact_final_evidence=job['llm'].get('compact_final_evidence', True),
                    memory_write=job['llm'].get('memory_write', False),
                    memory_retrieve=job['llm'].get('memory_retrieve', False),
                    boundary_tools=job['llm'].get('boundary_tools', False),
                    event_control=job['llm'].get('event_control', False),
                    unified_event_decision=job['llm'].get('unified_event_decision', False),
                    separate_final_decision=job['llm'].get('separate_final_decision', False),
                    rule_safety_fallback=job['llm'].get('rule_safety_fallback', False),
                    live_scoring=job['mode'] == 'live')
                llm_load_ms = (perf_counter() - load_began) * 1000
            if job['mode'] == 'fixed':
                with Path(job['source_trace']).open(encoding='utf-8') as stream:
                    rows = (json.loads(line) for line in stream if line.strip())
                    result = fixed_evidence_record(rows, job['task'], config, logs, context)
            else:
                if job['mode'] == 'cached':
                    values = np.load(Path(job['scores']) / f"{job['record']['record_id']}.npy", mmap_mode='r')
                    def score(lead, samples, location):
                        return tuple(map(float, values[location]))
                else:
                    score = live
                result = replay_event_record(job['cache'], job['record'], job['task'], job['channel'],
                                             config, score, logs, context, policy=job['policy'],
                                             raw_root=job['raw_root'], reviewer=reviewer,
                                             **({'checkpoint': checkpoint} if checkpoint else {}))
        finally:
            logs.close()
        result['execution'] = {'mode': job['mode'], 'actual_model_calls': actual_calls,
                               'llm_adapter_load_ms': llm_load_ms,
                               'device': job.get('device', 'cpu' if job['mode'] == 'live' else 'none'),
                               'model_info': _MODEL_INFO if job['mode'] == 'live' else None,
                               'neural_timing': 'wall clock including input transfer, forward and blocking scalar return',
                               'neural_ms': neural_ms, 'worker_model_load_ms': _LOAD_MS,
                               'model_load_accounting': 'once per worker, repeated here for attribution; do not sum',
                               'worker_pid': os.getpid(), 'threads': job['threads'],
                               'record_wall_ms': (perf_counter() - began) * 1000 +
                               (checkpoint.saved.get('elapsed_ms', 0.) if checkpoint else 0.)}
        save_json(staging / 'result.json', result)
        artifacts = {p.name: digest(p) for p in staging.iterdir() if p.is_file()}
        save_json(staging / 'job_manifest.json', {'job_hash': canonical_hash(job), 'job': job,
                                                 'artifacts': artifacts})
        staging.rename(target)
    return result, False


def split_traces(baseline, tasks, output):
    """One immutable copy per record; original files are never edited."""
    for channel in ('ECG1', 'ECG2'):
        source = Path(baseline) / f'{channel.lower()}-adaptive/window_trace.jsonl'
        folder = output / channel
        folder.mkdir(parents=True, exist_ok=True)
        metadata = folder / 'source.json'
        source_hash = digest(source)
        if metadata.exists():
            old = json.loads(metadata.read_text(encoding='utf-8'))
            if old['source_sha256'] != source_hash:
                raise ValueError('Frozen trace source changed')
            for name, expected in old['artifacts'].items():
                if digest(folder / name) != expected:
                    raise ValueError('Frozen per-record trace is corrupt')
            continue
        streams = {}
        try:
            with source.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    task_id = row['task_id']
                    if task_id not in tasks:
                        continue
                    record_id = tasks[task_id]['record_id']
                    if record_id not in streams:
                        streams[record_id] = (folder / f'{record_id}.jsonl').open('w', encoding='utf-8')
                    streams[record_id].write(line)
        finally:
            for stream in streams.values():
                stream.close()
        save_json(metadata, {'source_sha256': source_hash,
                             'artifacts': {p.name: digest(p) for p in folder.glob('*.jsonl')}})


def variants(mode, base=None):
    base = base or EventConfig()
    if mode == 'live':
        return [('event-adaptive', base, 'event-adaptive'), ('legacy-dual', base, 'legacy-dual')]
    if mode == 'fixed':
        return [('fixed-evidence', base, 'fixed'),
                ('fixed-no-end-confirm', replace(base, exit_windows=1), 'fixed'),
                ('fixed-grace-0', replace(base, unknown_grace_s=0), 'fixed'),
                ('fixed-grace-20', replace(base, unknown_grace_s=20), 'fixed')]
    return [('event-adaptive', base, 'event-adaptive'),
            ('event-grace-0', replace(base, unknown_grace_s=0), 'event-adaptive'),
            ('event-grace-20', replace(base, unknown_grace_s=20), 'event-adaptive'),
            *[(f'event-budget-{b}', replace(base, review_budget=b), 'event-adaptive') for b in (1, 2, 3)]]






def run(args):
    execution, workers = execution_settings(args.mode, getattr(args, 'device', 'cpu'), args.workers)
    if workers != args.workers:
        print(json.dumps({'execution': 'single_gpu_single_process', 'device': execution['device'],
                          'requested_workers': args.workers, 'effective_workers': workers}), flush=True)
    root = code_root()
    manifest = json.loads((args.cache / 'manifest.json').read_text(encoding='utf-8'))
    config = EventConfig(**json.loads(args.config.read_text(encoding='utf-8')))
    selected_variants=configured_variants(args.mode,config,getattr(args,'policies',None))
    llm_config = None
    if any(policy == 'event-react' for _, _, policy in selected_variants):
        if not getattr(args, 'llm_config', None):
            raise ValueError('event-react requires --llm-config')
        if workers != 1:
            raise ValueError('Use --workers 1 for the first ReAct experiment')
        llm_config = json.loads(args.llm_config.read_text(encoding='utf-8'))
        if any(k in llm_config for k in ('api_key', 'token', 'password')):
            raise ValueError('Use api_key_env, never credentials in a configuration file')
    if args.mode == 'fixed' and config.threshold != .5:
        raise ValueError('Frozen legacy evidence interpretation requires threshold 0.5')
    tasks = read_jsonl(args.tasks, 'task_id')
    if any({'answer', 'intervals', 'burden'} & task.keys() for task in tasks.values()):
        raise ValueError('Inference tasks must not contain answers')
    if args.limit_records:
        tasks = dict(list(tasks.items())[:args.limit_records])
    if args.duration_s:
        tasks = {key: dict(task, end_s=min(task['end_s'], task['start_s'] + args.duration_s))
                 for key, task in tasks.items()}
    records = {r['record_id']: r for r in manifest['records']}
    for task in tasks.values():
        if task.get('source_database') != 'afdb' or task['record_id'] not in records:
            raise ValueError('AFDB tasks/cache required')
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {'mode': args.mode, 'cache_sha256': digest(args.cache / 'manifest.json'),
                  'tasks_sha256': digest(args.tasks), 'checkpoint_sha256': digest(args.models / '1_lead_ECGFounder.pth'),
                  'inference_source': inference_hashes(root), 'threads': args.threads, 'workers': workers,
                  'execution': execution,
                  'platform': platform.platform(), 'python': platform.python_version(),
                  'torch_version': version('torch'), 'channels': args.channels,
                  'network_sha256': digest(root / 'third_party/ecgfounder/net1d.py'),
                  'labels_sha256': digest(root / 'third_party/ecgfounder/tasks.txt'),
                  'config_sha256': digest(args.config),
                  'raw_files': {p.name: digest(p) for name in sorted({t['record_id'] for t in tasks.values()})
                                for p in (Path(manifest['raw_root']) / f'{name}.hea',
                                          Path(manifest['raw_root']) / f'{name}.dat')},
                  'waveform_files': {p.name: digest(p) for name in sorted({t['record_id'] for t in tasks.values()})
                                     for p in (args.cache / f'{name}.npy', args.cache / f'{name}.windows.jsonl')},
                  'selection': list(tasks), 'duration_s': args.duration_s,
                  'variants': [(name, cfg.to_dict(), policy) for name, cfg, policy in selected_variants]}
    if llm_config is not None:
        from cardioagent.ecg_agent.harness.react import PROMPT, WINDOW_PROMPT
        from cardioagent.ecg_agent.harness.autonomous_decision import PROMPT_PATH
        prompt_path = WINDOW_PROMPT if llm_config.get('llm_window_interpretation') else PROMPT
        if llm_config.get('event_review_mode'):
            prompt_path = PROMPT_PATH
        if llm_config.get('window_assessment_mode') in ('verified','union_reference'):
            profile = llm_config['window_band_profile']
            if manifest['profile'] != profile['cache_profile'] or any(
                provenance[k] != v for k,v in profile['model_sha256'].items()
            ) or any(t['record_id'] in profile['fit_records'] for t in tasks.values()):
                raise ValueError('Frozen window reference requires matching model/preprocessing and separate records')
        provenance['llm'] = dict(config=llm_config, prompt_sha256=digest(prompt_path))
        if llm_config.get('split_event_review'):
            from cardioagent.ecg_agent.harness import split_event
            provenance['llm']['split_event_and_tool_guide_sha256'] = digest(Path(split_event.__file__))
        if llm_config.get('staged_event_review'):
            from cardioagent.ecg_agent.harness import staged_review
            provenance['llm']['staged_review_sha256'] = digest(Path(staged_review.__file__))
            if llm_config.get('window_assessment_mode','legacy') != 'legacy':
                provenance['llm']['window_semantics_sha256'] = digest(Path(staged_review.window_semantics.__file__))
        if llm_config.get('measured_event_review'):
            from cardioagent.ecg_agent.harness import measured_event
            provenance['llm']['measured_event_sha256'] = digest(Path(measured_event.__file__))
        if llm_config.get('provider') == 'transformers':
            model_root = Path(llm_config['model_path'])
            provenance['llm']['model_files'] = {p.name: digest(p) for p in sorted(model_root.iterdir())
                if p.is_file() and (p.suffix in ('.json', '.safetensors') or p.name == 'merges.txt')}
    if args.mode == 'cached':
        saved = json.loads((args.scores / 'provenance.json').read_text(encoding='utf-8'))
        if saved['cache_manifest_sha256'] != provenance['cache_sha256'] or saved['checkpoint_sha256'] != provenance['checkpoint_sha256']:
            raise ValueError('Cached score provenance mismatch')
        if any(saved[key] != provenance[key] for key in ('network_sha256', 'labels_sha256')):
            raise ValueError('Cached score network/label provenance mismatch')
        provenance['scores'] = {p.name: digest(p) for p in sorted(args.scores.glob('*.npy'))}
    if args.mode == 'fixed':
        split_traces(args.baseline, tasks, args.output / 'frozen_traces')
        provenance['source_traces'] = {lead: digest(args.output / f'frozen_traces/{lead}/source.json')
                                       for lead in ('ECG1', 'ECG2')}
    manifest_file = args.output / 'run_manifest.json'
    if manifest_file.exists():
        saved = json.loads(manifest_file.read_text(encoding='utf-8'))
        if saved['provenance'] != json.loads(json.dumps(provenance)):
            raise ValueError('Run provenance differs; choose a new directory')
        run_id = saved['run_id']
    else:
        from uuid import uuid4
        run_id = uuid4().hex
        save_json(manifest_file, {'run_id': run_id, 'provenance': provenance,
                                  'status': 'planned', 'claims': 'AFDB exploratory development only'})
    manifest_id = canonical_hash(provenance)
    jobs = []
    for channel in args.channels:
        for name, variant_config, policy in selected_variants:
            for task in tasks.values():
                record = records[task['record_id']]
                job = {'target': str(args.output / f'{channel.lower()}-{name}/records/{record["record_id"]}'),
                       'run_id': run_id, 'manifest_id': manifest_id, 'cache': str(args.cache),
                       'task': task, 'record': record, 'channel': channel, 'config': variant_config.to_dict(),
                       'mode': args.mode, 'policy': policy, 'raw_root': manifest['raw_root'],
                       'device': execution['device'],
                       'threads': args.threads, 'scores': str(args.scores),
                       'source_trace': str(args.output / f'frozen_traces/{channel}/{record["record_id"]}.jsonl')}
                jobs.append(job)
                if policy == 'event-react':
                    job['llm'] = llm_config
    began = perf_counter()
    completed = 0
    for job, result, resumed in execute_jobs(jobs, args.mode, str(args.models), args.threads,
                                            execution['device'], workers):
        completed += 1
        print(json.dumps({'completed': completed, 'total': len(jobs), 'record': result['record_id'],
                          'channel': job['channel'], 'policy': Path(job['target']).parent.parent.name,
                          'resumed': resumed, 'model_calls': result.get('counts', {}).get('model_calls'),
                          'actual_model_calls': result['execution']['actual_model_calls'],
                          'device': result['execution']['device'],
                          'wall_s': round(result['execution']['record_wall_ms'] / 1000, 2)}, ensure_ascii=False), flush=True)
    for channel in args.channels:
        for name, variant_config, _ in selected_variants:
            folder = args.output / f'{channel.lower()}-{name}'
            results = [json.loads((folder / f'records/{task["record_id"]}/result.json').read_text(encoding='utf-8'))
                       for task in tasks.values()]
            write_jsonl(folder / 'predictions.jsonl', [
                {key: r[key] for key in ('task_id', 'intervals', 'burden_intervals') if key in r}
                for r in results])
            save_json(folder / 'summary.json', {'records': len(results), 'mode': args.mode,
                      'config': variant_config.to_dict(), 'channel': channel,
                      'counts': {key: sum(r.get('counts', {}).get(key, 0) for r in results)
                                 for key in set().union(*(r.get('counts', {}) for r in results))},
                      'execution': {'device': execution['device'], 'workers': workers,
                                    'actual_model_calls': sum(r['execution']['actual_model_calls'] for r in results),
                                    'neural_ms': sum(r['execution']['neural_ms'] for r in results),
                                    'record_wall_ms': sum(r['execution']['record_wall_ms'] for r in results),
                                    'worker_model_load_ms': {str(r['execution']['worker_pid']): r['execution']['worker_model_load_ms']
                                                             for r in results}}})
    save_json(args.output / 'completion.json', {'completed': completed, 'jobs': len(jobs),
              'wall_s': perf_counter() - began, 'run_id': run_id, 'manifest_id': manifest_id})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('live', 'cached', 'fixed'), required=True)
    parser.add_argument('--cache', type=Path, default=default_inputs()['cache'])
    parser.add_argument('--tasks', type=Path, default=default_inputs()['tasks'])
    parser.add_argument('--scores', type=Path, default=default_inputs()['scores'])
    parser.add_argument('--baseline', type=Path, default=default_inputs()['baseline'])
    parser.add_argument('--models', type=Path, default=default_inputs()['models'])
    parser.add_argument('--config', type=Path, default=code_root() / 'configs/ecg/policy.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=normalize_ecg_device,
                        default=os.environ.get('CARDIOAGENT_ECG_DEVICE', 'cpu'),
                        help='cpu or one CUDA device (e.g. cuda:0); CUDA uses one process/model')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--policies',nargs='+',help='Explicit variant names; defaults to the existing mode matrix')
    parser.add_argument('--llm-config', type=Path, help='Explicit ReAct provider and loop configuration')
    parser.add_argument('--channels', choices=('ECG1', 'ECG2'), nargs='+', default=['ECG1', 'ECG2'])
    parser.add_argument('--limit-records', type=int, default=0)
    parser.add_argument('--duration-s', type=float, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1 or args.limit_records < 0 or args.duration_s < 0:
        parser.error('Invalid resource or subset argument')
    run(args)




from cardioagent.common.paths import code_root, default_inputs
from cardioagent.ecg_agent.harness import execution as _execution
execution_settings = _execution.execution_settings

def configured_variants(mode, config, selected=None):
    available=variants(mode,config)
    if selected and 'event-react' in selected and mode != 'fixed':
        available.append(('event-react', config, 'event-react'))
    if selected:
        unknown=set(selected)-{name for name,_,_ in available}
        if unknown:raise ValueError(f'Unknown variant names: {sorted(unknown)}')
        available=[item for item in available if item[0] in selected]
    if config.continue_threshold is not None and any(policy in ('legacy-dual','fixed') for _,_,policy in available):
        raise ValueError('State-dependent thresholds require event-adaptive variants; select --policies event-adaptive')
    return available

def initialize(mode, models, threads, device='cpu'):
    global _MODEL, _MODEL_INFO, _LOAD_MS
    _MODEL, _MODEL_INFO, _LOAD_MS = _execution.build_model(mode, models, threads, device)

def execute_jobs(jobs, mode, models, threads, device, workers):
    yield from _execution.execute_jobs(jobs, mode, models, threads, device, workers,
                                      initialize=initialize, run_job=run_job, pool_factory=ProcessPoolExecutor)

def inference_hashes(root):
    paths=list((root/'src/cardioagent/common').glob('*.py'))
    base=root/'src/cardioagent/ecg_agent'
    for p in list(base.rglob('*.py')) + list(base.glob('prompts/*.txt')):
        if not any(x in ('benchmarks','toolkit','evaluation','__pycache__') for x in p.relative_to(base).parts): paths.append(p)
    paths+=list((root/'src/cardioagent/signal').glob('*.py'))
    paths += [root/'third_party/ecgfounder/net1d.py',root/'third_party/ecgfounder/tasks.txt']
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(set(paths))}


if __name__ == '__main__':
    main()
