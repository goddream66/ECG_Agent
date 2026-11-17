"""Single-GPU and CPU execution; no policy, references or experiment imports."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from cardioagent.ecg_agent.models.ecgfounder import normalize_ecg_device, ecgfounder_execution

def build_model(mode, models, threads, device='cpu'):
    _MODEL, _LOAD_MS, _MODEL_INFO = None, 0., None
    if mode == 'live':
        import torch
        from cardioagent.ecg_agent.models.ecgfounder import create_ecgfounder_scorer
        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        before = perf_counter()
        _MODEL, _MODEL_INFO = create_ecgfounder_scorer(Path(models), device=device)
        _LOAD_MS = (perf_counter() - before) * 1000

    return _MODEL, _MODEL_INFO, _LOAD_MS

def execution_settings(mode, device, workers):
    requested = normalize_ecg_device(device)
    info = ecgfounder_execution(requested) if mode == 'live' else {'device': 'none'}
    info.update(requested_device=requested, requested_workers=workers)
    effective = 1 if info['device'].startswith('cuda:') else workers
    return info, effective


def execute_jobs(jobs, mode, models, threads, device, workers, *, initialize, run_job, pool_factory=ProcessPoolExecutor):
    """CUDA stays in one process with one model; the CPU pool retains its old behavior."""
    if mode == 'live' and device.startswith('cuda:'):
        initialized = False
        for job in jobs:
            if not initialized and not Path(job['target']).exists():
                initialize(mode, models, threads, device)
                initialized = True
            result, resumed = run_job(job)
            yield job, result, resumed
    else:
        with pool_factory(max_workers=workers, initializer=initialize,
                                 initargs=(mode, models, threads, device)) as pool:
            futures = {pool.submit(run_job, job): job for job in jobs}
            for future in as_completed(futures):
                result, resumed = future.result()
                yield futures[future], result, resumed
