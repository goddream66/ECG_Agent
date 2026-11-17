"""Physical storage is configured separately from the source checkout."""
import os
from pathlib import Path

def code_root():
    return Path(os.environ['ECG_AGENT_CODE']) if os.environ.get('ECG_AGENT_CODE') else Path(__file__).resolve().parents[3]

def default_inputs():
    storage=os.environ.get('ECG_AGENT_STORAGE')
    if storage:
        root=Path(storage)
        return dict(cache=root/'data/afdb_preprocessed_server',tasks=root/'data/afdb_tasks_server/agent_tasks.jsonl',
                    scores=root/'data/afdb_scores_server',baseline=root/'runs/afdb-full-review/full-cpu',models=root/'models')
    return dict(cache=Path('runs/afdb-preprocessed-full-20261113'),
                tasks=Path('runs/afdb-full-review/20261113-233212-64035fa8/agent_tasks.jsonl'),
                scores=Path('runs/afdb-full-review/full-cpu/model_scores'),baseline=Path('runs/afdb-full-review/full-cpu'),models=Path('models'))
