# ECG Agent

Current focus: a causal, evidence-grounded AF/AFL event agent using ECGFounder,
window-local measurements, an evidence ledger and event lifecycle memory.
Rules and foundation-model weights are unchanged by the layout refactor.

## Server

```bash
source /home/gaojiajun/ECGagent/activate.sh
./run_server_experiment.sh live
```

The default event run uses one visible GPU (physical GPU 0), one process and one
model. `ECG_AGENT_GPU=3` selects a different single GPU. `ECG_AGENT_DEVICE=cpu`
selects the CPU baseline. For a bounded engineering check, append
`--limit-records 4 --duration-s 300`.

Code lives in `/home/gaojiajun/ECGagent`. Data, model weights, environments,
package/GPU caches and all experiment outputs physically remain under
`/data/1shared/gaojiajun/ECGagent`. The checkout's data/runs/.venv entries are links.

## Code map

- `src/cardioagent/ecg_agent/agent.py`: current event controller.
- `harness/`: tool execution constraints, logging and CPU/GPU execution.
- `policies/`: current review rules, lifecycle rules and frozen baselines.
- `memory/`: raw evidence, event identity and interval history.
- `tools/`: tool allowlist and signal/model handlers.
- `models/`: ECGFounder predictor; weights are external.
- `data/`: signal-only AFDB arrivals and offline preprocessing.
- `evaluation/`: independent labels and frozen event/time metrics.
- `experiments/`: importable runner and fixed-evidence replay.
- `src/cardioagent/legacy/`: preserved multimodal and early measurement agents.
- `scripts/`: compatible commands and analysis/plot helpers.
- `ops/server/`: server operations, not agent decisions.

See [refactor and storage](docs/REFACTOR_20261116.md), [architecture](docs/ARCHITECTURE.md), [current status](docs/CURRENT_STATUS.md),
[tool development](docs/TOOL_DEVELOPMENT.md), and
[single GPU execution](docs/SINGLE_GPU_EXECUTION.md).
Old commands/imports remain supported through compatibility facades. Historical
plans and multimodal status documents are under `docs/history/multimodal`.

## Validation

```bash
python -m unittest discover -s tests -q
python -m cardioagent.cli track-afdb --help
```

Do not resume pre-refactor checkpoints as new runs: source fingerprints intentionally
change. Historical predictions/manifests remain immutable. No LLM or learned
stopping policy is implemented by this reorganization.
