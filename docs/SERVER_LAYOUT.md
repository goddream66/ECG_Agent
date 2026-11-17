# ECG Agent server layout

Code: `/home/gaojiajun/ECGagent`

All large or generated assets: `/data/1shared/gaojiajun/ECGagent`

| Content | Actual server directory |
| --- | --- |
| Raw AFDB and original project datasets | `.../data` |
| Active, portable AFDB inputs | `.../data/afdb_preprocessed_server`, `.../data/afdb_tasks_server`, `.../data/afdb_scores_server` |
| ECGFounder and existing non-LLM modality assets | `.../models` |
| Python interpreter and Linux virtual environment | `.../envs` |
| Baselines, latest reports, and future experiments | `.../runs` |
| Package, model and plotting caches | `.../cache` |
| Temporary files | `.../tmp` |
| Transfer checksums, package versions, path mapping and validation logs | `.../migration` |

`data`, `runs`, and `.venv` inside the code directory are symbolic links into the data disk. The `models` directory keeps the two Git-tracked asset-description files beside the code; its weights and upstream assets are links into the data disk. Working from the code directory therefore keeps the old relative paths useful. Source code and documentation remain in the home directory.

## Activate

```bash
source /home/gaojiajun/ECGagent/activate.sh
```

This selects the data-disk environment and puts package downloads, Python bytecode, model caches, matplotlib configuration and temporary files on the data disk.

## Experiments

Develop and test locally first. Publish code to GitHub when preparing a server GPU experiment, then pull the committed version here. Ordinary local edits do not trigger a push or server sync. See [Development workflow](DEVELOPMENT_WORKFLOW.md) for the update procedure and initial migration precautions.

```bash
# Small real-model check: one record, 60 seconds, both primary channels
./run_server_experiment.sh live --limit-records 1 --duration-s 60 --workers 1

# Full experiment using the already-cached scores
./run_server_experiment.sh cached

# Full experiment using real on-demand model execution
./run_server_experiment.sh live
```

The launcher always places output under the data-disk `runs` directory. To choose a reproducible output path or resume a run:

```bash
ECG_AGENT_OUTPUT=/data/1shared/gaojiajun/ECGagent/runs/my-experiment \
  ./run_server_experiment.sh live
```

Resume remains subject to the existing configuration/input/model/source fingerprint checks. Historical Windows results are preserved as historical results; do not resume them as Linux runs.

## Validation

```bash
source /home/gaojiajun/ECGagent/activate.sh
python -m unittest discover -s tests -q
cardioagent --help
cat "$ECG_AGENT_STORAGE/migration/integrity.json"
cat "$ECG_AGENT_STORAGE/migration/server-path-map.json"
```

Historical manifests and score files are left unchanged. Portable input copies adjust Windows absolute paths and their dependent manifest digest; waveform arrays and cached scores are linked to the verified originals. Reference answers remain separate from inference tasks.

The owner selected reports and figures instead of the large historical event-trajectory archives. The latest report archive is `/data/1shared/gaojiajun/ECGagent/runs/ecg-events-v1-r1-reports`, including `analysis/COMPARISON.md`, metrics, figures, and method-level predictions/summaries. It is a report archive, not a resumable full checkpoint. Full historical trajectories remain in the original local `D:\CardioAgent\runs` directory. Existing small runs and the baseline inputs already copied to the server remain on the data disk.

The Windows virtual environment is not copied. A Linux environment is rebuilt, with the original package inventory saved for comparison. The ECG event scorer supports an explicit single CUDA device; legacy callers retain their CPU default. The server event launcher selects CUDA explicitly. Windows llama.cpp binaries and the currently unused local GGUF model are outside this ECG-only runtime migration.

Local source data and results are preserved. No full research experiment is automatically started by the migration.

## GPU environment

The environment uses PyTorch 2.15.0+cu132 with CUDA 13.2. The event executor now accepts one CUDA device. `run_server_experiment.sh live` defaults to physical GPU 0, exposes one GPU, and runs one process with one model. `ECG_AGENT_GPU=3` selects physical GPU 3 as logical cuda:0. `ECG_AGENT_DEVICE=cpu` selects the CPU baseline. Cached and fixed modes do not use the GPU. Details of streaming guarantees and timing are in `docs/SINGLE_GPU_EXECUTION.md`.

CUDA, Triton, TorchInductor and extension caches are directed to the data disk by `activate.sh`. Details: `/data/1shared/gaojiajun/ECGagent/migration/gpu-upgrade/STATUS.md`.
