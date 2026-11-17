# Single-GPU ECG event execution

The ECG event experiment runner accepts `--device cpu` or one CUDA device such as
`--device cuda:0`. Existing scorer callers retain their CPU default. The server
activation script sets `CARDIOAGENT_ECG_DEVICE=cuda:0`, so the event script and
`cardioagent track-afdb` use a GPU by default on that server.

CUDA live runs execute serially in one process with one model. Requests for more
workers are explicitly reported and reduced to one. No CUDA process pool, model
replication, DataParallel, cross-window batches, future-window inference or
unrequested-channel precomputation is introduced. CPU live runs retain the old
process-pool implementation. Cached and fixed-evidence modes do not initialize
CUDA or execute the model, regardless of the requested device.

## Server commands

```bash
source /home/gaojiajun/ECGagent/activate.sh

# Default physical GPU 0, one process, both ECG primary-channel experiments
./run_server_experiment.sh live

# Select a different physical GPU; it is exposed as logical cuda:0
ECG_AGENT_GPU=3 ./run_server_experiment.sh live

# Explicit CPU baseline
ECG_AGENT_DEVICE=cpu ./run_server_experiment.sh live --workers 4

# Small real inference check, not a full study
./run_server_experiment.sh live --limit-records 4 --duration-s 300
```

The wrapper exposes one GPU using CUDA_VISIBLE_DEVICES. If a scheduler already
supplies a single visible GPU, it is respected; use ECG_AGENT_GPU to override it.
New experiment outputs remain under `/data/1shared/gaojiajun/ECGagent/runs`.

The generic CLI also accepts `--device`, `--limit-records` and `--duration-s`:

```bash
cardioagent track-afdb --mode live --device cuda:0 \
  --cache "$ECG_AGENT_STORAGE/data/afdb_preprocessed_server" \
  --tasks "$ECG_AGENT_STORAGE/data/afdb_tasks_server/agent_tasks.jsonl" \
  --run-dir "$ECG_AGENT_STORAGE/runs/my-single-gpu-run"
```

## Timing, failures and provenance

Inputs and model parameters are explicitly moved to the requested device.
Execution stays float32 with TF32 and cuDNN benchmarking disabled for the GPU.
Returned Python scores require a blocking transfer, so recorded neural wall time
includes input transfer, forward execution and scalar return. Model-load timing
also waits for CUDA. These timings are not pure CUDA-event kernel timings.

The run manifest records the requested and effective device/workers, CUDA runtime,
GPU name/UUID, visibility and precision. Record execution logs also retain actual
model-device metadata. Device changes participate in checkpoint identity, and the
runner source itself is included in the source fingerprint. CPU checkpoints cannot
silently resume as GPU results. Missing CUDA or an invalid device fails explicitly;
there is no automatic CPU fallback. Complete resumed GPU jobs skip model loading.

Rules, thresholds, event matching and evidence acquisition remain unchanged.
Compare CPU/GPU scores and report any threshold flips; hardware changes do not
justify claiming exact equality across all records. Old results remain intact.
