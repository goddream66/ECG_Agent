#!/usr/bin/env bash
set -euo pipefail
ECG_CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ECG_CODE_ROOT/activate.sh"
MODE="${1:-live}"
if (( $# )); then shift; fi
case "$MODE" in live|cached|fixed) ;; *) echo 'Mode must be live, cached, or fixed' >&2; exit 2 ;; esac
DEVICE="${ECG_AGENT_DEVICE:-cuda:0}"
WORKERS=4
if [[ "$MODE" == live && "$DEVICE" != cpu ]]; then
  export CUDA_VISIBLE_DEVICES="${ECG_AGENT_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
  case "$CUDA_VISIBLE_DEVICES" in *,*|"") echo 'Select one GPU with ECG_AGENT_GPU, for example ECG_AGENT_GPU=0' >&2; exit 2 ;; esac
  WORKERS=1
fi
for argument in "$@"; do
  case "$argument" in --output|--output=*) echo 'Set ECG_AGENT_OUTPUT to select a directory under the data runs directory' >&2; exit 2 ;; esac
done
OUTPUT="${ECG_AGENT_OUTPUT:-$ECG_AGENT_STORAGE/runs/server-$(date +%Y%m%d-%H%M%S)-$MODE}"
OUTPUT="$(realpath -m "$OUTPUT")"
case "$OUTPUT" in "$ECG_AGENT_STORAGE/runs/"*) ;; *) echo 'Experiment output must remain under the data runs directory' >&2; exit 2 ;; esac
python -m cardioagent.ecg_agent.experiments.runner \
  --mode "$MODE" \
  --cache "$ECG_AGENT_STORAGE/data/afdb_preprocessed_server" \
  --tasks "$ECG_AGENT_STORAGE/data/afdb_tasks_server/agent_tasks.jsonl" \
  --scores "$ECG_AGENT_STORAGE/data/afdb_scores_server" \
  --baseline "$ECG_AGENT_STORAGE/runs/afdb-full-review/full-cpu" \
  --models "$ECG_AGENT_STORAGE/models" \
  --config "$ECG_AGENT_CODE/configs/ecg/policy.json" \
  --output "$OUTPUT" \
  --device "$DEVICE" --workers "$WORKERS" --threads 2 "$@"
