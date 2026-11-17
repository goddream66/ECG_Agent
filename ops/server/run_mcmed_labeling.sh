#!/usr/bin/env bash
set -euo pipefail
ECG_LABEL_CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ECG_LABEL_CODE/activate.sh"
export CUDA_VISIBLE_DEVICES="${ECG_AGENT_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
case "$CUDA_VISIBLE_DEVICES" in *,*|"") echo 'Select exactly one GPU using ECG_AGENT_GPU' >&2; exit 2 ;; esac
python -m cardioagent.ecg_agent.data.label_mcmed_tasks "$@"
