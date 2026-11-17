# ECG event architecture

`experiments.runner` selects tasks, assets and execution. It feeds admitted windows
to `agent.replay_event_record`. The controller calls a rule policy, ToolSession,
EvidenceLedger and EventTracker. Only handlers requested by that policy execute.

Dependencies:

```
experiments -> agent -> policies / memory / harness / tools -> models / signal / common
evaluation -> independent references and pure metrics
```

Current online modules do not import evaluation, benchmarks, experiments or legacy.
Legacy class names exported by package __getattr__ are resolved lazily, only on an
explicit old API request. `benchmarks` and `toolkit` are compatibility facades.

Policy evidence interpretation is in policies/rule_review.py and labels.py.
Lifecycle thresholds are in policies/event_rules.py. Memory preserves event state,
support/counter-evidence/unknown intervals and actual commit timestamps.
No future samples, annotations or precomputed unrequested scores reach the policy.
Preprocessing stays offline on completed ten-second windows.

Tools are declared/versioned in tools/catalog.py, validated with schemas.py, and
executed through harness/runtime.py. Tools' algorithms and the model predictor are
separate from the runtime's budget, availability, caching and error protocol.

The single GPU dispatcher lives in harness/execution.py. It owns one model in one
process, with float32 and TF32 disabled. CPU keeps the process pool. Cached/fixed
replay never runs a neural model. A blocking scalar transfer is included in neural
wall-clock cost; model loading synchronizes CUDA.

Physical storage is configured by ECG_AGENT_CODE and ECG_AGENT_STORAGE. Weights,
waveforms, environments, caches and outputs stay on the data disk. Only the copied
ECGFounder architecture and class names are vendored in third_party/ecgfounder;
their bytes are verified against the previous assets.

Legacy root modules are small import aliases. Current CLI routing does not eagerly
load the legacy LLM workflow. scripts/run_ecg_events.py delegates to the package,
and root shell entry points delegate to ops/server. No new framework dependency,
classification head, inference threshold, event score or benchmark is introduced.
