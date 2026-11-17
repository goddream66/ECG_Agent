# Qwen ReAct review planner V1

Status: local implementation and deterministic integration validation. Actual Qwen
weights have not been loaded or exercised by these tests. No server experiment,
Git push, classification-head training or learned stopping policy in this change.

## Responsibility and code ownership

The design follows the separation of model adapters, runner, tools, session state,
configuration, examples and tests used by Qwen-Agent and the OpenAI Agents SDK:

- https://github.com/QwenLM/Qwen-Agent
- https://github.com/openai/openai-agents-python
- ReAct: https://arxiv.org/abs/2210.03629

These are architectural references. No framework implementation is vendored or
new orchestration dependency required. Signal tools still have one implementation.

```text
src/cardioagent/ecg_agent/
  llm/contracts.py             # provider-neutral message/usage contract
  llm/providers.py             # lazy local Qwen or compatible HTTP adapter
  prompts/react_review_v1.txt  # versioned acquisition instructions
  harness/react.py            # bounded model/tool/observation loop
  harness/runtime.py          # existing retained signal, tool budget and cache
  memory/events.py           # existing event identity and lifecycle
  agent.py                   # record/window controller, event-react opt-in
  experiments/runner.py       # CLI, provenance, checkpoint and result orchestration
configs/ecg/react_qwen_*.json  # provider, resources and memory limits
tests/runtime/test_react.py   # mock-provider and real-runtime integration tests
```

## One window

1. Admit the completed offline-preprocessed ten-second window.
2. Obtain the scheduled primary ECGFounder score through ToolSession.
3. Give Qwen this acquired result, active-event summary, recent-window evidence,
   available leads, current bounds and remaining budget.
4. Qwen emits one function call. The runner validates it, executes it through the
   same ToolSession, records real evidence and feeds the result back to Qwen.
5. Qwen may invoke `finish_review`. Budget exhaustion and round limits also stop
   acquisition. Model/protocol errors produce unavailable evidence, never normality.
6. The existing `decide_event` interpreter reads only already acquired results.
   It cannot execute extra hidden tools. The existing EventTracker updates events.

V1 deliberately separates learned tool planning from rule-based evidence
interpretation. AF/AFL thresholds, confirmation counts and unknown handling remain
those in the policy file. The model does not directly set intervals or revise raw
evidence. Stopping review is different from ending an event. If required evidence
is missing, stopping can result in unknown. Valid evidence may still support a
decision when the tool/round budget is reached.

Only `score_ecgfounder` and `assess_quality` are exposed initially. RR and morphology
tools remain available in the toolkit but do not acquire AF adjudication semantics.
The current session retains only the current window: model requests to any other
window are rejected, including historical signal. Memory contains past evidence,
not permission to read future or discarded raw signal.

## Memory and prompt

Each record/task/run receives a fresh reviewer and conversation. Local model
weights may be reused in a worker; patient memory may not. Recent history defaults
to six windows (zero disables it). The active-event snapshot contains boundary
timestamps, counters and the last six spans per list. Supporting/opposing/unknown
evidence remains separately traceable in the ledger. This is bounded working
memory, not a full EventMemAgent archive, vector search, or cross-patient memory.

Edit the independent versioned prompt when changing instructions. Prompt SHA-256,
provider settings, runtime source hashes and local model-file hashes enter the run
manifest. Tool schemas come from the existing registry; finish_review is a local
control action, not a signal measurement. Its evidence IDs must exist in the
current session. Logs contain action justifications rather than long private
reasoning transcripts. Endpoint weights are externally managed: endpoint identity
alone cannot verify weight identity or guarantee deterministic replay.

## Execution

The Transformers provider needs the optional `llm` dependencies in a compatible
environment; local tests need neither Transformers nor the 8B weights. Nothing is
installed automatically. It uses local files only, no remote model code, one
explicit CUDA device and deterministic greedy generation. Qwen's native
`<tool_call>` format is parsed into the provider-neutral contract. Malformed,
parallel or truncated calls fail to unknown.

Example for a later authorized server smoke experiment (paths for ECG inputs use
the existing server defaults). This is a command example, not an experiment run:

```bash
python -m cardioagent.ecg_agent.experiments.runner \
  --mode live --device cuda:0 --workers 1 --threads 2 \
  --policies event-react \
  --config configs/ecg/policy_hysteresis060040.json \
  --llm-config configs/ecg/react_qwen_local.json \
  --channels ECG1 --limit-records 1 --duration-s 120 \
  --output /data/1shared/gaojiajun/ECGagent/runs/react-qwen-v1-smoke
```

Use `react_qwen_endpoint.json` for an existing compatible local service; edit its
base_url/model to match that service. API keys are read from ECG_LLM_API_KEY, never
stored in configuration. Endpoint timeouts default to 60 seconds. Local generation
is token-bounded and has no hard wall-clock timeout.

`--mode cached` also supports event-react: ECGFounder returns cached values on
request, while the LLM is still real. It is not a real-model compute cost baseline.
`--mode fixed` does not support this interactive policy. Old variant defaults remain
unchanged; explicitly select `--policies event-react` and `--llm-config`.

## Outputs and validation

Per-record `llm_steps.jsonl` records selected actions, arguments, evidence IDs,
failures, stop reasons and the rule interpreter decision. Existing evidence,
acquisition, assessment and event-update logs remain separate. `result.json` adds
LLM request count, prompt/completion tokens, inference wall time and adapter-load
time. Tool and ECGFounder time remain separate. Adapter-load time on later records
can be only a cache lookup; do not count it as a fresh model load. Physical LLM
latency is reported independently of signal-cursor event timestamps.

Checkpoint artifacts are hashed. Configuration, model or prompt changes require a
new run directory. Do not select prompt/thresholds on full AFDB results and describe
that as held-out performance. Compare rules vs ReAct with the same policy settings,
tools and budget before changing event adjudication itself.

Local validation command:

```powershell
$env:TEMP='D:\CardioAgent\tmp'; $env:TMP=$env:TEMP
& .\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Validation on 2026-11-17: 164 tests passed, including 16 new ReAct tests; CLI help
and Git whitespace checks passed. The record-entry integration uses synthetic
signals and a scripted provider, with the actual ToolSession, logging and
checkpoint code. It verifies two arrived windows, real tool-result feedback,
unrequested-channel exclusion, per-record memory and verified resume. These are
engineering checks, not evidence of Qwen diagnostic accuracy or GPU compatibility.
