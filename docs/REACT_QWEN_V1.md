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

## 可选边界取证

在 `--llm-config` 对应的 JSON 中增加 `"boundary_tools": true`，仍使用
`--policies event-react`。Qwen 可在最近 120 秒已到达的波形内，为任一导联
选择小数秒区间，调用质量、R 峰、原始 RR、局部 RR 概况、P 波和 QRS 代理工具。
`get_beat_timeline_proxy` 返回按 R 峰对齐的逐拍记录及实际采样范围。测量至少
需要 5 秒，可用疑似转折前后的多拍信号复核更短的中断。波形工具仍按各自的
质量和检峰要求报告失败或证据不足。

`--mode live` 允许 ECGFounder 对已到达的任意完整 10 秒区间重新切窗并评分。
`--mode cached` 的 ECGFounder 读取当前网格窗，保留区间内的信号测量可照常执行。
例如可测量 `146.25–153.40 s` 内的逐拍证据，也可以在 live 模式对
`143.25–153.25 s` 进行 ECGFounder 评分。

存在已确认的活动事件时，Qwen 可在 `finish_review` 提交
`proposed_end_s` 和证据 ID。接受的建议写入 `llm_steps.jsonl` 并标记事件 ID
和当前游标。最终区间仍由现有状态机生成，建议供后续按标注评估。边界模式
沿用现有扫描推进与调用预算。

## Qwen 自主结束事件（实验性）

在边界取证配置中再加入 `"event_control": true`。该配置仍使用现有十秒流式
扫描和事件开始确认，但把**已确认事件的结束决策**交给 Qwen。此时模型可以
调用 `close_event(event_id, end_s, evidence_ids, reason)`，也可以用
`finish_review` 继续等待。`finish_review` 只结束本轮复核，不结束事件。
启用该模式后，连续阴性窗和连续未知窗不会替模型关闭已确认事件；到达记录
末尾仍会将未关闭事件标为右删失。候选事件的否定和超时规则保持原样。
可直接选用 `configs/ecg/react_qwen_event_control.json` 作为本地 Qwen 配置；
该文件只是入口配置，需在独立运行目录中执行实验。

`close_event` 要求当前存在对应的已确认事件，`end_s` 位于该事件起点之后、
当前游标之前，并且至少引用一次本轮已成功执行且覆盖结束时刻的局部边界
测量。运行时拒绝未来时刻、伪造证据或错误事件 ID。成功后，输出区间的结束
位置采用 `end_s`，同时保存原区间、估计结束时刻、实际决策时刻、证据 ID
和理由。局部逐拍与形态工具仍是代理测量；通过结构校验不等于医学验证。

新模式与原规则、仅记录 `proposed_end_s` 的模式用不同配置和运行目录配对
比较。先检查调用率、未关闭事件、区间 F1、误合并、碎片数、结束边界误差及
决定延迟，再判断 Qwen 的结束决策是否有增益。尚未在 AFDB 或外部数据上
测得该模式的效果。

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
