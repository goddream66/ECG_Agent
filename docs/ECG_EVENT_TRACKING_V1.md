# ECG evidence-grounded event tracking V1

This version implements M0–M1-C of ECG_AGENT_V2_RESEARCH_PLAN.md. M2 learning,
M3 LLM and M4 external validation remain future work. AFDB has already been used
for development; these experiments are exploratory, not a blind test.

Current canonical layout is documented in REFACTOR_20261116.md; old paths are compatibility facades.

## Core implementation

| Responsibility | Source |
|---|---|
| Event lifecycle and configuration | [event_state.py](D:/CardioAgent/src/cardioagent/ecg_agent/memory/events.py) |
| Immutable evidence and interpretations | [evidence.py](D:/CardioAgent/src/cardioagent/ecg_agent/memory/evidence.py) |
| State-aware tool selection | [review_policy.py](D:/CardioAgent/src/cardioagent/ecg_agent/policies/rule_review.py) |
| Causal current-window and fixed-evidence replay | [event_replay.py](D:/CardioAgent/src/cardioagent/ecg_agent/agent.py) |
| Independent metrics and paired statistics | [event_evaluate.py](D:/CardioAgent/src/cardioagent/ecg_agent/evaluation/metrics.py) |
| Record checkpoint orchestration | [run_ecg_events.py](D:/CardioAgent/scripts/run_ecg_events.py) |

## Runtime

- Keep the released single-channel ECGFounder checkpoint and existing NeuroECG
  ten-second offline cache. There is no new classifier training or model download.
- ToolSession can receive an EvidenceContext with run/task/record identity.
  Namespaced IDs include a session nonce. Old callers remain compatible.
- EvidenceLedger deep-copies raw returns. Interpretations and acquisitions are
  separate append-only logs. A cache hit links to its original evidence.
- EventTracker receives only window bounds, evidence interpretations and IDs;
  it has no model, waveform, label or filesystem access.
- Rules use only explicitly requested model scores and window-local quality.
  A below-threshold score near 0.5 is also reviewed without an active event.
  Missing primary evidence may fall back to a usable secondary channel, explicitly
  identified as a single-channel decision. RR is not part of the new decision rule.
- Raw I/O may preload a recording for efficiency. Each ToolSession owns only the
  current window; no future filtering, peak detection or model scoring is performed.
  This is completed-window causality, not sample-by-sample online preprocessing.

## State and output semantics

The state machine distinguishes candidate, active, pending end and unknown hold.
Two positive windows confirm; two consecutive usable negatives end. Unknown breaks
consecutive counts. A recovered candidate confirms from its latest consecutive
support run, preserving its original first-seen metadata.

A pending negative does not extend the report. Recovery may append a revision
bridging that short gap, while preserving counter-evidence. Unknown recovery within
the ten-second grace retains identity; longer unknown closes unresolved, without a
fabricated normal diagnosis or precise event offset. Candidate timeouts never become
confirmed events. Record-end truncation is separate from confirmed physiological end.

JSONL artifacts are written per record:

| Artifact | Content |
|---|---|
| evidence.jsonl | Original tool results, identity, source/version and timing |
| acquisitions.jsonl | Tool request trigger, prior state and remaining budget |
| assessments.jsonl | Interpretation, reason and cited evidence IDs |
| event_updates.jsonl | Causal state transitions and immutable interval writes |
| window_trace.jsonl | Window scores, interpretation, state and evidence links |
| result.json | Final intervals, event histories, counters and execution timing |
| job_manifest.json | Job fingerprint and artifact hashes for resumption |

Each method also exports compatible predictions.jsonl and summary.json.
Fixed-evidence runs retain original evidence IDs as source links, reconstruct the
old policy with its own shadow state, and require exact source/consumed hashes.
Their acquisitions are inherited from the old trace, not new tool executions.

## Commands

Single-GPU execution is now supported by `--device cuda:0`. On the server, use
`run_server_experiment.sh live`: it defaults to one visible GPU and one process.
See [SINGLE_GPU_EXECUTION.md](SINGLE_GPU_EXECUTION.md) for CPU overrides, timing,
device provenance and streaming guarantees. The original CPU commands below
remain valid on installations without a server-specific device default.

Run from `D:\CardioAgent` with the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts/run_ecg_events.py --mode fixed --output runs/ecg-events-v1-r1/fixed --workers 2 --threads 1
.\.venv\Scripts\python.exe scripts/run_ecg_events.py --mode cached --output runs/ecg-events-v1-r1/cached --workers 2 --threads 1
.\.venv\Scripts\python.exe scripts/run_ecg_events.py --mode live --output runs/ecg-events-v1-r1/live --workers 8 --threads 2
.\.venv\Scripts\python.exe scripts/analyze_ecg_events.py
.\.venv\Scripts\python.exe scripts/plot_ecg_events.py
.\.venv\Scripts\python.exe scripts/audit_ecg_events.py
```

The CLI equivalent is `cardioagent track-afdb --mode live --run-dir <directory>`;
`--config configs/ecg/policy.json` supplies the frozen primary settings. The script
also accepts `--limit-records` and `--duration-s` for explicit engineering smoke tests.
Live uses actual current-window model calls, with one model load per worker, no
cross-policy score sharing and no future score table. Cached mode never runs the
network. Fixed mode reuses only already acquired old Adaptive evidence.
Both execute a simulated recording clock: confirmation/ending timestamps are
recording seconds, while model/tool durations use measured wall time. Sensor
acquisition, transport and physical alarm delivery remain outside this replay.

Rerun the exact same command to resume completed records. Configuration, source,
model, data and result hashes must match. Incomplete/corrupt final checkpoints fail
explicitly. Temporary records are staged beside the destination and never replace
completed results. Changed inputs require a new output directory.

The initial `runs/ecg-events-v1` directory was a development attempt interrupted to
correct the inactive borderline-negative branch. The final experiment directory is
`runs/ecg-events-v1-r1`; do not combine the two versions' closed-loop results.

## Evaluation and cost

Original event F1 preserves prediction-order greedy IoU>=0.5. The evaluator reads
independent .atr timing only on the offline side. Time metrics exclude unlabeled
prefixes, retain unknown/undecided time in unconditional metrics, and report
conditional errors separately. Waveform missing spans and label availability are
different fields. Score success is not decision coverage.
Known AFDB rhythm labels are AFIB/AFL/J/N; other rhythm labels are excluded from
validity rather than treated as normal. The raw AFIB+AFL union is independently
audited against published targets, with a 0.01-second rounding tolerance and no
rewriting of the original reference intervals.
Time errors use the final reported intervals; waiting is separately represented by
causal unknown/undecided coverage and actual confirmation delay. These empirical
errors are not learned or calibrated stopping probabilities.

Fragment counts include every overlapping prediction; missed events remain explicit.
Merge counts are accompanied by crossed non-AF time. Actual first-confirmation delay
and complete-event IoU write delay are separate. Right-censored/unresolved offsets
are excluded from known-end error and counted separately.

Paired bootstrap uses 2,000 record resamples and seed 20261114. Subject identity
mapping is unavailable; the remaining grouping limitation is disclosed. Baselines
include causal median-3, 0.6/0.4 hysteresis and causal ten-second gap handling.

Real timings include neural execution, other tools, record wall time and model load.
Loads are recorded once per worker; repeated per-record load fields must not be
summed. Workers run concurrently, so summed service times are resource measurements,
not isolated clinical response latency or a direct wall-clock speedup guarantee.
Bootstrap intervals describe record variability, not unobserved external validity.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Tests cover evidence identity and immutability, preserved old tools, future isolation,
state transitions, timeout/truncation, inactive borderline-negative review, budget
failure, exact fixed-evidence reconstruction, and independent scoring examples.
See the status document and run analysis for measured outcomes; functional completion
does not imply the new strategy outperforms all simple baselines.
