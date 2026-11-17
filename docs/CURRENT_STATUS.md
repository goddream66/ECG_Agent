# Current ECG Agent status — 2026-11-16

The layout refactor is complete in the local source and deployed server checkout.
Current code is split into controller, harness, policies, memory, tools, models,
data, evaluation and experiments; earlier agents remain under legacy with compatible imports.

Server source: `/home/gaojiajun/ECGagent`.
Physical data/models/environments/caches/outputs: `/data/1shared/gaojiajun/ECGagent`.
The source checkout links to those asset directories. There is no environment or weight duplication.

Validation:

- Local and server unit suites: 101 tests passed (original 95 assertions retained).
- GPU and CPU live: 4 records × 300s × 2 primary channels × 2 strategies = 16 jobs per mode, 740 real model calls per mode.
- Cached: 48 jobs, zero actual neural calls. Fixed: first record's complete original evidence traces, 8 jobs, zero actual neural calls.
- CLI live: first record × 60s, ECG1, 2 jobs and 18 real model calls.
- Before/after refactor comparisons: identical scores, tool request sequences, window decisions, semantic event histories and final intervals on checked live/cached/CLI jobs.
- CPU versus GPU maximum score difference: 4.76837158203125e-07; no threshold flips at 0.4/0.5/0.6 and identical decisions/intervals.
- One GPU process/model. Resume reused all 16 GPU jobs; all 112 per-record artifacts remained unchanged.
- Architecture/label inputs and tool schemas match their previous bytes. Existing study outputs remain untouched.

These are bounded engineering regression checks including positive/negative windows, not a new full AFDB study or evidence of improved clinical performance.
Runtime remains the existing M0–M1-C rule-driven AF/AFL tracker with released ECGFounder weights and offline ten-second caches. M2 learning, M3 LLM and M4 external validation are unimplemented.

Legacy fixed replay consumes complete saved traces and does not truncate evidence with --duration-s. Validation uses fixed-full without that argument; the earlier fixed trial is excluded as a bounded-duration check.

Refactor backup and manifests: `/data/1shared/gaojiajun/ECGagent/migration/ecg-refactor-20261116`.
Verification report: `verification.json` there. New check outputs: `/data/1shared/gaojiajun/ECGagent/runs/ecg-refactor-20261116`.
Start with REFACTOR_20261116.md, ARCHITECTURE.md and TOOL_DEVELOPMENT.md.
