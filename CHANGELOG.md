# Change log

## Evidence-grounded ECG event tracking M0–M1-C — 2026-11-14

- Added namespaced immutable evidence, separate acquisition/assessment logs, pending-end and finite unknown-hold event tracking with causal confirmation timestamps.
- Preserved all four old policies; added exact fixed-evidence reconstruction and state-aware closed-loop review, including inactive borderline-negative and active negative review. RR remains outside the new decision rule.
- Added independent raw-annotation audits, time/fragment/merge/coverage metrics, paired record bootstrap, strong temporal controls, record checkpoints, CLI/configuration, actual CPU inference and cache/live parity checks.
- Completed 184 fixed, 276 cached and 92 live record jobs over all 23 AFDB records and both primary channels; 536,251 actual neural calls, zero threshold flips and no legacy dual regression. All 552 checkpoints passed the evidence/time/hash audit; 86 unit tests passed.
- Default new event F1 is ECG1 0.615 / ECG2 0.550 versus old Adaptive 0.328 / 0.375. Simple hysteresis remains competitive or better; time false alarms and short-event tradeoffs are reported separately. No new head training, LLM or external validation is claimed.

## ECG tool registry v1 — 2026-11-13

- Established local source control and a pre-refactor snapshot.
- Added typed ECG tool contracts, a registry, per-block execution context, argument/time validation, request budgets, caching and evidence traces.
- Routed the existing ECG-only rule agent through the same runtime.
- Adapted useful ECG-Scroll measurements with explicit experimental proxy names, preserved attribution, and removed full-record and annotation dependencies.
- Added tool catalog/export commands, documentation, repository instructions and a GitHub Actions test configuration.

## ECG-Scroll interval evaluation pilot — 2026-11-13

- Imported local AF tasks with separate inference manifests and evaluation references.
- Added upstream-compatible temporal F1 and burden scoring with strict task coverage checks.
- Added a six-record fixed-prefix engineering pilot; no AF model or online latency claims.
