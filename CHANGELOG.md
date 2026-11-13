# Change log

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
