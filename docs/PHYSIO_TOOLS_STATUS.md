# Physiological tool integration — 2026-11-16

Status: implemented locally and deployed on the server.

- New experimental tools: detect_beats_featuredb, extract_average_beat_featuredb, assess_template_quality.
- Local and server suites: 112 tests passed (101 existing tests plus 11 new cases).
- Real AFDB smoke check: all 23 records, first 30 seconds, ECG1/ECG2, 138 successful calls; no rhythm labels loaded.
- CLI invocation verified on record 04126, ECG1, first 10 seconds.
- Shared R-peak computation has explicit evidence linkage; raw RR, physical units, missing-signal failure and causal access are preserved.
- NeuroKit2 0.2.12 and its missing dependencies installed. Server Torch remains 2.15.0+cu132; environment remains on the data disk.
- Existing policy and old tool behavior are retained. New tools require explicit experimental enablement and are not automatically selected by event-adaptive.
- PQRST clinical intervals and independent measurement-accuracy validation remain future work.

Code: src/cardioagent/ecg_agent/tools/physio.py.
Usage and limitations: PHYSIO_TOOLS.md.
Real-data evidence: /data/1shared/gaojiajun/ECGagent/runs/physio-tools-20261116.
Backups, source hashes, dependency lock and tests: /data/1shared/gaojiajun/ECGagent/migration/physio-tools-20261116.
