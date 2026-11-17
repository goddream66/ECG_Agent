# Repository instructions

The active development focus is the ECG-only agent under src/cardioagent/ecg_agent.

- Preserve old CardioAgent experiment entry points and saved results unless the task requires changing them.
- Tools belong in ecg_agent/tools; contracts are in schemas.py and ToolSession is in ecg_agent/harness. Declare versions and limitations in tools/catalog.py. toolkit and benchmarks are compatibility facades only.
- Only the elapsed retained signal may reach a tool. Never use full-record peak caches for streaming inference.
- Annotation loaders stay in evaluation modules and must never appear in the inference registry.
- Missing evidence is insufficient/null; it is not a zero or a normal diagnosis.
- Clinical intervals require validated waveform boundaries. Approximate algorithms must have explicit proxy names and default to disabled.
- Keep raw RR and cleaning masks. Do not silently drop abnormal intervals.
- Preserve attribution in THIRD_PARTY_NOTICES.md and third_party/licenses when adapting upstream code.
- Keep data, weights, generated runs, credentials and local environments out of Git.
- Develop and test locally in D:/CardioAgent. Ordinary edits do not trigger a commit, push or server sync. Unless explicitly requested otherwise, commit and push to https://github.com/goddream66/ECG_Agent only when preparing an authorized server GPU experiment; then update the server with a fast-forward-only pull. Preserve uncommitted server changes before updating. Follow docs/DEVELOPMENT_WORKFLOW.md.
- For runtime changes, test causality, schemas, failures and cache behavior. Run `python -m unittest discover -s tests -q`.
- Add a new tool by following docs/TOOL_DEVELOPMENT.md. Update the generated catalog with scripts/export_ecg_tools.py.

- Current online code must not import ecg_agent.evaluation, ecg_agent.experiments, ecg_agent.benchmarks or cardioagent.legacy. Compatibility __getattr__ exports are lazy.
