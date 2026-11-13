# Repository instructions

The active development focus is the ECG-only agent under src/cardioagent/ecg_agent.

- Preserve old CardioAgent experiment entry points and saved results unless the task requires changing them.
- Tools belong in ecg_agent/toolkit. Declare names, argument models, versions and limitations in catalog.py; execute them through ToolSession.
- Only the elapsed retained signal may reach a tool. Never use full-record peak caches for streaming inference.
- Annotation loaders stay in evaluation modules and must never appear in the inference registry.
- Missing evidence is insufficient/null; it is not a zero or a normal diagnosis.
- Clinical intervals require validated waveform boundaries. Approximate algorithms must have explicit proxy names and default to disabled.
- Keep raw RR and cleaning masks. Do not silently drop abnormal intervals.
- Preserve attribution in THIRD_PARTY_NOTICES.md and third_party/licenses when adapting upstream code.
- Keep data, weights, generated runs, credentials and local environments out of Git.
- For runtime changes, test causality, schemas, failures and cache behavior. Run `python -m unittest discover -s tests -q`.
- Add a new tool by following docs/TOOL_DEVELOPMENT.md. Update the generated catalog with scripts/export_ecg_tools.py.
