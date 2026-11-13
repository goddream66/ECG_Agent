# Third-party references

## ECG-Scroll

Source: https://github.com/CuCl-2/ECG-Scroll
Local reviewed source: D:/ECGagent/ECG/ecg/tools.py
License: MIT; copyright (c) 2026 ECG-Scroll authors.
Retained license: third_party/licenses/ECG-Scroll-MIT.txt.

`src/cardioagent/ecg_agent/toolkit/measurements.py` adapts the local RR cleaning,
coarse profile and P/QRS/ST/T/QT measurement ideas and algorithms. It replaces
whole-record cached detection with elapsed-window tools, preserves original RR,
returns missing measurements as null, and explicitly names approximate outputs
as experimental proxies. Annotation-reading diagnostic tools were not copied.
No source data or pretrained weights were copied.
See third_party/ecg_scroll_source.json for the exact local source hash and commit.

## Architecture references (no source code copied)

- https://github.com/huggingface/smolagents/blob/main/src/smolagents/tools.py :
  named tools, descriptions, input schemas, validation and registration.
- https://github.com/langchain-ai/langgraph : separate state, execution and persistence.

CardioAgent uses its own provider-neutral contracts and runtime. It does not
install either framework as part of this refactor or claim API compatibility.
