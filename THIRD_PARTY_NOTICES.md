# Third-party references

## NeuroECG preprocessing snapshot

Source: https://github.com/goddream66/NeuroECG
Pinned commit: 95409f8df36737014a4be922f6464fe31f975fcb.
Unmodified source: third_party/NeuroECG/process_easy_500hz.py.
`src/cardioagent/signal/ecg_preprocessing.py` adapts the filter and normalization
recipe to individual completed AFDB windows. `scripts/preprocess_afdb.py` uses
that module to export offline caches with original sample coordinates.
See third_party/NeuroECG/source.json for its SHA256 and
third_party/NeuroECG/README.md for processing details and adaptation limitations.
No LICENSE/COPYING file was present in the inspected upstream tree at this
commit. This project does not assign a license to the copied source.

## ECG-Scroll

Source: https://github.com/CuCl-2/ECG-Scroll
Local reviewed source: D:/ECGagent/ECG/ecg/tools.py
License: MIT; copyright (c) 2026 ECG-Scroll authors.
Retained license: third_party/licenses/ECG-Scroll-MIT.txt.

`src/cardioagent/ecg_agent/tools/measurements.py` adapts the local RR cleaning,
coarse profile and P/QRS/ST/T/QT measurement ideas and algorithms. It replaces
whole-record cached detection with elapsed-window tools, preserves original RR,
returns missing measurements as null, and explicitly names approximate outputs
as experimental proxies. Annotation-reading diagnostic tools were not copied.
Its beat timeline combines these adapted local RR, P and QRS measurements at
each detected R-peak timestamp for event-boundary review.
No source data or pretrained weights were copied.
See third_party/ecg_scroll_source.json for the exact local source hash and commit.

## Architecture references (no source code copied)

- https://github.com/huggingface/smolagents/blob/main/src/smolagents/tools.py :
  named tools, descriptions, input schemas, validation and registration.
- https://github.com/langchain-ai/langgraph : separate state, execution and persistence.

CardioAgent uses its own provider-neutral contracts and runtime. It does not
install either framework as part of this refactor or claim API compatibility.

`src/cardioagent/ecg_agent/benchmarks/intervals.py` additionally adapts the temporal-F1 and relative-error verifiers from ECG-Scroll `ecg/verify.py`, under the retained MIT license.


## ECGFounder architecture and label vocabulary

The existing, previously used `net1d.py` and `tasks.txt` are now copied verbatim
into `third_party/ecgfounder` so runtime source code stays with the checkout.
Source: https://github.com/PKUDigitalHealth/ECGFounder
License: MIT; see `third_party/ecgfounder/LICENSE`.
Their contents are verified against the original model assets. Checkpoint files
remain external and subject to their upstream terms; no weights are redistributed.


## FeatureDB and ecg_signal_quality adapters

Source: https://github.com/PKUDigitalHealth/FeatureDB (fork of hsd1503/ECG_feature_extraction),
and https://github.com/PKUDigitalHealth/ecg_signal_quality.
Pinned, unmodified source excerpts and SHA-256 records are under third_party/physio_tools.
tools/physio.py adapts their local detector chain, average-beat geometry and template
correlation heuristic. NeuroKit2 supplies the peak detector. No upstream HTTP calls,
import-time demo execution, automatic PQRST boundaries or QT output are used.
No explicit LICENSE was found in these two inspected snapshots; do not infer MIT
permission for redistribution from neighboring repositories. Preserve provenance
and resolve upstream reuse terms before redistributing these sources.


## ECGFounder preprocessing adapter

`data/ecgfounder_preprocessing.py` adapts upstream `util.filter_bandpass` and
`dataset.py` z-score normalization. Source and hashes: third_party/ecgfounder/preprocessing_source.json.
License: existing third_party/ecgfounder/LICENSE (MIT). Raw-window missingness is
retained; upstream NaN-to-zero behavior is not adopted. Non-native rates explicitly
use polyphase resampling before the 500 Hz filter chain.


## 2026-11-17：取消分组数质量门槛

assess_template_quality 升级为1.1.0。template_consistent仅由有限的平均模板相关性>0.8决定，coefficient_group_count保留为order-sensitive审计字段，不再参与质量判定。其他预筛查和缺失规则保留。当前三任务导出为v0.5.1，会对核验后的旧证据重新解释质量规则，以派生证据记录source_evidence_id、source_tool_version及派生说明；原工具返回不变，不重跑检测器或模型。旧v0.5.0结果保留。
