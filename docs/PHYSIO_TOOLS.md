# FeatureDB and template-quality tools

Three experimental tools are registered: detect_beats_featuredb,
extract_average_beat_featuredb, assess_template_quality.
Install the optional dependency with `python -m pip install -e ".[ecg-tools]"`.
The existing server environment stays on the data disk; no model download is needed.

Use ToolSession(..., allow_experimental=True), or CLI call-ecg-tool with
--experimental. All tools accept lead/start_s/end_s. Physical mV/uV/V is required;
normalized ECGFounder tensors are rejected. Only the elapsed retained window is used.
Calls shorter than five seconds follow the existing insufficient-evidence protocol.

Example (server, activated environment):

```bash
cardioagent call-ecg-tool detect_beats_featuredb \
  --record "$ECG_AGENT_STORAGE/data/afdb_local/04126" \
  --lead ECG1 --end 10 --experimental
```

The detector reports absolute peak times, window-local indices, raw RR and median-RR
rate. RR intervals are not silently cleaned. Missing or flat windows return
insufficient, and unknown physical units or execution failures return error.
The average uses complete interior beats and reports excluded beats; no zero padding.
Average waveforms are morphology summaries, not per-beat event boundaries.

Template quality reports per-beat correlations, mean correlation, the upstream
coefficient-group heuristic and separate prescreen flags. Correlation is not a
probability. Abnormal RR or fewer than six peaks does not automatically veto quality.
Three complete, nondegenerate beat segments are required. No correlation is invented
for insufficient data. A flat signal can be flagged poor while template correlation
remains unavailable.

Shared detector results are cached within one ToolSession and window. A prior
measurement is referenced by evidence ID; otherwise the full detector runs inside
the requested tool, its wall cost is included and detector_executed is true.
Tools using the same detector are dependent evidence, not independent diagnostic votes.
Runtime cache hits retain the existing outer cached flag and evidence linkage.

The adapter uses only the local FeatureDB algorithm and does not import the archived
upstream scripts. It reports whether the 50 Hz notch was applied (requires fs>100 Hz).
Filters operate inside the admitted window after arrival, not on future windows.
This is window-causal execution, not a sample-by-sample streaming filter.

The original policy, old tool names, default allowlist and frozen old schema remain
unchanged. These tools are callable through the registry and CLI but are not yet
automatically requested by event-adaptive. No learned confidence, AF threshold change,
PQRST interval tool, or clinical validation is claimed. Sources, patches and license
status are documented in third_party/physio_tools/source.json.


## Integration verification (2026-11-16)

Local and server regression suites passed after integration; the final count is
recorded in the integration status. A real-data smoke check ran all three tools
on the first 30 seconds of all 23 AFDB records, both channels: 138 successful tool
calls. No rhythm labels were read and no diagnostic accuracy claim is made.
Results: /data/1shared/gaojiajun/ECGagent/runs/physio-tools-20261116.
Deployment backup, test logs and environment records:
/data/1shared/gaojiajun/ECGagent/migration/physio-tools-20261116.


## 2026-11-17：取消分组数质量门槛

assess_template_quality 升级为1.1.0。template_consistent仅由有限的平均模板相关性>0.8决定，coefficient_group_count保留为order-sensitive审计字段，不再参与质量判定。其他预筛查和缺失规则保留。当前三任务导出为v0.5.1，会对核验后的旧证据重新解释质量规则，以派生证据记录source_evidence_id、source_tool_version及派生说明；原工具返回不变，不重跑检测器或模型。旧v0.5.0结果保留。
