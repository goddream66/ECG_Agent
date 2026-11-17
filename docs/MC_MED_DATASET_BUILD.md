# MC-MED ECG Agent dataset builder — v0.4.0

Entry: `python -m cardioagent.ecg_agent.data.build_mcmed`.
Code: `/home/gaojiajun/ECGagent/src/cardioagent/ecg_agent/data/build_mcmed.py`.
Source: `/data/1shared/xiaoyujie/mc-med` (read-only).
Output: `/data/1shared/gaojiajun/ECGagent/datasets/mcmed-ecg-agent/v0.4.0/<UTC>-<mode>-<id>/`.
The existing v0.1.0 index package is retained unchanged. Runs never overwrite outputs.

```bash
source /home/gaojiajun/ECGagent/activate.sh
python -m cardioagent.ecg_agent.data.build_mcmed --mode index
CUDA_VISIBLE_DEVICES=0 python -m cardioagent.ecg_agent.data.build_mcmed \
  --mode prelabel --patients 10 --duration-s 600 --device cuda:0
```

prelabel produces auto_labels.jsonl with exactly one label per valid window:
label (original name), label_id, selection_rule=top1, human_verified=false. The
argmax uses all 150 classes; it has no threshold and does not filter low-scoring
outputs. Ties choose the first vocabulary index. The lists labels/label_ids remain
as length-one compatibility fields. Unavailable windows have label=null and
label_status=not_evaluated. This forced top-1 export is requested for this dataset,
not a claim that the pretrained model was trained as a mutually exclusive classifier.
Human review remains separate.

Scores are NOT saved by default. --save-scores optionally retains model_scores/*.npy.
Other outputs: manifest.json, records.json, windows.jsonl, labels.json,
label_mapping.json, model_inputs/, tool_evidence.jsonl and review/annotations.jsonl.
Index mode only writes source/selection/window metadata, without tools/models.
Mutable human review files are excluded from immutable artifact hashes.

Selection uses the official random patient split (train by default), seeded patient
hashing, and a bounded maximum scan. One continuous II segment per patient is chosen
by timestamp then path; diagnoses/outcomes are not exported. Each selected segment
contributes its first requested seconds (600 by default). Times are relative to
source-segment start; segments are not concatenated. No full-library scan is resumed.

Raw physical-unit ECG feeds tools. An independent profile is loaded from
configs/ecg/ecgfounder_preprocessing.json: 50 Hz notch (Q=30), fourth-order
0.67-40 Hz bandpass, 0.4s median-baseline subtraction, then per-window z-score
with epsilon=1e-8 and no clipping. These steps follow the upstream util.py and
dataset.py functions at native 500 Hz. Other source rates are explicitly resampled
to 500 Hz first. All processing is confined to the admitted ten-second window;
old AFDB NeuroECG preprocessing is unchanged. Missing and suspected-flatline
windows receive no label. No clinical PQRST intervals, beat counts or human gold
labels are automatically generated.
Original labels may be overlapping, contradictory or unsuitable for single-lead
clinical interpretation; the highest-scoring original name is retained for review.

v0.1.0 primarily exported scores plus a small candidate mapping. v0.2.0 exported multiple original class labels. v0.3.0 exports only the highest-scoring label with the independent preprocessing profile. The compatibility candidates() helper is
retained but is not used in this pipeline. No new MC-MED test package was generated
for this change, following the request to clear tests and preserve the index.


## Prepared labeling command (v0.3.1)

Configuration: `configs/ecg/mcmed_labeling.json`. Defaults: train, 10 patients,
one continuous II segment per patient, first 600 seconds, 10-second windows,
one GPU, independent ECGFounder preprocessing, original 150-class Top-1.
This remains a bounded development package, not a full-library labeling job.

```bash
cd /home/gaojiajun/ECGagent
# Readiness check: synthetic tools and one synthetic GPU inference only.
bash ops/server/run_mcmed_labeling.sh
# Start the configured real-data labeling job only when requested:
bash ops/server/run_mcmed_labeling.sh --run
# Optional smaller selection:
bash ops/server/run_mcmed_labeling.sh --run --patients 1 --duration-s 60
```

The launcher defaults to checking readiness, creates no patient dataset package,
does not install dependencies or download weights, and writes its readiness report
under the data-disk runs/mcmed-labeling-preflight directory. It checks existing
source tables, tools, checkpoint, preprocessing, GPU and the 150-label output.
ECG_AGENT_GPU selects the physical GPU (default 0); --device cpu is explicit CPU use.
Direct module invocation is also available after activating the environment:
`python -m cardioagent.ecg_agent.data.label_mcmed` (add --run to label).

Quality routing is conservative: nonfinite/suspected-flatline windows receive no
label and skip Founder. Preprocessing failures are retained with no label rather
than aborting the entire package. Template inconsistency, prescreen flags or
unavailable quality evidence set quality_status=review_required; they do not veto
Founder or classify an abnormal rhythm as noise. no_quality_flag is not proof of
clinical interpretability. The 1-second flatline heuristic is not validated for
MC-MED and does not prove lead disconnection. All windows remain in windows.jsonl.
auto_labels.jsonl and review/annotations.jsonl reference quality status/reasons;
label_summary.json counts labels, skipped inferences and quality-review flags.
Every model label still requires human review. No diagnosis/outcome propagation.


## Complete-patient selection (v0.3.2)

`python -m cardioagent.ecg_agent.data.run_batch` labels the configured ten 600-second selections, then every available II segment across all visits of the first additional eligible patient in the same seeded patient order. Patients cannot overlap. Source gaps are preserved. Final fragments shorter than ten seconds remain indexed and unlabeled; no padding or cross-segment windows. Selection, header failures, output paths and progress are stored under data-disk runs/mcmed-labeling-batch. Complete refers to available source ECG, not continuous coverage throughout the visit.


## Ranked candidates and visible scores (v0.4.0)

The current export replaces forced single-label presentation. Each valid window
retains ranked_candidates (all 150 labels with original rank and sigmoid score),
top1_label and top1_score (unmodified original winner), and display_candidates
(five highest-ranked labels excluding ABNORMAL ECG, BORDERLINE ECG and otherwise
normal ecg, case-insensitive). Those three broad descriptions remain in
summary_candidates and the full ranking, and are never deleted from source scores.
NORMAL ECG and rhythm labels remain eligible for display. labels/label_ids now
contain the five displayed candidates; legacy label/label_id still alias the
original first rank. Consumers must use selection_rule=ranked_candidates rather
than treating these five candidates as five positive diagnoses. No score threshold
is imposed and no scores are renormalized. Invalid windows have null score/rank fields.
Scores are uncalibrated sigmoid outputs, not established confidence probabilities.
The prepared configuration now also saves full score arrays. Earlier sections
describe prior versions. v0.3.2 packages and visualizations are preserved.


## 当前默认入口：v0.5.0 三任务

当前默认 run_mcmed_labeling.sh 已切换至 label_mcmed_tasks；参见 docs/MC_MED_THREE_TASK.md。只从核验后的 v0.4.0 证据包提取 AF、心率过快、心率过慢，不再输出其他分类标签。默认只显示协议，加 --run 生成三任务包；不进行额外神经网络推理，不扫描新患者。之前各版本为历史记录。


## 2026-11-17：取消分组数质量门槛

assess_template_quality 升级为1.1.0。template_consistent仅由有限的平均模板相关性>0.8决定，coefficient_group_count保留为order-sensitive审计字段，不再参与质量判定。其他预筛查和缺失规则保留。当前三任务导出为v0.5.1，会对核验后的旧证据重新解释质量规则，以派生证据记录source_evidence_id、source_tool_version及派生说明；原工具返回不变，不重跑检测器或模型。旧v0.5.0结果保留。
