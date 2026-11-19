# P3-1 证据理解与反事实实验状态（2026-11-19）

## 范围

固定 AFDB P3-0 的 35 个判断点、已到达的 ECGFounder 原窗分数和 Qwen3-8B，不更新模型参数。主组含 16 个参考结束点及 14 个普通持续点；困难组含 5 个短暂低分但参考仍持续的点。同一判断接口输出 `AF_AFL_ONGOING / AF_AFL_ENDED / UNCERTAIN`，并引用支持与反对结束的证据 ID。

- raw：原始测量列表。
- aligned：相同测量按时间窗口对齐，不添加诊断标签。
- reasoned：aligned 输入，另要求先分析支持与反对证据。
- counterfactual：在 reasoned 输入中单独改变备用导联分数、一次历史恢复分数，或仅反转测量排列。人为改分仅用于敏感性测试，不作为真实 ECG 或医学真值。

最初一次试跑使用含糊的 `TRANSITION` 输出，模型将其理解为进入 AF；那批结果作废。以下全部来自明确表述“当前已确认的 AF/AFL 是否结束”的重跑。实验不允许模型读取参考标签。

## 固定证据结果

| 输入 | 有效输出 | 真实结束识别 | 普通持续误结束 | 困难持续误结束 | UNCERTAIN |
|---|---:|---:|---:|---:|---:|
| raw | 32/35 | 7/16 | 0/14 | 3/5 | 0 |
| aligned | 35/35 | 7/16 | 0/14 | 0/5 | 0 |
| reasoned | 35/35 | 10/16 | 0/14 | 3/5 | 0 |

raw 的 3 次无效输出不能算作正确持续判断。aligned 减少困难点的错误结束，但没有提升真实结束识别。reasoned 提高真实结束识别，同时重新产生 3 个困难点错误结束。三组均未使用不确定选项，因此本实验无法在相同拒判覆盖率下验证更好的选择性判断。样本按错误分析目的选取，数量小且非独立，不能外推为 AFDB 全量性能。

## 证据敏感性

| 改动 | 有效案例 | 最终决定改变 | 改动证据 ID 被引用 |
|---|---:|---:|---:|
| 备用导联高低分互换 | 24 | 3 | 23 |
| 历史恢复分数互换 | 5 | 0 | 5 |
| 只反转呈现顺序 | 35 | 3 | 不适用 |

引用 ID 只说明模型提及测量，不能证明引用方向正确。两例末窗导联冲突均引用了备用导联作为反对结束的证据，但至少一例出现同一 ID 同时列在支持与反对栏。分数方向自动检查仅产生待复核提示；历史高分可能用于描述转变前的边界，不应把所有提示直接计为解释错误。反事实分数是人为修改，决定不变也不必然代表推理错误；需要逐例定义期望的证据关系变化。当前没有经过独立审核的反证识别率。

## P3-2：证据可判定性

已生成 35 份截止判断时刻的双导联波形图、无真值的测量文件和盲审表。参考答案单独存放。盲审在 `review.csv` 中记录 `review_status=reviewed`、`decidability`、`waveform_judgment`、审核者及依据。`decidability` 可填 `existing_sufficient`（已有证据可判定）、`need_measurement`（需要合法补测）、`need_wait`（需要未来窗口）、`tools_insufficient`（现有工具不能判定）或 `indeterminate`（波形仍不确定）；`waveform_judgment` 可填 `AF_AFL_ONGOING / AF_AFL_ENDED / UNCERTAIN`。审核完成后由 `analyze_decidability_review.py` 解盲汇总，并分别报告与参考标注及三组 Qwen 判断的差异。尚无独立审核结果，因此目前不能统计根因比例，也不能据此决定 SFT 或主动取证优先级。

## 后续门槛

1. 独立复核困难点、真实结束点及证据引用方向；先评估波形与已获取测量是否足以区分持续和结束。
2. 若主要是证据可判定但读取/推理错误，构建带人工审定依据的后训练与规则配对实验；若主要是信息不足，比较备用导联、RR 等合法补测与等待新窗口的增量收益。
3. 固定方法后才接回连续在线事件追踪，报告事件 F1、提前关闭、结束延迟、边界误差、拒判覆盖率与工具/LLM 成本，并与同证据规则比较。

## 复现位置

- 代码：`ops/server/evidence_reasoning_p31.py`、`summarize_evidence_reasoning_p31.py`、`prepare_decidability_review.py`、`analyze_decidability_review.py`。
- 服务器结果：`/data/1shared/gaojiajun/ECGagent/runs/afdb-evidence-reasoning-p31-end-20261119/`。
- 盲审材料：结果目录下 `decidability_review/review.csv`、`blinded_evidence.jsonl`、`case-*.png`；答案在单独的 `answer_key.jsonl`。
- 测试：服务器 208 项单元测试通过。
