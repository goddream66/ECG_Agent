# P3-2：两阶段证据可判定性盲审（2026-11-20）

## 要回答的问题

P3-1 的 Qwen 只看到了截止判断时刻已取得的 ECGFounder 分数；原 P3-2 审阅者同时看分数和原始波形。两者信息不同，因此原盲审即使发现 Qwen 错、审阅者对，也无法区分分数证据误读与波形信息缺失。新版分两阶段，同一案例先审分数，再审波形。

| 阶段 | 审阅信息 | 要记录什么 |
|---|---|---|
| A | 与 P3-1 Qwen 相同的测量条目、时间、状态和证据 ID；无参考标签、模型决策和原始波形 | AF/AFL 是否已结束、现有分数是否足以判定、1–5 置信度、引用的 ID 与理由 |
| B | A 的信息加上截至游标前最多 60 秒的 ECG1/ECG2 波形 | 波形是否新增可区分信息、判断、置信度、波形依据；不足时还缺什么 |

“足以判定”是审阅者对**当前可见证据**的判断，不等于参考标注无误，也不等于所有医生都会同意。B 阶段宜由另一名具备 ECG 判读能力的审阅者完成，或至少与 A 阶段隔开并盲于 A 的答案。审核者不得查看私有答案键、P3-1 输出或后续时刻信号。

## 防泄漏与审阅顺序

- 35 个判断点来自 3 条记录、21 个不同的记录时间点；两导联及相近时刻会重复。因此这些不是 35 个独立患者样本。
- 案例按匿名记录、游标升序排列。审阅者只接收 `stage_a/` 或 `stage_b/` 目录，不接收父目录；父目录保留真实记录映射、参考答案和尚未释放的案例。
- 每阶段先释放第一例。公开的 `review.csv` 也只含已释放的行，因此不暴露较晚案例的时间表。审阅者完成当前行后，研究者执行 `release-next`，才追加下一行并复制对应证据。B 阶段只有 A 阶段全部完成后才能生成；其第一例同样逐例释放。
- 每个分数测量的窗口终点不得晚于判断游标。B 阶段波形读取到游标为止。若 A 阶段 CSV 在 B 生成后被修改，分析程序会因哈希不一致而拒绝解盲。
- 仅用文件权限隔离私有父目录与公开阶段目录；共享同一服务器账号的研究者仍可访问父目录，所以审阅者应只拿到单独交付的公开目录。记录审阅顺序、审阅者、失败及偏离协议的情况。

## 执行顺序

```bash
python ops/server/prepare_decidability_review_v2.py stage-a \
  --cases /data/1shared/gaojiajun/ECGagent/runs/afdb-fixed-transition-p3-0-v2-20261119 \
  --output /data/1shared/gaojiajun/ECGagent/runs/afdb-decidability-p32-v2-gated-20261120

# 审阅者填写 stage_a/review.csv 当前已释放案例的行后，研究者逐次执行：
python ops/server/prepare_decidability_review_v2.py release-next \
  --output /data/1shared/gaojiajun/ECGagent/runs/afdb-decidability-p32-v2-gated-20261120 \
  --review-stage stage-a

# 只有 35 例 Stage A 全部完成后，才能释放 Stage B：
python ops/server/prepare_decidability_review_v2.py stage-b \
  --output /data/1shared/gaojiajun/ECGagent/runs/afdb-decidability-p32-v2-gated-20261120 \
  --raw-root /data/1shared/gaojiajun/ECGagent/data/afdb_local

# Stage B 同样逐例填写 review.csv 并执行 release-next --review-stage stage-b。
# 两阶段全部完成后再解盲：
python ops/server/analyze_decidability_review_v2.py \
  --review-root /data/1shared/gaojiajun/ECGagent/runs/afdb-decidability-p32-v2-gated-20261120 \
  --results /data/1shared/gaojiajun/ECGagent/runs/afdb-evidence-reasoning-p31-end-20261119 \
  --output /data/1shared/gaojiajun/ECGagent/runs/afdb-decidability-p32-v2-gated-20261120/analysis.json
```

## 解盲后的分析

逐例区分：A 已足以判定；A 不足但 B 足以判定；A、B 均不足。分别报告审阅者与 AFDB 参考标注的符合情况，再统计 Qwen 在各组的错误。若 B 仍不足，记录需要新的合法测量、等待后续窗口、现有工具缺失，或仍无法判定。所有数量同时报告案例数及不同记录时间点数；不给 35 个高度相关案例计算假精确的独立样本置信区间。

只有在**A 阶段高置信判断与参考一致、Qwen 却错误**时，才把该例列为“现有分数证据下可能的推理错误”；不是直接认定训练能解决。A 不足、B 足且 B 与参考一致，才列为“波形新增信息候选”。A、B 均不足不自动证明必须等待未来窗口，还要看审阅者指出的缺口。

## 本轮进度

代码已实现阶段门控、逐例释放、时间边界检查、封存校验和解盲统计。已用合成案例测试完整 A→B→分析流程；本地与服务器完整测试均为 209 项通过。服务器正式运行的 `preflight.json` 显示：35 例来自 3 条记录、21 个不同记录时间点；A 阶段只释放第 1 例，已完成审核 0 例，B 阶段未释放。实测尝试启动 B 阶段被 `Incomplete Stage A review: 1/35 cases released` 拒绝。

截至本轮，没有独立审阅结论，不能报告三类根因比例或声称改进 Event F1。反事实实验的证据识别与归因指标也需先定义逐例的正确解释，不能只以决策是否变化替代。
