# 先复用 ECG-Scroll 的任务和区间评分

本页记录任务导入、AFDB 离线缓存、ECGFounder 逐窗回放和区间评分。

## 已有工程入口

- ecg_agent/benchmarks/tasks.py：读取 ECG-Scroll AF JSONL，只选有本地波形的记录，生成无答案的 Agent 任务和独立评价真值。
- ecg_agent/benchmarks/intervals.py：保留上游按预测顺序的贪心 IoU 匹配、空-空 F1=1、相对误差负荷评分。
- ecg_agent/benchmarks/evaluate.py：严格核对任务 ID，计算每任务和汇总指标，禁止遗漏难例、越界区间或重复提交同一任务。
- `prepare-ecg-scroll` 和 `score-ecg-intervals` 两个 CLI 命令。

## 最小数据包

已经生成：D:/CardioAgent/runs/20261113-203626-a7c31aec/

```text
agent_tasks.jsonl             # 唯一交给 Agent 的任务清单，没有答案/事件数量
                             # 含本地波形路径、绝对时间范围和任务说明
 evaluation_references.jsonl  # 只给评分器，含目标区间和负荷
empty_predictions.jsonl       # 永远不报事件的基线/预测格式模板，不是模型结果
 task_manifest.json          # 来源哈希、抽样规则、标签口径、样本统计
```

6 条不同 AFDB 记录，各取固定的前 30 分钟，共 3 小时；2 个阳性任务，4 个阴性任务。按记录标识的确定性哈希选择，不按结果挑选片段。这个包只有工程验证用途，尚未划分训练/验证/测试，不用于调参后再声称独立测试。

保留上游 AFIB+AFL 标签；与 AFIB-only 有区别，不能混合比较。半小时片段会截断事件，评价的是该观察区间内事件，不宣称真实事件在片段末尾结束。使用 --duration-s 0 可导入完整记录。

## 第一步：导入任务（已运行，无需重复）

在 D:/CardioAgent：

```powershell
.venv\Scripts\python.exe -m cardioagent.cli prepare-ecg-scroll --max-records 6 --duration-s 1800
```

默认来源 D:/ECGagent/ECG/data/tasks/ecgscroll_lh.jsonl 和对应 data/raw。输出新建在 runs 时间戳目录，不复制大波形。

## 第二步：让 Agent 产生区间

每个任务返回一行 JSON；必须覆盖全部 task_id。时间是相对原记录的绝对秒数，不是相对当前 30 秒窗口。

```json
{"task_id":"与 agent_tasks.jsonl 完全相同的 ID","intervals":[[120.0,180.0],[420.0,510.0]]}
```

复制 empty_predictions.jsonl 为 my_predictions.jsonl 后，逐任务填入模型/规则真正预测的区间；无事件时保留空列表。不能把 evaluation_references.jsonl 的答案复制为模型输出。

`replay-afdb` 命令可用现有 ECGFounder 权重生成候选区间。在 D:/CardioAgent 运行：

```powershell
.venv\Scripts\python.exe scripts/preprocess_afdb.py --all --output runs/afdb-preprocessed-full-20261113
.venv\Scripts\python.exe -m cardioagent.cli replay-afdb --cache runs/afdb-preprocessed-full-20261113 --tasks runs/20261113-203626-a7c31aec/agent_tasks.jsonl --channel ECG1 --threshold 0.5
```

全量缓存已生成时可直接执行第二条命令。预处理按各通道连续切分 10 秒窗口，分别滤波、重采样和归一化。回放按窗口结束时刻推进模拟游标，每次把一份已到达的缓存行交给模型。回放只读取缓存与无答案的任务清单。窗口越过任务边界时只计入未评分时长。缺失和不完整窗口保留状态，AF/AFL 分数为 null。候选区间仅合并连续的阳性窗口，遇到缺失时断开。

输出目录包含 `window_trace.jsonl`、`replay_report.json` 和可交给现有评分器的 `predictions.jsonl`。轨迹包含窗口时间、模拟游标、AF/AFL 分数、事件更新时刻及推理耗时。报告包含覆盖率及未评分时长。运行时指定 ECG1 或 ECG2，解剖导联记为未知。`0.5` 是工程试跑阈值，须在独立开发数据上确定正式实验阈值。事件写入的模拟时刻与实际运行耗时分别记录，当前区间评分不计算检测延迟。

可在同一条 `replay-afdb` 命令末尾指定复核策略。

| 参数 | 行为 |
|---|---|
| `--policy threshold` | 默认基线。主通道单窗过阈值就写入区间。 |
| `--policy two-window` | 主通道连续两窗过阈值才写入区间，起点回填到已保存的首个支持窗。 |
| `--policy dual` | 每窗评分两条通道，连续两窗双通道阳性才确认。通道分歧记为待复核。 |
| `--policy adaptive` | 按需调用 ECG2、质量及 RR 工具。强阳性首窗可以先等，临界分数或待确认候选触发复核，确认后的持续候选每隔 3 个窗口复查。 |

`adaptive` 的额外调用预算由 `--review-budget` 设置，默认为每窗 4 次。临界分数区间为 `[阈值, 阈值+0.10)`。确认要求两个连续支持窗，且确认当窗两条通道均过阈值、质量检查可用。原始波形只用于当前窗的质量和 RR 测量。分歧时测得的 RR 保存在证据记录中，第一版策略将分歧留给复核。不可用或分歧的窗口打断连续支持，避免用未证实的区间跨越缺口。

复核策略另输出 `event_memory.jsonl`，用 `evidence_id` 关联 `window_trace.jsonl` 中的工具结果。候选起点与实际确认时刻分别保存。`review_seconds` 统计通道分歧或证据不足的已到达窗口时长，`coverage` 仍指主通道评分覆盖率。`model_calls` 与 `review_calls` 分别统计模型执行次数和额外工具请求次数。正式比较还应关注待复核时长和短事件召回。

## 第三步：评分

下面先跑已生成的始终为空基线，以验证通路：

```powershell
.venv\Scripts\python.exe -m cardioagent.cli score-ecg-intervals --references runs/20261113-203626-a7c31aec/evaluation_references.jsonl --predictions runs/20261113-203626-a7c31aec/empty_predictions.jsonl
```

之后将 --predictions 替换为自己的 my_predictions.jsonl。评分器不调用 LLM、不读取模型内部置信度，只比较最终区间。

输出包括：
- 每记录事件 F1（IoU>=0.5），TP/FP/FN，宏平均与微平均。
- 阳性和阴性任务分别统计；空预测在阴性任务上得 1 分，不能只看总平均。
- 负荷由预测区间并集时长/任务时长导出；同时报告上游负荷 score 和绝对误差百分点。
- 每小时误报事件数，分母为全部记录观察时长。
- 输入哈希与源码指纹。

原有空预测基线报告：runs/20261113-203721-49f7e3ba/interval_scores.json。它仅为工程基线，不是 Agent 成绩。

## 验证与差异

58 项本地测试通过。回放测试覆盖未来窗口隔离、无效窗口和错误缓存行号。区间和负荷评分另与本地上游 verify.py 对照了 200 组随机案例，数值一致。

本实验覆盖 AF/AFL 区间和负荷。完整记录任务保留上游四舍五入后的负荷参考。截取片段按截断后的真值区间重算负荷。预测区间对 F1 保留原顺序，负荷计算使用并集防止重复计时。
