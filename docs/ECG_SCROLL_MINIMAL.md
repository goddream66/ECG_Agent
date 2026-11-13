# 先复用 ECG-Scroll 的任务和区间评分

本轮只复用任务与离线评分，尚未增加 AF 推理控制器、事件记忆或在线延迟指标。

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

当前 ecg-agent 命令仅输出心搏、速率和复核，尚不能自动生成 AF 区间。还需一个候选规则/模型加事件区间组装器。最小接入路径是：每 30 秒调用 get_rate_rhythm -> 在开发数据确定的固定候选规则 -> 合并相邻阳性窗口 -> 输出 intervals。

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

55 项本地测试通过。另与本地上游 verify.py 对照了 200 组随机区间和负荷案例，数值一致。

这里复用两类核心评分，不是完整 ECG-Scroll 复现：没有变化点、缺血、PVC、稀有事件和延迟评分，没有复现论文 LLM 推理实验。完整记录任务保留上游四舍五入后的负荷参考；截取片段则按截断后的真值区间重算。预测区间对 F1 保留原顺序，不擅自排序合并；负荷计算使用并集防止重复计时。

下一步先实现规则候选到事件区间的转换，再用相同接口替换为 LLM 控制器；不需要为建立评分通路先训练大模型。
