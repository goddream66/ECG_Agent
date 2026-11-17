# CardioAgent V2.1 · M1.5.1

## 代码和数据流程

- 汇总先保留每个通道的尝试和证据，再在模态内判断一致性。每个模态在最终融合中只有一票。若两个 ECG 导联各给出一个可用但冲突的心率，保留两种解释。独立模态的唯一支持可用于消解冲突，无明确支持时保留分歧。
- 两条规则共用检测器和质量门控。`ppg_first` 在首选 ECG 需要复核时先尝试 PPG。`lead_first` 先尝试 II、V3 中可用的备用导联。报告保存动作、对应证据和剩余预算。
- PPG、PCG 的信号指标进入门控，质量报告分开保留信号、事件和模型字段。当前规则分数表示可观察的波形与事件特征，模型质量字段留空。PCG 仍报告周期。
- 新增只退化 ECG I 导联、PPG/PCG 各自加噪、两种辅助模态掉线和三模态同时加噪情景。配对结果保留按受试者聚类的置信区间。心率参考仍由 SensSmartTech 的 ECG 导联生成。 [数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12368122/)
- 统一用左闭右开的样本窗 `[start, end)`。事件报告分别保存发布方的采样网格和设备延迟校准状态。可导入来自独立共同时标的实测延迟与不确定度。当前数据只核验了共同网格，延迟校正路径用已知偏移的测试样例验证。
- 关键审计保存递归源码、脚本、配置、测试文件 SHA-256、软件版本、运行参数哈希和可用的 Git commit。当前项目目录没有 Git 历史，commit 字段为空。

## 退化实验

338 条记录各跑 12 个状态。下表列 69 条测试记录中的关键场景，单元格为返回条数、MAE、平均实际调用。MAE 在有输出的记录上计算。

| 场景 | PPG 优先 | 备用 ECG 导联优先 |
|---|---|---|
| 原始 | 69/69，0.297 bpm，1.01 次 | 69/69，0.297 bpm，1.01 次 |
| 仅 I 导联低信噪比 | 69/69，0.614 bpm，2.03 次 | 69/69，0.170 bpm，2.00 次 |
| 仅 I 导联局部掉线 | 69/69，0.614 bpm，2.03 次 | 69/69，0.170 bpm，2.00 次 |
| 全部 ECG 导联低信噪比 | 69/69，0.614 bpm，2.03 次 | 69/69，0.614 bpm，4.03 次 |
| PPG 和 PCG 局部掉线 | 68/69，0.301 bpm，1.03 次 | 69/69，0.297 bpm，1.01 次 |
| 三模态同时低信噪比 | 60/69，3.969 bpm，2.99 次 | 60/69，3.969 bpm，4.99 次 |

仅 I 导联受损时，两种规则都覆盖全部记录。备用导联优先的配对 MAE 变化为 −0.444 bpm，按 6 位受试者聚类的 95% 区间为 [−0.580, −0.318] bpm。所有 ECG 导联一起退化时，先复核备用导联平均多花两次调用，回退到 PPG 后的误差相同。这两组人工对照明确了备用导联的收益和成本条件。 [完整场景、逐条结果与置信区间](runs/20261111-224839-c276e0a3/incremental_value.json)

## 事件验证

在 PPG-BP 压缩包中核对 438 对原始文件与合并标注。合并的 `annot.os.t` 是标注脉搏的起止边界，`annot.pk.t` 是峰位置。旧报告混合计算了起点与终点。本轮只用开始边界核验足点，峰单独评分。 [发布包](https://zenodo.org/records/10523285) · [标注方法](https://pmc.ncbi.nlm.nih.gov/articles/PMC11003363/)

| PPG-BP 方法 | 标注者 MG F1 / 时间 MAE | 标注者 PC F1 / 时间 MAE |
|---|---:|---:|
| 波谷足点基线 | 98.62% / 20.14 ms | 98.61% / 19.08 ms |
| 上升沿修正足点（探索性） | 99.08% / 3.84 ms | 99.07% / 4.32 ms |
| 收缩峰 | 100.00% / 12.42 ms | 100.00% / 12.08 ms |

足点默认仍输出波谷。新增的上升沿修正候选单独导出，并附形态分数。它的参数在看过 PPG-BP 后选择，因此该数据上的结果用于探索，验证留给独立标注。 [PPG 与 PCG 外部审计](runs/20261111-224839-1d8b523d/external_aux_events.json)

SensSmartTech 有独立的开发盲标包和测试盲标包，每包 12 段。测试包覆盖全部 6 位测试受试者和两个活动状态。双人文件读入、漏标及时间争议队列、裁决拦截和按划分导出已用临时样例走通。真实表仍待两人独立复核。 [标注协议](data/annotations/annotation_protocol.md)

## 全量 Agent 对照

在原始 69 条测试记录、6 位受试者上重跑。四种方法的共同上限为 6 次工具调用，下表列实际调用和 4 个并发槽下每条记录的平均耗时。

| 方法 | 返回结果 | MAE (bpm) | 平均实际调用 | 平均耗时 |
|---|---:|---:|---:|---:|
| PPG 优先规则 | 69/69 | 0.297 | 1.01 | 0.14 s |
| 备用 ECG 导联优先规则 | 69/69 | 0.297 | 1.01 | 0.15 s |
| 单 Agent | 69/69 | 0.332 | 3.00 | 8.04 s |
| 多 Agent | 69/69 | 0.326 | 3.20 | 21.75 s |

全部返回的心率误差都在 5 bpm 内。多 Agent 相对单 Agent 的配对 MAE 变化为 −0.0054 bpm，按受试者聚类的 95% 区间为 [−0.0171, 0] bpm。并发吞吐和模型服务配置会影响耗时。 [Agent 配对报告](runs/20261111-224839-f0fc0124/agent_comparison.json) · [风险与实际开销图](runs/20261111-224839-f0fc0124/risk_actual_cost.png)

## 复现

```powershell
cd D:\CardioAgent
.\.venv\Scripts\python.exe -m cardioagent.cli analyze-ecg-first 1_10-09-54 --strategy lead_first
.\.venv\Scripts\python.exe -m cardioagent.cli audit-incremental
.\.venv\Scripts\python.exe -m cardioagent.cli audit-aux-events
.\.venv\Scripts\python.exe -m cardioagent.cli prepare-event-review --cohort test
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

复核者填完各自的文件后，运行 `review-event-annotations <盲标包目录>`。完成裁决后运行 `score-event-review <盲标包目录>`。

全量 Agent 对照可另开终端运行 `scripts/Start-LocalModel.ps1 -Slots 4 -ContextPerSlot 3072`，再运行 `audit-agents --budget 6 --workers 4`。用 `plot-agent-cost <agent_comparison.json 路径>` 生成风险与实际开销图。绘图与盲标导出依赖 `.[annotation]` 可选项。

设备延迟校准文件可以用 `events --calibration <JSON 路径>` 导入。文件字段包括 `record_id`、`method`（`shared_hardware_marker`）、指向独立测量的 `reference`，以及按 `ecg`、`ppg`、`pcg` 记录的 `modalities`。每个模态填写 `latency_s` 和 `uncertainty_s`，两者都使用秒。

14 项单元与集成测试通过。使用三个现有权重的 V1 固定推理返回 91.05 bpm，ECG、PPG、PCG 均成功保存表征。
