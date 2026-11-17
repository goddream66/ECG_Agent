# CardioAgent V2.2 · 条件证据获取

## 公用动作和成本

`ppg_fallback` 和 `alternate_lead_first` 都以首选 ECG 导联开始。前者在 ECG 需要复核时先查 PPG，后者先查备用 ECG 导联。旧策略名仍可用于现有命令，输出使用新名称。

主动单 Agent、主动多 Agent 与规则共享初始观察、测率工具、可用通道、合法动作和波形预算。动作包括复核备用 ECG 导联、调用 PPG 或 PCG、停止、拒绝。初始 ECG（可用时）计一次请求。多 Agent 每轮可先询问一个证据专家，协调者随后从同一组合法动作中选择。两次调用都计入 LLM 开销。审计输出每条记录的动作路径、实际消费、错误和无效动作。历史全模态 Agent 流程可用 `--protocol legacy` 单独复现。

| 字段 | 计量口径 |
|---|---|
| `waveform_requests` | 进入测量的请求，含缓存命中和已执行的失败 |
| `waveform_tool_calls` | 实际执行的波形工具，含失败执行 |
| `waveform_failures`、`waveform_insufficient` | 分别记录执行失败和证据不足 |
| `waveform_cache_hits` | 复用的测量结果，仍按一次请求计预算 |
| `coordinator_recheck_calls` | 协调者单独发出的复核请求。主动模式的备用导联测量计入波形开销 |
| `llm_invocations`、`input_tokens`、`output_tokens` | 实际模型请求次数与服务端返回的 token；`llm_usage_missing` 标记缺失的用量 |
| `action_budget_units`、`wall_time_s` | 已扣预算单位和单条记录实测耗时 |

质量规则只使用已获取证据。若 ECG 显示持续平坦、基线漂移或高频污染，优先复核备用导联。EIG 基线用固定的离散心率网格与高斯观测假设估算预期熵下降，每项测量成本为 1。其噪声参数为手工设定，未用公开心率拟合。

## 参考值和评测边界

原始表中的参考心率来自 ECG I、II、V3 导联。仅扰动 I 导联时，备用导联策略仍可使用参考值生成时用过的信号。本轮数字表示与 ECG 派生参考的心率一致性。69 条测试记录来自 6 位受试者，配对区间按受试者整簇抽样。 [SensSmartTech 数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12368122/)

`--adjudicated-packet` 接收已经双人盲标并裁决的 R 峰。参考心率从同一时间窗的人工作业计算，波形也裁到该窗。报告记录标注来源与文件哈希，人工参考与公开参考分别评测。空白盲标包无法产生参考值。验证已用临时测试样例跑通，真实 SensSmartTech 标签仍待两位标注者完成。既有逐事件评分仍可分别评估 R 峰、PPG 足点、收缩峰和 PCG 标注。 [双人盲标协议](data/annotations/annotation_protocol.md)

MAE、RMSE 和 P95 只对返回值计算。1、2、5 bpm 成功率以全部记录为分母，拒绝计入失败。条件实验定义风险为拒绝或误差超过 2 bpm，并报告配对风险下降、增加的波形请求、按 ECG 质量分层的结果、受试者级结果及风险与覆盖率曲线。覆盖率曲线用融合状态及已选证据的频带能量、事件稳定性排序，该排序是启发式指标。离线回放比较逻辑请求数，实际执行和延迟使用实时主动审计。

## 条件实验

69 条测试记录各跑 8 种条件。相对 ECG 单模态，固定加入辅助模态在以下条件下的风险变化不同。正值表示「拒绝或误差超过 2 bpm」的比例下降。可用辅助通道各增加一次请求。

| 条件 | 加 PPG 的风险下降 | 加 PCG 的风险下降 |
|---|---:|---:|
| 原始波形 | 1.45 个百分点 | 0.00 个百分点 |
| ECG 低信噪比 | 91.30 个百分点 | 89.86 个百分点 |
| ECG 和 PPG 低信噪比 | 1.45 个百分点 | 89.86 个百分点 |
| ECG 和 PCG 低信噪比 | 91.30 个百分点 | 68.12 个百分点 |

在所有 ECG 导联低信噪比、辅助模态原样的条件下，PPG 回退和备用导联优先的 MAE 都是 0.614 bpm，实际请求分别约 2.03 次与 4.03 次。在三模态都加噪的条件下，两种规则都返回 60/69 条，MAE 3.969 bpm，1/2/5 bpm 成功率和按受试者的结果见明细。

原始波形上两个规则 68/69 条的路径完全相同，均在 ECG 后停止。剩下一条 `11_18-59-02` 分别调用了 PPG 和 ECG II，最终都输出 104.53 bpm。因此两种规则在原始条件的 MAE 和平均请求数重合。

[八条件明细及受试者级区间](runs/20261111-234233-4afc19bf/incremental_value.json) · [条件风险与请求成本](runs/20261111-234233-4afc19bf/conditional_risk_cost.png) · [风险与覆盖率](runs/20261111-234233-4afc19bf/conditional_risk_coverage.png)

## 主动 Agent 实测

原始与仅 I 导联加噪分别运行 69 条记录，波形上限 6 次、LLM 请求上限 12 次。下表列每条记录的平均开销。耗时来自本机四并发槽运行。

| 条件与方法 | 返回条数 | MAE (bpm) | 实际波形调用 | LLM tokens | 耗时 |
|---|---:|---:|---:|---:|---:|
| 原始 · 两种规则 | 69/69 | 0.297 | 1.01 | 0 | 约 0.39 s |
| 原始 · 主动单 Agent | 68/69 | 0.301 | 1.00 | 173 | 2.58 s |
| 原始 · 主动多 Agent | 69/69 | 0.297 | 1.01 | 361 | 4.89 s |
| 仅 I 加噪 · PPG 回退 | 69/69 | 0.614 | 2.03 | 0 | 0.25 s |
| 仅 I 加噪 · 备用导联优先 | 69/69 | 0.170 | 2.00 | 0 | 0.43 s |
| 仅 I 加噪 · 主动单 Agent | 61/69 | 0.173 | 1.88 | 359 | 4.57 s |
| 仅 I 加噪 · 主动多 Agent | 69/69 | 0.170 | 2.00 | 787 | 9.39 s |

主动多 Agent 在 I 导联加噪时比单 Agent 多返回 8 条。两者在共同返回的 61 条上给出了相同结果。备用导联优先规则覆盖了全部 69 条，MAE 与多 Agent 相同。该条件仍使用包含 II、V3 信息的公开参考。 [主动对照明细](runs/20261111-233806-20daa69c/agent_comparison.json) · [波形、Token 和耗时对照图](runs/20261111-233806-20daa69c/risk_actual_cost.png)

## 运行

```powershell
cd D:\CardioAgent
.\.venv\Scripts\python.exe -m cardioagent.cli analyze-ecg-first 1_10-09-54 --strategy alternate_lead_first
.\.venv\Scripts\python.exe -m cardioagent.cli audit-incremental --split test
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

主动 Agent 对照先另开终端启动 `scripts/Start-LocalModel.ps1 -Slots 4 -ContextPerSlot 3072`，再执行 `audit-agents --budget 6 --workers 4 --scenario original --scenario lead_i_low_snr`。用 `plot-agent-cost <agent_comparison.json 路径>` 生成 MAE 对实际波形工具、token 和耗时的三轴对照图。用 `plot-conditional <incremental_value.json 路径>` 生成条件实验图。

18 项单元与集成测试通过。训练型 VOI 依赖经裁决的独立事件参考，当前可运行的 EIG 为规则基线。
