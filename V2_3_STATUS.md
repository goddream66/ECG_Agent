# CardioAgent V2.3 · 任务条件化的生理证据工具选择

V2.3 在已记录的 ECG、PPG 和 PCG 上决定下一步分析哪个通道、是否返回当前心率、是否拒答。V2.2 的源码与历史实验分开保存在 `runs/v2_2_frozen_source.zip` 和原运行目录。V2.3 只训练策略网络，沿用当前的 ECG、PPG、PCG 测率工具和基础模型权重。

## 实现与评价口径

| 模块 | V2.3 实现 |
| --- | --- |
| 任务和动作 | `heart_rate` 以 ECG 为起点，可以复核 ECG II/V3、测 PPG、测 PCG、停止或拒答。`peripheral_pulse_rate` 从 PPG 开始，独立输出外周脉率。当前策略训练和匹配实验使用 `heart_rate`。 |
| 通道质量 | 每个候选通道独立计算频带能量占比、周期强度和连续平线比例。质量预览不返回心率，其调用和耗时单独记录。 |
| 规则策略 | 对照 ECG-only、PPG fallback、备用 ECG 优先、通道质量、固定 Gaussian EIG、通道 EIG 和按训练集实测工具耗时归一化的通道 EIG。Gaussian 误差宽度是启发式设定。 |
| Learned VOI | 以可用通道、已观察测率、事件稳定性、分歧和剩余预算等为输入。带两层隐藏层的 MLP 用同一条记录中实际离线运行的工具输出监督，估计采取各个合法动作后到停止为止的预期损失。部署时只接收已获取证据和预览特征。 |
| 可靠停止 | 训练目标使用绝对误差大于 2 bpm 的二元损失，拒答损失固定为 0.40，工具惩罚为训练集相对耗时乘以 0.025。校准器在独立验证受试者的未加门控候选上拟合，按受试者均衡加权。预设风险上限 0.20 只限制 STOP 的可用性，可选动作按原始价值分数排序。 |
| 成本与 Oracle | 测率和 QC 的实测离线耗时用于回放，在线策略循环单列。动作目标的惩罚使用训练集的测率耗时，QC 记入报告总耗时。Oracle 知道参考值和所有工具结果，用于给出该目标的参考上界。 |
| Risk–coverage | 规则策略按启发式证据质量排序。验证集校准图对未加门控 VOI 已返回的结果扫阈值，闭环校准策略另标实际运行点。JSON 每个阈值含接受记录数、受试者数、覆盖率、选择性错误率、无条件失败率及按受试者重抽样的区间。 |
| LLM 对照 | `--protocol structured` 给 LLM 传观察证据的数值质量、事件特征、跨通道测率差和候选通道质量。Two-stage LLM Coordinator 用同一个 Qwen 先提议再决策。成本图同时报告覆盖率和无条件失败率。 |

选择性错误率以返回结果为分母，统计绝对误差大于 2 bpm 的比例。无条件失败率以全部记录为分母，拒答和绝对误差大于 2 bpm 都计为失败。每条曲线只报告该指标有定义的点。校准曲线和闭环运行点使用相同的验证校准器，但闭环策略可能为了达到停止条件请求额外证据。

## SensSmartTech 预实验

数据按受试者划分为训练 20 人（211 条）、校准 6 人（58 条）、测试 6 人（69 条）。每个扰动条件使用相同的测试记录，训练包含同样定义的 7 种条件。16 轮训练使用了 21,100 个由训练集工具回放生成的状态。

**实验 A，逐通道动作。** 只干扰 ECG I 时，固定 Gaussian EIG 的 MAE 为 0.602 bpm，通道 EIG 为 0.160 bpm，备用 ECG 优先为 0.170 bpm，PPG 优先为 0.614 bpm。通道 EIG 在 69 条里有 68 条先复核备用导联。参考值和备用导联存在输入重叠。

**实验 B，条件性的辅助价值。** 下表报告与 ECG-only 相比，误差 ≤2 bpm 的返回记录占比增加了多少个百分点，拒答也计入分母。对退化条件中的每一列，都使用同一批被质量门判为需复核的 ECG I 记录。

| 条件 | 加 PPG | 加 PCG |
| --- | ---: | ---: |
| 原始且 ECG I 通过质量门（68 条） | 0.0 | −1.5 |
| 所有 ECG 导联低 SNR | +91.3 | +89.9 |
| 所有 ECG 导联与 PPG 低 SNR | +1.4 | +89.9 |
| 所有 ECG 导联与 PCG 低 SNR | +91.3 | +68.1 |
| 三模态均低 SNR | 0.0 | +78.3 |

人工加噪的结果支持在 ECG 和 PPG 同时退化时测试 PCG 的条件价值。原始记录仅有 1 条 ECG I 被判为需复核。联合退化下 PCG 的 +78.3 个百分点以 ECG-only 的拒答结果为参照。

**实验 C，学习策略与规则。** Learned VOI 尚未稳定超过规则。只干扰 ECG I 时，其 MAE 为 0.524 bpm，平均回放分析耗时为 106.0 ms。备用 ECG 优先的 MAE 为 0.170 bpm，平均耗时 191.8 ms。全部 ECG 导联干扰时，按成本归一化的通道 EIG 无条件失败率为 1.45%，Learned VOI 为 4.35%。

**实验 D，校准与拒答。** 独立验证集的受试者均衡 Brier 从 0.0979 降至 0.0925。原始记录两种 VOI 策略都返回 69 条，MAE 均为 0.297 bpm。三模态均低 SNR 的结果如下。校准门槛减少了返回结果中的错误，也拒绝了更多原本正确的结果。

| 策略 | 返回数 | 覆盖率 | 选择性错误率 | 无条件失败率 | 平均回放分析耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 通道质量规则 | 60/69 | 87.0% | 10.0% | 21.7% | 132.2 ms |
| Learned VOI | 58/69 | 84.1% | 10.3% | 24.6% | 243.3 ms |
| Learned VOI + 校准 STOP | 46/69 | 66.7% | 6.5% | 37.7% | 243.3 ms |
| Oracle | 54/69 | 78.3% | 0.0% | 21.7% | 96.4 ms |

[完整报告](runs/20261112-132443-e2e9fd6c/v23_audit.json) · [策略权重](runs/20261112-132443-e2e9fd6c/value_network.pt) · [规则风险–覆盖率图](runs/20261112-132443-e2e9fd6c/heuristic_risk_coverage.png) · [校准风险–覆盖率图](runs/20261112-132443-e2e9fd6c/calibrated_risk_coverage.png) · [风险–成本图](runs/20261112-132443-e2e9fd6c/voi_risk_cost.png)

## 复现

训练和加载策略检查点使用现有 `models` 可选依赖中的 PyTorch。画图使用 `annotation` 可选依赖中的 Matplotlib。

```powershell
cd D:\CardioAgent
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m cardioagent.cli audit-v23 --allow-published-pilot
.\.venv\Scripts\python.exe -m cardioagent.cli plot-v23 '<运行目录>\v23_audit.json'
.\.venv\Scripts\python.exe -m cardioagent.cli analyze-v23 1_10-09-54 --task peripheral_pulse_rate
.\.venv\Scripts\python.exe -m cardioagent.cli analyze-v23 1_10-09-54 --method learned_voi_calibrated --checkpoint '<运行目录>\value_network.pt'
.\.venv\Scripts\python.exe -m cardioagent.cli audit-aux-events
```

结构化 Agent 验证使用 `scripts\Start-LocalModel.ps1 -Slots 4 -ContextPerSlot 3072` 启动本地模型，再执行 `audit-agents --protocol structured --budget 6 --workers 4 --per-subject 1 --scenario lead_i_low_snr`。用 `plot-agent-cost <agent_comparison.json>` 生成成本及覆盖率图。

结构化 Agent 冒烟测试包含 6 名测试受试者，每人一条 I 导联低 SNR 记录。单阶段、两阶段和备用 ECG 优先规则均返回 6 条且误差全部 ≤2 bpm。单阶段平均调用 LLM 2 次，两阶段为 4 次，二者均平均调用 4 次通道 QC。 [对照报告](runs/20261112-131653-f486b0a7/agent_comparison.json) · [覆盖率与无条件失败图](runs/20261112-131653-f486b0a7/coverage_unconditional_risk.png)

22 项单元测试通过。心率单记录推理加载最终权重，测量 ECG I 后输出 91.12 bpm。外周脉率单记录推理只运行 PPG 测量，输出 91.25 bpm。 [心率推理](runs/20261112-133506-ee770128/v23_analysis.json) · [外周脉率推理](runs/20261112-131234-ce8ccbf6/v23_analysis.json)

## 独立参考与外部评估

开发盲标包 `runs/20261111-224501-f9fb19c0` 和独立测试盲标包 `runs/20261111-224501-b9990f52` 各含 12 个窗口，等待双人提交及裁决。裁决完成后，用 `audit-v23 --development-packet <开发包> --test-packet <测试包>` 以对应时间窗的人工 R 峰重跑。命令校验训练、校准和测试受试者互斥，并分别记录参考来源。以公开参考做预实验需要显式提供 `--allow-published-pilot`。公开参考由 ECG I、II、V3 派生，备用 ECG 导联的收益需要结合参考重叠解读。 [SensSmartTech 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12368122/)

PPG-BP 每段只标注一个脉搏，当前 F1 的候选范围限定在标记点 ±150 ms，配对容差 50 ms。报告另计窗外候选数，连续检测 F1 留空。CirCor 读取的是已发布的 S1/S2 参考。整个多模态策略的外部泛化、独立 SensSmartTech 参考下的比较仍待数据。 [外部事件报告](runs/20261112-131556-2e1c5014/external_aux_events.json) · [pyPPG 标注说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC11003363/)
