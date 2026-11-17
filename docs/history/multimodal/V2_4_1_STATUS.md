# CardioAgent V2.4.1 · 动作价值与风险评价审计

V2.4.1 复用离线波形回放，重训小型策略网络。V2.4 归档和旧检查点仍可运行。

## 修改

- 训练时用同一反事实递归生成动作价值和优势 `A(s,a) = min_b Q(s,b) − Q(s,a)`。两者共享终止误差、拒答、测量成本与决策成本。优势头用于辅助监督，部署按 Q 选择动作。
- QC 调度使用训练集上按任务、候选通道、主通道质量和已观测跨模态分歧分组的经验分布。每组至少五名训练受试者，支持不足时逐级回退。报告验证集 CRPS 和各动作的分布查询层级。
- 匹配覆盖率同时使用方法自身评分和一个统一的风险评分器。统一评分器只用训练受试者的可返回测率和已观测证据拟合，规则与学习策略共用。保持各策略原生的返回与拒答决策，再对已返回结果作匹配覆盖率比较。
- BIDMC 分域验证保留 20% STOP 门槛，比较无门槛、SensSmartTech 校准直接迁移和 BIDMC 目标域验证集校准。主实验从策略训练和源域校准中排除了 BIDMC HR，目标域校准与最终测试的受试者互斥。
- 另跑同域共享实验，在相同 BIDMC 原始窗口同时具备 `HR` 与 `PULSE` 参考时配对比较单任务和共享编码器。HR 可以调用 ECG/PPG，PR 只有单路 PPG。 [BIDMC 数据说明](https://physionet.org/content/bidmc/1.0.0/)

## 实测

SensSmartTech 测试使用 6 名受试者、69 条记录。仅 I 导联退化时，V2.4.1 Conditional VOI 的首个追加测量均为 PPG，MAE **0.408 bpm**。V2.4 为 **0.304 bpm**，备用 ECG 优先为 **0.170 bpm**，两者配对的 `V2.4.1 − 备用 ECG 优先` MAE 差 95% 受试者重抽样区间为 **0.154–0.321 bpm**。动作策略未达到该场景的验收目标。验证集 QC 分布在原始条件上的 CRPS 从 0.159 降到 0.101，在 I 导联单独退化时则从 0.159 升到 0.178。SensSmartTech 公开参考由 ECG 派生，独立裁决仍待完成。 [SensSmartTech 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12368122/)

三模态低 SNR 的匹配覆盖率为 49/69。使用各自风险排序时，V2.4.1 Conditional VOI 的选择性错误率为 **2.0%**，通道质量规则为 **6.1%**。使用统一风险评分时，两者均为 **6.1%**，通道 EIG 为 **8.2%**。Conditional VOI 的原生返回为 60/69，无条件失败率与通道质量规则同为 **21.7%**。 [方法自身评分图](runs/20261112-182420-ec2a1dfa/v241_matched_coverage.png) · [统一评分图](runs/20261112-182420-ec2a1dfa/v241_shared_risk.png)

| BIDMC 外部 HR · ECG 低 SNR | 返回数 | 无条件失败率 |
| --- | ---: | ---: |
| Conditional VOI，无门槛 | 33/36 | 13.9% |
| SensSmartTech 门槛直接迁移 | 15/36 | 61.1% |
| BIDMC 独立验证受试者校准 | 1/36 | 100% |

目标域校准仍未解决低 SNR 的拒答问题。在同域配对实验中，BIDMC 原始窗口的共享模型 HR MAE 为 0.577 bpm，单任务为 0.593 bpm，两者都返回 35/36。PR 两者均为 0.614 bpm，返回 32/36。这个对照只有 9 名留出受试者，PR 也只有一个测量模态。 [同域配对报告](runs/20261112-182420-de6a0558/v241_audit.json)

## 复现与外部小样本

29 项测试通过，检查点已实际加载推理。MIMIC 流程用合成的同格式 WFDB 和分折元数据验证了端到端执行。实际 MIMIC 结果待授权用户下载样本。外部评估以发布的 ECG 派生 `median_30s_hr` 为参考，使用官方 fold 8 校准与 fold 9 测试，保存聚合指标。取样脚本按发布顺序扫描有限行数，默认每折最多取八名受试者各一个同步片段，用于小样本验证。 [MIMIC-III-Ext-PPG 发布说明](https://physionet.org/content/mimic-iii-ext-ppg/1.1.0/) · [PhysioNet 数据使用说明](https://physionet.org/news/post/llm-responsible-use/)

[主实验归档](runs/20261112-182420-ec2a1dfa/v241_experiment.zip) · [主实验 JSON](runs/20261112-182420-ec2a1dfa/v241_audit.json) · [同域实验归档](runs/20261112-182420-de6a0558/v241_experiment.zip)

```powershell
cd D:\CardioAgent
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m cardioagent.cli prepare-v24-bidmc runs\v24-replay\bidmc-paired.jsonl --paired-hr
.\.venv\Scripts\python.exe -m cardioagent.cli audit-v241 --v23-checkpoint runs\20261112-132443-e2e9fd6c\value_network.pt
.\.venv\Scripts\python.exe -m cardioagent.cli audit-v241 --paired-bidmc
.\.venv\Scripts\python.exe scripts\Get-MimicPilot.py --output data\mimic-iii-ext-ppg-pilot
.\.venv\Scripts\python.exe -m cardioagent.cli audit-v241-mimic --source data\mimic-iii-ext-ppg-pilot --checkpoint runs\20261112-182420-ec2a1dfa\models.pt
```
