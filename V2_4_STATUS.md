# CardioAgent V2.4 · 条件生理证据价值

V2.4 把心率与外周脉率的价值学习拆为误差、错误风险、返回概率和成本四个目标，并学习追加一次测量的增量价值。本版训练小型策略网络，沿用三个基础模型权重与测率工具。V2.3 的报告、权重和命令保留为对照。

## 实现与实验设计

- `objectives.py` 与 `targets.py` 用离线真实工具输出构造反事实标签。误差头学习截尾至 12 bpm 的条件绝对误差，另外三个头分别学习错误且返回的概率、返回概率、后续成本。训练与执行使用预先声明的损失权重。Oracle 知道参考值和真实工具结果，优化同一主目标。
- `features.py` 为事件间期、置信度和信号质量等特征加入可用标记。`complementarity.py` 和 `conditional_value.py` 使用已观察的同模态和异模态证据、质量与分歧，预测候选测量相对于立即停止或拒答的一步损失改善。
- `selection.py` 记录 QC、完整测量、决策耗时。选择性 QC 从训练集的质量分布近似预览带来的损失下降，只有预计收益覆盖 QC 和额外决策成本时才执行。全量 QC 对照预览所有尚未测量的候选通道。波形预算计完整测量，QC 单列计成本。
- 单任务与共享编码器、任务专用价值头使用相同的数据划分。报告提供关闭任务条件、互补特征和增量项、QC、成本、校准的对照。PR 训练来自 BIDMC 的单路 PPG，任务与训练数据来源相关，这个实验无法分离任务迁移和数据来源的影响。
- 校准比较受试者均衡、记录均衡、受试者内记录均衡与验证集逐受试者留一。报告按场景给出 Brier、50%/70%/85% 匹配覆盖率、验证集 5%/10%/20% 经验风险下选定的最大覆盖阈值，以及按受试者重抽样的区间。匹配覆盖率按接受数取整，验证阈值原样用于测试。

SensSmartTech 的 32 名受试者按 20/6/6 划分，同一组 69 条测试记录在七种条件下评测。BIDMC 下载了 53 条记录的波形和监护数值，按 46 个来源受试者分为 28/9/9。每条记录取四个 60 秒窗，要求至少 80% 的对应监护数值有效。BIDMC 的 `PULSE` 用于 PR 训练、校准和测试，`HR` 只用于 9 名留出受试者的外部 HR 测试。两者均为监护仪参考。16 轮训练使用了 104,719 个反事实状态。 [BIDMC 官方说明](https://physionet.org/content/bidmc/1.0.0/)

## 实际结果

| ECG I 低 SNR | 首次追加测量选备用 ECG | MAE | 无条件失败率 | 平均分析耗时 |
| --- | ---: | ---: | ---: | ---: |
| V2.3 二元 VOI | 0/69 | 0.524 bpm | 1.4% | 108.4 ms |
| V2.4 Conditional VOI | 12/69 | 0.304 bpm | 0.0% | 162.7 ms |
| 备用 ECG 优先 | 69/69 | 0.170 bpm | 0.0% | 194.5 ms |
| Channel EIG | 69/69 | 0.160 bpm | 0.0% | 204.8 ms |

备用 ECG II 的一步增量预测 MAE 从无互补对照的 0.253 降到 0.170（损失单位），PPG 的预测误差略升。移除任务条件的模型在该场景下 63/69 条先选备用导联，MAE 为 0.259 bpm，平均成本增至 320.5 ms。正常 ECG 上，Conditional VOI 的 MAE 为 0.348 bpm，耗时 136.7 ms，固定备用 ECG 策略为 0.297 bpm 和 99.0 ms。三模态低 SNR 下，Conditional VOI 与通道质量规则的无条件失败率都是 21.7%，平均耗时分别为 207.1 和 133.5 ms。选择性 QC 在正常信号上平均请求 2.06 次，全量 QC 请求 4 次。联合退化下两者平均都请求 4 次，选择性调度耗时略高。

三模态低 SNR、匹配覆盖 49/69 时，Conditional VOI 的选择性错误率是 4.1%，通道质量和通道 EIG 均为 6.1%，受试者级区间重叠。运行闭环校准门槛后，该条件下返回数从 60 降至 56，无条件失败率由 21.7% 升至 24.6%。验证 HR 的受试者内记录均衡 Brier 为 0.0754 → 0.0727，逐受试者留一后为 0.0754 → 0.0763。PR 留一校准有一折因其余受试者仅包含一种结局而无法拟合。

BIDMC 原始 PR 的单任务、共享任务和 PPG-only 均返回 32/36 窗，MAE 为 0.614 bpm。人工 PPG 低 SNR 条件下，质量门拒绝了全部 36 窗。外部 BIDMC 低 SNR HR 中，Conditional VOI 返回 33/36，MAE 为 0.705 bpm，与规则相同。直接使用 SensSmartTech 的校准门槛只返回 2/36，无条件失败率为 97.2%。这组对照未显示稳定的任务共享收益或校准迁移收益。

CirCor 的本地 3 对 WAV/TSV 用于核对 PCG 周期。以完整的 S1–S1 标注间期作参考，质量门通过 1 条，该条误差为 4.883 bpm。当前工具输出 PCG 周期，逐个 S1/S2 时间戳的准确度仍待评估。 [CirCor 周期报告](runs/20261112-141342-d73f4d2a/circor_pcg_period.json) · [CirCor 数据说明](https://physionet.org/content/circor-heart-sound/1.0.3/)

SensSmartTech 的公开参考由 ECG I、II、V3 派生，备用导联与参考有输入重叠。双人盲标裁决仍待提交。开发包已扩至 99 个窗口、25 人，独立测试包已扩至 36 个窗口、6 人。开发与测试受试者互斥，两个包的人工表单均为空。 [开发标注说明](runs/20261112-141756-4d0d5157/README.txt) · [测试标注说明](runs/20261112-141756-d8c1364f/README.txt) · [SensSmartTech 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12368122/)

MIMIC-III-Ext-PPG 需要受控访问权限，本轮的外部同步数据使用开放的 BIDMC。 [MIMIC-III-Ext-PPG 访问说明](https://physionet.org/content/mimic-iii-ext-ppg/1.1.0/)

## 复现与文件

26 项本地单元测试通过。使用归档权重在同步样例上实跑，HR 返回 91.12 bpm，PR 返回 91.25 bpm。 [HR 推理](runs/20261112-141342-dfd2269b/v24_analysis.json) · [PR 推理](runs/20261112-141342-7ef92d7b/v24_analysis.json)

[完整实验归档](runs/20261112-141018-5a18297d/v24_experiment.zip)包含训练时的源码、V2.4 权重、V2.3 对照权重、实验配置、数据和源码哈希、逐记录动作、评估 JSON 和匹配覆盖率图。 [评估 JSON](runs/20261112-141018-5a18297d/v24_audit.json) · [匹配覆盖率图](runs/20261112-141018-5a18297d/v24_matched_coverage.png)

```powershell
cd D:\CardioAgent
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m cardioagent.cli prepare-v24-sens runs\v24-replay\senssmarttech.jsonl --allow-published-pilot
.\.venv\Scripts\python.exe -m cardioagent.cli prepare-v24-bidmc runs\v24-replay\bidmc.jsonl --download
.\.venv\Scripts\python.exe -m cardioagent.cli audit-v24 --v23-checkpoint runs\20261112-132443-e2e9fd6c\value_network.pt
.\.venv\Scripts\python.exe -m cardioagent.cli plot-v24 '<运行目录>\v24_audit.json'
.\.venv\Scripts\python.exe -m cardioagent.cli analyze-v24 1_10-09-54 --checkpoint '<运行目录>\models.pt'
.\.venv\Scripts\python.exe -m cardioagent.cli audit-v24-pcg
.\.venv\Scripts\python.exe -m cardioagent.cli prepare-event-review --cohort development --expanded
.\.venv\Scripts\python.exe -m cardioagent.cli prepare-event-review --cohort test --expanded
```

缓存准备命令只在需要重建回放时执行。SensSmartTech 的独立参考裁决完成后，用 `--development-packet <开发包> --test-packet <测试包>` 生成新回放，再执行 `audit-v24 --sens-replay <独立参考回放>`。
