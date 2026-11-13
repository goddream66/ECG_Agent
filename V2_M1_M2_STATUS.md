# CardioAgent V2 / PhysioACE：M1 基线与 M2 事件基础

ECG 优先策略、辅助模态增量价值和独立事件验证见 [M1.5 实验记录](M1_5_STATUS.md)。

开发依据：[V2 方案](PROJECT_PLAN_V2.md)。数据来源：[SensSmartTech 1.0.0，PhysioNet](https://physionet.org/content/senssmarttech/1.0.0/)，CC BY 4.0。

## 可复现的数据与模型

- 已从本地 ZIP 导入全部 338 条记录、32 位受试者。WFDB 的 2,704 个文件及发布元数据通过包内 SHA-256 校验。原始 CSV 波形留在 ZIP 中。项目使用发布方已经放到同一时间轴的 1 kHz WFDB；ACC 文件一并导入，目前未作为模型输入。
- 划分文件为 `data/senssmarttech-1.0.0/manifest.json`：训练 20 人 / 211 条、验证 6 人 / 58 条、测试 6 人 / 69 条。V1 已经查看过的受试者 1 固定留在训练划分。训练划分指预留用途，此次没有拟合模型。
- 参考值是公开表的整条记录心率（由该记录 RR 间期中位数得到）。这是按记录评估测率的参考值。包中没有逐搏标签、脉搏足点标签、S1/S2 标签或疾病任务标签。测试划分来自同一数据集，不是跨数据集临床验证。

## M1：V1 基线和模型审计

原始记录加上分别缺失 ECG、PPG、PCG 三种条件，每种条件都在 338 条记录上运行。MAE 只在返回结果的记录上计算；应和覆盖率一起读。

| V1 固定融合 | 全量有结果 / 总数 | 全量 MAE | 测试集有结果 / 总数 | 测试集 MAE |
|---|---:|---:|---:|---:|
| 原始记录 | 334 / 338 | 0.337 bpm | 67 / 69 | 0.307 bpm |
| 缺 ECG | 329 / 338 | 0.586 bpm | 66 / 69 | 0.468 bpm |
| 缺 PPG | 330 / 338 | 1.021 bpm | 68 / 69 | 3.278 bpm |
| 缺 PCG | 331 / 338 | 0.381 bpm | 68 / 69 | 0.491 bpm |

同一组 6 条测试记录（每个测试受试者一条，均为活动后）启用了三个基础模型工具，三种 V1 模式均返回结果，单 Agent 和多 Agent 无执行报错：

| 模式 | MAE | 平均工具调用 | 平均分析耗时 |
|---|---:|---:|---:|
| 固定融合 | 0.382 bpm | 6.00 | 0.456 s |
| 本地 Qwen 单 Agent | 0.330 bpm | 6.00 | 9.733 s |
| 本地 Qwen 多 Agent | 0.337 bpm | 10.83 | 44.260 s |

这组对照用于核验流程及量级。误差差异小，样本少，不能据此判定 Agent 有稳定收益。耗时为本机分析流程用时，不含本地服务器启动。V1 依然用测率工具输出融合，预训练表征暂不进入最终数值。

| 模型 | 在 58 条验证记录上的核验 |
|---|---|
| ECGFounder | I 导联的 1,024 维表征均成功。已有 150 类头的分数没有对应标签用于核验。V1 的 50 Hz 陷波、0.67–40 Hz 通带与上游示例接近；上游滤波示例还减去中值基线，V1 未照搬。 |
| AnyPPG | 512 维表征均成功；一条重复输入逐元素一致。对同段加入 10 dB 白噪声后，余弦相似度中位数为 0.473、5% 分位数为 0.287；这只说明在该扰动下表征变化明显。 |
| CardioPHON | 3,072 维表征均成功。4 kHz、0.95 秒、FFT 512、步长 160、64 个 mel 滤波器、60–2,000 Hz 与发布配置匹配；V1 用逐片段标准化和平均池化，未复刻上游训练增强及推理全流程。SensSmartTech PCG 原生为 1 kHz，升采样不增加原生带宽。 |

权重、运行时归档和上游文件的哈希、V1 预处理和依赖版本保存在模型审计目录的 `frozen_assets.json`。三个模型均按现有权重运行，未训练任务头。

## M2：时间轴、质量与事件

`events` 可报告每个模态的采样率、样本数、公共时间范围以及同步来源。无依据时不宣称已对齐；本数据的共同网格由发布方文档和实际网格检查共同支持。系统不通过波形相关性平移信号；设备延迟未知。

每个通道分别记录非有限值、平坦/极值平台、基线与频带能量比例，以及事件间隔统计。已发布的 PPG 去均值，无法从该数据恢复传感器 DC，因此 AC/DC 留空；事件分数是启发式指标，模型校准质量也留空。

| 模态 | V2 输出 | 测试集原始记录测率 MAE |
|---|---|---:|
| ECG | WFDB XQRS + 幅值归一化 + 局部 QRS 极值，输出候选 R 峰和间期统计 | 0.299 bpm |
| PPG | 滤波峰和前导波谷候选；波谷仅作足点候选，未校准其时刻 | 2.078 bpm |
| PCG | 包络自相关周期；未确认的 S1/S2 不输出 | 3.442 bpm |

三种模态均对 69 条测试记录给出了率估计。PCG 有 2 条疑似倍频或半频错误，不能把现有周期结果当作心音事件标注。ECG 默认幅值的 XQRS 原先只在 38/338 条上返回可计算心率的事件；在验证集比较归一化后固定为 QRS 频段 P1–P99 振幅缩放，最终在 338/338 条上得到率估计。整条记录的心率对比不能核验逐搏时间误差。

## 复现

```powershell
cd D:\CardioAgent
.\.venv\Scripts\python.exe -m cardioagent.cli prepare-data 'C:\Users\34007\Downloads\senssmarttech-1.0.0.zip'
.\scripts\Run-V2-Benchmark.ps1 -Split all -MissingModalities
.\.venv\Scripts\python.exe -m cardioagent.cli events 1_10-09-54 --all-channels
.\.venv\Scripts\python.exe -m cardioagent.cli audit-models --split validation
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

启动本地 Qwen（另开终端运行 `scripts/Start-LocalModel.ps1`）后，可用 `Run-V2-Benchmark.ps1 -Split test -IncludeAgents -Limit 6` 重跑 Agent 对照。LangGraph 仍执行 V1 Agent 流程，V2 事件命令独立运行，便于控制实验变量。

本次运行文件：

- [全量基线与 V2 事件测率](runs/20261111-211510-0957240a/benchmark.json)
- [固定、单 Agent、多 Agent 同样本对照](runs/20261111-210925-19bcffd3/benchmark.json)
- [模型审计](runs/20261111-210847-b13067da/model_audit.json)及[权重冻结清单](runs/20261111-210847-b13067da/frozen_assets.json)
- [V2 事件示例（所有通道）](runs/20261111-211513-98fd2a4d/events.json)

下一步先给 ECG/PPG 事件时刻找独立标注验证，接入并评估有标注支持的 PCG S1/S2 检测。达到事件覆盖率和时间误差要求后，再训练 M3 证据图、M4 主动策略和 M5 风险校准。
