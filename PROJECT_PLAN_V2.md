# CardioAgent V2 开发方案 — PhysioACE

以下保留本次开发依据的方案原文。当前实现、验收结果与复现命令见 [V2 开发记录](V2_STATUS.md)。

# 十三、结合你真实代码，我对当前项目的进一步判断

这里我需要特别说明：这次上传的 `runs.zip` 不仅包含运行结果，还包含了 `src/cardioagent/` 下的核心 Python 源码。我检查了其中的 `cli.py`、`workflow.py`、`signal_tools.py`、`model_adapters.py` 和 `recording.py`，因此下面的 V2 设计不是单纯根据文件树推测，而是结合了你当前的实现。

我认为你目前已经完成了一个比较重要的工程阶段：

已经具备 V2 的工程起点

- 三种信号可以通过统一接口读取。
- ECG、PPG、PCG 都有实际的测量工具。
- 三个预训练模型已经有加载和推理适配器。
- LangGraph 已经管理初步的多专家工作流。
- 专家能够读取工具结果并决定是否复核。
- 有证据 ID、模型特征文件和 Agent 运行轨迹。
- 已经存在 fixed、single、multi 三种运行模式。

但是，我发现了几个非常值得在 V2 解决的问题。

## 13.1 当前代码存在的六个关键问题

问题一：模型特征尚未真正参与最终决策。

在 `model_adapters.py` 中：

- ECGFounder 能输出 embedding 和未校准的分类分数。
- AnyPPG 输出 embedding。
- CardioPHON 输出 embedding。

但在 `workflow.py` 的 `rate_summary()` 中，最终融合只使用测量工具输出的 `rate_bpm` 和 `quality`。

也就是说：

\\[ \text{Final Decision} = f(\text{HR}\_{ECG},\text{HR}\_{PPG},\text{HR}\_{PCG}) \\]

而不是：

\\[ \text{Final Decision} = f(\text{Waveform Features}, \text{Quality}, \text{Uncertainty}, \text{Physiological Relations}) \\]

这意味着你现在实际上还没有实现一个真正的多模态表征融合方法。

这不是坏事，反而是 V2 最自然的改进起点。

问题二：当前质量评价过于简单。

`signal_tools.py` 中：

- ECG 使用峰间隔变异程度。
- PPG 使用峰间隔变异程度。
- PCG 使用包络自相关强度。

这些指标可以作为第一版启发式质量检查，但不能充分代表信号质量。

例如，规则性很强的噪声也可能产生稳定的峰间隔；PCG 的自相关也可能受到心音周期倍频和半周期歧义影响。

V2 应该将质量评价拆分为：

\\[ Q_m = (Q_m^{\mathrm{signal}}, Q_m^{\mathrm{event}}, Q_m^{\mathrm{model}}) \\]

分别表示原始信号质量、事件检测质量和模型可靠性。

问题三：当前协调者主要通过 LLM 选择复核对象，缺乏可学习的决策依据。

你目前的 `specialist()` 和 `review()` 已经实现了工具型 Agent，但 Coordinator 主要根据提示词和工具观察来选择复核对象。

这使得系统具有一定的动态性，但还不能证明它学会了最优证据选择。

V2 应该增加独立的 Policy Network：

\\[ \pi\_\theta(a_t\mid s_t) \\]

由策略网络决定下一步分析动作，而不是完全依赖 LLM 的语言判断。

问题四：当前 rate_summary 使用固定规则进行融合。

现在采用的是可用结果、5 bpm 或 5% 阈值、两票以上一致等规则。

这适合工程原型，但存在两个局限：

- 不同模态的测量误差并不相同。
- 不同模态的质量和不确定性没有进入融合权重。

V2 应该学习一个带不确定性估计的证据融合器，同时保留现有规则作为 baseline。

问题五：PCG 当前只实现了周期性心率估计，并没有真正的心音事件级理解。

你的 `measure_pcg_period()` 使用包络和自相关估计周期。

这可以得到周期性信息，但不能直接证明已经实现 S1/S2 检测，更不能直接建立 R peak → S1 → S2 → PPG foot 的事件关系。

V2 必须把 PCG 从周期估计升级到事件检测，或者明确标记暂时没有经过验证的事件输出。

问题六：当前三模态时间关系还没有被真正建模。

`Recording.shared_duration` 取的是不同模态的最短时长，默认各信号从相同的时间零点开始。

但真实数据可能存在采集起始偏差、重采样延迟、通道延迟和时间戳差异。

V2 应该引入显式的时间轴和同步质量检查，而不是仅依赖相同的 `start_s` 和 `end_s`。

# 十四、CardioAgent V2：完整研究方案

我建议正式将 V2 命名为：

# CardioAgent V2 — PhysioACE

Physiology-Constrained Active Evidence Acquisition

副标题：

Reliable Multimodal Physiological Reasoning under Missingness, Signal Degradation and Limited Analysis Budgets

我建议 V2 的设计目标分为两条并行路线：

- 工程路线： 把你现在的 CardioAgent 升级为稳定、可复现、可扩展的多模态分析系统。
- 科研路线： 实现 PhysioACE，让证据图、主动策略和风险控制成为可以独立训练和评估的算法。

两条路线不能混在一起。否则最终很难判断实验收益到底来自新的算法，还是来自更多工具调用和更复杂的 Agent 工作流。

## 14.1 V2 的总体架构

## CardioAgent V2 Architecture

Multimodal Recording Input

ECG · PPG · PCG · Channel Metadata · Time Axis

Stage 1 — Signal Processing & Synchronization

ECG

R-peak

PPG

Pulse foot

PCG

S1 / S2

统一时间轴、预处理、质量评价、事件检测

Stage 2 — Modality Expert Encoders

ECGFounder · AnyPPG · CardioPHON

输出特征、预测、质量、不确定性

V2 核心算法

Stage 3 — Physiological Evidence Graph

事件节点、时间关系、跨模态置信度、质量条件化交互

V2 核心算法

Stage 4 — Active Evidence Policy

Value of Evidence · Action Selection · Budget Management

Acquire

获取证据

Verify

复核证据

Stop

结束或拒绝

Stage 5 — Reliable Decision

Prediction · Uncertainty · Abstention · Evidence Report

LangGraph Execution Layer + LLM Reporting

负责工具执行、任务管理、轨迹记录和自然语言报告，不代替核心策略网络

我特别强调一点：LangGraph 不需要被删除，也不需要换框架。

V2 只是改变它的职责。

V1 是：

\\[ \text{LLM} \rightarrow \text{Select Tool} \rightarrow \text{Observe} \rightarrow \text{Select Tool} \\]

V2 是：

\\[ \text{Evidence State} \rightarrow \text{Learned Policy} \rightarrow \text{Action} \rightarrow \text{New Evidence} \\]

LLM 仍然存在，但不再是核心决策算法。

# 十五、V2 的核心模块详细设计

## Module 1：Signal Synchronization & Quality Assessment

这是 V2 首先需要完成的底层模块。

建议新增 `synchronization.py` 和 `quality.py`。

### 1.1 统一信号时间轴

每个模态需要有自己的时间映射：

\\[ t_m(i)=t\_{0,m}+\frac{i}{f_m}+\delta_m \\]

其中：

- \\(t\_{0,m}\\)：该模态采集起始时间。
- \\(f_m\\)：采样率。
- \\(\delta_m\\)：经过验证的时间偏移修正。

注意：不能为了让不同模态对齐，就直接通过最大相关系数强行移动波形。生理延迟本身可能是真实信息，应该区分设备同步误差与生理传播延迟。

### 1.2 多维质量评价

建议对每个模态输出：

\\[ Q_m = [q\_{\mathrm{noise}}, q\_{\mathrm{artifact}}, q\_{\mathrm{event}}, q\_{\mathrm{consistency}}] \\]

具体指标：

| 模态  | 质量指标                     |
| --- | ------------------------ |
| ECG | SNR、基线漂移、QRS 检测可信度、峰值稳定性 |
| PPG | AC/DC、灌注波动、运动伪影、脉搏检测可信度  |
| PCG | SNR、心音事件可检测性、包络质量、周期歧义   |

不建议直接将这些指标手工加权为一个质量分数。先保留多维质量信息，再由 Evidence Encoder 学习如何利用它们。

### 1.3 输出接口

建议统一为：

```
{    "modality": "ecg",    "channel": "I",    "window_s": [0.0, 30.0],    "quality": {        "snr": 18.2,        "baseline_drift": 0.04,        "artifact_score": 0.12,        "event_confidence": 0.94    },    "status": "usable"}
```

上面的数值只是接口示例，不是你现有数据的实测结果。

## Module 2：Physiological Event Detection

这是 V2 区别于 V1 的一个关键升级。

不要继续只使用 HR、PR 作为三个专家的主要交互信息。

需要逐渐把它们升级为事件级信息。

## 三个模态的事件检测目标

[A Complete Pipeline for Heart Rate Extraction from Infant ECGs | MDPI](https://images.openai.com/static-rsc-4/WnQwY4bB4SeF5uztpFssShlKWk-h3khyzwBFiPyFlXlPVdBN7O-wBb-BIVJfrjIGHynpk8so8yunIZUURML-mLM_bS-8J9RMlkD100b7ZhlUgaD7ccY_bqnRrUZxpIV3u3-Sfi-D07v94TLdy9jFofSvvOXP1okFJVVT22taYtE?purpose=inline)

ECG Event Detector

- R peak
- RR interval
- Beat confidence

[Photoplethysmography Signal Processing and Synthesis | Peter Charlton](https://images.openai.com/static-rsc-4/sCpfa9XvUQH0fKrBA2_YNGviVr_Qv9_EX8_yzw29E08yi1tcC8Trs4fk6oaHNMjckzN1atgS_jOD5jaDaUBMsadxFiTMqxxS1qcmJw_NKwvfuzfQ49ArdkbZKIQQE70cuIDAGzguuBSfzm35qBBeIPflk2lSI7cWm22JQZVGPD8?purpose=inline)

PPG Event Detector

- Pulse foot
- Systolic peak
- Pulse interval
- Pulse confidence

[Classification of Normal/Abnormal Heart Sound Recordings: the PhysioNet/Computing in Cardiology Challenge 2016](https://images.openai.com/static-rsc-4/-pvfXQ9Nj0Makgck2dtvQ2Cdqq6hQpMV6Hdu98HDlVelOJUaw4wJqS0KLGrpv3Q9dm2CrO-fCK8p6FPzb9TJUbrvQVUFcEDy788opm1I_hrt7g3RJeSxHeAdGKM564XRLJHl7ix30DUkpMKpEp7hugvV4iIX6oN4N40okyaL8Fk?purpose=inline)

PCG Event Detector

- S1
- S2
- Systolic interval
- Diastolic interval

初期不必训练三个新的大型神经网络。

我建议：

- ECG：先使用成熟的 R-peak 检测工具。
- PPG：先使用经过验证的 pulse onset 检测算法。
- PCG：优先适配 CardioPHON 或公开的心音分割模型。

每个事件都必须输出时间戳和置信度。

例如：

```
{    "event_type": "R_peak",    "timestamps": [0.62, 1.31, 2.04],    "confidence": [0.98, 0.96, 0.91],    "source": "ecg_detector_v2"}
```

这为后面的跨模态关系学习提供监督。

## Module 3：Physiological Evidence Graph

这是 V2 最重要的表征模块之一。

我建议将每个心搏事件作为一个节点。

节点表示：

\\[ v_i = (e_i,t_i,m_i,q_i,u_i) \\]

其中：

- \\(e_i\\)：事件 embedding。
- \\(t_i\\)：事件时间。
- \\(m_i\\)：所属模态。
- \\(q_i\\)：质量。
- \\(u_i\\)：事件不确定性。

例如：

单个心搏周期的证据图示意

示意关系图：不同模态的事件在时间上相互关联，但并不意味着所有事件都必须落在固定时间差内。

图中的边可以分成：

1. Intra-modal temporal edge：同一模态内部的事件关系。
2. Cross-modal physiological edge：不同模态间的生理事件关系。
3. Uncertainty edge：事件之间的时间差及关系置信度。
4. Conflict edge：检测结果之间的异常或无法解释的差异。

建议采用一个轻量级 Graph Transformer：

\\[ h_i^{(l+1)} = \operatorname{GraphAttn} (h_i^{(l)},\mathcal N(i),r\_{ij}) \\]

其中 \\(r\_{ij}\\) 包含事件类型、时间差、质量和不确定性。

最终得到：

\\[ z_t = \operatorname{Pool}(G_t) \\]

这里有一个重要的工程原则：

不要在第一版中就把所有原始波形都转换成图节点。

先使用可验证的事件检测结果建立图，再逐渐加入模型 embedding。

否则你会同时面对三个难题：事件检测误差、跨模态时间对齐误差和图模型训练误差，很难定位性能下降的原因。

## Module 4：Counterfactual Value-of-Evidence Policy

这是 V2 最核心的算法创新。

建议新建独立的 `policy.py`。

策略网络输入：

\\[ s_t = [z_t,Q_t,U_t,M_t,B_t] \\]

其中：

- \\(z_t\\)：当前证据图表示。
- \\(Q_t\\)：各模态质量。
- \\(U_t\\)：预测不确定性。
- \\(M_t\\)：当前模态可用状态。
- \\(B_t\\)：剩余分析预算。

### 动作空间设计

不要只设计三个动作 ECG、PPG、PCG。

建议分成四个维度：

| 动作类型               | 示例                     |
| ------------------ | ---------------------- |
| Modality Selection | 选择 ECG、PPG 或 PCG       |
| Window Selection   | 选择某个局部时间窗              |
| Tool Selection     | 选择事件检测、质量检测或模型推理       |
| Decision Selection | Accept、Acquire、Abstain |

例如：

```
{    "action_type": "analyze",    "modality": "ppg",    "window_s": [12.0, 17.0],    "tool": "pulse_event_detector"}
```

注意：这是逻辑动作接口。具体工具必须经过注册、参数验证和成本预算检查。

### 核心优化目标

我建议 V2 使用：

\\[ V(a\mid s_t) = R^\*(s_t) - \mathbb E_o[R^\*(s_t,o)] - \lambda C(a) \\]

但进一步增加一个生理一致性项：

\\[ V\_{\mathrm{ACE}}(a\mid s_t) = \Delta R(a) + \beta\Delta P(a) - \lambda C(a) \\]

其中：

- \\(\Delta R(a)\\)：预测风险降低量。
- \\(\Delta P(a)\\)：生理关系推断改善量。
- \\(C(a)\\)：动作成本。
- \\(\beta,\lambda\\)：需要通过验证集选择的权重。

策略网络预测：

\\[ \pi\_\theta(a\mid s_t) = \operatorname{softmax}(g\_\theta(s_t)) \\]

第一版建议采用 supervised policy learning：

\\[ \mathcal L\_{\mathrm{policy}} = -\sum_t w_t\log\pi\_\theta(a_t^\*\mid s_t) \\]

其中 \\(a_t^\*\\) 来自反事实价值排序。

需要注意，实际数据只有已经观测到的完整信号，不代表每个动作都能得到一个独立的真实观测。模拟不同分析动作时，应该使用对应工具的真实输出和计算成本，不能仅靠假设“增加一个模态就会改善多少”。对于尚未验证的反事实动作，要报告它是离线模拟结果，而不是实际采集结果。

### 预算机制

建议将预算作为策略输入，而不是只在最后检查是否超限。

\\[ B\_{t+1}=B_t-C(a_t) \\]

如果：

\\[ B_t\<C(a) \\]

则动作不可选。

这样可以确保策略在训练阶段就学会成本意识。

## Module 5：Uncertainty & Reliable Stopping

最后一个算法模块负责决定何时停止。

建议输出：

```
{    "decision": "accept",    "prediction": "estimated_cardiac_rate",    "uncertainty": 0.08,    "coverage_state": "sufficient",    "evidence_ids": ["E001", "E005", "E009"],    "remaining_budget": 3}
```

这里的 `uncertainty` 应该来自经过验证的模型校准方法，不能直接把 softmax 最大值当成真实置信度。

我建议：

- 使用 Deep Ensemble 或其他可验证的不确定性估计作为基础对照。
- 使用 Temperature Scaling 校准分类概率。
- 使用 Conformal Prediction 评估选择性预测。
- 使用 Risk-Coverage 曲线评价拒绝判断机制。

特别注意：如果只对最终模型输出进行校准，不能自动保证每一次主动选择动作之后的预测都校准。最好单独评价不同动作路径和不同预算下的校准情况。

# 十六、V2 的代码目录应该怎么改？

这是我最建议你直接采用的目录设计。

不需要推倒重写现有代码，而是增量改造。

```

D:\CardioAgent\
│
├── PROJECT_PLAN_V1.md
├── PROJECT_PLAN_V2.md
├── README.md
├── pyproject.toml
│
├── configs\
│   ├── models.yaml
│   ├── tasks.yaml
│   ├── policy.yaml
│   └── evaluation.yaml
│
├── src\
│   └── cardioagent\
│       ├── cli.py
│       ├── recording.py
│       ├── model_adapters.py
│       │
│       ├── signal\
│       │   ├── preprocessing.py
│       │   ├── synchronization.py
│       │   ├── quality.py
│       │   └── events.py
│       │
│       ├── experts\
│       │   ├── base.py
│       │   ├── ecg.py
│       │   ├── ppg.py
│       │   └── pcg.py
│       │
│       ├── evidence\
│       │   ├── schema.py
│       │   ├── graph.py
│       │   ├── encoder.py
│       │   └── fusion.py
│       │
│       ├── policy\
│       │   ├── state.py
│       │   ├── actions.py
│       │   ├── value.py
│       │   ├── network.py
│       │   ├── trainer.py
│       │   └── stopping.py
│       │
│       ├── agents\
│       │   ├── specialists.py
│       │   ├── coordinator.py
│       │   └── reporting.py
│       │
│       ├── workflow.py
│       └── schemas.py
│
├── training\
│   ├── prepare_data.py
│   ├── train_evidence.py
│   ├── train_policy.py
│   └── calibrate.py
│
├── evaluation\
│   ├── baselines\
│   ├── robustness\
│   ├── cross_dataset\
│   ├── metrics.py
│   └── run_benchmarks.py
│
├── data\
│   ├── senssmarttech\
│   ├── auricle\
│   ├── mimic\
│   ├── ptbxl\
│   └── circor\
│
├── models\
│   ├── upstream\
│   ├── checkpoints\
│   └── manifests\
│
├── scripts\
│   ├── Start-LocalModel.ps1
│   ├── Get-SampleData.ps1
│   └── Run-V2-Benchmark.ps1
│
└── runs\

```

### 现有文件的迁移对应关系

| V1 文件               | V2 处理                           |
| ------------------- | ------------------------------- |
| `recording.py`      | 保留，增加时间轴和同步元数据                  |
| `signal_tools.py`   | 拆成 preprocessing、quality、events |
| `model_adapters.py` | 保留，增加统一模型输出 Schema              |
| `workflow.py`       | 保留执行层，移出核心策略计算                  |
| `cli.py`            | 保留，增加 train、benchmark、evaluate  |
| `report.json`       | 扩展为证据图、策略轨迹和预算报告                |

不要在 V2 一开始就删除旧代码。

保留 V1 作为实验基线，并通过 `mode=v1_fixed`、`mode=v1_multi` 等配置实现版本对照。新策略以独立模式运行。

# 十七、V2 具体使用哪些数据集？

我建议不要一开始下载所有数据集。先明确每个数据集承担什么科研任务。

| 数据集                 | V2 用途                | 优先级 |
| ------------------- | -------------------- | --- |
| SensSmartTech       | ECG、PPG、PCG 同步事件关系验证 | P0  |
| Auricle             | 三模态关系学习、质量变化、心衰相关探索  | P0  |
| MIMIC-III-Ext-PPG   | ECG–PPG 配对关系和跨数据集验证  | P1  |
| BIDMC               | ECG–PPG 心率与脉搏测量验证    | P1  |
| PTB-XL              | ECG 单模态表征和任务验证       | P2  |
| PhysioNet/CinC 2016 | PCG 事件与质量模型适配        | P1  |
| CirCor DigiScope    | PCG 杂音等后续任务          | P2  |

这里有一个重要原则：

不同数据集不必承担相同的监督任务。

例如：

- MIMIC 可以训练 ECG–PPG 配对关系，但不能用于训练 ECG–PCG 关系。
- CirCor 可以训练 PCG 相关模块，但不能直接作为同步三模态证据。
- PTB-XL 可以帮助验证 ECG 表征能力，但不意味着它能够监督三模态决策。

另外，Auricle 的心衰分类实验要严格控制受试者特征混杂，不能把健康人与心衰患者的设备或人口统计差异当成疾病特征。

## 数据划分建议

我建议采用三个实验层次：

1. In-Dataset：受试者级训练、验证、测试。
2. Cross-Dataset：一个数据集训练，另一个数据集测试。
3. Cross-Condition：不同质量、不同缺失模式、不同时间偏移条件测试。

尤其不要将不同窗口随机拆分到训练集和测试集。

# 十八、V2 的 Baseline 和消融实验

这部分我建议提前设计，不要等代码写完才考虑。

## 18.1 Baseline 体系

| 类别    | 对照方法                                           |
| ----- | ---------------------------------------------- |
| 单模态   | ECG Only、PPG Only、PCG Only                     |
| 普通融合  | Early Fusion、Late Fusion、Cross-Attention       |
| 缺失模态  | Modality Dropout、Missing-Modality Transformer  |
| 质量感知  | Quality-Weighted Fusion、QA-MoE                 |
| 主动决策  | Random、Fixed Priority、Entropy-First、Greedy EIG |
| 成本感知  | Cost-Aware EIG                                 |
| Agent | Single Agent、Fixed Multi-Agent、LLM Coordinator |
| 完整方法  | PhysioACE                                      |

尽可能复用相同的三个 Backbone、数据划分和推理预算。

## 18.2 核心消融实验

我建议至少做以下七组：

| 实验 | 移除什么                    | 验证内容      |
| -- | ----------------------- | --------- |
| A  | 完整 PhysioACE            | 完整方法      |
| B  | 去掉 Evidence Graph       | 生理关系是否有贡献 |
| C  | 去掉 Quality Conditioning | 质量信息是否有贡献 |
| D  | 去掉 Counterfactual VOI   | 主动策略是否有贡献 |
| E  | 去掉 Cost Penalty         | 成本约束是否有贡献 |
| F  | 去掉 Stopping Module      | 可靠停止是否有贡献 |
| G  | 去掉 Uncertainty Input    | 不确定性是否有贡献 |

其中 B、D、E 是我最希望看到显著差异的三组。

如果去掉 Evidence Graph 后结果几乎不变，说明图结构可能只是增加复杂度。

如果去掉 Counterfactual VOI 后结果不变，说明策略学习可能没有提供真正的价值。

这些都是需要提前验证的研究假设，而不是预先认定一定成立的结论。

# 十九、V2 的研发里程碑

我不建议你现在同时开发所有模块。按照下面顺序推进，风险最低。

## CardioAgent V2 Roadmap

1. M1：V1 工程冻结与审计

   第一优先级

   工作：
   - 固定三个模型权重与预处理配置。
   - 审计测量算法和模型输出。
   - 固定 SensSmartTech 受试者划分。
   - 生成 V1 benchmark。
   交付：可复现的 V1 baseline。
2. M2：生理事件和信号质量

   工作：
   - 时间轴和同步检查。
   - ECG R peak。
   - PPG pulse foot。
   - PCG S1/S2。
   - 多维质量评估。
   交付：统一的 Event Schema。
3. M3：Physiological Evidence Graph

   工作：
   - 事件节点。
   - 跨模态关系。
   - 时间差不确定性。
   - 质量条件化图编码。
   交付：可训练的 Evidence Encoder。
4. M4：Active Evidence Policy

   工作：
   - State 和 Action 定义。
   - 反事实风险计算。
   - VOI 网络。
   - Budget Controller。
   交付：可独立训练的策略模型。
5. M5：Reliable Stopping

   工作：
   - 不确定性校准。
   - 风险覆盖评估。
   - Abstention。
   - 预算耗尽处理。
   交付：完整决策模块。
6. M6：系统集成和论文实验

   工作：
   - 接入 LangGraph。
   - 与全部 baseline 比较。
   - 跨数据集测试。
   - 消融、效率和鲁棒性实验。
   交付：CardioAgent V2 + PhysioACE 实验报告。

# 二十、我认为你现在应该怎么做？

我给你一个非常明确的建议：

现在不要立即开始写 Evidence Graph 和 Policy Network。先完成 M1，并用现有代码做一次科研级审计。

因为你现在已经有三个预训练模型，但有几个问题必须先确认：

1. ECGFounder 的 embedding 和分类输出是否真的适合你的当前输入。
2. AnyPPG 的 embedding 在不同记录、不同质量下是否稳定。
3. CardioPHON 的 4 kHz 预处理是否与发布权重的实际训练配置一致。
4. 当前 PCG 周期估计与 ECG、PPG 测量之间的误差到底有多大。
5. 现有的 Agent 动态复核究竟有没有超过固定融合。

特别是第四个问题：如果当前 PCG 测量的误差很大，那么后续把 PCG 事件关系加入 Evidence Graph 可能反而会给模型引入错误监督。

我建议先拿你现在已有的两条 SensSmartTech 记录做开发验证，再逐步扩大到全部记录，不能用两条记录得出研究结论。

## 最终确定的 V2 研究定位

# CardioAgent V2

# PhysioACE

Physiology-Constrained Active Evidence Acquisition

研究目标

在生理信号存在质量变化、模态缺失、时间偏移和跨模态冲突的情况下，学习如何以有限的计算预算主动获取最有价值的生理证据，并做出可靠决策。

核心创新

1. Quality-Conditioned Physiological Evidence Graph。
2. Counterfactual Value-of-Evidence Policy。
3. Budget-Constrained Reliable Stopping。

基础设施

- ECGFounder
- AnyPPG
- CardioPHON
- LangGraph
- Qwen3-4B

主要实验

- ECG、PPG、PCG 事件推断。
- 跨模态生理关系。
- 质量退化与模态缺失。
- 主动证据选择。
- 风险—预算权衡。
- 跨数据集泛化。

## 我还想强调一个研究战略

你现在有两个潜在方向：

- 做一个三模态医疗 Agent，强调医学推理和多专家协作。
- 做一个能够主动获取证据的生理时间序列决策算法，强调机器学习方法。

我建议以第二个为主，第一个为辅。

因为第二个方向可以从 ECG–PPG 配对数据、单模态数据以及少量 ECG–PPG–PCG 同步数据中获得监督，不需要所有训练数据都具备三个模态。

同时，算法本身也有机会扩展到其他多模态生理信号，例如 EEG、呼吸、血压和其他可穿戴传感器。

这比把研究局限在 ECG、PPG、PCG 三模态医疗 Agent 上，具有更好的方法泛化空间。

下一步最值得做的具体工作，就是把 V1 的代码和运行结果转化成一份完整的 V2 开发规格，先锁定 M1 的验收标准，再逐个模块实施。 这样既不会浪费你已经完成的代码，也能避免在还没有验证基础方法的情况下，过早投入复杂的新算法。
