# CardioAgent v0.1

CardioAgent 对同步 ECG、PPG 和 PCG 运行心搏频率测量与跨模态一致性检查。LangChain 专家调用各自的波形工具，LangGraph 执行初步分析、协调复核和报告生成。最终频率由可追溯的工具结果计算。

这是项目方案中信号分析和 Agent 协作的第一版。当前验证针对研究用样例，任务范围为频率与信号质量，临床诊断需要对应的标注和验证。

## 在本机运行

从 PowerShell 进入 `D:\CardioAgent`。本机已准备好 `.venv`、SensSmartTech 样例、三个感知模型权重和本地语言模型。

```powershell
Set-Location D:\CardioAgent
.\.venv\Scripts\python.exe -m cardioagent.cli analyze 1_10-09-54 --with-models
```

默认使用固定规则分析，适合检查模型和波形工具。启动本地 Qwen 后，可运行单 Agent 和多 Agent。保留运行服务的窗口，在另一个 PowerShell 中执行分析命令。

```powershell
# 第一个窗口
Set-Location D:\CardioAgent
.\scripts\Start-LocalModel.ps1

# 第二个窗口
Set-Location D:\CardioAgent
.\.venv\Scripts\python.exe -m cardioagent.cli analyze 1_10-09-54 --mode multi --with-models
.\.venv\Scripts\python.exe -m cardioagent.cli analyze 1_10-09-54 --mode single --with-models
```

本机使用 Qwen3-4B 的 Q4_K_M 权重，通过 llama.cpp 在 RTX 5060 Laptop GPU 上运行。服务监听 `127.0.0.1:8089`。运行结束后在服务窗口按 Ctrl+C 停止。其他设备可通过 `llama-server.exe --list-devices` 查询支持的后端后修改脚本的 `-Device` 参数。也可以用 `--base-url` 和 `--model` 指定支持工具调用的 OpenAI 兼容服务，远程服务密钥由 `CARDIOAGENT_API_KEY` 提供。

## 输入与评测

数据目录包含以 `<记录前缀>_ecg`、`<记录前缀>_ppg` 和 `<记录前缀>_pcg` 命名的 WFDB 文件对。每个输入保留其文件头中的采样率和通道信息。同一记录的三个文件共享时间原点。

```powershell
# 按需获取公开样例
.\scripts\Get-SampleData.ps1

# 验证缺失模态
.\.venv\Scripts\python.exe -m cardioagent.cli analyze 1_10-09-54 --mode fixed --omit ppg

# 用公开表中的整条记录参考心率评价本地记录
.\.venv\Scripts\python.exe -m cardioagent.cli evaluate --mode fixed --with-models --reference data\senssmarttech\Demographics.csv
```

每次执行创建独立的 `runs/<时间>-<编号>/`。其中 `report.json` 保存实际工具调用、参数、证据编号、时间窗、失败原因和最终频率。Agent 模式还保存消息与 token 统计。模型表征保存为 `.npy`，对应条目记录实际模型输入时间窗。批量评测另生成 `evaluation.json`，给出覆盖率、参考值和 MAE。

当前样例来自同一名受试者，结果用于核验接口与流程。时间窗和标签的独立评价以受试者划分。频率一致性使用显式阈值，信号质量标记来自测量算法的启发式判断。

在 `1_10-09-54` 的一次多 Agent 验证中，协调者看到 ECG 的 II 导联测量和其他模态冲突，调用 ECG 专家重新检查了其他导联，最终汇总为 91.19 bpm。公开表的整条记录参考值为 91.2 bpm。报告路径为 `runs/20261111-174712-96f27063/report.json`。

## 模型能力

| 模型 | 当前实现 |
|---|---|
| ECGFounder 单导联 | 使用 I 导联，对 10 秒片段执行预处理并严格加载检查点。保存 1,024 维表征和原有 150 类输出中的前五项分数。分数尚未在本数据上校准。 |
| AnyPPG | 按 125 Hz 输入和逐窗标准化，对 10 秒片段提取 512 维表征。脉率由 PPG 峰检测工具计算。 |
| CardioPHON | 从现有 4 kHz 编码器权重提取 3,072 维表征。按发布配置计算 log-mel 特征，采用逐片段标准化并平均池化。PCG 周期由独立包络自相关工具估计。 |

模型以 CPU 执行推理，语言模型可使用 Vulkan GPU。当前输出无需新训练任务头。AnyPPG 和 CardioPHON 的表征可供后续分类或回归任务使用，当前频率融合使用工具测量值。

## 文件与依赖

| 路径 | 用途 |
|---|---|
| `src/cardioagent/recording.py` | 同步 WFDB 读取、采样率、通道和时间窗校验 |
| `src/cardioagent/signal_tools.py` | ECG、PPG 峰检测与 PCG 周期估计 |
| `src/cardioagent/model_adapters.py` | 三个预训练模型的加载与推理 |
| `src/cardioagent/workflow.py` | 专家与协调者工具、证据记录、LangGraph 流程 |
| `src/cardioagent/cli.py` | 分析、批量评价和结果保存 |

如需重建 Python 环境，可执行以下命令。模型推理需要安装 `models` 可选依赖，权重与对应上游实现的路径记录在 `models/ASSETS.md`。

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[models]"
```

相关资料见 [项目方案](PROJECT_PLAN_V1.md)、[LangChain 工具型 Agent](https://docs.langchain.com/oss/python/langchain/agents)、[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) 和 [SensSmartTech](https://physionet.org/content/senssmarttech/1.0.0/)。

## V2 / PhysioACE 开发版

V2.4.1 统一优势与动作价值定义，增加统一风险评价、条件 QC 和分域校准对照。 [V2.4.1 结果与复现命令](V2_4_1_STATUS.md)

V2.4 增加多目标条件价值学习、选择性 QC、HR/PR 共享模型、匹配覆盖率评估和 BIDMC 外部对照。 [V2.4 结果与复现命令](V2_4_STATUS.md)

V2.3 加入任务条件化的工具选择、逐通道质量预览、离线训练的证据价值网络和验证集停止校准。 [V2.3 实现、结果与复现命令](V2_3_STATUS.md)

V2.2 提供统一动作和预算的主动 Agent 对照、分项成本、条件证据实验及人工参考入口。 [V2.2 结果与命令](V2_2_STATUS.md)

V2.1 加入通道级证据、两种复核顺序和双人盲标裁决流程。 [V2.1 记录与命令](V2_1_STATUS.md)

M1.5 已接入 ECG 优先规则和辅助模态增量审计。可运行 `.\.venv\Scripts\python.exe -m cardioagent.cli analyze-ecg-first 1_10-09-54` 查看按需调用轨迹，或运行 `audit-incremental` 复现全量实验。 [M1.5 实验记录](M1_5_STATUS.md)

[V2 方案](PROJECT_PLAN_V2.md)和 [V2 开发记录](V2_STATUS.md)列出了里程碑、基线结果和待验证事项。原有 `analyze`、`evaluate` 命令继续读取 V1 样例。以下命令使用单独导入的 SensSmartTech 1.0.0 全量 WFDB 数据，默认不需要语言模型。

```powershell
cd D:\CardioAgent
.\.venv\Scripts\python.exe -m cardioagent.cli prepare-data 'C:\Users\34007\Downloads\senssmarttech-1.0.0.zip'
.\.venv\Scripts\python.exe -m cardioagent.cli events 1_10-09-54 --all-channels
.\scripts\Run-V2-Benchmark.ps1 -Split test -MissingModalities
.\.venv\Scripts\python.exe -m cardioagent.cli audit-models --split validation
```

`events` 输出共同时间轴、独立质量指标和候选事件。PCG 仍输出周期；S1/S2 明确标记为不可用。`benchmark` 按受试者划分报告心率误差与覆盖率。要对照本地 Qwen 单 Agent 和多 Agent，先另开窗口运行 `Start-LocalModel.ps1`，再为 benchmark 脚本加 `-IncludeAgents -Limit 6`。模型审计输出权重校验与冻结配置；上述命令均不训练模型。


V2.4.2 冻结策略机制诊断：见 [V2_4_2_STATUS.md](V2_4_2_STATUS.md)，包含连续 QC、统一 Linear/MLP 风险评价、BIDMC 受试者拆分校准与真实 MIMIC 小样本结果。


当前优先开发 ECG 单智能体：见 [ECG_AGENT_V1_STATUS.md](ECG_AGENT_V1_STATUS.md)。新增 `ecg-agent` 推理及 `audit-ecg-agent` 独立心搏标注评价命令。


ECG 工具工程化入口：
- [架构与目录](docs/ARCHITECTURE.md)
- [工具开发与调用](docs/TOOL_DEVELOPMENT.md)
- [工具目录](configs/ecg_tools_v1.json)
- [开发规范](CONTRIBUTING.md)
- [第三方来源](THIRD_PARTY_NOTICES.md)

默认只开放质量、心搏、RR 和粗扫描工具；P/QRS/ST/T/QT 代理测量需显式启用 experimental。


最小事件任务接入：见 [ECG-Scroll 任务与区间评分](docs/ECG_SCROLL_MINIMAL.md)。新增 prepare-ecg-scroll 和 score-ecg-intervals，任务答案与推理清单分离。
