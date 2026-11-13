# ECG Agent 工程结构

当前主工程：D:/CardioAgent。D:/ECGagent/ECG 是本轮审查的上游参考，未改动。

采用通用 Agent 项目常见的分层：控制器、工具契约、注册表、执行运行时、信号算法、评价与测试分开。参考 smolagents 的具名工具/schema 设计与 LangGraph 的状态/执行分离；本轮未复制这两个框架代码，也未额外引入框架依赖。

```text
src/cardioagent/
  ecg_agent/
    agent.py                 # ECG 规则控制器：通过 ToolSession 调用工具
    tools.py                 # 旧版测量兼容层，保留已验证测量语义
    toolkit/
      contracts.py           # Pydantic 参数、结果信封、ToolSpec、ToolContext
      runtime.py             # 注册、校验、因果边界、缓存、调用预算、轨迹
      catalog.py             # 唯一工具注册入口与默认白名单
      measurements.py        # RR、扫描及上游启发式测量的适配实现
    evaluation.py            # 人工标注读入与评价；不进入工具运行时
  signal/                    # 共享底层信号检测与质量算法
  policy/                    # 旧版策略实验，保留原入口
configs/ecg_tools_v1.json     # 从注册表生成的全工具/schema 目录
scripts/export_ecg_tools.py   # 目录生成器
third_party/                 # 来源指纹与保留许可证
  ecg_scroll_source.json
  licenses/ECG-Scroll-MIT.txt
tests/test_ecg_toolkit.py
.github/workflows/tests.yml  # GitHub Actions 测试配置（本地已测试，远程尚未运行）
AGENTS.md                    # 后续代码修改规范
CONTRIBUTING.md
docs/TOOL_DEVELOPMENT.md
THIRD_PARTY_NOTICES.md
```

## 工具执行

`controller -> ToolSession.call(name, arguments) -> 参数校验 -> 可见时间窗与预算检查 -> handler -> ToolResult -> 审计轨迹`。

运行时复制当前已到达信号，丢弃游标后的样本，避免全记录预检测。缓存仅在该不可变输入快照的 session 内复用；外部返回值的修改不会污染缓存与既有轨迹。该模块为单线程 session，不宣称并发执行保证。

工具没有网络、文件搜索或标注读取能力；不接受任意 Python 代码。未知工具/参数不会偷偷回退成其他工具。

默认 4 项：assess_quality、detect_beats、get_rate_rhythm、get_feature_profile。
显式启用后另外 5 项：get_p_wave_proxy、get_qrs_proxy、get_st_proxy、get_t_wave_proxy、get_qt_proxy。

近似算法的限制在工具描述和每次结果 warnings 中保留。标准 PR/QRS/QT 波形界定尚未实现；这些代理输出不能替代临床间期测量。ECGFounder 适配器保留原位置，本轮未把未经验证的分类结果注册成诊断工具。

## 成本与兼容性

ToolSession 的 max_calls 限制所有请求（含无效/缓存请求），每次请求生成证据编号并计时。ECGAgent 的 measurement_budget 仍独立限制导联测量次数；QC 不属于测量次数，但有单独成本。代理工具的内部峰检测成本包含在调用耗时中，未伪装为免费。

agent ledger 新增 tool_calls；原有 observations/previews/trace/cost 字段保留。
早期 V2.x 研究代码和数据未迁移，以免修改已有实验解释。

## 本地 Git

已初始化 D:/CardioAgent 的本地 main 分支，并保存工具重构前的源码快照。
数据、权重、运行输出、ZIP、临时目录和 .env 不纳入版本管理。未创建 GitHub 仓库或推送。
配置文件不存储凭据。当前任务的后续源码改动通过本地提交记录；实验报告在 runs 中独立保存。
