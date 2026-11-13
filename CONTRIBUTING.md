# Contributing

先读 AGENTS.md、docs/ARCHITECTURE.md 和 docs/TOOL_DEVELOPMENT.md。

- 每个改动围绕一项明确行为，源码、测试、工具目录和文档同步更新。
- 用 `git diff` 审查改动；提交描述说明行为变化和验证结果。
- 运行 `.venv\Scripts\python.exe -m unittest discover -s tests -q`。
- 运行报告留在 runs 下，外部数据和权重留在原数据目录，不放入提交。
- 未经标注验证的测量只允许实验接口，不能通过修改函数名变成诊断工具。
- 不把上游标注读取器、训练奖励或评估答案暴露给推理控制器。
- 修改现有检测/决策逻辑时，使用 `audit-ecg-agent` 做行为回归；纯文档改动无需重跑数据实验。

GitHub Actions 定义了 CPU 测试任务，只有发布到 GitHub 后才会在远程执行。本地测试不代表远程 CI 已执行。
