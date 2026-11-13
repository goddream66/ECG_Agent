# ECG 工具工程化交付（2026-11-13）

主工程：D:/CardioAgent。上游参考：D:/ECGagent/ECG，未修改上游目录。

## 完成内容

- 本地 Git main 分支与重构前快照 `84c47b6`；完成后的源码也存为本地提交，未推送。
- 控制器与工具分离；新增 contracts/runtime/catalog/measurements 四层。
- 统一 Pydantic 参数、结果信封、版本、证据编号、可见时间、耗时、结构化错误和调用日志。
- 当前 ECGAgent 已通过该运行时调用质量与心搏工具，原有函数作为兼容后端保留。
- 导出函数调用 JSON schema，可供后续 LLM 控制器使用。
- 原始 RR、清洗 mask、清洗前后统计与粗扫描；默认四项工具。
- 适配 ECG-Scroll 的五项 P/QRS/ST/T/QT 代理测量，明确实验名称并默认关闭，支持物理单位转换。
- 未移植整条记录峰缓存及人工标注诊断函数；没有加入固定 0.99 置信度。
- 保留上游 MIT 许可证、来源 commit 与本地源文件 SHA256；没有复制原始数据或模型权重。
- 增加 AGENTS.md、CONTRIBUTING、架构说明、工具开发说明、生成目录、变更日志和 GitHub Actions 配置。

## 验证

- 49 项本地测试通过；依赖检查通过。
- 六条 MIT-BIH、13,096 个参考心搏的回归评价：三种策略的心搏指标、条件速率 MAE、覆盖率和工具次数与旧版逐项完全一致。
- 回归报告：`runs/20261113-200701-60043f62/ecg_agent_audit.json`。
- 九工具真实片段调用结果：`runs/20261113-200701-60043f62/ecg_tool_smoke.json`；八项 ok，QT 代理 insufficient。该检查只验证可调用与有限 JSON 输出，不证明代理测量精度。
- 目录命令与单工具推理命令已运行。
- 自动测试工作流尚未在 GitHub 上执行，不宣称远程 CI 已通过。

## 使用入口

参见 docs/ARCHITECTURE.md 和 docs/TOOL_DEVELOPMENT.md。

```powershell
cd D:\CardioAgent
.venv\Scripts\python.exe -m cardioagent.cli list-ecg-tools
.venv\Scripts\python.exe -m cardioagent.cli list-ecg-tools --experimental
.venv\Scripts\python.exe -m cardioagent.cli call-ecg-tool get_rate_rhythm --record data/external-events/mitdb/100 --lead MLII --end 30
```

当前不宣称代理 PR/QRS/QT 的临床精度；没有新增 LLM 或后训练。真正波形边界测量仍需后续实现和独立标注验证。
