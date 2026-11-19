# 添加和使用 ECG 工具

## 新增一个工具

1. 在 schemas.py 中声明严格的参数模型（拒绝未知字段、非有限数值和非法时间范围）。
2. 在 tools/measurements.py 或独立模块实现 handler(session, args)，返回 `(payload, status, warnings)`。
3. 波形只能来自 session.window(args)，时间是绝对秒，结果明确标注 ms/mV。不要自行读取整条文件、标签、网络或模型编造的峰位置。
4. 在 tools/catalog.py 注册 ToolSpec：名称、描述、参数类、版本、来源、experimental。模型/算法版本或输出语义变化时更新版本。
5. 对未验证近似结果使用 `_proxy` 名称并设 experimental=True；无足够证据返回 insufficient/null，禁止默认 0 或固定 confidence=0.99。
6. 增加参数、因果性、异常输入、单位与实际语义测试。更换测量算法需独立标注验证。
7. 运行测试，执行 `python scripts/export_ecg_tools.py` 更新目录，再检查 Git 差异。

ToolResult 的外层由 Pydantic 验证，数值必须能严格 JSON 序列化。内部 result 是每工具的字典，由各算法及测试约束；目前还不是所有数值结果都有独立 Pydantic 结构，不应声称完整的临床语义校验。

## Python 调用

```python
from cardioagent.ecg_agent.harness import ToolContext, ToolSession, build_registry

# wave 是只包含 ECG 的 Waveform，时间原点为 0；游标是 30 秒。
session = ToolSession(ToolContext(wave, 0., 30.), max_calls=12)
response = session.call('get_rate_rhythm', {
    'lead': 'II', 'start_s': 0., 'end_s': 30.
})
# 先检查 response['status']，再使用 response['result']。
# 返回结果的修改不会影响 session.calls 中已保存的历史快照。
schemas = build_registry().function_schemas()
```

工具目录可供支持 function calling 的控制器读取。这里只导出通用函数 schema，没有接入任何线上 LLM 服务，也不保证所有厂商的 strict schema 变体直接兼容。

AFDB 复核回放会在 ToolSession 附加当前窗的缓存读取接口，从而启用 `score_ecgfounder`。工具参数为 `lead`、`start_s`、`end_s`，只接受当前已到达且有缓存的 10 秒窗口，返回未校准的 AF 和 AFL 分数。导出的目录包含这个工具，普通测量会话使用原有默认工具集。

## 命令行

在 D:/CardioAgent 下：

```powershell
.venv\Scripts\python.exe -m cardioagent.cli list-ecg-tools
.venv\Scripts\python.exe -m cardioagent.cli list-ecg-tools --experimental
.venv\Scripts\python.exe -m cardioagent.cli call-ecg-tool get_rate_rhythm --record data/external-events/mitdb/100 --lead MLII --end 30
.venv\Scripts\python.exe -m cardioagent.cli call-ecg-tool get_st_proxy --record data/external-events/mitdb/100 --lead MLII --end 30 --experimental
.venv\Scripts\python.exe -m cardioagent.cli ecg-agent --record data/external-events/mitdb/100
```

raw RR 与清洗 mask 完整保留，不按上游 MAX_LIST=60 截断。代理工具保存所有当前窗口的逐搏结果；未来接入 LLM 时应另外设置上下文摘要/分页层，不能静默裁剪或无限塞入模型。

## 上游适配清单

| ECG-Scroll 代码 | 当前实现 | 差异 |
|---|---|---|
| get_rhythm / _clean_rr | get_rate_rhythm | 原始 RR、清洗 mask、两组统计并存 |
| profile | get_feature_profile | 缺失分箱为 null，按请求窗口的峰检测统计，不取真值 |
| get_p_wave | get_p_wave_proxy | 输出 P峰-R峰距离，不叫 PR 间期 |
| get_beat_morphology | get_qrs_proxy | 输出阈值占用时间，不叫 QRS 时限，不计 PVC |
| get_rhythm + get_p_wave + get_beat_morphology | get_beat_timeline_proxy | 以 R 峰时间对齐逐拍原始 RR、P 候选时刻／振幅和 QRS 阈值占用；缺测字段为 null |
| get_st_level | get_st_proxy | 明确 R+120/R-40，未定位 J 点 |
| get_t_wave | get_t_wave_proxy | 明确固定搜索区间与幅度代理 |
| get_qt | get_qt_proxy | 保留假设与未知边界，不按正常范围丢弃异常值 |
| 全记录 _peaks 缓存 | 未移植 | 使用当前可见块，避免未来依赖 |
| classify_rhythm/beats/flag_ischemia | 未移植 | 它们读取人工真值，属于评价/Oracle |

ST/P/T/QT 幅度计算支持 mV、uV、V 并转换到 mV；未知物理单位明确报错。代理函数没有声称独立人工标注精度。


FeatureDB and template-quality integration: see [PHYSIO_TOOLS.md](PHYSIO_TOOLS.md).
