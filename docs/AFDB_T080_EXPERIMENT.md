# AFDB 全量 0.8 阈值对照实验

范围：现有 ECG-Scroll AFDB 全量 23 条完整记录，ECG1/ECG2 分别作为主通道；不包含 LTAFDB。已用于开发观察，结果属于探索性复测。

使用 `configs/ecg/policy_t080.json`，阈值为 `max(AF,AFL) >= 0.8`，其余生命周期和复核参数保持原配置。沿用 AFDB 已有 NeuroECG 离线十秒缓存，不混入 MC-MED 专用预处理。Agent 仅在已完成窗口上按实际工具请求执行 ECGFounder。

真实 GPU 运行 `cardioagent.ecg_agent.experiments.runner --mode live`，既有入口同时运行 event-adaptive 和 legacy-dual。两主通道使用独立输出目录和运行清单，可分别绑定一张 GPU，每张卡只运行一个进程。记录级检查点按现有哈希规则保存。

对照包括同阈值固定双通道、同阈值单通道单窗，以及固定的 0.6/0.4 滞回。简单基线可按时间顺序回放已核验缓存分数；这不产生真实 GPU 计算成本，不能将其耗时与实际模型执行耗时直接比较。固定双通道和 Agent 使用真实按需推理。

评价使用独立原始节律标注生成的有效范围；AFIB+AFL 并集作为参考。保留原预测顺序的一对一贪心匹配和 IoU>=0.5。报告事件级 TP/FP/FN、微/宏 F1、时间精确率/召回率、非 AF 报警分钟、短事件召回、碎片、负荷误差、确认延迟与实际调用成本。配对记录 Bootstrap 2000 次，种子 20261114；不根据结果改选参数。

运行目录必须在服务器 `/data/1shared/gaojiajun/ECGagent/runs` 下新建，既有 0.5 实验和 MC-MED 标注结果保持不变。最终附带原始预测、独立评价、配置和来源指纹。
