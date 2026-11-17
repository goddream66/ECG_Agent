# MC-MED AF/AFL 专用模型预标注

入口：`python -m cardioagent.ecg_agent.data.label_mcmed_af --help`。

对明确指定患者的全部本地可用 II 导联片段逐窗推理，使用独立 ECGFounder 预处理配置；完整窗口为不重叠十秒，不补齐片段末尾。只读取患者与就诊身份，不把诊断作为模型输入。

固定规则：`max(AF, AFL) > 0.8` 输出 `AF/AFL_candidate`，否则输出 `below_threshold`（未达阈值）；仅使用发布分类头的 AF 和 AFL 两个输出，不做 top-1 标签选择。阈值未经 MC-MED 临床校准，输出是待人工复核的弱标签。未达阈值不等于无 AF/AFL，也不代表正常心电图。

非有限值、至少一秒常值疑似脱落、不足十秒的尾窗和读取／预处理失败保留为 `unknown`，不能当阴性。该版本不使用模板分组或相关性规则否定模型标签。模型执行错误令运行失败，避免静默跳过故障。

`windows.jsonl` 给出逐窗 AF、AFL 分数与候选标签、片段相对秒数和脱敏绝对时间；`intervals.jsonl` 合并同片段中相邻阳性窗口，不跨阴性、未知、片段或就诊合并，边界精度为十秒。`target_scores.jsonl` 只保留 AF/AFL 两个分数供审计，没有其他类别。`records.json` 保存来源指纹；`manifest.json` 保存模型、预处理、Git 提交号与状态；`summary.json`、`patient_summary.csv` 和 `progress.json` 给出统计和进度。

输出必须使用新的 data 盘目录，不覆盖历史包；失败运行不能当完整标签使用。本入口是离线数据预标注，不是 Agent 流式评估，不将这些预测作为独立真值。

## 0.5 二值模式与双 GPU 运行

使用 `--threshold 0.5 --comparison ge --binary` 显式选择 `max(AF,AFL) >= 0.5`：输出 `AF/AFL` 和 `binary_label=1`，否则输出 `no_AF/AFL` 和 `binary_label=0`。不可用窗口仍是 `unknown` 和空值，不算阴性。这里的阴性是模型规则判断，不是人工确认。

并行运行时，每个进程通过 `CUDA_VISIBLE_DEVICES` 仅绑定一张物理卡，内部设备为 `cuda:0`。四个工作进程各分配一名患者、每卡两个进程，每进程指定 `--cpu-threads 1` 并设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`，控制主要 CPU 计算线程总预算为四；实际进程还可能有 CUDA 或系统辅助线程。不同进程写独立输出目录，全部成功且核对输入覆盖、分数与区间后才汇总结果。0.8 模式的默认行为保持不变。
