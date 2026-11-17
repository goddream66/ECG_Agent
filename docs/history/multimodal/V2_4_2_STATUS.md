# V2.4.2：冻结策略的机制诊断

本轮保留 V2.4.1 权重、反事实目标与既有报告。新增的是连续 QC 与统一风险评价的受控实验，不宣称新策略已优于基线。

## 实现
- `policy/qc_continuous.py`：仅训练受试者的连续核条件分布，按受试者均衡，向均衡边际分布平滑收缩，样本不足回退 V2.4.1。使用已有频带、基线、事件与分歧特征；没有把频带比例冒充真实 SNR。
- `scripts/audit_v242.py`：冻结相同权重比较 V2.4.1、连续 QC、Channel Quality、Channel EIG；统一增强特征下训练 Linear 与 16 单元 MLP；另报告原生和 Oracle 风险排序、验证/测试 AUC/Brier/ECE、风险覆盖曲线与测量成本。评分训练仅使用训练受试者；BIDMC HR 训练记录排除。
- `policy/domain_diagnostics.py` 与 `scripts/audit_v242_calibration.py`：目标域验证受试者拆成拟合与独立评估两组，阈值保持 0.2；补充 LOSO；提供训练分布距离 OOD 诊断函数（尚未接入部署拒绝逻辑）。

## 已完成验证
31 项单元测试通过，包含受试者记录重复不改变均衡 QC 分布及低支持回退测试。
BIDMC 校准诊断：5 个受试者拟合，4 个独立评估；105 个 STOP 候选的 Brier 从 0.1994 到 0.1968，候选通过率 89.5%，通过候选失败率 5.32%。这是混合场景的候选级诊断，不是低 SNR 最终部署覆盖率，不能与旧版 1/36 直接比较。
报告：`runs/v242_bidmc_calibration_diagnostic.json`。

真实 MIMIC 小样本外部测试已运行：3 个校准受试者、3 个测试受试者；原始条件冻结策略 MAE 0.527 bpm，低 SNR 和缺失 ECG 为 0.377 bpm，后两者与 Channel EIG 相同。源域校准在缺失 ECG 下拒绝全部三条，说明失效仍存在。目标校准样本没有同时覆盖成功/失败，未强行拟合。参考是发布数据的 ECG 派生 HR，不是独立人工事件标注。
报告：`runs/20261112-231609-503e4ee6/mimic_hr_aggregate.json`。

## 解释边界
完整冻结策略回放已执行完成，结果目录 `runs/v242-refinement`。当前风险曲线按各方法已返回样本的相同比例截取，是描述性曲线；跨方法严格相同总覆盖比较须按返回数对齐，不能直接比较不同 coverage 的点。尚未实现风险原生不确定性增量消融、独立阈值优化、扩大目标域样本、重新训练价值网络及 OOD 部署策略。H1/H2/H3 尚未得到证明。

运行：
```powershell
.venv\Scripts\python.exe scripts/audit_v242.py --checkpoint runs/20261112-182420-ec2a1dfa/models.pt --replays runs/v24-replay/senssmarttech.jsonl runs/v24-replay/bidmc-paired.jsonl --output runs/v242-refinement
.venv\Scripts\python.exe scripts/audit_v242_calibration.py
```


## 完整回放结果
- ECG I 低 SNR：冻结 V2.4.1 MAE 0.40797，新连续 QC 为 0.41710 bpm；两者均返回 69 条。未得到策略提升。
- 三模态低 SNR：两者均返回 60 条，MAE 均为 3.948 bpm；新版本风险 AUC：Linear 0.670、MLP 0.765、原生 0.846。这仅比较评分能力，不证明策略优势。
- 单导联低 SNR QC CRPS：连续 0.16855、边际 0.15920，仍未优于边际；三模态低 SNR 连续 0.09781、边际 0.11261。按场景聚合时 original 混合 HR/PR 与数据集，不应与历史单一组的数字直接比较。
- BIDMC 独立评估低 SNR 候选 Brier：校准前 0.13955，后 0.17816；总体略改善不能掩盖退化场景恶化。

本轮交付为 V2.4.2 的第一轮机制改进及实验，不代表审查建议全部完成。未实现项目见上文解释边界。完整报告及评分权重在 `runs/v242-refinement`。旧 V2.4.1 权重与实验报告未覆盖。
