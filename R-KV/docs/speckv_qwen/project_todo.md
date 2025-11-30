# SpecKV-Qwen TODO（单 Agent 用）

> 说明：使用 `- [ ]` 未完成，`- [x]` 已完成。无需填写 Agent 信息，按顺序推进即可；遇到关键算法偏差先停下并在 message 中记录。

## 一、准备与对齐
- [ ] 通读三份参考实现，整理关键算法点对照表（pruner 触发/状态、kv_budget、position/RoPE、round/窗口、prompt 模板、模型/数据入口、采样策略）。
- [ ] 明确“允许的次要差异”与“必须一致的关键要素”清单，为差异对比 MD 奠基。

## 二、脚本适配（Qwen）
- [ ] 以 `run_rkv_aime25_official_sampled8_qwen.sh` 为骨架，派生 SpecKV 版本（算法改为 SpecKV，其他保持 R-KV 风格与配置）。
- [ ] 以 `run_speckv_aime24_official_sampled8.sh` 为骨架，将模型改为 Qwen，保持 R-KV 风格，算法比对早期 Qwen 版本。
- [ ] 在实现过程中若发现与早期 Qwen/LazyEviction 版本的关键逻辑不一致，立即停止并在 message.md 记录告警。

## 三、对比与文档
- [ ] 编写差异对比 MD：SpecKV-Qwen 脚本 vs `LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh`，列明允许的次要差异与实际差异；标记任何关键不一致并警告。
- [ ] 在 status_overview.md 更新进展与风险结论。

## 四、验证（轻量）
- [ ] 代码静态检查：`python -m compileall R-KV/weian_script`（或最小子集），确认语法 OK。
- [ ] 规划 smoke 测试命令（如最小数据/单样本），待代码落地后执行；若无法跑，说明原因与缺口。

## 五、收尾
- [ ] 提交变更（含新脚本与文档），记录命令/路径；必要时附已运行的 smoke 结果。***
