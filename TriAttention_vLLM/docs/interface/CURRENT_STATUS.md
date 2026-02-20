# TriAttention_vLLM 当前状态

- 更新时间：2026-02-16
- 状态：Active
- 适用范围：vLLM 0.15.x

---

## 1. 执行摘要

项目已完成 V1 方案的大量算法与集成验证，但团队已确认进入 **V2 架构路线**：

1. 不再以“在 Attention 层后置压缩”作为主线。
2. 主线切换为“worker+runner+scheduler 非侵入式扩展”。
3. 文档系统已重构为 V2 体系，作为后续多人协作基线。

---

## 2. 已完成（可复用资产）

1. TriAttention 核心算法与配置体系（`triattention/config.py`, `triattention/compressor.py`）。
2. Triton 打分内核与相关验证资产。
3. V1 方案问题调查与根因沉淀（含 GQA 相关修复经验）。
4. 基于 vLLM 插件注册 custom attention backend 的实践路径。

---

## 3. 当前主线（V2）

1. 方案定义完成：`interface/V2_OVERVIEW.md`。
2. 技术规格主文档：`backend/ARCHITECTURE_REDESIGN.md`。
3. 决策日志已建立：`backend/DESIGN_DECISIONS.md`。
4. 开发规范与评审清单已重写：`backend/DEVELOPMENT_PRINCIPLES.md`, `backend/REVIEW_CHECKLIST.md`。
5. 新工程目录已创建：`triattention_v2/`（V2 独立开发入口）。
6. 旧版已单文件备份：`legacy_backup/v1_legacy_source_2026-02-13.tar.gz`。
7. Phase 1B 已接入执行闭环：runner 在触发后调用压缩 executor（hook 方式）并具备失败降级。
8. 默认运行模式为 plan-only；已提供实验性 KV compaction 原型（需显式开关启用）。
9. 已接入压缩事件 side-channel：runner 可将压缩执行事件回传 scheduler 侧日志。
10. scheduler 已接入 effective cache length tracker，用于对齐触发语义。
11. experimental compaction 已支持多 KV cache group 的基础映射（best-effort）。
12. 已补充 `protect_prefill=false` 裁剪路径测试与 batch>1 请求隔离测试。
13. 已新增 `tests_v2/run_smoke.py` 作为 Phase 1 无 pytest 依赖的最小回归入口。
14. 已新增 V2 专用评测 runner：`evaluation/runner/vllm_triattention_v2_runner.py`（通过 `worker_cls/scheduler_cls` 注入）。
15. 已新增 V2 quick 对齐实验配置与一键脚本：`evaluation/dispatch/configs/triattention_v2_aime24_quick.yaml`、`evaluation/scripts/run_v2_hf_alignment_quick.sh`。
16. dispatch 默认配置已切到 V2：`evaluation/dispatch/triattention_sharded_dispatch.py` 默认读取 `triattention_v2_aime24.yaml`。
17. dispatch 默认拒绝 legacy V1 runner（需显式 `--allow-legacy-v1` 才允许历史链路）。
18. 对齐对比脚本已升级为“严格判分优先 + 回退模式”：`benchmarks/reasoning/compare_results.py` 默认接入 `evaluation/eval` 的 parser/grader，并修复 `index=0` 分组冲突问题。
19. V2 评测分发已切到 HF 对齐的 run 分片语义：每个 shard 处理完整题集，但仅处理分配到的 draw/run 区间（sample8 可映射为 8 个任务）。
20. dispatch 评测脚本已复用 HF 官方 `R-KV/HuggingFace/evaluation/eval_math_multi.py`，减少评测语义漂移。
21. 新增 `per_head_selection_semantics` 开关并完成 V2 hook 落地：
   - `legacy_layer_local`：历史层内 per-head 选择（用于复现旧结果）；
   - `hf_aligned_global_per_head`：跨层聚合后的全局 per-head 选择（用于 HF 对齐）。
22. `tests_v2/run_smoke.py` 已扩展到 54 个用例，覆盖新语义开关与 hook/scheduler 分支。
23. 已补充 legacy 锚点配置：`evaluation/dispatch/configs/triattention_v2_aime24_hf_perhead_anchor_legacy.yaml`，用于复现旧语义结果。
24. 已完成“物理回收方案”技术澄清并形成执行文档：`backend/V2_RECLAIM_STRATEGY.md`（半侵入继承层路线）。
25. 已新增执行日志：`interface/V2_WORKLOG.md`，用于记录本轮实现事实/落点/验收口径，支持多人无聊天上下文接手。
26. 已落地 experimental block reclaim 原型闭环（hook 产出 reclaim 事件 -> runner 透传 -> scheduler 应用并释放 tail blocks），并补充最小单测覆盖。

---

## 4. 当前阻塞

当前阻塞主要在“原型能力向稳定能力收敛”：

1. V2 已实现实验性 KV gather/score/select/in-place compaction 闭环（hook 路径）；但当前仍是原型语义，尚未实现“物理页回收/块表重排”级别的生产闭环。
2. `protect_prefill` 与 `include_prefill_in_budget` 语义已落地，但默认策略和 KV usage 触发阈值仍需统一拍板（见 `PENDING_DECISIONS.md`）。
3. Phase 1 有本地 smoke 回归脚本（`tests_v2/run_smoke.py`），但尚未接入 CI/统一门禁流程。
4. 物理回收能力仍在开发中：已确定采用“继承层半侵入”实现，不直接修改上游 vLLM 源码文件。

---

## 4.1 2026-02-15 性能事件记录（P0）

1. 现象：AIME24 sample8 全量运行约 12 小时仍未完成，8 shard 长时间保持 active 但产出增量极小。
2. 关键证据：`evaluation/outputs/triattention_v2_aime24_hf_strict_run_20260214_230942/unattended_guard.log` 在长时间窗口内 `lines` 基本不增长。
3. 已确认主因 A（功能性缺陷）：V2 压缩事件原先只在 `sample_tokens()` 附加，vLLM 主链路通过 `execute_model()` 返回 `ModelRunnerOutput`，导致 scheduler 侧难以及时消费压缩事件、更新有效长度。
4. 已确认主因 B（性能缺陷）：experimental compaction 在 `triattention_v2/kv_compaction.py` 中存在 token 级 Python 循环（gather/scatter/zero），在长序列+多层下开销过大。
5. 已确认主因 C（实现策略）：`evaluation/runner/vllm_triattention_v2_runner.py` 固定 `enforce_eager=True`，会降低吞吐上限。
6. 本轮修复目标：
   - 修复事件回传链路（execute_model 路径）；
   - 复用既有 Triton scoring 路径；
   - 保持 gather/scatter 为 torch API，但去除核心 token 级 Python for 循环，降低循环层级；
   - 停止旧实验、清理旧输出目录后直接重跑完整实验。
7. 新增确定性修复：experimental compaction 不再写 zero tail，改为全量 permutation 重排，避免无效零 K 继续参与 softmax 污染生成。

---

## 5. 下一里程碑

1. M1：提交 V2 骨架代码（可加载、可运行、可观测）。
2. M2：完成 Phase 1 基础功能（单请求 + prefill 保护）。
3. M3：完成 Phase 2（batch>1 + prefill 可裁剪 + KV usage 触发）。

---

## 6. 风险

1. vLLM 内部接口并非全部稳定公开接口，需版本锁定与适配层。
2. 若文档不按规范维护，极易再次出现“状态冲突与旧结论污染”。
