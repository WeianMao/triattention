# TriAttention_vLLM 当前状态

- 更新时间：2026-02-23
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
27. 已完成一轮方案级复盘（2026-02-22）：确认当前 V2 主要问题已从“局部实现 bug”扩展为“方案边界偏航”（worker 热路径 patch 过重、hook 职责过载、长度语义事实源分散）。
28. 已新增方案重置文档与执行计划：
   - `backend/V2_FINAL_ARCHITECTURE.md`
   - `interface/V2_REFACTOR_EXECUTION_PLAN_2026-02-22.md`
29. 已进入重构实施中段（T2/T3 持续推进）：
   - `hook_impl.py` 已进一步瘦身（group 级选择/布局执行逻辑外移）；
   - `runner.py` 的 worker reclaim 同步逻辑已抽离到 `triattention_v2/worker_reclaim_sync.py`；
   - `gpu_seq_len_patch.py` 的低层 patch ops 已迁移至 `triattention_v2/input_patch_ops.py`（patch 正在降级为兼容/路由层）。
30. T2/T3 继续推进（后续增量）：
   - `hook_impl.py` 运行时口径/门禁逻辑已迁移到 `triattention_v2/hook_runtime_context.py`；
   - `hook_impl.py` group 循环编排与结果拼装已迁移到 `triattention_v2/hook_group_pipeline.py`；
   - `hook_impl.py` 前置校验（request/runtime state、KV cache/block_ids 校验）已迁移到 `triattention_v2/hook_preflight.py`；
   - `gpu_seq_len_patch.py` 的活动状态、vLLM patch 闭包与 backend facade 已拆分为：
     - `triattention_v2/input_patch_state.py`
     - `triattention_v2/input_patch_vllm_backend.py`
     - `triattention_v2/input_patch_backend.py`
   - `gpu_seq_len_patch.py` 的 patch 安装器已迁移到 `triattention_v2/input_patch_installer.py`；
   - `gpu_seq_len_patch.py` 已完成兼容层化（大部分 helper 变为别名转发到 backend/ops 模块）；
   - `runner.py` 的压缩执行块已迁移到 `triattention_v2/runner_compression_actions.py`；
   - `runner.py` 的生命周期/信号摄取逻辑已迁移到 `triattention_v2/runner_state_updates.py`；
   - `runner.py` 的 base execute + output side-channel 挂载逻辑已迁移到 `triattention_v2/runner_output_bridge.py`；
   - 当前 `hook_impl.py` 已缩减至 ~180 行，`runner.py` 已缩减至 ~140 行，`gpu_seq_len_patch.py` 已缩减至 ~50 行。
31. 重构稳定性复验（2026-02-22）：
   - `tests_v2` 全量通过（`128 passed`）；
   - `tests_v2/run_smoke.py` 通过（`smoke passed: 98 tests`）；
   - 运行时主路径已不再直接依赖 `gpu_seq_len_patch.py`（其已退化为兼容入口层）。
32. 方案调整共识（2026-02-23）已文档化：
   - 新增 `interface/V2_SCHEME_ADJUSTMENT_2026-02-23.md`；
   - 明确主线目标模式仅 `per_head` / `per_layer_per_head`（`per_layer` 非目标/非中间态）；
   - 明确压缩主线为低搬运 fill-hole，运行时走“压缩点更新持久状态 + decode 薄适配”。

---

## 4. 当前阻塞

当前阻塞主要在“原型能力向稳定能力收敛”：

1. V2 已实现实验性 KV gather/score/select/in-place compaction 闭环（hook 路径）；但当前仍是原型语义，尚未实现“物理页回收/块表重排”级别的生产闭环。
2. `protect_prefill` 与 `include_prefill_in_budget` 语义已落地；“按 KV usage/显存压力触发压缩”仍属于后续能力，当前重构阶段只要求保留清晰接入点。
3. Phase 1 有本地 smoke 回归脚本（`tests_v2/run_smoke.py`），已支持自动跳过需要 pytest fixture 的测试函数并恢复可用，但尚未接入 CI/统一门禁流程。
4. 物理回收能力仍在开发中：已确定采用“继承层半侵入”实现，不直接修改上游 vLLM 源码文件。
5. 当前实现路径存在方案级复杂度漂移，需先按重构计划拆分职责并收敛热路径，再继续在原型上叠加功能修复。
6. 当前阶段重点已从纯 T2/T3 模块拆分推进到“Runtime Adapter 简化 + 状态一致性验证”收敛阶段（见 `interface/V2_SCHEME_ADJUSTMENT_2026-02-23.md`）。

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
3. M3：完成 Phase 2（batch>1 + prefill 可裁剪 + KV usage/显存压力触发）。

---

## 6. 风险

1. vLLM 内部接口并非全部稳定公开接口，需版本锁定与适配层。
2. 若文档不按规范维护，极易再次出现“状态冲突与旧结论污染”。
