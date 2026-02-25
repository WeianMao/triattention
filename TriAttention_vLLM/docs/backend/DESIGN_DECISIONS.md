# TriAttention_vLLM 设计决策日志（Runtime）

- 更新时间：2026-02-23
- 状态：Active
- 适用范围：vLLM 0.15.x

> 说明：本文件只保留仍然有效的关键决策和必要历史摘要。
> 详细历史请参考 `docs/archive/snapshots/2026-02-13/` 与 `docs/backend/reference/`。

---

## D-001：Runtime 主线采用非侵入式扩展
- 日期：2026-02-13
- 问题：如何避免高维护成本并支持多人并行开发。
- 决策：通过 `worker_cls + scheduler_cls` 扩展接入，不修改 vLLM 源码。
- 影响范围：整体架构与后续开发流程。
- 替代方案：直接 fork/修改 vLLM（拒绝）。
- 状态：Accepted

## D-002：压缩主流程不再放在 Attention 层
- 日期：2026-02-13
- 问题：Attention 层缺乏完整生命周期与调度信息，易产生状态问题。
- 决策：Attention 保持纯计算；压缩执行迁移至 runner，触发决策在 scheduler。
- 影响范围：TriAttentionImpl 不再作为主逻辑承载点。
- 替代方案：继续后置压缩于 forward（拒绝）。
- 状态：Accepted

## D-003：request state 唯一键是 req_id
- 日期：2026-02-13
- 问题：batch_idx/block_id 在多请求、抢占、重排场景下不稳定。
- 决策：所有持久状态必须按 req_id 管理。
- 影响范围：状态字典、日志、测试、诊断工具。
- 替代方案：使用 batch_idx 或 first_block_id（拒绝）。
- 状态：Accepted

## D-004：Runtime 分阶段推进
- 日期：2026-02-13
- 问题：需求复杂，无法一轮交付。
- 决策：
  1. Phase 1：基础功能与正确性。
  2. Phase 2：batch>1、prefill 可裁剪、显存触发。
  3. Phase 3：性能与鲁棒性优化。
- 影响范围：任务拆分、里程碑管理、验收策略。
- 状态：Accepted

## D-005：prefill 策略双模式支持，Phase 1 默认保护
- 日期：2026-02-13
- 问题：需要兼顾稳定性与后续策略实验。
- 决策：保留 `protect_prefill=true/false` 双模式能力；Phase 1 默认保护。
- 影响范围：策略配置、回归测试、实验可比性。
- 状态：Accepted（默认值仍由 `PENDING_DECISIONS.md` 最终确认）

## D-006：显存触发压缩由 Scheduler 决策
- 日期：2026-02-13
- 问题：KV usage 信息在 scheduler 侧最稳定。
- 决策：显存触发（阈值策略）在 scheduler；runner 只执行动作。
- 影响范围：触发链路、策略单测、可观测性。
- 状态：Accepted

## D-007：V1 方案定位为历史参考，不再作为主线
- 日期：2026-02-13
- 问题：V1 对项目有贡献但边界不适合后续主线扩展。
- 决策：保留 V1 资产用于参考与回归对比，主线迁移到 Runtime。
- 影响范围：文档导航、任务分配、实现优先级。
- 状态：Accepted

## D-008：Runtime 新代码落位到独立目录 `triattention_runtime/`
- 日期：2026-02-13
- 问题：继续在旧目录叠加会造成新旧实现耦合和交付歧义。
- 决策：所有 Runtime 新功能统一放在 `triattention_runtime/`，旧版目录冻结为参考。
- 影响范围：代码组织、评审边界、多人协作。
- 状态：Accepted

## D-009：Phase 1 使用 runner proxy 包装原生 model runner
- 日期：2026-02-13
- 问题：直接重写 vLLM runner 风险高且维护成本大。
- 决策：通过 worker 注入 `TriAttentionModelRunner` 代理层，先实现生命周期与触发信号消费，不改原生 forward 主路径。
- 影响范围：Phase 1A 开发效率、后续压缩执行接入路径。
- 替代方案：fork 并大改 GPUModelRunner（拒绝）。
- 状态：Accepted

## D-010：Phase 1B 使用 runner hook 执行压缩动作
- 日期：2026-02-13
- 问题：需要接入压缩执行闭环，但不宜在当前阶段重写 vLLM runner 主链路。
- 决策：在 runner proxy 内引入 `executor`，默认通过 `triattention_apply_compression` hook 委托真实压缩执行；hook 缺失时自动 no-op 降级。
- 影响范围：压缩执行闭环、失败降级策略、后续真实 KV 操作接入点。
- 替代方案：直接在 proxy 中硬编码操作 `kv_caches`（暂缓）。
- 状态：Accepted

## D-011：KV compaction 采用“默认 plan-only + 实验开关”
- 日期：2026-02-13
- 问题：直接启用 in-place KV 改写存在高风险，需要先验证方向。
- 决策：默认只输出压缩计划（plan-only），不改写底层 KV；通过 `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION=true` 才启用原型 compaction。
- 影响范围：线上稳定性、灰度测试路径、Phase 1B 验证节奏。
- 状态：Accepted

## D-012：Scheduler 触发依据使用“有效缓存长度”语义
- 日期：2026-02-13
- 问题：单纯依赖 `num_computed_tokens` 会忽略压缩后的有效缓存长度，导致触发判断漂移。
- 决策：引入 `effective_len_tracker`，由 scheduler 结合压缩事件回传维护每请求有效缓存长度。
- 影响范围：触发精度、策略稳定性、后续显存触发策略一致性。
- 状态：Accepted

## D-013：Phase 1 原型 compaction 禁止 zero-tail 写法
- 日期：2026-02-16
- 问题：在 vLLM 现有调度语义下，请求逻辑长度仍按原 `num_computed_tokens` 前进；若 compaction 后将尾部 K/V 置零，这些位置仍参与 softmax，导致分布被系统性污染。
- 决策：Phase 1 原型 compaction 改为“全量重排（kept + dropped permutation）”，不再写 zero tail。先保证语义正确，再在后续阶段引入可验证的物理长度收缩方案。
- 影响范围：`triattention_runtime/kv_compaction.py` 的共享/分头路径行为定义与回归测试预期。
- 替代方案：继续 zero-tail（拒绝）。
- 状态：Accepted

## D-014：`per_head` 语义双轨并存，默认保留 legacy
- 日期：2026-02-16
- 问题：Runtime 旧实现中的 `per_head` 为“层内独立选择”，与 HF RKV-style 的“跨层聚合后按 KV head 统一选择”存在结构差异；但旧实验（约 45% 锚点）仍需可复现。
- 决策：
  - 新增 `per_head_selection_semantics` 配置开关；
  - `legacy_layer_local` 保留历史行为用于复现；
  - `hf_aligned_global_per_head` 用于 HF 对齐实验。
- 影响范围：`triattention_runtime/hook_impl.py` 组内选择逻辑、`evaluation/runner/vllm_triattention_runtime_runner.py` 参数与 env 映射、`tests_runtime/test_hook_impl.py`。
- 状态：Accepted

## D-015：物理回收采用“半侵入继承层”路线
- 日期：2026-02-16
- 问题：仅依赖 runner hook 可完成逻辑压缩，但无法稳定完成 request 级 block 回收与 scheduler/worker 一致性维护。
- 决策：
  - 不直接修改上游 vLLM 源码文件；
  - 通过 TriAttention 继承层（scheduler/worker/runner）扩展运行时契约；
  - 先落地“回收事件 -> scheduler 应用 -> block_pool 回收”闭环，再迭代 fill-in-place 优化。
- 影响范围：`triattention_runtime/scheduler.py`、`triattention_runtime/runner.py`、`triattention_runtime/hook_impl.py` 以及相关回归测试。
- 替代方案：
  - 仅 hook 层实现回收（拒绝，状态不同步风险高）；
  - 直接 fork/修改 vLLM 主源码（暂缓）。
- 状态：Accepted

## D-016：reclaim 事件契约采用 runner side-channel 透传
- 日期：2026-02-16
- 问题：hook 侧若只做局部 block_ids 修改，scheduler 不同步会导致 worker/scheduler 视图分裂，回收无法稳定闭环。
- 决策：
  - hook 返回结构化 `block_reclaim`（group 级 `block_ids_after`）；
  - executor 保留 hook `details`；
  - runner 将 `block_reclaim` 透传到 `ModelRunnerOutput.triattention_compression_events`；
  - scheduler 在 `update_from_output()` 消费并执行 tail block 回收（受实验开关保护）。
- 影响范围：`triattention_runtime/{executor.py,hook_impl.py,runner.py,scheduler.py}` 与对应单测。
- 替代方案：仅在 worker 侧更新 `req_state.block_ids`（拒绝，调度侧状态不同步）。
- 状态：Accepted

## D-017：Runtime 进入“方案重置”阶段，采用三层分离最终架构
- 日期：2026-02-22
- 问题：当前 Runtime 虽已实现大量原型能力，但出现方案级复杂度漂移：worker 热路径 `gpu_seq_len_patch.py` 承担长期主逻辑、`hook_impl.py` 职责过载、`effective/absolute/physical` 语义事实源分散，导致对齐/性能/规范难以同时满足。
- 决策：
  1. 保留现有最终目标不变（HF 对齐优先、物理回收、非侵入式优先）；
  2. 将 Runtime 最终架构重构为“三层分离”：
     - HF 语义层（selector，输出 `KeepPlan`）
     - 布局/回收层（layout engine，输出 `PlacementPlan/ReclaimEvent`）
     - 运行时输入适配层（runtime input adapter，显式维护 `absolute_progress/effective_cache_len/effective_slot_base`）
  3. `gpu_seq_len_patch.py` 降级为过渡兼容路径，不再作为长期热路径主设计；
  4. `hook_impl.py` 后续按执行计划拆分为编排层 + 语义层 + 布局层。
- 影响范围：
  - `triattention_runtime/hook_impl.py`
  - `triattention_runtime/gpu_seq_len_patch.py`
  - `triattention_runtime/runner.py`
  - 新增 `selector/layout/input_adapter` 相关模块
  - 文档 SSOT：`interface/RUNTIME_OVERVIEW.md`, `interface/OPEN_ISSUES.md`
- 替代方案：
  - 继续在现有 patch-heavy 路径上叠修复（拒绝，复杂度持续上升）
  - 回退到 V1 / Attention 层主方案（拒绝）
- 状态：Accepted

## D-018：Runtime 主线目标模式限定为 `per_head` / `per_layer_per_head`
- 日期：2026-02-23
- 问题：重构执行顺序中曾出现“先收敛 `per_layer` strict reclaim 再扩到 `per_head`”的临时思路，但该模式不属于实际交付目标，继续作为主线阶段门槛会误导实现优先级。
- 决策：
  - 当前主线目标模式仅保留 `per_head` 与 `per_layer_per_head`；
  - `per_layer` 不作为交付目标，也不作为合理中间收敛态；
  - 任务拆分与里程碑不得再以 `per_layer` full-run 作为主线验收门槛。
- 影响范围：
  - `docs/backend/RUNTIME_FINAL_ARCHITECTURE.md`
  - `docs/interface/RUNTIME_REFACTOR_EXECUTION_PLAN_2026-02-22.md`
  - 相关任务分解、验收口径、接手沟通
- 替代方案：先以 `per_layer` 跑通再迁移（拒绝，目标偏离且可能引入无效工程工作）。
- 状态：Accepted

## D-019：布局层主路径采用低搬运 fill-hole，物理保序不作为正确性要求
- 日期：2026-02-23
- 问题：严格保序的全量前缀重写（例如完整重写 2048 前缀）搬运量过大，压缩步 IO 成本不可接受；但完全只做 page/block 顺序交换又无法覆盖所有目标模式场景。
- 决策：
  - 布局层主路径采用 slot 级 fill-hole（低搬运）；
  - 只搬运必要保留 token 回填前缀空洞；
  - 不将物理保序作为正确性条件；
  - page/block 级重排仅作为可选 fast path（表达能力足够时使用）。
- 影响范围：
  - `triattention_runtime/kv_compaction.py`
  - `triattention_runtime/layout_engine.py`
  - strict reclaim 验收口径（关注 keep 集合一致性而非物理顺序）
- 替代方案：
  - 全量保序重写前缀（拒绝，搬运量过大）
  - 仅 page 级重排（拒绝，表达能力不足以覆盖目标模式）
- 状态：Accepted

## D-020：运行时输入适配采用“压缩点更新持久状态 + decode 薄适配层”
- 日期：2026-02-23
- 问题：当前 step-local runtime override（全局 `ACTIVE_*` + monkey patch + 激活窗口）时序脆弱且 decode 热路径 CPU/Python 开销偏高；但 vLLM 原生输入准备将 `positions` 与 `seq_lens` 绑定在同一 `num_computed_tokens`，无法只靠压缩时改一个状态变量解决。
- 决策：
  - 在压缩触发时更新 request-local 持久状态（至少覆盖绝对进度/有效长度/写入基址或等价语义）；
  - decode 每步通过 thin runtime adapter 应用这些状态；
  - 允许使用 monkey patch，但 patch 仅做薄适配，不再承担复杂推导与时序敏感状态管理主逻辑；
  - 将 step-local `ACTIVE_*` override 路径降级为过渡兼容/调试路径，逐步退出主热路径。
- 影响范围：
  - `triattention_runtime/input_adapter.py`
  - `triattention_runtime/effective_overrides.py`
  - `triattention_runtime/input_patch_state.py`
  - `triattention_runtime/input_patch_vllm_backend.py`
  - `triattention_runtime/runner_output_bridge.py`
- 替代方案：
  - 完全不改 decode 输入准备逻辑，仅压缩时 hack 单一状态（拒绝，无法同时满足 positions/seq_lens/slot_mapping 分叉语义）
  - 继续沿用 step-local override 主路径（拒绝，性能与稳定性风险持续）
- 状态：Accepted

## D-021：decode 热路径最小修改原则（代码改动与 metadata 最小化）
- 日期：2026-02-23
- 问题：当前 Runtime 性能瓶颈与复杂度问题的核心来自 decode 热路径持续承载 patch-heavy 逻辑与额外状态/metadata 构造，导致 CPU 拖 GPU。
- 决策：
  - decode 热路径代码改动与新增 metadata 引入必须最小化；
  - 若某 metadata 仅服务压缩触发步或调试，不得默认进入每步 decode 主路径；
  - 运行时适配优先使用 request-local 持久状态增量表达，不在每步重建稀疏映射/字典；
  - 若少量 monkey patch 能在保持薄适配的前提下减少热路径开销与实现复杂度，允许使用。
- 影响范围：
  - `triattention_runtime/input_adapter.py`
  - `triattention_runtime/effective_overrides.py`
  - `triattention_runtime/input_patch_*`
  - `triattention_runtime/runner_output_bridge.py`
  - 后续所有 decode 热路径相关实现评审标准
- 替代方案：
  - 为通用性预先引入额外每步 metadata（拒绝）
  - 继续以 step-local patch-heavy override 作为主路径（拒绝）
- 状态：Accepted

---

## 历史摘要（保留）

### H-001：GQA 相关准确率异常曾是关键问题
- 结论：该问题已定位并修复，相关经验保留用于排障。
- 参考：`archive/snapshots/2026-02-13/interface/OPEN_ISSUES.md`

### H-002：V1 对“Attention 层后置压缩”可行性进行了充分探索
- 结论：验证了部分可行性，但不满足 Runtime 长期可维护边界。
- 参考：`archive/snapshots/2026-02-13/backend/ARCHITECTURE_REDESIGN.md`
