# OPEN ISSUES（Runtime）

- 更新时间：2026-02-23
- 状态：Active
- 适用范围：vLLM 0.15.x

---

## [P0] 1. Runtime 触发链路需要从原型走向稳定版
- 背景：当前实现要求由 scheduler 侧决定何时压缩，runner 侧执行压缩。
- 影响：没有该链路就无法验证“显存触发压缩”主能力。
- 现状证据：`triattention_runtime/scheduler.py` 已挂载 `triattention_signals` 并接入 effective len tracker；`triattention_runtime/runner.py` 已消费信号并调用 executor。
- 下一步：将当前 experimental compaction 从原型升级为稳定实现，并覆盖多层/多组 KV cache 场景。
- 验收标准：可在日志中观测到“达到阈值 -> 触发压缩 -> 执行 hook -> 压缩执行完成”的稳定流程，且行为可回归验证。
- 状态：In Progress

## [P0] 1.0 Runtime 当前存在方案级复杂度偏航（需先重构边界）
- 背景：当前实现已跑通大量能力，但当前主要矛盾已不再是单点实现缺陷，而是方案边界偏航导致的复杂度累积。
- 影响：
  1. HF 对齐、性能、规范三者难以同时收敛；
  2. 修复一个局部问题时容易引入新的语义漂移或热路径开销；
  3. 新同事接手成本高，开发效率下降。
- 现状证据：
  - `triattention_runtime/gpu_seq_len_patch.py` 已承载 worker 热路径主逻辑（decode 每步介入 `seq_lens/slot_mapping` 修正）；
  - `triattention_runtime/hook_impl.py` 体量与职责明显过载（HF 语义、compaction、reclaim、guard、debug 同处一层）；
  - `effective length / absolute progress / physical block state` 的事实源分散在 scheduler/runner/worker patch/hook。
  - HF selector 仍存在对 base_runner/active runtime state 的隐式上下文依赖（ambient context），会增加 HF 偏差定位成本（属架构收敛中的未完成项）。
- 下一步：
  1. 按 `docs/backend/RUNTIME_FINAL_ARCHITECTURE.md` 执行“三层分离”重构（语义层 / 布局回收层 / 运行时输入适配层）；
  2. 按 `docs/interface/RUNTIME_REFACTOR_EXECUTION_PLAN_2026-02-22.md` 与 `docs/interface/RUNTIME_SCHEME_ADJUSTMENT_2026-02-23.md` 进行模块化重构，而不是继续在现方案上叠 patch；
  3. 将后续问题按“方案偏航类 / 实现 bug 类 / 实验验证类”重新归类。
- 2026-02-23 方案调整补充（已达成共识）：
  - 主线目标模式仅 `per_head` / `per_layer_per_head`，`per_layer` 不作为中间收敛态；
  - 布局层主路径收敛为低搬运 fill-hole（不以物理保序为正确性要求）；
  - runtime adapter 目标形态为“压缩点更新持久状态 + decode 薄适配层（可 patch，但必须 thin）”。
- 2026-02-22 当前进展（已开始代码重构）：
  - T1 已完成：`plan_models.py` 落地，`hook_impl` 内部开始使用结构化计划对象；
  - T2 已开始：HF selector 主实现迁出到 `selector_hf.py`，`hook_impl` 运行路径已切换；
  - T2/T3 起步：新增 `layout_engine.py` 与 `input_adapter.py`，开始收敛 `hook_impl` 与 `runner` 职责。
- 2026-02-22 当前进展（后续增量）：
  - T2 深化：新增 `selection_planner.py`、`hook_runtime_context.py`、`hook_group_pipeline.py`；
  - T3 深化：新增 `worker_reclaim_sync.py`、`input_patch_state.py`、`input_patch_vllm_backend.py`、`input_patch_backend.py`；
  - T2 继续推进：新增 `hook_preflight.py`，将 request/runtime_state 与 KV cache/block_ids 前置校验迁出 `hook_impl.py`；
  - T3 继续推进：新增 `input_patch_installer.py`，vLLM patch 安装器从 `gpu_seq_len_patch.py` 迁出，并将 `gpu_seq_len_patch.py` 大部分 helper 改为兼容别名转发；
  - `runner.py` 的压缩执行与生命周期/信号摄取逻辑已拆分到 `runner_compression_actions.py` / `runner_state_updates.py`；
  - `hook_impl.py` 已缩减至约 180 行，`runner.py` 已缩减至约 160 行，`gpu_seq_len_patch.py` 已缩减至约 50 行，职责明显收敛。
- 验收标准：
  1. decode 热路径不再依赖 patch-heavy Python 主逻辑；
  2. HF selector 语义可独立验证；
  3. strict reclaim full-run 的性能与正确性问题可在分层边界内定位。
- 状态：Open

## [P0] 1.1 Runtime 与 HF 等价性验证仍未完成
- 背景：项目终极目标是与 HF SpeckV 等价。
- 现状证据：已具备 Runtime quick 评测入口（`evaluation/runner/vllm_triattention_runtime_runner.py` + quick dispatch 配置），可快速产出对比样本。
- 差距：Runtime 当前压缩执行仍是原型 compaction，尚未接入完整 SpeckV score/topk 语义。
- 2026-02-22 补充（已修复一项确定性语义偏差）：
  - `triattention_runtime/hook_impl.py` 的 `per_layer` paged streaming 选点路径在 `sparse_normalize_scores=True` 时曾错误跳过归一化；
  - 该假设仅在“每头独立 top-k”时近似成立，但对跨 head 聚合（尤其 `max`）不成立，会造成 HF 对齐偏差；
  - 已改为两遍 chunk 统计（mean/std）+ 分块归一化，避免物化全序列分数。
- 2026-02-22 补充（已修复一项高风险 pre-step 语义错误）：
  - `triattention_runtime/hook_impl.py` 在 `execute_model()` 前执行压缩时，曾使用 `signal.estimated_cache_len`（= pre-step effective len + scheduled_tokens）直接作为 gather/score/select/compaction 的 `total_tokens`；
  - 这会把“本轮尚未写入 KV 的 scheduled token”错误计入压缩语义（decode 常见为 +1），同时把 `round_start` 也带偏；
  - 已修复为：
    1) 压缩执行使用 `pre-step effective len`（由 `estimated_cache_len - scheduled_tokens` 反推）；
    2) `round_start` 使用绝对解码进度 `req_state.num_computed_tokens`（对齐 HF `absolute_position` 口径）；
    3) 本地 re-trigger gate 仍使用 `estimated_cache_len`（保持调度触发语义）。
- 下一步：先跑 quick 对齐实验确认偏差规模，再决定优先补齐哪些语义差距。
- 验收标准：在固定小样本上得到可复现实验报告（accuracy/token match/长度差异）。
- 状态：Open

## [P0] 1.2 Runtime full-run 吞吐异常偏低（长时间低产出）
- 背景：当前 full run 在 8 shard 并发下运行近 12 小时仍未完成，明显偏离预期窗口。
- 影响：阻塞 HF 对齐验证节奏，且无法作为稳定开发基线。
- 现状证据：
  - `unattended_guard.log` 长时间 `active=8` 但 `lines` 近乎不增长；
  - GPU 利用率多卡长期处于低中位区间（非持续高负载）。
- 已定位风险点：
  - 压缩事件回传链路在 `execute_model` 路径存在缺口（已进入修复）；
  - experimental compaction 的 token 级 Python 循环开销过高；
  - `enforce_eager=True` 影响吞吐上限。
- 下一步：
  - 在继续实验前，优先执行方案级重构（见 `1.0`）以消除 worker 热路径 patch 主逻辑；
  - 在新架构下重新评估事件回传/有效长度同步与 compaction 热路径开销；
  - 重跑全量实验验证吞吐是否恢复。
- 2026-02-22 进展补充：
  - T2/T3 架构重构阶段已形成稳定代码基线（`tests_runtime` 全量 + `run_smoke.py` 均通过）；
  - 当前仍有一轮较早启动的 full-run 在运行（旧代码启动），其结果不可直接用于评估本轮重构后的性能/正确性。
- 验收标准：在相同配置下，全量运行时长回到可接受区间，且日志显示稳定产出增长。
- 状态：In Progress

## [P0] 1.3 experimental compaction 语义错误会污染注意力分布
- 背景：当前 Runtime 原型在逻辑长度不缩短的前提下执行 in-place compaction。
- 问题：旧实现将尾部 KV 置零；这些“零 K”仍参与 softmax，导致分母被大量无效项放大，生成质量显著劣化（可表现为乱码/重复/异常长输出）。
- 已确认修复方向：改为“全量 permutation（kept + dropped）”而非“kept + zero tail”，先保证语义与 FullKV 等价，再继续推进真正的物理长度收缩方案。
- 下一步：完成端到端对齐回归（itercheck + full run）验证该修复是否消除异常输出。
- 验收标准：修复后不再出现大规模乱码/重复退化；HF 对齐指标显著回升。
- 状态：In Progress

## [P0] 1.4 `per_head` 语义与 HF RKV-style 存在结构差异
- 背景：HF RKV-style 的 `per_head` 语义是“跨层聚合后按 KV head 独立选择，再将同一组 per-head 索引应用到各层”。
- 问题：旧实现在 hook 内按“每层独立 per-head 选择”执行，行为更接近 `per_layer_per_head`，会导致对齐实验存在系统性偏差。
- 修复：新增 `per_head_selection_semantics` 开关：
  - `legacy_layer_local`：保留旧行为用于历史复现；
  - `hf_aligned_global_per_head`：按组跨层聚合后统一 per-head 选择（当前对齐模式）。
- 当前状态：
  - 已落地“attention-head 打分 -> 组内（KV group）max -> 跨层 mean -> per-head topk”路径；
  - 已补齐头维适配：当 stats 头数与 runtime KV 头数不一致时，不再隐式使用前几个头；
  - 代码与单测已落地（`triattention_runtime/hook_impl.py`、`tests_runtime/test_hook_impl.py`），待全量 AIME24 sample8 复跑验证指标。
- 验收标准：在同一参数集下，Runtime 与 HF 的差异收敛到可解释范围，且 legacy 结果可复现。
- 状态：In Progress

## [P0] 1.5 物理回收能力未闭环（仅逻辑压缩）
- 背景：当前 Runtime experimental compaction 以逻辑重排为主，尚未稳定回收 request tail blocks 到 free pool。
- 影响：长跑显存/吞吐行为与“预算区+overflow”目标策略存在偏差，无法作为最终实现形态。
- 现状证据：
  - vLLM 默认契约为 append 路径：`vllm/v1/core/sched/output.py:116`、`vllm/v1/worker/gpu_model_runner.py:1037`。
  - `KVCacheManager` 公开 API 无 request 级局部 shrink/free：`vllm/v1/core/kv_cache_manager.py:378`。
  - 方案与分阶段边界已沉淀：`docs/backend/RUNTIME_RECLAIM_STRATEGY.md`。
- 下一步：
  - 已落地“半侵入继承层”回收原型闭环（runner 事件 + scheduler 应用 + block_pool 回收），下一步做端到端压测验证；
  - 以实验开关保护，默认保持主线行为不变；
  - 执行过程同步记录到 `interface/CURRENT_STATUS.md` 与 `interface/IMPLEMENTATION_OVERVIEW.md`（必要时补 `backend/DESIGN_DECISIONS.md`）；
  - 闭环稳定后再推进更严格 fill-in-place 页整理。
- 验收标准：压缩触发后可观测到回收事件并实际归还 tail blocks；默认关开时行为保持兼容。
- 状态：In Progress

## [P0] 1.6 fill-in-place 在 physical reclaim 模式下会打乱保留 KV 的时序
- 背景：当前实现在 `enable_experimental_block_reclaim=true` 时，会走 `preserve_dropped_tokens=False` 的 fill-in-place 快路径，再截断 tail blocks。
- 问题：当前 fill-in-place 仅保证“保留集合正确”，未保证“保留 token 顺序与 keep_indices 一致”。
- 代码证据：
  - `triattention_runtime/kv_compaction.py` 中 `compact_request_kv_in_place(..., preserve_dropped_tokens=False)` 逻辑：
    - 保留前缀内已存在 token 原地不动；
    - 仅把 tail survivor 回填到空槽；
    - 该过程会改变保留 token 的相对顺序。
  - `compact_request_kv_in_place_per_head(..., preserve_dropped_tokens=False)` 也存在同类问题（按 head 逐行回填，顺序不稳定）。
- 影响：KV 时间顺序被破坏后，后续 attention 的“位置-内容对应关系”失真，可能出现异常长输出、重复模式、准确率断崖下跌。
- 现状证据：
  - 低分运行（`acc=10.8`）样本显示大量 `output_tokens` 接近 `max_length`；
  - 历史较高分运行（`acc=45.4`）记录里缺少 reclaim strict 字段，语义路径不同。
- 最新排查结论（2026-02-22 更新）：
  - 已验证“仅修 recent-window 语义（改为 R 集合，不依赖逻辑尾部顺序）”仍不足以消除问题；
  - 在恢复低搬运 fill-hole 后，观测到 `recent_unabsorbed` 已稳定在合理值（首轮后约 134，符合调度/块粒度偏差），但单样本 strict reclaim 仍持续 runaway（压缩 step 持续上升）；
  - 新增证据：`tests_runtime/test_kv_compaction.py` 随机对照已验证低搬运 fill-hole 与全排列路径在“前缀有效区保留集合”上等价（shared/per-head 均通过），因此“compaction 前缀写坏数据”嫌疑下降；
  - 当前更可疑根因转向“长度语义混用”：vLLM worker GPU 输入准备将 `num_computed_tokens` 同时用于绝对位置与 `seq_lens`，而 Runtime 压缩只在 scheduler/hook 层维护 effective length。
- 当前判断：
  - 不能把问题简单归因于“乱序不安全”；更可能是模块边界内仍存在长度/状态不同步 bug；
  - 已在 Runtime 增加 worker `seq_lens` override 补丁（positions 保持绝对位置，seq_lens 使用 TriAttention effective cache len）作为 P0 修复方向，待端到端验证；
  - “保序重写前缀”仍不可作为最终方案（搬运量过大）。
- 验收标准：strict reclaim 在低搬运方案下不再出现 runaway，且 full-run 精度恢复到历史可接受区间。
- 状态：Open（P0，未解决）

## [P0] 1.7 worker 侧 `num_computed_tokens` 双重语义（绝对位置 + attention 长度）与压缩有效长度冲突
- 背景：vLLM v1 GPU 输入准备 `prepare_pos_seq_lens()` 用同一个 `num_computed_tokens` 同时构造：
  - `positions`（绝对位置，用于 RoPE / position ids）
  - `seq_lens`（attention 上下文长度）
- 问题：TriAttention Runtime 压缩后需要“绝对位置继续单调增长”，同时“attention 长度按压缩后 effective KV length 计算”；若不拆开，容易出现：
  - attention 仍按过长上下文（性能下降、可能读取无效尾部）；
  - 或错误地回退绝对位置（结果语义错误）。
- 代码证据：
  - `vllm/vllm/v1/worker/gpu/input_batch.py:_prepare_pos_seq_lens_kernel`
    - `seq_len = num_computed_tokens + query_len`
    - `pos = num_computed_tokens + block`
  - `vllm/vllm/v1/worker/gpu/model_runner.py:587-595` 同步调用该路径。
- 当前修复方向（已落地代码，作为过渡方案）：
  - 新增 `triattention_runtime/gpu_seq_len_patch.py` 修正 `seq_lens/slot_mapping` 口径；
  - `triattention_runtime/runner.py` 在 `execute_model` 前为当前 step 提供 effective 语义覆盖；
  - `triattention_runtime/worker.py` 在注入 runner 时安装补丁。
- 新判断（2026-02-22）：
  - 该补丁解决了部分确定性语义错误，但已演变为 decode 热路径长期主逻辑，成为性能与维护复杂度的主要来源之一；
  - 后续需按 `1.0` 转向 Runtime Input Adapter 方案，将 patch 降级为兼容路径。
- 2026-02-23 执行方向补充：
  - 不再以 step-local `ACTIVE_*` override 作为长期主路径；
  - 优先验证“压缩点更新持久状态 + decode 薄适配”是否可替代当前 override 链；
  - 在完全不改 decode 输入准备逻辑不可行的前提下，允许保留薄 patch。
  - decode 热路径新增 metadata 需最小化；能用持久状态增量表达的语义不再每步重建。
- 风险/边界：
  - 属于 worker 级半侵入补丁（但仍限定在 Runtime 注入路径内）；
  - 需要端到端验证不会破坏 prefill/spec decode 等未覆盖路径。
- 验收标准：
  1. 压缩激活后 GPU 利用率不再出现异常低占用/长时间等待 CPU；
  2. strict reclaim 单样本不再 runaway；
  3. 全量回归指标恢复到历史合理区间；
  4. decode 每步不再有重型 metadata/override 构造。
- 状态：In Progress（P0）

## [P0] 2. 请求级状态生命周期尚未在 Runtime 代码闭环
- 背景：V1 历史问题证明 request state 处理是高风险点。
- 影响：状态污染会直接导致压缩策略错误或结果漂移。
- 现状证据：`triattention_runtime/state.py` + `triattention_runtime/runner.py` 已接入生命周期骨架，覆盖 new/finished/preempt/resume。
- 下一步：补齐与真实压缩执行联动后的状态一致性校验。
- 验收标准：长跑测试无跨请求状态污染；请求结束后状态可回收。
- 状态：In Progress

## [P0] 3. Phase 1 回归门禁缺失
- 背景：多人并行开发需要固定最小回归集。
- 影响：修改后可能破坏核心路径且无人感知。
- 现状证据：`tests_runtime/run_smoke.py` 已恢复可用，支持自动跳过需要 pytest fixture 的测试函数（当前可运行并输出 `smoke passed`）。
- 下一步：将该脚本接入 CI 或统一 pre-merge 流程，形成强制门禁。
- 验收标准：PR 可自动/半自动执行并给出通过结论。
- 状态：In Progress

## [P1] 4. prefill 裁剪策略未落地
- 背景：Runtime 支持 `protect_prefill=false`，但 Phase 1 默认先保护。
- 影响：影响后续压缩率与策略实验。
- 现状证据：`triattention_runtime/kv_compaction.py` 已实现裁剪语义，`tests_runtime/test_kv_compaction.py` 与 `tests_runtime/test_hook_impl.py` 已覆盖关键路径。
- 下一步：在真实 vLLM 端到端链路中验证该模式（不仅是单元/冒烟）。
- 验收标准：两种 prefill 模式可配置切换且行为可验证。
- 状态：In Progress

## [P1] 4.1 `scheduled_tokens > 1` 场景下的 prefill 兼容风险
- 背景：当前 Runtime 触发链路包含 scheduler 估算长度 + runner 前置执行压缩的路径；在 chunked prefill 或单轮执行多 token 场景中，估算口径与真实执行口径可能出现偏差。
- 影响：可能导致压缩触发步与 HF strict 语义不一致，进而在 prefill 边界下出现“触发延后一轮/选点集合不同”的行为偏差；在极端场景可能放大为容量控制不稳定。
- 现状证据：`triattention_runtime/scheduler.py` 使用 `estimated_cache_len = effective_base_len + scheduled_tokens`，`triattention_runtime/hook_impl.py` 再基于 `req_state.num_computed_tokens` 做 clamp 后执行。
- 下一步：先记录为 P1，不在本轮 P0 修复中改动；后续设计 post-forward strict 模式，按真实执行增量（含 prefill/decode 拆分）决定触发。
- 验收标准：`scheduled_tokens > 1` 下，容量轨迹与触发语义可解释且有保护阈值（overflow guard），并可与 strict 参考链路对照验证。
- 状态：Open

## [P1] 4.2 `enable_experimental_block_reclaim=false` 时 `effective_len_regression` 门禁可能误杀
- 背景：为定位 strict reclaim 精度问题，执行单样本 A/B（同题同 seed，仅切 `enable_experimental_block_reclaim`）时，`reclaim=false` 分支在中途崩溃。
- 现象：
  - `triattention_runtime/hook_impl.py` 抛出 `TRIATTN_FATAL_TRITON_SCORING_REQUIRED:effective_len_regressed`
  - 示例：`effective_tokens=2511`, `num_computed_tokens=2511`, `guard_upper=2490`（step≈4655）
- 影响：
  - 会阻断 no-reclaim 对照实验（A/B baseline），造成 shard 失败或结果缺失；
  - 目前看更像调试/对照路径问题，不是 strict reclaim 主线路径的直接 blocker。
- 初步判断：
  - `effective_len_guard` 在 no-reclaim 模式下过严，scheduler/runner 异步步进与 block 粒度 slack 不足，触发 false positive。
- 已修复（2026-02-22）：
  - `triattention_runtime/hook_impl.py` 的 `effective_len_regression` 门禁仅在 `enable_experimental_block_reclaim=true` 时启用；
  - no-reclaim A/B 对照路径不再因该门禁 false positive 中断。
- 说明：
  - 该门禁本意是保护 strict reclaim 长度语义，no-reclaim 路径中触发误杀并不能指示真实错误。
- 状态：Resolved（代码已修，待端到端 A/B 验证）

## [P1] 5. batch>1 行为验证缺失
- 背景：Runtime 明确需要支持 batch>1。
- 影响：不验证将导致线上并发场景风险。
- 现状证据：`tests_runtime/test_runner.py::test_runner_batch_signals_keep_request_isolation` 已覆盖 batch 信号下的状态隔离。
- 下一步：补齐 scheduler 端 batch>1 触发一致性测试与长跑回归。
- 验收标准：batch>1 下结果稳定，且无 request identity 混淆。
- 状态：In Progress

## [P1] 6. 配置导致性能损失：默认 `enforce_eager=True`
- 背景：当前评测链路为保守稳定，历史上默认开启 eager 执行。
- 影响：会压低吞吐上限，导致 full-run 时长偏长，且更容易误判为“压缩逻辑慢”。
- 现状证据：`evaluation/runner/vllm_triattention_runtime_runner.py` 在未显式覆盖时沿用 eager 配置；近期慢跑样本中该项与低利用率同时出现。
- 已完成（代码默认值调整，2026-02-22）：
  - `evaluation/runner/vllm_triattention_runtime_runner.py` 默认 `--enforce-eager=False`；
  - 保留 CLI 开关，必要时可显式回退。
- 待验证：
  - 仍需执行 `false/true` A/B 冒烟并记录吞吐差异，形成量化结论。
- 验收标准：`enforce_eager=false` 在不破坏结果对齐的前提下，吞吐有可观测提升（以 tokens/s 和总时长为准）。
- 状态：In Progress（默认值已修，实验验证待完成）

## [P1] 6.1 评测链路会把“半成品分片结果”产出为正式指标，易误判
- 背景：分片任务在 4/8 已完成时，`eval_math_multi.py` 仍会输出 metrics（只打印 warning，不 fail-fast）。
- 代码证据：`R-KV/HuggingFace/evaluation/eval_math_multi.py` 中仅在 `len(preds) != num_samples` 时告警，不中止评测。
- 影响：会出现 `num_scores=120` 与 `num_scores=240` 共存，容易把半成品指标当最终结果（例如误判“只跑了 4 个 draw”）。
- 已完成（dispatch 门禁，2026-02-22）：
  - `evaluation/dispatch/triattention_sharded_dispatch.py` 新增评测前 merged 完整性检查：
    - 每个 `sample_idx` 的 draw 数必须等于 `num_samples`；
    - 若存在重复 `(sample_idx, draw_idx)` 或缺失 draw，则直接 fail-fast；
  - 避免半成品 merged 输入继续调用 HF `eval_math_multi.py` 产出误导性指标。
- 后续可选增强：
  - 若需要，可再给 `evaluation/eval/eval_math_multi.py` 增加 strict 模式（当前 dispatch 门禁已覆盖主工作流）。
- 验收标准：不完整 merged 输入不会生成“看似合法”的最终 accuracy 文件。
- 状态：Resolved（dispatch 主链路）

## [P1] 6.2 reclaim 保序修复当前实现的搬运量可能接近 `keep_count`（性能不可接受）
- 背景：为修复 strict reclaim 精度崩坏（runaway/异常长输出），已将 `preserve_dropped_tokens=False` 的 fill-in-place 路径改为“保序前缀写入”。
- 问题：当前保序实现在 shared/per_head 路径上可能重写整个保留前缀（最坏接近 `keep_count` 个 token 的 K/V 搬运）。
- 影响：
  - correctness 恢复，但压缩阶段显存读写开销增大；
  - 不符合目标中的“低搬运量/接近 A+B 分区式整理”的性能预期。
- 说明（重要）：
  - 这是 correctness stopgap，不是最终性能形态；
  - 且最新验证显示：仅靠“recent 语义修复”不足以支撑低搬运乱序 fill-hole，仍会出现 strict reclaim runaway；
  - 因此需要重新设计最终 reclaim 方案（不是简单在 stopgap 与乱序 fill-hole 间二选一）。
- 下一步：
  1. 明确目标语义：以“逻辑时序保序”为硬约束；
  2. 设计低搬运保序版本（优先只搬运 `src_idx != dst_idx` 的位置；必要时进一步引入 A/B 分区式布局）；
  3. 对 shared/per_head 分别统计每轮搬运 token 数，作为回归指标；
  4. 在不改变结果语义前提下做性能回归（tokens/s、压缩段耗时）。
- 验收标准：strict reclaim 结果保持稳定，同时压缩搬运量显著低于“重写整个 keep 前缀”的 stopgap 实现。
- 状态：Open

## [P2] 7. 进一步性能优化（TopK/Gather）
- 背景：当前阶段优先正确性，性能优化可后置。
- 影响：吞吐上限暂受限。
- 下一步：Phase 3 再评估是否需要 Triton TopK/Gather。
- 验收标准：有明确收益再实施，不强行提前优化。
- 状态：Open
