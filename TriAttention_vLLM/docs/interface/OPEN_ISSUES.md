# OPEN ISSUES（V2）

- 更新时间：2026-02-20
- 状态：Active
- 适用范围：vLLM 0.15.x

---

## [P0] 1. V2 触发链路需要从原型走向稳定版
- 背景：V2 要求由 scheduler 侧决定何时压缩，runner 侧执行压缩。
- 影响：没有该链路就无法验证“显存触发压缩”主能力。
- 现状证据：`triattention_v2/scheduler.py` 已挂载 `triattention_signals` 并接入 effective len tracker；`triattention_v2/runner.py` 已消费信号并调用 executor。
- 下一步：将当前 experimental compaction 从原型升级为稳定实现，并覆盖多层/多组 KV cache 场景。
- 验收标准：可在日志中观测到“达到阈值 -> 触发压缩 -> 执行 hook -> 压缩执行完成”的稳定流程，且行为可回归验证。
- 状态：In Progress

## [P0] 1.1 V2 与 HF 等价性验证仍未完成
- 背景：项目终极目标是与 HF SpeckV 等价。
- 现状证据：已具备 V2 quick 评测入口（`evaluation/runner/vllm_triattention_v2_runner.py` + quick dispatch 配置），可快速产出对比样本。
- 差距：V2 当前压缩执行仍是原型 compaction，尚未接入完整 SpeckV score/topk 语义。
- 下一步：先跑 quick 对齐实验确认偏差规模，再决定优先补齐哪些语义差距。
- 验收标准：在固定小样本上得到可复现实验报告（accuracy/token match/长度差异）。
- 状态：Open

## [P0] 1.2 V2 full-run 吞吐异常偏低（长时间低产出）
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
  - 修复事件回传与有效长度同步；
  - 复用 Triton scoring，保留 torch gather/scatter，同时去除关键 token 级 for 循环；
  - 清理旧输出并重跑全量实验验证修复效果。
- 验收标准：在相同配置下，全量运行时长回到可接受区间，且日志显示稳定产出增长。
- 状态：In Progress

## [P0] 1.3 experimental compaction 语义错误会污染注意力分布
- 背景：当前 V2 原型在逻辑长度不缩短的前提下执行 in-place compaction。
- 问题：旧实现将尾部 KV 置零；这些“零 K”仍参与 softmax，导致分母被大量无效项放大，生成质量显著劣化（可表现为乱码/重复/异常长输出）。
- 已确认修复方向：改为“全量 permutation（kept + dropped）”而非“kept + zero tail”，先保证语义与 FullKV 等价，再继续推进真正的物理长度收缩方案。
- 下一步：完成端到端对齐回归（itercheck + full run）验证该修复是否消除异常输出。
- 验收标准：修复后不再出现大规模乱码/重复退化；HF 对齐指标显著回升。
- 状态：In Progress

## [P0] 1.4 `per_head` 语义与 HF RKV-style 存在结构差异
- 背景：HF RKV-style 的 `per_head` 语义是“跨层聚合后按 KV head 独立选择，再将同一组 per-head 索引应用到各层”。
- 问题：V2 旧实现在 hook 内按“每层独立 per-head 选择”执行，行为更接近 `per_layer_per_head`，会导致对齐实验存在系统性偏差。
- 修复：新增 `per_head_selection_semantics` 开关：
  - `legacy_layer_local`：保留旧行为用于历史复现；
  - `hf_aligned_global_per_head`：按组跨层聚合后统一 per-head 选择（V2 对齐模式）。
- 当前状态：
  - 已落地“attention-head 打分 -> 组内（KV group）max -> 跨层 mean -> per-head topk”路径；
  - 已补齐头维适配：当 stats 头数与 runtime KV 头数不一致时，不再隐式使用前几个头；
  - 代码与单测已落地（`triattention_v2/hook_impl.py`、`tests_v2/test_hook_impl.py`），待全量 AIME24 sample8 复跑验证指标。
- 验收标准：在同一参数集下，V2 与 HF 的差异收敛到可解释范围，且 legacy 结果可复现。
- 状态：In Progress

## [P0] 1.5 物理回收能力未闭环（仅逻辑压缩）
- 背景：当前 V2 experimental compaction 以逻辑重排为主，尚未稳定回收 request tail blocks 到 free pool。
- 影响：长跑显存/吞吐行为与“预算区+overflow”目标策略存在偏差，无法作为最终实现形态。
- 现状证据：
  - vLLM 默认契约为 append 路径：`vllm/v1/core/sched/output.py:116`、`vllm/v1/worker/gpu_model_runner.py:1037`。
  - `KVCacheManager` 公开 API 无 request 级局部 shrink/free：`vllm/v1/core/kv_cache_manager.py:378`。
  - 方案与分阶段边界已沉淀：`docs/backend/V2_RECLAIM_STRATEGY.md`。
- 下一步：
  - 已落地“半侵入继承层”回收原型闭环（runner 事件 + scheduler 应用 + block_pool 回收），下一步做端到端压测验证；
  - 以实验开关保护，默认保持主线行为不变；
  - 执行过程同步记录到 `interface/V2_WORKLOG.md`，确保多人可追踪接手；
  - 闭环稳定后再推进更严格 fill-in-place 页整理。
- 验收标准：压缩触发后可观测到回收事件并实际归还 tail blocks；默认关开时行为保持兼容。
- 状态：In Progress

## [P0] 2. 请求级状态生命周期尚未在 V2 代码闭环
- 背景：V1 历史问题证明 request state 处理是高风险点。
- 影响：状态污染会直接导致压缩策略错误或结果漂移。
- 现状证据：`triattention_v2/state.py` + `triattention_v2/runner.py` 已接入生命周期骨架，覆盖 new/finished/preempt/resume。
- 下一步：补齐与真实压缩执行联动后的状态一致性校验。
- 验收标准：长跑测试无跨请求状态污染；请求结束后状态可回收。
- 状态：In Progress

## [P0] 3. Phase 1 回归门禁缺失
- 背景：多人并行开发需要固定最小回归集。
- 影响：修改后可能破坏核心路径且无人感知。
- 现状证据：`tests_v2/run_smoke.py` 已可执行 59 个基础用例（不依赖 pytest）。
- 下一步：将该脚本接入 CI 或统一 pre-merge 流程，形成强制门禁。
- 验收标准：PR 可自动/半自动执行并给出通过结论。
- 状态：In Progress

## [P1] 4. prefill 裁剪策略未落地
- 背景：V2 支持 `protect_prefill=false`，但 Phase 1 默认先保护。
- 影响：影响后续压缩率与策略实验。
- 现状证据：`triattention_v2/kv_compaction.py` 已实现裁剪语义，`tests_v2/test_kv_compaction.py` 与 `tests_v2/test_hook_impl.py` 已覆盖关键路径。
- 下一步：在真实 vLLM 端到端链路中验证该模式（不仅是单元/冒烟）。
- 验收标准：两种 prefill 模式可配置切换且行为可验证。
- 状态：In Progress

## [P1] 4.1 `scheduled_tokens > 1` 场景下的 prefill 兼容风险
- 背景：当前 V2 触发链路包含 scheduler 估算长度 + runner 前置执行压缩的路径；在 chunked prefill 或单轮执行多 token 场景中，估算口径与真实执行口径可能出现偏差。
- 影响：可能导致压缩触发步与 HF strict 语义不一致，进而在 prefill 边界下出现“触发延后一轮/选点集合不同”的行为偏差；在极端场景可能放大为容量控制不稳定。
- 现状证据：`triattention_v2/scheduler.py` 使用 `estimated_cache_len = effective_base_len + scheduled_tokens`，`triattention_v2/hook_impl.py` 再基于 `req_state.num_computed_tokens` 做 clamp 后执行。
- 下一步：先记录为 P1，不在本轮 P0 修复中改动；后续设计 post-forward strict 模式，按真实执行增量（含 prefill/decode 拆分）决定触发。
- 验收标准：`scheduled_tokens > 1` 下，容量轨迹与触发语义可解释且有保护阈值（overflow guard），并可与 strict 参考链路对照验证。
- 状态：Open

## [P1] 5. batch>1 行为验证缺失
- 背景：V2 明确需要支持 batch>1。
- 影响：不验证将导致线上并发场景风险。
- 现状证据：`tests_v2/test_runner.py::test_runner_batch_signals_keep_request_isolation` 已覆盖 batch 信号下的状态隔离。
- 下一步：补齐 scheduler 端 batch>1 触发一致性测试与长跑回归。
- 验收标准：batch>1 下结果稳定，且无 request identity 混淆。
- 状态：In Progress

## [P1] 6. 配置导致性能损失：默认 `enforce_eager=True`
- 背景：V2 评测链路为保守稳定，历史上默认开启 eager 执行。
- 影响：会压低吞吐上限，导致 full-run 时长偏长，且更容易误判为“压缩逻辑慢”。
- 现状证据：`evaluation/runner/vllm_triattention_v2_runner.py` 在未显式覆盖时沿用 eager 配置；近期慢跑样本中该项与低利用率同时出现。
- 下一步（必须执行）：
  1. 在同一配置上做 `enforce_eager=false/true` A/B 冒烟（1~2 题）并记录吞吐差异；
  2. A/B 通过后，将 full-run 默认切到 `enforce_eager=false`；
  3. 保留开关，若出现兼容性回归可一键回退到 eager。
- 验收标准：`enforce_eager=false` 在不破坏结果对齐的前提下，吞吐有可观测提升（以 tokens/s 和总时长为准）。
- 状态：Open

## [P2] 7. 进一步性能优化（TopK/Gather）
- 背景：当前阶段优先正确性，性能优化可后置。
- 影响：吞吐上限暂受限。
- 下一步：Phase 3 再评估是否需要 Triton TopK/Gather。
- 验收标准：有明确收益再实施，不强行提前优化。
- 状态：Open
