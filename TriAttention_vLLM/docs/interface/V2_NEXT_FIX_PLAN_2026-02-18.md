# V2 下一阶段修复清单（性能 + 语义）

- 更新时间：2026-02-18
- 状态：Active
- 适用范围：TriAttention_vLLM V2（vLLM 0.15.x）

---

## 1. 目标与范围

本文件用于固化“下一阶段必须完成”的修复项，避免开发过程中目标漂移。

短期目标（必须达成）：

1. 让压缩真正降低 attention 可见上下文成本，而不是仅做逻辑重排。
2. 严禁无声退化到接近 full-history 路径。
3. 修复当前可确定的性能损耗点（trig cache 未接线、eager 固定开启等）。

中期目标（紧随其后）：

1. 去掉“paged KV -> dense K”中间态搬运，改为 paged layout 直接打分+选点。
2. 进一步把 compaction 数据移动量收敛到“仅移动必要 token”。

---

## 2. 当前实现事实（代码证据）

1. 压缩打分前会先把 K gather 成连续 dense 张量：
   - `triattention_v2/kv_compaction.py:126`
   - `triattention_v2/kv_compaction.py:137`
   - `triattention_v2/kv_compaction.py:149`
2. 每次压缩不是只处理“新生成 128 token”，而是处理 `effective_tokens` 全段：
   - gather 输入：`triattention_v2/hook_impl.py:589`、`triattention_v2/hook_impl.py:622`
   - compaction 输入：`triattention_v2/hook_impl.py:671`、`triattention_v2/hook_impl.py:679`
3. 当前默认配置下 block reclaim 关闭时，只做逻辑重排，不做物理回收：
   - reclaim 开关分支：`triattention_v2/hook_impl.py:691`
4. Triton 打分已启用，但 trig cache 预计算分支未接入调用链：
   - 支持 `trig_cache`：`triattention/kernels/triton_scoring.py:508`
   - 当前调用未传 `trig_cache`：`triattention/scoring.py:143`
5. 评测 runner 里 `enforce_eager` 固定为 `True`：
   - `evaluation/runner/vllm_triattention_v2_runner.py:414`

---

## 3. 待修复问题列表（按优先级）

## [P0] A. 压缩未稳定转化为 attention 成本下降
- 背景：
  - 逻辑压缩已经可执行，但若无稳定物理回收/可见长度闭环，主 attention 成本可能仍接近长历史。
- 影响：
  - 运行时吞吐与预算控制目标不一致，full run 时间不可控。
- 下一步：
  - 建立“压缩后可见长度闭环”的强约束路径（见第 4 节推荐方案）。
- 验收标准：
  - 压缩后 attention 相关可观测长度稳定贴合预算口径（而非持续贴近 full-history）。
  - 8-shard 长跑吞吐恢复到可接受范围。
- 状态：In Progress（已补齐 before/after/budget/reclaim 观测字段与 summary 日志，仍需 8-shard 吞吐与可见长度轨迹验收）

## [P0] B. 禁止退化到全历史且无声继续
- 背景：
  - 若 tracker/event 链路异常，`effective_tokens` 可能回升，导致性能与语义偏离。
- 影响：
  - 实验可能“看起来在跑”，但实际已退化，结论不可用。
- 下一步：
  - 增加 fail-fast 门禁：一旦命中退化条件，直接报错退出。
- 验收标准：
  - 无 silent fallback；异常时日志包含明确 marker，便于 unattended 排查。
- 状态：Closed（hook + runner + eval 入口 strict 门禁已落地）

## [P0] C. Triton trig cache 分支未启用
- 背景：
  - 当前每次评分仍在 kernel 内计算 trig。
- 影响：
  - 打分阶段存在可避免计算开销。
- 下一步：
  - 在 compressor/scoring 初始化阶段创建并缓存 trig table；
  - 调用 `speckv_scoring(..., trig_cache=...)` 走 cached kernel。
- 验收标准：
  - 日志可确认走 cached 分支；
  - 小样本 A/B 显示打分阶段耗时下降。
- 状态：Closed

## [P0] D. `enforce_eager` 固定开启，吞吐上限受限
- 背景：
  - 当前评测路径硬编码 eager。
- 影响：
  - 禁用 cudagraph/编译优化，吞吐下降。
- 下一步：
  - runner 增加 `--enforce-eager` 开关；
  - 在小样本先验证 `false` 的正确性与稳定性，再决定全量策略。
- 验收标准：
  - 开关生效、默认行为明确；
  - 关闭 eager 的 smoke 与小样本对齐通过。
- 状态：Closed

## [P1] E. 去掉 paged->dense gather 的整段搬运
- 背景：
  - 当前打分前强制 gather 成 `[1,H,T,D]` dense。
- 影响：
  - 引入额外显存搬运；层数高时放大开销。
- 下一步：
  - 实现 paged layout 直接打分+选点（不创建 dense 中间态）。
- 验收标准：
  - 在等价语义下，压缩阶段总耗时明显下降。
- 状态：In Progress（单层与 global per-head 均改为 paged 流式 top-k，且 global per-head 已修复为全序列标准化语义；待 8-shard 实验验收）

## [P1] F. compaction 数据移动过大
- 背景：
  - 当前是按 `effective_tokens` 整段重排读写。
- 影响：
  - 每次触发压缩都产生较大内存流量。
- 下一步：
  - 向 fill-in-place 路线收敛：优先移动“新增区幸存 token”，减少历史区搬运。
- 验收标准：
  - 每轮压缩的数据移动量显著降低；
  - 语义与对齐不回退。
- 状态：Closed（keep-only 已切换为 fill-in-place 空洞填充语义，仅移动 tail 幸存者）

---

## 4. 推荐方案：让压缩真正影响 attention 计算长度

推荐采用“两阶段闭环”：

### 阶段 1（先落地，P0）

目标：先把“压缩 -> 可见长度下降 -> attention 成本下降”跑通并可验证。

执行要点：

1. 默认启用并稳定化 block reclaim 闭环（去实验态）：
   - worker 侧压缩后给出 `block_ids_after`；
   - scheduler 侧应用并回收 tail blocks。
2. 加强一致性约束（不满足即 fail-fast）：
   - `block_ids_after` 与当前前缀一致性；
   - `cache_len_after` 与 `required_blocks` 一致性；
   - tracker 与事件应用后长度关系一致性。
3. 增加“退化告警即报错”：
   - 已压缩请求若再次长期贴近 full-history 长度，直接中止。

说明：

1. `num_computed_tokens` 保持单调增长（用于绝对位置语义）；
2. 预算控制与可见长度由 reclaim+tracker 约束；
3. 若发现与 vLLM append 契约冲突，禁止静默降级，必须显式报错。

### 阶段 2（再优化，P1）

目标：去掉当前高开销中间态，提升压缩阶段效率。

执行要点：

1. 打分改为 paged layout 直算，不再 gather dense K。
2. compaction 改为 fill-in-place 风格，减少全段重排写回。
3. 结合 trig cache，进一步降低评分热路径开销。

---

## 5. 本轮建议执行顺序

1. C：先接 trig cache（改动小、收益确定）。
2. B：加退化 fail-fast（保障实验可信性）。
3. D：开放 `enforce_eager` 开关并做小样本验证。
4. A：推进可见长度闭环（block reclaim 稳定化 + 强约束）。
5. E/F：paged 直算与 fill-in-place 优化（中期实现）。

---

## 6. 回归与验收门禁

每个阶段至少满足：

1. `tests_v2/run_smoke.py` 全通过。
2. 小样本对齐（固定 seed）无明显回退。
3. 吞吐 sanity：
   - 记录 tokens/s；
   - 记录压缩触发次数与单次耗时；
   - 记录可见长度轨迹（不得长期回升到 full-history）。

---

## 7. 关联文档

1. `docs/interface/OPEN_ISSUES.md`
2. `docs/interface/V2_P0_ISSUES_EXPLAINED_2026-02-17.md`
3. `docs/interface/V2_WORKLOG.md`
4. `docs/backend/V2_RECLAIM_STRATEGY.md`
5. `docs/backend/reference/implementation/fill_in_place.md`

---

## 8. 执行进度记录（可交接）

### 8.1 2026-02-18 已完成

1. `P0-C`（trig cache 接线）已落地：
   - `triattention/config.py`：新增 trig cache 配置项（`use_trig_cache` 等）。
   - `triattention/compressor.py`：初始化并持有 `self.trig_cache`。
   - `triattention/scoring.py`：`compute_scores_triton` 支持 `trig_cache`；
     - 对齐 round_start：直接走 cached 分支；
     - 非对齐 round_start：通过“基准轮次 + 残差角修正”构造 trig 表，仍走 cached 分支；
     - 超出缓存范围：回退 on-the-fly 分支。
   - `triattention/kernels/triton_scoring.py`：`speckv_scoring` 新增 `trig_values` 参数，支持显式 trig 表输入（不依赖 strict round_start 对齐）。
   - `triattention_v2/hook_impl.py`：V2 selector 调用 `compute_scores_triton` 时传入 `compressor.trig_cache`。
2. `P0-B`（退化到近 full-history fail-fast）已落地：
   - `triattention_v2/config.py`：新增
     - `fail_on_effective_len_regression`
     - `effective_len_regression_ratio`
     - `effective_len_guard_divide_multiples`
   - `triattention_v2/hook_impl.py`：请求一旦出现“压缩后再次接近 full-history 长度”即抛出 `TRIATTN_FATAL_TRITON_SCORING_REQUIRED:effective_len_regressed:*`，禁止静默降级。
3. `P0-D`（`enforce_eager` 可配置）已落地：
   - `evaluation/runner/vllm_triattention_v2_runner.py`：
     - 新增 CLI 参数 `--enforce-eager`
     - 新增 regression guard 相关 CLI + env 透传
     - `LLM(..., enforce_eager=...)` 由参数控制，不再硬编码。
4. 测试已同步更新：
   - `tests_v2/test_config.py`
   - `tests_v2/test_v2_eval_runner.py`
   - `tests_v2/test_hook_impl.py`
5. strict“禁止降级”门禁加固：
   - `triattention_v2/runner.py`：
     - 开启 compaction 时，对压缩执行异常改为 fail-fast（不再记录后继续跑）；
     - 对 should_compress 下的异常 skip reason（除少数白名单）改为 fail-fast。
   - `evaluation/runner/vllm_triattention_v2_runner.py`：
     - strict 启动参数默认 compaction/reclaim/triton 全开启并做启动前强校验。
   - 新增回归用例：
     - `tests_v2/test_runner.py` strict missing-hook / executor-exception 两条。
6. trig cache 命中率修复回归：
   - 新增 `tests_v2/test_scoring_trig_cache.py`（对齐命中、非对齐命中、越界回退三条路径）。
   - `tests_v2/run_smoke.py` 纳入新模块。
7. global per-head 组内打分输入流式化：
   - `triattention_v2/hook_impl.py` group selector 新增 `layer_input_iter`；
   - hook 侧不再缓存整组 `keys_dense` 列表，改为逐层 gather/打分聚合。
8. 物理回收一致性门禁加固：
   - `triattention_v2/hook_impl.py` 在 `require_physical_reclaim=True` 时，若按长度应发生 block shrink 但实际未回收足量 block，直接 fail-fast。
9. 打分热路径额外拷贝削减：
   - `triattention_v2/kv_compaction.py` 的 K/KV gather 去除末端 `.contiguous()`；
   - `triattention/scoring.py` 去除 `key_states.contiguous()`；
   - `triattention/kernels/triton_scoring.py` 去除输入 contiguous 限制断言，按 stride 访问。
10. 条件 keep-only compaction 快路径：
   - `triattention_v2/kv_compaction.py` 新增 `preserve_dropped_tokens` 参数；
   - `triattention_v2/hook_impl.py` 在“确定会 block shrink”的场景启用 keep-only，仅搬运 keep token，减少整段 permutation 搬运。
11. P1-E 第一阶段（paged 分块打分）：
   - `triattention_v2/kv_compaction.py` 新增 `gather_request_k_dense_range`；
   - `triattention_v2/hook_impl.py` 内置 selector 新增 paged 分块打分路径（`_supports_paged=True`），默认不再强制先 gather 整段 `K`。
12. P1-E 第二阶段补全（global per-head）：
   - `triattention_v2/hook_impl.py` 在 group selector 调用点优先传 `layer_kv_iter`（`_supports_paged_group=True`），不支持时再回退 dense `layer_input_iter`；
   - `tests_v2/test_hook_impl.py` 对应兼容 paged/dense 两种 group 输入签名，并新增 `_supports_paged_group=True` 强断言回归（禁止无声回退 dense）。
13. P0-A 一致性门禁补强（scheduler 侧）：
   - `triattention_v2/scheduler.py` 新增规则：
     - 若 `require_physical_reclaim=True` 且按 `cache_len_after` 推导应 shrink blocks，但事件缺失 `block_reclaim` 或缺失应 shrink 的 group，直接报错；
     - 若回收后 `kept_len != required_blocks`（应 shrink group）直接报错。
   - `tests_v2/test_scheduler.py` 新增 `test_scheduler_missing_block_reclaim_raises_when_shrink_expected`。
14. P0-A 可观测性补齐（长跑验收准备）：
   - `triattention_v2/hook_impl.py` 压缩结果新增 `effective_tokens_before`、`budget_total`、`reclaimed_block_count`；
   - `triattention_v2/runner.py` 在 applied 事件输出 summary 日志（before/after/budget/reclaimed_blocks/reason）。
15. P1-E 单层 paged selector 去全段 scores 物化：
   - `triattention_v2/hook_impl.py` paged 分支改为分块流式 top-k（shared/per-head），避免 `torch.cat` 全段 chunk scores。
16. P1-F keep-only 搬运进一步缩减：
   - `triattention_v2/kv_compaction.py` keep-only 只搬运 `src!=dst`；
   - per-head keep-only 同步改为 `(head, token)` 稀疏搬运；
   - 新增 prefix no-op 快路径。
17. P1 运行稳定性：
   - `evaluation/dispatch/triattention_sharded_dispatch.py` 新增 `--stall-timeout-minutes`（log 无增长 fail-fast）；
   - 新增 `tests_v2/run_lite_gate.py` 作为稳定 lite 回归入口。
18. P1-F fill-in-place 终版落地：
   - `triattention_v2/kv_compaction.py` keep-only 从“前缀稳定压缩”升级为“空洞填充”；
   - per-head keep-only 同步为逐 head 空洞填充；
   - `tests_v2/test_kv_compaction.py` 新增 fill-holes 用例并更新旧断言。
19. P1-E global per-head 跨层聚合优化：
   - `triattention_v2/hook_impl.py` paged global per-head 改为 chunk 流式聚合 + 增量 top-k merge；
   - 去掉全序列 `aggregated_scores` 跨层累加张量。
20. 待验收风险（对齐）：
   - 已关闭：global per-head chunk 标准化风险已修复为“全序列标准化（两遍流式）”；
   - 下一步仅需在 8-shard HF 对齐实验中做结果级验收。

### 8.2 本轮验证结果

1. 定向 pytest：
   - `conda run -n rkv python -m pytest -q tests_v2/test_config.py tests_v2/test_v2_eval_runner.py tests_v2/test_hook_impl.py`
   - 结果：`26 passed`
2. smoke 回归：
   - `conda run -n rkv python tests_v2/run_smoke.py`
   - 结果：`smoke passed: 74 tests`
3. 全量 `tests_v2` 回归：
   - `conda run -n rkv python -m pytest -q tests_v2`
   - 结果：本机环境存在偶发卡住，当前以 `run_smoke + 定向 pytest` 为门禁（见 `V2_WORKLOG` 4.8）

### 8.3 当前未完成（下一棒直接接）

1. `P0-A`：压缩“默认”转化为 attention 可见长度下降（reclaim 闭环稳定化、去实验态）。
2. `P1-E`：paged layout 直算打分（去 dense gather 中间态）。
3. `P1-F`：compaction 数据移动量进一步缩减（向 fill-in-place 收敛）。

### 8.4 中断恢复指引

若本轮中断，接手人按以下顺序继续：

1. 先跑 `tests_v2/run_smoke.py` 确认当前基线。
2. 优先推进 `P0-A`，保持 fail-fast 约束不被弱化。
3. 每完成一个子步骤，更新本文件 8.x 小节（已完成/未完成/风险）。
