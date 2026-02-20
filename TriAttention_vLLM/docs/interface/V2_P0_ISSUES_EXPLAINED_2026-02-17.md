# V2 P0 问题详解（面向 HF 严格对齐）

- 更新时间：2026-02-20
- 状态：Active
- 适用范围：TriAttention_vLLM V2（vLLM 0.15.x）

---

## 1. 文档目的

这份文档专门回答一个问题：

> 为什么我们说当前 V2 还存在 P0 风险，不能直接宣称“与 HF 严格等价”？

重点不是“结果大概像不像”，而是“实现语义是否同构”。

---

## 2. 背景：我们在对齐什么

我们当前对齐目标是 HF SpeckV/R-KV 路线（AIME24 sample8），核心不是某一次 acc 数字，而是以下三层对齐：

1. 公式层：打分公式和参数来源一致。
2. 时序层：在同一时刻、基于同一缓存长度和位置信息触发压缩。
3. 选择层：per_head/per_layer 等模式下的 token 选择语义一致。

如果这三层有任何一层偏差，最终指标“接近”也可能只是偶然。

---

## 3. P0-1：`freq_scale_sq` 来源语义不一致

### 3.1 先讲概念：`q_abs_mean` 和 `freq_scale_sq` 不是一回事

1. `q_abs_mean`：来自统计文件，表示 query 统计量（和训练/采样数据分布相关）。
2. `freq_scale_sq`：来自 RoPE 频率缩放（和模型 RoPE 参数相关）。

它们都参与打分，但语义来源不同，不可互相替代。

### 3.2 HF 是怎么做的

HF R-KV/SpeckV 路径中：

1. 统计项 `q_abs_mean` 参与 `extra` 项构造：
   - `R-KV/weian_development/speckv/round_pruning_utils.py:256`
   - `R-KV/weian_development/speckv/round_pruning_utils.py:271`
2. `freq_scale_sq` 来自 rotary 计算，而不是 stats 的 `q_abs_mean`：
   - `R-KV/weian_development/speckv/speckv_rkv_style.py:131`
   - `R-KV/weian_development/speckv/speckv_rkv_style.py:132`
3. 两者在打分中分别使用：
   - base term 使用 `freq_scale_sq`：`R-KV/weian_development/speckv/round_pruning_utils.py:315`
   - additive term 也乘 `freq_scale_sq`：`R-KV/weian_development/speckv/round_pruning_utils.py:317`

### 3.3 V2 当前是怎么做的

V2 旧版曾在 R-KV stats 转换时把 `q_abs_mean**2` 直接写成 `freq_scale_sq`。
当前已修复为“优先从模型 RoPE 推导，失败时显式回退 ones”。

- `TriAttention_vLLM/triattention/utils.py:198`
- `TriAttention_vLLM/triattention/utils.py:201`

后续打分直接吃这个 `freq_scale_sq`：

- `TriAttention_vLLM/triattention/scoring.py:149`

### 3.4 为什么这是 P0

这会把“query 统计量”当成“RoPE 频率缩放”来用，属于公式参数语义混淆。  
即使结果有时看起来接近，也不代表严格等价。

### 3.5 修复方向（建议）

1. `freq_scale_sq` 改为按 HF 路径由 rotary 实时/初始化计算。
2. stats 只提供 `q_mean_* / q_abs_mean`，不再“伪造” `freq_scale_sq`。
3. 增加单测：同一输入下，V2 与 HF 的单层 score 张量误差阈值校验。

---

## 4. P0-2：Triton“必须使用”没有被完全强制

### 4.1 需求背景

当前阶段你已经明确要求：

1. 打分必须走 Triton。
2. Triton 失败应直接报错退出，不能静默回退。

### 4.2 V2 当前行为

目前只有“执行时抛出特定 marker 异常”才会 fatal（这是好的）：

- `TriAttention_vLLM/triattention_v2/runner.py:110`

但在 selector 构建阶段存在非 fatal 返回路径：

1. stats 路径未配置时返回 `None` selector：
   - `TriAttention_vLLM/triattention_v2/hook_impl.py:133`
2. stats 文件不存在时返回 `None` selector：
   - `TriAttention_vLLM/triattention_v2/hook_impl.py:138`
3. pruning_mode 不支持时返回 `None` selector：
   - `TriAttention_vLLM/triattention_v2/hook_impl.py:152`

之后会落到通用 fallback 选点逻辑：

- `TriAttention_vLLM/triattention_v2/hook_impl.py:524`

### 4.3 为什么这是 P0

这会导致“配置错误但流程继续跑完”，从外面看像正常实验，实际已偏离 Triton 严格路径。

### 4.4 修复方向（建议）

在 HF 对齐模式下（例如专用开关）：

1. selector 构建失败直接 raise（而不是返回 `None`）。
2. 禁止进入 `selection_mode=fallback`。
3. 运行日志打印“已启用 strict triton mode”并输出关键配置摘要。

---

## 5. （已降级为 P1）压缩触发时序与 HF 语义不完全同构

> 2026-02-17 决策：该项先不纳入本轮 P0 修复，转入 P1（重点关注 `scheduled_tokens > 1` 与 prefill 兼容）。
> 记录入口：`docs/interface/OPEN_ISSUES.md` 的 `[P1] 4.1`。

### 5.1 HF 时序（R-KV style）

HF 在 forward 中先根据 cache 真实增长更新位置，再判断是否触发压缩：

1. decode 追加后更新 `absolute_position`：
   - `R-KV/weian_development/speckv/speckv_rkv_style.py:1048`
   - `R-KV/weian_development/speckv/speckv_rkv_style.py:1063`
2. 之后按阈值/间隔判断压缩：
   - `R-KV/weian_development/speckv/speckv_rkv_style.py:1070`
   - `R-KV/weian_development/speckv/speckv_rkv_style.py:1081`

### 5.2 V2 当前时序

V2 在 scheduler 先估算下一步长度并下发 signal：

- `TriAttention_vLLM/triattention_v2/scheduler.py:108`

runner/hook 在 `execute_model` 前执行压缩动作（基于 signal 的 estimated len）：

- `TriAttention_vLLM/triattention_v2/hook_impl.py:385`
- `TriAttention_vLLM/triattention_v2/hook_impl.py:450`

### 5.3 为什么它仍是高风险语义差异

同一轮 decode 下，HF 与 V2 可能在“压缩前后是否包含最新 token、round_start 取值、触发边界”上有一步偏差。  
这类偏差会直接影响 topk 选点，属于严格对齐风险。

### 5.4 后续修复方向（P1）

先明确一条“唯一标准时序”（建议以 HF 语义为准），再把 V2 触发点绑定到该时序：

1. 定义“触发时 cache_len 的口径”（是否含本轮新增 token）。
2. 定义“round_start 的口径”（与 HF 同步）。
3. 做 deterministic 子集（固定 seed）token-level 对照，验证触发步完全一致。

---

## 6. P0-4：`per_head` 聚合细节仍有结构差异

### 6.1 HF `per_head` 当前语义

HF 路径（修复后的实现）是：

1. 按 `(layer, kv_head)` 分组；
2. 每层组内先 `max`；
3. 再跨层 `mean`；
4. 最后每个 KV head 自己 topk。

代码：

- `R-KV/weian_development/speckv/speckv_rkv_style.py:409`
- `R-KV/weian_development/speckv/speckv_rkv_style.py:449`
- `R-KV/weian_development/speckv/speckv_rkv_style.py:455`

### 6.2 V2 `hf_aligned_global_per_head` 当前语义（已修复）

V2 已改为：

1. 当 stats 头数与 runtime KV 头数不一致时，先在打分侧扩展到 attention-head 粒度；
2. 每层内按 KV group 做 `max`；
3. 再跨层 `mean`；
4. 最后按 KV head 独立 topk。

代码：

- `TriAttention_vLLM/triattention_v2/hook_impl.py`
- `TriAttention_vLLM/tests_v2/test_hook_impl.py::test_selector_hf_global_per_head_uses_attention_head_scores_and_group_max`
- `TriAttention_vLLM/tests_v2/test_hook_impl.py::test_selector_reduces_stats_heads_to_runtime_heads_for_legacy_path`

### 6.3 当前剩余差异（仍需跟踪）

虽然主聚合顺序已对齐，但仍有两点需要实验确认：

1. tie-break 细节：HF 路径在部分模式下会加极小噪声破平分，V2 当前未引入该噪声；
2. 端到端等价仍需通过 full-run 指标和固定子集 token-level 对照来证明。

### 6.4 当前结论

P0-4 的“结构性偏差”已完成主修复，状态从“阻塞实现”转为“待实验验收”。

---

## 7. 补充：为什么你看到“两个准确率”

这不是 P0 本体，但会干扰判断，所以这里说明：

1. 官方评测 `acc`（我们最终看这个）：
   - 来自 `eval_math_multi.py -> evaluate.py`，按每题 8 次平均再全局平均：
   - `R-KV/HuggingFace/evaluation/evaluate.py:87`
2. `compare_results.py` 是“any-correct”口径：
   - 题目只要任意一次命中就记对：
   - `TriAttention_vLLM/benchmarks/reasoning/compare_results.py:156`

所以 `76.67` 与 `43.8` 可以同时成立，不冲突，只是口径不同。

---

## 8. 当前状态结论（本轮）

1. 当前 V2 已经“接近 HF”，不是“明显跑偏”。
2. P0-1 已完成：`freq_scale_sq` 不再来自 `q_abs_mean**2`，并新增防回退测试（`tests_v2/test_utils_rkv_stats.py`）。
3. P0-2 已完成：strict Triton 模式下不可用即 fail-fast，不再静默 fallback。
4. P0-4 主修复已完成：`hf_aligned_global_per_head` 已改为 attention-head 打分后按 KV group 层内 max、跨层 mean；剩余为实验验收项（full-run + 子集 token-level 对照）。
5. 时序问题（`scheduled_tokens > 1` / prefill 兼容）按决策转入 P1，后续继续跟进避免场景扩展时放大风险。

---

## 9. P1 问题记录（本轮同步）

以下是本轮审计已确认、但优先级低于 P0 的问题，后续要持续追踪：

1. `max_new_tokens` 由外部 tokenizer 长度估算，和 vLLM 内部 tokenization 可能有微小偏差：
   - `TriAttention_vLLM/evaluation/runner/vllm_triattention_v2_runner.py:526`
2. eval runner 固定 `enforce_eager=True`，会限制吞吐上限：
   - `TriAttention_vLLM/evaluation/runner/vllm_triattention_v2_runner.py:405`

---

## 10. 2026-02-20 复发事故与热修（strict guard）

### 10.1 现象

`hf_strict_freegpu` 跑到约第 4647 step 时多分片同时崩溃，报错一致：

- `TRIATTN_FATAL_TRITON_SCORING_REQUIRED:effective_len_regressed`
- 典型值：`effective_tokens=2507`，`guard_upper=2490`

日志：

- `TriAttention_vLLM/evaluation/logs/triattention_v2_aime24_hf_strict_freegpu_20260220_150327/triattention_shard00_20260220_150414.log`

### 10.2 根因（两层）

1. strict guard 的 token 口径过于依赖 scheduler signal（`estimated_cache_len`），
   在异步条件下会出现 `+1~+3` 的步间偏差。
2. 更关键的是：当 scheduler 侧的有效长度跟踪尚未同步时，signal 会暂时偏向单调增长口径，
   与 runner 侧真实可用 block 容量不一致，导致 guard 误判。

这属于 strict 门禁误杀，不是实际退化到 full-history 的证据。

### 10.3 修复

1. 新增每请求 `scheduled_tokens` 解析（best-effort）：
   - `TriAttention_vLLM/triattention_v2/hook_impl.py:_scheduled_tokens_for_req`
2. guard token 口径增加“物理容量上限”保护：
   - 用 `req_state.block_ids * block_size` 推导 `block_capacity_hint`
   - 允许 1 个 block 的异步余量后再 clamp：
   - `effective_tokens = min(effective_tokens, block_capacity_hint + block_size)`
3. guard slack 保持为：
   - `block_size + estimated_slack + scheduled_tokens`
4. 新增回归测试覆盖 `16 + 1`、`16 + 2（含 estimate skew）`、`16 + 3 + 1` 场景：
   - `TriAttention_vLLM/tests_v2/test_hook_impl.py::test_effective_len_guard_allows_block_plus_one_overflow`
   - `TriAttention_vLLM/tests_v2/test_hook_impl.py::test_effective_len_guard_allows_block_plus_two_with_estimate_skew`
   - `TriAttention_vLLM/tests_v2/test_hook_impl.py::test_effective_len_guard_allows_block_plus_skew_plus_scheduled`

### 10.4 边界说明

此修复只放宽 strict guard 误杀，不改变 selector/compaction 的核心语义。  
若出现“真实回归到接近 full-history”，`effective_len_regression_ratio` 条件仍会触发 fatal。
3. 对齐报告口径易混淆：
   - `compare_results.py` 使用 any-correct；
   - 官方 `eval_math_multi` 使用 per-question draw 均值。
   - 需要在输出中强制同时打印并标注两种口径。
