# TriAttention_vLLM V2 开发执行日志

- 更新时间：2026-02-17
- 状态：Active
- 适用范围：vLLM 0.15.x（V1 Engine）

---

## 1. 用途与边界

本文件用于记录“正在执行中的开发信息”，作为多人协作交接入口。  
当前结论仍以 SSOT 为准：

1. 方案定义：`interface/V2_OVERVIEW.md`
2. 当前状态：`interface/CURRENT_STATUS.md`
3. 当前问题：`interface/OPEN_ISSUES.md`
4. 技术决策：`backend/DESIGN_DECISIONS.md`

---

## 2. 2026-02-16（回收闭环与性能修复阶段）

### 2.1 本轮目标

1. 先记录并固化当前开发信息，避免“口头状态”丢失。
2. 在 V2 路线上继续推进“物理回收闭环”原型（半侵入继承层）。
3. 保持默认行为兼容（实验开关关闭时与当前主线一致）。

### 2.2 已确认事实（来自代码与运行现象）

1. V2 已具备 hook 路径的逻辑压缩（in-place compaction）与事件回传基础。
2. 仅逻辑压缩不足以稳定改善长跑性能；需要回收 request tail blocks，避免长期处理“越来越长的历史”。
3. vLLM 默认运行契约偏 append-only：仅 hook 层无法独立完成 scheduler/worker 双侧一致的回收闭环。
4. 已确认采用“半侵入继承层”路线（见 `backend/V2_RECLAIM_STRATEGY.md`）。

### 2.3 已接受的本轮实现范围

1. 新增实验开关：`TRIATTN_V2_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM`。
2. executor 支持传递 hook 结构化详情（`details`），避免 runner/scheduler 信息丢失。
3. hook 在 compaction 成功后可生成 `block_reclaim` 事件（含 group 级 `block_ids_after`）。
4. runner 负责透传 reclaim 事件到 `ModelRunnerOutput` side-channel。
5. scheduler 在 `update_from_output()` 侧消费 reclaim 事件并执行 tail block 回收（实验开关保护）。

### 2.4 本轮明确不做（避免范围失控）

1. 不一次性完成严格 fill-in-place 页级整理全量方案。
2. 不改上游 vLLM 源码文件（仅在 TriAttention 继承层实现）。
3. 不在本轮追求生产级最优吞吐，只先完成“可验证闭环 + 正确性”。

### 2.5 代码落点（已完成）

1. `triattention_v2/config.py`
2. `triattention_v2/executor.py`
3. `triattention_v2/hook_impl.py`
4. `triattention_v2/runner.py`
5. `triattention_v2/scheduler.py`
6. `tests_v2/test_config.py`
7. `tests_v2/test_executor.py`
8. `tests_v2/test_hook_impl.py`
9. `tests_v2/test_runner.py`
10. `tests_v2/test_scheduler.py`（新增）

### 2.6 验收口径

1. 开关关闭：行为与当前 V2 主线保持一致。
2. 开关开启：压缩后可观测 reclaim 事件，且 scheduler 侧实际执行回收。
3. 基础回归通过：配置解析、hook 输出契约、scheduler 事件应用单测通过。
4. 文档同步：`CURRENT_STATUS` 与 `OPEN_ISSUES` 状态不冲突。

### 2.7 交接提示

1. 若出现“worker 已改 block_ids、scheduler 未同步”必须优先修复一致性，禁止带病跑全量实验。
2. 若 reclaim 影响正确性，先关 `ENABLE_EXPERIMENTAL_BLOCK_RECLAIM` 回退到逻辑压缩模式。
3. 大规模实验前先跑 `tests_v2/run_smoke.py`，再跑小样本对齐实验，最后再上 full run。

### 2.8 本轮验证记录（2026-02-16）

1. 针对改动点执行定向 pytest（`trivllm` 环境）：
   - `test_config.py`、`test_executor.py`、`test_hook_impl.py`、`test_runner.py`、`test_scheduler.py`
   - 结果：`25 passed`
2. 执行 `tests_v2/run_smoke.py`（`trivllm` 环境）：
   - 结果：`smoke passed: 54 tests`
   - 注：日志中的 `TRIATTN_FATAL_TRITON_SCORING_REQUIRED` 为单测 `test_runner_raises_on_triton_required_failure` 的预期路径。

### 2.9 复查修复记录（2026-02-16 第二轮）

1. 修复 smoke 门禁漏测：`tests_v2/run_smoke.py` 已纳入 `tests_v2.test_scheduler`。
2. 修复实验入口缺口：`evaluation/runner/vllm_triattention_v2_runner.py` 新增
   `--enable-experimental-block-reclaim` 并写入
   `TRIATTN_V2_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM`。
3. 修复 reclaim 一致性策略：`triattention_v2/scheduler.py` 对
   `block_ids_after` 改为 fail-fast 校验（长度/重复/前缀一致性），不再静默兜底。
4. 验证：
   - `pytest TriAttention_vLLM/tests_v2 -q` -> `54 passed`；
   - 已终止旧 `hf_perhead_anchor` 任务并重新拉起修复后实验（新输出目录）。

---

## 3. 2026-02-17（P0 修复推进）

### 3.1 本轮新增记录要求

1. 将 `scheduled_tokens > 1` 的 prefill 兼容风险明确记录为 P1（先不修复）：
   - `docs/interface/OPEN_ISSUES.md` `[P1] 4.1`
2. 先固化 P0 修复计划，再按计划落地与回归。

### 3.2 已落地修复

1. P0-1：`freq_scale_sq` 来源语义修正
   - 文件：`triattention/utils.py`
   - 结果：移除 `q_abs_mean**2` 伪造路径；改为 RoPE 推导优先 + ones 回退。
2. P0-2：strict Triton 打分强约束
   - 文件：`triattention_v2/config.py`、`triattention_v2/hook_impl.py`、`evaluation/runner/vllm_triattention_v2_runner.py`
   - 结果：strict 模式下 selector 不可用/打分失败会 fail-fast。
3. 回归测试补齐
   - 新增：`tests_v2/test_utils_rkv_stats.py`
   - 冒烟纳入：`tests_v2/run_smoke.py`
4. 兼容性修复
   - `triattention_v2/signals.py` 增加 `from __future__ import annotations`，修复 Python 3.9 导入 `float | None` 失败问题。

### 3.3 本轮验证结果

1. `conda run -n rkv python tests_v2/run_smoke.py`
   - 结果：`smoke passed: 57 tests`
2. `conda run -n rkv python -m pytest -q tests_v2`
   - 结果：`57 passed`
3. 说明：冒烟日志中 `TRIATTN_FATAL_TRITON_SCORING_REQUIRED` 为预期单测路径（非线上异常）。

### 3.4 当前剩余

1. P0-4 已进入“实验验收”阶段：
   - 关键实现差异已修复；
   - 剩余工作是 full-run 与 token-level 对照验证。

### 3.5 P0-4 主修复进展（2026-02-17）

1. `triattention_v2/hook_impl.py` 已完成关键改造：
   - `hf_aligned_global_per_head` 路径改为 attention-head 打分后按 KV group 做层内 `max`，再跨层 `mean`；
   - 当 stats 头数与 runtime KV 头数不一致时，新增显式头维适配，避免隐式前缀头偏置。
2. 新增对齐回归用例：
   - `tests_v2/test_hook_impl.py::test_selector_hf_global_per_head_uses_attention_head_scores_and_group_max`
   - `tests_v2/test_hook_impl.py::test_selector_reduces_stats_heads_to_runtime_heads_for_legacy_path`
3. 回归结果更新：
   - `conda run -n rkv python tests_v2/run_smoke.py` -> `smoke passed: 59 tests`
   - `conda run -n rkv python -m pytest -q tests_v2` -> `59 passed`

---

## 4. 2026-02-18（P0 性能/可靠性修复推进）

### 4.1 本轮目标

1. 接通 Triton trig cache 分支（减少评分热路径开销）。
2. 新增“退化到近 full-history” fail-fast，禁止静默降级。
3. 将 `enforce_eager` 从硬编码改为可配置开关。

### 4.2 已落地改动

1. trig cache 接线：
   - `triattention/config.py` 增加 trig cache 配置项；
   - `triattention/compressor.py` 初始化并缓存 trig table；
   - `triattention/scoring.py` 在 round 对齐时走 cached kernel，非对齐自动回退。
2. full-history 退化防护：
   - `triattention_v2/config.py` 新增 regression guard 参数；
   - `triattention_v2/hook_impl.py` 新增 fail-fast 判定与 marker：
     - `TRIATTN_FATAL_TRITON_SCORING_REQUIRED:effective_len_regressed:*`
3. runner 可配置 eager：
   - `evaluation/runner/vllm_triattention_v2_runner.py` 新增 `--enforce-eager` 及相关 env 透传。
4. 测试补齐：
   - `tests_v2/test_config.py`
   - `tests_v2/test_v2_eval_runner.py`
   - `tests_v2/test_hook_impl.py`

### 4.3 回归结果

1. `conda run -n rkv python -m pytest -q tests_v2/test_config.py tests_v2/test_v2_eval_runner.py tests_v2/test_hook_impl.py`
   - 结果：`26 passed`
2. `conda run -n rkv python tests_v2/run_smoke.py`
   - 结果：`smoke passed: 67 tests`

### 4.4 strict 禁止降级（2026-02-18 补充）

1. `triattention_v2/runner.py`
   - 开启 compaction 时，compression executor 任意异常直接 fail-fast；
   - 开启 compaction 时，`should_compress` 下若出现非白名单 skip reason，直接 fail-fast。
2. `evaluation/runner/vllm_triattention_v2_runner.py`
   - strict 默认参数保持 `experimental_compaction=true`、`experimental_block_reclaim=true`、
     `require_triton_scoring=true`、`require_physical_reclaim=true`；
   - 启动前做强校验，配置不满足直接抛错。
3. 新增回归用例：
   - `tests_v2/test_runner.py::test_runner_strict_mode_raises_when_hook_missing`
   - `tests_v2/test_runner.py::test_runner_strict_mode_raises_on_executor_exception`
   - `tests_v2/test_v2_eval_runner.py` 新增 strict 参数断言与启动强校验断言。
4. 全量回归：
   - `conda run -n rkv python -m pytest -q tests_v2`
   - 结果：`67 passed`

### 4.5 trig cache 命中率修复（2026-02-18 继续）

1. 问题背景：
   - 原实现仅在 `round_start % divide_length == 0` 时才走 cached kernel；
   - decode 场景下 `round_start` 常受 prefill 偏移影响，导致长期不整除，实际高频落在 on-the-fly trig 路径。
2. 修复方案（数学等价）：
   - `triattention/scoring.py`：
     - 新增“基准轮次 + 残差角”路径：
       - 取 `base_round_start = floor(round_start / divide_length) * divide_length`；
       - 从 trig cache 取 `cos(base+offset)` / `sin(base+offset)`；
       - 用残差 `residual = round_start - base_round_start` 进行角加法修正，得到 `cos(round_start+offset)` / `sin(round_start+offset)`；
       - 将修正后的 trig 表传入 cached kernel。
   - `triattention/kernels/triton_scoring.py`：
     - `speckv_scoring` 增加 `trig_values` 参数（显式传入 `(cos_table, sin_table)`）；
     - `trig_values` 与 `trig_cache` 统一走 cached kernel 分支。
3. 测试补齐：
   - 新增 `tests_v2/test_scoring_trig_cache.py`：
     - 对齐 round_start 命中 cache；
     - 非对齐 round_start 命中“残差修正 cache”；
     - 超出 cache 范围回退 on-the-fly。
   - `tests_v2/run_smoke.py` 纳入新模块。
4. 回归结果：
   - `conda run -n rkv python -m pytest -q tests_v2/test_scoring_trig_cache.py tests_v2/test_runner.py tests_v2/test_v2_eval_runner.py`
   - 结果：`19 passed`
   - `conda run -n rkv python tests_v2/run_smoke.py`
   - 结果：`smoke passed: 70 tests`
   - `conda run -n rkv python -m pytest -q tests_v2`
   - 结果：`70 passed`

### 4.6 压缩路径继续优化（2026-02-18）

1. `hf_aligned_global_per_head` 路径内存峰值优化：
   - 文件：`triattention_v2/hook_impl.py`
   - 改动：group selector 支持 `layer_input_iter`，在 hook 中改为“逐层 gather -> 逐层打分聚合”流式输入，不再先构造并持有整组 `layer_inputs` dense 张量列表。
   - 影响：降低该路径压缩步骤的瞬时显存和分配压力，减少 Python 侧大对象列表持有。
2. 物理回收一致性门禁加固：
   - 文件：`triattention_v2/hook_impl.py`
   - 改动：当 `require_physical_reclaim=True` 且按长度计算应发生 block shrink 时，若实际 `removed_block_ids` 不足则 fail-fast（`TRIATTN_FATAL_TRITON_SCORING_REQUIRED:physical_reclaim_missing:*`）。
   - 影响：防止“逻辑压缩已执行但物理回收未生效”的隐性退化继续运行。
3. 回归与验证：
   - `conda run -n rkv python -m pytest -q tests_v2/test_hook_impl.py tests_v2/test_runner.py tests_v2/test_scoring_trig_cache.py` -> `25 passed`
   - `conda run -n rkv python tests_v2/run_smoke.py` -> `smoke passed: 70 tests`
   - `conda run -n rkv python -m pytest -q tests_v2` -> `70 passed`

### 4.7 打分热路径拷贝削减（2026-02-18）

1. 文件：`triattention_v2/kv_compaction.py`
   - `gather_request_k_dense` / `gather_request_kv_dense` 去掉末端 `.contiguous()`；
   - 返回 transpose 后 view，避免额外一次大张量拷贝。
2. 文件：`triattention/scoring.py`
   - `compute_scores_triton` 不再对 `key_states` 强制 `.contiguous()`，直接按 stride 传入 kernel wrapper。
3. 文件：`triattention/kernels/triton_scoring.py`
   - 移除 `K_rot must be contiguous` 断言，允许非 contiguous 输入（kernel 使用显式 stride 访问）。
4. 目的：
   - 在不改语义的前提下减少 scoring 前后的中间复制成本，降低压缩阶段额外带宽开销。
5. 回归结果：
   - `conda run -n rkv python -m pytest -q tests_v2/test_kv_compaction.py tests_v2/test_hook_impl.py tests_v2/test_scoring_trig_cache.py` -> `25 passed`
   - `conda run -n rkv python tests_v2/run_smoke.py` -> `smoke passed: 72 tests`
   - `conda run -n rkv python -m pytest -q tests_v2` -> `72 passed`

### 4.8 条件 keep-only compaction 快路径（2026-02-18）

1. 文件：`triattention_v2/kv_compaction.py`
   - `compact_request_kv_in_place` / `compact_request_kv_in_place_per_head` 新增参数
     `preserve_dropped_tokens`（默认 `True` 保持旧语义）。
   - 当 `preserve_dropped_tokens=False` 时，仅搬运 keep 集合到前缀区，不再构造/搬运整段 `[kept + dropped]` permutation。
2. 文件：`triattention_v2/hook_impl.py`
   - 新增 `_selected_keep_count(...)`；
   - 在每层 compaction 前按 block 数变化决定快路径是否可用：
     - 若 `after_required_blocks < before_required_blocks`（确定会物理 shrink）则启用 keep-only 快路径；
     - 否则保持旧模式，避免语义风险。
3. 测试：
   - `tests_v2/test_kv_compaction.py` 新增 keep-only 两条用例：
     - `test_compact_request_kv_in_place_keep_only_fast_path`
     - `test_compact_request_kv_in_place_per_head_keep_only_fast_path`
   - `tests_v2/test_hook_impl.py` 兼容新参数签名。
4. 回归结果：
   - `conda run -n rkv python -m pytest -q tests_v2/test_kv_compaction.py tests_v2/test_hook_impl.py tests_v2/test_runner.py` -> `32 passed`
   - `conda run -n rkv python tests_v2/run_smoke.py` -> `smoke passed: 74 tests`
   - `conda run -n rkv python -m compileall triattention_v2/kv_compaction.py triattention_v2/hook_impl.py tests_v2/test_kv_compaction.py` -> pass
5. 备注：
   - 本机环境下 `pytest -q tests_v2` 存在偶发长时间无输出卡住现象（非确定性，疑似环境/插件干扰），本轮以 `run_smoke.py + 定向 pytest + compileall` 作为可复现门禁。

### 4.9 P1-E 第一阶段：paged 分块打分接线（2026-02-19）

1. 文件：`triattention_v2/kv_compaction.py`
   - 新增 `gather_request_k_dense_range(...)`，支持按 token 区间从 paged KV 中提取 K 子段（含连续 block 零拷贝路径与非连续通用路径）。
2. 文件：`triattention_v2/hook_impl.py`
   - selector 新增 paged 路径：
     - `_compute_layer_scores_paged(...)` 通过 `gather_request_k_dense_range` 按 chunk 打分；
     - `_select_keep_indices` 支持 `(kv_cache, block_ids, block_size)` 输入；
     - 默认 selector 暴露 `_supports_paged=True`，hook 在该标记存在时不再先构造整段 `keys_dense`，改为 paged 分块评分。
   - 保留兼容：
     - 若 selector 非内置（测试桩/外部注入）且无 `_supports_paged`，仍走旧版 `keys_dense` 调用路径。
3. 测试补齐：
   - `tests_v2/test_kv_compaction.py` 新增：
     - `test_gather_request_k_dense_range_layout0`
     - `test_gather_request_k_dense_range_non_consecutive_blocks`
   - `tests_v2/test_hook_impl.py` 增加 selector paged 能力断言（`_supports_paged`）。
4. 回归结果（本轮可复现门禁）：
   - `timeout 300s conda run -n rkv python -m pytest -q tests_v2/test_kv_compaction.py` -> `14 passed`
   - `timeout 300s conda run -n rkv python -m pytest -q tests_v2/test_hook_impl.py` -> `14 passed`
   - `timeout 300s conda run -n rkv python -m pytest -q tests_v2/test_runner.py` -> `8 passed`
   - `timeout 300s conda run -n rkv python -m pytest -q tests_v2/test_scoring_trig_cache.py` -> `3 passed`
   - 合计关键回归：`39 passed`
5. 环境说明：
   - `run_smoke.py` 与全量 `pytest tests_v2` 在本机存在偶发“进程长时间活跃但无输出”的非确定性耗时问题；
   - 本轮以逐文件超时门禁保证可复现性与可交接性。

### 4.10 P1-E 补全：global per-head 走 paged 输入（2026-02-19）

1. 变更背景：
   - 4.9 已完成单层 selector 的 paged 分块打分；
   - 但 `per_head + hf_aligned_global_per_head` 的组内聚合调用点仍优先走 `layer_input_iter`（dense gather）。
2. 本轮改动：
   - 文件：`triattention_v2/hook_impl.py`
   - 在 hook 调用点新增分流：
     - 若 group selector 标记 `_supports_paged_group=True`，传 `layer_kv_iter`（`layer_idx, kv_cache, block_ids, block_size`）；
     - 仅当外部 selector 不支持 paged group 时，回退旧 `layer_input_iter` dense 路径。
   - group selector 本体保持双栈兼容（`layer_kv_iter` / `layer_input_iter`）。
3. 测试同步：
   - 文件：`tests_v2/test_hook_impl.py`
   - `test_hook_uses_hf_global_per_head_selector_once_per_group` 调整为同时兼容：
     - paged group 输入（`layer_kv_iter`）；
     - legacy dense 输入（`layer_input_iter`）。
   - 新增 `test_hook_uses_hf_global_per_head_selector_paged_group_when_supported`：
     - 显式断言 `_supports_paged_group=True` 时必须走 `layer_kv_iter`，禁止无声回退 dense。
4. 本轮回归（`dc` 环境）：
   - `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests_v2/test_hook_impl.py` -> `15 passed`
   - `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests_v2/test_kv_compaction.py` -> `14 passed`
   - `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests_v2/test_runner.py` -> `8 passed`
   - `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests_v2/test_scoring_trig_cache.py` -> `3 passed`
   - `python tests_v2/run_smoke.py` -> `smoke passed: 76 tests`
5. 结果：
   - V2 默认路径下，global per-head 组内选择已不再强制先构造整组 dense K；
   - paged 分块打分覆盖到 group 语义分支，进一步减少压缩阶段无谓搬运。

### 4.11 P0-A 补强：scheduler 侧 reclaim 缺失即 fail-fast（2026-02-19）

1. 问题：
   - 即使 worker 侧完成压缩，若事件链路丢失/缺失 `block_reclaim`，scheduler 侧不会缩短 `req_to_blocks`；
   - 这会导致“逻辑上已压缩，但 attention 仍可能按长 block 列表计算”的隐性退化风险。
2. 修复：
   - 文件：`triattention_v2/scheduler.py`
   - 新增一致性规则（仅在 `require_physical_reclaim=True` 生效）：
     - 按 `cache_len_after` 计算 `required_blocks`；
     - 若某些 group 明确应 shrink，但事件缺失 `block_reclaim` 或 `groups` 非法，直接 `RuntimeError`；
     - 若应 shrink 的 group 未出现在事件里，直接 `RuntimeError`；
     - 若应 shrink group 的 `kept_len != required_blocks`，直接 `RuntimeError`。
3. 测试：
   - 文件：`tests_v2/test_scheduler.py`
   - 新增 `test_scheduler_missing_block_reclaim_raises_when_shrink_expected`。
4. 回归结果：
   - `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests_v2/test_scheduler.py` -> `4 passed`
   - `python tests_v2/run_smoke.py` -> `smoke passed: 78 tests`
5. 影响：
   - P0-A“可见长度闭环”进一步收紧：当事件链路异常时，不再静默继续，而是立即暴露错误。

### 4.12 按 1/2/3/4 顺序推进（2026-02-19）

1. [1] P0-A 可观测性补齐（长跑验收前置）：
   - 文件：`triattention_v2/hook_impl.py`
     - 压缩结果新增字段：
       - `effective_tokens_before`
       - `budget_total`
       - `reclaimed_block_count`
   - 文件：`triattention_v2/runner.py`
     - 在压缩 `applied` 时输出 summary 日志：
       - `before/after/budget/reclaimed_blocks/reason`
     - 目的：8-shard 长跑时可直接从日志确认“压缩是否真实降低可见长度”。
2. [2] P1-E 继续收敛（paged 路径去全段 scores 物化）：
   - 文件：`triattention_v2/hook_impl.py`
   - 单层 selector 的 paged 分支改为“分块流式 top-k”：
     - 不再 `torch.cat` 全序列 chunk scores；
     - 直接按 chunk merge top-k（shared/per-head 均支持）。
   - 语义说明：
     - 对单层选择，`normalize_scores` 为 token 轴 z-score（仿射单调），不改变 top-k 排序，因此可跳过全段 normalize 物化而不改变排序语义。
3. [3] P1-F 基础优化（keep-only 只搬运必要 token）：
   - 文件：`triattention_v2/kv_compaction.py`
   - 新增 no-op/最小搬运逻辑：
     - prefix keep 直接返回（identity permutation）；
     - keep-only 路径只搬运 `src!=dst` 的 token；
     - per-head keep-only 同步改为按 `(head, token)` 稀疏搬运。
   - 测试新增：
     - `test_compact_request_kv_in_place_keep_only_prefix_noop`
     - `test_compact_request_kv_in_place_per_head_keep_only_prefix_noop`
4. [4] 环境稳定性治理（防“无输出长时间挂起”）：
   - 文件：`evaluation/dispatch/triattention_sharded_dispatch.py`
     - 新增 `--stall-timeout-minutes`；
     - shard log 长时间无增长时 fail-fast 终止并报错。
   - 文件：`tests_v2/run_lite_gate.py`
     - 新增稳定 lite 门禁脚本：
       - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
       - 分模块 pytest + smoke（带 timeout）。
5. 回归结果（`dc` 环境）：
   - `pytest -q tests_v2/test_runner.py` -> `8 passed`
   - `pytest -q tests_v2/test_hook_impl.py` -> `15 passed`
   - `pytest -q tests_v2/test_kv_compaction.py tests_v2/test_scheduler.py tests_v2/test_scoring_trig_cache.py` -> `23 passed`
   - `tests_v2/run_lite_gate.py` 过程通过并观测到 `smoke passed: 80 tests`（环境仍有偶发会话残留现象，见下一条）。
6. 风险备注：
   - 本机工具链在长命令会话上偶发“进程退出后会话未即时回收”现象；
   - 新增 `run_lite_gate.py` + dispatcher stall timeout 作为当前阶段的工程级规避措施。

### 4.13 Fill-in-place 终版 + global per-head 跨层流式聚合（2026-02-19）

1. Fill-in-place（keep-only）终版语义落地：
   - 文件：`triattention_v2/kv_compaction.py`
   - 共享模式（`compact_request_kv_in_place`, `preserve_dropped_tokens=False`）：
     - 从“前缀稳定压缩（会移动大量 prefix kept）”改为“空洞填充”：
       - prefix 内已保留 token 原地不动；
       - 仅将 tail 幸存 token 填入 prefix 空槽。
   - per-head 模式（`compact_request_kv_in_place_per_head`）同步改为逐 head 空洞填充。
   - 结果：更接近 `fill_in_place.md` 的“仅移动必要 token”目标，减少无效搬运。
2. global per-head 跨层聚合去全序列 `aggregated_scores`：
   - 文件：`triattention_v2/hook_impl.py`
   - 在 paged 分支中改为 chunk 流式聚合：
     - 每个 chunk 内遍历层并求和/平均；
     - chunk 级 top-k 与全局候选集做增量 merge；
     - 不再构造整段 `[H, T]` 的跨层聚合张量。
3. 测试补齐：
   - 文件：`tests_v2/test_kv_compaction.py`
   - 新增：
     - `test_compact_request_kv_in_place_keep_only_fill_holes_not_shift_prefix_kept`
     - `test_compact_request_kv_in_place_per_head_keep_only_fill_holes`
   - 旧用例按 fill-in-place 新语义更新断言：
     - `test_compact_request_kv_in_place_per_head_keep_only_fast_path`
4. 回归结果：
   - `pytest -q tests_v2/test_kv_compaction.py` -> `18 passed`
   - `pytest -q tests_v2/test_hook_impl.py tests_v2/test_runner.py tests_v2/test_scheduler.py tests_v2/test_scoring_trig_cache.py` -> `30 passed`
5. 重要说明（对齐风险提示）：
   - 后续 4.14 已将该风险修复为“全序列标准化语义（两遍流式）”。

### 4.14 global per-head 标准化严格对齐修复（2026-02-19）

1. 问题：
   - 4.13 的 paged 跨层流式聚合在 `sparse_normalize_scores=True` 下使用了 chunk 内标准化；
   - 与 HF / 既有 dense 路径的“全序列标准化”不严格等价，可能影响最终 top-k。
2. 修复：
   - 文件：`triattention_v2/hook_impl.py`
   - paged global per-head 路径改为“两遍流式”：
     - Pass-1：按层、按 chunk 累积全序列统计量（sum/sumsq）得到每层每头全序列 mean/std；
     - Pass-2：按 chunk 评分并应用全序列 mean/std，再做 guard、group-max、跨层平均与增量 top-k merge。
   - 语义对齐：
     - 顺序与 dense 路径一致：normalize -> guard -> group-max（HF aligned）。
3. 回归：
   - 新增 `tests_v2/test_hook_impl.py::test_selector_hf_global_per_head_paged_matches_dense_with_normalize`，
     断言 dense 与 paged 在 `sparse_normalize_scores=True` 下输出一致。
   - `tests_v2/test_hook_impl.py` -> `16 passed`
   - `tests_v2/test_kv_compaction.py` -> `18 passed`
   - `tests_v2/test_runner.py tests_v2/test_scheduler.py tests_v2/test_scoring_trig_cache.py` -> `15 passed`

### 4.15 paged streaming top-k 越界修复（2026-02-20）

1. 问题：
   - 线上 4 卡任务报错：
     - `RuntimeError: selected index k out of range`
     - 位置：`triattention_v2/hook_impl.py` paged streaming merge `torch.topk(...)`。
   - 触发机制：
     - 流式 chunk 合并早期，候选池长度 `< k`；
     - 代码仍以固定 `k=budget_total` 做 merge top-k，导致越界。
2. 修复：
   - 文件：`triattention_v2/hook_impl.py`
   - 三处 merge 逻辑统一改为：
     - `merge_k = min(k, merged_scores.shape[-1])`
     - `torch.topk(..., k=merge_k, ...)`
   - 覆盖路径：
     - 单层 paged shared merge；
     - 单层 paged per-head merge；
     - global per-head paged merge。
3. 回归测试：
   - 文件：`tests_v2/test_hook_impl.py`
   - 新增：
     - `test_selector_paged_streaming_merge_topk_clamps_k_to_current_pool`
   - 用例构造 `budget_total > 单个/前几块候选池长度`，验证不再抛越界且输出长度正确。
4. 当前验证状态：
   - 语法检查通过：`python -m compileall triattention_v2/hook_impl.py tests_v2/test_hook_impl.py`
   - 本机 `pytest` 进程存在间歇性 `D` 态（I/O 挂起）导致未能在本机完成该新增用例执行；需在稳定节点补跑 `tests_v2/test_hook_impl.py`。

### 4.16 性能慢点定位与修复（2026-02-20）

1. 主要慢点（确认）：
   - 文件：`triattention_v2/planner.py`
   - 问题：长度触发阈值固定为 `kv_budget + divide_length`，未考虑
     `protect_prefill=true 且 include_prefill_in_budget=false` 的有效预算语义。
   - 影响：压缩后 cache_len 仍可能高于固定阈值，导致“几乎每步都触发压缩”，
     出现 GPU 利用率周期性下降、整体吞吐显著变慢。
2. 触发阈值修复：
   - 文件：`triattention_v2/planner.py`
   - 变更：当 `protect_prefill=true && include_prefill_in_budget=false` 时，
     阈值改为 `kv_budget + divide_length + prefill_len`，与 compaction 预算语义一致。
3. 严格守卫误报修复（稳定性）：
   - 文件：`triattention_v2/hook_impl.py`
   - 问题：`effective_len_regressed` 在异步调度轻微时序偏差（+1 token 级）下会误报并终止任务。
   - 变更：守卫条件加入基于 `signal.estimated_cache_len` 的最小异步容差，
     保留对“接近 full history 回退”的 fail-fast 检测。
4. gather 热点轻量化（性能）：
   - 文件：`triattention_v2/kv_compaction.py`
   - 问题：`_consecutive_block_span` 在 tensor `block_ids` 下高频构造全长 `arange` + `torch.equal`，
     在压缩循环中重复触发，增加不必要的 kernel 与同步开销。
   - 变更：
     - 改为“首尾范围门控 + 相邻差分检查”；
     - 增加 span 缓存（按 tensor ptr/shape/device key）以复用结果。
5. 回归测试（本机通过）：
   - `tests_v2/test_planner.py::test_length_trigger_respects_prefill_outside_budget_mode`
   - `tests_v2/test_hook_impl.py::test_effective_len_guard_allows_one_step_async_slack`
   - `tests_v2/test_hook_impl.py::test_hook_fail_fast_when_effective_len_regresses_to_full_history`
   - `tests_v2/test_hook_impl.py::test_hook_clamps_effective_tokens_to_block_capacity_in_paged_path`
   - `tests_v2/test_kv_compaction.py::test_gather_request_k_dense_range_with_tensor_block_ids`
   - `tests_v2/test_kv_compaction.py::test_gather_request_k_dense_range_non_consecutive_blocks`
