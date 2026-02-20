# V2 P0 修复计划（2026-02-17）

- 更新时间：2026-02-17
- 状态：In Progress（Step A/B 完成，Step C 主修复完成待实验验收）
- 适用范围：TriAttention_vLLM V2（vLLM 0.15.x）

---

## 1. 范围与边界

本轮只修复以下 P0：

1. P0-1：`freq_scale_sq` 来源语义修正（不再由 `q_abs_mean**2` 伪造）。
2. P0-2：Triton 打分强约束（对齐模式下禁止静默 fallback）。
3. P0-4：`per_head` 聚合语义朝 HF 约束收敛（先修当前实现里明确可修的差异）。

不在本轮修复：

1. 时序/Prefill 兼容项（原 P0-3）已按决策转入 P1：
   - 见 `docs/interface/OPEN_ISSUES.md` `[P1] 4.1`。

---

## 2. 执行步骤

### Step A（先修）`freq_scale_sq` 语义

目标：

1. 停止 `q_abs_mean -> freq_scale_sq` 映射。
2. 优先从模型 RoPE 配置推导频率缩放；不可得时使用保守降级（显式标注来源）。

落点：

1. `triattention/utils.py`
2. `triattention/compressor.py`（必要时）

验收：

1. 代码中不再存在 `q_abs_mean**2 -> freq_scale_sq` 的主路径。
2. 单测/冒烟通过。

### Step B（并行修）Triton 强约束

目标：

1. 在启用压缩且要求 Triton 的模式下，selector 不可用必须直接失败。
2. 禁止在 strict 模式进入 fallback 选点。

落点：

1. `triattention_v2/config.py`（增加开关）
2. `triattention_v2/hook_impl.py`
3. `evaluation/runner/vllm_triattention_v2_runner.py`（CLI/env 透传）

验收：

1. 缺 stats / selector 不可用时，运行报错退出而不是继续跑。
2. 现有单测更新后通过。

### Step C（可修部分）per_head 聚合语义

目标：

1. 在 `hf_aligned_global_per_head` 模式下，把当前可表达的聚合语义向 HF 约束收敛。
2. 增加对应测试，防止回退。

说明：

1. 若受限于当前统计张量粒度，无法一步做到数学完全同构，需要在文档里明确残余差异。

落点：

1. `triattention_v2/hook_impl.py`
2. `tests_v2/test_hook_impl.py`

验收：

1. 代码路径与测试能证明语义变化生效。
2. 文档明确“已修复部分”和“剩余差异”。

---

## 3. 验证与回归

1. 先跑：
   - `tests_v2/run_smoke.py`
2. 再跑：
   - `pytest TriAttention_vLLM/tests_v2 -q`
3. 如需快速功能验证：
   - 小样本 V2 runner（不覆盖历史输出目录）。

---

## 4. 风险控制

1. 严格保持“开关关闭时兼容”原则。
2. 所有报错路径使用可检索 marker，便于 unattended 运行定位。
3. 文档与代码同轮更新，避免状态漂移。

---

## 5. 执行进展（2026-02-17）

### 5.1 Step A 完成（P0-1）

1. 已移除 `q_abs_mean**2 -> freq_scale_sq` 主路径：
   - `triattention/utils.py`
2. `freq_scale_sq` 现逻辑：
   - 优先从模型 RoPE 配置推导；
   - 推导失败时显式回退为 `ones`（不再伪造为 query 统计）。
3. 新增回归测试（防回退）：
   - `tests_v2/test_utils_rkv_stats.py`
   - 断言 `freq_scale_sq != q_abs_mean**2`。

### 5.2 Step B 完成（P0-2）

1. 新增 strict 开关与环境透传：
   - `triattention_v2/config.py`
   - `evaluation/runner/vllm_triattention_v2_runner.py`
2. strict 模式下 selector 不可用/打分失败会 fail-fast：
   - `triattention_v2/hook_impl.py`
   - marker：`TRIATTN_FATAL_TRITON_SCORING_REQUIRED:*`
3. 单测已覆盖 strict 路径：
   - `tests_v2/test_hook_impl.py`
   - `tests_v2/test_v2_eval_runner.py`

### 5.3 Step C 主修复完成（P0-4）

1. 已实现 `hf_aligned_global_per_head` 的关键聚合语义：
   - attention-head 打分；
   - 按 KV group 层内 `max`；
   - 跨层 `mean`；
   - 每个 KV head 独立 topk。
2. 已修复头维适配：
   - 当 stats 头数与 runtime KV 头数不一致时，显式映射，不再隐式使用前几个头。
3. 新增回归测试：
   - `tests_v2/test_hook_impl.py::test_selector_hf_global_per_head_uses_attention_head_scores_and_group_max`
   - `tests_v2/test_hook_impl.py::test_selector_reduces_stats_heads_to_runtime_heads_for_legacy_path`
4. 当前剩余为实验验收项（full-run 与 token-level 对照），不再是实现阻塞项。

### 5.4 本轮回归结果

1. `conda run -n rkv python tests_v2/run_smoke.py`：
   - `smoke passed: 59 tests`
2. `conda run -n rkv python -m pytest -q tests_v2`：
   - `59 passed`
