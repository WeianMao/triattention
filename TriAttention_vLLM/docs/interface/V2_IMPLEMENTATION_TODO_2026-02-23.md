# V2 实施 TODO（本轮代码改造）

- 开始时间：2026-02-23
- 目的：把本轮代码改造过程、取舍和结果记录下来，方便后续接手
- 范围：围绕 `fill-hole + thin runtime adapter` 方案，对 V2 运行时路径做收敛和减复杂

## 本轮目标（按优先级）

1. 保持/提升 HF `per_head` / `per_layer_per_head` 语义对齐的实现基础
2. 降低 decode 热路径额外复杂度与时序脆弱性（尤其 runtime override 链）
3. 在不改 vLLM 源码目录前提下，优先做小而稳的收敛改造

## TODO 列表

- [x] 建立 request-local 持久状态语义，减少 `compression_events` 对 runtime override 构建的影响
- [x] 收敛 `effective_overrides` 的“本步压缩已应用”判断逻辑到 request state
- [x] 保持/补齐测试覆盖（`state`, `runner_state_updates`, `effective_overrides`）
- [x] 自检代码路径，确认 decode 热路径未新增不必要 metadata/重计算
- [x] 将 `compression_events` 从 runtime 输入适配主链路中降级（仅保留输出桥接/调试语义）
- [ ] 排查并修复 HF 不对齐问题（优先检查“slot/显存地址映射错位：应改 A 实改 B”）
- [ ] 必要时对 runtime patch / adapter 路径做定向重构（以正确性与 decode 性能为准，不被历史结构绑定）
- [ ] 项目阶段性完成后做 HF 对齐复核（不要预设“地址错位”就是根因；按现象→证据重新验证）
- [ ] 在真实 `vllm.v1` 运行路径做端到端复核（当前仅完成单测/冒烟门禁，尚未完成真实 runtime/HF 对齐验收）

## 工作记录

### 2026-02-23 / 初始化

- 新建本文件作为本轮实施记录。
- 先阅读 `runner.py / input_adapter.py / effective_overrides.py / input_patch_*` 与相关测试，确认切入点。
- 初步判断：`effective_overrides` 仍依赖 step-local `compression_events` 区分“本步压缩后长度语义”，这与最终方案中“压缩点更新持久状态 + decode 薄适配”的方向不完全一致，可作为本轮第一收敛点。

### 2026-02-23 / 第一轮代码收敛（已完成）

- 在 `RequestCompressionState` 中增加 `current_cache_len_semantics` / `current_cache_len_step`，显式记录当前 `current_cache_len` 的语义来源（scheduler 估计值 vs 压缩后 effective pre-step）。
- `consume_runner_signals()` 更新 cache 长度时同步写入 step（通过 `signal.step`）。
- `effective_overrides.py` 新增基于 request state 语义标记的有效 base 计算路径：
  - 优先使用 request-local 显式语义；
  - `compression_events` 仅作为兼容 fallback（用于旧测试替身对象或旧状态对象）。
- 目标收益：减少 decode 热路径对 step-local 事件列表语义的依赖，降低时序脆弱性。

### 2026-02-23 / 第二轮代码收敛（已完成）

- 进一步去除 `effective_overrides.py` 对 `compression_events` 的主逻辑依赖：
  - cache key 不再包含 `compression_events`；
  - 早退判断改为仅依赖 request-local 压缩状态；
  - 是否需要覆盖、以及有效 base 计算由 request state 语义标记决定。
- 对“已压缩请求但缺少状态语义标记”的情况改为 fail-fast（不再靠事件列表猜测）。
- 目标收益：继续收缩 step-local 时序耦合，让 Runtime Adapter 更接近“压缩点更新持久状态 + decode 薄适配”。

### 2026-02-23 / 待验证

- 运行定点测试：`test_state.py`、`test_runner_state_updates.py`、`test_effective_overrides.py`
- 若通过，再检查是否有其他调用方受 `update_cache_len(..., step=...)` 签名变化影响

### 2026-02-23 / 自检与验证结果（本轮）

- 已完成：
  - `python -m compileall` 检查本轮改动文件（代码与测试文件）通过
  - diff 自检：本轮改动未在 decode 热路径新增大块 metadata 构造；主要是把语义判断前移到 request state
  - `runner._needs_effective_input_overrides()` 去掉对 `_pending_compression_events` 的直接依赖，进一步收敛到 request-local 状态判断
  - 手动调用一组纯 Python 定点测试函数（`input_patch_ops` / `input_patch_vllm_backend` 新增护栏用例）通过
  - 手动验证新增 `idx_mapping` / `query_start_loc` 双重一致性护栏的 fail-fast 行为（定点函数）通过
  - 发现并确认可用测试环境：`trivllm`（Conda），`pytest 9.0.2` 可用，`vllm==0.15.0`
  - 在 `trivllm` 环境下分批执行本轮相关 pytest 定点测试并通过：
    - `test_state.py`
    - `test_runner_state_updates.py`
    - `test_effective_overrides.py`
    - `test_input_adapter.py`
    - `test_input_patch_ops.py`
    - `test_input_patch_vllm_backend.py`
    - `test_runner_output_bridge.py`
  - 在 `trivllm` 环境下运行 `tests_v2/run_smoke.py` 成功（`smoke passed: 148 tests`）
    - 说明：包含大量零参测试执行；需 pytest fixture 的用例按设计跳过
  - 在 `trivllm` 环境下补充执行更宽范围 pytest，并通过：
    - `test_runner.py`
    - `test_runner_compression_actions.py`
    - `test_kv_compaction.py`
    - `test_layout_engine.py`
    - `test_plan_models.py`
    - `test_selection_planner.py`
    - `test_hook_preflight.py`
    - `test_hook_runtime_context.py`
    - `test_hook_group_pipeline.py`
    - `test_dispatch_sharding.py`
    - `test_v2_eval_runner.py`
    - `test_hook_impl.py`
    - `test_config.py`
    - `test_effective_len_tracker.py`
    - `test_executor.py`
    - `test_planner.py`
    - `test_scheduler.py`
    - `test_scoring_trig_cache.py`
    - `test_utils_rkv_stats.py`
  - 在 `trivllm` 环境下执行 `tests_v2/run_lite_gate.py` 成功（串行关键 pytest + smoke）
- 未完成（环境限制）：
  - 默认环境（base）无 `pytest` 模块，不适合作为 V2 测试执行环境
  - `trivllm` 环境下直接导入 `vllm.v1.worker.gpu.*` 在 30s 超时内未完成（需在真实运行机/完整依赖下再做端到端 runtime 复核）
  - 仍未执行端到端 HF 对齐实验（本轮主要完成运行时正确性护栏与定点/冒烟/门禁验证）
- 风险提示：
  - `update_cache_len()` 签名新增 `step` 参数，但保留了默认值，旧调用方理论上兼容；仍建议后续在完整测试环境下跑一轮 `tests_v2`

### 2026-02-23 / 新增待办（负责人补充）

- 已知当前版本仍存在 HF 行为不对齐问题，优先怀疑 runtime 映射错误（示例症状：目标应写/改 A 槽位，但实际命中 B 槽位）。
- 本轮后续工作需将该问题纳入主线排查，不仅做架构收敛。
- 若排查中发现历史 patch/adapter 结构是主要阻碍，允许进行必要的定向重构（不以“少改代码”为唯一目标）。

### 2026-02-23 / 映射错位风险排查（进行中）

- 对 runtime `slot_mappings` 路径做代码审查后，先修复一个高风险点：
  - `shift_positions_from_sparse_deltas()` 原实现默认把 token-level delta 写入 `positions` 前缀（隐含 `query_start_loc == [0..N]` 的连续布局假设）。
  - 一旦未来/某些路径出现非连续 query token 布局，可能发生“应改 A 实改 B”的错位写入风险。
- 本轮修复策略：
  - 保留连续布局 fast path（不增加常规 decode 开销）；
  - 非连续布局使用显式 token index 应用 delta（避免前缀假设导致错位）；
  - 若检测到非单调/越界等异常布局，直接返回失败信号，由上层 fail-fast，避免部分错误写入。
- 新增一层更直接的护栏：
  - Runtime adapter 在激活 override 时同步记录“本步预期 req_state 行索引顺序”；
  - patch backend 在改写 `seq_lens/slot_mapping` 前先校验 `idx_mapping` 与预期完全一致，不一致直接 fail-fast。
- 备注：该修复是降低错位风险的防御性改造，不等于已证明它就是 HF 不对齐的唯一根因。

### 2026-02-23 / 开发原则落地（负责人强调：禁止静默降级）

- 已按要求将本轮新增的 runtime patch 关键路径改为 fail-fast：
  - sparse `seq_lens` base 覆盖未成功应用时直接报错；
  - sparse `slot_mapping` 位置位移构造失败时直接报错。
- 并进一步收紧为“部分命中也视为失败”：
  - 若 sparse 覆盖只命中部分请求行，视为潜在映射错位风险，不允许继续执行。
- 目的：避免 A/B 两套路径中 A 失败后悄悄降级到 B，导致优化是否生效不可观测。

### 2026-02-23 / 额外一致性护栏（已落地）

- `effective_overrides` 侧新增 fail-fast：
  - `compression_events` 显示本步已压缩，但 request state 却报告“无压缩请求”时直接报错（状态闭环不一致）。
  - 已压缩请求缺少 `current_cache_len` 语义标记时直接报错（不再靠事件列表猜）。
- runtime patch 侧新增 fail-fast：
  - `idx_mapping` 与 Runtime Adapter 在激活时记录的预期 req 行索引顺序不一致时直接报错；
  - `query_start_loc` 推导出的每请求 query 长度与 Runtime Adapter 记录的预期长度不一致时直接报错；
  - dense `effective_positions` shape 不匹配时直接报错。
- 执行路径新增 fail-fast：
  - Runtime override 已激活，但 patched 输入准备路径未实际消费 override 时直接报错（防止时序/路径偏差导致 override 没生效却继续跑）。

### 2026-02-23 / 收尾验收提醒（负责人新增）

- 即使中途定位过“疑似地址映射错位（想覆盖 A 实际覆盖 B）”，也不要把该说法当成既定事实。
- 在本轮/本阶段开发基本完成后，必须做一次面向结果的复核：
  - 当前实现与 HF `per_head` / `per_layer_per_head` 行为是否对齐；
  - 端到端是否仍有 bug（包含但不限于地址映射问题）；
  - 如发现不对齐，再回到证据链定位真实根因。

### 2026-02-24 / Runtime Adapter 接口瘦身（已完成）

- 将 `compression_events` 从 runtime 输入适配主链路接口移除：
  - `runner.py -> runner_output_bridge.py -> input_adapter.py` 不再为输入适配传递事件列表；
  - `compression_events` 继续保留在输出桥接（`execute_model` / `sample_tokens` side-channel）路径中。
- `effective_overrides.py` 保留 `compression_events` 可选参数（默认 `None`），仅用于定点测试/一致性校验，不再作为运行时输入适配的必需输入。
- 目的：
  - 继续降低 decode 热路径时序耦合；
  - 让 runtime adapter 更接近“request-local state 主导 + 薄适配”的最终方案。
- 回归验证（`trivllm` 环境）：
  - 定点 pytest 通过：`test_input_adapter.py`, `test_runner_output_bridge.py`, `test_runner.py`, `test_effective_overrides.py`
  - `run_lite_gate.py` 再次通过（含关键 pytest + `run_smoke.py`）

## 备注

- 若后续发现本轮范围过大，应优先完成“状态语义收敛 + 测试补齐”，其余优化留到下一轮。
