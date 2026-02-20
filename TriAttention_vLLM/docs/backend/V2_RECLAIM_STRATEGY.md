# V2 物理回收策略（半侵入继承层）

- 更新时间：2026-02-16
- 状态：Active
- 适用范围：vLLM 0.15.x（V1 Engine）

---

## 1. 背景与目标

当前 V2 已具备 request 级 in-place compaction（逻辑压缩）能力，但尚未完成“物理回收 block/page”闭环。  
目标是在不直接修改上游 vLLM 源码文件的前提下，实现可运行、可验证的回收路径。

本阶段采用“半侵入继承层”策略：

1. 不改 `site-packages/vllm/...` 源码。
2. 通过 `TriAttentionScheduler/TriAttentionWorker/TriAttentionModelRunner` 的继承与包装扩展关键行为。
3. 允许覆盖 vLLM 内部运行时契约（block_ids 同步、回收事件应用）。

---

## 2. 为什么“纯 hook 层”不够

以下约束来自 vLLM 真实实现（已在 `trivllm` 环境核对）：

1. Scheduler->Worker 的 `new_block_ids` 默认是 append 语义。  
   证据：`vllm/v1/core/sched/output.py:116`
2. Worker 对 running request 也是 append。只有 resumed from preemption 才 replace。  
   证据：`vllm/v1/worker/gpu_model_runner.py:1037`
3. `KVCacheManager` 公开接口没有“运行中 request 级局部 shrink/free”。  
   证据：`vllm/v1/core/kv_cache_manager.py:378`
4. BlockPool/Prefix-cache 设计对 block table 采用 append-only 假设。  
   证据：`vllm/v1/core/block_pool.py:46`

结论：仅在 runner hook 做 KV 重排，无法完整完成“回收后保持调度一致性”。

---

## 3. 方案定义：半侵入继承层

### 3.1 数据面（worker/runner）

1. 压缩执行后生成结构化 `block_reclaim` 事件（每组 `block_ids_after`）。
2. runner 本地先更新 worker 侧 `req_state.block_ids`（避免 worker 内视图滞后）。
3. 事件通过 `ModelRunnerOutput` side-channel 回传 scheduler。

### 3.2 控制面（scheduler/kv manager）

1. scheduler 在 `update_from_output()` 解析 `block_reclaim` 事件。
2. 将 `block_ids_after` 应用到 scheduler 侧 kv manager 内部 request->blocks 映射。
3. 对 tail removed blocks 调 `block_pool.free_blocks(...)` 归还 free pool。

### 3.3 范围与限制

1. Phase A 先做“尾部回收（truncate reclaim）”闭环，不追求一次性完成最优页整理。
2. 默认关闭，走实验开关，避免影响现有对齐实验主线。
3. `cache_positions` 与绝对位置一致性问题按阶段推进，不在 Phase A 一次性求全。

---

## 4. 与 fill_in_place 的关系

`docs/backend/reference/implementation/fill_in_place.md` 给出的是目标策略：  
“预算区 + overflow 区 + 幸存者回填 + 释放 overflow pages”。

当前半侵入策略是其工程化落地路径：

1. 先打通 “压缩事件 -> block_ids 收缩 -> block_pool 回收” 的闭环。
2. 在闭环稳定后，再升级到更严格的 fill-in-place 页级整理与边界优化。

---

## 5. 风险清单

1. **一致性风险**：worker 与 scheduler 若未同时更新 block_ids，会出现读错页/行为漂移。
2. **prefix cache 风险**：回收路径若不处理 cached-block 元数据，可能引入 hash 链不一致。
3. **语义风险**：仅回收而不处理长度语义，可能造成性能改善有限或触发频率异常。

---

## 6. 验收标准（本阶段）

1. 功能：压缩触发后可观测到 `block_reclaim` 事件，并完成 scheduler 侧回收。
2. 正确性：基础 smoke/单测通过；无 request 级崩溃或明显状态污染。
3. 兼容性：默认开关关闭时行为与当前 V2 主线一致。

---

## 7. 关联文档

1. `docs/backend/DESIGN_DECISIONS.md`
2. `docs/backend/V2_IMPLEMENTATION_BLUEPRINT.md`
3. `docs/interface/CURRENT_STATUS.md`
4. `docs/interface/OPEN_ISSUES.md`
5. `docs/backend/reference/implementation/fill_in_place.md`

