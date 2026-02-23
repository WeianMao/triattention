# V2 重构执行计划（方案重置后）

- 更新时间：2026-02-23
- 状态：Draft
- 适用范围：vLLM 0.15.x（V1 Engine）

> 本文是 `backend/V2_FINAL_ARCHITECTURE.md` 的落地计划，聚焦“怎么改代码、先改哪里、每步怎么验收”。
>
> 2026-02-23 调整：执行主线与阶段顺序已按 `interface/V2_SCHEME_ADJUSTMENT_2026-02-23.md` 更新（不再以 `per_layer` 为中间收敛目标）。

---

## 1. 本轮目标（重构阶段，不是继续补丁）

1. 把 V2 的复杂度从“worker 热路径 patch + 巨型 hook”迁回清晰边界。
2. 保持 HF 对齐优先，不通过改语义换性能。
3. 为 `per_head/per_layer_per_head` 共用同一架构打基础。
4. 明确将 decode 热路径改动与新增 metadata 压到最小集合（性能优先）。

---

## 2. 重构范围与非范围

### 2.1 本轮范围（要做）

1. 拆分 `hook_impl.py` 的职责（语义层 / 布局层 / 编排层）
2. 引入结构化计划对象（`KeepPlan` / `PlacementPlan` / `ReclaimEvent`）
3. 设计并落地 Runtime Input Adapter（先接入 runner，逐步替代 patch 热路径）
4. 将 `gpu_seq_len_patch.py` 从主逻辑降为兼容/调试路径

### 2.2 本轮非范围（先不做）

1. 新增 Triton TopK/Gather
2. 一次性解决 batch>1 全部边角场景
3. 新的算法语义实验（HF 语义以外）

---

## 3. 任务拆分（按依赖顺序）

## T1. 结构化计划对象（先做）

目标：

1. 把“选点结果”和“物理布局动作”从 dict 风格返回值里拆出来。

交付：

1. 新增数据结构（建议落位 `triattention_v2/layout_plan.py` 或 `models.py`）
   - `KeepPlan`
   - `PlacementPlan`
   - `ReclaimEvent`

验收：

1. 不改变现有功能开关行为；
2. `hook_impl.py` 内部开始使用结构化对象（至少局部）。

---

## T2. 拆分 `hook_impl.py`（语义层 vs 布局层）

目标：

1. 将 HF 打分/选点逻辑从 compaction/reclaim 执行逻辑中拆开。

建议拆分：

1. `selector_hf.py`（或 `hf_selector.py`）
   - 只做 HF strict scoring + selection
2. `layout_engine.py`
   - 只做 compaction / fill / reclaim plan
3. `hook_impl.py`
   - 负责 glue code（读取 req_state、调用 selector、调用 layout engine、组装返回事件）

验收：

1. `hook_impl.py` 行数明显下降；
2. selector 层可独立单测；
3. layout 层可独立单测。

---

## T3. Runtime Input Adapter（核心性能重构）

目标：

1. 不再让 `gpu_seq_len_patch.py` 承担长期 decode 热路径主逻辑。

做法：

1. 在 `TriAttentionModelRunner` 内维护每请求：
   - `absolute_progress`
   - `effective_cache_len`
   - `effective_slot_base`（或等价 delta）
2. 新建 `input_adapter.py`（建议）
   - 负责基于 scheduler_output + runner state 生成输入准备所需 `effective_*` 数据
3. 先让 adapter 输出与现 patch 兼容的最小数据；
4. 再逐步把 patch 改成“薄适配层”或移除热路径逻辑。
5. decode 每步禁止引入非必要 metadata 组装；优先复用持久状态增量。

验收：

1. decode 热路径 CPU/Python 开销显著下降（代码层面先看 `.item()/to(\"cpu\")` 热路径减少）；
2. `gpu_seq_len_patch.py` 不再包含主要语义推导逻辑；
3. 语义不回退（长度/slot mapping 口径保持正确）；
4. decode 每步新增 metadata 规模可解释且为最小必要集合。

---

## T4. Runtime Adapter 简化 + 目标模式状态一致性收敛

目标：

1. 不以 `per_layer` 为过渡目标，直接围绕 `per_head` / `per_layer_per_head` 收敛运行时语义与布局一致性。

内容：

1. 建立最小“状态一致性验证”测试（压缩后 keep 集合 / `seq_lens` / `slot_mappings` 一致）
2. Runtime Input Adapter 切到“压缩点更新持久状态 + decode 薄适配”新路径
3. 布局层启 low-move fill-hole + strict reclaim 主路径（目标模式）

验收：

1. 不再依赖 step-local patch-heavy override 主路径；
2. decode 热路径 CPU/Python 开销显著下降；
3. 状态一致性测试稳定通过，且问题可在分层边界内定位。

---

## T5. `per_head` / `per_layer_per_head` 接入（同架构）

目标：

1. 在不重写架构的前提下完成 `per_head` 与 `per_layer_per_head` 的主线验证与收敛。

要求：

1. 打分/选点严格 HF 语义；
2. 布局策略可插拔（允许跨层同步搬运 policy）；
3. 不把 `per_head` 特例硬塞回 `hook_impl.py` 大函数里。

验收：

1. 同一架构下稳定支持两种目标模式（`per_head` / `per_layer_per_head`）；
2. 代码边界清晰，不新增 patch-heavy 热路径。

---

## 4. 每阶段统一门禁（轻量但有效）

为避免再次“边改边跑偏”，每个任务合并前至少满足：

1. **代码门禁**
   - `python -m compileall TriAttention_vLLM/triattention_v2`
2. **结构门禁**
   - 新增逻辑是否落在目标模块（不是继续塞 `hook_impl.py` / patch）
3. **语义门禁**
   - HF selector 语义未改（除非文档明确写了决策）
4. **文档门禁**
   - `CURRENT_STATUS.md` / `OPEN_ISSUES.md` 状态同步

---

## 5. 风险与止损规则

1. 若某一步为了性能被迫改 HF 语义：立即停止，先记录到 `PENDING_DECISIONS.md`
2. 若某一步继续增加 `gpu_seq_len_patch.py` 热路径逻辑：视为偏离重构目标，需重新审视方案
3. 若 `hook_impl.py` 拆分后反而引入更多全局状态：回滚拆分策略，优先保证单一事实源

---

## 6. 负责人视角的里程碑（便于汇报）

1. M-Reset-1：方案定稿文档完成（本文 + `backend/V2_FINAL_ARCHITECTURE.md`）
2. M-Reset-2：`hook_impl.py` 拆分完成，结构化计划对象落地
3. M-Reset-3：状态一致性验证门禁落地（压缩后 keep/seq_lens/slot_mappings 一致）
4. M-Reset-4：Runtime Adapter 新路径接管主热路径（thin adapter + 持久状态）
5. M-Reset-5：`per_head` / `per_layer_per_head` 主线验证完成
