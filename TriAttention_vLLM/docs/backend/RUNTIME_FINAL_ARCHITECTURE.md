# TriAttention_vLLM V2 最终架构方案（重构定稿）

- 更新时间：2026-02-23
- 状态：Draft
- 适用范围：vLLM 0.15.x（V1 Engine）

> 目的：在不改变项目最终目标（HF 对齐 / 物理回收 / 高性能 / 非侵入式优先）的前提下，重置 V2 的实现边界，避免继续在补丁链上叠复杂度。
>
> 2026-02-23 补充：执行计划已明确收敛到 `per_head` / `per_layer_per_head` 两种目标模式；`per_layer` 不作为主线交付或中间态。具体落地调整见 `interface/V2_SCHEME_ADJUSTMENT_2026-02-23.md`。

---

## 1. 适用背景（为什么需要重构方案）

当前 V2 已验证了大量关键能力（调度触发、hook 压缩、reclaim 闭环、HF 打分路径、Triton 打分），但实现复杂度明显偏高，且出现如下结构性问题：

1. `worker` 热路径长期依赖 `gpu_seq_len_patch.py` 修正 `seq_lens/slot_mapping` 语义，decode 全程引入额外 CPU/Python 开销。
2. `hook_impl.py` 同时承担 HF 语义、压缩执行、reclaim、guard、debug 等职责，边界混杂。
3. `effective length / absolute progress / physical block state` 的事实源分散在 scheduler/runner/worker patch/hook，多处推导易漂移。

结论：需要从“继续补实现细节”切换为“重构方案边界”。

---

## 2. 不变约束（必须保留）

以下约束保持不变：

1. **HF 对齐优先**
   - 主线目标模式（`per_head` / `per_layer_per_head`）的打分与选点语义严格按 HF SpeckV。
2. **物理回收必须做**
   - 不是只做逻辑压缩。
3. **非侵入式优先**
   - 不修改上游 vLLM 源码文件；
   - 允许通过 `worker/scheduler/runner` 继承层实现“半侵入运行时契约”。
   - 若为满足 HF 对齐与 decode 性能需使用少量 monkey patch/函数替换，允许，但需集中管理且保持薄适配。
4. **Attention 保持纯计算**
   - 不在 Attention 层承载主压缩流程。
5. **decode 热路径极简**
   - decode 每步代码改动和 metadata 引入必须最小化；
   - 优先使用持久状态增量，不做重型 metadata 组装。

---

## 3. 新架构核心：三层分离

### 3.1 语义层（HF Selector Layer）

职责：只回答“应该保留哪些 token”。

输入：

1. 当前请求的有效 token 集合（逻辑视图）
2. HF 对齐配置（主线为 `per_head/per_layer_per_head`；`per_layer` 仅作历史/调试参考）
3. 打分统计与参数（stats、offset、normalize、aggregation 等）

输出：

1. `KeepPlan`（逻辑保留计划）
   - shared keep indices 或 per-head keep indices
   - 元信息：语义模式、预算、window/prefill 保护结果等

约束：

1. 严禁在该层修改 KV tensor 或 block ids。
2. 严禁在该层做物理回收决策。
3. 该层必须可独立对照 HF 验证（不依赖 vLLM worker patch）。

---

### 3.2 布局层（Layout + Reclaim Engine）

职责：把 `KeepPlan` 落成“怎么搬、怎么放、怎么回收”。

输入：

1. `KeepPlan`
2. 当前 request 的物理布局视图（block ids、group/layer 映射、block size）
3. 请求运行时状态（prefill/recent 语义状态）

输出：

1. `PlacementPlan`
   - 哪些 token 保留到哪些物理槽位（允许乱序）
   - 搬运统计（供性能观测）
2. `ReclaimEvent`
   - 各 group 的 `block_ids_after` / removed blocks
3. `cache_len_after`

关键约束：

1. **物理顺序允许乱序**（性能优先）
2. **任何策略语义不得依赖物理顺序**
   - recent window / prefill 保护 / 触发判断只能依赖显式状态
3. 低搬运是该层目标，不通过篡改 HF 语义换性能。

---

### 3.3 运行时适配层（Runtime Input Adapter）

职责：在 worker/runner 输入准备阶段显式提供正确的三套语义，不再让全局 patch 成为长期主逻辑。

必须显式维护的语义：

1. `absolute_progress`
   - 用于 `positions` / RoPE（单调增长）
2. `effective_cache_len`
   - 用于 `seq_lens`（attention 上下文长度）
3. `effective_slot_base`（或等价表示）
   - 用于 `slot_mappings`（写入位置）

实现目标：

1. 在 `TriAttentionModelRunner` / `TriAttentionWorker` 内形成明确的数据适配路径；
2. 尽量使用 GPU tensor 增量更新；
3. 逐步替代当前 `gpu_seq_len_patch.py` 的全局 monkey patch 热路径职责。
4. decode 每步只应用最小必要状态，不在热路径构建大体量临时 metadata。

说明：

1. 过渡期允许保留 patch 作为兼容路径；
2. 允许在压缩触发时更新 request-local 持久状态（“压缩点 hack”），以降低 decode 热路径复杂度；
3. 但 patch 不再作为最终主设计，且不应承载复杂推导。

---

## 4. 语义与布局解耦原则（避免再次走偏）

### 4.1 recent/prefill 的语义定义

V2 运行时状态需要显式维护：

1. `S`：稳定已吸收集合（长期保留语义）
2. `R`：recent 未吸收集合（自上次压缩后新增 decode token）
3. `P`：prefill 受保护集合（若开启保护）

当前有效集合：

1. `Active = P ∪ S ∪ R`

要求：

1. `recent window` 的保护语义来自 `R`（或其显式派生状态）
2. 不再通过“物理尾部/逻辑前缀排列”推导 recent

### 4.2 允许乱序的边界

以下前提满足时，布局层允许乱序：

1. K/V 成对移动；
2. HF 语义层不读取物理顺序；
3. recent/prefill/trigger 语义来自显式状态；
4. worker 输入适配层使用显式 `effective_*` 语义，而不是复用单一 `num_computed_tokens`。

---

## 5. `per_head` / `per_layer_per_head` 在新架构中的统一位置

### 5.1 统一原则

1. 差异只体现在 **语义层的 `KeepPlan` 生成**。
2. 布局层与运行时适配层复用同一框架。

### 5.2 `per_head` / `per_layer_per_head` 要求（固定）

1. 打分与选点严格按 HF `per_head` 语义。
2. 布局策略可采用“同一 `kv_head` 跨层同步搬运”的 policy（若与 HF 语义不冲突）。
3. 若布局优化与 HF 语义冲突，优先保持 HF 语义，布局层退回保守实现。

---

## 6. 模块职责重划分（相对当前实现）

### 6.1 Scheduler（保留）

继续负责：

1. 触发决策
2. 生命周期状态
3. `effective_len_tracker`（或其后续等价实现）

不新增：

1. 打分/选点
2. KV tensor 操作

### 6.2 Runner（瘦身为编排器）

负责：

1. 调用语义层生成 `KeepPlan`
2. 调用布局层执行 compaction/reclaim
3. 维护 request 运行时状态
4. 回传结构化事件给 scheduler
5. 通过 Runtime Input Adapter 生成 worker 输入所需 `effective_*` 语义

不负责：

1. 在一个函数里塞完所有 HF 语义 + reclaim + guard + debug 逻辑

### 6.3 `gpu_seq_len_patch.py`（降级为过渡适配层）

当前状态：主逻辑（问题）

目标状态：

1. 只作为过渡期兼容路径；
2. 最终由 Runtime Input Adapter 替代其热路径职责；
3. debug 校验能力可保留，但默认关闭且不进入热路径。

---

## 7. 为什么这个方案更简洁（相对现状）

1. **把复杂度放回“压缩触发时”**
   - decode 常规步只更新轻量状态，不再每步跑复杂 patch 逻辑。
   - 同时减少 decode 每步 metadata 组装与 Python 字典/映射处理。
2. **HF 对齐问题与性能问题可分开处理**
   - 语义层出错看 HF 对照；
   - 布局层慢看搬运/回收；
   - 运行时适配层慢看 worker 输入准备。
3. **`per_head` / `per_layer_per_head` 不再分别走一套工程路径**
   - 统一 `KeepPlan -> LayoutPlan -> ReclaimEvent` 链路。

---

## 8. 落地顺序（最终架构的分阶段实现，不是临时方案）

### Phase A：边界重构（先不追求功能新增）

1. 抽出 `KeepPlan` / `PlacementPlan` / `ReclaimEvent` 数据结构。
2. 拆分 `hook_impl.py`（语义层 vs 布局层）。
3. 定义 Runtime Input Adapter 接口（先可复用现 patch 结果）。

### Phase B：Runtime Adapter 简化 + 目标模式状态一致性收敛

1. 运行时适配层：从 step-local override 转向“压缩点持久状态 + decode 薄适配”
2. 布局层：低搬运 fill-hole + 物理回收路径稳定
3. 验收：状态一致性测试通过、decode 热路径 patch-heavy 逻辑显著收敛

### Phase C：`per_head` / `per_layer_per_head` 主线验证

1. 保持 HF strict 语义（仅目标模式）
2. 接入布局层策略（含跨层同步搬运 policy）
3. 验收：HF 对齐 + 性能/稳定性不回退

---

## 9. 与当前文档的关系（SSOT）

1. `interface/V2_OVERVIEW.md`
   - 保留为负责人视角总览，增加对本文件的指向。
2. `backend/ARCHITECTURE_REDESIGN.md`
   - 保留 V2 原始边界定义；本文件补充“方案偏航后的重构定稿”。
3. `interface/OPEN_ISSUES.md`
   - 需将“方案偏航类问题”与“实现 bug”分组，避免继续混写。

---

## 10. 成功判据（重构完成时）

1. decode 热路径不再依赖 patch-heavy Python 逻辑导致长期 GPU util 下滑；
2. HF 对齐语义可在语义层单独验证（与布局实现解耦）；
3. strict reclaim full-run 可稳定运行并产出可解释指标；
4. 新同事阅读 `V2_OVERVIEW.md + 本文 + OPEN_ISSUES.md` 能直接接手开发。
