# TriAttention_vLLM V2 方案总览

- 更新时间：2026-02-23
- 状态：Active
- 适用范围：vLLM 0.15.x（V1 Engine）

---

## 1. V2 目标

在 **不修改 vLLM 源码目录** 的前提下，实现可扩展、可维护且可高性能收敛的 KV 压缩体系：

1. 当前阶段先保证基础功能正确（单请求/低并发可用）。
2. 后续支持 batch>1、prefill 保护/裁剪、按显存压力触发压缩。
3. 避免 V1 方案中“Attention 层可见性不足”导致的状态与生命周期问题。

> 2026-02-22 补充（方案重置）：在保留上述目标不变的前提下，当前实现已暴露“worker 热路径 patch 过重 / hook 职责过载 / 长度语义事实源分散”等结构性问题。  
> 新的最终架构定稿见 `backend/V2_FINAL_ARCHITECTURE.md`，后续重构执行计划见 `interface/V2_REFACTOR_EXECUTION_PLAN_2026-02-22.md`。
>
> 2026-02-23 补充（执行调整）：当前主线目标模式仅保留 `per_head` / `per_layer_per_head`；`per_layer` 不作为交付目标或中间收敛态。压缩主线采用低搬运 fill-hole，运行时允许使用薄 patch/adapter（不可继续 patch-heavy）。详见 `interface/V2_SCHEME_ADJUSTMENT_2026-02-23.md`。

### 1.1 当前拍板优先级（2026-02-23）

后续实现取舍按以下顺序判断：

1. **HF 对齐优先**：必须对齐 HF 的 `per_head` / `per_layer_per_head` 行为与逻辑。
2. **decode 性能次之**：decode 热路径尽量轻量，避免 CPU 拖 GPU；代码改动与额外 metadata 引入都应最小化。
3. **工程取舍第三**：在“不改 vLLM 源码目录”前提下平衡非侵入式、代码简洁性与开发复杂度。

说明：

1. 允许 monkey patch / 函数替换，但不是默认推荐；
2. 若少量 patch 明显更简洁且更有利于前两项目标，可以用；
3. 若多种方案效果相近，优先选择更规范、侵入更低方案。

---

## 2. 为什么从 Attention 层迁移

历史实践表明，压缩逻辑放在 Attention 层会遇到结构性问题：

1. 请求生命周期不透明（无法稳定拿到 request identity）。
2. `seq_lens/slot_mapping` 由 runner 侧统一计算，Attention 层难以做一致修正。
3. 压缩触发策略与显存水位信息属于 scheduler 侧职责。

因此 V2 明确：**Attention 只做 attention 计算，不做 KV 管理决策**。

---

## 3. V2 接入架构（非侵入式优先，但不教条）

### 3.1 扩展点

优先通过 vLLM 可配置扩展点注入：

1. `--worker-cls`: 注入 `TriAttentionWorker`
2. `--scheduler-cls`: 注入 `TriAttentionScheduler`
3. `--attention-backend`: 继续使用标准 FlashAttention（V2 默认不自定义压缩后置逻辑）

在可配置扩展点不足以满足前两优先级（HF 对齐、decode 性能）时：

1. 允许使用少量 monkey patch / 函数替换补足运行时语义适配；
2. 但仍禁止直接修改 `vLLM` 源码目录；
3. patch 必须尽量薄、集中管理，不再走 patch-heavy 热路径；
4. decode 每步新增 metadata 默认不引入，除非能证明是最小必要集合。

当前 class path：

1. `triattention_runtime.worker.TriAttentionWorker`
2. `triattention_runtime.scheduler.TriAttentionScheduler`

### 3.3 当前阶段说明（重要）

V2 当前代码已实现大量原型与实验能力，但并非最终稳定形态。尤其以下路径已被识别为“需重构”的过渡方案：

1. `triattention_runtime/gpu_seq_len_patch.py`（worker 输入准备热路径补丁）
2. `triattention_runtime/hook_impl.py`（职责过载的大型编排函数）

后续主线将采用 `backend/V2_FINAL_ARCHITECTURE.md` 定义的“三层分离”：

1. HF 语义层（selector）
2. 布局/回收层（layout engine）
3. 运行时输入适配层（runtime input adapter）

### 3.2 组件职责（强约束）

1. `TriAttentionScheduler`
- 决定“是否触发压缩”（可基于 KV usage / 固定策略）。
- 后续主线目标包含“显存未满不压缩、显存压力触发压缩”（本轮重构只要求保留该能力的实现便利性，不要求当轮交付）。
- 维护与 request 生命周期一致的策略状态。
- 不做具体 gather/scatter。

2. `TriAttentionModelRunner`
- 执行压缩动作（gather -> score -> select -> scatter）。
- 基于 request 维度维护压缩状态（key 必须是 req_id）。
- 负责与 input 准备流程保持一致性（positions/slot mapping/seq lens）。
- 运行时适配目标形态为“压缩点更新持久状态 + decode 薄适配”。
- decode 热路径优先使用最小状态增量，不做重型 metadata 组装。

3. `TriAttentionWorker`
- 负责替换默认 model runner 为 TriAttention runner。
- 不承载压缩策略本体。

4. `TriAttentionCompressor`（已有）
- 保持算法核心（打分、TopK 选择等）独立可测试。

代码落位：

- `TriAttention_vLLM/triattention_runtime/`（V2 新开发目录）

---

## 4. 明确的行为定义（避免歧义）

### 4.1 请求标识

- 必须使用真实 `req_id`（来自 scheduler/runner state）。
- 禁止使用 `batch_idx`、`block_id`、固定字符串作为长期 request key。

### 4.2 prefill 策略

V2 支持两种模式（默认先走保护模式）：

1. `protect_prefill=true`
- prefill token 不参与裁剪。

2. `protect_prefill=false`
- prefill token 允许参与裁剪（用于更激进压缩策略）。

### 4.3 触发策略

V2 允许两条触发线并存（先实现一条，再叠加）：

1. 长度策略：`seq_len >= budget + divide_length`
2. 显存策略：`kv_usage >= trigger_threshold`（由 scheduler 提供）

说明：

1. 当前重构阶段以“边界收敛、HF 对齐、decode 热路径降复杂度”为主；
2. 显存压力触发属于后续功能，当前方案要求为其保留清晰接入点，而非本轮必须完成。

### 4.4 失败降级

压缩异常时必须：

1. 不中断主推理流程。
2. 打结构化日志（request/layer/step）。
3. 将请求回退到“本步不压缩”路径。

### 4.5 `per_head` 语义开关（避免实现歧义）

1. `legacy_layer_local`
- 每层独立执行 per-head TopK；用于复现早期 V2 结果（含约 45% 的历史锚点实验）。

2. `hf_aligned_global_per_head`
- 先在同一 KV group 内做跨层聚合，再按 KV head 独立 TopK；
- 选出的同一组 per-head 索引应用到组内各层；
- 该模式用于 HF RKV-style `per_head` 对齐实验。

### 4.6 当前主线模式范围（2026-02-23 明确）

1. 主线交付与重构收敛仅围绕：
   - `per_head`
   - `per_layer_per_head`
2. `per_layer` 不作为主线中间态，不用于定义重构里程碑。

---

## 5. 分阶段实施（V2）

### Phase 1（基础功能）

目标：

1. 跑通非侵入式框架（worker+runner+scheduler 扩展注入）。
2. 单请求路径稳定。
3. prefill 保护模式可用。

不要求：

1. Triton TopK/Gather。
2. 高并发吞吐最优。
3. 完整显存回收闭环优化。

### Phase 2（能力扩展）

目标：

1. batch>1 稳定支持。
2. prefill 可裁剪模式。
3. 基于 KV usage 的触发策略上线。

### Phase 3（优化与鲁棒性）

目标：

1. 性能优化（必要时再评估 Triton TopK/Gather）。
2. 长上下文压力测试与回归体系。
3. 更细粒度的内存策略（含回收/复用策略优化）。

---

## 6. 成功标准

1. 架构层面：不改 vLLM 源码目录；优先使用可配置扩展点，必要时允许少量集中式 patch/替换。
2. 正确性层面：压缩行为与策略定义一致，生命周期无状态污染。
3. 维护层面：新同事可根据 `GUIDED_TOUR.md` 与本文件直接接手。
