# TriAttention_vLLM V2 方案总览

- 更新时间：2026-02-13
- 状态：Active
- 适用范围：vLLM 0.15.x（V1 Engine）

---

## 1. V2 目标

在 **不修改 vLLM 源码** 的前提下，实现可扩展、可维护的 KV 压缩体系：

1. 当前阶段先保证基础功能正确（单请求/低并发可用）。
2. 后续支持 batch>1、prefill 保护/裁剪、按显存压力触发压缩。
3. 避免 V1 方案中“Attention 层可见性不足”导致的状态与生命周期问题。

---

## 2. 为什么从 Attention 层迁移

历史实践表明，压缩逻辑放在 Attention 层会遇到结构性问题：

1. 请求生命周期不透明（无法稳定拿到 request identity）。
2. `seq_lens/slot_mapping` 由 runner 侧统一计算，Attention 层难以做一致修正。
3. 压缩触发策略与显存水位信息属于 scheduler 侧职责。

因此 V2 明确：**Attention 只做 attention 计算，不做 KV 管理决策**。

---

## 3. V2 非侵入式架构

### 3.1 扩展点

全部通过 vLLM 可配置扩展点注入：

1. `--worker-cls`: 注入 `TriAttentionWorker`
2. `--scheduler-cls`: 注入 `TriAttentionScheduler`
3. `--attention-backend`: 继续使用标准 FlashAttention（V2 默认不自定义压缩后置逻辑）

### 3.2 组件职责（强约束）

1. `TriAttentionScheduler`
- 决定“是否触发压缩”（可基于 KV usage / 固定策略）。
- 维护与 request 生命周期一致的策略状态。
- 不做具体 gather/scatter。

2. `TriAttentionModelRunner`
- 执行压缩动作（gather -> score -> select -> scatter）。
- 基于 request 维度维护压缩状态（key 必须是 req_id）。
- 负责与 input 准备流程保持一致性（positions/slot mapping/seq lens）。

3. `TriAttentionWorker`
- 负责替换默认 model runner 为 TriAttention runner。
- 不承载压缩策略本体。

4. `TriAttentionCompressor`（已有）
- 保持算法核心（打分、TopK 选择等）独立可测试。

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

### 4.4 失败降级

压缩异常时必须：

1. 不中断主推理流程。
2. 打结构化日志（request/layer/step）。
3. 将请求回退到“本步不压缩”路径。

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

1. 架构层面：不改 vLLM 源码，全部通过可配置扩展点接入。
2. 正确性层面：压缩行为与策略定义一致，生命周期无状态污染。
3. 维护层面：新同事可根据 `GUIDED_TOUR.md` 与本文件直接接手。

