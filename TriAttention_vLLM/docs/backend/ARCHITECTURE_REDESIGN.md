# TriAttention_vLLM V2 架构规格

- 更新时间：2026-02-13
- 状态：Active
- 适用范围：vLLM 0.15.x（V1 Engine）

---

## 1. 设计目标

V2 架构目标：

1. 保持对 vLLM **非侵入式**（不改上游源码）。
2. 将压缩策略与执行拆分，避免职责混乱。
3. 为后续功能预留稳定扩展空间（batch>1、prefill 裁剪、显存触发）。

---

## 2. 非目标（当前阶段不做）

1. 立即追求 Triton TopK/Gather。
2. 一步到位覆盖全部复杂场景。
3. 依赖临时 hack 作为长期方案。

---

## 3. 总体架构

```
Scheduler (TriAttentionScheduler)
  -> 产生触发决策/策略信号
  -> 输出给 Worker/Runner 路径

Worker (TriAttentionWorker)
  -> 初始化并注入 TriAttentionModelRunner

ModelRunner (TriAttentionModelRunner)
  -> 准备输入
  -> 执行模型前向
  -> 执行 KV 压缩动作（按策略信号）

Attention Backend (FlashAttention)
  -> 保持标准 attention 计算
```

---

## 4. 扩展点与接入方式

1. `--worker-cls triattention.worker.TriAttentionWorker`
2. `--scheduler-cls triattention.scheduler.TriAttentionScheduler`
3. `--attention-backend` 保持标准路径（V2 默认不承载压缩主逻辑）

约束：

- 不允许通过 monkey patch vLLM 源码路径实现核心能力。
- 不允许在 Attention.forward 中挂主压缩流程。

---

## 5. 组件职责（精确定义）

### 5.1 TriAttentionScheduler

输入：request 队列、KV cache usage、请求状态。
输出：调度结果 + 压缩触发信号（策略级）。

必须做：

1. 根据策略判断是否触发压缩。
2. 与 request 生命周期一致地推进/清理策略状态。

不能做：

1. 操作 KV tensor。
2. 实现算法打分与 token 选择。

### 5.2 TriAttentionModelRunner

输入：scheduler_output、模型状态、KV cache。
输出：标准 model output（并在内部完成压缩动作）。

必须做：

1. 根据 req_id 执行 request 级压缩状态管理。
2. 在正确时机执行 gather/score/select/scatter。
3. 保证输入准备流程一致性（positions/slot mapping/seq_lens）。

不能做：

1. 替代 scheduler 的全局策略决策。
2. 依赖 batch_idx/block_id 作为长期 request key。

### 5.3 TriAttentionWorker

必须做：

1. 用官方机制注入自定义 runner。
2. 维持与 vLLM worker 生命周期一致。

不能做：

1. 承担策略决策。
2. 承担算法逻辑。

---

## 6. 关键数据与状态模型

### 6.1 Request 级状态（最小集合）

每个 `req_id` 至少维护：

1. `current_cache_len`
2. `compression_count`
3. `last_compression_step`
4. `prefill_len`
5. `mode`（prefill 保护/可裁剪）

### 6.2 生命周期事件

1. `on_request_start(req_id)`：创建状态
2. `on_step(req_id)`：更新长度/触发判断
3. `on_request_finish(req_id)`：清理状态
4. `on_request_preempt(req_id)`：标记与策略处理
5. `on_request_resume(req_id)`：恢复并校验

---

## 7. 触发策略规范

### 7.1 长度触发

- 条件：`effective_len >= kv_budget + divide_length`

### 7.2 显存触发（Phase 2）

- 条件：`kv_usage >= trigger_threshold`
- 建议配套：`release_threshold`（避免抖动）

### 7.3 策略冲突处理

当多策略同时触发：

1. 先按显存安全优先级执行。
2. 同一步最多执行一次压缩动作。

---

## 8. prefill 行为规范

### 8.1 保护模式

- `protect_prefill=true`
- prefill tokens 不进入候选裁剪集合。

### 8.2 可裁剪模式

- `protect_prefill=false`
- prefill tokens 可进入候选集合。

要求：

1. 模式必须显式可配置。
2. 两种模式必须分别有测试与日志标识。

---

## 9. 错误处理与回退

1. 压缩失败时本步回退“不压缩”。
2. 不得让单次压缩异常直接中断整个推理请求。
3. 日志必须包含 req_id、layer、触发条件、失败原因。

---

## 10. 分阶段实施计划

### Phase 1：基础功能

交付：

1. Worker + Scheduler + Runner 扩展接入。
2. 单请求稳定压缩。
3. prefill 保护模式。

验收：

1. 功能可运行且无状态污染。
2. 关键日志可观测。

### Phase 2：功能扩展

交付：

1. batch>1 支持。
2. prefill 可裁剪模式。
3. KV usage 触发策略。

验收：

1. 多请求场景行为稳定。
2. 策略切换结果可复现。

### Phase 3：优化与鲁棒性

交付：

1. 性能优化项（按收益评估）。
2. 长序列压力与回归体系。

---

## 11. 可观测性要求

每次压缩事件至少记录：

1. req_id
2. step / layer
3. 触发原因（长度/显存）
4. 压缩前后长度
5. 模式（prefill 保护/可裁剪）
6. 执行耗时

---

## 12. 兼容与版本策略

1. 锁定 vLLM 目标版本（0.15.x）。
2. 对非公开接口依赖处必须集中封装，避免散落。
3. 升级 vLLM 时先跑兼容检查，再做功能开发。

---

## 13. 与历史方案关系

1. V1 方案保留为参考资产与回归基线。
2. V2 方案是当前唯一主线。
3. 历史细节追溯见：
   - `docs/archive/snapshots/2026-02-13/`
   - `docs/backend/reference/`

