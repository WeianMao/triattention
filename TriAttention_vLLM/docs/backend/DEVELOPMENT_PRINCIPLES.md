# TriAttention_vLLM 开发原则（Runtime）

- 更新时间：2026-02-13
- 状态：Active
- 适用范围：vLLM 0.15.x

---

## 1. 总原则

1. **非侵入式优先**：不修改 vLLM 源码，优先使用 `worker_cls` / `scheduler_cls` / 插件机制。
2. **职责分离**：策略在 scheduler，执行在 runner，attention 保持纯计算。
3. **先正确后优化**：先确保状态生命周期与触发逻辑正确，再做性能优化。
4. **单一事实源**：实现边界以 `backend/ARCHITECTURE_REDESIGN.md` 为准。

---

## 2. 模块职责边界（强约束）

### 2.1 Scheduler（TriAttentionScheduler）

允许：

1. 压缩触发决策（长度阈值/显存阈值）。
2. request 生命周期相关策略状态。

禁止：

1. 直接操作 KV tensor。
2. 混入算法打分与 gather/scatter 实现。

### 2.2 ModelRunner（TriAttentionModelRunner）

允许：

1. 执行压缩（gather -> score -> select -> scatter）。
2. req_id 维度状态维护与清理。
3. 输入准备流程一致性处理（positions/slot mapping/seq lens）。

禁止：

1. 直接决定全局调度策略。
2. 使用 batch_idx/block_id 作为长期 request identity。

### 2.3 Attention Backend

允许：

1. 维持标准 attention 计算路径。

禁止：

1. 承载主要压缩流程。
2. 承担 request 生命周期管理。

---

## 3. 状态管理约束

1. request state 唯一键必须为 `req_id`。
2. 必须覆盖 `new / running / finished / preempted / resumed` 生命周期。
3. 任何临时 fallback 不得作为长期主逻辑。
4. 清理动作必须可验证（日志与断言）。

---

## 4. 触发策略约束

1. Phase 1：可先实现长度触发（`budget + divide_length`）。
2. Phase 2：加入 KV usage 触发。
3. 触发条件必须可配置，并在日志中可观测。
4. 触发策略实现必须可单测（不依赖真实大模型推理）。

---

## 5. prefill 策略约束

1. 必须支持 `protect_prefill=true/false` 两种模式。
2. Phase 1 默认只启用一种模式，另一种在 Phase 2 落地。
3. prefill 模式变化必须在 `PENDING_DECISIONS.md` 先拍板。

---

## 6. 错误处理与降级

1. 压缩失败不得直接中断请求主流程。
2. 出错时本步回退为“不压缩”，并记录结构化日志。
3. 对于重复失败请求必须有节流机制，避免日志风暴。

---

## 7. 测试策略（最低要求）

### 7.1 Phase 1 必测

1. 单请求压缩触发正确性。
2. request 生命周期无污染。
3. prefill 保护模式行为正确。
4. 压缩异常降级路径。

### 7.2 Phase 2 增测

1. batch>1 请求隔离。
2. prefill 可裁剪模式。
3. 显存触发策略正确性。

---

## 8. 文档同步要求

代码涉及以下变化时，必须同 PR 更新文档：

1. 架构边界变化。
2. 配置参数变化。
3. 问题状态变化。
4. 决策状态变化。

对应更新文件：

1. `interface/CURRENT_STATUS.md`
2. `interface/OPEN_ISSUES.md`
3. `backend/DESIGN_DECISIONS.md`

---

## 9. 禁止事项

1. 禁止将历史长叙事继续堆入 `CURRENT_STATUS.md`。
2. 禁止在多个文件重复维护同一“当前状态”。
3. 禁止以“临时调试开关”替代正式架构边界。

