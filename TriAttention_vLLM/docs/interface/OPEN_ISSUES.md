# OPEN ISSUES（V2）

- 更新时间：2026-02-13
- 状态：Active
- 适用范围：vLLM 0.15.x

---

## [P0] 1. V2 触发链路尚未实现
- 背景：V2 要求由 scheduler 侧决定何时压缩，runner 侧执行压缩。
- 影响：没有该链路就无法验证“显存触发压缩”主能力。
- 现状证据：`interface/V2_OVERVIEW.md` 已定义，但代码未落地。
- 下一步：实现 TriAttentionScheduler -> Runner 的触发信号传递与消费。
- 验收标准：可在日志中观测到“达到阈值 -> 触发压缩 -> 执行完成”的稳定流程。
- 状态：Open

## [P0] 2. 请求级状态生命周期尚未在 V2 代码闭环
- 背景：V1 历史问题证明 request state 处理是高风险点。
- 影响：状态污染会直接导致压缩策略错误或结果漂移。
- 现状证据：V2 文档已要求 req_id 作为唯一 key，但实现未提交。
- 下一步：在 runner/scheduler 路径实现 request start/update/finish/preempt 生命周期处理。
- 验收标准：长跑测试无跨请求状态污染；请求结束后状态可回收。
- 状态：Open

## [P0] 3. Phase 1 回归门禁缺失
- 背景：多人并行开发需要固定最小回归集。
- 影响：修改后可能破坏核心路径且无人感知。
- 下一步：建立 Phase 1 必跑用例（功能、稳定性、日志完整性）。
- 验收标准：PR 可自动/半自动执行并给出通过结论。
- 状态：Open

## [P1] 4. prefill 裁剪策略未落地
- 背景：V2 支持 `protect_prefill=false`，但 Phase 1 默认先保护。
- 影响：影响后续压缩率与策略实验。
- 下一步：在 Phase 2 引入 prefill 可裁剪模式并补齐测试。
- 验收标准：两种 prefill 模式可配置切换且行为可验证。
- 状态：Open

## [P1] 5. batch>1 行为验证缺失
- 背景：V2 明确需要支持 batch>1。
- 影响：不验证将导致线上并发场景风险。
- 下一步：补齐 batch>1 的请求映射、状态隔离、触发一致性测试。
- 验收标准：batch>1 下结果稳定，且无 request identity 混淆。
- 状态：Open

## [P2] 6. 进一步性能优化（TopK/Gather）
- 背景：当前阶段优先正确性，性能优化可后置。
- 影响：吞吐上限暂受限。
- 下一步：Phase 3 再评估是否需要 Triton TopK/Gather。
- 验收标准：有明确收益再实施，不强行提前优化。
- 状态：Open

