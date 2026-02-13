# TriAttention_vLLM 设计决策日志（V2）

- 更新时间：2026-02-13
- 状态：Active
- 适用范围：vLLM 0.15.x

> 说明：本文件只保留仍然有效的关键决策和必要历史摘要。
> 详细历史请参考 `docs/archive/snapshots/2026-02-13/` 与 `docs/backend/reference/`。

---

## D-001：V2 主线采用非侵入式扩展
- 日期：2026-02-13
- 问题：如何避免高维护成本并支持多人并行开发。
- 决策：通过 `worker_cls + scheduler_cls` 扩展接入，不修改 vLLM 源码。
- 影响范围：整体架构与后续开发流程。
- 替代方案：直接 fork/修改 vLLM（拒绝）。
- 状态：Accepted

## D-002：压缩主流程不再放在 Attention 层
- 日期：2026-02-13
- 问题：Attention 层缺乏完整生命周期与调度信息，易产生状态问题。
- 决策：Attention 保持纯计算；压缩执行迁移至 runner，触发决策在 scheduler。
- 影响范围：TriAttentionImpl 不再作为主逻辑承载点。
- 替代方案：继续后置压缩于 forward（拒绝）。
- 状态：Accepted

## D-003：request state 唯一键是 req_id
- 日期：2026-02-13
- 问题：batch_idx/block_id 在多请求、抢占、重排场景下不稳定。
- 决策：所有持久状态必须按 req_id 管理。
- 影响范围：状态字典、日志、测试、诊断工具。
- 替代方案：使用 batch_idx 或 first_block_id（拒绝）。
- 状态：Accepted

## D-004：V2 分阶段推进
- 日期：2026-02-13
- 问题：需求复杂，无法一轮交付。
- 决策：
  1. Phase 1：基础功能与正确性。
  2. Phase 2：batch>1、prefill 可裁剪、显存触发。
  3. Phase 3：性能与鲁棒性优化。
- 影响范围：任务拆分、里程碑管理、验收策略。
- 状态：Accepted

## D-005：prefill 策略双模式支持，Phase 1 默认保护
- 日期：2026-02-13
- 问题：需要兼顾稳定性与后续策略实验。
- 决策：保留 `protect_prefill=true/false` 双模式能力；Phase 1 默认保护。
- 影响范围：策略配置、回归测试、实验可比性。
- 状态：Accepted（默认值仍由 `PENDING_DECISIONS.md` 最终确认）

## D-006：显存触发压缩由 Scheduler 决策
- 日期：2026-02-13
- 问题：KV usage 信息在 scheduler 侧最稳定。
- 决策：显存触发（阈值策略）在 scheduler；runner 只执行动作。
- 影响范围：触发链路、策略单测、可观测性。
- 状态：Accepted

## D-007：V1 方案定位为历史参考，不再作为主线
- 日期：2026-02-13
- 问题：V1 对项目有贡献但边界不适合后续主线扩展。
- 决策：保留 V1 资产用于参考与回归对比，主线迁移 V2。
- 影响范围：文档导航、任务分配、实现优先级。
- 状态：Accepted

---

## 历史摘要（保留）

### H-001：GQA 相关准确率异常曾是关键问题
- 结论：该问题已定位并修复，相关经验保留用于排障。
- 参考：`archive/snapshots/2026-02-13/interface/OPEN_ISSUES.md`

### H-002：V1 对“Attention 层后置压缩”可行性进行了充分探索
- 结论：验证了部分可行性，但不满足 V2 长期可维护边界。
- 参考：`archive/snapshots/2026-02-13/backend/ARCHITECTURE_REDESIGN.md`

