# TriAttention_vLLM 接手导读（V2）

- 更新时间：2026-02-23
- 状态：Active

---

## 目标

让新同事在 20~30 分钟内完成接手，不需要先读历史长文。

---

## Step 1（5 分钟）：先看项目目标

阅读：`interface/PROJECT_GOAL.md`

你需要确认：

1. 项目终极目标是“与 HF SpeckV 行为对齐 + vLLM 工程化落地”。
2. 当前阶段优先级已明确为：
   - HF `per_head` / `per_layer_per_head` 对齐优先；
   - decode 性能（热路径极简）第二；
   - 工程侵入性/开发复杂度平衡第三。

---

## Step 2（5 分钟）：看 V2 方案

阅读：

1. `interface/V2_OVERVIEW.md`
2. `interface/V2_REFACTOR_EXECUTION_PLAN_2026-02-22.md`
3. `interface/V2_SCHEME_ADJUSTMENT_2026-02-23.md`

你需要确认：

1. V2 不再把压缩主逻辑放在 Attention 层。
2. V2 采用 worker/scheduler/runner 扩展点，保持对 vLLM 非侵入。
3. 当前主线目标模式是 `per_head` / `per_layer_per_head`（`per_layer` 非中间态）。
4. 当前方案调整强调“fill-hole + thin runtime adapter”，复杂度尽量放在压缩触发时。
5. decode 热路径代码改动与新增 metadata 必须最小化（性能约束）。
6. 新开发目录为 `triattention_v2/`，旧版只做参考。

---

## Step 3（5 分钟）：看当前状态与问题

阅读：

1. `interface/CURRENT_STATUS.md`
2. `interface/OPEN_ISSUES.md`
3. `interface/PENDING_DECISIONS.md`

你需要确认：

1. 当前阻塞是工程落位，不是算法公式。
2. 哪些问题是 P0，先做哪些。
3. 有哪些决策需要负责人拍板。

---

## Step 4（10 分钟）：看开发约束与技术规格

阅读：

1. `backend/DEVELOPMENT_PRINCIPLES.md`
2. `backend/ARCHITECTURE_REDESIGN.md`
3. `backend/DESIGN_DECISIONS.md`
4. `backend/V2_IMPLEMENTATION_BLUEPRINT.md`

你需要确认：

1. 每个模块职责边界（scheduler 决策 / runner 执行 / attention 保持纯计算）。
2. 禁止的实现方式（例如 request 标识使用 batch_idx 等）。
3. 本次任务涉及哪些决策与验收标准。

---

## 开发前自检

1. 我能复述 V2 主链路吗？
2. 我知道本次任务更新哪些文档吗？
3. 我知道完成后如何更新 `CURRENT_STATUS/OPEN_ISSUES` 吗？

如果以上任一问题回答不了，先不要写代码。
