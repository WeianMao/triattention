# TriAttention_vLLM 文档维护规范（V2）

> 这是强约束规范。后续开发同事必须遵守。

---

## 1. 适用范围与目标

本规范适用于 `TriAttention_vLLM/docs/` 全部文档。
目标：

1. 防止新旧信息耦合导致误导。
2. 防止“重点缺失、细节堆积”。
3. 保证跨人协作时可稳定交接。

---

## 2. 角色与责任

- **实现同事**：代码变更后同步更新文档（同一 PR）。
- **评审同事**：按 `backend/REVIEW_CHECKLIST.md` 检查文档一致性。
- **任务负责人**：对 `PENDING_DECISIONS.md` 给出决策并关闭条目。

---

## 3. 文档级别与信息边界

### 3.1 Interface 层（面向负责人）

- 只写“当前可行动信息”。
- 不写冗长推导过程。
- 每条问题必须带：影响、下一步、验收标准。

### 3.2 Backend 层（面向开发）

- 写技术边界、模块职责、数据流、失败场景。
- 写“为什么这样设计”，但不重复 interface 状态。

### 3.3 Archive/Reference 层（面向追溯）

- 保存历史细节与深度分析。
- 不得覆盖当前结论。

---

## 4. 必填元信息（每个核心文档）

核心文档（interface + backend 顶层）必须在开头包含：

- `更新时间`（YYYY-MM-DD）
- `状态`（Draft / Active / Deprecated）
- `适用范围`（例如 vLLM 0.15.x）

---

## 5. 单一信息源规则（SSOT）

- 当前方案定义：`interface/V2_OVERVIEW.md`
- 当前状态：`interface/CURRENT_STATUS.md`
- 当前问题：`interface/OPEN_ISSUES.md`
- 待决策：`interface/PENDING_DECISIONS.md`
- 技术规格：`backend/ARCHITECTURE_REDESIGN.md`
- 决策日志：`backend/DESIGN_DECISIONS.md`

若其他文档与 SSOT 冲突，必须在 24 小时内修复。

---

## 6. 更新触发条件（必须同步）

出现以下任一情况，必须更新文档：

1. 架构边界改变（模块职责、数据流、扩展点）。
2. 新增或移除关键配置参数。
3. 问题状态变化（新问题、问题解决、优先级变化）。
4. 决策生效（从待决策进入已决策）。
5. 验收标准变化。

---

## 7. 信息写作约束（防止混乱）

- **当前态文档禁止写长历史**：历史放 `reference/` 或 `archive/`。
- **禁止重复结论**：同一结论只维护在 SSOT 文件。
- **禁止模糊措辞**：使用可验证表述（路径、模块、条件、阈值）。
- **禁止无主语待办**：每个 P0/P1 必须有下一动作与验收标准。

---

## 8. Open Issues 统一模板

```markdown
### [P0|P1|P2] 问题标题
- 背景：
- 影响：
- 现状证据：文件路径:行号（可多条）
- 下一步：
- 验收标准：
- 状态：Open / In Progress / Blocked
```

---

## 9. Pending Decisions 统一模板

```markdown
## 决策标题
- 背景：
- 选项A：
- 选项B：
- 推荐：
- 不决策风险：
- 截止时间：YYYY-MM-DD
- 状态：Pending / Decided
```

---

## 10. Design Decisions 统一模板

```markdown
### D-XXX 标题
- 日期：YYYY-MM-DD
- 问题：
- 决策：
- 影响范围：
- 替代方案：
- 状态：Accepted / Superseded / Deprecated
```

---

## 11. 清理与归档规则

- 已解决问题：从 `OPEN_ISSUES.md` 删除；必要历史放 `reference/`。
- 已决策事项：从 `PENDING_DECISIONS.md` 删除，并写入 `DESIGN_DECISIONS.md`。
- 大规模重写前：先做快照到 `archive/snapshots/YYYY-MM-DD/`。

---

## 12. PR 合并门禁（文档）

满足以下条件才允许合并：

1. 代码行为变更对应文档已更新。
2. `CURRENT_STATUS` 与 `OPEN_ISSUES` 无互相矛盾条目。
3. 关键结论可在 SSOT 中直接找到。
4. 新同事按 `GUIDED_TOUR` 可以完成接手。

---

*版本：1.0*
*生效日期：2026-02-13*
