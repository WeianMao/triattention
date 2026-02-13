# TriAttention_vLLM 文档系统（V2）

> 本文档是文档入口与治理规则摘要。
> 详细规则见 `docs/DOCS_STANDARDS.md`。

---

## 1. 文档目标

文档系统必须满足三件事：

1. **可接手**：新同事 30 分钟内搞清楚“目标、现状、问题、下一步”。
2. **可执行**：每个待办有明确 owner、动作、验收标准。
3. **可追溯**：决策与历史可以追踪，且不污染当前状态文档。

---

## 2. 文档分层

```
docs/
├── README.md                     # 入口（本文件）
├── DOCS_STANDARDS.md             # 文档维护规范（强约束）
├── interface/                    # 对外层：项目负责人/接手同事先读
│   ├── PROJECT_GOAL.md           # 终极目标（长期稳定）
│   ├── V2_OVERVIEW.md            # V2 方案总览（当前主线）
│   ├── CURRENT_STATUS.md         # 当前状态（滚动更新）
│   ├── OPEN_ISSUES.md            # 当前未解决问题（滚动更新）
│   ├── PENDING_DECISIONS.md      # 需要负责人拍板的问题
│   └── GUIDED_TOUR.md            # 快速接手导读
├── backend/                      # 开发层：技术实现与约束
│   ├── ARCHITECTURE_REDESIGN.md  # V2 架构规格（技术主文档）
│   ├── DESIGN_DECISIONS.md       # 关键决策日志（长期保留）
│   ├── DEVELOPMENT_PRINCIPLES.md # 开发原则（必读）
│   ├── REVIEW_CHECKLIST.md       # 评审清单
│   └── reference/                # 参考资料（历史分析、实现细节）
├── archive/                      # 归档（历史快照，不参与当前判断）
└── V0/                           # 已废弃目录（仅历史参考）
```

---

## 3. 单一信息源（必须遵守）

- **当前方案与边界**：`interface/V2_OVERVIEW.md`
- **当前进度**：`interface/CURRENT_STATUS.md`
- **当前问题**：`interface/OPEN_ISSUES.md`
- **待决策项**：`interface/PENDING_DECISIONS.md`
- **技术落地规范**：`backend/ARCHITECTURE_REDESIGN.md`
- **已确认决策**：`backend/DESIGN_DECISIONS.md`

禁止在多个文件重复维护同一状态结论。若冲突，以以上单一信息源为准。

---

## 4. 更新纪律（摘要）

- 代码行为变化后，**同一提交/同一 PR** 必须同步更新文档。
- `CURRENT_STATUS.md`、`OPEN_ISSUES.md` 必须保持“当前态”，不写长历史叙事。
- 历史细节放 `backend/reference/` 或 `archive/`，并在当前文档里用链接引用。
- 任何“已解决/已失效”内容应从当前态文档移除，避免堆积。

完整规则见 `docs/DOCS_STANDARDS.md`。

---

## 5. 快速接手路径（建议顺序）

1. `interface/PROJECT_GOAL.md`
2. `interface/V2_OVERVIEW.md`
3. `interface/CURRENT_STATUS.md`
4. `interface/OPEN_ISSUES.md`
5. `backend/DEVELOPMENT_PRINCIPLES.md`
6. `backend/ARCHITECTURE_REDESIGN.md`
7. `backend/DESIGN_DECISIONS.md`

---

## 6. 历史快照说明

为了避免信息丢失并减少主文档噪音，已保留阶段快照：

- `archive/snapshots/2026-02-13/interface/`
- `archive/snapshots/2026-02-13/backend/`

这些快照只用于追溯，不作为当前执行依据。

---

*版本：V2 文档体系*
*更新日期：2026-02-13*
