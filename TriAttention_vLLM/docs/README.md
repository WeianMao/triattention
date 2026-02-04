# 文档维护指南

> **重要**：开发前请先阅读本文档，理解文档结构和维护原则。

---

## ⚠️ 废弃目录说明

| 目录 | 状态 | 说明 |
|------|------|------|
| `V0/` | **已废弃** | 旧版文档，仅保留历史参考，**不再维护** |
| `archive/` | 归档 | 历史文档归档 |

**新内容请维护在 `interface/` 和 `backend/` 目录中。**

---

## 文档架构

```
docs/
├── README.md                  # 本文档（入口，维护原则）
├── interface/                 # 用户接口层（面向项目负责人）
│   ├── PROJECT_GOAL.md        # 终极目标（只读）
│   ├── CURRENT_STATUS.md      # 当前状态（持续更新）
│   ├── PENDING_DECISIONS.md   # 待决策问题（决策后删除）
│   └── OPEN_ISSUES.md         # 待解决问题（解决后删除）
├── backend/                   # 后端层（面向开发同事）
│   ├── DESIGN_DECISIONS.md    # 已确定的设计决策
│   ├── DEVELOPMENT_PRINCIPLES.md  # 开发原则
│   ├── REVIEW_CHECKLIST.md    # 审查清单
│   └── reference/             # 参考资料（历史分析、技术细节）
├── V0/                        # ⚠️ 已废弃，不再维护
└── archive/                   # 历史归档
```

---

## 两层设计理念

### Interface 层（用户接口）
- **面向**：项目负责人（你）
- **目的**：快速了解项目状态、做出决策
- **原则**：保持精简，只展示需要关注的信息

| 文档 | 用途 | 维护频率 |
|------|------|----------|
| `PROJECT_GOAL.md` | 终极目标 | **不可修改** |
| `CURRENT_STATUS.md` | 最新进度 | 每次开发后更新 |
| `PENDING_DECISIONS.md` | 需要你决策的问题 | 决策后删除 |
| `OPEN_ISSUES.md` | 待解决的问题 | 解决后删除 |

### Backend 层（后端）
- **面向**：开发同事
- **目的**：提供开发规范、技术细节、历史记录
- **原则**：详尽但有组织

| 文档 | 用途 |
|------|------|
| `DESIGN_DECISIONS.md` | 已确定的设计决策（永久记录） |
| `DEVELOPMENT_PRINCIPLES.md` | 开发规范（所有同事必读） |
| `REVIEW_CHECKLIST.md` | Review 时使用的检查清单 |
| `reference/` | 技术分析、历史文档（按需查阅） |

---

## 三种工作场景

### 场景 1：继续执行任务

**同事需要做的**：
1. 阅读 `interface/CURRENT_STATUS.md` 了解当前进度
2. 阅读 `interface/OPEN_ISSUES.md` 了解待解决问题
3. 阅读 `backend/DEVELOPMENT_PRINCIPLES.md` 了解开发规范
4. 按优先级完成任务
5. **任务完成后**：
   - 从 `OPEN_ISSUES.md` 删除已解决的问题
   - 更新 `CURRENT_STATUS.md` 反映最新状态

### 场景 2：Review 项目状态

**同事需要做的**：
1. 运行测试，检查代码状态
2. 使用 `backend/REVIEW_CHECKLIST.md` 逐项检查
3. **发现新问题** → 添加到 `interface/OPEN_ISSUES.md`
4. **确认问题已解决** → 从 `OPEN_ISSUES.md` 删除
5. 更新 `interface/CURRENT_STATUS.md`

### 场景 3：需要用户决策

**同事发现需要决策的问题时**：
1. 添加到 `interface/PENDING_DECISIONS.md`
2. 提供选项和建议
3. **用户决策后**：
   - 从 `PENDING_DECISIONS.md` 删除该问题
   - 将决策结果添加到 `backend/DESIGN_DECISIONS.md`

---

## 维护原则

### 原则 1：及时清理
- 问题解决后 → **立即从 `OPEN_ISSUES.md` 删除**
- 决策完成后 → **立即从 `PENDING_DECISIONS.md` 删除**
- 不要让已完成的事项堆积

### 原则 2：持续更新状态
- `CURRENT_STATUS.md` 应反映**最新状态**
- 旧信息应被**替换**而非累积
- 每次开发后检查是否需要更新

### 原则 3：保持 Interface 精简
- Interface 层文档应**一眼能看完**
- 详细技术分析放在 `backend/reference/`
- 只在 Interface 中放必要信息

### 原则 4：Backend 详尽但有组织
- 技术细节、历史分析放在 `backend/reference/`
- `DESIGN_DECISIONS.md` 是永久记录，不删除内容
- 新的参考文档放在 `reference/` 目录下

### 原则 5：不重复
- 同一信息只在一个地方维护
- Interface 可以引用 Backend，但不复制内容

---

## 文档模板

### 新增待解决问题（OPEN_ISSUES.md）

```markdown
### N. 问题标题
- **位置**：文件路径:行号
- **影响**：简述影响
- **修复方案**：简述方案（如已知）
- **工作量**：预估时间
```

### 新增待决策问题（PENDING_DECISIONS.md）

```markdown
## 问题标题

### 背景
[问题描述]

### 选项
- **选项 A**：[描述] （工作量：xxx）
- **选项 B**：[描述] （工作量：xxx）

### 建议
[如有建议]
```

### 记录已确定决策（DESIGN_DECISIONS.md）

```markdown
### 决策 N: 标题
- **日期**：YYYY-MM-DD
- **问题**：[问题描述]
- **决策**：[选择的方案]
- **理由**：[决策理由]
- **状态**：待执行/已完成
```

---

## 检查清单：开发前

- [ ] 阅读本 README
- [ ] 阅读 `interface/CURRENT_STATUS.md`
- [ ] 阅读 `interface/OPEN_ISSUES.md`
- [ ] 阅读 `backend/DEVELOPMENT_PRINCIPLES.md`

## 检查清单：开发后

- [ ] 已解决的问题从 `OPEN_ISSUES.md` 删除
- [ ] `CURRENT_STATUS.md` 已更新
- [ ] 如有新问题，已添加到 `OPEN_ISSUES.md`
- [ ] 如有待决策问题，已添加到 `PENDING_DECISIONS.md`

---

## 快速导航

### Interface Layer（优先阅读）

| 想了解... | 阅读文档 |
|----------|---------|
| ⭐ **项目终极目标** | [interface/PROJECT_GOAL.md](interface/PROJECT_GOAL.md) |
| ⭐ **当前状态** | [interface/CURRENT_STATUS.md](interface/CURRENT_STATUS.md) |
| 待决策问题 | [interface/PENDING_DECISIONS.md](interface/PENDING_DECISIONS.md) |
| 待解决问题 | [interface/OPEN_ISSUES.md](interface/OPEN_ISSUES.md) |

### Backend Layer（开发参考）

| 想了解... | 阅读文档 |
|----------|---------|
| 已确定的设计决策 | [backend/DESIGN_DECISIONS.md](backend/DESIGN_DECISIONS.md) |
| 开发原则 | [backend/DEVELOPMENT_PRINCIPLES.md](backend/DEVELOPMENT_PRINCIPLES.md) |
| 审查清单 | [backend/REVIEW_CHECKLIST.md](backend/REVIEW_CHECKLIST.md) |

### Reference Materials（详细技术文档）

#### 实施规划
- [实施路线图](backend/reference/roadmap.md)
- [Phase1 状态报告](backend/reference/PHASE1_STATUS_REPORT.md)
- [总体路线图](backend/reference/ROADMAP.md)

#### 修复记录
- [Triton BF16 编译错误修复](backend/reference/fixes/BF16_TRITON_FIX.md) ⭐ **最新**
- [相位计算公式修正](backend/reference/fixes/RKV_EQUIVALENCE_FIX.md)
- [MLR 公式修正](backend/reference/fixes/MLR_FIX.md)
- [Triton-PyTorch 等价性修正](backend/reference/fixes/FP32_EQUIVALENCE_FIX.md)
- [序列长度同步修正](backend/reference/fixes/FIX_SEQ_LEN_SYNC.md)
- [位置索引修正](backend/reference/fixes/POSITION_INDICES_FIX.md)
- [Per-Request 隔离修正](backend/reference/fixes/PER_REQUEST_ISOLATION_FIX.md)
- [KV Cache 格式修正](backend/reference/fixes/KV_CACHE_FORMAT_FIX.md)
- [Kernel 接口变更](backend/reference/fixes/KERNEL_INTERFACE_CHANGES.md)
- [API 变更（Per-Request）](backend/reference/fixes/API_CHANGES_PER_REQUEST.md)
- [vLLM 集成审查](backend/reference/fixes/VLLM_INTEGRATION_REVIEW.md)

#### 实现总结
- [实现状态](backend/reference/summaries/IMPLEMENTATION_STATUS.md)
- [结构总结](backend/reference/summaries/STRUCTURE_SUMMARY.md)
- [快速开始](backend/reference/summaries/QUICK_START.md)
- [运行推理](backend/reference/summaries/RUNNING_INFERENCE.md)
- [调试总结](backend/reference/summaries/DEBUG_SUMMARY.md)
- [vLLM Hook 总结](backend/reference/summaries/VLLM_HOOK_SUMMARY.md)
- [Agent2 总结](backend/reference/summaries/AGENT2_SUMMARY.md)

#### 实现细节
- [Fill-in-Place 策略](backend/reference/implementation/fill_in_place.md)
- [数据结构](backend/reference/implementation/data_structures.md)
- [vLLM 集成](backend/reference/implementation/vllm_integration.md)
- [vLLM Hook 实现](backend/reference/implementation/vllm_hook_implementation.md)

#### 设计文档
- [算法设计](backend/reference/design/algorithm.md)
- [优化设计](backend/reference/design/optimization.md)

#### Phase1 文档
- [Phase1 README](backend/reference/phase1/README.md)
- [技术笔记](backend/reference/phase1/TECHNICAL_NOTES.md)

#### R-KV 对比分析
- [分析概述](backend/reference/r-kv-analysis/README.md)
- [需求覆盖对比](backend/reference/r-kv-analysis/Q1_requirement_coverage.md)
- [优缺点分析](backend/reference/r-kv-analysis/Q2_pros_cons_analysis.md)
- [可复用代码分析](backend/reference/r-kv-analysis/Q3_reusable_code.md)

---

*文档版本：6.0*
*创建日期：2025-01-30*
*更新日期：2026-02-02（基于 R-KV/vLLM/docs/weian_development/README.md 重构）*
