# 常见工作流模式

> 学习如何组合命令完成复杂任务，提升开发效率

## 🎯 什么是工作流？

工作流是**一系列命令的组合**，用于完成特定的开发目标。Claude DMS3 提供了多种工作流模式，覆盖从规划到测试的完整开发周期。

**核心概念**：
- **工作流（Workflow）**：一组相关任务的集合
- **任务（Task）**：独立的工作单元，有明确的输入和输出
- **Session**：工作流的执行实例，记录所有任务状态
- **上下文（Context）**：任务执行所需的代码、文档、配置等信息

---

## 💡 Pattern 0: 头脑风暴（从0到1的第一步）

**⚠️ 重要**：这是**从0到1开发的起点**！在开始编码之前，通过多角色头脑风暴明确需求、技术选型和架构决策。

**适用场景**：
- 全新项目启动，需求和技术方案不明确
- 重大功能开发，涉及多个技术领域和权衡
- 架构决策，需要多角色视角分析

**流程**：话题分析 → 角色选择 → 角色问答 → 冲突解决 → 生成指导文档

### 模式 A：交互式头脑风暴（推荐）

**特点**：通过问答交互，逐步明确需求和决策

```bash
# 步骤 1：启动头脑风暴
/workflow:brainstorm:artifacts "
GOAL: 实现实时协作编辑平台
SCOPE: 支持100+用户同时在线，低延迟(<100ms)，冲突自动解决
CONTEXT: MVP阶段，3个月上线，团队5人（2前端+2后端+1全栈）
" --count 3

# 系统输出 Phase 0：自动收集项目上下文
# ✅ 分析现有代码库结构
# ✅ 加载相关文档
# ✅ 识别技术栈和依赖

# 系统输出 Phase 1：意图分析（2-4个问题）
# 【问题1 - 核心挑战】实时协作的主要技术挑战？
# a) 实时数据同步
# b) 可扩展性架构
# c) 冲突解决机制
# 【问题2 - 优先级】MVP阶段最关注？
# a) 功能完整性
# b) 用户体验
# c) 系统稳定性
# 请回答 (格式: 1a 2c)：

# 用户输入：1c 2b

# 系统输出 Phase 2：角色选择（推荐 count+2 个角色）
# 【角色选择】请选择 3 个角色参与头脑风暴
# a) system-architect (系统架构师) - 实时同步架构设计
# b) ui-designer (UI设计师) - 协作界面用户体验
# c) product-manager (产品经理) - 功能优先级决策
# d) data-architect (数据架构师) - 数据同步模型设计
# e) ux-expert (UX专家) - 多用户协作交互流程
# 请输入选择 (格式: 2acd 或 2a 2c 2d)：

# 用户输入：2acd

# 系统输出 Phase 3：角色问题（每个角色 3-4 个问题）
# ===== system-architect 问题 =====
# 【问题1】100+ 用户实时状态同步方案？
# a) Event Sourcing - 完整事件历史，易于回溯
# b) 集中式状态管理 - 简单直接，适合MVP
# c) CRDT - 自动冲突解决，分布式友好
# 【问题2】低延迟通信协议选择？
# ...
# 请回答：

# 用户输入：1c 2a 3b 4c

# 系统输出 Phase 4：冲突检测和解决
# 【冲突1】CRDT 与 UI 回滚期望冲突
# Background: system-architect 选择 CRDT，但 ui-designer 期望回滚UI
# a) 采用 CRDT，显示合并状态
# b) 切换到 OT 算法支持回滚
# c) 混合方案：CRDT + 本地撤销栈
# ...

# 系统输出 Phase 5：生成指导文档
# ✅ 生成 guidance-specification.md
# ✅ 记录所有决策和理由
# ✅ 标注冲突解决方案
# 📁 文件位置：.workflow/WFS-realtime-collab/.brainstorming/guidance-specification.md

# 步骤 2：查看生成的指导文档
cat .workflow/WFS-*//.brainstorming/guidance-specification.md
```

### 模式 B：自动并行头脑风暴（快速）

**特点**：自动选择角色，并行执行，快速生成多角色分析

```bash
# 步骤 1：一键启动并行头脑风暴
/workflow:brainstorm:auto-parallel "
GOAL: 实现支付处理模块
SCOPE: 支持微信/支付宝/银行卡，日交易10万笔，99.99%可用性
CONTEXT: 金融合规要求，PCI DSS认证，风控系统集成
" --count 4

# 系统输出：
# ✅ Phase 0: 收集项目上下文
# ✅ Phase 1-2: artifacts 交互式框架生成
# ⏳ Phase 3: 4个角色并行分析
#   - system-architect → 分析中...
#   - data-architect → 分析中...
#   - product-manager → 分析中...
#   - subject-matter-expert → 分析中...
# ✅ Phase 4: synthesis 综合分析
# 📁 输出文件：
#   - .brainstorming/guidance-specification.md (框架)
#   - system-architect/analysis.md
#   - data-architect/analysis.md
#   - product-manager/analysis.md
#   - subject-matter-expert/analysis.md
#   - synthesis/final-recommendations.md

# 步骤 2：查看综合建议
cat .workflow/WFS-*//.brainstorming/synthesis/final-recommendations.md
```

### 模式 C：单角色深度分析（特定领域）

**特点**：针对特定领域问题，调用单个角色深度分析

```bash
# 系统架构分析
/workflow:brainstorm:system-architect "API 网关架构设计，支持10万QPS，微服务集成"

# UI 设计分析
/workflow:brainstorm:ui-designer "管理后台界面设计，复杂数据展示，操作效率优先"

# 数据架构分析
/workflow:brainstorm:data-architect "分布式数据存储方案，MySQL+Redis+ES 组合"
```

### 关键点

1. **Phase 0 自动上下文收集**：
   - 自动分析现有代码库、文档、技术栈
   - 识别潜在冲突和集成点
   - 为后续问题生成提供上下文

2. **动态问题生成**：
   - 基于话题关键词和项目上下文生成问题
   - 不使用预定义模板
   - 问题直接针对你的具体场景

3. **智能角色推荐**：
   - 基于话题分析推荐最相关的角色
   - 推荐 count+2 个角色供选择
   - 每个角色都有基于话题的推荐理由

4. **输出物**：
   - `guidance-specification.md` - 确认的指导规范（决策、理由、集成点）
   - `{role}/analysis.md` - 各角色详细分析（仅 auto-parallel 模式）
   - `synthesis/final-recommendations.md` - 综合建议（仅 auto-parallel 模式）

5. **下一步**：
   - 头脑风暴完成后，使用 `/workflow:plan` 基于指导文档生成实施计划
   - 指导文档作为规划和实现的权威参考

### 使用场景对比

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| 全新项目启动 | 交互式 (artifacts) | 需要充分澄清需求和约束 |
| 重大架构决策 | 交互式 (artifacts) | 需要深入讨论权衡 |
| 快速原型验证 | 自动并行 (auto-parallel) | 快速获得多角色建议 |
| 特定技术问题 | 单角色 (specific role) | 专注某个领域深度分析 |

---

## 📋 Pattern 1: 规划→执行（最常用）

**适用场景**：实现新功能、新模块

**流程**：规划 → 执行 → 查看状态

### 完整示例

```bash
# 步骤 1：规划任务
/workflow:plan --agent "实现用户认证模块"

# 系统输出：
# ✅ 规划完成
# 📁 Session: WFS-20251106-123456
# 📋 生成 5 个任务

# 步骤 2：执行任务
/workflow:execute

# 系统输出：
# ⏳ 执行 task-001-user-model...
# ✅ task-001 完成
# ⏳ 执行 task-002-login-api...
# ...

# 步骤 3：查看状态
/workflow:status

# 系统输出：
# Session: WFS-20251106-123456
# Total: 5 | Completed: 5 | Pending: 0
```

**关键点**：
- `--agent` 参数使用 AI 生成更详细的计划
- 系统自动发现最新 session，无需手动指定
- 所有任务按依赖顺序自动执行

---

## 🧪 Pattern 2: TDD测试驱动开发

**适用场景**：需要高质量代码和测试覆盖

**流程**：TDD规划 → 执行（红→绿→重构）→ 验证

### 完整示例

```bash
# 步骤 1：TDD 规划
/workflow:tdd-plan --agent "实现购物车功能"

# 系统输出：
# ✅ TDD 任务链生成
# 📋 Red-Green-Refactor 周期：
#   - task-001-cart-tests (RED)
#   - task-002-cart-implement (GREEN)
#   - task-003-cart-refactor (REFACTOR)

# 步骤 2：执行 TDD 周期
/workflow:execute

# 系统会自动：
# 1. 生成失败的测试（RED）
# 2. 实现代码让测试通过（GREEN）
# 3. 重构代码（REFACTOR）

# 步骤 3：验证 TDD 合规性
/workflow:tdd-verify

# 系统输出：
# ✅ TDD 周期完整
# ✅ 测试覆盖率: 95%
# ✅ Red-Green-Refactor 合规
```

**关键点**：
- TDD 模式自动生成测试优先的任务链
- 每个任务有依赖关系，确保正确的顺序
- 验证命令检查 TDD 合规性

---

## 🔄 Pattern 3: 测试生成

**适用场景**：已有代码，需要生成测试

**流程**：分析代码 → 生成测试策略 → 执行测试生成

### 完整示例

```bash
# 步骤 1：实现功能（已完成）
# 假设已经完成实现，session 为 WFS-20251106-123456

# 步骤 2：生成测试
/workflow:test-gen WFS-20251106-123456

# 系统输出：
# ✅ 分析实现代码
# ✅ 生成测试策略
# 📋 创建测试任务：WFS-test-20251106-789

# 步骤 3：执行测试生成
/workflow:test-cycle-execute --resume-session WFS-test-20251106-789

# 系统输出：
# ⏳ 生成测试用例...
# ⏳ 执行测试...
# ❌ 3 tests failed
# ⏳ 修复失败测试...
# ✅ All tests passed
```

**关键点**：
- `test-gen` 分析现有代码生成测试
- `test-cycle-execute` 自动生成→测试→修复循环
- 最多迭代 N 次直到所有测试通过

---

## 🎨 Pattern 4: UI 设计工作流

**适用场景**：基于设计稿或现有网站实现 UI

**流程**：提取样式 → 提取布局 → 生成原型 → 更新

### 完整示例

```bash
# 步骤 1：提取设计样式
/workflow:ui-design:style-extract \
  --images "design/*.png" \
  --mode imitate \
  --variants 3

# 系统输出：
# ✅ 提取颜色系统
# ✅ 提取字体系统
# ✅ 生成 3 个样式变体

# 步骤 2：提取页面布局
/workflow:ui-design:layout-extract \
  --urls "https://example.com/dashboard" \
  --device-type responsive

# 系统输出：
# ✅ 提取布局结构
# ✅ 识别组件层次
# ✅ 生成响应式布局

# 步骤 3：生成 UI 原型
/workflow:ui-design:generate \
  --style-variants 2 \
  --layout-variants 2

# 系统输出：
# ✅ 生成 4 个原型组合
# 📁 输出：.workflow/ui-design/prototypes/

# 步骤 4：更新最终版本
/workflow:ui-design:update \
  --session ui-session-id \
  --selected-prototypes "proto-1,proto-3"

# 系统输出：
# ✅ 应用最终设计系统
# ✅ 更新所有原型
```

**关键点**：
- 支持从图片或 URL 提取设计
- 可生成多个变体供选择
- 最终更新使用确定的设计系统

---

## 🔍 Pattern 5: 代码分析→重构

**适用场景**：优化现有代码，提高可维护性

**流程**：分析现状 → 制定计划 → 执行重构 → 生成测试

### 完整示例

```bash
# 步骤 1：分析代码质量
/cli:analyze --tool gemini --cd src/auth \
  "评估认证模块的代码质量、可维护性和潜在问题"

# 系统输出：
# ✅ 识别 3 个设计问题
# ✅ 发现 5 个性能瓶颈
# ✅ 建议 7 项改进

# 步骤 2：制定重构计划
/cli:mode:plan --tool gemini --cd src/auth \
  "基于上述分析，制定认证模块重构方案"

# 系统输出：
# ✅ 重构计划生成
# 📋 包含 8 个重构任务

# 步骤 3：执行重构
/cli:execute --tool codex \
  "按照重构计划执行认证模块重构"

# 步骤 4：生成测试确保正确性
/workflow:test-gen WFS-refactor-session-id
```

**关键点**：
- Gemini 用于分析和规划（理解）
- Codex 用于执行实现（重构）
- 重构后必须生成测试验证

---

## 📚 Pattern 6: 文档生成

**适用场景**：为项目或模块生成文档

**流程**：分析代码 → 生成文档 → 更新索引

### 完整示例

```bash
# 方式 1：为单个模块生成文档
/memory:docs src/auth --tool gemini --mode full

# 系统输出：
# ✅ 分析模块结构
# ✅ 生成 CLAUDE.md
# ✅ 生成 API 文档
# ✅ 生成使用指南

# 方式 2：更新所有模块文档
/memory:update-full --tool gemini

# 系统输出：
# ⏳ 按层级更新文档...
# ✅ Layer 3: 12 modules updated
# ✅ Layer 2: 5 modules updated
# ✅ Layer 1: 2 modules updated

# 方式 3：只更新修改过的模块
/memory:update-related --tool gemini

# 系统输出：
# ✅ 检测 git 变更
# ✅ 更新 3 个相关模块
```

**关键点**：
- `--mode full` 生成完整文档
- `update-full` 适用于初始化或大规模更新
- `update-related` 适用于日常增量更新

---

## 🔄 Pattern 7: 恢复和继续

**适用场景**：中断后继续工作，或修复失败的任务

**流程**：查看状态 → 恢复 session → 继续执行

### 完整示例

```bash
# 步骤 1：查看所有 session
/workflow:status

# 系统输出：
# Session: WFS-20251106-123456 (5/10 completed)
# Session: WFS-20251105-234567 (10/10 completed)

# 步骤 2：恢复特定 session
/workflow:resume WFS-20251106-123456

# 系统输出：
# ✅ Session 恢复
# 📋 5/10 tasks completed
# ⏳ 待执行: task-006, task-007, ...

# 步骤 3：继续执行
/workflow:execute --resume-session WFS-20251106-123456

# 系统输出：
# ⏳ 继续执行 task-006...
# ✅ task-006 完成
# ...
```

**关键点**：
- 所有 session 状态都被保存
- 可以随时恢复中断的工作流
- 恢复时自动分析进度和待办任务

---

## 🎯 Pattern 8: 快速实现（Codex YOLO）

**适用场景**：快速实现简单功能，跳过规划

**流程**：直接执行 → 完成

### 完整示例

```bash
# 一键实现功能
/cli:codex-execute --verify-git \
  "实现用户头像上传功能：
  - 支持 jpg/png 格式
  - 自动裁剪为 200x200
  - 压缩到 100KB 以下
  - 上传到 OSS
  "

# 系统输出：
# ⏳ 分析需求...
# ⏳ 生成代码...
# ⏳ 集成现有代码...
# ✅ 功能实现完成
# 📁 修改文件:
#   - src/api/upload.ts
#   - src/utils/image.ts
```

**关键点**：
- 适合简单、独立的功能
- `--verify-git` 确保 git 状态干净
- 自动分析需求并完整实现

---

## 🤝 Pattern 9: 多工具协作

**适用场景**：复杂任务需要多个 AI 工具配合

**流程**：Gemini 分析 → Gemini/Qwen 规划 → Codex 实现

### 完整示例

```bash
# 步骤 1：Gemini 深度分析
/cli:analyze --tool gemini \
  "分析支付模块的安全性和性能问题"

# 步骤 2：多工具讨论方案
/cli:discuss-plan --topic "支付模块重构方案" --rounds 3

# 系统输出：
# Round 1:
#   Gemini: 建议方案 A（关注安全）
#   Codex: 建议方案 B（关注性能）
# Round 2:
#   Gemini: 综合分析...
#   Codex: 技术实现评估...
# Round 3:
#   最终方案: 方案 C（安全+性能）

# 步骤 3：Codex 执行实现
/cli:execute --tool codex "按照方案 C 重构支付模块"
```

**关键点**：
- `discuss-plan` 让多个 AI 讨论方案
- 每个工具贡献自己的专长
- 最终选择综合最优方案

---

## 📊 工作流选择指南

**核心区分**：从0到1 vs 功能新增
- **从0到1**：全新项目、新产品、重大架构决策 → **必须头脑风暴**
- **功能新增**：已有项目中添加功能 → **可直接规划**

```mermaid
graph TD
    A[我要做什么?] --> B{项目阶段?}

    B -->|从0到1<br/>全新项目/产品| Z[💡头脑风暴<br/>必经阶段]
    B -->|功能新增<br/>已有项目| C{任务类型?}

    Z --> Z1[/workflow:brainstorm:artifacts<br/>或<br/>/workflow:brainstorm:auto-parallel]
    Z1 --> Z2[⬇️ 生成指导文档]
    Z2 --> C

    C -->|新功能| D[规划→执行]
    C -->|需要测试| E{代码是否存在?}
    C -->|UI开发| F[UI设计工作流]
    C -->|代码优化| G[分析→重构]
    C -->|生成文档| H[文档生成]
    C -->|快速实现| I[Codex YOLO]

    E -->|不存在| J[TDD工作流]
    E -->|已存在| K[测试生成]

    D --> L[/workflow:plan<br/>↓<br/>/workflow:execute]
    J --> M[/workflow:tdd-plan<br/>↓<br/>/workflow:execute]
    K --> N[/workflow:test-gen<br/>↓<br/>/workflow:test-cycle-execute]
    F --> O[/workflow:ui-design:*]
    G --> P[/cli:analyze<br/>↓<br/>/cli:mode:plan<br/>↓<br/>/cli:execute]
    H --> Q[/memory:docs]
    I --> R[/cli:codex-execute]
```

**说明**：
- **从0到1场景**：创业项目、新产品线、系统重构 → 头脑风暴明确方向后再规划
- **功能新增场景**：现有系统添加模块、优化现有功能 → 直接进入规划或分析

---

## 💡 最佳实践

### ✅ 推荐做法

1. **复杂任务使用完整工作流**
   ```bash
   /workflow:plan → /workflow:execute → /workflow:test-gen
   ```

2. **简单任务使用 Codex YOLO**
   ```bash
   /cli:codex-execute "快速实现xxx"
   ```

3. **重要代码使用 TDD**
   ```bash
   /workflow:tdd-plan → /workflow:execute → /workflow:tdd-verify
   ```

4. **定期更新文档**
   ```bash
   /memory:update-related  # 每次提交前
   ```

5. **善用恢复功能**
   ```bash
   /workflow:status → /workflow:resume
   ```

---

### ❌ 避免做法

1. **⚠️ 不要在从0到1场景跳过头脑风暴**
   - ❌ 全新项目直接 `/workflow:plan`
   - ✅ 先 `/workflow:brainstorm:artifacts` 明确方向再规划

2. **不要跳过规划直接执行复杂任务**
   - ❌ 直接 `/cli:execute` 实现复杂功能
   - ✅ 先 `/workflow:plan` 再 `/workflow:execute`

3. **不要忽略测试**
   - ❌ 实现完成后不生成测试
   - ✅ 使用 `/workflow:test-gen` 生成测试

4. **不要遗忘文档**
   - ❌ 代码实现后忘记更新文档
   - ✅ 使用 `/memory:update-related` 自动更新

---

## 🔗 相关资源

- **快速入门**：[Getting Started](getting-started.md) - 5分钟上手
- **CLI 工具**：[CLI Tools Guide](cli-tools-guide.md) - Gemini/Qwen/Codex 详解
- **UI设计工作流**：[UI Design Workflow Guide](ui-design-workflow-guide.md) - UI设计完整指南
- **问题排查**：[Troubleshooting](troubleshooting.md) - 常见问题解决
- **完整命令列表**：查看 `index/all-commands.json`

---

**最后更新**: 2025-11-06

记住：选择合适的工作流模式，事半功倍！不确定用哪个？使用 `ccw` 询问 Command Guide！
