# Usage Examples

Comprehensive examples of command-guide skill in action across all 5 operation modes.

---

## Mode 1: Command Search 🔍

### Example 1: Keyword Search

**User**: "搜索包含 'execute' 的命令"

**Skill Response**:
```
找到 4 个相关命令：

1. **cli:execute** (高级)
   描述: Auto-execution of implementation tasks with YOLO permissions
   参数: [--agent] [--tool codex|gemini|qwen] [--enhance] description or task-id
   场景: implementation

2. **task:execute** (中级)
   描述: Execute tasks with appropriate agents and context-aware orchestration
   参数: task-id
   场景: implementation

3. **workflow:execute** (中级)
   描述: Coordinate agents for existing workflow tasks with automatic discovery
   参数: [--resume-session="session-id"]
   场景: implementation

4. **codex-execute** (高级)
   描述: Automated task decomposition and execution with Codex
   参数: task description or task-id
   场景: implementation

💡 相关命令: workflow:plan, task:create
📖 详细文档: [Implementation Commands](workflow-patterns.md#implementation)
```

---

### Example 2: Category Search

**User**: "cli 类别下有哪些命令？"

**Skill Response**:
```
CLI 类别包含 9 个命令：

**核心命令** (6个):
- cli:analyze - Quick codebase analysis
- cli:chat - Direct codebase interaction
- cli:execute - Auto-execution with YOLO
- cli:cli-init - Initialize CLI tool configs
- cli:codex-execute - Codex-powered execution
- cli:discuss-plan - Multi-model planning discussion

**模式命令** (3个):
- cli:mode:plan - Project planning mode
- cli:mode:code-analysis - Deep code analysis mode
- cli:mode:bug-diagnosis - Bug diagnosis mode

📚 使用指南: [CLI Tools Guide](cli-tools-guide.md)
```

---

### Example 3: Use-Case Search

**User**: "planning 相关的命令有哪些？"

**Skill Response**:
```
Planning 场景包含 8 个命令：

**核心规划**:
1. workflow:plan (高级) - 5阶段规划工作流
2. cli:mode:plan (高级) - 项目规划和架构分析
3. workflow:tdd-plan (高级) - TDD 规划工作流

**任务分解**:
4. task:breakdown (中级) - 智能任务分解
5. workflow:tools:task-generate (中级) - 生成任务 JSON

**架构设计**:
6. workflow:brainstorm:system-architect (中级) - 系统架构师头脑风暴
7. workflow:brainstorm:api-designer (中级) - API 设计师头脑风暴

**评估验证**:
8. workflow:action-plan-verify (中级) - 规划质量验证

💡 常见工作流: Plan → Verify → Execute
📖 模式参考: [Workflow Patterns](workflow-patterns.md#planning-workflow)
```

---

## Mode 2: Smart Recommendations 🤖

### Example 1: After Planning

**User**: "执行完 /workflow:plan 后下一步做什么？"

**Skill Response**:
```
推荐的后续步骤：

**🎯 主要选择**:
1. **/workflow:execute** - 开始执行规划的任务
   ✅ 这是最常见的下一步
   作用: 实际执行 workflow:plan 生成的任务

**📋 可选步骤**:
2. **/workflow:action-plan-verify** - 验证规划质量
   作用: 在执行前检查规划的一致性和完整性
   建议: 重要项目建议先验证

3. **/workflow:status** - 查看规划状态
   作用: 查看生成的任务和当前进度

**📚 完整工作流模式**:
Plan → Verify (可选) → Execute → Review

详见: [Planning Workflow Pattern](workflow-patterns.md#planning-execution)
```

---

### Example 2: After Execution

**User**: "workflow:execute 执行完成后，推荐做什么？"

**Skill Response**:
```
推荐的后续步骤：

**🎯 主要选择**:
1. **/workflow:review** - 审查实现结果
   ✅ 检查代码质量、安全性、架构
   参数: [--type=quality|security|architecture]

**📊 可选步骤**:
2. **/workflow:status** - 查看执行状态
   作用: 确认所有任务都已完成

3. **/workflow:test-gen** - 生成测试
   作用: 为实现的功能创建测试用例

4. **/workflow:session:complete** - 完成会话
   作用: 标记工作流已完成，归档经验

**🔄 如果需要修改**:
- /task:replan - 重新规划某个任务
- /workflow:resume - 恢复工作流

详见: [Execution Review Pattern](workflow-patterns.md#execution-review)
```

---

## Mode 3: Full Documentation 📖

### Example 1: Command Parameters

**User**: "/workflow:plan 的参数是什么？"

**Skill Response**:
```
# workflow:plan

**描述**: Orchestrate 5-phase planning workflow with quality gate, executing commands and passing context between phases

**参数**:
```bash
/workflow:plan [--agent] [--cli-execute] "text description"|file.md
```

**参数说明**:
- `--agent` (可选): 使用 agent 模式执行规划
- `--cli-execute` (可选): 允许在规划阶段执行 CLI 命令
- `"text description"`: 任务描述文本
- `file.md`: 或者提供任务描述文件路径

**使用场景**: planning
**难度级别**: 高级

**使用示例**:
```bash
# 基本用法
/workflow:plan "实现用户认证功能"

# 使用 agent 模式
/workflow:plan --agent "重构支付模块"

# 从文件读取
/workflow:plan --agent feature-spec.md

# 允许 CLI 执行
/workflow:plan --cli-execute "添加搜索功能"
```

**执行流程**:
1. 上下文收集
2. 概念规划
3. 实现规划
4. 任务生成
5. 质量验证

**相关命令**:
- workflow:execute - 执行规划的任务
- workflow:action-plan-verify - 验证规划质量
- workflow:status - 查看规划状态

**详细文档**: ../../commands/workflow/plan.md
```

---

## Mode 4: Beginner Onboarding 🎓

### Example 1: Complete Beginner

**User**: "我是新手，如何开始使用工作流系统？"

**Skill Response**:
```
欢迎使用 Claude DMS3 工作流系统！🎉

## 🚀 快速开始（3 步）

### Step 1: 了解核心命令（Top 5）

1. **/workflow:plan** - 规划一个功能或任务
   用法: /workflow:plan "你想做的事情"

2. **/workflow:execute** - 执行规划的任务
   用法: /workflow:execute

3. **/cli:execute** - 快速自动执行
   用法: /cli:execute "实现XXX功能"

4. **/task:create** - 创建单个任务
   用法: /task:create "任务标题"

5. **/enhance-prompt** - 优化你的提示词
   用法: /enhance-prompt "你的想法"

### Step 2: 尝试第一个工作流

```bash
# 1. 规划一个简单功能
/workflow:plan "添加用户登录功能"

# 2. 执行规划
/workflow:execute

# 3. 审查结果
/workflow:review
```

### Step 3: 学习资源

📖 **必读指南**:
- [5分钟快速开始](getting-started.md)
- [常见工作流模式](workflow-patterns.md)

🛠️ **工具指南**:
- [CLI 工具使用](cli-tools-guide.md)
- [问题排查](troubleshooting.md)

💡 **下一步建议**:
1. 阅读 getting-started.md（5分钟）
2. 尝试上面的第一个工作流
3. 探索更多命令: 询问 "常用命令有哪些"

有问题随时问 "CCW-help"！
```

---

### Example 2: Intermediate User

**User**: "常用命令有哪些？"

**Skill Response**:
```
## 核心命令列表（Top 14）

### 🎯 工作流管理 (5个)
1. workflow:plan - 5阶段规划
2. workflow:execute - 执行任务
3. workflow:review - 审查实现
4. workflow:tdd-plan - TDD 规划
5. workflow:test-gen - 生成测试

### 🛠️ CLI 工具 (3个)
6. cli:execute - 自动执行
7. cli:mode:plan - 规划模式
8. cli:analyze - 代码分析

### 📋 任务管理 (3个)
9. task:create - 创建任务
10. task:execute - 执行任务
11. task:breakdown - 分解任务

### 🔧 实用工具 (2个)
12. enhance-prompt - 提示词增强
13. version - 版本信息

### 💡 使用频率
- **每天**: workflow:plan, workflow:execute, cli:execute
- **经常**: task:create, enhance-prompt
- **按需**: workflow:review, test-gen, tdd-plan

📚 详细文档: [Index Structure](index-structure.md#essential-commands)
```

---

## Mode 5: Issue Reporting 📝

### Example 1: Bug Report

**User**: "CCW-issue"

**Skill (Interactive)**:
```
请选择问题类型：
1. 🐛 Bug 报告
2. ✨ 功能请求
3. ❓ 问题咨询
```

**User**: "1"

**Skill**:
```
好的，我将帮您创建 Bug 报告。请回答以下问题：

1. 简要描述问题：
```

**User**: "/workflow:plan 执行后没有生成任务文件"

**Skill**:
```
2. 您执行的完整命令是什么？
```

**User**: "/workflow:plan '添加登录功能'"

**Skill**:
```
3. 期望的结果是什么？
```

**User**: "应该在 .workflow/ 目录生成 task.json"

**Skill**:
```
4. 实际发生了什么？
```

**User**: "命令执行了但没有生成文件"

**Skill**:
```
✅ Bug 报告已生成！

文件路径: templates/.generated/bug-2025-01-06-123456.md

--- 报告内容 ---
# Bug 报告

## 问题描述
/workflow:plan 执行后没有生成任务文件

## 执行的命令
```bash
/workflow:plan '添加登录功能'
```

## 期望行为
应该在 .workflow/ 目录生成 task.json

## 实际行为
命令执行了但没有生成文件

## 环境信息
- 操作系统: Windows
- 报告日期: 2025-01-06
- 相关命令: /workflow:plan

---

**下一步操作**:
1. 复制以上内容到 GitHub Issues
2. 或使用: gh issue create -F templates/.generated/bug-2025-01-06-123456.md

需要更多帮助？使用 "CCW-help"
```

---

### Example 2: Feature Request

**User**: "我想提个功能建议"

**Skill (Interactive)**:
```
请选择问题类型：
1. 🐛 Bug 报告
2. ✨ 功能请求
3. ❓ 问题咨询
```

**User**: "2"

**Skill**:
```
好的，功能请求。请描述：

1. 您希望添加什么功能？
```

**User**: "希望 /workflow:plan 能自动估算任务时间"

**Skill**:
```
2. 这个功能解决什么问题？
```

**User**: "帮助更好地规划项目进度和资源分配"

**Skill**:
```
3. 具体的使用场景是什么？（可选）
```

**User**: "规划大型项目时，能看到预计完成时间"

**Skill**:
```
✅ 功能请求已生成！

文件路径: templates/.generated/feature-2025-01-06-123457.md

--- 内容预览 ---
# 功能请求

## 功能概述
希望 /workflow:plan 能自动估算任务时间

## 问题背景
帮助更好地规划项目进度和资源分配

## 使用场景
规划大型项目时，能看到预计完成时间

## 优先级
- [ ] 高 - 严重影响工作效率
- [x] 中 - 有明显改善但有变通方案
- [ ] 低 - 锦上添花

---

可以复制到 GitHub 提交，或使用:
gh issue create -F templates/.generated/feature-2025-01-06-123457.md
```

---

## Edge Cases & Error Handling

### Case 1: No Results Found

**User**: "搜索 xyz 命令"

**Skill**:
```
❌ 未找到包含 "xyz" 的命令

💡 建议：
1. 尝试更广泛的关键词（如 "plan", "execute", "test"）
2. 按类别浏览：workflow, cli, memory, task
3. 查看所有常用命令：询问 "常用命令"
4. 需要帮助：使用 "CCW-help"

📚 快速参考：[All Commands Index](index-structure.md#all-commands)
```

---

### Case 2: Ambiguous Query

**User**: "命令"

**Skill**:
```
您想了解什么？请选择：

1. 🔍 搜索特定命令
2. 🤖 获取命令推荐
3. 📖 查看命令文档
4. 🎓 新手入门指导
5. 📝 报告问题或建议

或者直接说出具体需求，例如：
- "搜索 planning 命令"
- "执行完 /workflow:plan 下一步做什么"
- "/task:create 怎么用"
```

---

### Case 3: Multiple Matches

**User**: "execute 命令"

**Skill**:
```
找到多个相关命令，请选择您需要的：

1. **cli:execute** - CLI 自动执行（推荐新手）
2. **task:execute** - 执行单个任务
3. **workflow:execute** - 执行整个工作流

或者询问：
- "cli:execute 的详细文档"
- "三者有什么区别"
- "我该用哪个"
```

---

**Last Updated**: 2025-01-06
