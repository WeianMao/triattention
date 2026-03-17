# 5分钟快速上手指南

> 欢迎使用 Claude DMS3！本指南将帮助您快速上手，5分钟内开始第一个工作流。

## 🎯 Claude DMS3 是什么？

Claude DMS3 是一个**智能开发管理系统**，集成了 69 个命令，帮助您：
- 📋 规划和分解复杂任务
- ⚡ 自动化代码实现
- 🧪 生成和执行测试
- 📚 生成项目文档
- 🤖 使用 AI 工具（Gemini、Qwen、Codex）加速开发

**核心理念**：用自然语言描述需求 → 系统自动规划和执行 → 获得结果

---

## 🚀 最常用的14个命令

### 工作流类（必会）

| 命令 | 用途 | 何时使用 |
|------|------|----------|
| `/workflow:plan` | 规划任务 | 开始新功能、新项目 |
| `/workflow:execute` | 执行任务 | plan 之后，实现功能 |
| `/workflow:test-gen` | 生成测试 | 实现完成后，生成测试 |
| `/workflow:status` | 查看进度 | 查看工作流状态 |
| `/workflow:resume` | 恢复任务 | 继续之前的工作流 |

### CLI 工具类（常用）

| 命令 | 用途 | 何时使用 |
|------|------|----------|
| `/cli:analyze` | 代码分析 | 理解代码、分析架构 |
| `/cli:execute` | 执行实现 | 精确控制实现过程 |
| `/cli:codex-execute` | 自动化实现 | 快速实现功能 |
| `/cli:chat` | 问答交互 | 询问代码库问题 |

### Memory 类（知识管理）

| 命令 | 用途 | 何时使用 |
|------|------|----------|
| `/memory:docs` | 生成文档 | 生成模块文档 |
| `/memory:load` | 加载上下文 | 获取任务相关上下文 |

### Task 类（任务管理）

| 命令 | 用途 | 何时使用 |
|------|------|----------|
| `/task:create` | 创建任务 | 手动创建单个任务 |
| `/task:execute` | 执行任务 | 执行特定任务 |

---

## 📝 第一个工作流：实现一个新功能

让我们通过一个实际例子来体验完整的工作流：**实现用户登录功能**

### 步骤 1：规划任务

```bash
/workflow:plan --agent "实现用户登录功能，包括邮箱密码验证和JWT令牌"
```

**发生什么**：
- 系统分析需求
- 自动生成任务计划（IMPL_PLAN.md）
- 创建多个子任务（task JSON 文件）
- 返回 workflow session ID（如 WFS-20251106-xxx）

**你会看到**：
- ✅ 规划完成
- 📋 任务列表（如：task-001-user-model, task-002-login-api 等）
- 📁 Session 目录创建

---

### 步骤 2：执行实现

```bash
/workflow:execute
```

**发生什么**：
- 系统自动发现最新的 workflow session
- 按顺序执行所有任务
- 使用 Codex 自动生成代码
- 实时显示进度

**你会看到**：
- ⏳ Task 1 执行中...
- ✅ Task 1 完成
- ⏳ Task 2 执行中...
- （依次执行所有任务）

---

### 步骤 3：生成测试

```bash
/workflow:test-gen WFS-20251106-xxx
```

**发生什么**：
- 分析实现的代码
- 生成测试策略
- 创建测试任务

---

### 步骤 4：查看状态

```bash
/workflow:status
```

**发生什么**：
- 显示当前工作流状态
- 列出所有任务及其状态
- 显示已完成/进行中/待执行任务

---

## 🎓 其他常用场景

### 场景 1：快速代码分析

**需求**：理解陌生代码

```bash
# 分析整体架构
/cli:analyze --tool gemini "分析项目架构和模块关系"

# 追踪执行流程
/cli:mode:code-analysis --tool gemini "追踪用户注册的执行流程"
```

---

### 场景 2：快速实现功能

**需求**：实现一个简单功能

```bash
# 方式 1：完整工作流（推荐）
/workflow:plan "添加用户头像上传功能"
/workflow:execute

# 方式 2：直接实现（快速）
/cli:codex-execute "添加用户头像上传功能，支持图片裁剪和压缩"
```

---

### 场景 3：恢复之前的工作

**需求**：继续上次的任务

```bash
# 查看可恢复的 session
/workflow:status

# 恢复特定 session
/workflow:resume WFS-20251106-xxx
```

---

### 场景 4：生成文档

**需求**：为模块生成文档

```bash
/memory:docs src/auth --tool gemini --mode full
```

---

## 💡 快速记忆法

记住这个流程，就能完成大部分任务：

```
规划 → 执行 → 测试 → 完成
  ↓      ↓      ↓
plan → execute → test-gen
```

**扩展场景**：
- 需要分析理解 → 使用 `/cli:analyze`
- 需要精确控制 → 使用 `/cli:execute`
- 需要快速实现 → 使用 `/cli:codex-execute`

---

## 🆘 遇到问题？

### 命令记不住？

使用 Command Guide SKILL：
```bash
ccw  # 或 ccw-help
```

然后说：
- "搜索 planning 命令"
- "执行完 /workflow:plan 后做什么"
- "我是新手，如何开始"

---

### 执行失败？

1. **查看错误信息**：仔细阅读错误提示
2. **使用诊断模板**：`ccw-issue` → 选择 "诊断模板"
3. **查看排查指南**：[Troubleshooting Guide](troubleshooting.md)

---

### 想深入学习？

- **工作流模式**：[Workflow Patterns](workflow-patterns.md) - 学习更多工作流组合
- **CLI 工具使用**：[CLI Tools Guide](cli-tools-guide.md) - 了解 Gemini/Qwen/Codex 的高级用法
- **完整命令列表**：查看 `index/essential-commands.json`

---

## 🎯 下一步

现在你已经掌握了基础！尝试：

1. **实践基础工作流**：选择一个小功能，走一遍 plan → execute → test-gen 流程
2. **探索 CLI 工具**：尝试用 `/cli:analyze` 分析你的代码库
3. **学习工作流模式**：阅读 [Workflow Patterns](workflow-patterns.md) 了解更多高级用法

**记住**：Claude DMS3 的设计理念是让你用自然语言描述需求，系统自动完成繁琐的工作。不要担心命令记不住，随时可以使用 `ccw` 获取帮助！

---

**祝你使用愉快！** 🎉

有任何问题，使用 `ccw-issue` 提交问题或查询帮助。
