# CLI 工具使用指南

> **SKILL 参考文档**：用于回答用户关于 CLI 工具（Gemini、Qwen、Codex）的使用问题
>
> **用途**：当用户询问 CLI 工具的能力、使用方法、调用方式时，从本文档中提取相关信息，根据用户具体需求加工后返回

## 🎯 快速理解：CLI 工具是什么？

CLI 工具是集成在 Claude DMS3 中的**智能分析和执行助手**。

**工作流程**：
1. **用户** → 用自然语言向 Claude Code 描述需求（如"分析认证模块的安全性"）
2. **Claude Code** → 识别用户意图，决定使用哪种方式：
   - **CLI 工具语义调用**：生成并执行 `gemini`/`qwen`/`codex` 命令
   - **Slash 命令调用**：执行预定义的工作流命令（如 `/workflow:plan`）
3. **工具** → 自动完成任务并返回结果

**核心理念**：用户用自然语言描述需求 → Claude Code 选择最佳方式 → 工具执行 → 返回结果

---

## 📋 三大工具能力对比

| 工具 | 擅长领域 | 典型场景 | 何时使用 |
|------|----------|----------|----------|
| **Gemini** | 分析、理解、规划 | 代码分析、架构设计、问题诊断 | 需要深入理解代码或系统 |
| **Qwen** | 分析、备选方案 | 代码审查、模式识别 | Gemini 不可用时的备选 |
| **Codex** | 实现、测试、执行 | 功能开发、测试生成、自动化任务 | 需要生成代码或自动执行 |

**简单记忆**：
- 想**理解**什么 → Gemini / Qwen
- 想**实现**什么 → Codex

---

## 🚀 Claude Code 的两种响应方式

当用户用自然语言描述需求时，Claude Code 会根据任务特性选择最佳方式：

### 方式 1：CLI 工具语义调用（灵活、强大）

**用户明确指示使用 CLI 工具**，Claude Code 生成并执行相应命令。

#### 示例 1：Gemini 语义分析

**用户输入**（必须明确提到工具）：
```
使用 gemini 分析认证模块的安全性，识别潜在的安全漏洞，
关注 JWT 令牌处理、密码存储、会话管理等方面，
提供具体的修复建议
```

**Claude Code 生成并执行**（CLI 命令）：
```bash
cd src/auth && gemini -p "
分析认证模块的安全性，识别潜在的安全漏洞，
关注 JWT 令牌处理、密码存储、会话管理等方面，
提供具体的修复建议
" -m gemini-3-pro-preview-11-2025
```

**工具输出**：Gemini 自动理解任务，组合多个分析步骤，返回结构化报告

---

#### 示例 2：Qwen 语义分析（Gemini 的备选）

**用户输入**：
```
用 qwen 追踪用户登录的完整执行流程，
从 API 入口到数据库查询，
列出所有调用的函数和依赖关系
```

**Claude Code 生成并执行**：
```bash
cd src/auth && qwen -p "
追踪用户登录的完整执行流程，
从 API 入口到数据库查询，
列出所有调用的函数和依赖关系
"
```

**工具输出**：Qwen 理解需求，自动追踪执行路径

---

#### 示例 3：Codex 语义实现

**用户输入**：
```
让 codex 实现 JWT 令牌刷新机制：
1. 添加 refreshToken 端点
2. 验证旧令牌的有效性
3. 生成新的访问令牌
4. 确保安全性（防重放攻击）
```

**Claude Code 生成并执行**：
```bash
codex -C src/auth --full-auto exec "
实现 JWT 令牌刷新机制：
1. 添加 refreshToken 端点
2. 验证旧令牌的有效性
3. 生成新的访问令牌
4. 确保安全性（防重放攻击）
" -m gpt-5 --skip-git-repo-check -s danger-full-access
```

**工具输出**：Codex 理解需求，自动生成代码并集成到现有系统

---

**核心特点**：
- ✅ **用户明确指定工具**：必须说"使用 gemini"、"用 qwen"、"让 codex"等触发工具调用
- ✅ **Claude 生成命令**：识别工具名称后，自动构造最优的 CLI 工具调用
- ✅ **工具自动理解**：CLI 工具解析需求，组合分析/实现步骤
- ✅ **灵活强大**：不受预定义工作流限制
- ✅ **精确控制**：Claude 可指定工作目录、文件范围、模型参数

**触发方式**：
- "使用 gemini ..."
- "用 qwen ..."
- "让 codex ..."
- "通过 gemini 工具..."

**Claude Code 何时选择此方式**：
- 用户明确指定使用某个 CLI 工具
- 复杂分析任务（跨模块、多维度）
- 自定义工作流需求
- 需要精确控制上下文范围

---

### 方式 2：Slash 命令调用（标准工作流）

**用户直接输入 Slash 命令**，或 **Claude Code 建议使用 Slash 命令**，系统执行预定义工作流（内部调用 CLI 工具）。

#### Workflow 类命令（系统自动选择工具）

**示例 1：规划任务**

**用户输入**：
```
/workflow:plan --agent "实现用户认证功能"
```

**系统执行**：内部调用 gemini/qwen 分析 + action-planning-agent 生成任务

---

**示例 2：执行任务**

**用户输入**：
```
/workflow:execute
```

**系统执行**：内部调用 codex 实现代码

---

**示例 3：生成测试**

**用户输入**：
```
/workflow:test-gen WFS-xxx
```

**系统执行**：内部调用 gemini 分析 + codex 生成测试

---

#### CLI 类命令（指定工具）

**示例 1：封装的分析命令**

**用户输入**：
```
/cli:analyze --tool gemini "分析认证模块"
```

**系统执行**：使用 gemini 工具进行分析

---

**示例 2：封装的执行命令**

**用户输入**：
```
/cli:execute --tool codex "实现 JWT 刷新"
```

**系统执行**：使用 codex 工具实现功能

---

**示例 3：快速执行（YOLO 模式）**

**用户输入**：
```
/cli:codex-execute "添加用户头像上传"
```

**系统执行**：使用 codex 快速实现

---

**核心特点**：
- ✅ **用户可直接输入**：Slash 命令格式固定，用户可以直接输入（如 `/workflow:plan`）
- ✅ **Claude 可建议**：Claude Code 也可以识别需求后建议或执行 Slash 命令
- ✅ **预定义流程**：标准化的工作流模板
- ✅ **自动工具选择**：workflow 命令内部自动选择合适的 CLI 工具
- ✅ **集成完整**：包含规划、执行、测试、文档等环节
- ✅ **简单易用**：无需了解底层 CLI 工具细节

**Claude Code 何时选择此方式**：
- 标准开发任务（功能开发、测试、重构）
- 团队协作（统一工作流）
- 适合新手（降低学习曲线）
- 快速开发（减少配置时间）

---

## 🔄 两种方式对比

| 维度 | CLI 工具语义调用 | Slash 命令调用 |
|------|------------------|----------------|
| **用户输入** | 纯自然语言描述需求 | `/` 开头的固定命令格式 |
| **Claude Code 行为** | 生成并执行 `gemini`/`qwen`/`codex` 命令 | 执行预定义工作流（内部调用 CLI 工具） |
| **灵活性** | 完全自定义任务和执行方式 | 固定工作流模板 |
| **学习曲线** | 用户无需学习（纯自然语言） | 需要知道 Slash 命令名称 |
| **适用复杂度** | 复杂、探索性、定制化任务 | 标准、重复性、工作流化任务 |
| **工具选择** | Claude 自动选择最佳 CLI 工具 | 系统自动选择（workflow 类）<br>或用户指定（cli 类） |
| **典型场景** | 深度分析、自定义流程、探索研究 | 日常开发、团队协作、标准流程 |

**使用建议**：
- **日常开发** → 优先使用 Slash 命令（标准化、快速）
- **复杂分析** → Claude 自动选择 CLI 工具语义调用（灵活、强大）
- **用户角度** → 只需用自然语言描述需求，Claude Code 会选择最佳方式

---

## 💡 工具能力速查

### Gemini - 分析与规划
- 执行流程追踪、依赖分析、代码模式识别
- 架构设计、技术方案评估、任务分解
- 文档生成（API 文档、模块说明）

**触发示例**：`使用 gemini 追踪用户登录的完整流程`

---

### Qwen - Gemini 的备选
- 代码分析、模式识别、架构评审
- 作为 Gemini 不可用时的备选方案

**触发示例**：`用 qwen 分析数据处理模块`

---

### Codex - 实现与执行
- 功能开发、组件实现、API 创建
- 单元测试、集成测试、TDD 支持
- 代码重构、性能改进、Bug 修复

**触发示例**：`让 codex 实现用户注册功能，包含邮箱验证`

---

## 🔄 典型使用场景

### 场景 1：理解陌生代码库

**需求**：接手新项目，需要快速理解代码结构

**方式 1：CLI 工具语义调用**（推荐，灵活）

- **用户输入**：`使用 gemini 分析这个项目的架构设计，识别主要模块、依赖关系和架构模式`
- **Claude Code 生成并执行**：`cd project-root && gemini -p "..."`

**方式 2：Slash 命令**
- **用户输入**：`/cli:analyze --tool gemini "分析项目架构"`

---

### 场景 2：实现新功能

**需求**：实现用户认证功能

**方式 1：CLI 工具语义调用**
- **用户输入**：`让 codex 实现用户认证功能：注册（邮箱+密码+验证）、登录（JWT token）、刷新令牌，技术栈 Node.js + Express`
- **Claude Code 生成并执行**：`codex -C src/auth --full-auto exec "..." --skip-git-repo-check -s danger-full-access`

**方式 2：Slash 命令**（工作流化）
- **用户输入**：`/workflow:plan --agent "实现用户认证功能"` → `/workflow:execute`

---

### 场景 3：诊断 Bug

**需求**：登录功能偶尔超时

**方式 1：CLI 工具语义调用**
- **用户输入**：`使用 gemini 诊断登录超时问题，分析处理流程、性能瓶颈、数据库查询效率`
- **Claude Code 生成并执行**：`cd src/auth && gemini -p "..."`
- **用户输入**：`让 codex 根据上述分析修复登录超时，优化查询、添加缓存`
- **Claude Code 生成并执行**：`codex -C src/auth --full-auto exec "..." --skip-git-repo-check -s danger-full-access`

**方式 2：Slash 命令**
- **用户输入**：`/cli:mode:bug-diagnosis --tool gemini "诊断登录超时"` → `/cli:execute --tool codex "修复登录超时"`

---

### 场景 4：生成文档

**需求**：为 API 模块生成完整文档

**方式 1：CLI 工具语义调用**
- **用户输入**：`使用 gemini 为 API 模块生成技术文档，包含端点说明、数据模型、使用示例`
- **Claude Code 生成并执行**：`cd src/api && gemini -p "..." --approval-mode yolo`

**方式 2：Slash 命令**
- **用户输入**：`/memory:docs src/api --tool gemini --mode full`

---

## 🎯 常用工作流程

### 简单 Bug 修复
```
使用 gemini 诊断问题（可选其他 cli 工具）
→ Claude 分析
→ Claude 直接执行修复
```

### 复杂 Bug 修复
```
/cli:mode:plan 或 /cli:mode:bug-diagnosis
→ Claude 分析
→ Claude 执行修复
```

### 简单功能增加
```
/cli:mode:plan
→ Claude 执行
```

### 复杂功能增加
```
/cli:mode:plan --agent
→ Claude 执行 或 /cli:codex-execute

或

/cli:mode:plan
→ 进入工作流模式（/workflow:execute）
```

### 项目内存管理

**建立技术栈文档**（为项目提供技术参考）
```
/memory:tech-research [session-id | tech-stack-name]
```

**为项目重建多级结构的 CLAUDE.md 内存**
```
/memory:docs [path] [--tool gemini|qwen|codex] [--mode full|partial]
```

---

## 📚 常用命令速查

| 需求 | 推荐命令 |
|------|----------|
| **代码分析** | `使用 gemini 分析...` 或 `/cli:analyze --tool gemini` |
| **Bug 诊断** | `/cli:mode:bug-diagnosis` |
| **功能实现** | `/cli:codex-execute` 或 `让 codex 实现...` |
| **架构规划** | `/cli:mode:plan` |
| **生成测试** | `/workflow:test-gen WFS-xxx` |
| **完整工作流** | `/workflow:plan` → `/workflow:execute` |
| **技术文档** | `/memory:tech-research [tech-name]` |
| **项目文档** | `/memory:docs [path]` |

---

## 🆘 快速提示

**触发 CLI 工具语义调用**：
- "使用 gemini ..."
- "用 qwen ..."
- "让 codex ..."

**选择工具**：
- **理解/分析/规划** → Gemini
- **实现/测试/执行** → Codex
- **不确定** → 使用 Slash 命令让系统选择

**提升质量**：
- 清晰描述需求和期望
- 提供上下文信息
- 使用 `--agent` 处理复杂任务

---

**最后更新**: 2025-11-06
