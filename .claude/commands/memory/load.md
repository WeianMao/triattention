---
name: load
description: Delegate to universal-executor agent to analyze project via Gemini/Qwen CLI and return JSON core content package for task context
argument-hint: "[--tool gemini|qwen] \"task context description\""
allowed-tools: Task(*), Bash(*)
examples:
  - /memory:load "在当前前端基础上开发用户认证功能"
  - /memory:load --tool qwen -p "重构支付模块API"
---

# Memory Load Command (/memory:load)

## 1. Overview

The `memory:load` command **delegates to a universal-executor agent** to analyze the project and return a structured "Core Content Pack". This pack is loaded into the main thread's memory, providing essential context for subsequent agent operations while minimizing token consumption.

**Core Philosophy**:
- **Agent-Driven**: Fully delegates execution to universal-executor agent
- **Read-Only Analysis**: Does not modify code, only extracts context
- **Structured Output**: Returns standardized JSON content package
- **Memory Optimization**: Package loaded directly into main thread memory
- **Token Efficiency**: CLI analysis executed within agent to save tokens

## 2. Parameters

- `"task context description"` (Required): Task description to guide context extraction
  - Example: "在当前前端基础上开发用户认证功能"
  - Example: "重构支付模块API"
  - Example: "修复数据库查询性能问题"

- `--tool <gemini|qwen>` (Optional): Specify CLI tool for agent to use (default: gemini)
  - gemini: Large context window, suitable for complex project analysis
  - qwen: Alternative to Gemini with similar capabilities

## 3. Agent-Driven Execution Flow

The command fully delegates to **universal-executor agent**, which autonomously:

1. **Analyzes Project Structure**: Executes `get_modules_by_depth.sh` to understand architecture
2. **Loads Documentation**: Reads CLAUDE.md, README.md and other key docs
3. **Extracts Keywords**: Derives core keywords from task description
4. **Discovers Files**: Uses MCP code-index or rg/find to locate relevant files
5. **CLI Deep Analysis**: Executes Gemini/Qwen CLI for deep context analysis
6. **Generates Content Package**: Returns structured JSON core content package

## 4. Core Content Package Structure

**Output Format** - Loaded into main thread memory for subsequent use:

```json
{
  "task_context": "在当前前端基础上开发用户认证功能",
  "keywords": ["前端", "用户", "认证", "auth", "login"],
  "project_summary": {
    "architecture": "TypeScript + React frontend with Vite build system",
    "tech_stack": ["React", "TypeScript", "Vite", "TailwindCSS"],
    "key_patterns": [
      "State management via Context API",
      "Functional components with Hooks pattern",
      "API calls encapsulated in custom hooks"
    ]
  },
  "relevant_files": [
    {
      "path": "src/components/Auth/LoginForm.tsx",
      "relevance": "Existing login form component",
      "priority": "high"
    },
    {
      "path": "src/contexts/AuthContext.tsx",
      "relevance": "Authentication state management context",
      "priority": "high"
    },
    {
      "path": "CLAUDE.md",
      "relevance": "Project development standards",
      "priority": "high"
    }
  ],
  "integration_points": [
    "Must integrate with existing AuthContext",
    "Follow component organization pattern: src/components/[Feature]/",
    "API calls should use src/hooks/useApi.ts wrapper"
  ],
  "constraints": [
    "Maintain backward compatibility",
    "Follow TypeScript strict mode",
    "Use existing UI component library"
  ]
}
```

## 5. Agent Invocation

```javascript
Task(
  subagent_type="universal-executor",
  description="Load project memory: ${task_description}",
  prompt=`
## Mission: Load Project Memory Context

**Task**: Load project memory context for: "${task_description}"
**Mode**: analysis
**Tool Preference**: ${tool || 'gemini'}

## Execution Steps

### Step 1: Foundation Analysis

1. **Project Structure**
   \`\`\`bash
   bash(ccw tool exec get_modules_by_depth '{}')
   \`\`\`

2. **Core Documentation**
   \`\`\`javascript
   Read(CLAUDE.md)
   Read(README.md)
   \`\`\`

### Step 2: Keyword Extraction & File Discovery

1. Extract core keywords from task description
2. Discover relevant files using ripgrep and find:
   \`\`\`bash
   # Find files by name
   find . -name "*{keyword}*" -type f

   # Search content with ripgrep
   rg "{keyword}" --type ts --type md -C 2
   rg -l "{keyword}" --type ts --type md  # List files only
   \`\`\`

### Step 3: Deep Analysis via CLI

Execute Gemini/Qwen CLI for deep analysis (saves main thread tokens):

\`\`\`bash
cd . && ${tool} -p "
PURPOSE: Extract project core context for task: ${task_description}
TASK: Analyze project architecture, tech stack, key patterns, relevant files
MODE: analysis
CONTEXT: @CLAUDE.md,README.md @${discovered_files}
EXPECTED: Structured project summary and integration point analysis
RULES:
- Focus on task-relevant core information
- Identify key architecture patterns and technical constraints
- Extract integration points and development standards
- Output concise, structured format
"
\`\`\`

### Step 4: Generate Core Content Package

Generate structured JSON content package (format shown above)

**Required Fields**:
- task_context: Original task description
- keywords: Extracted keyword array
- project_summary: Architecture, tech stack, key patterns
- relevant_files: File list with path, relevance, priority
- integration_points: Integration guidance
- constraints: Development constraints

### Step 5: Return Content Package

Return JSON content package as final output for main thread to load into memory.

## Quality Checklist

Before returning:
- [ ] Valid JSON format
- [ ] All required fields complete
- [ ] relevant_files contains 3-10 files minimum
- [ ] project_summary accurately reflects architecture
- [ ] integration_points clearly specify integration paths
- [ ] keywords accurately extracted (3-8 keywords)
- [ ] Content concise, avoiding redundancy (< 5KB total)

  `
)
```

## 6. Usage Examples

### Example 1: Load Context for New Feature

```bash
/memory:load "在当前前端基础上开发用户认证功能"
```

**Agent Execution**:
1. Analyzes project structure (`get_modules_by_depth.sh`)
2. Reads CLAUDE.md, README.md
3. Extracts keywords: ["前端", "用户", "认证", "auth"]
4. Uses MCP to search relevant files
5. Executes Gemini CLI for deep analysis
6. Returns core content package

**Returned Package** (loaded into memory):
```json
{
  "task_context": "在当前前端基础上开发用户认证功能",
  "keywords": ["前端", "认证", "auth", "login"],
  "project_summary": { ... },
  "relevant_files": [ ... ],
  "integration_points": [ ... ],
  "constraints": [ ... ]
}
```

### Example 2: Using Qwen Tool

```bash
/memory:load --tool qwen -p "重构支付模块API"
```

Agent uses Qwen CLI for analysis, returns same structured package.

### Example 3: Bug Fix Context

```bash
/memory:load "修复登录验证错误"
```

Returns core context related to login validation, including test files and validation logic.

### Memory Persistence

- **Session-Scoped**: Content package valid for current session
- **Subsequent Reference**: All subsequent agents/commands can access
- **Reload Required**: New sessions need to re-execute /memory:load

## 8. Notes

- **Read-Only**: Does not modify any code, pure analysis
- **Token Optimization**: CLI analysis executed within agent, saves main thread tokens
- **Memory Loading**: Returned JSON loaded directly into main thread memory
- **Subsequent Use**: Other commands/agents can reference this package for development
- **Session-Level**: Content package valid for current session
