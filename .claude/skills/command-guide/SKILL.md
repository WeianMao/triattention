---
name: command-guide
description: Workflow command guide for Claude Code Workflow (78 commands). Search/browse commands, get next-step recommendations, view documentation, report issues. Triggers "CCW-help", "CCW-issue", "ccw-help", "ccw-issue", "ccw"
allowed-tools: Read, Grep, Glob, AskUserQuestion
version: 5.8.0
---

# Command Guide Skill

Comprehensive command guide for Claude Code Workflow (CCW) system covering 78 commands across 5 categories (workflow, cli, memory, task, general).

## 🆕 What's New in v5.8.0

### Major Features

**🎨 UI Design Style Memory Workflow** (Primary Focus)
- **`/memory:style-skill-memory`** - Generate reusable SKILL packages from design systems
- **`/workflow:ui-design:codify-style`** - Extract design tokens from code with automatic file discovery
- **`/workflow:ui-design:reference-page-generator`** - Generate multi-component reference pages
- **Workflow**: Design extraction → Token documentation → SKILL package → Easy loading

**⚡ `/workflow:lite-plan`** - Intelligent Planning & Execution (Testing Phase)
- Dynamic workflow adaptation (smart exploration, adaptive planning, progressive clarification)
- Two-dimensional confirmation (task approval + execution method selection)
- Direct execution with live TodoWrite progress tracking
- Faster than `/workflow:plan` (1-3 min vs 5-10 min) for simple to medium tasks

**🗺️ `/memory:code-map-memory`** - Code Flow Mapping Generator (Testing Phase)
- Uses cli-explore-agent for deep code flow analysis with dual-source strategy
- Generates Mermaid diagrams for architecture, functions, data flow, conditional paths
- Creates feature-specific SKILL packages for code understanding
- Progressive loading (2K → 30K tokens) for efficient context management

### Agent 

- **cli-explore-agent**  - Specialized code exploration with Deep Scan mode (Bash + Gemini)
- **cli-planning-agent** - Enhanced task generation with improved context handling
- **ui-design-agent** - Major refactoring for better design system extraction

### Additional Improvements
- Enhanced brainstorming workflows with parallel execution
- Improved test workflow documentation and task attachment models
- Updated CLI tool default models (Gemini 2.5-pro)

## 🧠 Core Principle: Intelligent Integration

**⚠️ IMPORTANT**: This SKILL provides **reference materials** for intelligent integration, NOT templates for direct copying.

**Response Strategy**:
1. **Analyze user's specific context** - Understand their exact need, workflow stage, and technical level
2. **Extract relevant information** - Select only the pertinent parts from reference guides
3. **Synthesize and customize** - Combine multiple sources, add context-specific examples
4. **Deliver targeted response** - Provide concise, actionable guidance tailored to the user's situation

**Never**:
- ❌ Copy-paste entire template sections verbatim
- ❌ Return raw reference documentation without processing
- ❌ Provide generic responses that ignore user context

**Always**:
- ✅ Understand the user's specific situation first
- ✅ Integrate information from multiple sources (indexes, guides, reference docs)
- ✅ Customize examples and explanations to match user's use case
- ✅ Provide progressive depth - brief answers with "more detail available" prompts

---

## 🎯 Operation Modes

### Mode 1: Command Search 🔍

**When**: User searches by keyword, category, or use-case

**Triggers**: "搜索命令", "find command", "planning 相关", "search"

**Process**:
1. Identify search type (keyword/category/use-case)
2. Query appropriate index (all-commands/by-category/by-use-case)
3. **Intelligently filter and rank** results based on user's implied context
4. **Synthesize concise response** with command names, brief descriptions, and use-case fit
5. **Suggest next steps** - related commands or workflow patterns

**Example**: "搜索 planning 命令" → Analyze user's likely goal → Present top 3-5 most relevant planning commands with context-specific usage hints, NOT raw JSON dump

---

### Mode 2: Smart Recommendations 🤖

**When**: User asks for next steps after a command

**Triggers**: "下一步", "what's next", "after /workflow:plan", "推荐"

**Process**:
1. **Analyze workflow context** - Understand where user is in their development cycle
2. Query `index/command-relationships.json` for possible next commands
3. **Evaluate and prioritize** recommendations based on:
   - User's stated goals
   - Common workflow patterns
   - Project complexity indicators
4. **Craft contextual guidance** - Explain WHY each recommendation fits, not just WHAT to run
5. **Provide workflow examples** - Show complete flow, not isolated commands

**Example**: "执行完 /workflow:plan 后做什么？" → Analyze plan output quality → Recommend `/workflow:action-plan-verify` (if complex) OR `/workflow:execute` (if straightforward) with reasoning for each choice

---

### Mode 3: Full Documentation 📖

**When**: User requests command details

**Triggers**: "参数说明", "怎么用", "how to use", "详情"

**Process**:
1. Locate command in `index/all-commands.json`
2. Read original command file for full details
3. **Extract user-relevant sections** - Focus on what they asked about (parameters OR examples OR workflow)
4. **Enhance with context** - Add use-case specific examples if user's scenario is clear
5. **Progressive disclosure** - Provide core info first, offer "need more details?" prompts

**Example**: "/workflow:plan 的参数是什么？" → Identify user's experience level → Present parameters with context-appropriate explanations (beginner: verbose + examples; advanced: concise + edge cases), NOT raw documentation dump

---

### Mode 4: Beginner Onboarding 🎓

**When**: New user needs guidance

**Triggers**: "新手", "getting started", "如何开始", "常用命令", **"从0到1"**, **"全新项目"**

**Process**:
1. **Assess user background** - Ask clarifying questions if needed (coding experience? project type?)
2. **⚠️ Identify project stage** - FROM-ZERO-TO-ONE vs FEATURE-ADDITION:
   - **从0到1场景** (全新项目/产品/架构决策) → **MUST START with brainstorming workflow**
   - **功能新增场景** (已有项目中添加功能) → Start with planning workflow
3. **Design personalized learning path** based on their goals and stage
4. **Curate essential commands** from `index/essential-commands.json` - Select 3-5 most relevant for their use case
5. **Provide guided first example** - Walk through ONE complete workflow with explanation, **emphasizing brainstorming for 0-to-1 scenarios**
6. **Set clear next steps** - What to try next, where to get help

**Example 1 (从0到1)**: "我是新手，如何开始全新项目？" → Identify as FROM-ZERO-TO-ONE → Emphasize brainstorming workflow (`/workflow:brainstorm:artifacts`) as mandatory first step → Explain brainstorm → plan → execute flow

**Example 2 (功能新增)**: "我是新手，如何在已有项目中添加功能？" → Identify as FEATURE-ADDITION → Guide to planning workflow (`/workflow:plan`) → Explain plan → execute → test flow

**Example 3 (探索)**: "我是新手，如何开始？" → Ask clarifying question: "是全新项目启动（从0到1）还是在已有项目中添加功能？" → Based on answer, route to appropriate workflow

---

### Mode 5: Issue Reporting 📝

**When**: User wants to report issue or request feature

**Triggers**: **"CCW-issue"**, **"CCW-help"**, **"ccw-issue"**, **"ccw-help"**, **"ccw"**, "报告 bug", "功能建议", "问题咨询", "交互式诊断"

**Process**:
1. **Understand issue context** - Use AskUserQuestion to confirm type AND gather initial context
2. **Intelligently guide information collection**:
   - Adapt questions based on previous answers
   - Skip irrelevant sections
   - Probe for missing critical details
3. **Select and customize template**:
   - `issue-diagnosis.md`, `issue-bug.md`, `issue-feature.md`, or `issue-question.md`
   - **Adapt template sections** to match user's specific scenario
4. **Synthesize coherent issue report**:
   - Integrate collected information with appropriate template sections
   - **Highlight key details** - Don't bury critical info in boilerplate
   - Add privacy-protected command history
5. **Provide actionable next steps** - Immediate troubleshooting OR submission guidance

**Example**: "CCW-issue" → Detect user frustration level → For urgent: fast-track to critical info collection; For exploratory: comprehensive diagnostic flow, NOT one-size-fits-all questionnaire

**🆕 Enhanced Features**:
- Complete command history with privacy protection
- Interactive diagnostic checklists
- Decision tree navigation (diagnosis template)
- Execution environment capture

---

### Mode 6: Deep Command Analysis 🔬

**When**: User asks detailed questions about specific commands or agents

**Triggers**: "详细说明", "命令原理", "agent 如何工作", "实现细节", specific command/agent name mentioned

**Data Sources**:
- `reference/agents/*.md` - All agent documentation (11 agents)
- `reference/commands/**/*.md` - All command documentation (69 commands)

**Process**:

**Simple Query** (direct documentation lookup):
1. Identify target command/agent from user query
2. Locate corresponding markdown file in `reference/`
3. **Extract contextually relevant sections** - Not entire document
4. **Synthesize focused explanation**:
   - Address user's specific question
   - Add context-appropriate examples
   - Link related concepts
5. **Offer progressive depth** - "Want to know more about X?"

**Complex Query** (CLI-assisted analysis):
1. **Detect complexity indicators** (多个命令对比、工作流程分析、最佳实践)
2. **Design targeted analysis prompt** for gemini/qwen:
   - Frame user's question precisely
   - Specify required analysis depth
   - Request structured comparison/synthesis
   ```bash
   gemini -p "
   PURPOSE: Analyze command documentation to answer user query
   TASK: [extracted user question with context]
   MODE: analysis
   CONTEXT: @**/*
   EXPECTED: Comprehensive answer with examples and recommendations
   RULES: $(cat ~/.claude/workflows/cli-templates/prompts/analysis/02-analyze-code-patterns.txt) | Focus on practical usage | analysis=READ-ONLY
   " -m gemini-3-pro-preview-11-2025 --include-directories ~/.claude/skills/command-guide/reference
   ```
   Note: Use absolute path `~/.claude/skills/command-guide/reference` for reference documentation access
3. **Process and integrate CLI analysis**:
   - Extract key insights from CLI output
   - Add context-specific examples
   - Synthesize actionable recommendations
4. **Deliver tailored response** - Not raw CLI output

**Query Classification**:
- **Simple**: Single command explanation, parameter list, basic usage
- **Complex**: Cross-command workflows, performance comparison, architectural analysis, best practices across multiple commands

**Examples**:

*Simple Query*:
```
User: "action-planning-agent 如何工作？"
→ Read reference/agents/action-planning-agent.md
→ **Identify user's knowledge gap** (mechanism? inputs/outputs? when to use?)
→ **Extract relevant sections** addressing their need
→ **Synthesize focused explanation** with examples
→ NOT: Dump entire agent documentation
```

*Complex Query*:
```
User: "对比 workflow:plan 和 workflow:tdd-plan 的使用场景和最佳实践"
→ Detect: 多命令对比 + 最佳实践
→ **Design comparison framework** (when to use, trade-offs, workflow integration)
→ Use gemini to analyze both commands with structured comparison prompt
→ **Synthesize insights** into decision matrix and usage guidelines
→ NOT: Raw command documentation side-by-side
```

---

## 📚 Index Files

All command metadata is stored in JSON indexes for fast querying:

- **all-commands.json** - Complete catalog (69 commands) with full metadata
- **by-category.json** - Hierarchical organization (workflow/cli/memory/task)
- **by-use-case.json** - Grouped by scenario (planning/implementation/testing/docs/session)
- **essential-commands.json** - Top 14 most-used commands
- **command-relationships.json** - Next-step recommendations and dependencies

📖 Detailed structure: [Index Structure Reference](guides/index-structure.md)

---

## 🗂️ Supporting Guides

- **[Getting Started](guides/getting-started.md)** - 5-minute quickstart for beginners
- **[Workflow Patterns](guides/workflow-patterns.md)** - Common workflow examples (Plan→Execute, TDD, UI design)
- **[CLI Tools Guide](guides/cli-tools-guide.md)** - Gemini/Qwen/Codex usage
- **[Troubleshooting](guides/troubleshooting.md)** - Common issues and solutions
- **[Implementation Details](guides/implementation-details.md)** - Detailed logic for each mode
- **[Usage Examples](guides/examples.md)** - Example dialogues and edge cases

## 📦 Reference Documentation

Complete backup of all command and agent documentation for deep analysis:

- **[reference/agents/](reference/agents/)** - 14 agent markdown files with implementation details
  - **New in v5.8**: cli-explore-agent (code exploration), cli-planning-agent (enhanced)
- **[reference/commands/](reference/commands/)** - 78 command markdown files organized by category
  - `cli/` - CLI tool commands (10 files) - **New**: document-analysis mode
  - `memory/` - Memory management commands (12 files) - **New**: docs-full-cli, docs-related-cli, code-map-memory, style-skill-memory
  - `task/` - Task management commands (4 files)
  - `workflow/` - Workflow commands (50 files) - **New**: lite-plan, lite-fix, ui-design enhancements

**Installation Path**: `~/.claude/skills/command-guide/` (skill designed for global installation)

**Absolute Reference Path**: `~/.claude/skills/command-guide/reference/`

**Usage**: Mode 6 queries these files directly for detailed command/agent analysis, or uses CLI tools (gemini/qwen) with absolute paths for complex cross-command analysis.

---

## 🛠️ Issue Templates

Generate standardized GitHub issue templates with **execution flow emphasis**:

- **[Interactive Diagnosis](templates/issue-diagnosis.md)** - 🆕 Comprehensive diagnostic workflow with decision tree, checklists, and full command history
- **[Bug Report](templates/issue-bug.md)** - Report command errors with complete execution flow and environment details
- **[Feature Request](templates/issue-feature.md)** - Suggest improvements with current workflow analysis and pain points
- **[Question](templates/issue-question.md)** - Ask usage questions with detailed attempt history and context

**All templates now include**:
- ✅ Complete command history sections (with privacy protection)
- ✅ Execution environment details
- ✅ Interactive problem-locating checklists
- ✅ Structured troubleshooting guidance

Templates are auto-populated during Mode 5 (Issue Reporting) interaction.

---

## 📊 System Statistics

- **Total Commands**: 78
- **Total Agents**: 14
- **Categories**: 5 (workflow: 50, cli: 10, memory: 12, task: 4, general: 2)
- **Use Cases**: 7 (planning, implementation, testing, documentation, session-management, analysis, general)
- **Difficulty Levels**: 3 (Beginner, Intermediate, Advanced)
- **Essential Commands**: 13
- **Reference Docs**: 92 markdown files (14 agents + 78 commands)

---

## 🔧 Maintenance

### Updating Indexes

When commands are added/modified/removed:

```bash
cd /d/Claude_dms3/.claude/skills/command-guide
python scripts/analyze_commands.py
```

This script:
1. Scans all command files in `../../commands/`
2. Extracts metadata from YAML frontmatter
3. Analyzes command relationships
4. Regenerates all 5 index files

### Committing Updates

```bash
git add .claude/skills/command-guide/
git commit -m "docs: update command indexes"
git push
```

Team members get latest indexes via `git pull`.

---

## 📖 Related Documentation

- [Workflow Architecture](../../workflows/workflow-architecture.md) - System design overview
- [Intelligent Tools Strategy](../../workflows/intelligent-tools-strategy.md) - CLI tool selection
- [Context Search Strategy](../../workflows/context-search-strategy.md) - Search patterns
- [Task Core](../../workflows/task-core.md) - Task system fundamentals

---

## 🔄 Maintenance

### Documentation Updates

This SKILL documentation is kept in sync with command implementations through a standardized update process.

**Update Guideline**: See [UPDATE-GUIDELINE.md](UPDATE-GUIDELINE.md) for the complete documentation maintenance process.

**Update Process**:
1. **Analyze**: Identify changed commands/agents from git commits
2. **Extract**: Gather change information and impact assessment
3. **Update**: Sync reference docs, guides, and examples
4. **Regenerate**: Run `scripts/analyze_commands.py` to rebuild indexes
5. **Validate**: Test examples and verify consistency
6. **Commit**: Follow standardized commit message format

**Key Capabilities**:
- 6 operation modes (Search, Recommendations, Full Docs, Onboarding, Issue Reporting, Deep Analysis)
- 80 reference documentation files (11 agents + 69 commands)
- 5 JSON indexes for fast command lookup
- 8 comprehensive guides covering all workflow patterns
- 4 issue templates for standardized problem reporting
- CLI-assisted complex query analysis with gemini/qwen integration

**Maintainer**: CCW Team
