---
name: context-search-strategy
description: Strategic guidelines for context search commands
type: search-guideline
---

# Context Search Strategy

## ⚡ Execution Environment

**CRITICAL**: All commands execute in **Bash environment** (Git Bash on Windows)

**❌ Forbidden**: Windows commands (`findstr`, `dir`, `where`) - Use Bash (`grep`, `find`, `cat`)

## ⚡ Core Search Tools

**Skill()**: FASTEST way to get context - use FIRST if SKILL exists. Three types: (1) `workflow-progress` for WFS sessions (2) tech SKILLs for stack docs (3) `{project-name}` for project docs
**codebase-retrieval**: Semantic file discovery via Gemini CLI with all files analysis
**rg (ripgrep)**: Fast content search with regex support
**find**: File/directory location by name patterns
**grep**: Built-in pattern matching (fallback when rg unavailable)
**get_modules_by_depth.sh**: Program architecture analysis (MANDATORY before planning)

## 📋 Tool Selection Matrix

| Need | Tool | Use Case |
|------|------|----------|
| **Workflow history** | Skill(workflow-progress) | WFS sessions lessons/conflicts - `/memory:workflow-skill-memory` |
| **Tech stack docs** | Skill({tech-name}) | Stack APIs/guides - `/memory:tech-research` |
| **Project docs** | Skill({project-name}) | Project modules/architecture - `/memory:skill-memory` |
| **Semantic discovery** | codebase-retrieval | Find files relevant to task/feature context |
| **Pattern matching** | rg | Search code content with regex |
| **File name lookup** | find | Locate files by name patterns |
| **Architecture** | get_modules_by_depth.sh | Understand program structure |

## 🔧 Quick Command Reference

```bash
# SKILL Packages (FIRST PRIORITY - fastest context loading)
Skill(command: "workflow-progress")  # Workflow: WFS sessions history, lessons, conflicts
Skill(command: "react-dev")          # Tech: React APIs, patterns, best practices
Skill(command: "claude_dms3")        # Project: Project modules, architecture, examples

# Semantic File Discovery (codebase-retrieval)
cd [directory] && gemini -p "
PURPOSE: Discover files relevant to task/feature
TASK: List all files related to [task/feature description]
MODE: analysis
CONTEXT: @**/*
EXPECTED: Relevant file paths with relevance explanation
RULES: Focus on direct relevance to task requirements
"

# Program Architecture (MANDATORY before planning)
ccw tool exec get_modules_by_depth '{}'

# Content Search (rg preferred)
rg "pattern" --type js -n        # Search JS files with line numbers
rg -i "case-insensitive"         # Ignore case
rg -C 3 "context"                # Show 3 lines before/after

# File Search
find . -name "*.ts" -type f      # Find TypeScript files
find . -path "*/node_modules" -prune -o -name "*.js" -print

# Workflow Examples
rg "IMPL-\d+" .workflow/ --type json                    # Find task IDs
find .workflow/ -name "*.json" -path "*/.task/*"        # Locate task files
rg "status.*pending" .workflow/.task/                   # Find pending tasks
```

## ⚡ Performance Tips

- **rg > grep** for content search
- **Use --type filters** to limit file types
- **Exclude dirs**: `--glob '!node_modules'`
- **Use -F** for literal strings (no regex)
