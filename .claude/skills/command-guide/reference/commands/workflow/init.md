---
name: init
description: Initialize project-level state with intelligent project analysis using cli-explore-agent
argument-hint: "[--regenerate]"
examples:
  - /workflow:init
  - /workflow:init --regenerate
---

# Workflow Init Command (/workflow:init)

## Overview
Initialize `.workflow/project.json` with comprehensive project understanding by delegating analysis to **cli-explore-agent**.

**Note**: This command may be called by other workflow commands. Upon completion, return immediately to continue the calling workflow without interrupting the task flow.

## Usage
```bash
/workflow:init                 # Initialize (skip if exists)
/workflow:init --regenerate    # Force regeneration
```

## Execution Process

```
Input Parsing:
   └─ Parse --regenerate flag → regenerate = true | false

Decision:
   ├─ EXISTS + no --regenerate → Exit: "Already initialized"
   ├─ EXISTS + --regenerate → Backup existing → Continue analysis
   └─ NOT_FOUND → Continue analysis

Analysis Flow:
   ├─ Get project metadata (name, root)
   ├─ Invoke cli-explore-agent
   │   ├─ Structural scan (get_modules_by_depth.sh, find, wc)
   │   ├─ Semantic analysis (Gemini CLI)
   │   ├─ Synthesis and merge
   │   └─ Write .workflow/project.json
   └─ Display summary

Output:
   └─ .workflow/project.json (+ .backup if regenerate)
```

## Implementation

### Step 1: Parse Input and Check Existing State

**Parse --regenerate flag**:
```javascript
const regenerate = $ARGUMENTS.includes('--regenerate')
```

**Check existing state**:

```bash
bash(test -f .workflow/project.json && echo "EXISTS" || echo "NOT_FOUND")
```

**If EXISTS and no --regenerate**: Exit early
```
Project already initialized at .workflow/project.json
Use /workflow:init --regenerate to rebuild
Use /workflow:status --project to view state
```

### Step 2: Get Project Metadata

```bash
bash(basename "$(git rev-parse --show-toplevel 2>/dev/null || pwd)")
bash(git rev-parse --show-toplevel 2>/dev/null || pwd)
bash(mkdir -p .workflow)
```

### Step 3: Invoke cli-explore-agent

**For --regenerate**: Backup and preserve existing data
```bash
bash(cp .workflow/project.json .workflow/project.json.backup)
```

**Delegate analysis to agent**:

```javascript
Task(
  subagent_type="cli-explore-agent",
  description="Deep project analysis",
  prompt=`
Analyze project for workflow initialization and generate .workflow/project.json.

## MANDATORY FIRST STEPS
1. Execute: cat ~/.claude/workflows/cli-templates/schemas/project-json-schema.json (get schema reference)
2. Execute: ccw tool exec get_modules_by_depth '{}' (get project structure)

## Task
Generate complete project.json with:
- project_name: ${projectName}
- initialized_at: current ISO timestamp
- overview: {description, technology_stack, architecture, key_components}
- features: ${regenerate ? 'preserve from backup' : '[] (empty)'}
- development_index: ${regenerate ? 'preserve from backup' : '{feature: [], enhancement: [], bugfix: [], refactor: [], docs: []}'}
- statistics: ${regenerate ? 'preserve from backup' : '{total_features: 0, total_sessions: 0, last_updated}'}
- _metadata: {initialized_by: "cli-explore-agent", analysis_timestamp, analysis_mode}

## Analysis Requirements

**Technology Stack**:
- Languages: File counts, mark primary
- Frameworks: From package.json, requirements.txt, go.mod, etc.
- Build tools: npm, cargo, maven, webpack, vite
- Test frameworks: jest, pytest, go test, junit

**Architecture**:
- Style: MVC, microservices, layered (from structure & imports)
- Layers: presentation, business-logic, data-access
- Patterns: singleton, factory, repository
- Key components: 5-10 modules {name, path, description, importance}

## Execution
1. Structural scan: get_modules_by_depth.sh, find, wc -l
2. Semantic analysis: Gemini for patterns/architecture
3. Synthesis: Merge findings
4. ${regenerate ? 'Merge with preserved features/development_index/statistics from .workflow/project.json.backup' : ''}
5. Write JSON: Write('.workflow/project.json', jsonContent)
6. Report: Return brief completion summary

Project root: ${projectRoot}
`
)
```

### Step 4: Display Summary

```javascript
const projectJson = JSON.parse(Read('.workflow/project.json'));

console.log(`
✓ Project initialized successfully

## Project Overview
Name: ${projectJson.project_name}
Description: ${projectJson.overview.description}

### Technology Stack
Languages: ${projectJson.overview.technology_stack.languages.map(l => l.name).join(', ')}
Frameworks: ${projectJson.overview.technology_stack.frameworks.join(', ')}

### Architecture
Style: ${projectJson.overview.architecture.style}
Components: ${projectJson.overview.key_components.length} core modules

---
Project state: .workflow/project.json
${regenerate ? 'Backup: .workflow/project.json.backup' : ''}
`);
```

## Error Handling

**Agent Failure**: Fall back to basic initialization with placeholder overview
**Missing Tools**: Agent uses Qwen fallback or bash-only
**Empty Project**: Create minimal JSON with all gaps identified
