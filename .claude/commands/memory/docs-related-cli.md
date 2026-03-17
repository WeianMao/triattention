---
name: docs-related-cli
description: Generate/update documentation for git-changed modules using CLI execution with batched agents (4 modules/agent) and gemini→qwen→codex fallback, <15 modules uses direct parallel
argument-hint: "[--tool <gemini|qwen|codex>]"
---

# Related Documentation Generation - CLI Mode (/memory:docs-related-cli)

## Overview

Orchestrates context-aware documentation generation/update for changed modules using CLI-based execution with batched agents and automatic tool fallback (gemini→qwen→codex).

**Parameters**:
- `--tool <gemini|qwen|codex>`: Primary tool (default: gemini)

**Execution Flow**:
1. Change Detection → 2. Plan Presentation → 3. Batched Execution → 4. Verification

## Core Rules

1. **Detect Changes First**: Use git diff to identify affected modules
2. **Wait for Approval**: Present plan, no execution without user confirmation
3. **Execution Strategy**:
   - **<15 modules**: Direct parallel execution (max 4 concurrent per depth, no agent overhead)
   - **≥15 modules**: Agent batch processing (4 modules/agent, 73% overhead reduction)
4. **Tool Fallback**: Auto-retry with fallback tools on failure
5. **Depth Sequential**: Process depths N→0, parallel batches within depth (both modes)
6. **Related Mode**: Generate/update only changed modules and their parent contexts
7. **Single Strategy**: Always use `single` strategy (incremental update)

## Tool Fallback Hierarchy

```javascript
--tool gemini  →  [gemini, qwen, codex]  // default
--tool qwen    →  [qwen, gemini, codex]
--tool codex   →  [codex, gemini, qwen]
```

**Trigger**: Non-zero exit code from generation script

| Tool   | Best For                       | Fallback To    |
|--------|--------------------------------|----------------|
| gemini | Documentation, patterns        | qwen → codex   |
| qwen   | Architecture, system design    | gemini → codex |
| codex  | Implementation, code quality   | gemini → qwen  |

## Phase 1: Change Detection & Analysis

```javascript
// Get project metadata
Bash({command: "pwd && basename \"$(pwd)\" && git rev-parse --show-toplevel 2>/dev/null || pwd", run_in_background: false});

// Detect changed modules
Bash({command: "ccw tool exec detect_changed_modules '{\"format\":\"list\"}'", run_in_background: false});

// Cache git changes
Bash({command: "git add -A 2>/dev/null || true", run_in_background: false});
```

**Parse output** `depth:N|path:<PATH>|change:<TYPE>|type:<code|navigation>` to extract affected modules.

**Smart filter**: Auto-detect and skip tests/build/config/vendor based on project tech stack (Node.js/Python/Go/Rust/etc).

**Fallback**: If no changes detected, use recent modules (first 10 by depth).

## Phase 2: Plan Presentation

**Present filtered plan**:
```
Related Documentation Generation Plan:
  Tool: gemini (fallback: qwen → codex)
  Changed: 4 modules | Batching: 4 modules/agent
  Project: myproject
  Output: .workflow/docs/myproject/

  Will generate/update docs for:
  - ./src/api/auth (5 files, type: code) [new module]
  - ./src/api (12 files, type: code) [parent of changed auth/]
  - ./src (8 files, type: code) [parent context]
  - . (14 files, type: code) [root level]

  Documentation Strategy:
  - Strategy: single (all modules - incremental update)
  - Output: API.md + README.md (code folders), README.md only (navigation folders)
  - Context: Current dir code + child docs

  Auto-skipped (12 paths):
  - Tests: ./src/api/auth.test.ts (8 paths)
  - Config: tsconfig.json (3 paths)
  - Other: node_modules (1 path)

  Agent allocation:
  - Depth 3 (1 module): 1 agent [1]
  - Depth 2 (1 module): 1 agent [1]
  - Depth 1 (1 module): 1 agent [1]
  - Depth 0 (1 module): 1 agent [1]

  Estimated time: ~5-10 minutes

  Confirm execution? (y/n)
```

**Decision logic**:
- User confirms "y": Proceed with execution
- User declines "n": Abort, no changes
- <15 modules: Direct execution
- ≥15 modules: Agent batch execution

## Phase 3A: Direct Execution (<15 modules)

**Strategy**: Parallel execution within depth (max 4 concurrent), no agent overhead.

**CRITICAL**: All Bash commands use `run_in_background: false` for synchronous execution.

```javascript
let project_name = detect_project_name();

for (let depth of sorted_depths.reverse()) {  // N → 0
  let batches = batch_modules(modules_by_depth[depth], 4);

  for (let batch of batches) {
    let parallel_tasks = batch.map(module => {
      return async () => {
        for (let tool of tool_order) {
          Bash({
            command: `cd ${module.path} && ccw tool exec generate_module_docs '{"strategy":"single","sourcePath":".","projectName":"${project_name}","tool":"${tool}"}'`,
            run_in_background: false
          });
          if (bash_result.exit_code === 0) {
            report(`✅ ${module.path} docs generated with ${tool}`);
            return true;
          }
        }
        report(`❌ FAILED: ${module.path} failed all tools`);
        return false;
      };
    });
    await Promise.all(parallel_tasks.map(task => task()));
  }
}
```

## Phase 3B: Agent Batch Execution (≥15 modules)

### Batching Strategy

```javascript
// Batch modules into groups of 4
function batch_modules(modules, batch_size = 4) {
  let batches = [];
  for (let i = 0; i < modules.length; i += batch_size) {
    batches.push(modules.slice(i, i + batch_size));
  }
  return batches;
}
// Examples: 10→[4,4,2] | 8→[4,4] | 3→[3]
```

### Coordinator Orchestration

```javascript
let modules_by_depth = group_by_depth(changed_modules);
let tool_order = construct_tool_order(primary_tool);
let project_name = detect_project_name();

for (let depth of sorted_depths.reverse()) {  // N → 0
  let batches = batch_modules(modules_by_depth[depth], 4);
  let worker_tasks = [];

  for (let batch of batches) {
    worker_tasks.push(
      Task(
        subagent_type="memory-bridge",
        description=`Generate docs for ${batch.length} modules at depth ${depth}`,
        prompt=generate_batch_worker_prompt(batch, tool_order, depth, project_name, "related")
      )
    );
  }

  await parallel_execute(worker_tasks);  // Batches run in parallel
}
```

### Batch Worker Prompt Template

```
PURPOSE: Generate/update documentation for assigned modules with tool fallback (related mode)

TASK:
Generate documentation for the following modules based on recent changes. For each module, try tools in order until success.

PROJECT: {{project_name}}
OUTPUT: .workflow/docs/{{project_name}}/

MODULES:
{{module_path_1}} (type: {{folder_type_1}})
{{module_path_2}} (type: {{folder_type_2}})
{{module_path_3}} (type: {{folder_type_3}})
{{module_path_4}} (type: {{folder_type_4}})

TOOLS (try in order):
1. {{tool_1}}
2. {{tool_2}}
3. {{tool_3}}

EXECUTION:
For each module above:
  1. Try tool 1:
     Bash({
       command: `cd "{{module_path}}" && ccw tool exec generate_module_docs '{"strategy":"single","sourcePath":".","projectName":"{{project_name}}","tool":"{{tool_1}}"}'`,
       run_in_background: false
     })
     → Success: Report "✅ {{module_path}} docs generated with {{tool_1}}", proceed to next module
     → Failure: Try tool 2
  2. Try tool 2:
     Bash({
       command: `cd "{{module_path}}" && ccw tool exec generate_module_docs '{"strategy":"single","sourcePath":".","projectName":"{{project_name}}","tool":"{{tool_2}}"}'`,
       run_in_background: false
     })
     → Success: Report "✅ {{module_path}} docs generated with {{tool_2}}", proceed to next module
     → Failure: Try tool 3
  3. Try tool 3:
     Bash({
       command: `cd "{{module_path}}" && ccw tool exec generate_module_docs '{"strategy":"single","sourcePath":".","projectName":"{{project_name}}","tool":"{{tool_3}}"}'`,
       run_in_background: false
     })
     → Success: Report "✅ {{module_path}} docs generated with {{tool_3}}", proceed to next module
     → Failure: Report "❌ FAILED: {{module_path}} failed all tools", proceed to next module

FOLDER TYPE HANDLING:
  - code: Generate API.md + README.md
  - navigation: Generate README.md only

REPORTING:
Report final summary with:
- Total processed: X modules
- Successful: Y modules
- Failed: Z modules
- Tool usage: {{tool_1}}:X, {{tool_2}}:Y, {{tool_3}}:Z
```

## Phase 4: Verification

```javascript
// Check documentation files created/updated
Bash({command: 'find .workflow/docs -type f -name "*.md" 2>/dev/null | wc -l', run_in_background: false});

// Display recent changes
Bash({command: 'find .workflow/docs -type f -name "*.md" -mmin -60 2>/dev/null', run_in_background: false});
```

**Aggregate results**:
```
Documentation Generation Summary:
  Total: 4 | Success: 4 | Failed: 0

  Tool usage:
  - gemini: 4 modules
  - qwen: 0 modules (fallback)
  - codex: 0 modules

  Changes:
  .workflow/docs/myproject/src/api/auth/API.md      (new)
  .workflow/docs/myproject/src/api/auth/README.md   (new)
  .workflow/docs/myproject/src/api/API.md           (updated)
  .workflow/docs/myproject/src/api/README.md        (updated)
  .workflow/docs/myproject/src/API.md               (updated)
  .workflow/docs/myproject/src/README.md            (updated)
  .workflow/docs/myproject/API.md                   (updated)
  .workflow/docs/myproject/README.md                (updated)
```

## Execution Summary

**Module Count Threshold**:
- **<15 modules**: Coordinator executes Phase 3A (Direct Execution)
- **≥15 modules**: Coordinator executes Phase 3B (Agent Batch Execution)

**Agent Hierarchy** (for ≥15 modules):
- **Coordinator**: Handles batch division, spawns worker agents per depth
- **Worker Agents**: Each processes 4 modules with tool fallback (related mode)

## Error Handling

**Batch Worker**:
- Tool fallback per module (auto-retry)
- Batch isolation (failures don't propagate)
- Clear per-module status reporting

**Coordinator**:
- No changes: Use fallback (recent 10 modules)
- User decline: No execution
- Verification fail: Report incomplete modules
- Partial failures: Continue execution, report failed modules

**Fallback Triggers**:
- Non-zero exit code
- Script timeout
- Unexpected output

## Output Structure

```
.workflow/docs/{project_name}/
├── src/                           # Mirrors source structure
│   ├── modules/
│   │   ├── README.md
│   │   ├── auth/
│   │   │   ├── API.md             # Updated based on code changes
│   │   │   └── README.md          # Updated based on code changes
│   │   └── api/
│   │       ├── API.md
│   │       └── README.md
│   └── utils/
│       └── README.md
└── README.md
```

## Usage Examples

```bash
# Daily development documentation update
/memory:docs-related-cli

# After feature work with specific tool
/memory:docs-related-cli --tool qwen

# Code quality documentation review after implementation
/memory:docs-related-cli --tool codex
```

## Key Advantages

**Efficiency**: 30 modules → 8 agents (73% reduction)
**Resilience**: 3-tier fallback per module
**Performance**: Parallel batches, no concurrency limits
**Context-aware**: Updates based on actual git changes
**Fast**: Only affected modules, not entire project
**Incremental**: Single strategy for focused updates

## Coordinator Checklist

- Parse `--tool` (default: gemini)
- Get project metadata (name, root)
- Detect changed modules via detect_changed_modules.sh
- **Smart filter modules** (auto-detect tech stack, skip tests/build/config/vendor)
- Cache git changes
- Apply fallback if no changes (recent 10 modules)
- Construct tool fallback order
- **Present filtered plan** with skip reasons and change types
- **Wait for y/n confirmation**
- Determine execution mode:
  - **<15 modules**: Direct execution (Phase 3A)
    - For each depth (N→0): Sequential module updates with tool fallback
  - **≥15 modules**: Agent batch execution (Phase 3B)
    - For each depth (N→0): Batch modules (4 per batch), spawn batch workers in parallel
- Wait for depth/batch completion
- Aggregate results
- Verification check (documentation files created/updated)
- Display summary + recent changes

## Comparison with Full Documentation Generation

| Aspect | Related Generation | Full Generation |
|--------|-------------------|-----------------|
| **Scope** | Changed modules only | All project modules |
| **Speed** | Fast (minutes) | Slower (10-30 min) |
| **Use case** | Daily development | Initial setup, major refactoring |
| **Strategy** | `single` (all) | `full` (L3) + `single` (L1-2) |
| **Trigger** | After commits | After setup or major changes |
| **Batching** | 4 modules/agent | 4 modules/agent |
| **Fallback** | gemini→qwen→codex | gemini→qwen→codex |
| **Complexity threshold** | ≤15 modules | ≤20 modules |

## Template Reference

Templates used from `~/.claude/workflows/cli-templates/prompts/documentation/`:
- `api.txt`: Code API documentation
- `module-readme.txt`: Module purpose, usage, dependencies
- `folder-navigation.txt`: Navigation README for folders

## Related Commands

- `/memory:docs-full-cli` - Full project documentation generation
- `/memory:docs` - Agent-based documentation planning workflow
- `/memory:update-related` - Update CLAUDE.md for changed modules
