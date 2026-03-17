---
name: update-full
description: Update all CLAUDE.md files using layer-based execution (Layer 3→1) with batched agents (4 modules/agent) and gemini→qwen→codex fallback, <20 modules uses direct parallel
argument-hint: "[--tool gemini|qwen|codex] [--path <directory>]"
---

# Full Documentation Update (/memory:update-full)

## Overview

Orchestrates project-wide CLAUDE.md updates using batched agent execution with automatic tool fallback and 3-layer architecture support.

**Parameters**:
- `--tool <gemini|qwen|codex>`: Primary tool (default: gemini)
- `--path <directory>`: Target specific directory (default: entire project)

**Execution Flow**: Discovery → Plan Presentation → Execution → Safety Verification

## 3-Layer Architecture & Auto-Strategy Selection

### Layer Definition & Strategy Assignment

| Layer | Depth | Strategy | Purpose | Context Pattern |
|-------|-------|----------|---------|----------------|
| **Layer 3** (Deepest) | ≥3 | `multi-layer` | Handle unstructured files, generate docs for all subdirectories | `@**/*` (all files) |
| **Layer 2** (Middle) | 1-2 | `single-layer` | Aggregate from children + current code | `@*/CLAUDE.md @*.{ts,tsx,js,...}` |
| **Layer 1** (Top) | 0 | `single-layer` | Aggregate from children + current code | `@*/CLAUDE.md @*.{ts,tsx,js,...}` |

**Update Direction**: Layer 3 → Layer 2 → Layer 1 (bottom-up dependency flow)

**Strategy Auto-Selection**: Strategies are automatically determined by directory depth - no user configuration needed.

### Strategy Details

#### Multi-Layer Strategy (Layer 3 Only)
- **Use Case**: Deepest directories with unstructured file layouts
- **Behavior**: Generates CLAUDE.md for current directory AND each subdirectory containing files
- **Context**: All files in current directory tree (`@**/*`)


#### Single-Layer Strategy (Layers 1-2)
- **Use Case**: Upper layers that aggregate from existing documentation
- **Behavior**: Generates CLAUDE.md only for current directory
- **Context**: Direct children CLAUDE.md files + current directory code files

### Example Flow
```
src/auth/handlers/ (depth 3) → MULTI-LAYER STRATEGY
  CONTEXT: @**/* (all files in handlers/ and subdirs)
  GENERATES: ./CLAUDE.md + CLAUDE.md in each subdir with files
  ↓
src/auth/ (depth 2) → SINGLE-LAYER STRATEGY
  CONTEXT: @*/CLAUDE.md @*.ts (handlers/CLAUDE.md + current code)
  GENERATES: ./CLAUDE.md only
  ↓
src/ (depth 1) → SINGLE-LAYER STRATEGY
  CONTEXT: @*/CLAUDE.md (auth/CLAUDE.md, utils/CLAUDE.md)
  GENERATES: ./CLAUDE.md only
  ↓
./ (depth 0) → SINGLE-LAYER STRATEGY
  CONTEXT: @*/CLAUDE.md (src/CLAUDE.md, tests/CLAUDE.md)
  GENERATES: ./CLAUDE.md only
```

## Core Execution Rules

1. **Analyze First**: Git cache + module discovery before updates
2. **Wait for Approval**: Present plan, no execution without user confirmation
3. **Execution Strategy**:
   - **<20 modules**: Direct parallel execution (max 4 concurrent per layer)
   - **≥20 modules**: Agent batch processing (4 modules/agent, 73% overhead reduction)
4. **Tool Fallback**: Auto-retry with fallback tools on failure
5. **Layer Sequential**: Process layers 3→2→1 (bottom-up), parallel batches within layer
6. **Safety Check**: Verify only CLAUDE.md files modified
7. **Layer-based Grouping**: Group modules by LAYER (not depth) for execution

## Tool Fallback Hierarchy

```javascript
--tool gemini  →  [gemini, qwen, codex]  // default
--tool qwen    →  [qwen, gemini, codex]
--tool codex   →  [codex, gemini, qwen]
```

**Trigger**: Non-zero exit code from update script

| Tool   | Best For                       | Fallback To    |
|--------|--------------------------------|----------------|
| gemini | Documentation, patterns        | qwen → codex   |
| qwen   | Architecture, system design    | gemini → codex |
| codex  | Implementation, code quality   | gemini → qwen  |

## Execution Phases

### Phase 1: Discovery & Analysis

```javascript
// Cache git changes
Bash({command: "git add -A 2>/dev/null || true", run_in_background: false});

// Get module structure
Bash({command: "ccw tool exec get_modules_by_depth '{\"format\":\"list\"}'", run_in_background: false});

// OR with --path
Bash({command: "cd <target-path> && ccw tool exec get_modules_by_depth '{\"format\":\"list\"}'", run_in_background: false});
```

**Parse output** `depth:N|path:<PATH>|...` to extract module paths and count.

**Smart filter**: Auto-detect and skip tests/build/config/docs based on project tech stack.

### Phase 2: Plan Presentation

**For <20 modules**:
```
Update Plan:
  Tool: gemini (fallback: qwen → codex)
  Total: 7 modules
  Execution: Direct parallel (< 20 modules threshold)

  Will update:
  - ./core/interfaces (12 files) - depth 2 [Layer 2] - single-layer strategy
  - ./core (22 files) - depth 1 [Layer 2] - single-layer strategy
  - ./models (9 files) - depth 1 [Layer 2] - single-layer strategy
  - ./utils (12 files) - depth 1 [Layer 2] - single-layer strategy
  - . (5 files) - depth 0 [Layer 1] - single-layer strategy

  Context Strategy (Auto-Selected):
  - Layer 2 (depth 1-2): @*/CLAUDE.md + current code files
  - Layer 1 (depth 0): @*/CLAUDE.md + current code files

  Auto-skipped: ./tests, __pycache__, setup.py (15 paths)
  Execution order: Layer 2 → Layer 1
  Estimated time: ~5-10 minutes

  Confirm execution? (y/n)
```

**For ≥20 modules**:
```
Update Plan:
  Tool: gemini (fallback: qwen → codex)
  Total: 31 modules
  Execution: Agent batch processing (4 modules/agent)

  Will update:
  - ./src/features/auth (12 files) - depth 3 [Layer 3] - multi-layer strategy
  - ./.claude/commands/cli (6 files) - depth 3 [Layer 3] - multi-layer strategy
  - ./src/utils (8 files) - depth 2 [Layer 2] - single-layer strategy
  ...

  Context Strategy (Auto-Selected):
  - Layer 3 (depth ≥3): @**/* (all files)
  - Layer 2 (depth 1-2): @*/CLAUDE.md + current code files
  - Layer 1 (depth 0): @*/CLAUDE.md + current code files
  
  Auto-skipped: ./tests, __pycache__, setup.py (15 paths)
  Execution order: Layer 2 → Layer 1
  Estimated time: ~5-10 minutes

  Agent allocation (by LAYER):
  - Layer 3 (14 modules, depth ≥3): 4 agents [4, 4, 4, 2]
  - Layer 2 (15 modules, depth 1-2): 4 agents [4, 4, 4, 3]
  - Layer 1 (2 modules, depth 0): 1 agent [2]

  Estimated time: ~15-25 minutes

  Confirm execution? (y/n)
```

### Phase 3A: Direct Execution (<20 modules)

**Strategy**: Parallel execution within layer (max 4 concurrent), no agent overhead.

**CRITICAL**: All Bash commands use `run_in_background: false` for synchronous execution.

```javascript
for (let layer of [3, 2, 1]) {
  if (modules_by_layer[layer].length === 0) continue;
  let batches = batch_modules(modules_by_layer[layer], 4);

  for (let batch of batches) {
    let parallel_tasks = batch.map(module => {
      return async () => {
        let strategy = module.depth >= 3 ? "multi-layer" : "single-layer";
        for (let tool of tool_order) {
          Bash({
            command: `cd ${module.path} && ccw tool exec update_module_claude '{"strategy":"${strategy}","path":".","tool":"${tool}"}'`,
            run_in_background: false
          });
          if (bash_result.exit_code === 0) {
            report(`✅ ${module.path} (Layer ${layer}) updated with ${tool}`);
            return true;
          }
        }
        report(`❌ FAILED: ${module.path} (Layer ${layer}) failed all tools`);
        return false;
      };
    });
    await Promise.all(parallel_tasks.map(task => task()));
  }
}
```

### Phase 3B: Agent Batch Execution (≥20 modules)

**Strategy**: Batch modules into groups of 4, spawn memory-bridge agents per batch.

```javascript
// Group modules by LAYER and batch within each layer
let modules_by_layer = group_by_layer(module_list);
let tool_order = construct_tool_order(primary_tool);

for (let layer of [3, 2, 1]) {
  if (modules_by_layer[layer].length === 0) continue;

  let batches = batch_modules(modules_by_layer[layer], 4);
  let worker_tasks = [];

  for (let batch of batches) {
    worker_tasks.push(
      Task(
        subagent_type="memory-bridge",
        description=`Update ${batch.length} modules in Layer ${layer}`,
        prompt=generate_batch_worker_prompt(batch, tool_order, layer)
      )
    );
  }

  await parallel_execute(worker_tasks);
}
```

**Batch Worker Prompt Template**:
```
PURPOSE: Update CLAUDE.md for assigned modules with tool fallback

TASK: Update documentation for assigned modules using specified strategies.

MODULES:
{{module_path_1}} (strategy: {{strategy_1}})
{{module_path_2}} (strategy: {{strategy_2}})
...

TOOLS (try in order): {{tool_1}}, {{tool_2}}, {{tool_3}}

EXECUTION SCRIPT: ccw tool exec update_module_claude
  - Accepts strategy parameter: multi-layer | single-layer
  - Tool execution via direct CLI commands (gemini/qwen/codex)

EXECUTION FLOW (for each module):
  1. Tool fallback loop (exit on first success):
     for tool in {{tool_1}} {{tool_2}} {{tool_3}}; do
       Bash({
         command: `cd "{{module_path}}" && ccw tool exec update_module_claude '{"strategy":"{{strategy}}","path":".","tool":"${tool}"}'`,
         run_in_background: false
       })
       exit_code=$?

       if [ $exit_code -eq 0 ]; then
         report "✅ {{module_path}} updated with $tool"
         break
       else
         report "⚠️  {{module_path}} failed with $tool, trying next..."
         continue
       fi
     done

  2. Handle complete failure (all tools failed):
     if [ $exit_code -ne 0 ]; then
       report "❌ FAILED: {{module_path}} - all tools exhausted"
       # Continue to next module (do not abort batch)
     fi

FAILURE HANDLING:
  - Module-level isolation: One module's failure does not affect others
  - Exit code detection: Non-zero exit code triggers next tool
  - Exhaustion reporting: Log modules where all tools failed
  - Batch continuation: Always process remaining modules

REPORTING FORMAT:
  Per-module status:
    ✅ path/to/module updated with {tool}
    ⚠️  path/to/module failed with {tool}, trying next...
    ❌ FAILED: path/to/module - all tools exhausted
```
### Phase 4: Safety Verification

```javascript
// Check only CLAUDE.md files modified
Bash({command: 'git diff --cached --name-only | grep -v "CLAUDE.md" || echo "Only CLAUDE.md files modified"', run_in_background: false});

// Display status
Bash({command: "git status --short", run_in_background: false});
```

**Result Summary**:
```
Update Summary:
  Total: 31 | Success: 29 | Failed: 2
  Tool usage: gemini: 25, qwen: 4, codex: 0
  Failed: path1, path2
```

## Error Handling

**Batch Worker**: Tool fallback per module, batch isolation, clear status reporting
**Coordinator**: Invalid path abort, user decline handling, safety check with auto-revert
**Fallback Triggers**: Non-zero exit code, script timeout, unexpected output

## Usage Examples

```bash
# Full project update (auto-strategy selection)
/memory:update-full

# Target specific directory
/memory:update-full --path .claude
/memory:update-full --path src/features/auth

# Use specific tool
/memory:update-full --tool qwen
/memory:update-full --path .claude --tool qwen
```

## Key Advantages

- **Efficiency**: 30 modules → 8 agents (73% reduction from sequential)
- **Resilience**: 3-tier tool fallback per module
- **Performance**: Parallel batches, no concurrency limits
- **Observability**: Per-module tool usage, batch-level metrics
- **Automation**: Zero configuration - strategy auto-selected by directory depth
