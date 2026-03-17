---
name: docs-full-cli
description: Generate full project documentation using CLI execution (Layer 3→1) with batched agents (4 modules/agent) and gemini→qwen→codex fallback, <20 modules uses direct parallel
argument-hint: "[path] [--tool <gemini|qwen|codex>]"
---

# Full Documentation Generation - CLI Mode (/memory:docs-full-cli)

## Overview

Orchestrates project-wide documentation generation using CLI-based execution with batched agents and automatic tool fallback.

**Parameters**:
- `path`: Target directory (default: current directory)
- `--tool <gemini|qwen|codex>`: Primary tool (default: gemini)

**Execution Flow**: Discovery → Plan Presentation → Execution → Verification

## 3-Layer Architecture & Auto-Strategy Selection

### Layer Definition & Strategy Assignment

| Layer | Depth | Strategy | Purpose | Context Pattern |
|-------|-------|----------|---------|----------------|
| **Layer 3** (Deepest) | ≥3 | `full` | Generate docs for all subdirectories with code | `@**/*` (all files) |
| **Layer 2** (Middle) | 1-2 | `single` | Current dir + child docs | `@*/API.md @*/README.md @*.{ts,tsx,js,...}` |
| **Layer 1** (Top) | 0 | `single` | Current dir + child docs | `@*/API.md @*/README.md @*.{ts,tsx,js,...}` |

**Generation Direction**: Layer 3 → Layer 2 → Layer 1 (bottom-up dependency flow)

**Strategy Auto-Selection**: Strategies are automatically determined by directory depth - no user configuration needed.

### Strategy Details

#### Full Strategy (Layer 3 Only)
- **Use Case**: Deepest directories with comprehensive file coverage
- **Behavior**: Generates API.md + README.md for current directory AND subdirectories containing code
- **Context**: All files in current directory tree (`@**/*`)
- **Output**: `.workflow/docs/{project_name}/{path}/API.md` + `README.md`


#### Single Strategy (Layers 1-2)
- **Use Case**: Upper layers that aggregate from existing documentation
- **Behavior**: Generates API.md + README.md only in current directory
- **Context**: Direct children docs + current directory code files
- **Output**: `.workflow/docs/{project_name}/{path}/API.md` + `README.md`

### Example Flow
```
src/auth/handlers/ (depth 3) → FULL STRATEGY
  CONTEXT: @**/* (all files in handlers/ and subdirs)
  GENERATES: .workflow/docs/project/src/auth/handlers/{API.md,README.md} + subdirs
  ↓
src/auth/ (depth 2) → SINGLE STRATEGY
  CONTEXT: @*/API.md @*/README.md @*.ts (handlers docs + current code)
  GENERATES: .workflow/docs/project/src/auth/{API.md,README.md} only
  ↓
src/ (depth 1) → SINGLE STRATEGY
  CONTEXT: @*/API.md @*/README.md (auth docs, utils docs)
  GENERATES: .workflow/docs/project/src/{API.md,README.md} only
  ↓
./ (depth 0) → SINGLE STRATEGY
  CONTEXT: @*/API.md @*/README.md (src docs, tests docs)
  GENERATES: .workflow/docs/project/{API.md,README.md} only
```

## Core Execution Rules

1. **Analyze First**: Module discovery + folder classification before generation
2. **Wait for Approval**: Present plan, no execution without user confirmation
3. **Execution Strategy**:
   - **<20 modules**: Direct parallel execution (max 4 concurrent per layer)
   - **≥20 modules**: Agent batch processing (4 modules/agent, 73% overhead reduction)
4. **Tool Fallback**: Auto-retry with fallback tools on failure
5. **Layer Sequential**: Process layers 3→2→1 (bottom-up), parallel batches within layer
6. **Safety Check**: Verify only docs files modified in .workflow/docs/
7. **Layer-based Grouping**: Group modules by LAYER (not depth) for execution

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

## Execution Phases

### Phase 1: Discovery & Analysis

```javascript
// Get project metadata
Bash({command: "pwd && basename \"$(pwd)\" && git rev-parse --show-toplevel 2>/dev/null || pwd", run_in_background: false});

// Get module structure with classification
Bash({command: "ccw tool exec get_modules_by_depth '{\"format\":\"list\"}' | ccw tool exec classify_folders '{}'", run_in_background: false});

// OR with path parameter
Bash({command: "cd <target-path> && ccw tool exec get_modules_by_depth '{\"format\":\"list\"}' | ccw tool exec classify_folders '{}'", run_in_background: false});
```

**Parse output** `depth:N|path:<PATH>|type:<code|navigation>|...` to extract module paths, types, and count.

**Smart filter**: Auto-detect and skip tests/build/config/vendor based on project tech stack.

### Phase 2: Plan Presentation

**For <20 modules**:
```
Documentation Generation Plan:
  Tool: gemini (fallback: qwen → codex)
  Total: 7 modules
  Execution: Direct parallel (< 20 modules threshold)
  Project: myproject
  Output: .workflow/docs/myproject/

  Will generate docs for:
  - ./core/interfaces (12 files, type: code) - depth 2 [Layer 2] - single strategy
  - ./core (22 files, type: code) - depth 1 [Layer 2] - single strategy
  - ./models (9 files, type: code) - depth 1 [Layer 2] - single strategy
  - ./utils (12 files, type: navigation) - depth 1 [Layer 2] - single strategy
  - . (5 files, type: code) - depth 0 [Layer 1] - single strategy

  Documentation Strategy (Auto-Selected):
  - Layer 2 (depth 1-2): API.md + README.md (current dir only, reference child docs)
  - Layer 1 (depth 0): API.md + README.md (current dir only, reference child docs)

  Output Structure:
  - Code folders: API.md + README.md
  - Navigation folders: README.md only

  Auto-skipped: ./tests, __pycache__, node_modules (15 paths)
  Execution order: Layer 2 → Layer 1
  Estimated time: ~5-10 minutes

  Confirm execution? (y/n)
```

**For ≥20 modules**:
```
Documentation Generation Plan:
  Tool: gemini (fallback: qwen → codex)
  Total: 31 modules
  Execution: Agent batch processing (4 modules/agent)
  Project: myproject
  Output: .workflow/docs/myproject/

  Will generate docs for:
  - ./src/features/auth (12 files, type: code) - depth 3 [Layer 3] - full strategy
  - ./.claude/commands/cli (6 files, type: code) - depth 3 [Layer 3] - full strategy
  - ./src/utils (8 files, type: code) - depth 2 [Layer 2] - single strategy
  ...

  Documentation Strategy (Auto-Selected):
  - Layer 3 (depth ≥3): API.md + README.md (all subdirs with code)
  - Layer 2 (depth 1-2): API.md + README.md (current dir only)
  - Layer 1 (depth 0): API.md + README.md (current dir only)

  Output Structure:
  - Code folders: API.md + README.md
  - Navigation folders: README.md only

  Auto-skipped: ./tests, __pycache__, node_modules (15 paths)
  Execution order: Layer 3 → Layer 2 → Layer 1

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
let project_name = detect_project_name();

for (let layer of [3, 2, 1]) {
  if (modules_by_layer[layer].length === 0) continue;
  let batches = batch_modules(modules_by_layer[layer], 4);

  for (let batch of batches) {
    let parallel_tasks = batch.map(module => {
      return async () => {
        let strategy = module.depth >= 3 ? "full" : "single";
        for (let tool of tool_order) {
          Bash({
            command: `cd ${module.path} && ccw tool exec generate_module_docs '{"strategy":"${strategy}","sourcePath":".","projectName":"${project_name}","tool":"${tool}"}'`,
            run_in_background: false
          });
          if (bash_result.exit_code === 0) {
            report(`✅ ${module.path} (Layer ${layer}) docs generated with ${tool}`);
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
let project_name = detect_project_name();

for (let layer of [3, 2, 1]) {
  if (modules_by_layer[layer].length === 0) continue;

  let batches = batch_modules(modules_by_layer[layer], 4);
  let worker_tasks = [];

  for (let batch of batches) {
    worker_tasks.push(
      Task(
        subagent_type="memory-bridge",
        description=`Generate docs for ${batch.length} modules in Layer ${layer}`,
        prompt=generate_batch_worker_prompt(batch, tool_order, layer, project_name)
      )
    );
  }

  await parallel_execute(worker_tasks);
}
```

**Batch Worker Prompt Template**:
```
PURPOSE: Generate documentation for assigned modules with tool fallback

TASK: Generate API.md + README.md for assigned modules using specified strategies.

PROJECT: {{project_name}}
OUTPUT: .workflow/docs/{{project_name}}/

MODULES:
{{module_path_1}} (strategy: {{strategy_1}}, type: {{folder_type_1}})
{{module_path_2}} (strategy: {{strategy_2}}, type: {{folder_type_2}})
...

TOOLS (try in order): {{tool_1}}, {{tool_2}}, {{tool_3}}

EXECUTION SCRIPT: ccw tool exec generate_module_docs
  - Accepts strategy parameter: full | single
  - Accepts folder type detection: code | navigation
  - Tool execution via direct CLI commands (gemini/qwen/codex)
  - Output path: .workflow/docs/{{project_name}}/{module_path}/

EXECUTION FLOW (for each module):
  1. Tool fallback loop (exit on first success):
     for tool in {{tool_1}} {{tool_2}} {{tool_3}}; do
       Bash({
         command: `cd "{{module_path}}" && ccw tool exec generate_module_docs '{"strategy":"{{strategy}}","sourcePath":".","projectName":"{{project_name}}","tool":"${tool}"}'`,
         run_in_background: false
       })
       exit_code=$?

       if [ $exit_code -eq 0 ]; then
         report "✅ {{module_path}} docs generated with $tool"
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

FOLDER TYPE HANDLING:
  - code: Generate API.md + README.md
  - navigation: Generate README.md only

FAILURE HANDLING:
  - Module-level isolation: One module's failure does not affect others
  - Exit code detection: Non-zero exit code triggers next tool
  - Exhaustion reporting: Log modules where all tools failed
  - Batch continuation: Always process remaining modules

REPORTING FORMAT:
  Per-module status:
    ✅ path/to/module docs generated with {tool}
    ⚠️  path/to/module failed with {tool}, trying next...
    ❌ FAILED: path/to/module - all tools exhausted
```

### Phase 4: Project-Level Documentation

**After all module documentation is generated, create project-level documentation files.**

```javascript
let project_name = detect_project_name();
let project_root = get_project_root();

// Step 1: Generate Project README
report("Generating project README.md...");
for (let tool of tool_order) {
  Bash({
    command: `cd ${project_root} && ccw tool exec generate_module_docs '{"strategy":"project-readme","sourcePath":".","projectName":"${project_name}","tool":"${tool}"}'`,
    run_in_background: false
  });
  if (bash_result.exit_code === 0) {
    report(`✅ Project README generated with ${tool}`);
    break;
  }
}

// Step 2: Generate Architecture & Examples
report("Generating ARCHITECTURE.md and EXAMPLES.md...");
for (let tool of tool_order) {
  Bash({
    command: `cd ${project_root} && ccw tool exec generate_module_docs '{"strategy":"project-architecture","sourcePath":".","projectName":"${project_name}","tool":"${tool}"}'`,
    run_in_background: false
  });
  if (bash_result.exit_code === 0) {
    report(`✅ Architecture docs generated with ${tool}`);
    break;
  }
}

// Step 3: Generate HTTP API documentation (if API routes detected)
Bash({command: 'rg "router\\.|@Get|@Post" -g "*.{ts,js,py}" 2>/dev/null && echo "API_FOUND" || echo "NO_API"', run_in_background: false});
if (bash_result.stdout.includes("API_FOUND")) {
  report("Generating HTTP API documentation...");
  for (let tool of tool_order) {
    Bash({
      command: `cd ${project_root} && ccw tool exec generate_module_docs '{"strategy":"http-api","sourcePath":".","projectName":"${project_name}","tool":"${tool}"}'`,
      run_in_background: false
    });
    if (bash_result.exit_code === 0) {
      report(`✅ HTTP API docs generated with ${tool}`);
      break;
    }
  }
}
```

**Expected Output**:
```
Project-Level Documentation:
  ✅ README.md (project root overview)
  ✅ ARCHITECTURE.md (system design)
  ✅ EXAMPLES.md (usage examples)
  ✅ api/README.md (HTTP API reference) [optional]
```

### Phase 5: Verification

```javascript
// Check documentation files created
Bash({command: 'find .workflow/docs -type f -name "*.md" 2>/dev/null | wc -l', run_in_background: false});

// Display structure
Bash({command: 'tree -L 3 .workflow/docs/', run_in_background: false});
```

**Result Summary**:
```
Documentation Generation Summary:
  Total: 31 | Success: 29 | Failed: 2
  Tool usage: gemini: 25, qwen: 4, codex: 0
  Failed: path1, path2

  Generated documentation:
    .workflow/docs/myproject/
    ├── src/
    │   ├── auth/
    │   │   ├── API.md
    │   │   └── README.md
    │   └── utils/
    │       └── README.md
    └── README.md
```

## Error Handling

**Batch Worker**: Tool fallback per module, batch isolation, clear status reporting
**Coordinator**: Invalid path abort, user decline handling, verification with cleanup
**Fallback Triggers**: Non-zero exit code, script timeout, unexpected output

## Output Structure

```
.workflow/docs/{project_name}/
├── src/                           # Mirrors source structure
│   ├── modules/
│   │   ├── README.md              # Navigation
│   │   ├── auth/
│   │   │   ├── API.md             # API signatures
│   │   │   ├── README.md          # Module docs
│   │   │   └── middleware/
│   │   │       ├── API.md
│   │   │       └── README.md
│   │   └── api/
│   │       ├── API.md
│   │       └── README.md
│   └── utils/
│       └── README.md
├── lib/
│   └── core/
│       ├── API.md
│       └── README.md
├── README.md                      # ✨ Project root overview (auto-generated)
├── ARCHITECTURE.md                # ✨ System design (auto-generated)
├── EXAMPLES.md                    # ✨ Usage examples (auto-generated)
└── api/                           # ✨ Optional (auto-generated if HTTP API detected)
    └── README.md                  # HTTP API reference
```

## Usage Examples

```bash
# Full project documentation generation
/memory:docs-full-cli

# Target specific directory
/memory:docs-full-cli src/features/auth
/memory:docs-full-cli .claude

# Use specific tool
/memory:docs-full-cli --tool qwen
/memory:docs-full-cli src --tool qwen
```

## Key Advantages

- **Efficiency**: 30 modules → 8 agents (73% reduction from sequential)
- **Resilience**: 3-tier tool fallback per module
- **Performance**: Parallel batches, no concurrency limits
- **Observability**: Per-module tool usage, batch-level metrics
- **Automation**: Zero configuration - strategy auto-selected by directory depth
- **Path Mirroring**: Clear 1:1 mapping between source and documentation structure

## Template Reference

Templates used from `~/.claude/workflows/cli-templates/prompts/documentation/`:
- `api.txt`: Code API documentation (Part A: Code API, Part B: HTTP API)
- `module-readme.txt`: Module purpose, usage, dependencies
- `folder-navigation.txt`: Navigation README for folders with subdirectories

## Related Commands

- `/memory:docs` - Agent-based documentation planning workflow
- `/memory:docs-related-cli` - Update docs for changed modules only
- `/workflow:execute` - Execute documentation tasks (when using agent mode)
