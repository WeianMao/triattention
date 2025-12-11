---
name: skill-memory
description: 4-phase autonomous orchestrator: check docs → /memory:docs planning → /workflow:execute → generate SKILL.md with progressive loading index (skips phases 2-3 if docs exist)
argument-hint: "[path] [--tool <gemini|qwen|codex>] [--regenerate] [--mode <full|partial>] [--cli-execute]"
allowed-tools: SlashCommand(*), TodoWrite(*), Bash(*), Read(*), Write(*)
---

# Memory SKILL Package Generator

## Orchestrator Role

**Pure Orchestrator**: Execute documentation generation workflow, then generate SKILL.md index. Does NOT create task JSON files.

**Auto-Continue Workflow**: This command runs **fully autonomously** once triggered. Each phase completes and automatically triggers the next phase without user interaction.

**Execution Paths**:
- **Full Path**: All 4 phases (no existing docs OR `--regenerate` specified)
- **Skip Path**: Phase 1 → Phase 4 (existing docs found AND no `--regenerate` flag)
- **Phase 4 Always Executes**: SKILL.md index is never skipped, always generated or updated

## Core Rules

1. **Start Immediately**: First action is TodoWrite initialization, second action is Phase 1 execution
2. **No Task JSON**: This command does not create task JSON files - delegates to /memory:docs
3. **Parse Every Output**: Extract required data from each command output (session_id, task_count, file paths)
4. **Auto-Continue**: After completing each phase, update TodoWrite and immediately execute next phase
5. **Track Progress**: Update TodoWrite after EVERY phase completion before starting next phase
6. **Direct Generation**: Phase 4 directly generates SKILL.md using Write tool
7. **No Manual Steps**: User should never be prompted for decisions between phases

---

## 4-Phase Execution

### Phase 1: Prepare Arguments

**Goal**: Parse command arguments and check existing documentation

**Step 1: Get Target Path and Project Name**
```bash
# Get current directory (or use provided path)
bash(pwd)

# Get project name from directory
bash(basename "$(pwd)")

# Get project root
bash(git rev-parse --show-toplevel 2>/dev/null || pwd)
```

**Output**:
- `target_path`: `/d/my_project`
- `project_name`: `my_project`
- `project_root`: `/d/my_project`

**Step 2: Set Default Parameters**
```bash
# Default values (use these unless user specifies otherwise):
# - tool: "gemini"
# - mode: "full"
# - regenerate: false (no --regenerate flag)
# - cli_execute: false (no --cli-execute flag)
```

**Step 3: Check Existing Documentation**
```bash
# Check if docs directory exists
bash(test -d .workflow/docs/my_project && echo "exists" || echo "not_exists")

# Count existing documentation files
bash(find .workflow/docs/my_project -name "*.md" 2>/dev/null | wc -l || echo 0)
```

**Output**:
- `docs_exists`: `exists` or `not_exists`
- `existing_docs`: `5` (or `0` if no docs)

**Step 4: Determine Execution Path**

**Decision Logic**:
```javascript
if (existing_docs > 0 && !regenerate_flag) {
  // Documentation exists and no regenerate flag
  SKIP_DOCS_GENERATION = true
  message = "Documentation already exists, skipping Phase 2 and Phase 3. Use --regenerate to force regeneration."
} else if (regenerate_flag) {
  // Force regeneration: delete existing docs
  bash(rm -rf .workflow/docs/my_project 2>/dev/null || true)
  SKIP_DOCS_GENERATION = false
  message = "Regenerating documentation from scratch."
} else {
  // No existing docs
  SKIP_DOCS_GENERATION = false
  message = "No existing documentation found, generating new documentation."
}
```

**Summary Variables**:
- `PROJECT_NAME`: `my_project`
- `TARGET_PATH`: `/d/my_project`
- `DOCS_PATH`: `.workflow/docs/my_project`
- `TOOL`: `gemini` (default) or user-specified
- `MODE`: `full` (default) or user-specified
- `CLI_EXECUTE`: `false` (default) or `true` if --cli-execute flag
- `REGENERATE`: `false` (default) or `true` if --regenerate flag
- `EXISTING_DOCS`: Count of existing documentation files
- `SKIP_DOCS_GENERATION`: `true` if skipping Phase 2/3, `false` otherwise

**Completion & TodoWrite**:
- If `SKIP_DOCS_GENERATION = true`: Mark phase 1 completed, phase 2&3 completed (skipped), phase 4 in_progress
- If `SKIP_DOCS_GENERATION = false`: Mark phase 1 completed, phase 2 in_progress

**Next Action**:
- If skipping: Display skip message → Jump to Phase 4 (SKILL.md generation)
- If not skipping: Display preparation results → Continue to Phase 2 (documentation planning)

---

### Phase 2: Call /memory:docs

**Skip Condition**: This phase is **skipped if SKIP_DOCS_GENERATION = true** (documentation already exists without --regenerate flag)

**Goal**: Trigger documentation generation workflow

**Command**:
```bash
SlashCommand(command="/memory:docs [targetPath] --tool [tool] --mode [mode] [--cli-execute]")
```

**Example**:
```bash
/memory:docs /d/my_app --tool gemini --mode full
/memory:docs /d/my_app --tool gemini --mode full --cli-execute
```

**Note**: The `--regenerate` flag is handled in Phase 1 by deleting existing documentation. This command always calls `/memory:docs` without the regenerate flag, relying on docs.md's built-in update detection.

**Parse Output**:
- Extract session ID: `WFS-docs-[timestamp]` (store as `docsSessionId`)
- Extract task count (store as `taskCount`)

**Completion Criteria**:
- `/memory:docs` command executed successfully
- Session ID extracted and stored
- Task count retrieved
- Task files created in `.workflow/[docsSessionId]/.task/`
- workflow-session.json exists

**TodoWrite**: Mark phase 2 completed, phase 3 in_progress

**Next Action**: Display docs planning results (session ID, task count) → Auto-continue to Phase 3

---

### Phase 3: Execute Documentation Generation

**Skip Condition**: This phase is **skipped if SKIP_DOCS_GENERATION = true** (documentation already exists without --regenerate flag)

**Goal**: Execute documentation generation tasks

**Command**:
```bash
SlashCommand(command="/workflow:execute")
```

**Note**: `/workflow:execute` automatically discovers active session from Phase 2

**Completion Criteria**:
- `/workflow:execute` command executed successfully
- Documentation files generated in `.workflow/docs/[projectName]/`
- All tasks marked as completed in session
- At minimum: module documentation files exist (API.md and/or README.md)
- For full mode: Project README, ARCHITECTURE, EXAMPLES files generated

**TodoWrite**: Mark phase 3 completed, phase 4 in_progress

**Next Action**: Display execution results (file count, module count) → Auto-continue to Phase 4

---

### Phase 4: Generate SKILL.md Index

**Note**: This phase is **NEVER skipped** - it always executes to generate or update the SKILL index.

**Step 1: Read Key Files** (Use Read tool)
- `.workflow/docs/{project_name}/README.md` (required)
- `.workflow/docs/{project_name}/ARCHITECTURE.md` (optional)

**Step 2: Discover Structure**
```bash
bash(find .workflow/docs/{project_name} -name "*.md" | sed 's|.workflow/docs/{project_name}/||' | awk -F'/' '{if(NF>=2) print $1"/"$2}' | sort -u)
```

**Step 3: Generate Intelligent Description**

Extract from README + structure: Function (capabilities), Modules (names), Keywords (API/CLI/auth/etc.)

**Format**: `{Project} {core capabilities} (located at {project_path}). Load this SKILL when analyzing, modifying, or learning about {domain_description} or files under this path, especially when no relevant context exists in memory.`

**Key Elements**:
- **Path Reference**: Use `TARGET_PATH` from Phase 1 for precise location identification
- **Domain Description**: Extract human-readable domain/feature area from README (e.g., "workflow management", "thermal modeling")
- **Trigger Optimization**: Include project path, emphasize "especially when no relevant context exists in memory"
- **Action Coverage**: analyzing (分析), modifying (修改), learning (了解)

**Example**: "Workflow orchestration system with CLI tools and documentation generation (located at /d/Claude_dms3). Load this SKILL when analyzing, modifying, or learning about workflow management or files under this path, especially when no relevant context exists in memory."

**Step 4: Write SKILL.md** (Use Write tool)
```bash
bash(mkdir -p .claude/skills/{project_name})
```

`.claude/skills/{project_name}/SKILL.md`:
```yaml
---
name: {project_name}
description: {intelligent description from Step 3}
version: 1.0.0
---
# {Project Name} SKILL Package

## Documentation: `../../../.workflow/docs/{project_name}/`

## Progressive Loading
### Level 0: Quick Start (~2K)
- [README](../../../.workflow/docs/{project_name}/README.md)
### Level 1: Core Modules (~8K)
{Module READMEs}
### Level 2: Complete (~25K)
All modules + [Architecture](../../../.workflow/docs/{project_name}/ARCHITECTURE.md)
### Level 3: Deep Dive (~40K)
Everything + [Examples](../../../.workflow/docs/{project_name}/EXAMPLES.md)
```

**Completion Criteria**:
- SKILL.md file created at `.claude/skills/{project_name}/SKILL.md`
- Intelligent description generated from documentation
- Progressive loading levels (0-3) properly structured
- Module index includes all documented modules
- All file references use relative paths

**TodoWrite**: Mark phase 4 completed

**Final Action**: Report completion summary to user

**Return to User**:
```
SKILL Package Generation Complete

Project: {project_name}
Documentation: .workflow/docs/{project_name}/ ({doc_count} files)
SKILL Index: .claude/skills/{project_name}/SKILL.md

Generated:
- {task_count} documentation tasks completed
- SKILL.md with progressive loading (4 levels)
- Module index with {module_count} modules

Usage:
- Load Level 0: Quick project overview (~2K tokens)
- Load Level 1: Core modules (~8K tokens)
- Load Level 2: Complete docs (~25K tokens)
- Load Level 3: Everything (~40K tokens)
```

---

## Implementation Details

### Critical Rules

1. **No User Prompts Between Phases**: Never ask user questions or wait for input between phases
2. **Immediate Phase Transition**: After TodoWrite update, immediately execute next phase command
3. **Status-Driven Execution**: Check TodoList status after each phase:
   - If next task is "pending" → Mark it "in_progress" and execute
   - If all tasks are "completed" → Report final summary
4. **Phase Completion Pattern**:
   ```
   Phase N completes → Update TodoWrite (N=completed, N+1=in_progress) → Execute Phase N+1
   ```

### TodoWrite Patterns

#### Initialization (Before Phase 1)

**FIRST ACTION**: Create TodoList with all 4 phases
```javascript
TodoWrite({todos: [
  {"content": "Parse arguments and prepare", "status": "in_progress", "activeForm": "Parsing arguments"},
  {"content": "Call /memory:docs to plan documentation", "status": "pending", "activeForm": "Calling /memory:docs"},
  {"content": "Execute documentation generation", "status": "pending", "activeForm": "Executing documentation"},
  {"content": "Generate SKILL.md index", "status": "pending", "activeForm": "Generating SKILL.md"}
]})
```

**SECOND ACTION**: Execute Phase 1 immediately

#### Full Path (SKIP_DOCS_GENERATION = false)

**After Phase 1**:
```javascript
TodoWrite({todos: [
  {"content": "Parse arguments and prepare", "status": "completed", "activeForm": "Parsing arguments"},
  {"content": "Call /memory:docs to plan documentation", "status": "in_progress", "activeForm": "Calling /memory:docs"},
  {"content": "Execute documentation generation", "status": "pending", "activeForm": "Executing documentation"},
  {"content": "Generate SKILL.md index", "status": "pending", "activeForm": "Generating SKILL.md"}
]})
// Auto-continue to Phase 2
```

**After Phase 2**:
```javascript
TodoWrite({todos: [
  {"content": "Parse arguments and prepare", "status": "completed", "activeForm": "Parsing arguments"},
  {"content": "Call /memory:docs to plan documentation", "status": "completed", "activeForm": "Calling /memory:docs"},
  {"content": "Execute documentation generation", "status": "in_progress", "activeForm": "Executing documentation"},
  {"content": "Generate SKILL.md index", "status": "pending", "activeForm": "Generating SKILL.md"}
]})
// Auto-continue to Phase 3
```

**After Phase 3**:
```javascript
TodoWrite({todos: [
  {"content": "Parse arguments and prepare", "status": "completed", "activeForm": "Parsing arguments"},
  {"content": "Call /memory:docs to plan documentation", "status": "completed", "activeForm": "Calling /memory:docs"},
  {"content": "Execute documentation generation", "status": "completed", "activeForm": "Executing documentation"},
  {"content": "Generate SKILL.md index", "status": "in_progress", "activeForm": "Generating SKILL.md"}
]})
// Auto-continue to Phase 4
```

**After Phase 4**:
```javascript
TodoWrite({todos: [
  {"content": "Parse arguments and prepare", "status": "completed", "activeForm": "Parsing arguments"},
  {"content": "Call /memory:docs to plan documentation", "status": "completed", "activeForm": "Calling /memory:docs"},
  {"content": "Execute documentation generation", "status": "completed", "activeForm": "Executing documentation"},
  {"content": "Generate SKILL.md index", "status": "completed", "activeForm": "Generating SKILL.md"}
]})
// Report completion summary to user
```

#### Skip Path (SKIP_DOCS_GENERATION = true)

**After Phase 1** (detects existing docs, skips Phase 2 & 3):
```javascript
TodoWrite({todos: [
  {"content": "Parse arguments and prepare", "status": "completed", "activeForm": "Parsing arguments"},
  {"content": "Call /memory:docs to plan documentation", "status": "completed", "activeForm": "Calling /memory:docs"},
  {"content": "Execute documentation generation", "status": "completed", "activeForm": "Executing documentation"},
  {"content": "Generate SKILL.md index", "status": "in_progress", "activeForm": "Generating SKILL.md"}
]})
// Display skip message: "Documentation already exists, skipping Phase 2 and Phase 3. Use --regenerate to force regeneration."
// Jump directly to Phase 4
```

**After Phase 4**:
```javascript
TodoWrite({todos: [
  {"content": "Parse arguments and prepare", "status": "completed", "activeForm": "Parsing arguments"},
  {"content": "Call /memory:docs to plan documentation", "status": "completed", "activeForm": "Calling /memory:docs"},
  {"content": "Execute documentation generation", "status": "completed", "activeForm": "Executing documentation"},
  {"content": "Generate SKILL.md index", "status": "completed", "activeForm": "Generating SKILL.md"}
]})
// Report completion summary to user
```

### Execution Flow Diagrams

#### Full Path Flow
```
User triggers command
  ↓
[TodoWrite] Initialize 4 phases (Phase 1 = in_progress)
  ↓
[Execute] Phase 1: Parse arguments
  ↓
[TodoWrite] Phase 1 = completed, Phase 2 = in_progress
  ↓
[Execute] Phase 2: Call /memory:docs
  ↓
[TodoWrite] Phase 2 = completed, Phase 3 = in_progress
  ↓
[Execute] Phase 3: Call /workflow:execute
  ↓
[TodoWrite] Phase 3 = completed, Phase 4 = in_progress
  ↓
[Execute] Phase 4: Generate SKILL.md
  ↓
[TodoWrite] Phase 4 = completed
  ↓
[Report] Display completion summary
```

#### Skip Path Flow
```
User triggers command
  ↓
[TodoWrite] Initialize 4 phases (Phase 1 = in_progress)
  ↓
[Execute] Phase 1: Parse arguments, detect existing docs
  ↓
[TodoWrite] Phase 1 = completed, Phase 2&3 = completed (skipped), Phase 4 = in_progress
  ↓
[Display] Skip message: "Documentation already exists, skipping Phase 2 and Phase 3"
  ↓
[Execute] Phase 4: Generate SKILL.md (always runs)
  ↓
[TodoWrite] Phase 4 = completed
  ↓
[Report] Display completion summary
```

### Error Handling

- If any phase fails, mark it as "in_progress" (not completed)
- Report error details to user
- Do NOT auto-continue to next phase on failure

---

## Parameters

```bash
/memory:skill-memory [path] [--tool <gemini|qwen|codex>] [--regenerate] [--mode <full|partial>] [--cli-execute]
```

- **path**: Target directory (default: current directory)
- **--tool**: CLI tool for documentation (default: gemini)
  - `gemini`: Comprehensive documentation
  - `qwen`: Architecture analysis
  - `codex`: Implementation validation
- **--regenerate**: Force regenerate all documentation
  - When enabled: Deletes existing `.workflow/docs/{project_name}/` before regeneration
  - Ensures fresh documentation from source code
- **--mode**: Documentation mode (default: full)
  - `full`: Complete docs (modules + README + ARCHITECTURE + EXAMPLES)
  - `partial`: Module docs only
- **--cli-execute**: Enable CLI-based documentation generation (optional)
  - When enabled: CLI generates docs directly in implementation_approach
  - When disabled (default): Agent generates documentation content

---

## Examples

### Example 1: Generate SKILL Package (Default)

```bash
/memory:skill-memory
```

**Workflow**:
1. Phase 1: Detects current directory, checks existing docs
2. Phase 2: Calls `/memory:docs . --tool gemini --mode full` (Agent Mode)
3. Phase 3: Executes documentation generation via `/workflow:execute`
4. Phase 4: Generates SKILL.md at `.claude/skills/{project_name}/SKILL.md`

### Example 2: Regenerate with Qwen

```bash
/memory:skill-memory /d/my_app --tool qwen --regenerate
```

**Workflow**:
1. Phase 1: Parses target path, detects regenerate flag, deletes existing docs
2. Phase 2: Calls `/memory:docs /d/my_app --tool qwen --mode full`
3. Phase 3: Executes documentation regeneration
4. Phase 4: Generates updated SKILL.md

### Example 3: Partial Mode (Modules Only)

```bash
/memory:skill-memory --mode partial
```

**Workflow**:
1. Phase 1: Detects partial mode
2. Phase 2: Calls `/memory:docs . --tool gemini --mode partial` (Agent Mode)
3. Phase 3: Executes module documentation only
4. Phase 4: Generates SKILL.md with module-only index

### Example 4: CLI Execute Mode

```bash
/memory:skill-memory --cli-execute
```

**Workflow**:
1. Phase 1: Detects CLI execute mode
2. Phase 2: Calls `/memory:docs . --tool gemini --mode full --cli-execute` (CLI Mode)
3. Phase 3: Executes CLI-based documentation generation
4. Phase 4: Generates SKILL.md at `.claude/skills/{project_name}/SKILL.md`

### Example 5: Skip Path (Existing Docs)

```bash
/memory:skill-memory
```

**Scenario**: Documentation already exists in `.workflow/docs/{project_name}/`

**Workflow**:
1. Phase 1: Detects existing docs (5 files), sets SKIP_DOCS_GENERATION = true
2. Display: "Documentation already exists, skipping Phase 2 and Phase 3. Use --regenerate to force regeneration."
3. Phase 4: Generates or updates SKILL.md index only (~5-10x faster)

---



## Architecture

```
skill-memory (orchestrator)
  ├─ Phase 1: Prepare (bash commands, skip decision)
  ├─ Phase 2: /memory:docs (task planning, skippable)
  ├─ Phase 3: /workflow:execute (task execution, skippable)
  └─ Phase 4: Write SKILL.md (direct file generation, always runs)

No task JSON created by this command
All documentation tasks managed by /memory:docs
Smart skip logic: 5-10x faster when docs exist
```
