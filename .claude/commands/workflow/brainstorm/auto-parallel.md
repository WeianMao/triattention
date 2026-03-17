---
name: auto-parallel
description: Parallel brainstorming automation with dynamic role selection and concurrent execution across multiple perspectives
argument-hint: "topic or challenge description" [--count N]
allowed-tools: SlashCommand(*), Task(*), TodoWrite(*), Read(*), Write(*), Bash(*), Glob(*)
---

# Workflow Brainstorm Parallel Auto Command

## Coordinator Role

**This command is a pure orchestrator**: Dispatches 3 phases in sequence (interactive framework → parallel role analysis → synthesis), coordinating specialized commands/agents through task attachment model.

**Task Attachment Model**:
- SlashCommand dispatch **expands workflow** by attaching sub-tasks to current TodoWrite
- Task agent dispatch **attaches analysis tasks** to orchestrator's TodoWrite
- Phase 1: artifacts command attaches its internal tasks (Phase 1-5)
- Phase 2: N conceptual-planning-agent tasks attached in parallel
- Phase 3: synthesis command attaches its internal tasks
- Orchestrator **executes these attached tasks** sequentially (Phase 1, 3) or in parallel (Phase 2)
- After completion, attached tasks are **collapsed** back to high-level phase summary
- This is **task expansion**, not external delegation

**Execution Model - Auto-Continue Workflow**:

This workflow runs **fully autonomously** once triggered. Phase 1 (artifacts) handles user interaction, Phase 2 (role agents) runs in parallel.

1. **User triggers**: `/workflow:brainstorm:auto-parallel "topic" [--count N]`
2. **Dispatch Phase 1** → artifacts command (tasks ATTACHED) → Auto-continues
3. **Dispatch Phase 2** → Parallel role agents (N tasks ATTACHED concurrently) → Auto-continues
4. **Dispatch Phase 3** → Synthesis command (tasks ATTACHED) → Reports final summary

**Auto-Continue Mechanism**:
- TodoList tracks current phase status and dynamically manages task attachment/collapse
- When Phase 1 (artifacts) finishes executing, automatically load roles and launch Phase 2 agents
- When Phase 2 (all agents) finishes executing, automatically execute Phase 3 synthesis
- **⚠️ CONTINUOUS EXECUTION** - Do not stop until all phases complete

## Core Rules

1. **Start Immediately**: First action is TodoWrite initialization, second action is dispatch Phase 1 command
2. **No Preliminary Analysis**: Do not analyze topic before Phase 1 - artifacts handles all analysis
3. **Parse Every Output**: Extract selected_roles from workflow-session.json after Phase 1
4. **Auto-Continue via TodoList**: Check TodoList status to dispatch next pending phase automatically
5. **Track Progress**: Update TodoWrite dynamically with task attachment/collapse pattern
6. **Task Attachment Model**: SlashCommand and Task dispatches **attach** sub-tasks to current workflow. Orchestrator **executes** these attached tasks itself, then **collapses** them after completion
7. **⚠️ CRITICAL: DO NOT STOP**: Continuous multi-phase workflow. After executing all attached tasks, immediately collapse them and dispatch next phase
8. **Parallel Execution**: Phase 2 attaches multiple agent tasks simultaneously for concurrent execution

## Usage

```bash
/workflow:brainstorm:auto-parallel "<topic>" [--count N] [--style-skill package-name]
```

**Recommended Structured Format**:
```bash
/workflow:brainstorm:auto-parallel "GOAL: [objective] SCOPE: [boundaries] CONTEXT: [background]" [--count N] [--style-skill package-name]
```

**Parameters**:
- `topic` (required): Topic or challenge description (structured format recommended)
- `--count N` (optional): Number of roles to select (default: 3, max: 9)
- `--style-skill package-name` (optional): Style SKILL package to load for UI design (located at `.claude/skills/style-{package-name}/`)

## 3-Phase Execution

### Phase 1: Interactive Framework Generation

**Step 1: Dispatch** - Interactive framework generation via artifacts command

```javascript
SlashCommand(command="/workflow:brainstorm:artifacts \"{topic}\" --count {N}")
```

**What It Does**:
- Topic analysis: Extract challenges, generate task-specific questions
- Role selection: Recommend count+2 roles, user selects via AskUserQuestion
- Role questions: Generate 3-4 questions per role, collect user decisions
- Conflict resolution: Detect and resolve cross-role conflicts
- Guidance generation: Transform Q&A to declarative guidance-specification.md

**Parse Output**:
- **⚠️ Memory Check**: If `selected_roles[]` already in conversation memory from previous load, skip file read
- Extract: `selected_roles[]` from workflow-session.json (if not in memory)
- Extract: `session_id` from workflow-session.json (if not in memory)
- Verify: guidance-specification.md exists

**Validation**:
- guidance-specification.md created with confirmed decisions
- workflow-session.json contains selected_roles[] (metadata only, no content duplication)
- Session directory `.workflow/active/WFS-{topic}/.brainstorming/` exists

**TodoWrite Update (Phase 1 SlashCommand dispatched - tasks attached)**:
```json
[
  {"content": "Phase 0: Parameter Parsing", "status": "completed", "activeForm": "Parsing count parameter"},
  {"content": "Phase 1: Interactive Framework Generation", "status": "in_progress", "activeForm": "Executing artifacts interactive framework"},
  {"content": "  → Topic analysis and question generation", "status": "in_progress", "activeForm": "Analyzing topic"},
  {"content": "  → Role selection and user confirmation", "status": "pending", "activeForm": "Selecting roles"},
  {"content": "  → Role questions and user decisions", "status": "pending", "activeForm": "Collecting role questions"},
  {"content": "  → Conflict detection and resolution", "status": "pending", "activeForm": "Resolving conflicts"},
  {"content": "  → Guidance specification generation", "status": "pending", "activeForm": "Generating guidance"},
  {"content": "Phase 2: Parallel Role Analysis", "status": "pending", "activeForm": "Executing parallel role analysis"},
  {"content": "Phase 3: Synthesis Integration", "status": "pending", "activeForm": "Executing synthesis integration"}
]
```

**Note**: SlashCommand dispatch **attaches** artifacts' 5 internal tasks. Orchestrator **executes** these tasks sequentially.

**Next Action**: Tasks attached → **Execute Phase 1.1-1.5** sequentially

**TodoWrite Update (Phase 1 completed - tasks collapsed)**:
```json
[
  {"content": "Phase 0: Parameter Parsing", "status": "completed", "activeForm": "Parsing count parameter"},
  {"content": "Phase 1: Interactive Framework Generation", "status": "completed", "activeForm": "Executing artifacts interactive framework"},
  {"content": "Phase 2: Parallel Role Analysis", "status": "pending", "activeForm": "Executing parallel role analysis"},
  {"content": "Phase 3: Synthesis Integration", "status": "pending", "activeForm": "Executing synthesis integration"}
]
```

**Note**: Phase 1 tasks completed and collapsed to summary.

**After Phase 1**: Auto-continue to Phase 2 (parallel role agent execution)

---

### Phase 2: Parallel Role Analysis Execution

**For Each Selected Role**:
```bash
Task(conceptual-planning-agent): "
[FLOW_CONTROL]

Execute {role-name} analysis for existing topic framework

## Context Loading
ASSIGNED_ROLE: {role-name}
OUTPUT_LOCATION: .workflow/active/WFS-{session}/.brainstorming/{role}/
TOPIC: {user-provided-topic}

## Flow Control Steps
1. load_topic_framework → .workflow/active/WFS-{session}/.brainstorming/guidance-specification.md
2. load_role_template → ~/.claude/workflows/cli-templates/planning-roles/{role}.md
3. load_session_metadata → .workflow/active/WFS-{session}/workflow-session.json
4. load_style_skill (ui-designer only, if style_skill_package) → .claude/skills/style-{style_skill_package}/

## Analysis Requirements
**Primary Reference**: Original user prompt from workflow-session.json is authoritative
**Framework Source**: Address all discussion points in guidance-specification.md from {role-name} perspective
**Role Focus**: {role-name} domain expertise aligned with user intent
**Structured Approach**: Create analysis.md addressing framework discussion points
**Template Integration**: Apply role template guidelines within framework structure

## Expected Deliverables
1. **analysis.md** (optionally with analysis-{slug}.md sub-documents)
2. **Framework Reference**: @../guidance-specification.md
3. **User Intent Alignment**: Validate against session_context

## Completion Criteria
- Address each discussion point from guidance-specification.md with {role-name} expertise
- Provide actionable recommendations from {role-name} perspective within analysis files
- All output files MUST start with `analysis` prefix (no recommendations.md or other naming)
- Reference framework document using @ notation for integration
- Update workflow-session.json with completion status
"
```

**Parallel Dispatch**:
- Launch N agents simultaneously (one message with multiple Task calls)
- Each agent task **attached** to orchestrator's TodoWrite
- All agents execute concurrently, each attaching their own analysis sub-tasks
- Each agent operates independently reading same guidance-specification.md

**Input**:
- `selected_roles[]` from Phase 1
- `session_id` from Phase 1
- guidance-specification.md path

**Validation**:
- Each role creates `.workflow/active/WFS-{topic}/.brainstorming/{role}/analysis.md`
- Optionally with `analysis-{slug}.md` sub-documents (max 5)
- **File pattern**: `analysis*.md` for globbing
- **FORBIDDEN**: `recommendations.md` or any non-`analysis` prefixed files
- All N role analyses completed

**TodoWrite Update (Phase 2 agents dispatched - tasks attached in parallel)**:
```json
[
  {"content": "Phase 0: Parameter Parsing", "status": "completed", "activeForm": "Parsing count parameter"},
  {"content": "Phase 1: Interactive Framework Generation", "status": "completed", "activeForm": "Executing artifacts interactive framework"},
  {"content": "Phase 2: Parallel Role Analysis", "status": "in_progress", "activeForm": "Executing parallel role analysis"},
  {"content": "  → Execute system-architect analysis", "status": "in_progress", "activeForm": "Executing system-architect analysis"},
  {"content": "  → Execute ui-designer analysis", "status": "in_progress", "activeForm": "Executing ui-designer analysis"},
  {"content": "  → Execute product-manager analysis", "status": "in_progress", "activeForm": "Executing product-manager analysis"},
  {"content": "Phase 3: Synthesis Integration", "status": "pending", "activeForm": "Executing synthesis integration"}
]
```

**Note**: Multiple Task dispatches **attach** N role analysis tasks simultaneously. Orchestrator **executes** these tasks in parallel.

**Next Action**: Tasks attached → **Execute Phase 2.1-2.N** concurrently

**TodoWrite Update (Phase 2 completed - tasks collapsed)**:
```json
[
  {"content": "Phase 0: Parameter Parsing", "status": "completed", "activeForm": "Parsing count parameter"},
  {"content": "Phase 1: Interactive Framework Generation", "status": "completed", "activeForm": "Executing artifacts interactive framework"},
  {"content": "Phase 2: Parallel Role Analysis", "status": "completed", "activeForm": "Executing parallel role analysis"},
  {"content": "Phase 3: Synthesis Integration", "status": "pending", "activeForm": "Executing synthesis integration"}
]
```

**Note**: Phase 2 parallel tasks completed and collapsed to summary.

**After Phase 2**: Auto-continue to Phase 3 (synthesis)

---

### Phase 3: Synthesis Generation

**Step 3: Dispatch** - Synthesis integration via synthesis command

```javascript
SlashCommand(command="/workflow:brainstorm:synthesis --session {sessionId}")
```

**What It Does**:
- Load original user intent from workflow-session.json
- Read all role analysis.md files
- Integrate role insights into synthesis-specification.md
- Validate alignment with user's original objectives

**Input**: `sessionId` from Phase 1

**Validation**:
- `.workflow/active/WFS-{topic}/.brainstorming/synthesis-specification.md` exists
- Synthesis references all role analyses

**TodoWrite Update (Phase 3 SlashCommand dispatched - tasks attached)**:
```json
[
  {"content": "Phase 0: Parameter Parsing", "status": "completed", "activeForm": "Parsing count parameter"},
  {"content": "Phase 1: Interactive Framework Generation", "status": "completed", "activeForm": "Executing artifacts interactive framework"},
  {"content": "Phase 2: Parallel Role Analysis", "status": "completed", "activeForm": "Executing parallel role analysis"},
  {"content": "Phase 3: Synthesis Integration", "status": "in_progress", "activeForm": "Executing synthesis integration"},
  {"content": "  → Load role analysis files", "status": "in_progress", "activeForm": "Loading role analyses"},
  {"content": "  → Integrate insights across roles", "status": "pending", "activeForm": "Integrating insights"},
  {"content": "  → Generate synthesis specification", "status": "pending", "activeForm": "Generating synthesis"}
]
```

**Note**: SlashCommand dispatch **attaches** synthesis' internal tasks. Orchestrator **executes** these tasks sequentially.

**Next Action**: Tasks attached → **Execute Phase 3.1-3.3** sequentially

**TodoWrite Update (Phase 3 completed - tasks collapsed)**:
```json
[
  {"content": "Phase 0: Parameter Parsing", "status": "completed", "activeForm": "Parsing count parameter"},
  {"content": "Phase 1: Interactive Framework Generation", "status": "completed", "activeForm": "Executing artifacts interactive framework"},
  {"content": "Phase 2: Parallel Role Analysis", "status": "completed", "activeForm": "Executing parallel role analysis"},
  {"content": "Phase 3: Synthesis Integration", "status": "completed", "activeForm": "Executing synthesis integration"}
]
```

**Note**: Phase 3 tasks completed and collapsed to summary.

**Return to User**:
```
Brainstorming complete for session: {sessionId}
Roles analyzed: {count}
Synthesis: .workflow/active/WFS-{topic}/.brainstorming/synthesis-specification.md

✅ Next Steps:
1. /workflow:concept-clarify --session {sessionId}  # Optional refinement
2. /workflow:plan --session {sessionId}  # Generate implementation plan
```

## TodoWrite Pattern

**Core Concept**: Dynamic task attachment and collapse for parallel brainstorming workflow with interactive framework generation and concurrent role analysis.

### Key Principles

1. **Task Attachment** (when SlashCommand/Task dispatched):
   - Sub-command's or agent's internal tasks are **attached** to orchestrator's TodoWrite
   - Phase 1: `/workflow:brainstorm:artifacts` attaches 5 internal tasks (Phase 1.1-1.5)
   - Phase 2: Multiple `Task(conceptual-planning-agent)` calls attach N role analysis tasks simultaneously
   - Phase 3: `/workflow:brainstorm:synthesis` attaches 3 internal tasks (Phase 3.1-3.3)
   - First attached task marked as `in_progress`, others as `pending`
   - Orchestrator **executes** these attached tasks (sequentially for Phase 1, 3; in parallel for Phase 2)

2. **Task Collapse** (after sub-tasks complete):
   - Remove detailed sub-tasks from TodoWrite
   - **Collapse** to high-level phase summary
   - Example: Phase 1.1-1.5 collapse to "Execute artifacts interactive framework generation: completed"
   - Phase 2: Multiple role tasks collapse to "Execute parallel role analysis: completed"
   - Phase 3: Synthesis tasks collapse to "Execute synthesis integration: completed"
   - Maintains clean orchestrator-level view

3. **Continuous Execution**:
   - After collapse, automatically proceed to next pending phase
   - No user intervention required between phases
   - TodoWrite dynamically reflects current execution state

**Lifecycle Summary**: Initial pending tasks → Phase 1 dispatched (artifacts tasks ATTACHED) → Artifacts sub-tasks executed → Phase 1 completed (tasks COLLAPSED) → Phase 2 dispatched (N role tasks ATTACHED in parallel) → Role analyses executed concurrently → Phase 2 completed (tasks COLLAPSED) → Phase 3 dispatched (synthesis tasks ATTACHED) → Synthesis sub-tasks executed → Phase 3 completed (tasks COLLAPSED) → Workflow complete.

### Brainstorming Workflow Specific Features

- **Phase 1**: Interactive framework generation with user Q&A (SlashCommand attachment)
- **Phase 2**: Parallel role analysis execution with N concurrent agents (Task agent attachments)
- **Phase 3**: Cross-role synthesis integration (SlashCommand attachment)
- **Dynamic Role Count**: `--count N` parameter determines number of Phase 2 parallel tasks (default: 3, max: 9)
- **Mixed Execution**: Sequential (Phase 1, 3) and Parallel (Phase 2) task execution


## Input Processing

**Count Parameter Parsing**:
```javascript
// Extract --count from user input
IF user_input CONTAINS "--count":
    EXTRACT count_value FROM "--count N" pattern
    IF count_value > 9:
        count_value = 9  // Cap at maximum 9 roles
ELSE:
    count_value = 3  // Default to 3 roles

// Pass to artifacts command
EXECUTE: /workflow:brainstorm:artifacts "{topic}" --count {count_value}
```

**Style-Skill Parameter Parsing**:
```javascript
// Extract --style-skill from user input
IF user_input CONTAINS "--style-skill":
    EXTRACT style_skill_name FROM "--style-skill package-name" pattern

    // Validate SKILL package exists
    skill_path = ".claude/skills/style-{style_skill_name}/SKILL.md"
    IF file_exists(skill_path):
        style_skill_package = style_skill_name
        style_reference_path = ".workflow/reference_style/{style_skill_name}"
        echo("✓ Style SKILL package found: style-{style_skill_name}")
        echo("  Design reference: {style_reference_path}")
    ELSE:
        echo("⚠ WARNING: Style SKILL package not found: {style_skill_name}")
        echo("  Expected location: {skill_path}")
        echo("  Continuing without style reference...")
        style_skill_package = null
ELSE:
    style_skill_package = null
    echo("No style-skill specified, ui-designer will use default workflow")

// Store for Phase 2 ui-designer context
CONTEXT_VARS:
    - style_skill_package: {style_skill_package}
    - style_reference_path: {style_reference_path}
```

**Topic Structuring**:
1. **Already Structured** → Pass directly to artifacts
   ```
   User: "GOAL: Build platform SCOPE: 100 users CONTEXT: Real-time"
   → Pass as-is to artifacts
   ```

2. **Simple Text** → Pass directly (artifacts handles structuring)
   ```
   User: "Build collaboration platform"
   → artifacts will analyze and structure
   ```

## Session Management

**⚡ FIRST ACTION**: Check `.workflow/active/` for existing sessions before Phase 1

**Multiple Sessions Support**:
- Different Claude instances can have different brainstorming sessions
- If multiple sessions found, prompt user to select
- If single session found, use it
- If no session exists, create `WFS-[topic-slug]`

**Session Continuity**:
- MUST use selected session for all phases
- Each role's context stored in session directory
- Session isolation: Each session maintains independent state

## Output Structure

**Phase 1 Output**:
- `.workflow/active/WFS-{topic}/.brainstorming/guidance-specification.md` (framework content)
- `.workflow/active/WFS-{topic}/workflow-session.json` (metadata: selected_roles[], topic, timestamps, style_skill_package)

**Phase 2 Output**:
- `.workflow/active/WFS-{topic}/.brainstorming/{role}/analysis.md` (one per role)
- `.superdesign/design_iterations/` (ui-designer artifacts, if --style-skill provided)

**Phase 3 Output**:
- `.workflow/active/WFS-{topic}/.brainstorming/synthesis-specification.md` (integrated analysis)

**⚠️ Storage Separation**: Guidance content in .md files, metadata in .json (no duplication)
**⚠️ Style References**: When --style-skill provided, workflow-session.json stores style_skill_package name, ui-designer loads from `.claude/skills/style-{package-name}/`

## Available Roles

- data-architect (数据架构师)
- product-manager (产品经理)
- product-owner (产品负责人)
- scrum-master (敏捷教练)
- subject-matter-expert (领域专家)
- system-architect (系统架构师)
- test-strategist (测试策略师)
- ui-designer (UI 设计师)
- ux-expert (UX 专家)

**Role Selection**: Handled by artifacts command (intelligent recommendation + user selection)

## Error Handling

- **Role selection failure**: artifacts defaults to product-manager with explanation
- **Agent execution failure**: Agent-specific retry with minimal dependencies
- **Template loading issues**: Agent handles graceful degradation
- **Synthesis conflicts**: Synthesis highlights disagreements without resolution

## Reference Information

**File Structure**:
```
.workflow/active/WFS-[topic]/
├── workflow-session.json              # Session metadata ONLY
└── .brainstorming/
    ├── guidance-specification.md      # Framework (Phase 1)
    ├── {role}/
    │   ├── analysis.md                # Main document (with optional @references)
    │   └── analysis-{slug}.md         # Section documents (max 5)
    └── synthesis-specification.md     # Integration (Phase 3)
```

**Template Source**: `~/.claude/workflows/cli-templates/planning-roles/`

