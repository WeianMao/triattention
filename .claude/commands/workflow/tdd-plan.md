---
name: tdd-plan
description: TDD workflow planning with Red-Green-Refactor task chain generation, test-first development structure, and cycle tracking
argument-hint: "\"feature description\"|file.md"
allowed-tools: SlashCommand(*), TodoWrite(*), Read(*), Bash(*)
---

# TDD Workflow Plan Command (/workflow:tdd-plan)

## Coordinator Role

**This command is a pure orchestrator**: Dispatches 6 slash commands in sequence, parse outputs, pass context, and ensure complete TDD workflow creation with Red-Green-Refactor task generation.

**CLI Tool Selection**: CLI tool usage is determined semantically from user's task description. Include "use Codex/Gemini/Qwen" in your request for CLI execution.

**Task Attachment Model**:
- SlashCommand dispatch **expands workflow** by attaching sub-tasks to current TodoWrite
- When dispatching a sub-command (e.g., `/workflow:tools:test-context-gather`), its internal tasks are attached to the orchestrator's TodoWrite
- Orchestrator **executes these attached tasks** sequentially
- After completion, attached tasks are **collapsed** back to high-level phase summary
- This is **task expansion**, not external delegation

**Auto-Continue Mechanism**:
- TodoList tracks current phase status and dynamically manages task attachment/collapse
- When each phase finishes executing, automatically dispatch next pending phase
- All phases run autonomously without user interaction
- **⚠️ CONTINUOUS EXECUTION** - Do not stop until all phases complete

## Core Rules

1. **Start Immediately**: First action is TodoWrite initialization, second action is dispatch Phase 1
2. **No Preliminary Analysis**: Do not read files before Phase 1
3. **Parse Every Output**: Extract required data for next phase
4. **Auto-Continue via TodoList**: Check TodoList status to dispatch next pending phase automatically
5. **Track Progress**: Update TodoWrite dynamically with task attachment/collapse pattern
6. **TDD Context**: All descriptions include "TDD:" prefix
7. **Task Attachment Model**: SlashCommand dispatch **attaches** sub-tasks to current workflow. Orchestrator **executes** these attached tasks itself, then **collapses** them after completion
8. **⚠️ CRITICAL: DO NOT STOP**: Continuous multi-phase workflow. After executing all attached tasks, immediately collapse them and dispatch next phase

## 6-Phase Execution (with Conflict Resolution)

### Phase 1: Session Discovery

**Step 1.1: Dispatch** - Session discovery and initialization

```javascript
SlashCommand(command="/workflow:session:start --type tdd --auto \"TDD: [structured-description]\"")
```

**TDD Structured Format**:
```
TDD: [Feature Name]
GOAL: [Objective]
SCOPE: [Included/excluded]
CONTEXT: [Background]
TEST_FOCUS: [Test scenarios]
```

**Parse**: Extract sessionId

**TodoWrite**: Mark phase 1 completed, phase 2 in_progress

**After Phase 1**: Return to user showing Phase 1 results, then auto-continue to Phase 2

---

### Phase 2: Context Gathering

**Step 2.1: Dispatch** - Context gathering and analysis

```javascript
SlashCommand(command="/workflow:tools:context-gather --session [sessionId] \"TDD: [structured-description]\"")
```

**Use Same Structured Description**: Pass the same structured format from Phase 1

**Input**: `sessionId` from Phase 1

**Parse Output**:
- Extract: context-package.json path (store as `contextPath`)
- Typical pattern: `.workflow/active/[sessionId]/.process/context-package.json`

**Validation**:
- Context package path extracted
- File exists and is valid JSON

**TodoWrite**: Mark phase 2 completed, phase 3 in_progress

**After Phase 2**: Return to user showing Phase 2 results, then auto-continue to Phase 3

---

### Phase 3: Test Coverage Analysis

**Step 3.1: Dispatch** - Test coverage analysis and framework detection

```javascript
SlashCommand(command="/workflow:tools:test-context-gather --session [sessionId]")
```

**Purpose**: Analyze existing codebase for:
- Existing test patterns and conventions
- Current test coverage
- Related components and integration points
- Test framework detection

**Parse**: Extract testContextPath (`.workflow/active/[sessionId]/.process/test-context-package.json`)



<!-- TodoWrite: When test-context-gather dispatched, INSERT 3 test-context-gather tasks -->

**TodoWrite Update (Phase 3 SlashCommand dispatched - tasks attached)**:
```json
[
  {"content": "Phase 1: Session Discovery", "status": "completed", "activeForm": "Executing session discovery"},
  {"content": "Phase 2: Context Gathering", "status": "completed", "activeForm": "Executing context gathering"},
  {"content": "Phase 3: Test Coverage Analysis", "status": "in_progress", "activeForm": "Executing test coverage analysis"},
  {"content": "  → Detect test framework and conventions", "status": "in_progress", "activeForm": "Detecting test framework"},
  {"content": "  → Analyze existing test coverage", "status": "pending", "activeForm": "Analyzing test coverage"},
  {"content": "  → Identify coverage gaps", "status": "pending", "activeForm": "Identifying coverage gaps"},
  {"content": "Phase 5: TDD Task Generation", "status": "pending", "activeForm": "Executing TDD task generation"},
  {"content": "Phase 6: TDD Structure Validation", "status": "pending", "activeForm": "Validating TDD structure"}
]
```

**Note**: SlashCommand dispatch **attaches** test-context-gather's 3 tasks. Orchestrator **executes** these tasks.

**Next Action**: Tasks attached → **Execute Phase 3.1-3.3** sequentially

<!-- TodoWrite: After Phase 3 tasks complete, REMOVE Phase 3.1-3.3, restore to orchestrator view -->

**TodoWrite Update (Phase 3 completed - tasks collapsed)**:
```json
[
  {"content": "Phase 1: Session Discovery", "status": "completed", "activeForm": "Executing session discovery"},
  {"content": "Phase 2: Context Gathering", "status": "completed", "activeForm": "Executing context gathering"},
  {"content": "Phase 3: Test Coverage Analysis", "status": "completed", "activeForm": "Executing test coverage analysis"},
  {"content": "Phase 5: TDD Task Generation", "status": "pending", "activeForm": "Executing TDD task generation"},
  {"content": "Phase 6: TDD Structure Validation", "status": "pending", "activeForm": "Validating TDD structure"}
]
```

**Note**: Phase 3 tasks completed and collapsed to summary.

**After Phase 3**: Return to user showing test coverage results, then auto-continue to Phase 4/5 (depending on conflict_risk)

---

### Phase 4: Conflict Resolution (Optional - auto-triggered by conflict risk)

**Trigger**: Only execute when context-package.json indicates conflict_risk is "medium" or "high"

**Step 4.1: Dispatch** - Conflict detection and resolution

```javascript
SlashCommand(command="/workflow:tools:conflict-resolution --session [sessionId] --context [contextPath]")
```

**Input**:
- sessionId from Phase 1
- contextPath from Phase 2
- conflict_risk from context-package.json

**Parse Output**:
- Extract: Execution status (success/skipped/failed)
- Verify: CONFLICT_RESOLUTION.md file path (if executed)

**Validation**:
- File `.workflow/active/[sessionId]/.process/CONFLICT_RESOLUTION.md` exists (if executed)

**Skip Behavior**:
- If conflict_risk is "none" or "low", skip directly to Phase 5
- Display: "No significant conflicts detected, proceeding to TDD task generation"

<!-- TodoWrite: If conflict_risk ≥ medium, INSERT 3 conflict-resolution tasks when dispatched -->

**TodoWrite Update (Phase 4 SlashCommand dispatched - tasks attached, if conflict_risk ≥ medium)**:
```json
[
  {"content": "Phase 1: Session Discovery", "status": "completed", "activeForm": "Executing session discovery"},
  {"content": "Phase 2: Context Gathering", "status": "completed", "activeForm": "Executing context gathering"},
  {"content": "Phase 3: Test Coverage Analysis", "status": "completed", "activeForm": "Executing test coverage analysis"},
  {"content": "Phase 4: Conflict Resolution", "status": "in_progress", "activeForm": "Executing conflict resolution"},
  {"content": "  → Detect conflicts with CLI analysis", "status": "in_progress", "activeForm": "Detecting conflicts"},
  {"content": "  → Present conflicts to user", "status": "pending", "activeForm": "Presenting conflicts"},
  {"content": "  → Apply resolution strategies", "status": "pending", "activeForm": "Applying resolution strategies"},
  {"content": "Phase 5: TDD Task Generation", "status": "pending", "activeForm": "Executing TDD task generation"},
  {"content": "Phase 6: TDD Structure Validation", "status": "pending", "activeForm": "Validating TDD structure"}
]
```

**Note**: SlashCommand dispatch **attaches** conflict-resolution's 3 tasks. Orchestrator **executes** these tasks.

**Next Action**: Tasks attached → **Execute Phase 4.1-4.3** sequentially

<!-- TodoWrite: After Phase 4 tasks complete, REMOVE Phase 4.1-4.3, restore to orchestrator view -->

**TodoWrite Update (Phase 4 completed - tasks collapsed)**:
```json
[
  {"content": "Phase 1: Session Discovery", "status": "completed", "activeForm": "Executing session discovery"},
  {"content": "Phase 2: Context Gathering", "status": "completed", "activeForm": "Executing context gathering"},
  {"content": "Phase 3: Test Coverage Analysis", "status": "completed", "activeForm": "Executing test coverage analysis"},
  {"content": "Phase 4: Conflict Resolution", "status": "completed", "activeForm": "Executing conflict resolution"},
  {"content": "Phase 5: TDD Task Generation", "status": "pending", "activeForm": "Executing TDD task generation"},
  {"content": "Phase 6: TDD Structure Validation", "status": "pending", "activeForm": "Validating TDD structure"}
]
```

**Note**: Phase 4 tasks completed and collapsed to summary.

**After Phase 4**: Return to user showing conflict resolution results (if executed) and selected strategies, then auto-continue to Phase 5

**Memory State Check**:
- Evaluate current context window usage and memory state
- If memory usage is high (>110K tokens or approaching context limits):

  **Step 4.5: Dispatch** - Memory compaction

  ```javascript
  SlashCommand(command="/compact")
  ```

  - This optimizes memory before proceeding to Phase 5
- Memory compaction is particularly important after analysis phase which may generate extensive documentation
- Ensures optimal performance and prevents context overflow

---

### Phase 5: TDD Task Generation

**Step 5.1: Dispatch** - TDD task generation via action-planning-agent

```javascript
SlashCommand(command="/workflow:tools:task-generate-tdd --session [sessionId]")
```

**Note**: CLI tool usage is determined semantically from user's task description.

**Parse**: Extract feature count, task count (not chain count - tasks now contain internal TDD cycles)

**Validate**:
- IMPL_PLAN.md exists (unified plan with TDD Implementation Tasks section)
- IMPL-*.json files exist (one per feature, or container + subtasks for complex features)
- TODO_LIST.md exists with internal TDD phase indicators
- Each IMPL task includes:
  - `meta.tdd_workflow: true`
  - `flow_control.implementation_approach` with 3 steps (red/green/refactor)
  - Green phase includes test-fix-cycle configuration
- IMPL_PLAN.md contains workflow_type: "tdd" in frontmatter
- Task count ≤10 (compliance with task limit)

<!-- TodoWrite: When task-generate-tdd dispatched, INSERT 3 task-generate-tdd tasks -->

**TodoWrite Update (Phase 5 SlashCommand dispatched - tasks attached)**:
```json
[
  {"content": "Phase 1: Session Discovery", "status": "completed", "activeForm": "Executing session discovery"},
  {"content": "Phase 2: Context Gathering", "status": "completed", "activeForm": "Executing context gathering"},
  {"content": "Phase 3: Test Coverage Analysis", "status": "completed", "activeForm": "Executing test coverage analysis"},
  {"content": "Phase 5: TDD Task Generation", "status": "in_progress", "activeForm": "Executing TDD task generation"},
  {"content": "  → Discovery - analyze TDD requirements", "status": "in_progress", "activeForm": "Analyzing TDD requirements"},
  {"content": "  → Planning - design Red-Green-Refactor cycles", "status": "pending", "activeForm": "Designing TDD cycles"},
  {"content": "  → Output - generate IMPL tasks with internal TDD phases", "status": "pending", "activeForm": "Generating TDD tasks"},
  {"content": "Phase 6: TDD Structure Validation", "status": "pending", "activeForm": "Validating TDD structure"}
]
```

**Note**: SlashCommand dispatch **attaches** task-generate-tdd's 3 tasks. Orchestrator **executes** these tasks. Each generated IMPL task will contain internal Red-Green-Refactor cycle.

**Next Action**: Tasks attached → **Execute Phase 5.1-5.3** sequentially

<!-- TodoWrite: After Phase 5 tasks complete, REMOVE Phase 5.1-5.3, restore to orchestrator view -->

**TodoWrite Update (Phase 5 completed - tasks collapsed)**:
```json
[
  {"content": "Phase 1: Session Discovery", "status": "completed", "activeForm": "Executing session discovery"},
  {"content": "Phase 2: Context Gathering", "status": "completed", "activeForm": "Executing context gathering"},
  {"content": "Phase 3: Test Coverage Analysis", "status": "completed", "activeForm": "Executing test coverage analysis"},
  {"content": "Phase 5: TDD Task Generation", "status": "completed", "activeForm": "Executing TDD task generation"},
  {"content": "Phase 6: TDD Structure Validation", "status": "in_progress", "activeForm": "Validating TDD structure"}
]
```

**Note**: Phase 5 tasks completed and collapsed to summary. Each generated IMPL task contains complete Red-Green-Refactor cycle internally.

### Phase 6: TDD Structure Validation & Action Plan Verification (RECOMMENDED)
**Internal validation first, then recommend external verification**

**Internal Validation**:
1. Each task contains complete TDD workflow (Red-Green-Refactor internally)
2. Task structure validation:
   - `meta.tdd_workflow: true` in all IMPL tasks
   - `flow_control.implementation_approach` has exactly 3 steps
   - Each step has correct `tdd_phase`: "red", "green", "refactor"
3. Dependency validation:
   - Sequential features: IMPL-N depends_on ["IMPL-(N-1)"] if needed
   - Complex features: IMPL-N.M depends_on ["IMPL-N.(M-1)"] for subtasks
4. Agent assignment: All IMPL tasks use @code-developer
5. Test-fix cycle: Green phase step includes test-fix-cycle logic with max_iterations
6. Task count: Total tasks ≤10 (simple + subtasks)

**Return Summary**:
```
TDD Planning complete for session: [sessionId]

Features analyzed: [N]
Total tasks: [M] (1 task per simple feature + subtasks for complex features)

Task breakdown:
- Simple features: [K] tasks (IMPL-1 to IMPL-K)
- Complex features: [L] features with [P] subtasks
- Total task count: [M] (within 10-task limit)

Structure:
- IMPL-1: {Feature 1 Name} (Internal: Red → Green → Refactor)
- IMPL-2: {Feature 2 Name} (Internal: Red → Green → Refactor)
- IMPL-3: {Complex Feature} (Container)
  - IMPL-3.1: {Sub-feature A} (Internal: Red → Green → Refactor)
  - IMPL-3.2: {Sub-feature B} (Internal: Red → Green → Refactor)
[...]

Plans generated:
- Unified Implementation Plan: .workflow/active/[sessionId]/IMPL_PLAN.md
  (includes TDD Implementation Tasks section with workflow_type: "tdd")
- Task List: .workflow/active/[sessionId]/TODO_LIST.md
  (with internal TDD phase indicators)

TDD Configuration:
- Each task contains complete Red-Green-Refactor cycle
- Green phase includes test-fix cycle (max 3 iterations)
- Auto-revert on max iterations reached

Recommended Next Steps:
1. /workflow:action-plan-verify --session [sessionId]  # Verify TDD plan quality and dependencies
2. /workflow:execute --session [sessionId]  # Start TDD execution
3. /workflow:tdd-verify [sessionId]  # Post-execution TDD compliance check

Quality Gate: Consider running /workflow:action-plan-verify to validate TDD task structure and dependencies
```

## TodoWrite Pattern

**Core Concept**: Dynamic task attachment and collapse for TDD workflow with test coverage analysis and Red-Green-Refactor cycle generation.

### Key Principles

1. **Task Attachment** (when SlashCommand dispatched):
   - Sub-command's internal tasks are **attached** to orchestrator's TodoWrite
   - Example: `/workflow:tools:test-context-gather` attaches 3 sub-tasks (Phase 3.1, 3.2, 3.3)
   - First attached task marked as `in_progress`, others as `pending`
   - Orchestrator **executes** these attached tasks sequentially

2. **Task Collapse** (after sub-tasks complete):
   - Remove detailed sub-tasks from TodoWrite
   - **Collapse** to high-level phase summary
   - Example: Phase 3.1-3.3 collapse to "Execute test coverage analysis: completed"
   - Maintains clean orchestrator-level view

3. **Continuous Execution**:
   - After collapse, automatically proceed to next pending phase
   - No user intervention required between phases
   - TodoWrite dynamically reflects current execution state

**Lifecycle Summary**: Initial pending tasks → Phase dispatched (tasks ATTACHED) → Sub-tasks executed sequentially → Phase completed (tasks COLLAPSED to summary) → Next phase begins (conditional Phase 4 if conflict_risk ≥ medium) → Repeat until all phases complete.

### TDD-Specific Features

- **Phase 3**: Test coverage analysis detects existing patterns and gaps
- **Phase 5**: Generated IMPL tasks contain internal Red-Green-Refactor cycles
- **Conditional Phase 4**: Conflict resolution only if conflict_risk ≥ medium



**Note**: See individual Phase descriptions (Phase 3, 4, 5) for detailed TodoWrite Update examples with full JSON structures.

## Execution Flow Diagram

```
TDD Workflow Orchestrator
│
├─ Phase 1: Session Discovery
│  └─ /workflow:session:start --auto
│     └─ Returns: sessionId
│
├─ Phase 2: Context Gathering
│  └─ /workflow:tools:context-gather
│     └─ Returns: context-package.json path
│
├─ Phase 3: Test Coverage Analysis                    ← ATTACHED (3 tasks)
│  └─ /workflow:tools:test-context-gather
│     ├─ Phase 3.1: Detect test framework
│     ├─ Phase 3.2: Analyze existing test coverage
│     └─ Phase 3.3: Identify coverage gaps
│     └─ Returns: test-context-package.json           ← COLLAPSED
│
├─ Phase 4: Conflict Resolution (conditional)
│  IF conflict_risk ≥ medium:
│  └─ /workflow:tools:conflict-resolution             ← ATTACHED (3 tasks)
│     ├─ Phase 4.1: Detect conflicts with CLI
│     ├─ Phase 4.2: Present conflicts to user
│     └─ Phase 4.3: Apply resolution strategies
│     └─ Returns: CONFLICT_RESOLUTION.md              ← COLLAPSED
│  ELSE:
│  └─ Skip to Phase 5
│
├─ Phase 5: TDD Task Generation                       ← ATTACHED (3 tasks)
│  └─ /workflow:tools:task-generate-tdd
│     ├─ Phase 5.1: Discovery - analyze TDD requirements
│     ├─ Phase 5.2: Planning - design Red-Green-Refactor cycles
│     └─ Phase 5.3: Output - generate IMPL tasks with internal TDD phases
│     └─ Returns: IMPL-*.json, IMPL_PLAN.md           ← COLLAPSED
│        (Each IMPL task contains internal Red-Green-Refactor cycle)
│
└─ Phase 6: TDD Structure Validation
   └─ Internal validation + summary returned
   └─ Recommend: /workflow:action-plan-verify

Key Points:
• ← ATTACHED: SlashCommand attaches sub-tasks to orchestrator TodoWrite
• ← COLLAPSED: Sub-tasks executed and collapsed to phase summary
• TDD-specific: Each generated IMPL task contains complete Red-Green-Refactor cycle
```

## Input Processing

Convert user input to TDD-structured format:

**Simple text** → Add TDD context
**Detailed text** → Extract components with TEST_FOCUS
**File/Issue** → Read and structure with TDD

## Error Handling

- **Parsing failure**: Retry once, then report
- **Validation failure**: Report missing/invalid data
- **Command failure**: Keep phase in_progress, report error
- **TDD validation failure**: Report incomplete chains or wrong dependencies

## Related Commands

**Prerequisite Commands**:
- None - TDD planning is self-contained (can optionally run brainstorm commands before)

**Called by This Command** (6 phases):
- `/workflow:session:start` - Phase 1: Create or discover TDD workflow session
- `/workflow:tools:context-gather` - Phase 2: Gather project context and analyze codebase
- `/workflow:tools:test-context-gather` - Phase 3: Analyze existing test patterns and coverage
- `/workflow:tools:conflict-resolution` - Phase 4: Detect and resolve conflicts (auto-triggered if conflict_risk ≥ medium)
- `/compact` - Phase 4: Memory optimization (if context approaching limits)
- `/workflow:tools:task-generate-tdd` - Phase 5: Generate TDD tasks (CLI tool usage determined semantically)

**Follow-up Commands**:
- `/workflow:action-plan-verify` - Recommended: Verify TDD plan quality and structure before execution
- `/workflow:status` - Review TDD task breakdown
- `/workflow:execute` - Begin TDD implementation
- `/workflow:tdd-verify` - Post-execution: Verify TDD compliance and generate quality report

