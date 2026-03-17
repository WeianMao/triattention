---
name: test-gen
description: Create independent test-fix workflow session from completed implementation session, analyzes code to generate test tasks
argument-hint: "source-session-id"
allowed-tools: SlashCommand(*), TodoWrite(*), Read(*), Bash(*)
---

# Workflow Test Generation Command (/workflow:test-gen)

## Coordinator Role

**This command is a pure orchestrator**: Creates an independent test-fix workflow session for validating a completed implementation. It reuses the standard planning toolchain with automatic cross-session context gathering.

**Core Principles**:
- **Session Isolation**: Creates new `WFS-test-[source]` session to keep verification separate from implementation
- **Context-First**: Prioritizes gathering code changes and summaries from source session
- **Format Reuse**: Creates standard `IMPL-*.json` task, using `meta.type: "test-fix"` for agent assignment
- **Parameter Simplification**: Tools auto-detect test session type via metadata, no manual cross-session parameters needed
- **Semantic CLI Selection**: CLI tool usage is determined by user's task description (e.g., "use Codex for fixes")

**Task Attachment Model**:
- SlashCommand dispatch **expands workflow** by attaching sub-tasks to current TodoWrite
- When a sub-command is dispatched (e.g., `/workflow:tools:test-context-gather`), its internal tasks are attached to the orchestrator's TodoWrite
- Orchestrator **executes these attached tasks** sequentially
- After completion, attached tasks are **collapsed** back to high-level phase summary
- This is **task expansion**, not external delegation

**Auto-Continue Mechanism**:
- TodoList tracks current phase status and dynamically manages task attachment/collapse
- When each phase finishes executing, automatically execute next pending phase
- All phases run autonomously without user interaction
- **⚠️ CONTINUOUS EXECUTION** - Do not stop until all phases complete

**Execution Flow**:
1. Initialize TodoWrite → Create test session → Parse session ID
2. Gather cross-session context (automatic) → Parse context path
3. Analyze implementation with concept-enhanced → Parse ANALYSIS_RESULTS.md
4. Generate test task from analysis → Return summary

**Command Scope**: This command ONLY prepares test workflow artifacts. It does NOT execute tests or implementation. Task execution requires separate user action.

## Core Rules

1. **Start Immediately**: First action is TodoWrite initialization, second action is Phase 1 test session creation
2. **No Preliminary Analysis**: Do not read files or analyze before Phase 1
3. **Parse Every Output**: Extract required data from each phase for next phase
4. **Sequential Execution**: Each phase depends on previous phase's output
5. **Complete All Phases**: Do not return to user until Phase 5 completes (summary returned)
6. **Track Progress**: Update TodoWrite dynamically with task attachment/collapse pattern
7. **Automatic Detection**: context-gather auto-detects test session and gathers source session context
8. **Semantic CLI Selection**: CLI tool usage determined from user's task description, passed to Phase 4
9. **Command Boundary**: This command ends at Phase 5 summary. Test execution is NOT part of this command.
10. **Task Attachment Model**: SlashCommand dispatch **attaches** sub-tasks to current workflow. Orchestrator **executes** these attached tasks itself, then **collapses** them after completion
11. **⚠️ CRITICAL: DO NOT STOP**: Continuous multi-phase workflow. After executing all attached tasks, immediately collapse them and execute next phase

## 5-Phase Execution

### Phase 1: Create Test Session

**Step 1.0: Load Source Session Intent** - Preserve user's original task description for semantic CLI selection

```javascript
// Read source session metadata to get original task description
Read(".workflow/active/[sourceSessionId]/workflow-session.json")
// OR if context-package exists:
Read(".workflow/active/[sourceSessionId]/.process/context-package.json")

// Extract: metadata.task_description or project/description field
// This preserves user's CLI tool preferences (e.g., "use Codex for fixes")
```

**Step 1.1: Dispatch** - Create new test workflow session with preserved intent

```javascript
// Include original task description to enable semantic CLI selection
SlashCommand(command="/workflow:session:start --new \"Test validation for [sourceSessionId]: [originalTaskDescription]\"")
```

**Input**:
- `sourceSessionId` from user argument (e.g., `WFS-user-auth`)
- `originalTaskDescription` from source session metadata (preserves CLI tool preferences)

**Expected Behavior**:
- Creates new session with pattern `WFS-test-[source-slug]` (e.g., `WFS-test-user-auth`)
- Writes metadata to `workflow-session.json`:
  - `workflow_type: "test_session"`
  - `source_session_id: "[sourceSessionId]"`
  - Description includes original user intent for semantic CLI selection
- Returns new session ID for subsequent phases

**Parse Output**:
- Extract: new test session ID (store as `testSessionId`)
- Pattern: `WFS-test-[slug]`

**Validation**:
- Source session `.workflow/[sourceSessionId]/` exists
- Source session has completed IMPL tasks (`.summaries/IMPL-*-summary.md`)
- New test session directory created
- Metadata includes `workflow_type` and `source_session_id`

**TodoWrite**: Mark phase 1 completed, phase 2 in_progress

---

### Phase 2: Gather Test Context

**Step 2.1: Dispatch** - Gather test coverage context from source session

```javascript
SlashCommand(command="/workflow:tools:test-context-gather --session [testSessionId]")
```

**Input**: `testSessionId` from Phase 1 (e.g., `WFS-test-user-auth`)

**Expected Behavior**:
- Load source session implementation context and summaries
- Analyze test coverage using MCP tools (find existing tests)
- Identify files requiring tests (coverage gaps)
- Detect test framework and conventions
- Generate `test-context-package.json`

**Parse Output**:
- Extract: test context package path (store as `testContextPath`)
- Pattern: `.workflow/[testSessionId]/.process/test-context-package.json`

**Validation**:
- Test context package created
- Contains source session summaries
- Includes coverage gap analysis
- Test framework detected
- Test conventions documented

<!-- TodoWrite: When test-context-gather dispatched, INSERT 3 test-context-gather tasks -->

**TodoWrite Update (Phase 2 SlashCommand dispatched - tasks attached)**:
```json
[
  {"content": "Create independent test session", "status": "completed", "activeForm": "Creating test session"},
  {"content": "Phase 2.1: Load source session summaries (test-context-gather)", "status": "in_progress", "activeForm": "Loading source session summaries"},
  {"content": "Phase 2.2: Analyze test coverage with MCP tools (test-context-gather)", "status": "pending", "activeForm": "Analyzing test coverage"},
  {"content": "Phase 2.3: Identify coverage gaps and framework (test-context-gather)", "status": "pending", "activeForm": "Identifying coverage gaps"},
  {"content": "Analyze test requirements with Gemini", "status": "pending", "activeForm": "Analyzing test requirements"},
  {"content": "Generate test generation and execution tasks", "status": "pending", "activeForm": "Generating test tasks"},
  {"content": "Return workflow summary", "status": "pending", "activeForm": "Returning workflow summary"}
]
```

**Note**: SlashCommand dispatch **attaches** test-context-gather's 3 tasks. Orchestrator **executes** these tasks.

**Next Action**: Tasks attached → **Execute Phase 2.1-2.3** sequentially

<!-- TodoWrite: After Phase 2 tasks complete, REMOVE Phase 2.1-2.3, restore to orchestrator view -->

**TodoWrite Update (Phase 2 completed - tasks collapsed)**:
```json
[
  {"content": "Create independent test session", "status": "completed", "activeForm": "Creating test session"},
  {"content": "Gather test coverage context", "status": "completed", "activeForm": "Gathering test coverage context"},
  {"content": "Analyze test requirements with Gemini", "status": "pending", "activeForm": "Analyzing test requirements"},
  {"content": "Generate test generation and execution tasks", "status": "pending", "activeForm": "Generating test tasks"},
  {"content": "Return workflow summary", "status": "pending", "activeForm": "Returning workflow summary"}
]
```

**Note**: Phase 2 tasks completed and collapsed to summary.

---

### Phase 3: Test Generation Analysis

**Step 3.1: Dispatch** - Analyze test requirements with Gemini

```javascript
SlashCommand(command="/workflow:tools:test-concept-enhanced --session [testSessionId] --context [testContextPath]")
```

**Input**:
- `testSessionId` from Phase 1
- `testContextPath` from Phase 2

**Expected Behavior**:
- Use Gemini to analyze coverage gaps and implementation context
- Study existing test patterns and conventions
- Generate test requirements for each missing test file
- Design test generation strategy
- Generate `TEST_ANALYSIS_RESULTS.md`

**Parse Output**:
- Verify `.workflow/[testSessionId]/.process/TEST_ANALYSIS_RESULTS.md` created
- Contains test requirements and generation strategy
- Lists test files to create with specifications

**Validation**:
- TEST_ANALYSIS_RESULTS.md exists with complete sections:
  - Coverage Assessment
  - Test Framework & Conventions
  - Test Requirements by File
  - Test Generation Strategy
  - Implementation Targets (test files to create)
  - Success Criteria

<!-- TodoWrite: When test-concept-enhanced dispatched, INSERT 3 concept-enhanced tasks -->

**TodoWrite Update (Phase 3 SlashCommand dispatched - tasks attached)**:
```json
[
  {"content": "Create independent test session", "status": "completed", "activeForm": "Creating test session"},
  {"content": "Gather test coverage context", "status": "completed", "activeForm": "Gathering test coverage context"},
  {"content": "Phase 3.1: Analyze coverage gaps with Gemini (test-concept-enhanced)", "status": "in_progress", "activeForm": "Analyzing coverage gaps"},
  {"content": "Phase 3.2: Study existing test patterns (test-concept-enhanced)", "status": "pending", "activeForm": "Studying test patterns"},
  {"content": "Phase 3.3: Generate test generation strategy (test-concept-enhanced)", "status": "pending", "activeForm": "Generating test strategy"},
  {"content": "Generate test generation and execution tasks", "status": "pending", "activeForm": "Generating test tasks"},
  {"content": "Return workflow summary", "status": "pending", "activeForm": "Returning workflow summary"}
]
```

**Note**: SlashCommand dispatch **attaches** test-concept-enhanced's 3 tasks. Orchestrator **executes** these tasks.

**Next Action**: Tasks attached → **Execute Phase 3.1-3.3** sequentially

<!-- TodoWrite: After Phase 3 tasks complete, REMOVE Phase 3.1-3.3, restore to orchestrator view -->

**TodoWrite Update (Phase 3 completed - tasks collapsed)**:
```json
[
  {"content": "Create independent test session", "status": "completed", "activeForm": "Creating test session"},
  {"content": "Gather test coverage context", "status": "completed", "activeForm": "Gathering test coverage context"},
  {"content": "Analyze test requirements with Gemini", "status": "completed", "activeForm": "Analyzing test requirements"},
  {"content": "Generate test generation and execution tasks", "status": "pending", "activeForm": "Generating test tasks"},
  {"content": "Return workflow summary", "status": "pending", "activeForm": "Returning workflow summary"}
]
```

**Note**: Phase 3 tasks completed and collapsed to summary.

---

### Phase 4: Generate Test Tasks

**Step 4.1: Dispatch** - Generate test task JSON files and planning documents

```javascript
SlashCommand(command="/workflow:tools:test-task-generate --session [testSessionId]")
```

**Input**:
- `testSessionId` from Phase 1

**Note**: CLI tool usage for fixes is determined semantically from user's task description (e.g., "use Codex for automated fixes").

**Expected Behavior**:
- Parse TEST_ANALYSIS_RESULTS.md from Phase 3
- Extract test requirements and generation strategy
- Generate **TWO task JSON files**:
  - **IMPL-001.json**: Test Generation task (calls @code-developer)
  - **IMPL-002.json**: Test Execution and Fix Cycle task (calls @test-fix-agent)
- Generate IMPL_PLAN.md with test generation and execution strategy
- Generate TODO_LIST.md with both tasks

**Parse Output**:
- Verify `.workflow/[testSessionId]/.task/IMPL-001.json` exists (test generation)
- Verify `.workflow/[testSessionId]/.task/IMPL-002.json` exists (test execution & fix)
- Verify `.workflow/[testSessionId]/IMPL_PLAN.md` created
- Verify `.workflow/[testSessionId]/TODO_LIST.md` created

**Validation - IMPL-001.json (Test Generation)**:
- Task ID: `IMPL-001`
- `meta.type: "test-gen"`
- `meta.agent: "@code-developer"`
- `context.requirements`: Generate tests based on TEST_ANALYSIS_RESULTS.md
- `flow_control.pre_analysis`: Load TEST_ANALYSIS_RESULTS.md and test context
- `flow_control.implementation_approach`: Test generation steps
- `flow_control.target_files`: Test files to create from analysis section 5

**Validation - IMPL-002.json (Test Execution & Fix)**:
- Task ID: `IMPL-002`
- `meta.type: "test-fix"`
- `meta.agent: "@test-fix-agent"`
- `context.depends_on: ["IMPL-001"]`
- `context.requirements`: Execute and fix tests
- `flow_control.implementation_approach.test_fix_cycle`: Complete cycle specification
  - **Cycle pattern**: test → gemini_diagnose → fix (agent or CLI based on `command` field) → retest
  - **Tools configuration**: Gemini for analysis with bug-fix template, agent or CLI for fixes
  - **Exit conditions**: Success (all pass) or failure (max iterations)
- `flow_control.implementation_approach.modification_points`: 3-phase execution flow
  - Phase 1: Initial test execution
  - Phase 2: Iterative Gemini diagnosis + fixes (agent or CLI based on step's `command` field)
  - Phase 3: Final validation and certification

<!-- TodoWrite: When test-task-generate dispatched, INSERT 3 test-task-generate tasks -->

**TodoWrite Update (Phase 4 SlashCommand dispatched - tasks attached)**:
```json
[
  {"content": "Create independent test session", "status": "completed", "activeForm": "Creating test session"},
  {"content": "Gather test coverage context", "status": "completed", "activeForm": "Gathering test coverage context"},
  {"content": "Analyze test requirements with Gemini", "status": "completed", "activeForm": "Analyzing test requirements"},
  {"content": "Phase 4.1: Parse TEST_ANALYSIS_RESULTS.md (test-task-generate)", "status": "in_progress", "activeForm": "Parsing test analysis"},
  {"content": "Phase 4.2: Generate IMPL-001.json and IMPL-002.json (test-task-generate)", "status": "pending", "activeForm": "Generating task JSONs"},
  {"content": "Phase 4.3: Generate IMPL_PLAN.md and TODO_LIST.md (test-task-generate)", "status": "pending", "activeForm": "Generating plan documents"},
  {"content": "Return workflow summary", "status": "pending", "activeForm": "Returning workflow summary"}
]
```

**Note**: SlashCommand dispatch **attaches** test-task-generate's 3 tasks. Orchestrator **executes** these tasks.

**Next Action**: Tasks attached → **Execute Phase 4.1-4.3** sequentially

<!-- TodoWrite: After Phase 4 tasks complete, REMOVE Phase 4.1-4.3, restore to orchestrator view -->

**TodoWrite Update (Phase 4 completed - tasks collapsed)**:
```json
[
  {"content": "Create independent test session", "status": "completed", "activeForm": "Creating test session"},
  {"content": "Gather test coverage context", "status": "completed", "activeForm": "Gathering test coverage context"},
  {"content": "Analyze test requirements with Gemini", "status": "completed", "activeForm": "Analyzing test requirements"},
  {"content": "Generate test generation and execution tasks", "status": "completed", "activeForm": "Generating test tasks"},
  {"content": "Return workflow summary", "status": "in_progress", "activeForm": "Returning workflow summary"}
]
```

**Note**: Phase 4 tasks completed and collapsed to summary.

---

### Phase 5: Return Summary (Command Ends Here)

**Important**: This is the final phase of `/workflow:test-gen`. The command completes and returns control to the user. No automatic execution occurs.

**Return to User**:
```
Test workflow preparation complete!

Source Session: [sourceSessionId]
Test Session: [testSessionId]

Artifacts Created:
- Test context analysis
- Test generation strategy
- Task definitions (IMPL-001, IMPL-002)
- Implementation plan

Test Framework: [detected framework]
Test Files to Generate: [count]
Fix Mode: [Agent|CLI] (based on `command` field in implementation_approach steps)

Review Generated Artifacts:
- Test plan: .workflow/[testSessionId]/IMPL_PLAN.md
- Task list: .workflow/[testSessionId]/TODO_LIST.md
- Analysis: .workflow/[testSessionId]/.process/TEST_ANALYSIS_RESULTS.md

Ready for execution. Use appropriate workflow commands to proceed.
```

**TodoWrite**: Mark phase 5 completed

**Command Boundary**: After this phase, the command terminates and returns to user prompt.

---

## TodoWrite Pattern

**Core Concept**: Dynamic task attachment and collapse for test-gen workflow with cross-session context gathering and test generation strategy.

### Key Principles

1. **Task Attachment** (when SlashCommand dispatched):
   - Sub-command's internal tasks are **attached** to orchestrator's TodoWrite
   - Example: `/workflow:tools:test-context-gather` attaches 3 sub-tasks (Phase 2.1, 2.2, 2.3)
   - First attached task marked as `in_progress`, others as `pending`
   - Orchestrator **executes** these attached tasks sequentially

2. **Task Collapse** (after sub-tasks complete):
   - Remove detailed sub-tasks from TodoWrite
   - **Collapse** to high-level phase summary
   - Example: Phase 2.1-2.3 collapse to "Gather test coverage context: completed"
   - Maintains clean orchestrator-level view

3. **Continuous Execution**:
   - After collapse, automatically proceed to next pending phase
   - No user intervention required between phases
   - TodoWrite dynamically reflects current execution state

**Lifecycle Summary**: Initial pending tasks → Phase dispatched (tasks ATTACHED) → Sub-tasks executed sequentially → Phase completed (tasks COLLAPSED to summary) → Next phase begins → Repeat until all phases complete.

### Test-Gen Specific Features

- **Phase 2**: Cross-session context gathering from source implementation session
- **Phase 3**: Test requirements analysis with Gemini for generation strategy
- **Phase 4**: Dual-task generation (IMPL-001 for test generation, IMPL-002 for test execution)
- **Fix Mode Configuration**: CLI tool usage determined semantically from user's task description



**Note**: See individual Phase descriptions (Phase 2, 3, 4) for detailed TodoWrite Update examples with full JSON structures.

## Execution Flow Diagram

```
Test-Gen Workflow Orchestrator
│
├─ Phase 1: Create Test Session
│  └─ /workflow:session:start --new
│     └─ Returns: testSessionId (WFS-test-[source])
│
├─ Phase 2: Gather Test Context                           ← ATTACHED (3 tasks)
│  └─ /workflow:tools:test-context-gather
│     ├─ Phase 2.1: Load source session summaries
│     ├─ Phase 2.2: Analyze test coverage with MCP tools
│     └─ Phase 2.3: Identify coverage gaps and framework
│     └─ Returns: test-context-package.json               ← COLLAPSED
│
├─ Phase 3: Test Generation Analysis                      ← ATTACHED (3 tasks)
│  └─ /workflow:tools:test-concept-enhanced
│     ├─ Phase 3.1: Analyze coverage gaps with Gemini
│     ├─ Phase 3.2: Study existing test patterns
│     └─ Phase 3.3: Generate test generation strategy
│     └─ Returns: TEST_ANALYSIS_RESULTS.md                ← COLLAPSED
│
├─ Phase 4: Generate Test Tasks                           ← ATTACHED (3 tasks)
│  └─ /workflow:tools:test-task-generate
│     ├─ Phase 4.1: Parse TEST_ANALYSIS_RESULTS.md
│     ├─ Phase 4.2: Generate IMPL-001.json and IMPL-002.json
│     └─ Phase 4.3: Generate IMPL_PLAN.md and TODO_LIST.md
│     └─ Returns: Task JSONs and plans                    ← COLLAPSED
│
└─ Phase 5: Return Summary
   └─ Command ends, control returns to user

Artifacts Created:
├── .workflow/active/WFS-test-[session]/
│   ├── workflow-session.json
│   ├── IMPL_PLAN.md
│   ├── TODO_LIST.md
│   ├── .task/
│   │   ├── IMPL-001.json (test generation task)
│   │   └── IMPL-002.json (test execution task)
│   └── .process/
│       ├── test-context-package.json
│       └── TEST_ANALYSIS_RESULTS.md

Key Points:
• ← ATTACHED: SlashCommand attaches sub-tasks to orchestrator TodoWrite
• ← COLLAPSED: Sub-tasks executed and collapsed to phase summary
```

## Session Metadata

Test session includes `workflow_type: "test_session"` and `source_session_id` for automatic context gathering.

## Task Output

Generates two task definition files:
- **IMPL-001.json**: Test generation task specification
  - Agent: @code-developer
  - Input: TEST_ANALYSIS_RESULTS.md
  - Output: Test files based on analysis
- **IMPL-002.json**: Test execution and fix cycle specification
  - Agent: @test-fix-agent
  - Dependency: IMPL-001 must complete first
  - Max iterations: 5
  - Fix mode: Agent or CLI (based on `command` field in implementation_approach)

See `/workflow:tools:test-task-generate` for complete task JSON schemas.

## Error Handling

| Phase | Error | Action |
|-------|-------|--------|
| 1 | Source session not found | Return error with source session ID |
| 1 | No completed IMPL tasks | Return error, source incomplete |
| 2 | Context gathering failed | Return error, check source artifacts |
| 3 | Analysis failed | Return error, check context package |
| 4 | Task generation failed | Retry once, then error with details |

## Output Files

Created in `.workflow/active/WFS-test-[session]/`:
- `workflow-session.json` - Session metadata
- `.process/test-context-package.json` - Coverage analysis
- `.process/TEST_ANALYSIS_RESULTS.md` - Test requirements
- `.task/IMPL-001.json` - Test generation task
- `.task/IMPL-002.json` - Test execution & fix task
- `IMPL_PLAN.md` - Test plan
- `TODO_LIST.md` - Task checklist

## Task Specifications

**IMPL-001.json Structure**:
- `meta.type: "test-gen"`
- `meta.agent: "@code-developer"`
- `context.requirements`: Generate tests based on TEST_ANALYSIS_RESULTS.md
- `flow_control.target_files`: Test files to create
- `flow_control.implementation_approach`: Test generation strategy

**IMPL-002.json Structure**:
- `meta.type: "test-fix"`
- `meta.agent: "@test-fix-agent"`
- `context.depends_on: ["IMPL-001"]`
- `flow_control.implementation_approach.test_fix_cycle`: Complete cycle specification
  - Gemini diagnosis template
  - Fix application mode (agent or CLI based on `command` field)
  - Max iterations: 5
- `flow_control.implementation_approach.modification_points`: 3-phase flow

See `/workflow:tools:test-task-generate` for complete JSON schemas.

## Best Practices

1. **Prerequisites**: Ensure source session has completed IMPL tasks with summaries
2. **Clean State**: Commit implementation changes before running test-gen
3. **Review Artifacts**: Check generated IMPL_PLAN.md and TODO_LIST.md before proceeding
4. **Understand Scope**: This command only prepares artifacts; it does not execute tests

## Related Commands

**Prerequisite Commands**:
- `/workflow:plan` or `/workflow:execute` - Complete implementation session that needs test validation

**Dispatched by This Command** (4 phases):
- `/workflow:session:start` - Phase 1: Create independent test workflow session
- `/workflow:tools:test-context-gather` - Phase 2: Analyze test coverage and gather source session context
- `/workflow:tools:test-concept-enhanced` - Phase 3: Generate test requirements and strategy using Gemini
- `/workflow:tools:test-task-generate` - Phase 4: Generate test task JSONs (CLI tool usage determined semantically)

**Follow-up Commands**:
- `/workflow:status` - Review generated test tasks
- `/workflow:test-cycle-execute` - Execute test generation and fix cycles
- `/workflow:execute` - Execute generated test tasks