---
name: execute
description: Coordinate agent execution for workflow tasks with automatic session discovery, parallel task processing, and status tracking
argument-hint: "[--resume-session=\"session-id\"]"
---

# Workflow Execute Command

## Overview
Orchestrates autonomous workflow execution through systematic task discovery, agent coordination, and progress tracking. **Executes entire workflow without user interruption** (except initial session selection if multiple active sessions exist), providing complete context to agents and ensuring proper flow control execution with comprehensive TodoWrite tracking.

**Resume Mode**: When called with `--resume-session` flag, skips discovery phase and directly enters TodoWrite generation and agent execution for the specified session.

## Performance Optimization Strategy

**Lazy Loading**: Task JSONs read **on-demand** during execution, not upfront. TODO_LIST.md + IMPL_PLAN.md provide metadata for planning.

**Loading Strategy**:
- **TODO_LIST.md**: Read in Phase 3 (task metadata, status, dependencies for TodoWrite generation)
- **IMPL_PLAN.md**: Check existence in Phase 2 (normal mode), parse execution strategy in Phase 4A
- **Task JSONs**: Lazy loading - read only when task is about to execute (Phase 4B)

## Core Rules
**Complete entire workflow autonomously without user interruption, using TodoWrite for comprehensive progress tracking.**
**Execute all discovered pending tasks until workflow completion or blocking dependency.**
**Auto-complete session when all tasks finished: Call `/workflow:session:complete` upon workflow completion.**
**ONE AGENT = ONE TASK JSON: Each agent instance executes exactly one task JSON file - never batch multiple tasks into single agent execution.**

## Core Responsibilities
- **Session Discovery**: Identify and select active workflow sessions
- **Execution Strategy Parsing**: Extract execution model from IMPL_PLAN.md
- **TodoWrite Progress Tracking**: Maintain real-time execution status throughout entire workflow
- **Agent Orchestration**: Coordinate specialized agents with complete context
- **Status Synchronization**: Update task JSON files and workflow state
- **Autonomous Completion**: Continue execution until all tasks complete or reach blocking state
- **Session Auto-Complete**: Call `/workflow:session:complete` when all workflow tasks finished

## Execution Philosophy
- **Progress tracking**: Continuous TodoWrite updates throughout entire workflow execution
- **Autonomous completion**: Execute all tasks without user interruption until workflow complete

## Execution Process

```
Normal Mode:
Phase 1: Discovery
   ├─ Count active sessions
   └─ Decision:
      ├─ count=0 → ERROR: No active sessions
      ├─ count=1 → Auto-select session → Phase 2
      └─ count>1 → AskUserQuestion (max 4 options) → Phase 2

Phase 2: Planning Document Validation
   ├─ Check IMPL_PLAN.md exists
   ├─ Check TODO_LIST.md exists
   └─ Validate .task/ contains IMPL-*.json files

Phase 3: TodoWrite Generation
   ├─ Parse TODO_LIST.md for task statuses
   ├─ Generate TodoWrite for entire workflow
   └─ Prepare session context paths

Phase 4: Execution Strategy & Task Execution
   ├─ Step 4A: Parse execution strategy from IMPL_PLAN.md
   └─ Step 4B: Execute tasks with lazy loading
      └─ Loop:
         ├─ Get next in_progress task from TodoWrite
         ├─ Lazy load task JSON
         ├─ Launch agent with task context
         ├─ Mark task completed (update IMPL-*.json status)
         │  # Quick fix: Update task status for ccw dashboard
         │  # TS=$(date -Iseconds) && jq --arg ts "$TS" '.status="completed" | .status_history=(.status_history // [])+[{"from":"in_progress","to":"completed","changed_at":$ts}]' IMPL-X.json > tmp.json && mv tmp.json IMPL-X.json
         └─ Advance to next task

Phase 5: Completion
   ├─ Update task statuses in JSON files
   ├─ Generate summaries
   └─ Auto-call /workflow:session:complete

Resume Mode (--resume-session):
   ├─ Skip Phase 1 & Phase 2
   └─ Entry Point: Phase 3 (TodoWrite Generation)
      └─ Continue: Phase 4 → Phase 5
```

## Execution Lifecycle

### Phase 1: Discovery
**Applies to**: Normal mode only (skipped in resume mode)

**Purpose**: Find and select active workflow session with user confirmation when multiple sessions exist

**Process**:

#### Step 1.1: Count Active Sessions
```bash
bash(find .workflow/active/ -name "WFS-*" -type d 2>/dev/null | wc -l)
```

#### Step 1.2: Handle Session Selection

**Case A: No Sessions** (count = 0)
```
ERROR: No active workflow sessions found
Run /workflow:plan "task description" to create a session
```

**Case B: Single Session** (count = 1)
```bash
bash(find .workflow/active/ -name "WFS-*" -type d 2>/dev/null | head -1 | xargs basename)
```
Auto-select and continue to Phase 2.

**Case C: Multiple Sessions** (count > 1)

List sessions with metadata and prompt user selection:
```bash
bash(for dir in .workflow/active/WFS-*/; do
  session=$(basename "$dir")
  project=$(jq -r '.project // "Unknown"' "$dir/workflow-session.json" 2>/dev/null)
  total=$(grep -c "^- \[" "$dir/TODO_LIST.md" 2>/dev/null || echo "0")
  completed=$(grep -c "^- \[x\]" "$dir/TODO_LIST.md" 2>/dev/null || echo "0")
  [ "$total" -gt 0 ] && progress=$((completed * 100 / total)) || progress=0
  echo "${session} | ${project} | ${completed}/${total} tasks (${progress}%)"
done)
```

Use AskUserQuestion to present formatted options (max 4 options shown):
```javascript
// If more than 4 sessions, show most recent 4 with "Other" option for manual input
const sessions = getActiveSessions()  // sorted by last modified
const displaySessions = sessions.slice(0, 4)

AskUserQuestion({
  questions: [{
    question: "Multiple active sessions detected. Select one:",
    header: "Session",
    multiSelect: false,
    options: displaySessions.map(s => ({
      label: s.id,
      description: `${s.project} | ${s.progress}`
    }))
    // Note: User can select "Other" to manually enter session ID
  }]
})
```

**Input Validation**:
- If user selects from options: Use selected session ID
- If user selects "Other" and provides input: Validate session exists
- If validation fails: Show error and re-prompt or suggest available sessions

Parse user input (supports: number "1", full ID "WFS-auth-system", or partial "auth"), validate selection, and continue to Phase 2.

#### Step 1.3: Load Session Metadata
```bash
bash(cat .workflow/active/${sessionId}/workflow-session.json)
```

**Output**: Store session metadata in memory
**DO NOT read task JSONs yet** - defer until execution phase (lazy loading)

**Resume Mode**: This entire phase is skipped when `--resume-session="session-id"` flag is provided.

### Phase 2: Planning Document Validation
**Applies to**: Normal mode only (skipped in resume mode)

**Purpose**: Validate planning artifacts exist before execution

**Process**:
1. **Check IMPL_PLAN.md**: Verify file exists (defer detailed parsing to Phase 4A)
2. **Check TODO_LIST.md**: Verify file exists (defer reading to Phase 3)
3. **Validate Task Directory**: Ensure `.task/` contains at least one IMPL-*.json file

**Key Optimization**: Only existence checks here. Actual file reading happens in later phases.

**Resume Mode**: This phase is skipped when `--resume-session` flag is provided. Resume mode entry point is Phase 3.

### Phase 3: TodoWrite Generation
**Applies to**: Both normal and resume modes (resume mode entry point)

**Process**:
1. **Create TodoWrite List**: Generate task list from TODO_LIST.md (not from task JSONs)
   - Parse TODO_LIST.md to extract all tasks with current statuses
   - Identify first pending task with met dependencies
   - Generate comprehensive TodoWrite covering entire workflow
2. **Prepare Session Context**: Inject workflow paths for agent use (using provided session-id)
3. **Validate Prerequisites**: Ensure IMPL_PLAN.md and TODO_LIST.md exist and are valid

**Resume Mode Behavior**:
- Load existing TODO_LIST.md directly from `.workflow/active/{session-id}/`
- Extract current progress from TODO_LIST.md
- Generate TodoWrite from TODO_LIST.md state
- Proceed immediately to agent execution (Phase 4)

### Phase 4: Execution Strategy Selection & Task Execution
**Applies to**: Both normal and resume modes

**Step 4A: Parse Execution Strategy from IMPL_PLAN.md**

Read IMPL_PLAN.md Section 4 to extract:
- **Execution Model**: Sequential | Parallel | Phased | TDD Cycles
- **Parallelization Opportunities**: Which tasks can run in parallel
- **Serialization Requirements**: Which tasks must run sequentially
- **Critical Path**: Priority execution order

If IMPL_PLAN.md lacks execution strategy, use intelligent fallback (analyze task structure).

**Step 4B: Execute Tasks with Lazy Loading**

**Key Optimization**: Read task JSON **only when needed** for execution

**Execution Loop Pattern**:
```
while (TODO_LIST.md has pending tasks) {
  next_task_id = getTodoWriteInProgressTask()
  task_json = Read(.workflow/active/{session}/.task/{next_task_id}.json)  // Lazy load
  executeTaskWithAgent(task_json)
  updateTodoListMarkCompleted(next_task_id)
  advanceTodoWriteToNextTask()
}
```

**Execution Process per Task**:
1. **Identify Next Task**: From TodoWrite, get the next `in_progress` task ID
2. **Load Task JSON on Demand**: Read `.task/{task-id}.json` for current task ONLY
3. **Validate Task Structure**: Ensure all 5 required fields exist (id, title, status, meta, context, flow_control)
4. **Launch Agent**: Invoke specialized agent with complete context including flow control steps
5. **Monitor Progress**: Track agent execution and handle errors without user interruption
6. **Collect Results**: Gather implementation results and outputs
7. **Continue Workflow**: Identify next pending task from TODO_LIST.md and repeat

**Note**: TODO_LIST.md updates are handled by agents (e.g., code-developer.md), not by the orchestrator.


### Phase 5: Completion
**Applies to**: Both normal and resume modes

**Process**:
1. **Update Task Status**: Mark completed tasks in JSON files
2. **Generate Summary**: Create task summary in `.summaries/`
3. **Update TodoWrite**: Mark current task complete, advance to next
4. **Synchronize State**: Update session state and workflow status
5. **Check Workflow Complete**: Verify all tasks are completed
6. **Auto-Complete Session**: Call `/workflow:session:complete` when all tasks finished

## Execution Strategy (IMPL_PLAN-Driven)

### Strategy Priority

**IMPL_PLAN-Driven Execution (Recommended)**:
1. **Read IMPL_PLAN.md execution strategy** (Section 4: Implementation Strategy)
2. **Follow explicit guidance**:
   - Execution Model (Sequential/Parallel/Phased/TDD)
   - Parallelization Opportunities (which tasks can run in parallel)
   - Serialization Requirements (which tasks must run sequentially)
   - Critical Path (priority execution order)
3. **Use TODO_LIST.md for status tracking** only
4. **IMPL_PLAN decides "HOW"**, execute.md implements it

**Intelligent Fallback (When IMPL_PLAN lacks execution details)**:
1. **Analyze task structure**:
   - Check `meta.execution_group` in task JSONs
   - Analyze `depends_on` relationships
   - Understand task complexity and risk
2. **Apply smart defaults**:
   - No dependencies + same execution_group → Parallel
   - Has dependencies → Sequential (wait for deps)
   - Critical/high-risk tasks → Sequential
3. **Conservative approach**: When uncertain, prefer sequential execution

### Execution Models

#### 1. Sequential Execution
**When**: IMPL_PLAN specifies "Sequential" OR no clear parallelization guidance
**Pattern**: Execute tasks one by one in TODO_LIST order
**TodoWrite**: ONE task marked as `in_progress` at a time

#### 2. Parallel Execution
**When**: IMPL_PLAN specifies "Parallel" with clear parallelization opportunities
**Pattern**: Execute independent task groups concurrently by launching multiple agent instances
**TodoWrite**: MULTIPLE tasks (in same batch) marked as `in_progress` simultaneously
**Agent Instantiation**: Launch one agent instance per task (respects ONE AGENT = ONE TASK JSON rule)

#### 3. Phased Execution
**When**: IMPL_PLAN specifies "Phased" with phase breakdown
**Pattern**: Execute tasks in phases, respect phase boundaries
**TodoWrite**: Within each phase, follow Sequential or Parallel rules

#### 4. Intelligent Fallback
**When**: IMPL_PLAN lacks execution strategy details
**Pattern**: Analyze task structure and apply smart defaults
**TodoWrite**: Follow Sequential or Parallel rules based on analysis

### Task Status Logic
```
pending + dependencies_met → executable
completed → skip
blocked → skip until dependencies clear
```

## TodoWrite Coordination

### TodoWrite Rules (Unified)

**Rule 1: Initial Creation**
- **Normal Mode**: Generate TodoWrite from discovered pending tasks for entire workflow
- **Resume Mode**: Generate from existing session state and current progress

**Rule 2: In-Progress Task Count (Execution-Model-Dependent)**
- **Sequential execution**: Mark ONLY ONE task as `in_progress` at a time
- **Parallel batch execution**: Mark ALL tasks in current batch as `in_progress` simultaneously
- **Execution group indicator**: Show `[execution_group: group-id]` for parallel tasks

**Rule 3: Status Updates**
- **Immediate Updates**: Update status after each task/batch completion without user interruption
- **Status Synchronization**: Sync with JSON task files after updates
- **Continuous Tracking**: Maintain TodoWrite throughout entire workflow execution until completion

**Rule 4: Workflow Completion Check**
- When all tasks marked `completed`, auto-call `/workflow:session:complete`

### TodoWrite Tool Usage

**Example 1: Sequential Execution**
```javascript
TodoWrite({
  todos: [
    {
      content: "Execute IMPL-1.1: Design auth schema [code-developer] [FLOW_CONTROL]",
      status: "in_progress",  // ONE task in progress
      activeForm: "Executing IMPL-1.1: Design auth schema"
    },
    {
      content: "Execute IMPL-1.2: Implement auth logic [code-developer] [FLOW_CONTROL]",
      status: "pending",
      activeForm: "Executing IMPL-1.2: Implement auth logic"
    }
  ]
});
```

**Example 2: Parallel Batch Execution**
```javascript
TodoWrite({
  todos: [
    {
      content: "Execute IMPL-1.1: Build Auth API [code-developer] [execution_group: parallel-auth-api]",
      status: "in_progress",  // Batch task 1
      activeForm: "Executing IMPL-1.1: Build Auth API"
    },
    {
      content: "Execute IMPL-1.2: Build User UI [code-developer] [execution_group: parallel-ui-comp]",
      status: "in_progress",  // Batch task 2 (running concurrently)
      activeForm: "Executing IMPL-1.2: Build User UI"
    },
    {
      content: "Execute IMPL-1.3: Setup Database [code-developer] [execution_group: parallel-db-schema]",
      status: "in_progress",  // Batch task 3 (running concurrently)
      activeForm: "Executing IMPL-1.3: Setup Database"
    },
    {
      content: "Execute IMPL-2.1: Integration Tests [test-fix-agent] [depends_on: IMPL-1.1, IMPL-1.2, IMPL-1.3]",
      status: "pending",  // Next batch (waits for current batch completion)
      activeForm: "Executing IMPL-2.1: Integration Tests"
    }
  ]
});
```

## Agent Execution Pattern

### Flow Control Execution
**[FLOW_CONTROL]** marker indicates task JSON contains `flow_control.pre_analysis` steps for context preparation.

**Note**: Orchestrator does NOT execute flow control steps - Agent interprets and executes them autonomously.

### Agent Prompt Template
**Dynamic Generation**: Before agent invocation, read task JSON and extract key requirements.

```bash
Task(subagent_type="{meta.agent}",
     prompt="Execute task: {task.title}

     {[FLOW_CONTROL]}

     **Task Objectives** (from task JSON):
     {task.context.objective}

     **Expected Deliverables** (from task JSON):
     {task.context.deliverables}

     **Quality Standards** (from task JSON):
     {task.context.acceptance_criteria}

     **MANDATORY FIRST STEPS**:
     1. Read complete task JSON: {session.task_json_path}
     2. Load context package: {session.context_package_path}

     Follow complete execution guidelines in @.claude/agents/{meta.agent}.md

     **Session Paths**:
     - Workflow Dir: {session.workflow_dir}
     - TODO List: {session.todo_list_path}
     - Summaries Dir: {session.summaries_dir}
     - Context Package: {session.context_package_path}

     **Success Criteria**: Complete all objectives, meet all quality standards, deliver all outputs as specified above.",
     description="Executing: {task.title}")
```

### Agent Assignment Rules
```
meta.agent specified → Use specified agent
meta.agent missing → Infer from meta.type:
  - "feature" → @code-developer
  - "test-gen" → @code-developer
  - "test-fix" → @test-fix-agent
  - "review" → @universal-executor
  - "docs" → @doc-generator
```

## Workflow File Structure Reference
```
.workflow/active/WFS-[topic-slug]/
├── workflow-session.json     # Session state and metadata
├── IMPL_PLAN.md             # Planning document and requirements
├── TODO_LIST.md             # Progress tracking (updated by agents)
├── .task/                   # Task definitions (JSON only)
│   ├── IMPL-1.json          # Main task definitions
│   └── IMPL-1.1.json        # Subtask definitions
├── .summaries/              # Task completion summaries
│   ├── IMPL-1-summary.md    # Task completion details
│   └── IMPL-1.1-summary.md  # Subtask completion details
└── .process/                # Planning artifacts
    ├── context-package.json # Smart context package
    └── ANALYSIS_RESULTS.md  # Planning analysis results
```

## Error Handling & Recovery

### Common Errors & Recovery

| Error Type | Cause | Recovery Strategy | Max Attempts |
|-----------|-------|------------------|--------------|
| **Discovery Errors** |
| No active session | No sessions in `.workflow/active/` | Create or resume session: `/workflow:plan "project"` | N/A |
| Multiple sessions | Multiple sessions in `.workflow/active/` | Prompt user selection | N/A |
| Corrupted session | Invalid JSON files | Recreate session structure or validate files | N/A |
| **Execution Errors** |
| Agent failure | Agent crash/timeout | Retry with simplified context | 2 |
| Flow control error | Command failure | Skip optional, fail critical | 1 per step |
| Context loading error | Missing dependencies | Reload from JSON, use defaults | 3 |
| JSON file corruption | File system issues | Restore from backup/recreate | 1 |

### Error Prevention
- **Pre-flight Checks**: Validate session integrity before execution
- **Backup Strategy**: Create task snapshots before major operations
- **Atomic Updates**: Update JSON files atomically to prevent corruption
- **Dependency Validation**: Check all depends_on references exist
- **Context Verification**: Ensure all required context is available

