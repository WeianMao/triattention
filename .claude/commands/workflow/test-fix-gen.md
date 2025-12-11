---
name: test-fix-gen
description: Create test-fix workflow session from session ID, description, or file path with test strategy generation and task planning
argument-hint: "(source-session-id | \"feature description\" | /path/to/file.md)"
allowed-tools: SlashCommand(*), TodoWrite(*), Read(*), Bash(*)
---

# Workflow Test-Fix Generation Command (/workflow:test-fix-gen)

## Overview

### What It Does

This command creates an independent test-fix workflow session for existing code. It orchestrates a 5-phase process to analyze implementation, generate test requirements, and create executable test generation and fix tasks.

**CRITICAL - Command Scope**:
- **This command ONLY generates task JSON files** (IMPL-001.json, IMPL-002.json)
- **Does NOT execute tests or apply fixes** - all execution happens in separate orchestrator
- **Must call `/workflow:test-cycle-execute`** after this command to actually run tests and fixes
- **Test failure handling happens in test-cycle-execute**, not here

### Dual-Mode Support

**Automatic mode detection** based on input pattern:

| Mode | Input Pattern | Context Source | Use Case |
|------|--------------|----------------|----------|
| **Session Mode** | `WFS-xxx` | Source session summaries | Test validation for completed workflow |
| **Prompt Mode** | Text or file path | Direct codebase analysis | Test generation from description |

**Detection Logic**:
```bash
if [[ "$input" == WFS-* ]]; then
  MODE="session"  # Use test-context-gather
else
  MODE="prompt"   # Use context-gather
fi
```

### Core Principles

- **Dual Input Support**: Accepts session ID (WFS-xxx) or feature description/file path
- **Session Isolation**: Creates independent `WFS-test-[slug]` session
- **Context-First**: Gathers implementation context via appropriate method
- **Format Reuse**: Creates standard `IMPL-*.json` tasks with `meta.type: "test-fix"`
- **Semantic CLI Selection**: CLI tool usage determined from user's task description
- **Automatic Detection**: Input pattern determines execution mode

### Coordinator Role

This command is a **pure planning coordinator**:
- Does NOT analyze code directly
- Does NOT generate tests or documentation
- Does NOT execute tests or apply fixes
- Does NOT handle test failures or iterations
- ONLY coordinates slash commands to generate task JSON files
- Parses outputs to pass data between phases
- Creates independent test workflow session
- **All execution delegated to `/workflow:test-cycle-execute`**

**Task Attachment Model**:
- SlashCommand dispatch **expands workflow** by attaching sub-tasks to current TodoWrite
- When dispatching a sub-command (e.g., `/workflow:tools:test-context-gather`), its internal tasks are attached to the orchestrator's TodoWrite
- Orchestrator **executes these attached tasks** sequentially
- After completion, attached tasks are **collapsed** back to high-level phase summary
- This is **task expansion**, not external delegation

**Auto-Continue Mechanism**:
- TodoList tracks current phase status and dynamically manages task attachment/collapse
- When each phase finishes executing, automatically execute next pending phase
- All phases run autonomously without user interaction
- **⚠️ CONTINUOUS EXECUTION** - Do not stop until all phases complete

---

## Usage

### Command Syntax

```bash
# Basic syntax
/workflow:test-fix-gen <INPUT>

# Input
<INPUT>            # Session ID, description, or file path
```

**Note**: CLI tool usage is determined semantically from the task description. To request CLI execution, include it in your description (e.g., "use Codex for automated fixes").

### Usage Examples

#### Session Mode
```bash
# Test validation for completed implementation
/workflow:test-fix-gen WFS-user-auth-v2

# With semantic CLI request
/workflow:test-fix-gen WFS-api-endpoints  # Add "use Codex" in description for automated fixes
```

#### Prompt Mode - Text Description
```bash
# Generate tests from feature description
/workflow:test-fix-gen "Test the user authentication API endpoints in src/auth/api.ts"

# With CLI execution (semantic)
/workflow:test-fix-gen "Test user registration and login flows, use Codex for automated fixes"
```

#### Prompt Mode - File Reference
```bash
# Generate tests from requirements file
/workflow:test-fix-gen ./docs/api-requirements.md
```

### Mode Comparison

| Aspect | Session Mode | Prompt Mode |
|--------|-------------|-------------|
| **Phase 1** | Create `WFS-test-[source]` with `source_session_id` | Create `WFS-test-[slug]` without `source_session_id` |
| **Phase 2** | `/workflow:tools:test-context-gather` | `/workflow:tools:context-gather` |
| **Phase 3-5** | Identical | Identical |
| **Context** | Source session summaries + artifacts | Direct codebase analysis |

---

## Execution Flow

### Core Execution Rules

1. **Start Immediately**: First action is TodoWrite, second is dispatch Phase 1 session creation
2. **No Preliminary Analysis**: Do not read files before Phase 1
3. **Parse Every Output**: Extract required data from each phase for next phase
4. **Sequential Execution**: Each phase depends on previous phase's output
5. **Complete All Phases**: Do not return until Phase 5 completes
6. **Track Progress**: Update TodoWrite dynamically with task attachment/collapse pattern
7. **Automatic Detection**: Mode auto-detected from input pattern
8. **Semantic CLI Detection**: CLI tool usage determined from user's task description for Phase 4
9. **Task Attachment Model**: SlashCommand dispatch **attaches** sub-tasks to current workflow. Orchestrator **executes** these attached tasks itself, then **collapses** them after completion
10. **⚠️ CRITICAL: DO NOT STOP**: Continuous multi-phase workflow. After executing all attached tasks, immediately collapse them and execute next phase

### 5-Phase Execution

#### Phase 1: Create Test Session

**Step 1.0: Load Source Session Intent (Session Mode Only)** - Preserve user's original task description for semantic CLI selection

```javascript
// Session Mode: Read source session metadata to get original task description
Read(".workflow/active/[sourceSessionId]/workflow-session.json")
// OR if context-package exists:
Read(".workflow/active/[sourceSessionId]/.process/context-package.json")

// Extract: metadata.task_description or project/description field
// This preserves user's CLI tool preferences (e.g., "use Codex for fixes")
```

**Step 1.1: Dispatch** - Create test workflow session with preserved intent

```javascript
// Session Mode - Include original task description to enable semantic CLI selection
SlashCommand(command="/workflow:session:start --type test --new \"Test validation for [sourceSessionId]: [originalTaskDescription]\"")

// Prompt Mode - User's description already contains their intent
SlashCommand(command="/workflow:session:start --type test --new \"Test generation for: [description]\"")
```

**Input**: User argument (session ID, description, or file path)

**Expected Behavior**:
- Creates new session: `WFS-test-[slug]`
- Writes `workflow-session.json` metadata with `type: "test"`
  - **Session Mode**: Additionally includes `source_session_id: "[sourceId]"`, description with original user intent
  - **Prompt Mode**: Uses user's description (already contains intent)
- Returns new session ID

**Parse Output**:
- Extract: `testSessionId` (pattern: `WFS-test-[slug]`)

**Validation**:
- **Session Mode**: Source session exists with completed IMPL tasks
- **Both Modes**: New test session directory created with metadata

**TodoWrite**: Mark phase 1 completed, phase 2 in_progress

---

#### Phase 2: Gather Test Context

**Step 2.1: Dispatch** - Gather test context via appropriate method

```javascript
// Session Mode
SlashCommand(command="/workflow:tools:test-context-gather --session [testSessionId]")

// Prompt Mode
SlashCommand(command="/workflow:tools:context-gather --session [testSessionId] \"[task_description]\"")
```

**Input**: `testSessionId` from Phase 1

**Expected Behavior**:
- **Session Mode**:
  - Load source session implementation context and summaries
  - Analyze test coverage using MCP tools
  - Identify files requiring tests
- **Prompt Mode**:
  - Analyze codebase based on description
  - Identify relevant files and dependencies
- Detect test framework and conventions
- Generate context package JSON

**Parse Output**:
- Extract: `contextPath` (pattern: `.workflow/[testSessionId]/.process/[test-]context-package.json`)

**Validation**:
- Context package created with coverage analysis
- Test framework detected
- Test conventions documented

**TodoWrite**: Mark phase 2 completed, phase 3 in_progress

---

#### Phase 3: Test Generation Analysis

**Step 3.1: Dispatch** - Generate test requirements using Gemini

```javascript
SlashCommand(command="/workflow:tools:test-concept-enhanced --session [testSessionId] --context [contextPath]")
```

**Input**:
- `testSessionId` from Phase 1
- `contextPath` from Phase 2

**Expected Behavior**:
- Use Gemini to analyze coverage gaps and implementation
- Study existing test patterns and conventions
- Generate **multi-layered test requirements** (L0: Static Analysis, L1: Unit, L2: Integration, L3: E2E)
- Design test generation strategy with quality assurance criteria
- Generate `TEST_ANALYSIS_RESULTS.md` with structured test layers

**Enhanced Test Requirements**:
For each targeted file/function, Gemini MUST generate:
1. **L0: Static Analysis Requirements**:
   - Linting rules to enforce (ESLint, Prettier)
   - Type checking requirements (TypeScript)
   - Anti-pattern detection rules
2. **L1: Unit Test Requirements**:
   - Happy path scenarios (valid inputs → expected outputs)
   - Negative path scenarios (invalid inputs → error handling)
   - Edge cases (null, undefined, 0, empty strings/arrays)
3. **L2: Integration Test Requirements**:
   - Successful component interactions
   - Failure handling scenarios (service unavailable, timeout)
4. **L3: E2E Test Requirements** (if applicable):
   - Key user journeys from start to finish

**Parse Output**:
- Verify `.workflow/[testSessionId]/.process/TEST_ANALYSIS_RESULTS.md` created

**Validation**:
- TEST_ANALYSIS_RESULTS.md exists with complete sections:
  - Coverage Assessment
  - Test Framework & Conventions
  - **Multi-Layered Test Plan** (NEW):
    - L0: Static Analysis Plan
    - L1: Unit Test Plan
    - L2: Integration Test Plan
    - L3: E2E Test Plan (if applicable)
  - Test Requirements by File (with layer annotations)
  - Test Generation Strategy
  - Implementation Targets
  - Quality Assurance Criteria (NEW):
    - Minimum coverage thresholds
    - Required test types per function
    - Acceptance criteria for test quality
  - Success Criteria

**TodoWrite**: Mark phase 3 completed, phase 4 in_progress

---

#### Phase 4: Generate Test Tasks

**Step 4.1: Dispatch** - Generate test task JSONs

```javascript
SlashCommand(command="/workflow:tools:test-task-generate --session [testSessionId]")
```

**Input**:
- `testSessionId` from Phase 1

**Note**: CLI tool usage is determined semantically from user's task description.

**Expected Behavior**:
- Parse TEST_ANALYSIS_RESULTS.md from Phase 3 (multi-layered test plan)
- Generate **minimum 3 task JSON files** (expandable based on complexity):
  - **IMPL-001.json**: Test Understanding & Generation (`@code-developer`)
  - **IMPL-001.5-review.json**: Test Quality Gate (`@test-fix-agent`) ← **NEW**
  - **IMPL-002.json**: Test Execution & Fix Cycle (`@test-fix-agent`)
  - **IMPL-003+**: Additional tasks if needed for complex projects
- Generate `IMPL_PLAN.md` with multi-layered test strategy
- Generate `TODO_LIST.md` with task checklist

**Parse Output**:
- Verify `.workflow/[testSessionId]/.task/IMPL-001.json` exists
- Verify `.workflow/[testSessionId]/.task/IMPL-001.5-review.json` exists ← **NEW**
- Verify `.workflow/[testSessionId]/.task/IMPL-002.json` exists
- Verify additional `.task/IMPL-*.json` if applicable
- Verify `IMPL_PLAN.md` and `TODO_LIST.md` created

**TodoWrite**: Mark phase 4 completed, phase 5 in_progress

---

#### Phase 5: Return Summary

**Return to User**:
```
Independent test-fix workflow created successfully!

Input: [original input]
Mode: [Session|Prompt]
Test Session: [testSessionId]

Tasks Created:
- IMPL-001: Test Understanding & Generation (@code-developer)
- IMPL-001.5: Test Quality Gate - Static Analysis & Coverage (@test-fix-agent) ← NEW
- IMPL-002: Test Execution & Fix Cycle (@test-fix-agent)
[- IMPL-003+: Additional tasks if applicable]

Test Strategy: Multi-Layered (L0: Static, L1: Unit, L2: Integration, L3: E2E)
Test Framework: [detected framework]
Test Files to Generate: [count]
Quality Thresholds:
- Minimum Coverage: 80%
- Static Analysis: Zero critical issues
Max Fix Iterations: 5
Fix Mode: [Manual|Codex Automated]

Review artifacts:
- Test plan: .workflow/[testSessionId]/IMPL_PLAN.md
- Task list: .workflow/[testSessionId]/TODO_LIST.md

CRITICAL - Next Steps:
1. Review IMPL_PLAN.md (now includes multi-layered test strategy)
2. **MUST execute: /workflow:test-cycle-execute**
   - This command only generated task JSON files
   - Test execution and fix iterations happen in test-cycle-execute
   - Do NOT attempt to run tests or fixes in main workflow
3. IMPL-001.5 will validate test quality before fix cycle begins
```

**TodoWrite**: Mark phase 5 completed

**BOUNDARY NOTE**:
- Command completes here - only task JSON files generated
- All test execution, failure detection, CLI analysis, fix generation happens in `/workflow:test-cycle-execute`
- This command does NOT handle test failures or apply fixes

---

### TodoWrite Pattern

**Core Concept**: Dynamic task attachment and collapse for test-fix-gen workflow with dual-mode support (Session Mode and Prompt Mode).

#### Initial TodoWrite Structure

```json
[
  {"content": "Phase 1: Create Test Session", "status": "in_progress", "activeForm": "Creating test session"},
  {"content": "Phase 2: Gather Test Context", "status": "pending", "activeForm": "Gathering test context"},
  {"content": "Phase 3: Test Generation Analysis", "status": "pending", "activeForm": "Analyzing test generation"},
  {"content": "Phase 4: Generate Test Tasks", "status": "pending", "activeForm": "Generating test tasks"},
  {"content": "Phase 5: Return Summary", "status": "pending", "activeForm": "Completing"}
]
```

#### Key Principles

1. **Task Attachment** (when SlashCommand dispatched):
   - Sub-command's internal tasks are **attached** to orchestrator's TodoWrite
   - Example - Phase 2 with sub-tasks:
   ```json
   [
     {"content": "Phase 1: Create Test Session", "status": "completed", "activeForm": "Creating test session"},
     {"content": "Phase 2: Gather Test Context", "status": "in_progress", "activeForm": "Gathering test context"},
     {"content": "  → Load context and analyze coverage", "status": "in_progress", "activeForm": "Loading context"},
     {"content": "  → Detect test framework and conventions", "status": "pending", "activeForm": "Detecting framework"},
     {"content": "  → Generate context package", "status": "pending", "activeForm": "Generating context"},
     {"content": "Phase 3: Test Generation Analysis", "status": "pending", "activeForm": "Analyzing test generation"},
     {"content": "Phase 4: Generate Test Tasks", "status": "pending", "activeForm": "Generating test tasks"},
     {"content": "Phase 5: Return Summary", "status": "pending", "activeForm": "Completing"}
   ]
   ```

2. **Task Collapse** (after sub-tasks complete):
   - Remove detailed sub-tasks from TodoWrite
   - **Collapse** to high-level phase summary
   - Example - Phase 2 completed:
   ```json
   [
     {"content": "Phase 1: Create Test Session", "status": "completed", "activeForm": "Creating test session"},
     {"content": "Phase 2: Gather Test Context", "status": "completed", "activeForm": "Gathering test context"},
     {"content": "Phase 3: Test Generation Analysis", "status": "in_progress", "activeForm": "Analyzing test generation"},
     {"content": "Phase 4: Generate Test Tasks", "status": "pending", "activeForm": "Generating test tasks"},
     {"content": "Phase 5: Return Summary", "status": "pending", "activeForm": "Completing"}
   ]
   ```

3. **Continuous Execution**:
   - After collapse, automatically proceed to next pending phase
   - No user intervention required between phases
   - TodoWrite dynamically reflects current execution state

**Lifecycle Summary**: Initial pending tasks → Phase dispatched (tasks ATTACHED with mode-specific context gathering) → Sub-tasks executed sequentially → Phase completed (tasks COLLAPSED to summary) → Next phase begins → Repeat until all phases complete.

#### Test-Fix-Gen Specific Features

- **Dual-Mode Support**: Automatic mode detection based on input pattern
  - **Session Mode**: Input pattern `WFS-*` → uses `test-context-gather` for cross-session context
  - **Prompt Mode**: Text or file path → uses `context-gather` for direct codebase analysis
- **Phase 2**: Mode-specific context gathering (session summaries vs codebase analysis)
- **Phase 3**: Multi-layered test requirements analysis (L0: Static, L1: Unit, L2: Integration, L3: E2E)
- **Phase 4**: Multi-task generation with quality gate (IMPL-001, IMPL-001.5-review, IMPL-002)
- **Fix Mode Configuration**: CLI tool usage determined semantically from user's task description


---

## Task Specifications

Generates minimum 3 tasks (expandable for complex projects):

### IMPL-001: Test Understanding & Generation

**Agent**: `@code-developer`

**Purpose**: Understand source implementation and generate test files following multi-layered test strategy

**Task Configuration**:
- Task ID: `IMPL-001`
- `meta.type: "test-gen"`
- `meta.agent: "@code-developer"`
- `context.requirements`: Understand source implementation and generate tests across all layers (L0-L3)
- `flow_control.target_files`: Test files to create from TEST_ANALYSIS_RESULTS.md section 5

**Execution Flow**:
1. **Understand Phase**:
   - Load TEST_ANALYSIS_RESULTS.md and test context
   - Understand source code implementation patterns
   - Analyze multi-layered test requirements (L0: Static, L1: Unit, L2: Integration, L3: E2E)
   - Identify test scenarios, edge cases, and error paths
2. **Generation Phase**:
   - Generate L1 unit test files following existing patterns
   - Generate L2 integration test files (if applicable)
   - Generate L3 E2E test files (if applicable)
   - Ensure test coverage aligns with multi-layered requirements
   - Include both positive and negative test cases
3. **Verification Phase**:
   - Verify test completeness and correctness
   - Ensure each test has meaningful assertions
   - Check for test anti-patterns (tests without assertions, overly broad mocks)

### IMPL-001.5: Test Quality Gate ← **NEW**

**Agent**: `@test-fix-agent`

**Purpose**: Validate test quality before entering fix cycle - prevent "hollow tests" from becoming the source of truth

**Task Configuration**:
- Task ID: `IMPL-001.5-review`
- `meta.type: "test-quality-review"`
- `meta.agent: "@test-fix-agent"`
- `context.depends_on: ["IMPL-001"]`
- `context.requirements`: Validate generated tests meet quality standards
- `context.quality_config`: Load from `.claude/workflows/test-quality-config.json`

**Execution Flow**:
1. **L0: Static Analysis**:
   - Run linting on test files (ESLint, Prettier)
   - Check for test anti-patterns:
     - Tests without assertions (`expect()` missing)
     - Empty test bodies (`it('should...', () => {})`)
     - Disabled tests without justification (`it.skip`, `xit`)
   - Verify TypeScript type safety (if applicable)
2. **Coverage Analysis**:
   - Run coverage analysis on generated tests
   - Calculate coverage percentage for target source files
   - Identify uncovered branches and edge cases
3. **Test Quality Metrics**:
   - Verify minimum coverage threshold met (default: 80%)
   - Verify all critical functions have negative test cases
   - Verify integration tests cover key component interactions
4. **Quality Gate Decision**:
   - **PASS**: Coverage ≥ 80%, zero critical anti-patterns → Proceed to IMPL-002
   - **FAIL**: Coverage < 80% OR critical anti-patterns found → Loop back to IMPL-001 with feedback

**Acceptance Criteria**:
- Static analysis: Zero critical issues
- Test coverage: ≥ 80% for target files
- Test completeness: All targeted functions have unit tests
- Negative test coverage: Each public API has at least one error handling test
- Integration coverage: Key component interactions have integration tests (if applicable)

**Failure Handling**:
If quality gate fails:
1. Generate detailed feedback report (`.process/test-quality-report.md`)
2. Update IMPL-001 task with specific improvement requirements
3. Trigger IMPL-001 re-execution with enhanced context
4. Maximum 2 quality gate retries before escalating to user

### IMPL-002: Test Execution & Fix Cycle

**Agent**: `@test-fix-agent`

**Purpose**: Execute initial tests and trigger orchestrator-managed fix cycles

**Note**: This task executes tests and reports results. The test-cycle-execute orchestrator manages all fix iterations, CLI analysis, and fix task generation.

**Task Configuration**:
- Task ID: `IMPL-002`
- `meta.type: "test-fix"`
- `meta.agent: "@test-fix-agent"`
- `context.depends_on: ["IMPL-001"]`
- `context.requirements`: Execute and fix tests

**Test-Fix Cycle Specification**:
**Note**: This specification describes what test-cycle-execute orchestrator will do. The agent only executes single tasks.
- **Cycle Pattern** (orchestrator-managed): test → gemini_diagnose → fix (agent or CLI) → retest
- **Tools Configuration** (orchestrator-controlled):
  - Gemini for analysis with bug-fix template → surgical fix suggestions
  - Agent fix application (default) OR CLI if `command` field present in implementation_approach
- **Exit Conditions** (orchestrator-enforced):
  - Success: All tests pass
  - Failure: Max iterations reached (5)

**Execution Flow**:
1. **Phase 1**: Initial test execution
2. **Phase 2**: Iterative Gemini diagnosis + manual/Codex fixes
3. **Phase 3**: Final validation and certification

### IMPL-003+: Additional Tasks (Optional)

**Scenarios for Multiple Tasks**:
- Large projects requiring per-module test generation
- Separate integration vs unit test tasks
- Specialized test types (performance, security, etc.)

**Agent**: `@code-developer` or specialized agents based on requirements

---

## Artifacts & Output

### Output Files Structure

Created in `.workflow/active/WFS-test-[session]/`:

```
WFS-test-[session]/
├── workflow-session.json          # Session metadata
├── IMPL_PLAN.md                   # Test generation and execution strategy
├── TODO_LIST.md                   # Task checklist
├── .task/
│   ├── IMPL-001.json              # Test understanding & generation
│   ├── IMPL-002.json              # Test execution & fix cycle
│   └── IMPL-*.json                # Additional tasks (if applicable)
└── .process/
    ├── [test-]context-package.json # Context and coverage analysis
    └── TEST_ANALYSIS_RESULTS.md    # Test requirements and strategy
```

### Session Metadata

**File**: `workflow-session.json`

**Session Mode** includes:
- `type: "test"` (set by session:start --type test)
- `source_session_id: "[sourceSessionId]"` (enables automatic cross-session context)

**Prompt Mode** includes:
- `type: "test"` (set by session:start --type test)
- No `source_session_id` field

### Execution Flow Diagram

```
Test-Fix-Gen Workflow Orchestrator (Dual-Mode Support)
│
├─ Phase 1: Create Test Session
│  ├─ Session Mode: /workflow:session:start --new (with source_session_id)
│  └─ Prompt Mode: /workflow:session:start --new (without source_session_id)
│     └─ Returns: testSessionId (WFS-test-[slug])
│
├─ Phase 2: Gather Context                                   ← ATTACHED (3 tasks)
│  ├─ Session Mode: /workflow:tools:test-context-gather
│  │  └─ Load source session summaries + analyze coverage
│  └─ Prompt Mode: /workflow:tools:context-gather
│     └─ Analyze codebase from description
│     ├─ Phase 2.1: Load context and analyze coverage
│     ├─ Phase 2.2: Detect test framework and conventions
│     └─ Phase 2.3: Generate context package
│     └─ Returns: [test-]context-package.json                ← COLLAPSED
│
├─ Phase 3: Test Generation Analysis                         ← ATTACHED (3 tasks)
│  └─ /workflow:tools:test-concept-enhanced
│     ├─ Phase 3.1: Analyze coverage gaps with Gemini
│     ├─ Phase 3.2: Study existing test patterns
│     └─ Phase 3.3: Generate test generation strategy
│     └─ Returns: TEST_ANALYSIS_RESULTS.md                   ← COLLAPSED
│
├─ Phase 4: Generate Test Tasks                              ← ATTACHED (3 tasks)
│  └─ /workflow:tools:test-task-generate
│     ├─ Phase 4.1: Parse TEST_ANALYSIS_RESULTS.md
│     ├─ Phase 4.2: Generate task JSONs (IMPL-001, IMPL-002)
│     └─ Phase 4.3: Generate IMPL_PLAN.md and TODO_LIST.md
│     └─ Returns: Task JSONs and plans                       ← COLLAPSED
│
└─ Phase 5: Return Summary
   └─ Command ends, control returns to user

Artifacts Created:
├── .workflow/active/WFS-test-[session]/
│   ├── workflow-session.json
│   ├── IMPL_PLAN.md
│   ├── TODO_LIST.md
│   ├── .task/
│   │   ├── IMPL-001.json (test understanding & generation)
│   │   ├── IMPL-002.json (test execution & fix cycle)
│   │   └── IMPL-003.json (optional: test review & certification)
│   └── .process/
│       ├── [test-]context-package.json
│       └── TEST_ANALYSIS_RESULTS.md

Key Points:
• ← ATTACHED: SlashCommand attaches sub-tasks to orchestrator TodoWrite
• ← COLLAPSED: Sub-tasks executed and collapsed to phase summary
• Dual-Mode: Session Mode and Prompt Mode share same attachment pattern
• Command Boundary: Execution delegated to /workflow:test-cycle-execute
```

---

## Reference

### Error Handling

| Phase | Error Condition | Action |
|-------|----------------|--------|
| 1 | Source session not found (session mode) | Return error with source session ID |
| 1 | No completed IMPL tasks (session mode) | Return error, source incomplete |
| 2 | Context gathering failed | Return error, check source artifacts |
| 3 | Gemini analysis failed | Return error, check context package |
| 4 | Task generation failed | Retry once, then return error with details |

### Best Practices

1. **Before Running**:
   - Ensure implementation is complete (session mode: check summaries exist)
   - Commit all implementation changes
   - Review source code quality

2. **After Running**:
   - Review generated `IMPL_PLAN.md` before execution
   - Check `TEST_ANALYSIS_RESULTS.md` for completeness
   - Verify task dependencies in `TODO_LIST.md`

3. **During Execution**:
   - Monitor iteration logs in `.process/fix-iteration-*`
   - Track progress with `/workflow:status`
   - Review Gemini diagnostic outputs

4. **Mode Selection**:
   - Use **Session Mode** for completed workflow validation
   - Use **Prompt Mode** for ad-hoc test generation
   - Include "use Codex" in description for autonomous fix application

## Related Commands

**Prerequisite Commands**:
- `/workflow:plan` or `/workflow:execute` - Complete implementation session (for Session Mode)
- None for Prompt Mode (ad-hoc test generation)

**Called by This Command** (5 phases):
- `/workflow:session:start` - Phase 1: Create independent test workflow session
- `/workflow:tools:test-context-gather` - Phase 2 (Session Mode): Gather source session context
- `/workflow:tools:context-gather` - Phase 2 (Prompt Mode): Analyze codebase directly
- `/workflow:tools:test-concept-enhanced` - Phase 3: Generate test requirements using Gemini
- `/workflow:tools:test-task-generate` - Phase 4: Generate test task JSONs (CLI tool usage determined semantically)

**Follow-up Commands**:
- `/workflow:status` - Review generated test tasks
- `/workflow:test-cycle-execute` - Execute test generation and iterative fix cycles
- `/workflow:execute` - Standard execution of generated test tasks

