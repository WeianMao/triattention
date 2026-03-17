---
name: test-cycle-execute
description: Execute test-fix workflow with dynamic task generation and iterative fix cycles until test pass rate >= 95% or max iterations reached. Uses @cli-planning-agent for failure analysis and task generation.
argument-hint: "[--resume-session=\"session-id\"] [--max-iterations=N]"
allowed-tools: SlashCommand(*), TodoWrite(*), Read(*), Bash(*), Task(*)
---

# Workflow Test-Cycle-Execute Command

## Quick Start

```bash
# Execute test-fix workflow (auto-discovers active session)
/workflow:test-cycle-execute

# Resume interrupted session
/workflow:test-cycle-execute --resume-session="WFS-test-user-auth"

# Custom iteration limit (default: 10)
/workflow:test-cycle-execute --max-iterations=15
```

**Quality Gate**: Test pass rate >= 95% (criticality-aware) or 100%
**Max Iterations**: 10 (default, adjustable)
**CLI Tools**: Gemini → Qwen → Codex (fallback chain)

## What & Why

### Core Concept
Dynamic test-fix orchestrator with **adaptive task generation** based on runtime analysis.

**vs Standard Execute**:
- **Standard**: Pre-defined tasks → Execute sequentially → Done
- **Test-Cycle**: Initial tasks → **Test → Analyze failures → Generate fix tasks → Fix → Re-test** → Repeat until pass

### Value Proposition
1. **Automatic Problem Solving**: No manual intervention needed until 95% pass rate
2. **Intelligent Adaptation**: Strategy adjusts based on progress (conservative → aggressive → surgical)
3. **Fast Feedback**: Progressive testing runs only affected tests (70-90% faster)

### Orchestrator Boundary (CRITICAL)
- **ONLY command** handling test failures - always delegate here
- Manages: iteration loop, strategy selection, pass rate validation
- Delegates: CLI analysis to @cli-planning-agent, execution to @test-fix-agent

## How It Works

### Execution Flow 

```
1. Discovery
   └─ Load session, tasks, iteration state

2. Main Loop (for each task):
   ├─ Execute → Test → Calculate pass_rate
   ├─ Decision:
   │  ├─ 100% → SUCCESS: Next task
   │  ├─ 95-99% + low criticality → PARTIAL SUCCESS: Approve with note
   │  └─ <95% or critical failures → Fix Loop ↓
   └─ Fix Loop:
      ├─ Detect: stuck tests, regression, progress trend
      ├─ Select strategy: conservative/aggressive/surgical
      ├─ Generate fix task via @cli-planning-agent (IMPL-fix-N.json)
      ├─ Execute fix via @test-fix-agent
      └─ Re-test → Back to step 2

3. Completion
   └─ Final validation → Generate summary → Auto-complete session
```

### Agent Roles

| Agent | Responsibility |
|-------|---------------|
| **Orchestrator** | Loop control, strategy selection, pass rate calculation, threshold decisions |
| **@cli-planning-agent** | CLI analysis (Gemini/Qwen/Codex), root cause extraction, task generation, affected test detection |
| **@test-fix-agent** | Test execution, code fixes, criticality assignment, result reporting |

## Enhanced Features

### 1. Intelligent Strategy Engine

**Auto-selects optimal strategy based on iteration context:**

| Strategy | Trigger | Behavior |
|----------|---------|----------|
| **Conservative** | Iteration 1-2 (default) | Single targeted fix, full validation |
| **Aggressive** | Pass rate >80% + similar failures | Batch fix related issues |
| **Surgical** | Regression detected (pass rate drops >10%) | Minimal changes, rollback focus |

**Selection Logic** (in orchestrator):
```javascript
if (iteration <= 2) return "conservative";
if (passRate > 80 && failurePattern.similarity > 0.7) return "aggressive";
if (regressionDetected) return "surgical";
return "conservative";
```

**Integration**: Strategy passed to @cli-planning-agent in prompt for tailored analysis.

### 2. Progressive Testing

**Runs affected tests during iterations, full suite only for final validation.**

**How It Works**:
1. @cli-planning-agent analyzes fix_strategy.modification_points
2. Maps modified files to test files (via imports + integration patterns)
3. Returns `affected_tests[]` in task JSON
4. @test-fix-agent runs: `npm test -- ${affected_tests.join(' ')}`
5. Final validation: `npm test` (full suite)


**Benefits**: 70-90% iteration speed improvement, instant feedback on fix effectiveness.

## Core Responsibilities

### Orchestrator
- Session discovery, task queue management
- Pass rate calculation: `(passed / total) * 100` from test-results.json
- Criticality assessment (high/medium/low)
- Strategy selection based on context
- **Runtime Calculations** (from iteration-state.json):
  - Current iteration: `iterations.length + 1`
  - Stuck tests: Tests appearing in `failed_tests` for 3+ consecutive iterations
  - Regression: Compare consecutive `pass_rate` values (>10% drop)
  - Max iterations: Read from `task.meta.max_iterations`
- Iteration control (max 10, default)
- CLI tool fallback chain: Gemini → Qwen → Codex
- TodoWrite progress tracking
- Session auto-complete (pass rate >= 95%)

### @cli-planning-agent
- Execute CLI analysis with bug diagnosis template
- Parse output for root causes and fix strategy
- Generate IMPL-fix-N.json task definition
- Detect affected tests for progressive testing
- Save: analysis.md, cli-output.txt

### @test-fix-agent
- Execute tests, save results to test-results.json
- Apply fixes from task.context.fix_strategy
- Assign criticality to failures
- Update task status

## Reference

### CLI Tool Configuration

**Fallback Chain**: Gemini → Qwen → Codex
**Template**: `~/.claude/workflows/cli-templates/prompts/analysis/01-diagnose-bug-root-cause.txt`
**Timeout**: 40min (2400000ms)

**Tool Details**:
1. **Gemini** (primary): `gemini-2.5-pro`
2. **Qwen** (fallback): `coder-model`
3. **Codex** (fallback): `gpt-5.1-codex`

**When to Fallback**: HTTP 429, timeout, analysis quality degraded

### Session File Structure

```
.workflow/active/WFS-test-{session}/
├── workflow-session.json           # Session metadata
├── IMPL_PLAN.md, TODO_LIST.md
├── .task/
│   ├── IMPL-{001,002}.json         # Initial tasks
│   └── IMPL-fix-{N}.json           # Generated fix tasks
├── .process/
│   ├── iteration-state.json        # Current iteration + strategy + stuck tests
│   ├── test-results.json           # Latest results (pass_rate, criticality)
│   ├── test-output.log             # Full test output
│   ├── fix-history.json            # All fix attempts
│   ├── iteration-{N}-analysis.md   # CLI analysis report
│   └── iteration-{N}-cli-output.txt
└── .summaries/iteration-summaries/
```

### Iteration State JSON

**Purpose**: Persisted state machine for iteration loop - enables Resume and historical analysis.

```json
{
  "current_task": "IMPL-002",
  "selected_strategy": "aggressive",
  "next_action": "execute_fix_task",
  "iterations": [
    {
      "iteration": 1,
      "pass_rate": 70,
      "strategy": "conservative",
      "failed_tests": ["test_auth_flow", "test_user_permissions"]
    },
    {
      "iteration": 2,
      "pass_rate": 82,
      "strategy": "conservative",
      "failed_tests": ["test_user_permissions", "test_token_expiry"]
    },
    {
      "iteration": 3,
      "pass_rate": 89,
      "strategy": "aggressive",
      "failed_tests": ["test_auth_edge_case"]
    }
  ]
}
```

**Field Descriptions**:
- `current_task`: Pointer to active task (essential for Resume)
- `selected_strategy`: Current iteration strategy (runtime state)
- `next_action`: State machine next step (`execute_fix_task` | `retest` | `complete`)
- `iterations[]`: Historical log of all iterations (source of truth for trends)


### Agent Invocation Template

**@cli-planning-agent** (failure analysis):
```javascript
Task(
  subagent_type="cli-planning-agent",
  description=`Analyze test failures (iteration ${N}) - ${strategy} strategy`,
  prompt=`
    ## Task Objective
    Analyze test failures and generate fix task JSON for iteration ${N}

    ## Strategy
    ${selectedStrategy} - ${strategyDescription}

    ## MANDATORY FIRST STEPS
    1. Read test results: ${session.test_results_path}
    2. Read test output: ${session.test_output_path}
    3. Read iteration state: ${session.iteration_state_path}

    ## Context Metadata (Orchestrator-Calculated)
    - Session ID: ${sessionId} (from file path)
    - Current Iteration: ${N} (= iterations.length + 1)
    - Max Iterations: ${maxIterations} (from task.meta.max_iterations)
    - Current Pass Rate: ${passRate}%
    - Selected Strategy: ${selectedStrategy} (from iteration-state.json)
    - Stuck Tests: ${stuckTests} (calculated from iterations[].failed_tests history)

    ## CLI Configuration
    - Tool Priority: gemini & codex
    - Template: 01-diagnose-bug-root-cause.txt
    - Timeout: 2400000ms

    ## Expected Deliverables
    1. Task JSON: ${session.task_dir}/IMPL-fix-${N}.json
       - Must include: fix_strategy.test_execution.affected_tests[]
       - Must include: fix_strategy.confidence_score
    2. Analysis report: ${session.process_dir}/iteration-${N}-analysis.md
    3. CLI output: ${session.process_dir}/iteration-${N}-cli-output.txt

    ## Strategy-Specific Requirements
    - Conservative: Single targeted fix, high confidence required
    - Aggressive: Batch fix similar failures, pattern-based approach
    - Surgical: Minimal changes, focus on rollback safety

    ## Success Criteria
    - Concrete fix strategy with modification points (file:function:lines)
    - Affected tests list for progressive testing
    - Root cause analysis (not just symptoms)
  `
)
```

**@test-fix-agent** (execution):
```javascript
Task(
  subagent_type="test-fix-agent",
  description=`Execute ${task.meta.type}: ${task.title}`,
  prompt=`
    ## Task Objective
    ${taskTypeObjective[task.meta.type]}

    ## MANDATORY FIRST STEPS
    1. Read task JSON: ${session.task_json_path}
    2. Read iteration state: ${session.iteration_state_path}
    3. ${taskTypeSpecificReads[task.meta.type]}

    ## CRITICAL: Syntax Check Priority
    **Before any code modification or test execution:**
    - Run project syntax checker (TypeScript: tsc --noEmit, ESLint, etc.)
    - Verify zero syntax errors before proceeding
    - If syntax errors found: Fix immediately before other work
    - Syntax validation is MANDATORY gate - no exceptions

    ## Session Paths
    - Workflow Dir: ${session.workflow_dir}
    - Task JSON: ${session.task_json_path}
    - Test Results Output: ${session.test_results_path}
    - Test Output Log: ${session.test_output_path}
    - Iteration State: ${session.iteration_state_path}

    ## Task Type: ${task.meta.type}
    ${taskTypeGuidance[task.meta.type]}

    ## Expected Deliverables
    ${taskTypeDeliverables[task.meta.type]}

    ## Success Criteria
    - ${taskTypeSuccessCriteria[task.meta.type]}
    - Update task status in task JSON
    - Save all outputs to specified paths
    - Report completion to orchestrator
  `
)

// Task Type Configurations
const taskTypeObjective = {
  "test-gen": "Generate comprehensive tests based on requirements",
  "test-fix": "Execute test suite and report results with criticality assessment",
  "test-fix-iteration": "Apply fixes from strategy and validate with tests"
};

const taskTypeSpecificReads = {
  "test-gen": "Read test context: ${session.test_context_path}",
  "test-fix": "Read previous results (if exists): ${session.test_results_path}",
  "test-fix-iteration": "Read fix strategy: ${session.analysis_path}, fix history: ${session.fix_history_path}"
};

const taskTypeGuidance = {
  "test-gen": `
    - Review task.context.requirements for test scenarios
    - Analyze codebase to understand implementation
    - Generate tests covering: happy paths, edge cases, error handling
    - Follow existing test patterns and framework conventions
  `,
  "test-fix": `
    - Run test command from task.context or project config
    - Capture: pass/fail counts, error messages, stack traces
    - Assess criticality for each failure:
      * high: core functionality broken, security issues
      * medium: feature degradation, data integrity issues
      * low: edge cases, flaky tests, env-specific issues
    - Save structured results to test-results.json
  `,
  "test-fix-iteration": `
    - Load fix_strategy from task.context.fix_strategy
    - Identify modification_points: ${task.context.fix_strategy.modification_points}
    - Apply surgical fixes (minimal changes)
    - Test execution mode: ${task.context.fix_strategy.test_execution.mode}
      * affected_only: Run ${task.context.fix_strategy.test_execution.affected_tests}
      * full_suite: Run complete test suite
    - If failures persist: Document in test-results.json, DO NOT analyze (orchestrator handles)
  `
};

const taskTypeDeliverables = {
  "test-gen": "- Test files in target directories\n    - Test coverage report\n    - Summary in .summaries/",
  "test-fix": "- test-results.json (pass_rate, criticality, failures)\n    - test-output.log (full test output)\n    - Summary in .summaries/",
  "test-fix-iteration": "- Modified source files\n    - test-results.json (updated pass_rate)\n    - test-output.log\n    - Summary in .summaries/"
};

const taskTypeSuccessCriteria = {
  "test-gen": "All test files created, executable without errors, coverage documented",
  "test-fix": "Test results saved with accurate pass_rate and criticality, all failures documented",
  "test-fix-iteration": "Fixes applied per strategy, tests executed, results reported (pass/fail to orchestrator)"
};
```

### Completion Conditions

**Full Success**:
- All tasks completed
- Pass rate === 100%
- Action: Auto-complete session

**Partial Success**:
- All tasks completed
- Pass rate >= 95% and < 100%
- All failures are "low" criticality
- Action: Auto-approve with review note

**Failure**:
- Max iterations (10) reached without 95% pass rate
- Pass rate < 95% after max iterations
- Action: Generate failure report, mark blocked, return to user

### Error Handling

| Scenario | Action |
|----------|--------|
| Test execution error | Log, retry with error context |
| CLI analysis failure | Fallback: Gemini → Qwen → Codex → manual |
| Agent execution error | Save state, retry with simplified context |
| Max iterations reached | Generate failure report, mark blocked |
| Regression detected | Rollback last fix, switch to surgical strategy |
| Stuck tests detected | Continue with alternative strategy, document in failure report |

**CLI Fallback Triggers** (Gemini → Qwen → Codex → manual):

Fallback is triggered when any of these conditions occur:

1. **Invalid Output**:
   - CLI tool fails to generate valid `IMPL-fix-N.json` (JSON parse error)
   - Missing required fields: `fix_strategy.modification_points` or `fix_strategy.affected_tests`

2. **Low Confidence**:
   - `fix_strategy.confidence_score < 0.4` (indicates uncertain analysis)

3. **Technical Failures**:
   - HTTP 429 (rate limit) or 5xx errors
   - Timeout (exceeds 2400000ms / 40min)
   - Connection errors

4. **Quality Degradation**:
   - Analysis report < 100 words (too brief, likely incomplete)
   - No concrete modification points provided (only general suggestions)
   - Same root cause identified 3+ consecutive times (stuck analysis)

**Fallback Sequence**:
- Try primary tool (Gemini)
- If trigger detected → Try fallback (Qwen)
- If trigger detected again → Try final fallback (Codex)
- If all fail → Mark as degraded, use basic pattern matching from fix-history.json, notify user

### TodoWrite Structure

```javascript
TodoWrite({
  todos: [
    {
      content: "Execute IMPL-001: Generate tests [code-developer]",
      status: "completed",
      activeForm: "Executing test generation"
    },
    {
      content: "Execute IMPL-002: Test & Fix Cycle [ITERATION]",
      status: "in_progress",
      activeForm: "Running test-fix iteration cycle"
    },
    {
      content: "  → Iteration 1: Initial test (pass: 70%, conservative)",
      status: "completed",
      activeForm: "Running initial tests"
    },
    {
      content: "  → Iteration 2: Fix validation (pass: 82%, conservative)",
      status: "completed",
      activeForm: "Fixing validation issues"
    },
    {
      content: "  → Iteration 3: Batch fix auth (pass: 89%, aggressive)",
      status: "in_progress",
      activeForm: "Fixing authentication issues"
    }
  ]
});
```

**Update Rules**:
- Add iteration item with: strategy, pass rate
- Mark completed after each iteration
- Update parent task when all complete

## Commit Strategy

**Automatic Commits** (orchestrator-managed):

The orchestrator automatically creates git commits at key checkpoints to enable safe rollback:

1. **After Successful Iteration** (pass rate increased):
   ```bash
   git add .
   git commit -m "test-cycle: iteration ${N} - ${strategy} strategy (pass: ${oldRate}% → ${newRate}%)"
   ```

2. **Before Rollback** (regression detected):
   ```bash
   # Current state preserved, then:
   git revert HEAD
   git commit -m "test-cycle: rollback iteration ${N} - regression detected (pass: ${newRate}% < ${oldRate}%)"
   ```

**Commit Content**:
- Modified source files from fix application
- Updated test-results.json, iteration-state.json
- Excludes: temporary files, logs

**Benefits**:
- **Rollback Safety**: Each iteration is a revert point
- **Progress Tracking**: Git history shows iteration evolution
- **Audit Trail**: Clear record of which strategy/iteration caused issues
- **Resume Capability**: Can resume from any checkpoint

**Note**: Final session completion creates additional commit with full summary.

## Best Practices

1. **Default Settings Work**: 10 iterations sufficient for most cases
2. **Automatic Commits**: Orchestrator commits after each successful iteration - no manual intervention needed
3. **Trust Strategy Engine**: Auto-selection based on proven heuristics
4. **Monitor Logs**: Check `.process/iteration-N-analysis.md` for CLI analysis insights
5. **Progressive Testing**: Saves 70-90% iteration time automatically
