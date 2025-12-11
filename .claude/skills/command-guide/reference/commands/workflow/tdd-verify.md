---
name: tdd-verify
description: Verify TDD workflow compliance against Red-Green-Refactor cycles, generate quality report with coverage analysis

argument-hint: "[optional: WFS-session-id]"
allowed-tools: SlashCommand(*), TodoWrite(*), Read(*), Bash(gemini:*)
---

# TDD Verification Command (/workflow:tdd-verify)

## Coordinator Role

**This command is a pure orchestrator**: Execute 4 phases to verify TDD workflow compliance, test coverage, and Red-Green-Refactor cycle execution.

## Core Responsibilities
- Verify TDD task chain structure
- Analyze test coverage
- Validate TDD cycle execution
- Generate compliance report

## Execution Process

```
Input Parsing:
   └─ Decision (session argument):
      ├─ session-id provided → Use provided session
      └─ No session-id → Auto-detect active session

Phase 1: Session Discovery
   ├─ Validate session directory exists
   └─ TodoWrite: Mark phase 1 completed

Phase 2: Task Chain Validation
   ├─ Load all task JSONs from .task/
   ├─ Extract task IDs and group by feature
   ├─ Validate TDD structure:
   │  ├─ TEST-N.M → IMPL-N.M → REFACTOR-N.M chain
   │  ├─ Dependency verification
   │  └─ Meta field validation (tdd_phase, agent)
   └─ TodoWrite: Mark phase 2 completed

Phase 3: Test Execution Analysis
   └─ /workflow:tools:tdd-coverage-analysis
      ├─ Coverage metrics extraction
      ├─ TDD cycle verification
      └─ Compliance score calculation

Phase 4: Compliance Report Generation
   ├─ Gemini analysis for comprehensive report
   ├─ Generate TDD_COMPLIANCE_REPORT.md
   └─ Return summary to user
```

## 4-Phase Execution

### Phase 1: Session Discovery
**Auto-detect or use provided session**

```bash
# If session-id provided
sessionId = argument

# Else auto-detect active session
find .workflow/active/ -name "WFS-*" -type d | head -1 | sed 's/.*\///'
```

**Extract**: sessionId

**Validation**: Session directory exists

**TodoWrite**: Mark phase 1 completed, phase 2 in_progress

---

### Phase 2: Task Chain Validation
**Validate TDD structure using bash commands**

```bash
# Load all task JSONs
find .workflow/active/{sessionId}/.task/ -name '*.json'

# Extract task IDs
find .workflow/active/{sessionId}/.task/ -name '*.json' -exec jq -r '.id' {} \;

# Check dependencies
find .workflow/active/{sessionId}/.task/ -name 'IMPL-*.json' -exec jq -r '.context.depends_on[]?' {} \;
find .workflow/active/{sessionId}/.task/ -name 'REFACTOR-*.json' -exec jq -r '.context.depends_on[]?' {} \;

# Check meta fields
find .workflow/active/{sessionId}/.task/ -name '*.json' -exec jq -r '.meta.tdd_phase' {} \;
find .workflow/active/{sessionId}/.task/ -name '*.json' -exec jq -r '.meta.agent' {} \;
```

**Validation**:
- For each feature N, verify TEST-N.M → IMPL-N.M → REFACTOR-N.M exists
- IMPL-N.M.context.depends_on includes TEST-N.M
- REFACTOR-N.M.context.depends_on includes IMPL-N.M
- TEST tasks have tdd_phase="red" and agent="@code-review-test-agent"
- IMPL/REFACTOR tasks have tdd_phase="green"/"refactor" and agent="@code-developer"

**Extract**: Chain validation report

**TodoWrite**: Mark phase 2 completed, phase 3 in_progress

---

### Phase 3: Test Execution Analysis
**Command**: `SlashCommand(command="/workflow:tools:tdd-coverage-analysis --session [sessionId]")`

**Input**: sessionId from Phase 1

**Parse Output**:
- Coverage metrics (line, branch, function percentages)
- TDD cycle verification results
- Compliance score

**Validation**:
- `.workflow/active/{sessionId}/.process/test-results.json` exists
- `.workflow/active/{sessionId}/.process/coverage-report.json` exists
- `.workflow/active/{sessionId}/.process/tdd-cycle-report.md` exists

**TodoWrite**: Mark phase 3 completed, phase 4 in_progress

---

### Phase 4: Compliance Report Generation
**Gemini analysis for comprehensive TDD compliance report**

```bash
cd project-root && gemini -p "
PURPOSE: Generate TDD compliance report
TASK: Analyze TDD workflow execution and generate quality report
CONTEXT: @{.workflow/active/{sessionId}/.task/*.json,.workflow/active/{sessionId}/.summaries/*,.workflow/active/{sessionId}/.process/tdd-cycle-report.md}
EXPECTED:
- TDD compliance score (0-100)
- Chain completeness verification
- Test coverage analysis summary
- Quality recommendations
- Red-Green-Refactor cycle validation
- Best practices adherence assessment
RULES: Focus on TDD best practices and workflow adherence. Be specific about violations and improvements.
" > .workflow/active/{sessionId}/TDD_COMPLIANCE_REPORT.md
```

**Output**: TDD_COMPLIANCE_REPORT.md

**TodoWrite**: Mark phase 4 completed

**Return to User**:
```
TDD Verification Report - Session: {sessionId}

## Chain Validation
[COMPLETE] Feature 1: TEST-1.1 → IMPL-1.1 → REFACTOR-1.1 (Complete)
[COMPLETE] Feature 2: TEST-2.1 → IMPL-2.1 → REFACTOR-2.1 (Complete)
[INCOMPLETE] Feature 3: TEST-3.1 → IMPL-3.1 (Missing REFACTOR phase)

## Test Execution
All TEST tasks produced failing tests
All IMPL tasks made tests pass
All REFACTOR tasks maintained green tests

## Coverage Metrics
Line Coverage: {percentage}%
Branch Coverage: {percentage}%
Function Coverage: {percentage}%

## Compliance Score: {score}/100

Detailed report: .workflow/active/{sessionId}/TDD_COMPLIANCE_REPORT.md

Recommendations:
- Complete missing REFACTOR-3.1 task
- Consider additional edge case tests for Feature 2
- Improve test failure message clarity in Feature 1
```

## TodoWrite Pattern

```javascript
// Initialize (before Phase 1)
TodoWrite({todos: [
  {"content": "Identify target session", "status": "in_progress", "activeForm": "Identifying target session"},
  {"content": "Validate task chain structure", "status": "pending", "activeForm": "Validating task chain structure"},
  {"content": "Analyze test execution", "status": "pending", "activeForm": "Analyzing test execution"},
  {"content": "Generate compliance report", "status": "pending", "activeForm": "Generating compliance report"}
]})

// After Phase 1
TodoWrite({todos: [
  {"content": "Identify target session", "status": "completed", "activeForm": "Identifying target session"},
  {"content": "Validate task chain structure", "status": "in_progress", "activeForm": "Validating task chain structure"},
  {"content": "Analyze test execution", "status": "pending", "activeForm": "Analyzing test execution"},
  {"content": "Generate compliance report", "status": "pending", "activeForm": "Generating compliance report"}
]})

// Continue pattern for Phase 2, 3, 4...
```

## Validation Logic

### Chain Validation Algorithm
```
1. Load all task JSONs from .workflow/active/{sessionId}/.task/
2. Extract task IDs and group by feature number
3. For each feature:
   - Check TEST-N.M exists
   - Check IMPL-N.M exists
   - Check REFACTOR-N.M exists (optional but recommended)
   - Verify IMPL-N.M depends_on TEST-N.M
   - Verify REFACTOR-N.M depends_on IMPL-N.M
   - Verify meta.tdd_phase values
   - Verify meta.agent assignments
4. Calculate chain completeness score
5. Report incomplete or invalid chains
```

### Compliance Scoring
```
Base Score: 100 points

Deductions:
- Missing TEST task: -30 points per feature
- Missing IMPL task: -30 points per feature
- Missing REFACTOR task: -10 points per feature
- Wrong dependency: -15 points per error
- Wrong agent: -5 points per error
- Wrong tdd_phase: -5 points per error
- Test didn't fail initially: -10 points per feature
- Tests didn't pass after IMPL: -20 points per feature
- Tests broke during REFACTOR: -15 points per feature

Final Score: Max(0, Base Score - Deductions)
```

## Output Files
```
.workflow/active/{session-id}/
├── TDD_COMPLIANCE_REPORT.md     # Comprehensive compliance report ⭐
└── .process/
    ├── test-results.json         # From tdd-coverage-analysis
    ├── coverage-report.json      # From tdd-coverage-analysis
    └── tdd-cycle-report.md       # From tdd-coverage-analysis
```

## Error Handling

### Session Discovery Errors
| Error | Cause | Resolution |
|-------|-------|------------|
| No active session | No WFS-* directories | Provide session-id explicitly |
| Multiple active sessions | Multiple WFS-* directories | Provide session-id explicitly |
| Session not found | Invalid session-id | Check available sessions |

### Validation Errors
| Error | Cause | Resolution |
|-------|-------|------------|
| Task files missing | Incomplete planning | Run tdd-plan first |
| Invalid JSON | Corrupted task files | Regenerate tasks |
| Missing summaries | Tasks not executed | Execute tasks before verify |

### Analysis Errors
| Error | Cause | Resolution |
|-------|-------|------------|
| Coverage tool missing | No test framework | Configure testing first |
| Tests fail to run | Code errors | Fix errors before verify |
| Gemini analysis fails | Token limit / API error | Retry or reduce context |

## Integration & Usage

### Command Chain
- **Called After**: `/workflow:execute` (when TDD tasks completed)
- **Calls**: `/workflow:tools:tdd-coverage-analysis`, Gemini CLI
- **Related**: `/workflow:tdd-plan`, `/workflow:status`

### Basic Usage
```bash
# Auto-detect active session
/workflow:tdd-verify

# Specify session
/workflow:tdd-verify WFS-auth
```

### When to Use
- After completing all TDD tasks in a workflow
- Before merging TDD workflow branch
- For TDD process quality assessment
- To identify missing TDD steps

## TDD Compliance Report Structure

```markdown
# TDD Compliance Report - {Session ID}

**Generated**: {timestamp}
**Session**: {sessionId}
**Workflow Type**: TDD

## Executive Summary
Overall Compliance Score: {score}/100
Status: {EXCELLENT | GOOD | NEEDS IMPROVEMENT | FAILED}

## Chain Analysis

### Feature 1: {Feature Name}
**Status**: Complete
**Chain**: TEST-1.1 → IMPL-1.1 → REFACTOR-1.1

- **Red Phase**: Test created and failed with clear message
- **Green Phase**: Minimal implementation made test pass
- **Refactor Phase**: Code improved, tests remained green

### Feature 2: {Feature Name}
**Status**: Incomplete
**Chain**: TEST-2.1 → IMPL-2.1 (Missing REFACTOR-2.1)

- **Red Phase**: Test created and failed
- **Green Phase**: Implementation seems over-engineered
- **Refactor Phase**: Missing

**Issues**:
- REFACTOR-2.1 task not completed
- IMPL-2.1 implementation exceeded minimal scope

[Repeat for all features]

## Test Coverage Analysis

### Coverage Metrics
- Line Coverage: {percentage}% {status}
- Branch Coverage: {percentage}% {status}
- Function Coverage: {percentage}% {status}

### Coverage Gaps
- {file}:{lines} - Uncovered error handling
- {file}:{lines} - Uncovered edge case

## TDD Cycle Validation

### Red Phase (Write Failing Test)
- {N}/{total} features had failing tests initially
- Feature 3: No evidence of initial test failure

### Green Phase (Make Test Pass)
- {N}/{total} implementations made tests pass
- All implementations minimal and focused

### Refactor Phase (Improve Quality)
- {N}/{total} features completed refactoring
- Feature 2, 4: Refactoring step skipped

## Best Practices Assessment

### Strengths
- Clear test descriptions
- Good test coverage
- Consistent naming conventions
- Well-structured code

### Areas for Improvement
- Some implementations over-engineered in Green phase
- Missing refactoring steps
- Test failure messages could be more descriptive

## Recommendations

### High Priority
1. Complete missing REFACTOR tasks (Features 2, 4)
2. Verify initial test failures for Feature 3
3. Simplify over-engineered implementations

### Medium Priority
1. Add edge case tests for Features 1, 3
2. Improve test failure message clarity
3. Increase branch coverage to >85%

### Low Priority
1. Add more descriptive test names
2. Consider parameterized tests for similar scenarios
3. Document TDD process learnings

## Conclusion
{Summary of compliance status and next steps}
```

