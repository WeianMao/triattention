---
name: tdd-coverage-analysis
description: Analyze test coverage and TDD cycle execution with Red-Green-Refactor compliance verification
argument-hint: "--session WFS-session-id"
allowed-tools: Read(*), Write(*), Bash(*)
---

# TDD Coverage Analysis Command

## Overview
Analyze test coverage and verify Red-Green-Refactor cycle execution for TDD workflow validation.

## Core Responsibilities
- Extract test files from TEST tasks
- Run test suite with coverage
- Parse coverage metrics
- Verify TDD cycle execution (Red -> Green -> Refactor)
- Generate coverage and cycle reports

## Execution Process

```
Input Parsing:
   ├─ Parse flags: --session
   └─ Validation: session_id REQUIRED

Phase 1: Extract Test Tasks
   └─ Find TEST-*.json files and extract focus_paths

Phase 2: Run Test Suite
   └─ Decision (test framework):
      ├─ Node.js → npm test --coverage --json
      ├─ Python → pytest --cov --json-report
      └─ Other → [test_command] --coverage --json

Phase 3: Parse Coverage Data
   ├─ Extract line coverage percentage
   ├─ Extract branch coverage percentage
   ├─ Extract function coverage percentage
   └─ Identify uncovered lines/branches

Phase 4: Verify TDD Cycle
   └─ FOR each TDD chain (TEST-N.M → IMPL-N.M → REFACTOR-N.M):
      ├─ Red Phase: Verify tests created and failed initially
      ├─ Green Phase: Verify tests now pass
      └─ Refactor Phase: Verify code quality improved

Phase 5: Generate Analysis Report
   └─ Create tdd-cycle-report.md with coverage metrics and cycle verification
```

## Execution Lifecycle

### Phase 1: Extract Test Tasks
```bash
find .workflow/active/{session_id}/.task/ -name 'TEST-*.json' -exec jq -r '.context.focus_paths[]' {} \;
```

**Output**: List of test directories/files from all TEST tasks

### Phase 2: Run Test Suite
```bash
# Node.js/JavaScript
npm test -- --coverage --json > .workflow/active/{session_id}/.process/test-results.json

# Python
pytest --cov --json-report > .workflow/active/{session_id}/.process/test-results.json

# Other frameworks (detect from project)
[test_command] --coverage --json-output .workflow/active/{session_id}/.process/test-results.json
```

**Output**: test-results.json with coverage data

### Phase 3: Parse Coverage Data
```bash
jq '.coverage' .workflow/active/{session_id}/.process/test-results.json > .workflow/active/{session_id}/.process/coverage-report.json
```

**Extract**:
- Line coverage percentage
- Branch coverage percentage
- Function coverage percentage
- Uncovered lines/branches

### Phase 4: Verify TDD Cycle

For each TDD chain (TEST-N.M -> IMPL-N.M -> REFACTOR-N.M):

**1. Red Phase Verification**
```bash
# Check TEST task summary
cat .workflow/active/{session_id}/.summaries/TEST-N.M-summary.md
```

Verify:
- Tests were created
- Tests failed initially
- Failure messages were clear

**2. Green Phase Verification**
```bash
# Check IMPL task summary
cat .workflow/active/{session_id}/.summaries/IMPL-N.M-summary.md
```

Verify:
- Implementation was completed
- Tests now pass
- Implementation was minimal

**3. Refactor Phase Verification**
```bash
# Check REFACTOR task summary
cat .workflow/active/{session_id}/.summaries/REFACTOR-N.M-summary.md
```

Verify:
- Refactoring was completed
- Tests still pass
- Code quality improved

### Phase 5: Generate Analysis Report

Create `.workflow/active/{session_id}/.process/tdd-cycle-report.md`:

```markdown
# TDD Cycle Analysis - {Session ID}

## Coverage Metrics
- **Line Coverage**: {percentage}%
- **Branch Coverage**: {percentage}%
- **Function Coverage**: {percentage}%

## Coverage Details
### Covered
- {covered_lines} lines
- {covered_branches} branches
- {covered_functions} functions

### Uncovered
- Lines: {uncovered_line_numbers}
- Branches: {uncovered_branch_locations}

## TDD Cycle Verification

### Feature 1: {Feature Name}
**Chain**: TEST-1.1 -> IMPL-1.1 -> REFACTOR-1.1

- [PASS] **Red Phase**: Tests created and failed initially
- [PASS] **Green Phase**: Implementation made tests pass
- [PASS] **Refactor Phase**: Refactoring maintained green tests

### Feature 2: {Feature Name}
**Chain**: TEST-2.1 -> IMPL-2.1 -> REFACTOR-2.1

- [PASS] **Red Phase**: Tests created and failed initially
- [WARN] **Green Phase**: Tests pass but implementation seems over-engineered
- [PASS] **Refactor Phase**: Refactoring maintained green tests

[Repeat for all features]

## TDD Compliance Summary
- **Total Chains**: {N}
- **Complete Cycles**: {N}
- **Incomplete Cycles**: {0}
- **Compliance Score**: {score}/100

## Gaps Identified
- Feature 3: Missing initial test failure verification
- Feature 5: No refactoring step completed

## Recommendations
- Complete missing refactoring steps
- Add edge case tests for Feature 2
- Verify test failure messages are descriptive
```

## Output Files
```
.workflow/active//{session-id}/
└── .process/
    ├── test-results.json         # Raw test execution results
    ├── coverage-report.json      # Parsed coverage data
    └── tdd-cycle-report.md       # TDD cycle analysis
```

## Test Framework Detection

Auto-detect test framework from project:

```bash
# Check for test frameworks
if [ -f "package.json" ] && grep -q "jest\|mocha\|vitest" package.json; then
    TEST_CMD="npm test -- --coverage --json"
elif [ -f "pytest.ini" ] || [ -f "setup.py" ]; then
    TEST_CMD="pytest --cov --json-report"
elif [ -f "Cargo.toml" ]; then
    TEST_CMD="cargo test -- --test-threads=1 --nocapture"
elif [ -f "go.mod" ]; then
    TEST_CMD="go test -coverprofile=coverage.out -json ./..."
else
    TEST_CMD="echo 'No supported test framework found'"
fi
```

## TDD Cycle Verification Algorithm

```
For each feature N:
  1. Load TEST-N.M-summary.md
     IF summary missing:
       Mark: "Red phase incomplete"
       SKIP to next feature

     CHECK: Contains "test" AND "fail"
     IF NOT found:
       Mark: "Red phase verification failed"
     ELSE:
       Mark: "Red phase [PASS]"

  2. Load IMPL-N.M-summary.md
     IF summary missing:
       Mark: "Green phase incomplete"
       SKIP to next feature

     CHECK: Contains "pass" OR "green"
     IF NOT found:
       Mark: "Green phase verification failed"
     ELSE:
       Mark: "Green phase [PASS]"

  3. Load REFACTOR-N.M-summary.md
     IF summary missing:
       Mark: "Refactor phase incomplete"
       CONTINUE (refactor is optional)

     CHECK: Contains "refactor" AND "pass"
     IF NOT found:
       Mark: "Refactor phase verification failed"
     ELSE:
       Mark: "Refactor phase [PASS]"

  4. Calculate chain score:
     - Red + Green + Refactor all [PASS] = 100%
     - Red + Green [PASS], Refactor missing = 80%
     - Red [PASS], Green missing = 40%
     - All missing = 0%
```

## Coverage Metrics Calculation

```bash
# Parse coverage from test-results.json
line_coverage=$(jq '.coverage.lineCoverage' test-results.json)
branch_coverage=$(jq '.coverage.branchCoverage' test-results.json)
function_coverage=$(jq '.coverage.functionCoverage' test-results.json)

# Calculate overall score
overall_score=$(echo "($line_coverage + $branch_coverage + $function_coverage) / 3" | bc)
```

## Error Handling

### Test Execution Errors
| Error | Cause | Resolution |
|-------|-------|------------|
| Test framework not found | No test config | Configure test framework first |
| Tests fail to run | Syntax errors | Fix code before analysis |
| Coverage not available | Missing coverage tool | Install coverage plugin |

### Cycle Verification Errors
| Error | Cause | Resolution |
|-------|-------|------------|
| Summary missing | Task not executed | Execute tasks before analysis |
| Invalid summary format | Corrupted file | Re-run task to regenerate |
| No test evidence | Tests not committed | Ensure tests are committed |

## Integration & Usage

### Command Chain
- **Called By**: `/workflow:tdd-verify` (Phase 3)
- **Calls**: Test framework commands (npm test, pytest, etc.)
- **Followed By**: Compliance report generation

### Basic Usage
```bash
/workflow:tools:tdd-coverage-analysis --session WFS-auth
```

### Expected Output
```
TDD Coverage Analysis complete for session: WFS-auth

## Coverage Results
Line Coverage: 87%
Branch Coverage: 82%
Function Coverage: 91%

## TDD Cycle Verification
[PASS] Feature 1: Complete (Red -> Green -> Refactor)
[PASS] Feature 2: Complete (Red -> Green -> Refactor)
[WARN] Feature 3: Incomplete (Red -> Green, missing Refactor)

Overall Compliance: 93/100

Detailed report: .workflow/active/WFS-auth/.process/tdd-cycle-report.md
```

