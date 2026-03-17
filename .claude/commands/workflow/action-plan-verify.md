---
name: action-plan-verify
description: Perform non-destructive cross-artifact consistency analysis between IMPL_PLAN.md and task JSONs with quality gate validation
argument-hint: "[optional: --session session-id]"
allowed-tools: Read(*), TodoWrite(*), Glob(*), Bash(*)
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Goal

Identify inconsistencies, duplications, ambiguities, and underspecified items between action planning artifacts (`IMPL_PLAN.md`, `task.json`) and brainstorming artifacts (`role analysis documents`) before implementation. This command MUST run only after `/workflow:plan` has successfully produced complete `IMPL_PLAN.md` and task JSON files.

## Operating Constraints

**STRICTLY READ-ONLY**: Do **not** modify any files. Output a structured analysis report. Offer an optional remediation plan (user must explicitly approve before any follow-up editing commands).

**Synthesis Authority**: The `role analysis documents` is **authoritative** for requirements and design decisions. Any conflicts between IMPL_PLAN/tasks and synthesis are automatically CRITICAL and require adjustment of the plan/tasks—not reinterpretation of requirements.

## Execution Steps

### 1. Initialize Analysis Context

```bash
# Detect active workflow session
IF --session parameter provided:
    session_id = provided session
ELSE:
    # Auto-detect active session
    active_sessions = bash(find .workflow/active/ -name "WFS-*" -type d 2>/dev/null)
    IF active_sessions is empty:
        ERROR: "No active workflow session found. Use --session <session-id>"
        EXIT
    ELSE IF active_sessions has multiple entries:
        # Use most recently modified session
        session_id = bash(ls -td .workflow/active/WFS-*/ 2>/dev/null | head -1 | xargs basename)
    ELSE:
        session_id = basename(active_sessions[0])

# Derive absolute paths
session_dir = .workflow/active/WFS-{session}
brainstorm_dir = session_dir/.brainstorming
task_dir = session_dir/.task

# Validate required artifacts
# Note: "role analysis documents" refers to [role]/analysis.md files (e.g., product-manager/analysis.md)
SYNTHESIS_DIR = brainstorm_dir  # Contains role analysis files: */analysis.md
IMPL_PLAN = session_dir/IMPL_PLAN.md
TASK_FILES = Glob(task_dir/*.json)

# Abort if missing
SYNTHESIS_FILES = Glob(brainstorm_dir/*/analysis.md)
IF SYNTHESIS_FILES.count == 0:
    ERROR: "No role analysis documents found in .brainstorming/*/analysis.md. Run /workflow:brainstorm:synthesis first"
    EXIT

IF NOT EXISTS(IMPL_PLAN):
    ERROR: "IMPL_PLAN.md not found. Run /workflow:plan first"
    EXIT

IF TASK_FILES.count == 0:
    ERROR: "No task JSON files found. Run /workflow:plan first"
    EXIT
```

### 2. Load Artifacts (Progressive Disclosure)

Load only minimal necessary context from each artifact:

**From workflow-session.json** (NEW - PRIMARY REFERENCE):
- Original user prompt/intent (project or description field)
- User's stated goals and objectives
- User's scope definition

**From role analysis documents**:
- Functional Requirements (IDs, descriptions, acceptance criteria)
- Non-Functional Requirements (IDs, targets)
- Business Requirements (IDs, success metrics)
- Key Architecture Decisions
- Risk factors and mitigation strategies
- Implementation Roadmap (high-level phases)

**From IMPL_PLAN.md**:
- Summary and objectives
- Context Analysis
- Implementation Strategy
- Task Breakdown Summary
- Success Criteria
- Brainstorming Artifacts References (if present)

**From task.json files**:
- Task IDs
- Titles and descriptions
- Status
- Dependencies (depends_on, blocks)
- Context (requirements, focus_paths, acceptance, artifacts)
- Flow control (pre_analysis, implementation_approach)
- Meta (complexity, priority)

### 3. Build Semantic Models

Create internal representations (do not include raw artifacts in output):

**Requirements inventory**:
- Each functional/non-functional/business requirement with stable ID
- Requirement text, acceptance criteria, priority

**Architecture decisions inventory**:
- ADRs from synthesis
- Technology choices
- Data model references

**Task coverage mapping**:
- Map each task to one or more requirements (by ID reference or keyword inference)
- Map each requirement to covering tasks

**Dependency graph**:
- Task-to-task dependencies (depends_on, blocks)
- Requirement-level dependencies (from synthesis)

### 4. Detection Passes (Token-Efficient Analysis)

Focus on high-signal findings. Limit to 50 findings total; aggregate remainder in overflow summary.

#### A. User Intent Alignment (NEW - CRITICAL)

- **Goal Alignment**: IMPL_PLAN objectives match user's original intent
- **Scope Drift**: Plan covers user's stated scope without unauthorized expansion
- **Success Criteria Match**: Plan's success criteria reflect user's expectations
- **Intent Conflicts**: Tasks contradicting user's original objectives

#### B. Requirements Coverage Analysis

- **Orphaned Requirements**: Requirements in synthesis with zero associated tasks
- **Unmapped Tasks**: Tasks with no clear requirement linkage
- **NFR Coverage Gaps**: Non-functional requirements (performance, security, scalability) not reflected in tasks

#### C. Consistency Validation

- **Requirement Conflicts**: Tasks contradicting synthesis requirements
- **Architecture Drift**: IMPL_PLAN architecture not matching synthesis ADRs
- **Terminology Drift**: Same concept named differently across IMPL_PLAN and tasks
- **Data Model Inconsistency**: Tasks referencing entities/fields not in synthesis data model

#### D. Dependency Integrity

- **Circular Dependencies**: Task A depends on B, B depends on C, C depends on A
- **Missing Dependencies**: Task requires outputs from another task but no explicit dependency
- **Broken Dependencies**: Task depends on non-existent task ID
- **Logical Ordering Issues**: Implementation tasks before foundational setup without dependency note

#### E. Synthesis Alignment

- **Priority Conflicts**: High-priority synthesis requirements mapped to low-priority tasks
- **Success Criteria Mismatch**: IMPL_PLAN success criteria not covering synthesis acceptance criteria
- **Risk Mitigation Gaps**: Critical risks in synthesis without corresponding mitigation tasks

#### F. Task Specification Quality

- **Ambiguous Focus Paths**: Tasks with vague or missing focus_paths
- **Underspecified Acceptance**: Tasks without clear acceptance criteria
- **Missing Artifacts References**: Tasks not referencing relevant brainstorming artifacts in context.artifacts
- **Weak Flow Control**: Tasks without clear implementation_approach or pre_analysis steps
- **Missing Target Files**: Tasks without flow_control.target_files specification

#### G. Duplication Detection

- **Overlapping Task Scope**: Multiple tasks with nearly identical descriptions
- **Redundant Requirements Coverage**: Same requirement covered by multiple tasks without clear partitioning

#### H. Feasibility Assessment

- **Complexity Misalignment**: Task marked "simple" but requires multiple file modifications
- **Resource Conflicts**: Parallel tasks requiring same resources/files
- **Skill Gap Risks**: Tasks requiring skills not in team capability assessment (from synthesis)

### 5. Severity Assignment

Use this heuristic to prioritize findings:

- **CRITICAL**:
  - Violates user's original intent (goal misalignment, scope drift)
  - Violates synthesis authority (requirement conflict)
  - Core requirement with zero coverage
  - Circular dependencies
  - Broken dependencies

- **HIGH**:
  - NFR coverage gaps
  - Priority conflicts
  - Missing risk mitigation tasks
  - Ambiguous acceptance criteria

- **MEDIUM**:
  - Terminology drift
  - Missing artifacts references
  - Weak flow control
  - Logical ordering issues

- **LOW**:
  - Style/wording improvements
  - Minor redundancy not affecting execution

### 6. Produce Compact Analysis Report

**Report Generation**: Generate report content and save to file.

Output a Markdown report with the following structure:

```markdown
## Action Plan Verification Report

**Session**: WFS-{session-id}
**Generated**: {timestamp}
**Artifacts Analyzed**: role analysis documents, IMPL_PLAN.md, {N} task files

---

### Executive Summary

- **Overall Risk Level**: CRITICAL | HIGH | MEDIUM | LOW
- **Recommendation**: (See decision matrix below)
  - BLOCK_EXECUTION: Critical issues exist (must fix before proceeding)
  - PROCEED_WITH_FIXES: High issues exist, no critical (fix recommended before execution)
  - PROCEED_WITH_CAUTION: Medium issues only (proceed with awareness)
  - PROCEED: Low issues only or no issues (safe to execute)
- **Critical Issues**: {count}
- **High Issues**: {count}
- **Medium Issues**: {count}
- **Low Issues**: {count}

---

### Findings Summary

| ID | Category | Severity | Location(s) | Summary | Recommendation |
|----|----------|----------|-------------|---------|----------------|
| C1 | Coverage | CRITICAL | synthesis:FR-03 | Requirement "User auth" has zero task coverage | Add authentication implementation task |
| H1 | Consistency | HIGH | IMPL-1.2 vs synthesis:ADR-02 | Task uses REST while synthesis specifies GraphQL | Align task with ADR-02 decision |
| M1 | Specification | MEDIUM | IMPL-2.1 | Missing context.artifacts reference | Add @synthesis reference |
| L1 | Duplication | LOW | IMPL-3.1, IMPL-3.2 | Similar scope | Consider merging |

(Add one row per finding; generate stable IDs prefixed by severity initial.)

---

### Requirements Coverage Analysis

| Requirement ID | Requirement Summary | Has Task? | Task IDs | Priority Match | Notes |
|----------------|---------------------|-----------|----------|----------------|-------|
| FR-01 | User authentication | Yes | IMPL-1.1, IMPL-1.2 | Match | Complete |
| FR-02 | Data export | Yes | IMPL-2.3 | Mismatch | High req → Med priority task |
| FR-03 | Profile management | No | - | - | **CRITICAL: Zero coverage** |
| NFR-01 | Response time <200ms | No | - | - | **HIGH: No performance tasks** |

**Coverage Metrics**:
- Functional Requirements: 85% (17/20 covered)
- Non-Functional Requirements: 40% (2/5 covered)
- Business Requirements: 100% (5/5 covered)

---

### Unmapped Tasks

| Task ID | Title | Issue | Recommendation |
|---------|-------|-------|----------------|
| IMPL-4.5 | Refactor utils | No requirement linkage | Link to technical debt or remove |

---

### Dependency Graph Issues

**Circular Dependencies**: None detected

**Broken Dependencies**:
- IMPL-2.3 depends on "IMPL-2.4" (non-existent)

**Logical Ordering Issues**:
- IMPL-5.1 (integration test) has no dependency on IMPL-1.* (implementation tasks)

---

### Synthesis Alignment Issues

| Issue Type | Synthesis Reference | IMPL_PLAN/Task | Impact | Recommendation |
|------------|---------------------|----------------|--------|----------------|
| Architecture Conflict | synthesis:ADR-01 (JWT auth) | IMPL_PLAN uses session cookies | HIGH | Update IMPL_PLAN to use JWT |
| Priority Mismatch | synthesis:FR-02 (High) | IMPL-2.3 (Medium) | MEDIUM | Elevate task priority |
| Missing Risk Mitigation | synthesis:Risk-03 (API rate limits) | No mitigation tasks | HIGH | Add rate limiting implementation task |

---

### Task Specification Quality Issues

**Missing Artifacts References**: 12 tasks lack context.artifacts
**Weak Flow Control**: 5 tasks lack implementation_approach
**Missing Target Files**: 8 tasks lack flow_control.target_files

**Sample Issues**:
- IMPL-1.2: No context.artifacts reference to synthesis
- IMPL-3.1: Missing flow_control.target_files specification
- IMPL-4.2: Vague focus_paths ["src/"] - needs refinement

---

### Feasibility Concerns

| Concern | Tasks Affected | Issue | Recommendation |
|---------|----------------|-------|----------------|
| Skill Gap | IMPL-6.1, IMPL-6.2 | Requires Kubernetes expertise not in team | Add training task or external consultant |
| Resource Conflict | IMPL-3.1, IMPL-3.2 | Both modify src/auth/service.ts in parallel | Add dependency or serialize |

---

### Metrics

- **Total Requirements**: 30 (20 functional, 5 non-functional, 5 business)
- **Total Tasks**: 25
- **Overall Coverage**: 77% (23/30 requirements with ≥1 task)
- **Critical Issues**: 2
- **High Issues**: 5
- **Medium Issues**: 8
- **Low Issues**: 3

---

### Next Actions

#### Action Recommendations

**Recommendation Decision Matrix**:

| Condition | Recommendation | Action |
|-----------|----------------|--------|
| Critical > 0 | BLOCK_EXECUTION | Must resolve all critical issues before proceeding |
| Critical = 0, High > 0 | PROCEED_WITH_FIXES | Fix high-priority issues before execution |
| Critical = 0, High = 0, Medium > 0 | PROCEED_WITH_CAUTION | Proceed with awareness of medium issues |
| Only Low or None | PROCEED | Safe to execute workflow |

**If CRITICAL Issues Exist** (BLOCK_EXECUTION):
- Resolve all critical issues before proceeding
- Use TodoWrite to track required fixes
- Fix broken dependencies and circular references first

**If HIGH Issues Exist** (PROCEED_WITH_FIXES):
- Fix high-priority issues before execution
- Use TodoWrite to systematically track and complete improvements

**If Only MEDIUM/LOW Issues** (PROCEED_WITH_CAUTION / PROCEED):
- Can proceed with execution
- Address issues during or after implementation

#### TodoWrite-Based Remediation Workflow

**Report Location**: `.workflow/active/WFS-{session}/.process/ACTION_PLAN_VERIFICATION.md`

**Recommended Workflow**:
1. **Create TodoWrite Task List**: Extract all findings from report
2. **Process by Priority**: CRITICAL → HIGH → MEDIUM → LOW
3. **Complete Each Fix**: Mark tasks as in_progress/completed as you work
4. **Validate Changes**: Verify each modification against requirements

**TodoWrite Task Structure Example**:
```markdown
Priority Order:
1. Fix coverage gaps (CRITICAL)
2. Resolve consistency conflicts (CRITICAL/HIGH)
3. Add missing specifications (MEDIUM)
4. Improve task quality (LOW)
```

**Notes**:
- TodoWrite provides real-time progress tracking
- Each finding becomes a trackable todo item
- User can monitor progress throughout remediation
- Architecture drift in IMPL_PLAN requires manual editing
```

### 7. Save Report and Execute TodoWrite-Based Remediation

**Step 7.1: Save Analysis Report**:
```bash
report_path = ".workflow/active/WFS-{session}/.process/ACTION_PLAN_VERIFICATION.md"
Write(report_path, full_report_content)
```

**Step 7.2: Display Report Summary to User**:
- Show executive summary with counts
- Display recommendation (BLOCK/PROCEED_WITH_FIXES/PROCEED_WITH_CAUTION/PROCEED)
- List critical and high issues if any

**Step 7.3: After Report Generation**:

1. **Extract Findings**: Parse all issues by severity
2. **Create TodoWrite Task List**: Convert findings to actionable todos
3. **Execute Fixes**: Process each todo systematically
4. **Update Task Files**: Apply modifications directly to task JSON files
5. **Update IMPL_PLAN**: Apply strategic changes if needed

At end of report, provide remediation guidance:

```markdown
### 🔧 Remediation Workflow

**Recommended Approach**:
1. **Initialize TodoWrite**: Create comprehensive task list from all findings
2. **Process by Severity**: Start with CRITICAL, then HIGH, MEDIUM, LOW
3. **Apply Fixes Directly**: Modify task.json files and IMPL_PLAN.md as needed
4. **Track Progress**: Mark todos as completed after each fix

**TodoWrite Execution Pattern**:
```bash
# Step 1: Create task list from verification report
TodoWrite([
  { content: "Fix FR-03 coverage gap - add authentication task", status: "pending", activeForm: "Fixing FR-03 coverage gap" },
  { content: "Fix IMPL-1.2 consistency - align with ADR-02", status: "pending", activeForm: "Fixing IMPL-1.2 consistency" },
  { content: "Add context.artifacts to IMPL-1.2", status: "pending", activeForm: "Adding context.artifacts to IMPL-1.2" },
  # ... additional todos for each finding
])

# Step 2: Process each todo systematically
# Mark as in_progress when starting
# Apply fix using Read/Edit tools
# Mark as completed when done
# Move to next priority item
```

**File Modification Workflow**:
```bash
# For task JSON modifications:
1. Read(.workflow/active/WFS-{session}/.task/IMPL-X.Y.json)
2. Edit() to apply fixes
3. Mark todo as completed

# For IMPL_PLAN modifications:
1. Read(.workflow/active/WFS-{session}/IMPL_PLAN.md)
2. Edit() to apply strategic changes
3. Mark todo as completed
```

**Note**: All fixes execute immediately after user confirmation without additional commands.
