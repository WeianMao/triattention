---
name: review-session-cycle
description: Session-based comprehensive multi-dimensional code review. Analyzes git changes from workflow session across 7 dimensions with hybrid parallel-iterative execution, aggregates findings, and performs focused deep-dives on critical issues until quality gates met.
argument-hint: "[session-id] [--dimensions=security,architecture,...] [--max-iterations=N]"
allowed-tools: SlashCommand(*), TodoWrite(*), Read(*), Bash(*), Task(*)
---

# Workflow Review-Session-Cycle Command

## Quick Start

```bash
# Execute comprehensive session review (all 7 dimensions)
/workflow:review-session-cycle

# Review specific session with custom dimensions
/workflow:review-session-cycle WFS-payment-integration --dimensions=security,architecture,quality

# Specify session and iteration limit
/workflow:review-session-cycle WFS-payment-integration --max-iterations=5
```

**Review Scope**: Git changes from session creation to present (via `git log --since`)
**Session Requirement**: Requires active or completed workflow session
**Output Directory**: `.workflow/active/WFS-{session-id}/.review/` (session-based)
**Default Dimensions**: Security, Architecture, Quality, Action-Items, Performance, Maintainability, Best-Practices
**Max Iterations**: 3 (adjustable via --max-iterations)
**Default Iterations**: 1 (deep-dive runs once; use --max-iterations=0 to skip)
**CLI Tools**: Gemini → Qwen → Codex (fallback chain)

## What & Why

### Core Concept
Session-based multi-dimensional code review orchestrator with **hybrid parallel-iterative execution** for comprehensive quality assessment of **git changes within a workflow session**.

**Review Scope**:
- **Session-based**: Reviews only files changed during the workflow session (via `git log --since="${sessionCreatedAt}"`)
- **For independent module review**: Use `/workflow:review-module-cycle` command instead

**vs Standard Review**:
- **Standard**: Sequential manual reviews → Inconsistent coverage → Missed cross-cutting concerns
- **Review-Session-Cycle**: **Parallel automated analysis → Aggregate findings → Deep-dive critical issues** → Comprehensive coverage

### Value Proposition
1. **Comprehensive Coverage**: 7 specialized dimensions analyze all quality aspects simultaneously
2. **Intelligent Prioritization**: Automatic identification of critical issues and cross-cutting concerns
3. **Actionable Insights**: Deep-dive iterations provide step-by-step remediation plans

### Orchestrator Boundary (CRITICAL)
- **ONLY command** for comprehensive multi-dimensional review
- Manages: dimension coordination, aggregation, iteration control, progress tracking
- Delegates: Code exploration and analysis to @cli-explore-agent, dimension-specific reviews via Deep Scan mode

## How It Works

### Execution Flow (Simplified)

```
Phase 1: Discovery & Initialization
   └─ Validate session, initialize state, create output structure

Phase 2: Parallel Reviews (for each dimension)
   ├─ Launch 7 review agents simultaneously
   ├─ Each executes CLI analysis via Gemini/Qwen
   ├─ Generate dimension JSON + markdown reports
   └─ Update review-progress.json

Phase 3: Aggregation
   ├─ Load all dimension JSON files
   ├─ Calculate severity distribution (critical/high/medium/low)
   ├─ Identify cross-cutting concerns (files in 3+ dimensions)
   └─ Decision:
      ├─ Critical findings OR high > 5 OR critical files → Phase 4 (Iterate)
      └─ Else → Phase 5 (Complete)

Phase 4: Iterative Deep-Dive (optional)
   ├─ Select critical findings (max 5 per iteration)
   ├─ Launch deep-dive agents for root cause analysis
   ├─ Generate remediation plans with impact assessment
   ├─ Re-assess severity based on analysis
   └─ Loop until no critical findings OR max iterations

Phase 5: Completion
   └─ Finalize review-progress.json
```

### Agent Roles

| Agent | Responsibility |
|-------|---------------|
| **Orchestrator** | Phase control, session discovery, state management, aggregation logic, iteration control |
| **@cli-explore-agent** (Review) | Execute dimension-specific code analysis via Deep Scan mode, generate findings JSON with dual-source strategy (Bash + Gemini), create structured analysis reports |
| **@cli-explore-agent** (Deep-dive) | Focused root cause analysis using dependency mapping, remediation planning with architectural insights, impact assessment, severity re-assessment |

## Enhanced Features

### 1. Review Dimensions Configuration

**7 Specialized Dimensions** with priority-based allocation:

| Dimension | Template | Priority | Timeout |
|-----------|----------|----------|---------|
| **Security** | 03-assess-security-risks.txt | 1 (Critical) | 60min |
| **Architecture** | 02-review-architecture.txt | 2 (High) | 60min |
| **Quality** | 02-review-code-quality.txt | 3 (Medium) | 40min |
| **Action-Items** | 02-analyze-code-patterns.txt | 2 (High) | 40min |
| **Performance** | 03-analyze-performance.txt | 3 (Medium) | 60min |
| **Maintainability** | 02-review-code-quality.txt* | 3 (Medium) | 40min |
| **Best-Practices** | 03-review-quality-standards.txt | 3 (Medium) | 40min |

*Custom focus: "Assess technical debt and maintainability"

**Category Definitions by Dimension**:

```javascript
const CATEGORIES = {
  security: ['injection', 'authentication', 'authorization', 'encryption', 'input-validation', 'access-control', 'data-exposure'],
  architecture: ['coupling', 'cohesion', 'layering', 'dependency', 'pattern-violation', 'scalability', 'separation-of-concerns'],
  quality: ['code-smell', 'duplication', 'complexity', 'naming', 'error-handling', 'testability', 'readability'],
  'action-items': ['requirement-coverage', 'acceptance-criteria', 'documentation', 'deployment-readiness', 'missing-functionality'],
  performance: ['n-plus-one', 'inefficient-query', 'memory-leak', 'blocking-operation', 'caching', 'resource-usage'],
  maintainability: ['technical-debt', 'magic-number', 'long-method', 'large-class', 'dead-code', 'commented-code'],
  'best-practices': ['convention-violation', 'anti-pattern', 'deprecated-api', 'missing-validation', 'inconsistent-style']
};
```

### 2. Aggregation Logic

**Cross-Cutting Concern Detection**:
1. Files appearing in 3+ dimensions = **Critical Files**
2. Same issue pattern across dimensions = **Systemic Issue**
3. Severity clustering in specific files = **Hotspots**

**Deep-Dive Selection Criteria**:
- All critical severity findings (priority 1)
- Top 3 high-severity findings in critical files (priority 2)
- Max 5 findings per iteration (prevent overwhelm)

### 3. Severity Assessment

**Severity Levels**:
- **Critical**: Security vulnerabilities, data corruption risks, system-wide failures, authentication/authorization bypass
- **High**: Feature degradation, performance bottlenecks, architecture violations, significant technical debt
- **Medium**: Code smells, minor performance issues, style inconsistencies, maintainability concerns
- **Low**: Documentation gaps, minor refactoring opportunities, cosmetic issues

**Iteration Trigger**:
- Critical findings > 0 OR
- High findings > 5 OR
- Critical files count > 0

## Core Responsibilities

### Orchestrator

**Phase 1: Discovery & Initialization**

**Step 1: Session Discovery**
```javascript
// If session ID not provided, auto-detect
if (!providedSessionId) {
  // Check for active sessions
  const activeSessions = Glob('.workflow/active/WFS-*');
  if (activeSessions.length === 1) {
    sessionId = activeSessions[0].match(/WFS-[^/]+/)[0];
  } else if (activeSessions.length > 1) {
    // List sessions and prompt user
    error("Multiple active sessions found. Please specify session ID.");
  } else {
    error("No active session found. Create session first with /workflow:session:start");
  }
} else {
  sessionId = providedSessionId;
}

// Validate session exists
Bash(`test -d .workflow/active/${sessionId} && echo "EXISTS"`);
```

**Step 2: Session Validation**
- Ensure session has implementation artifacts (check `.summaries/` or `.task/` directory)
- Extract session creation timestamp from `workflow-session.json`
- Use timestamp for git log filtering: `git log --since="${sessionCreatedAt}"`

**Step 3: Changed Files Detection**
```bash
# Get files changed since session creation
git log --since="${sessionCreatedAt}" --name-only --pretty=format: | sort -u
```

**Step 4: Output Directory Setup**
- Output directory: `.workflow/active/${sessionId}/.review/`
- Create directory structure:
  ```bash
  mkdir -p ${sessionDir}/.review/{dimensions,iterations,reports}
  ```

**Step 5: Initialize Review State**
- State initialization: Create `review-state.json` with metadata, dimensions, max_iterations (merged metadata + state)
- Progress tracking: Create `review-progress.json` for progress tracking

**Step 6: TodoWrite Initialization**
- Set up progress tracking with hierarchical structure
- Mark Phase 1 completed, Phase 2 in_progress

**Phase 2: Parallel Review Coordination**
- Launch 7 @cli-explore-agent instances simultaneously (Deep Scan mode)
- Pass dimension-specific context (template, timeout, custom focus)
- Monitor completion via review-progress.json updates
- TodoWrite updates: Mark dimensions as completed
- CLI tool fallback: Gemini → Qwen → Codex (on error/timeout)

**Phase 3: Aggregation**
- Load all dimension JSON files from dimensions/
- Calculate severity distribution: Count by critical/high/medium/low
- Identify cross-cutting concerns: Files in 3+ dimensions
- Select deep-dive findings: Critical + high in critical files (max 5)
- Decision logic: Iterate if critical > 0 OR high > 5 OR critical files exist
- Update review-state.json with aggregation results

**Phase 4: Iteration Control**
- Check iteration count < max_iterations (default 3)
- Launch deep-dive agents for selected findings
- Collect remediation plans and re-assessed severities
- Update severity distribution based on re-assessments
- Record iteration in review-state.json
- Loop back to aggregation if still have critical/high findings

**Phase 5: Completion**
- Finalize review-progress.json with completion statistics
- Update review-state.json with completion_time and phase=complete
- TodoWrite completion: Mark all tasks done



### Session File Structure

```
.workflow/active/WFS-{session-id}/.review/
├── review-state.json                    # Orchestrator state machine (includes metadata)
├── review-progress.json                 # Real-time progress for dashboard
├── dimensions/                          # Per-dimension results
│   ├── security.json
│   ├── architecture.json
│   ├── quality.json
│   ├── action-items.json
│   ├── performance.json
│   ├── maintainability.json
│   └── best-practices.json
├── iterations/                          # Deep-dive results
│   ├── iteration-1-finding-{uuid}.json
│   └── iteration-2-finding-{uuid}.json
└── reports/                             # Human-readable reports
    ├── security-analysis.md
    ├── security-cli-output.txt
    ├── deep-dive-1-{uuid}.md
    └── ...
```

**Session Context**:
```
.workflow/active/WFS-{session-id}/
├── workflow-session.json
├── IMPL_PLAN.md
├── TODO_LIST.md
├── .task/
├── .summaries/
└── .review/                             # Review results (this command)
    └── (structure above)
```

### Review State JSON

**Purpose**: Unified state machine and metadata (merged from metadata + state)

```json
{
  "session_id": "WFS-payment-integration",
  "review_id": "review-20250125-143022",
  "review_type": "session",
  "metadata": {
    "created_at": "2025-01-25T14:30:22Z",
    "git_changes": {
      "commit_range": "abc123..def456",
      "files_changed": 15,
      "insertions": 342,
      "deletions": 128
    },
    "dimensions": ["security", "architecture", "quality", "action-items", "performance", "maintainability", "best-practices"],
    "max_iterations": 3
  },
  "phase": "parallel|aggregate|iterate|complete",
  "current_iteration": 1,
  "dimensions_reviewed": ["security", "architecture", "quality", "action-items", "performance", "maintainability", "best-practices"],
  "selected_strategy": "comprehensive",
  "next_action": "execute_parallel_reviews|aggregate_findings|execute_deep_dive|generate_final_report|complete",
  "severity_distribution": {
    "critical": 2,
    "high": 5,
    "medium": 12,
    "low": 8
  },
  "critical_files": [
    {
      "file": "src/payment/processor.ts",
      "finding_count": 5,
      "dimensions": ["security", "architecture", "quality"]
    }
  ],
  "iterations": [
    {
      "iteration": 1,
      "findings_analyzed": ["uuid-1", "uuid-2"],
      "findings_resolved": 1,
      "findings_escalated": 1,
      "severity_change": {
        "before": {"critical": 2, "high": 5, "medium": 12, "low": 8},
        "after": {"critical": 1, "high": 6, "medium": 12, "low": 8}
      },
      "timestamp": "2025-01-25T14:30:00Z"
    }
  ],
  "completion_criteria": {
    "target": "no_critical_findings_and_high_under_5",
    "current_status": "in_progress",
    "estimated_completion": "2 iterations remaining"
  }
}
```

**Field Descriptions**:
- `phase`: Current execution phase (state machine pointer)
- `current_iteration`: Iteration counter (used for max check)
- `next_action`: Next step orchestrator should execute
- `severity_distribution`: Aggregated counts across all dimensions
- `critical_files`: Files appearing in 3+ dimensions with metadata
- `iterations[]`: Historical log for trend analysis

### Review Progress JSON

**Purpose**: Real-time dashboard updates via polling

```json
{
  "review_id": "review-20250125-143022",
  "last_update": "2025-01-25T14:35:10Z",
  "phase": "parallel|aggregate|iterate|complete",
  "current_iteration": 1,
  "progress": {
    "parallel_review": {
      "total_dimensions": 7,
      "completed": 5,
      "in_progress": 2,
      "percent_complete": 71
    },
    "deep_dive": {
      "total_findings": 6,
      "analyzed": 2,
      "in_progress": 1,
      "percent_complete": 33
    }
  },
  "agent_status": [
    {
      "agent_type": "review-agent",
      "dimension": "security",
      "status": "completed",
      "started_at": "2025-01-25T14:30:00Z",
      "completed_at": "2025-01-25T15:15:00Z",
      "duration_ms": 2700000
    },
    {
      "agent_type": "deep-dive-agent",
      "finding_id": "sec-001-uuid",
      "status": "in_progress",
      "started_at": "2025-01-25T14:32:00Z"
    }
  ],
  "estimated_completion": "2025-01-25T16:00:00Z"
}
```

### Agent Output Schemas

**Agent-produced JSON files follow standardized schemas**:

1. **Dimension Results** (cli-explore-agent output from parallel reviews)
   - Schema: `~/.claude/workflows/cli-templates/schemas/review-dimension-results-schema.json`
   - Output: `.review-cycle/dimensions/{dimension}.json`
   - Contains: findings array, summary statistics, cross_references

2. **Deep-Dive Results** (cli-explore-agent output from iterations)
   - Schema: `~/.claude/workflows/cli-templates/schemas/review-deep-dive-results-schema.json`
   - Output: `.review-cycle/iterations/iteration-{N}-finding-{uuid}.json`
   - Contains: root_cause, remediation_plan, impact_assessment, reassessed_severity

### Agent Invocation Template

**Review Agent** (parallel execution, 7 instances):

```javascript
Task(
  subagent_type="cli-explore-agent",
  description=`Execute ${dimension} review analysis via Deep Scan`,
  prompt=`
    ## Task Objective
    Conduct comprehensive ${dimension} code exploration and analysis using Deep Scan mode (Bash + Gemini dual-source strategy) for completed implementation in session ${sessionId}

    ## Analysis Mode Selection
    Use **Deep Scan mode** for this review:
    - Phase 1: Bash structural scan for standard patterns (classes, functions, imports)
    - Phase 2: Gemini semantic analysis for design intent, non-standard patterns, ${dimension}-specific concerns
    - Phase 3: Synthesis with attribution (bash-discovered vs gemini-discovered findings)

    ## MANDATORY FIRST STEPS (Execute by Agent)
    **You (cli-explore-agent) MUST execute these steps in order:**
    1. Read session metadata: ${sessionMetadataPath}
    2. Read completed task summaries: bash(find ${summariesDir} -name "IMPL-*.md" -type f)
    3. Get changed files: bash(cd ${workflowDir} && git log --since="${sessionCreatedAt}" --name-only --pretty=format: | sort -u)
    4. Read review state: ${reviewStateJsonPath}
    5. Execute: cat ~/.claude/workflows/cli-templates/schemas/review-dimension-results-schema.json (get output schema reference)

    ## Session Context
    - Session ID: ${sessionId}
    - Review Dimension: ${dimension}
    - Review ID: ${reviewId}
    - Implementation Phase: Complete (all tests passing)
    - Output Directory: ${outputDir}

    ## CLI Configuration
    - Tool Priority: gemini → qwen → codex (fallback chain)
    - Template: ~/.claude/workflows/cli-templates/prompts/analysis/${dimensionTemplate}
    - Custom Focus: ${customFocus || 'Standard dimension analysis'}
    - Timeout: ${timeout}ms
    - Mode: analysis (READ-ONLY)

    ## Expected Deliverables

    **Schema Reference**: Schema obtained in MANDATORY FIRST STEPS step 5, follow schema exactly

    1. Dimension Results JSON: ${outputDir}/dimensions/${dimension}.json

       **⚠️ CRITICAL JSON STRUCTURE REQUIREMENTS**:

       Root structure MUST be array: \`[{ ... }]\` NOT \`{ ... }\`

       Required top-level fields:
       - dimension, review_id, analysis_timestamp (NOT timestamp/analyzed_at)
       - cli_tool_used (gemini|qwen|codex), model, analysis_duration_ms
       - summary (FLAT structure), findings, cross_references

       Summary MUST be FLAT (NOT nested by_severity):
       \`{ "total_findings": N, "critical": N, "high": N, "medium": N, "low": N, "files_analyzed": N, "lines_reviewed": N }\`

       Finding required fields:
       - id: format \`{dim}-{seq}-{uuid8}\` e.g., \`sec-001-a1b2c3d4\` (lowercase)
       - severity: lowercase only (critical|high|medium|low)
       - snippet (NOT code_snippet), impact (NOT exploit_scenario)
       - metadata, iteration (0), status (pending_remediation), cross_references

    2. Analysis Report: ${outputDir}/reports/${dimension}-analysis.md
       - Human-readable summary with recommendations
       - Grouped by severity: critical → high → medium → low
       - Include file:line references for all findings

    3. CLI Output Log: ${outputDir}/reports/${dimension}-cli-output.txt
       - Raw CLI tool output for debugging
       - Include full analysis text

    ## Dimension-Specific Guidance
    ${getDimensionGuidance(dimension)}

    ## Success Criteria
    - [ ] Schema obtained via cat review-dimension-results-schema.json
    - [ ] All changed files analyzed for ${dimension} concerns
    - [ ] All findings include file:line references with code snippets
    - [ ] Severity assessment follows established criteria (see reference)
    - [ ] Recommendations are actionable with code examples
    - [ ] JSON output follows schema exactly
    - [ ] Report is comprehensive and well-organized
  `
)
```

**Deep-Dive Agent** (iteration execution):

```javascript
Task(
  subagent_type="cli-explore-agent",
  description=`Deep-dive analysis for critical finding: ${findingTitle} via Dependency Map + Deep Scan`,
  prompt=`
    ## Task Objective
    Perform focused root cause analysis using Dependency Map mode (for impact analysis) + Deep Scan mode (for semantic understanding) to generate comprehensive remediation plan for critical ${dimension} issue

    ## Analysis Mode Selection
    Use **Dependency Map mode** first to understand dependencies:
    - Build dependency graph around ${file} to identify affected components
    - Detect circular dependencies or tight coupling related to this finding
    - Calculate change risk scores for remediation impact

    Then apply **Deep Scan mode** for semantic analysis:
    - Understand design intent and architectural context
    - Identify non-standard patterns or implicit dependencies
    - Extract remediation insights from code structure

    ## Finding Context
    - Finding ID: ${findingId}
    - Original Dimension: ${dimension}
    - Title: ${findingTitle}
    - File: ${file}:${line}
    - Severity: ${severity}
    - Category: ${category}
    - Original Description: ${description}
    - Iteration: ${iteration}

    ## MANDATORY FIRST STEPS (Execute by Agent)
    **You (cli-explore-agent) MUST execute these steps in order:**
    1. Read original finding: ${dimensionJsonPath}
    2. Read affected file: ${file}
    3. Identify related code: bash(grep -r "import.*${basename(file)}" ${workflowDir}/src --include="*.ts")
    4. Read test files: bash(find ${workflowDir}/tests -name "*${basename(file, '.ts')}*" -type f)
    5. Execute: cat ~/.claude/workflows/cli-templates/schemas/review-deep-dive-results-schema.json (get output schema reference)

    ## CLI Configuration
    - Tool Priority: gemini → qwen → codex
    - Template: ~/.claude/workflows/cli-templates/prompts/analysis/01-diagnose-bug-root-cause.txt
    - Timeout: 2400000ms (40 minutes)
    - Mode: analysis (READ-ONLY)

    ## Expected Deliverables

    **Schema Reference**: Schema obtained in MANDATORY FIRST STEPS step 5, follow schema exactly

    1. Deep-Dive Results JSON: ${outputDir}/iterations/iteration-${iteration}-finding-${findingId}.json

       **⚠️ CRITICAL JSON STRUCTURE REQUIREMENTS**:

       Root structure MUST be array: \`[{ ... }]\` NOT \`{ ... }\`

       Required top-level fields:
       - finding_id, dimension, iteration, analysis_timestamp
       - cli_tool_used, model, analysis_duration_ms
       - original_finding, root_cause, remediation_plan
       - impact_assessment, reassessed_severity, confidence_score, cross_references

       All nested objects must follow schema exactly - read schema for field names

    2. Analysis Report: ${outputDir}/reports/deep-dive-${iteration}-${findingId}.md
       - Detailed root cause analysis
       - Step-by-step remediation plan
       - Impact assessment and rollback strategy

    ## Success Criteria
    - [ ] Schema obtained via cat review-deep-dive-results-schema.json
    - [ ] Root cause clearly identified with supporting evidence
    - [ ] Remediation plan is step-by-step actionable with exact file:line references
    - [ ] Each step includes specific commands and validation tests
    - [ ] Impact fully assessed (files, tests, breaking changes, dependencies)
    - [ ] Severity re-evaluation justified with evidence
    - [ ] Confidence score accurately reflects certainty of analysis
    - [ ] JSON output follows schema exactly
    - [ ] References include project-specific and external documentation
  `
)
```

### Dimension Guidance Reference

```javascript
function getDimensionGuidance(dimension) {
  const guidance = {
    security: `
      Focus Areas:
      - Input validation and sanitization
      - Authentication and authorization mechanisms
      - Data encryption (at-rest and in-transit)
      - SQL/NoSQL injection vulnerabilities
      - XSS, CSRF, and other web vulnerabilities
      - Sensitive data exposure
      - Access control and privilege escalation

      Severity Criteria:
      - Critical: Authentication bypass, SQL injection, RCE, sensitive data exposure
      - High: Missing authorization checks, weak encryption, exposed secrets
      - Medium: Missing input validation, insecure defaults, weak password policies
      - Low: Security headers missing, verbose error messages, outdated dependencies
    `,
    architecture: `
      Focus Areas:
      - Layering and separation of concerns
      - Coupling and cohesion
      - Design pattern adherence
      - Dependency management
      - Scalability and extensibility
      - Module boundaries
      - API design consistency

      Severity Criteria:
      - Critical: Circular dependencies, god objects, tight coupling across layers
      - High: Violated architectural principles, scalability bottlenecks
      - Medium: Missing abstractions, inconsistent patterns, suboptimal design
      - Low: Minor coupling issues, documentation gaps, naming inconsistencies
    `,
    quality: `
      Focus Areas:
      - Code duplication
      - Complexity (cyclomatic, cognitive)
      - Naming conventions
      - Error handling patterns
      - Code readability
      - Comment quality
      - Dead code

      Severity Criteria:
      - Critical: Severe complexity (CC > 20), massive duplication (>50 lines)
      - High: High complexity (CC > 10), significant duplication, poor error handling
      - Medium: Moderate complexity (CC > 5), naming issues, code smells
      - Low: Minor duplication, documentation gaps, cosmetic issues
    `,
    'action-items': `
      Focus Areas:
      - Requirements coverage verification
      - Acceptance criteria met
      - Documentation completeness
      - Deployment readiness
      - Missing functionality
      - Test coverage gaps
      - Configuration management

      Severity Criteria:
      - Critical: Core requirements not met, deployment blockers
      - High: Significant functionality missing, acceptance criteria not met
      - Medium: Minor requirements gaps, documentation incomplete
      - Low: Nice-to-have features missing, minor documentation gaps
    `,
    performance: `
      Focus Areas:
      - N+1 query problems
      - Inefficient algorithms (O(n²) where O(n log n) possible)
      - Memory leaks
      - Blocking operations on main thread
      - Missing caching opportunities
      - Resource usage (CPU, memory, network)
      - Database query optimization

      Severity Criteria:
      - Critical: Memory leaks, O(n²) in hot path, blocking main thread
      - High: N+1 queries, missing indexes, inefficient algorithms
      - Medium: Suboptimal caching, unnecessary computations, lazy loading issues
      - Low: Minor optimization opportunities, redundant operations
    `,
    maintainability: `
      Focus Areas:
      - Technical debt indicators
      - Magic numbers and hardcoded values
      - Long methods (>50 lines)
      - Large classes (>500 lines)
      - Dead code and commented code
      - Code documentation
      - Test coverage

      Severity Criteria:
      - Critical: Massive methods (>200 lines), severe technical debt blocking changes
      - High: Large methods (>100 lines), significant dead code, undocumented complex logic
      - Medium: Magic numbers, moderate technical debt, missing tests
      - Low: Minor refactoring opportunities, cosmetic improvements
    `,
    'best-practices': `
      Focus Areas:
      - Framework conventions adherence
      - Language idioms
      - Anti-patterns
      - Deprecated API usage
      - Coding standards compliance
      - Error handling patterns
      - Logging and monitoring

      Severity Criteria:
      - Critical: Severe anti-patterns, deprecated APIs with security risks
      - High: Major convention violations, poor error handling, missing logging
      - Medium: Minor anti-patterns, style inconsistencies, suboptimal patterns
      - Low: Cosmetic style issues, minor convention deviations
    `
  };

  return guidance[dimension] || 'Standard code review analysis';
}
```

### Completion Conditions

**Full Success**:
- All dimensions reviewed
- Critical findings = 0
- High findings ≤ 5
- Action: Generate final report, mark phase=complete

**Partial Success**:
- All dimensions reviewed
- Max iterations reached
- Still have critical/high findings
- Action: Generate report with warnings, recommend follow-up

### Error Handling

**Phase-Level Error Matrix**:

| Phase | Error | Blocking? | Action |
|-------|-------|-----------|--------|
| Phase 1 | Session not found | Yes | Error and exit |
| Phase 1 | No completed tasks | Yes | Error and exit |
| Phase 1 | No changed files | Yes | Error and exit |
| Phase 2 | Single dimension fails | No | Log warning, continue other dimensions |
| Phase 2 | All dimensions fail | Yes | Error and exit |
| Phase 3 | Missing dimension JSON | No | Skip in aggregation, log warning |
| Phase 4 | Deep-dive agent fails | No | Skip finding, continue others |
| Phase 4 | Max iterations reached | No | Generate partial report |

**CLI Fallback Chain**: Gemini → Qwen → Codex → degraded mode

**Fallback Triggers**:
1. HTTP 429, 5xx errors, connection timeout
2. Invalid JSON output (parse error, missing required fields)
3. Low confidence score < 0.4
4. Analysis too brief (< 100 words in report)

**Fallback Behavior**:
- On trigger: Retry with next tool in chain
- After Codex fails: Enter degraded mode (skip analysis, log error)
- Degraded mode: Continue workflow with available results

### TodoWrite Structure

```javascript
TodoWrite({
  todos: [
    { content: "Phase 1: Discovery & Initialization", status: "completed", activeForm: "Initializing" },
    { content: "Phase 2: Parallel Reviews (7 dimensions)", status: "in_progress", activeForm: "Reviewing" },
    { content: "  → Security review", status: "in_progress", activeForm: "Analyzing security" },
    // ... other dimensions as sub-items
    { content: "Phase 3: Aggregation", status: "pending", activeForm: "Aggregating" },
    { content: "Phase 4: Deep-dive", status: "pending", activeForm: "Deep-diving" },
    { content: "Phase 5: Completion", status: "pending", activeForm: "Completing" }
  ]
});
```

## Best Practices

1. **Default Settings Work**: 7 dimensions + 3 iterations sufficient for most cases
2. **Parallel Execution**: ~60 minutes for full initial review (7 dimensions)
3. **Trust Aggregation Logic**: Auto-selection based on proven heuristics
4. **Monitor Logs**: Check reports/ directory for CLI analysis insights

## Related Commands

### View Review Progress
Use `ccw view` to open the review dashboard in browser:

```bash
ccw view
```

### Automated Fix Workflow
After completing a review, use the generated findings JSON for automated fixing:

```bash
# Step 1: Complete review (this command)
/workflow:review-session-cycle

# Step 2: Run automated fixes using dimension findings
/workflow:review-fix .workflow/active/WFS-{session-id}/.review/
```

See `/workflow:review-fix` for automated fixing with smart grouping, parallel execution, and test verification.

