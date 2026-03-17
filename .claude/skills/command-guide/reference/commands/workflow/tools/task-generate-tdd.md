---
name: task-generate-tdd
description: Autonomous TDD task generation using action-planning-agent with Red-Green-Refactor cycles, test-first structure, and cycle validation
argument-hint: "--session WFS-session-id"
examples:
  - /workflow:tools:task-generate-tdd --session WFS-auth
---

# Autonomous TDD Task Generation Command

## Overview
Autonomous TDD task JSON and IMPL_PLAN.md generation using action-planning-agent with two-phase execution: discovery and document generation. Generates complete Red-Green-Refactor cycles contained within each task.

## Core Philosophy
- **Agent-Driven**: Delegate execution to action-planning-agent for autonomous operation
- **Two-Phase Flow**: Discovery (context gathering) → Output (document generation)
- **Memory-First**: Reuse loaded documents from conversation memory
- **MCP-Enhanced**: Use MCP tools for advanced code analysis and research
- **Semantic CLI Selection**: CLI tool usage determined from user's task description, not flags
- **Agent Simplicity**: Agent generates content with semantic CLI detection
- **Path Clarity**: All `focus_paths` prefer absolute paths (e.g., `D:\\project\\src\\module`), or clear relative paths from project root (e.g., `./src/module`)
- **TDD-First**: Every feature starts with a failing test (Red phase)
- **Feature-Complete Tasks**: Each task contains complete Red-Green-Refactor cycle
- **Quantification-Enforced**: All test cases, coverage requirements, and implementation scope MUST include explicit counts and enumerations

## Task Strategy & Philosophy

### Optimized Task Structure (Current)
- **1 feature = 1 task** containing complete TDD cycle internally
- Each task executes Red-Green-Refactor phases sequentially
- Task count = Feature count (typically 5 features = 5 tasks)

**Previous Approach** (Deprecated):
- 1 feature = 3 separate tasks (TEST-N.M, IMPL-N.M, REFACTOR-N.M)
- 5 features = 15 tasks with complex dependency chains
- High context switching cost between phases

### When to Use Subtasks
- Feature complexity >2500 lines or >6 files per TDD cycle
- Multiple independent sub-features needing parallel execution
- Strong technical dependency blocking (e.g., API before UI)
- Different tech stacks or domains within feature

### Task Limits
- **Maximum 18 tasks** (hard limit for TDD workflows)
- **Feature-based**: Complete functional units with internal TDD cycles
- **Hierarchy**: Flat (≤5 simple features) | Two-level (6-10 for complex features with sub-features)
- **Re-scope**: If >18 tasks needed, break project into multiple TDD workflow sessions

### TDD Cycle Mapping
- **Old approach**: 1 feature = 3 tasks (TEST-N.M, IMPL-N.M, REFACTOR-N.M)
- **Current approach**: 1 feature = 1 task (IMPL-N with internal Red-Green-Refactor phases)
- **Complex features**: 1 container (IMPL-N) + subtasks (IMPL-N.M) when necessary

## Execution Process

```
Input Parsing:
   ├─ Parse flags: --session
   └─ Validation: session_id REQUIRED

Phase 1: Discovery & Context Loading (Memory-First)
   ├─ Load session context (if not in memory)
   ├─ Load context package (if not in memory)
   ├─ Load test context package (if not in memory)
   ├─ Extract & load role analyses from context package
   ├─ Load conflict resolution (if exists)
   └─ Optional: MCP external research

Phase 2: Agent Execution (Document Generation)
   ├─ Pre-agent template selection (semantic CLI detection)
   ├─ Invoke action-planning-agent
   ├─ Generate TDD Task JSON Files (.task/IMPL-*.json)
   │  └─ Each task: complete Red-Green-Refactor cycle internally
   ├─ Create IMPL_PLAN.md (TDD variant)
   └─ Generate TODO_LIST.md with TDD phase indicators
```

## Execution Lifecycle

### Phase 1: Discovery & Context Loading
**⚡ Memory-First Rule**: Skip file loading if documents already in conversation memory

**Agent Context Package**:
```javascript
{
  "session_id": "WFS-[session-id]",
  "workflow_type": "tdd",
  // Note: CLI tool usage is determined semantically by action-planning-agent based on user's task description
  "session_metadata": {
    // If in memory: use cached content
    // Else: Load from .workflow/active//{session-id}/workflow-session.json
  },
  "brainstorm_artifacts": {
    // Loaded from context-package.json → brainstorm_artifacts section
    "role_analyses": [
      {
        "role": "system-architect",
        "files": [{"path": "...", "type": "primary|supplementary"}]
      }
    ],
    "guidance_specification": {"path": "...", "exists": true},
    "synthesis_output": {"path": "...", "exists": true},
    "conflict_resolution": {"path": "...", "exists": true}  // if conflict_risk >= medium
  },
  "context_package_path": ".workflow/active//{session-id}/.process/context-package.json",
  "context_package": {
    // If in memory: use cached content
    // Else: Load from .workflow/active//{session-id}/.process/context-package.json
  },
  "test_context_package_path": ".workflow/active//{session-id}/.process/test-context-package.json",
  "test_context_package": {
    // Existing test patterns and coverage analysis
  },
  "mcp_capabilities": {
    "code_index": true,
    "exa_code": true,
    "exa_web": true
  }
}
```

**Discovery Actions**:
1. **Load Session Context** (if not in memory)
   ```javascript
   if (!memory.has("workflow-session.json")) {
     Read(.workflow/active//{session-id}/workflow-session.json)
   }
   ```

2. **Load Context Package** (if not in memory)
   ```javascript
   if (!memory.has("context-package.json")) {
     Read(.workflow/active//{session-id}/.process/context-package.json)
   }
   ```

3. **Load Test Context Package** (if not in memory)
   ```javascript
   if (!memory.has("test-context-package.json")) {
     Read(.workflow/active//{session-id}/.process/test-context-package.json)
   }
   ```

4. **Extract & Load Role Analyses** (from context-package.json)
   ```javascript
   // Extract role analysis paths from context package
   const roleAnalysisPaths = contextPackage.brainstorm_artifacts.role_analyses
     .flatMap(role => role.files.map(f => f.path));

   // Load each role analysis file
   roleAnalysisPaths.forEach(path => Read(path));
   ```

5. **Load Conflict Resolution** (from context-package.json, if exists)
   ```javascript
   if (contextPackage.brainstorm_artifacts.conflict_resolution?.exists) {
     Read(contextPackage.brainstorm_artifacts.conflict_resolution.path)
   }
   ```

6. **Code Analysis with Native Tools** (optional - enhance understanding)
   ```bash
   # Find relevant test files and patterns
   find . -name "*test*" -type f
   rg "describe|it\(|test\(" -g "*.ts"
   ```

7. **MCP External Research** (optional - gather TDD best practices)
   ```javascript
   // Get external TDD examples and patterns
   mcp__exa__get_code_context_exa(
     query="TypeScript TDD best practices Red-Green-Refactor",
     tokensNum="dynamic"
   )
   ```

### Phase 2: Agent Execution (Document Generation)

**Pre-Agent Template Selection** (Command decides path before invoking agent):
```javascript
// Command checks flag and selects template PATH (not content)
const templatePath = hasCliExecuteFlag
  ? "~/.claude/workflows/cli-templates/prompts/workflow/task-json-cli-mode.txt"
  : "~/.claude/workflows/cli-templates/prompts/workflow/task-json-agent-mode.txt";
```

**Agent Invocation**:
```javascript
Task(
  subagent_type="action-planning-agent",
  description="Generate TDD task JSON and implementation plan",
  prompt=`
## Execution Context

**Session ID**: WFS-{session-id}
**Workflow Type**: TDD
**Note**: CLI tool usage is determined semantically from user's task description

## Phase 1: Discovery Results (Provided Context)

### Session Metadata
{session_metadata_content}

### Role Analyses (Enhanced by Synthesis)
{role_analyses_content}
- Includes requirements, design specs, enhancements, and clarifications from synthesis phase

### Artifacts Inventory
- **Guidance Specification**: {guidance_spec_path}
- **Role Analyses**: {role_analyses_list}

### Context Package
{context_package_summary}
- Includes conflict_risk assessment

### Test Context Package
{test_context_package_summary}
- Existing test patterns, framework config, coverage analysis

### Conflict Resolution (Conditional)
If conflict_risk was medium/high, modifications have been applied to:
- **guidance-specification.md**: Design decisions updated to resolve conflicts
- **Role analyses (*.md)**: Recommendations adjusted for compatibility
- **context-package.json**: Marked as "resolved" with conflict IDs
- NO separate CONFLICT_RESOLUTION.md file (conflicts resolved in-place)

### MCP Analysis Results (Optional)
**Code Structure**: {mcp_code_index_results}
**External Research**: {mcp_exa_research_results}

## Phase 2: TDD Document Generation Task

**Agent Configuration Reference**: All TDD task generation rules, quantification requirements, Red-Green-Refactor cycle structure, quality standards, and execution details are defined in action-planning-agent.

Refer to: @.claude/agents/action-planning-agent.md for:
- TDD Task Decomposition Standards
- Red-Green-Refactor Cycle Requirements
- Quantification Requirements (MANDATORY)
- 5-Field Task JSON Schema
- IMPL_PLAN.md Structure (TDD variant)
- TODO_LIST.md Format
- TDD Execution Flow & Quality Validation

### TDD-Specific Requirements Summary

#### Task Structure Philosophy
- **1 feature = 1 task** containing complete TDD cycle internally
- Each task executes Red-Green-Refactor phases sequentially
- Task count = Feature count (typically 5 features = 5 tasks)
- Subtasks only when complexity >2500 lines or >6 files per cycle
- **Maximum 18 tasks** (hard limit for TDD workflows)

#### TDD Cycle Mapping
- **Simple features**: IMPL-N with internal Red-Green-Refactor phases
- **Complex features**: IMPL-N (container) + IMPL-N.M (subtasks)
- Each cycle includes: test_count, test_cases array, implementation_scope, expected_coverage

#### Required Outputs Summary

##### 1. TDD Task JSON Files (.task/IMPL-*.json)
- **Location**: `.workflow/active//{session-id}/.task/`
- **Schema**: 5-field structure with TDD-specific metadata
  - `meta.tdd_workflow`: true (REQUIRED)
  - `meta.max_iterations`: 3 (Green phase test-fix cycle limit)
  - `context.tdd_cycles`: Array with quantified test cases and coverage
  - `flow_control.implementation_approach`: Exactly 3 steps with `tdd_phase` field
    1. Red Phase (`tdd_phase: "red"`): Write failing tests
    2. Green Phase (`tdd_phase: "green"`): Implement to pass tests
    3. Refactor Phase (`tdd_phase: "refactor"`): Improve code quality
  - CLI tool usage determined semantically (add `command` field when user requests CLI execution)
- **Details**: See action-planning-agent.md § TDD Task JSON Generation

##### 2. IMPL_PLAN.md (TDD Variant)
- **Location**: `.workflow/active//{session-id}/IMPL_PLAN.md`
- **Template**: `~/.claude/workflows/cli-templates/prompts/workflow/impl-plan-template.txt`
- **TDD-Specific Frontmatter**: workflow_type="tdd", tdd_workflow=true, feature_count, task_breakdown
- **TDD Implementation Tasks Section**: Feature-by-feature with internal Red-Green-Refactor cycles
- **Details**: See action-planning-agent.md § TDD Implementation Plan Creation

##### 3. TODO_LIST.md
- **Location**: `.workflow/active//{session-id}/TODO_LIST.md`
- **Format**: Hierarchical task list with internal TDD phase indicators (Red → Green → Refactor)
- **Status**: ▸ (container), [ ] (pending), [x] (completed)
- **Details**: See action-planning-agent.md § TODO List Generation

### Quantification Requirements (MANDATORY)

**Core Rules**:
1. **Explicit Test Case Counts**: Red phase specifies exact number with enumerated list
2. **Quantified Coverage**: Acceptance includes measurable percentage (e.g., ">=85%")
3. **Detailed Implementation Scope**: Green phase enumerates files, functions, line counts
4. **Enumerated Refactoring Targets**: Refactor phase lists specific improvements with counts

**TDD Phase Formats**:
- **Red Phase**: "Write N test cases: [test1, test2, ...]"
- **Green Phase**: "Implement N functions in file lines X-Y: [func1() X1-Y1, func2() X2-Y2, ...]"
- **Refactor Phase**: "Apply N refactorings: [improvement1 (details), improvement2 (details), ...]"
- **Acceptance**: "All N tests pass with >=X% coverage: verify by [test command]"

**Validation Checklist**:
- [ ] Every Red phase specifies exact test case count with enumerated list
- [ ] Every Green phase enumerates files, functions, and estimated line counts
- [ ] Every Refactor phase lists specific improvements with counts
- [ ] Every acceptance criterion includes measurable coverage percentage
- [ ] tdd_cycles array contains test_count and test_cases for each cycle
- [ ] No vague language ("comprehensive", "complete", "thorough")

### Agent Execution Summary

**Key Steps** (Detailed instructions in action-planning-agent.md):
1. Load task JSON template from provided path
2. Extract and decompose features with TDD cycles
3. Generate TDD task JSON files enforcing quantification requirements
4. Create IMPL_PLAN.md using TDD template variant
5. Generate TODO_LIST.md with TDD phase indicators
6. Update session state with TDD metadata

**Quality Gates** (Full checklist in action-planning-agent.md):
- ✓ Quantification requirements enforced (explicit counts, measurable acceptance, exact targets)
- ✓ Task count ≤18 (hard limit)
- ✓ Each task has meta.tdd_workflow: true
- ✓ Each task has exactly 3 implementation steps with tdd_phase field
- ✓ Green phase includes test-fix cycle logic
- ✓ Artifact references mapped correctly
- ✓ MCP tool integration added
- ✓ Documents follow TDD template structure

## Output

Generate all three documents and report completion status:
- TDD task JSON files created: N files (IMPL-*.json)
- TDD cycles configured: N cycles with quantified test cases
- Artifacts integrated: synthesis-spec, guidance-specification, N role analyses
- Test context integrated: existing patterns and coverage
- MCP enhancements: code-index, exa-research
- Session ready for TDD execution: /workflow:execute
`
)
```

### Agent Context Passing

**Memory-Aware Context Assembly**:
```javascript
// Assemble context package for agent
const agentContext = {
  session_id: "WFS-[id]",
  workflow_type: "tdd",

  // Use memory if available, else load
  session_metadata: memory.has("workflow-session.json")
    ? memory.get("workflow-session.json")
    : Read(.workflow/active/WFS-[id]/workflow-session.json),

  context_package_path: ".workflow/active/WFS-[id]/.process/context-package.json",

  context_package: memory.has("context-package.json")
    ? memory.get("context-package.json")
    : Read(".workflow/active/WFS-[id]/.process/context-package.json"),

  test_context_package_path: ".workflow/active/WFS-[id]/.process/test-context-package.json",

  test_context_package: memory.has("test-context-package.json")
    ? memory.get("test-context-package.json")
    : Read(".workflow/active/WFS-[id]/.process/test-context-package.json"),

  // Extract brainstorm artifacts from context package
  brainstorm_artifacts: extractBrainstormArtifacts(context_package),

  // Load role analyses using paths from context package
  role_analyses: brainstorm_artifacts.role_analyses
    .flatMap(role => role.files)
    .map(file => Read(file.path)),

  // Load conflict resolution if exists (from context package)
  conflict_resolution: brainstorm_artifacts.conflict_resolution?.exists
    ? Read(brainstorm_artifacts.conflict_resolution.path)
    : null,

  // Optional MCP enhancements
  mcp_analysis: executeMcpDiscovery()
}
```

## TDD Task Structure Reference

This section provides quick reference for TDD task JSON structure. For complete implementation details, see the agent invocation prompt in Phase 2 above.

**Quick Reference**:
- Each TDD task contains complete Red-Green-Refactor cycle
- Task ID format: `IMPL-N` (simple) or `IMPL-N.M` (complex subtasks)
- Required metadata: `meta.tdd_workflow: true`, `meta.max_iterations: 3`
- Flow control: Exactly 3 steps with `tdd_phase` field (red, green, refactor)
- Context: `tdd_cycles` array with quantified test cases and coverage
- See Phase 2 agent prompt for full schema and requirements

## Output Files Structure
```
.workflow/active//{session-id}/
├── IMPL_PLAN.md                     # Unified plan with TDD Implementation Tasks section
├── TODO_LIST.md                     # Progress tracking with internal TDD phase indicators
├── .task/
│   ├── IMPL-1.json                  # Complete TDD task (Red-Green-Refactor internally)
│   ├── IMPL-2.json                  # Complete TDD task
│   ├── IMPL-3.json                  # Complex feature container (if needed)
│   ├── IMPL-3.1.json                # Complex feature subtask (if needed)
│   ├── IMPL-3.2.json                # Complex feature subtask (if needed)
│   └── ...
└── .process/
    ├── CONFLICT_RESOLUTION.md       # Conflict resolution strategies (if conflict_risk ≥ medium)
    ├── test-context-package.json    # Test coverage analysis
    ├── context-package.json         # Input from context-gather
    ├── context_package_path         # Path to smart context package
    └── green-fix-iteration-*.md     # Fix logs from Green phase test-fix cycles
```

**File Count**:
- **Old approach**: 5 features = 15 task JSON files (TEST/IMPL/REFACTOR × 5)
- **New approach**: 5 features = 5 task JSON files (IMPL-N × 5)
- **Complex feature**: 1 feature = 1 container + M subtasks (IMPL-N + IMPL-N.M)

## Validation Rules

### Task Completeness
- Every IMPL-N must contain complete TDD workflow in `flow_control.implementation_approach`
- Each task must have 3 steps with `tdd_phase`: "red", "green", "refactor"
- Every task must have `meta.tdd_workflow: true`

### Dependency Enforcement
- Sequential features: IMPL-N depends_on ["IMPL-(N-1)"] if needed
- Complex feature subtasks: IMPL-N.M depends_on ["IMPL-N.(M-1)"] or parent dependencies
- No circular dependencies allowed

### Task Limits
- Maximum 10 total tasks (simple + subtasks)
- Flat hierarchy (≤5 tasks) or two-level (6-10 tasks with containers)
- Re-scope requirements if >10 tasks needed

### TDD Workflow Validation
- `meta.tdd_workflow` must be true
- `flow_control.implementation_approach` must have exactly 3 steps
- Each step must have `tdd_phase` field ("red", "green", or "refactor")
- Green phase step must include test-fix cycle logic
- `meta.max_iterations` must be present (default: 3)

## Error Handling

### Input Validation Errors
| Error | Cause | Resolution |
|-------|-------|------------|
| Session not found | Invalid session ID | Verify session exists |
| Context missing | Incomplete planning | Run context-gather first |

### TDD Generation Errors
| Error | Cause | Resolution |
|-------|-------|------------|
| Task count exceeds 10 | Too many features or subtasks | Re-scope requirements or merge features |
| Missing test framework | No test config | Configure testing first |
| Invalid TDD workflow | Missing tdd_phase or incomplete flow_control | Fix TDD structure in ANALYSIS_RESULTS.md |
| Missing tdd_workflow flag | Task doesn't have meta.tdd_workflow: true | Add TDD workflow metadata |

## Integration & Usage

**Command Chain**:
- Called by: `/workflow:tdd-plan` (Phase 4)
- Invokes: `action-planning-agent` for autonomous task generation
- Followed by: `/workflow:execute`, `/workflow:tdd-verify`

**Basic Usage**:
```bash
# Standard execution
/workflow:tools:task-generate-tdd --session WFS-auth

# With semantic CLI request (include in task description)
# e.g., "Generate TDD tasks for auth module, use Codex for implementation"
```

**CLI Tool Selection**: Determined semantically from user's task description. Include "use Codex/Gemini/Qwen" in your request for CLI execution.

**Output**:
- TDD task JSON files in `.task/` directory (IMPL-N.json format)
- IMPL_PLAN.md with TDD Implementation Tasks section
- TODO_LIST.md with internal TDD phase indicators
- Session state updated with task count and TDD metadata
- MCP enhancements integrated (if available)

## Test Coverage Analysis Integration

The TDD workflow includes test coverage analysis (via `/workflow:tools:test-context-gather`) to:
- Detect existing test patterns and conventions
- Identify current test coverage gaps
- Discover test framework and configuration
- Enable integration with existing tests

This makes TDD workflow context-aware instead of assuming greenfield scenarios.

## Iterative Green Phase with Test-Fix Cycle

IMPL (Green phase) tasks include automatic test-fix cycle:

**Process Flow**:
1. **Initial Implementation**: Write minimal code to pass tests
2. **Test Execution**: Run test suite
3. **Success Path**: Tests pass → Complete task
4. **Failure Path**: Tests fail → Enter iterative fix cycle:
   - **Gemini Diagnosis**: Analyze failures with bug-fix template
   - **Fix Application**: Agent (default) or CLI (if `command` field present)
   - **Retest**: Verify fix resolves failures
   - **Repeat**: Up to max_iterations (default: 3)
5. **Safety Net**: Auto-revert all changes if max iterations reached



## Configuration Options
- **meta.max_iterations**: Number of fix attempts (default: 3 for TDD, 5 for test-gen)
- **CLI tool usage**: Determined semantically from user's task description via `command` field in implementation_approach

