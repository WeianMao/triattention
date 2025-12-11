---
name: test-task-generate
description: Generate test planning documents (IMPL_PLAN.md, test task JSONs, TODO_LIST.md) using action-planning-agent - produces test planning artifacts, does NOT execute tests
argument-hint: "--session WFS-test-session-id"
examples:
  - /workflow:tools:test-task-generate --session WFS-test-auth
---

# Generate Test Planning Documents Command

## Overview
Generate test planning documents (IMPL_PLAN.md, test task JSONs, TODO_LIST.md) using action-planning-agent. This command produces **test planning artifacts only** - it does NOT execute tests or implement code. Actual test execution requires separate execution command (e.g., /workflow:test-cycle-execute).

## Core Philosophy
- **Planning Only**: Generate test planning documents (IMPL_PLAN.md, task JSONs, TODO_LIST.md) - does NOT execute tests
- **Agent-Driven Document Generation**: Delegate test plan generation to action-planning-agent
- **Two-Phase Flow**: Context Preparation (command) → Test Document Generation (agent)
- **Memory-First**: Reuse loaded documents from conversation memory
- **MCP-Enhanced**: Use MCP tools for test pattern research and analysis
- **Path Clarity**: All `focus_paths` prefer absolute paths (e.g., `D:\\project\\src\\module`), or clear relative paths from project root
- **Leverage Existing Test Infrastructure**: Prioritize using established testing frameworks and tools present in the project

## Test-Specific Execution Modes

### Test Generation (IMPL-001)
- **Agent Mode** (default): @code-developer generates tests within agent context
- **CLI Mode**: Use CLI tools when `command` field present in implementation_approach (determined semantically)

### Test Execution & Fix (IMPL-002+)
- **Agent Mode** (default): Gemini diagnosis → agent applies fixes
- **CLI Mode**: Gemini diagnosis → CLI applies fixes (when `command` field present in implementation_approach)

## Execution Process

```
Input Parsing:
   ├─ Parse flags: --session
   └─ Validation: session_id REQUIRED

Phase 1: Context Preparation (Command)
   ├─ Assemble test session paths
   │  ├─ session_metadata_path
   │  ├─ test_analysis_results_path (REQUIRED)
   │  └─ test_context_package_path
   └─ Provide metadata (session_id, source_session_id)

Phase 2: Test Document Generation (Agent)
   ├─ Load TEST_ANALYSIS_RESULTS.md as primary requirements source
   ├─ Generate Test Task JSON Files (.task/IMPL-*.json)
   │  ├─ IMPL-001: Test generation (meta.type: "test-gen")
   │  └─ IMPL-002+: Test execution & fix (meta.type: "test-fix")
   ├─ Create IMPL_PLAN.md (test_session variant)
   └─ Generate TODO_LIST.md with test phase indicators
```

## Document Generation Lifecycle

### Phase 1: Context Preparation (Command Responsibility)

**Command prepares test session paths and metadata for planning document generation.**

**Test Session Path Structure**:
```
.workflow/active/WFS-test-{session-id}/
├── workflow-session.json          # Test session metadata
├── .process/
│   ├── TEST_ANALYSIS_RESULTS.md   # Test requirements and strategy
│   ├── test-context-package.json  # Test patterns and coverage
│   └── context-package.json       # General context artifacts
├── .task/                         # Output: Test task JSON files
├── IMPL_PLAN.md                   # Output: Test implementation plan
└── TODO_LIST.md                   # Output: Test TODO list
```

**Command Preparation**:
1. **Assemble Test Session Paths** for agent prompt:
   - `session_metadata_path`
   - `test_analysis_results_path` (REQUIRED)
   - `test_context_package_path`
   - Output directory paths

2. **Provide Metadata** (simple values):
   - `session_id`
   - `source_session_id` (if exists)
   - `mcp_capabilities` (available MCP tools)

**Note**: CLI tool usage is now determined semantically from user's task description, not by flags.

### Phase 2: Test Document Generation (Agent Responsibility)

**Purpose**: Generate test-specific IMPL_PLAN.md, task JSONs, and TODO_LIST.md - planning documents only, NOT test execution.

**Agent Invocation**:
```javascript
Task(
  subagent_type="action-planning-agent",
  description="Generate test planning documents (IMPL_PLAN.md, task JSONs, TODO_LIST.md)",
  prompt=`
## TASK OBJECTIVE
Generate test planning documents (IMPL_PLAN.md, task JSONs, TODO_LIST.md) for test workflow session

IMPORTANT: This is TEST PLANNING ONLY - you are generating planning documents, NOT executing tests.

CRITICAL:
- Use existing test frameworks and utilities from the project
- Follow the progressive loading strategy defined in your agent specification (load context incrementally from memory-first approach)

## AGENT CONFIGURATION REFERENCE
All test task generation rules, schemas, and quality standards are defined in your agent specification:
@.claude/agents/action-planning-agent.md

Refer to your specification for:
- Test Task JSON Schema (6-field structure with test-specific metadata)
- Test IMPL_PLAN.md Structure (test_session variant with test-fix cycle)
- TODO_LIST.md Format (with test phase indicators)
- Progressive Loading Strategy (memory-first, load TEST_ANALYSIS_RESULTS.md as primary source)
- Quality Validation Rules (task count limits, requirement quantification)

## SESSION PATHS
Input:
  - Session Metadata: .workflow/active/{test-session-id}/workflow-session.json
  - TEST_ANALYSIS_RESULTS: .workflow/active/{test-session-id}/.process/TEST_ANALYSIS_RESULTS.md (REQUIRED - primary requirements source)
  - Test Context Package: .workflow/active/{test-session-id}/.process/test-context-package.json
  - Context Package: .workflow/active/{test-session-id}/.process/context-package.json
  - Source Session Summaries: .workflow/active/{source-session-id}/.summaries/IMPL-*.md (if exists)

Output:
  - Task Dir: .workflow/active/{test-session-id}/.task/
  - IMPL_PLAN: .workflow/active/{test-session-id}/IMPL_PLAN.md
  - TODO_LIST: .workflow/active/{test-session-id}/TODO_LIST.md

## CONTEXT METADATA
Session ID: {test-session-id}
Workflow Type: test_session
Source Session: {source-session-id} (if exists)
MCP Capabilities: {exa_code, exa_web, code_index}

## CLI TOOL SELECTION
Determine CLI tool usage per-step based on user's task description:
- If user specifies "use Codex/Gemini/Qwen for X" → Add command field to relevant steps
- Default: Agent execution (no command field) unless user explicitly requests CLI

## TEST-SPECIFIC REQUIREMENTS SUMMARY
(Detailed specifications in your agent definition)

### Task Structure Requirements
- Minimum 2 tasks: IMPL-001 (test generation) + IMPL-002 (test execution & fix)
- Expandable for complex projects: Add IMPL-003+ (per-module, integration, E2E tests)

Task Configuration:
  IMPL-001 (Test Generation):
    - meta.type: "test-gen"
    - meta.agent: "@code-developer"
    - meta.test_framework: Specify existing framework (e.g., "jest", "vitest", "pytest")
    - flow_control: Test generation strategy from TEST_ANALYSIS_RESULTS.md
    - CLI execution: Add `command` field when user requests (determined semantically)

  IMPL-002+ (Test Execution & Fix):
    - meta.type: "test-fix"
    - meta.agent: "@test-fix-agent"
    - flow_control: Test-fix cycle with iteration limits and diagnosis configuration
    - CLI execution: Add `command` field when user requests (determined semantically)

### Test-Fix Cycle Specification (IMPL-002+)
Required flow_control fields:
  - max_iterations: 5
  - diagnosis_tool: "gemini"
  - diagnosis_template: "~/.claude/workflows/cli-templates/prompts/analysis/01-diagnose-bug-root-cause.txt"
  - cycle_pattern: "test → gemini_diagnose → fix → retest"
  - exit_conditions: ["all_tests_pass", "max_iterations_reached"]
  - auto_revert_on_failure: true
  - CLI fix: Add `command` field when user specifies CLI tool usage

### Automation Framework Configuration
Select automation tools based on test requirements from TEST_ANALYSIS_RESULTS.md:
- UI interaction testing → E2E browser automation (meta.e2e_framework)
- API/database integration → integration test tools (meta.test_tools)
- Performance metrics → load testing tools (meta.perf_framework)
- Logic verification → unit test framework (meta.test_framework)

**Tool Selection**: Detect from project config > suggest based on requirements

### TEST_ANALYSIS_RESULTS.md Mapping
PRIMARY requirements source - extract and map to task JSONs:
  - Test framework config → meta.test_framework (use existing framework from project)
  - Existing test utilities → flow_control.reusable_test_tools (discovered test helpers, fixtures, mocks)
  - Test runner commands → flow_control.test_commands (from package.json or pytest config)
  - Coverage targets → meta.coverage_target
  - Test requirements → context.requirements (quantified with explicit counts)
  - Test generation strategy → IMPL-001 flow_control.implementation_approach
  - Implementation targets → context.files_to_test (absolute paths)

## EXPECTED DELIVERABLES
1. Test Task JSON Files (.task/IMPL-*.json)
   - 6-field schema with quantified requirements from TEST_ANALYSIS_RESULTS.md
   - Test-specific metadata: type, agent, test_framework, coverage_target
   - flow_control includes: reusable_test_tools, test_commands (from project config)
   - CLI execution via `command` field when user requests (determined semantically)
   - Artifact references from test-context-package.json
   - Absolute paths in context.files_to_test

2. Test Implementation Plan (IMPL_PLAN.md)
   - Template: ~/.claude/workflows/cli-templates/prompts/workflow/impl-plan-template.txt
   - Test-specific frontmatter: workflow_type="test_session", test_framework, source_session_id
   - Test-Fix-Retest Cycle section with diagnosis configuration
   - Source session context integration (if applicable)

3. TODO List (TODO_LIST.md)
   - Hierarchical structure with test phase containers
   - Links to task JSONs with status markers
   - Matches task JSON hierarchy

## QUALITY STANDARDS
Hard Constraints:
  - Task count: minimum 2, maximum 18
  - All requirements quantified from TEST_ANALYSIS_RESULTS.md
  - Test framework matches existing project framework
  - flow_control includes reusable_test_tools and test_commands from project
  - Absolute paths for all focus_paths
  - Acceptance criteria include verification commands
  - CLI `command` field added only when user explicitly requests CLI tool usage

## SUCCESS CRITERIA
- All test planning documents generated successfully
- Return completion status: task count, test framework, coverage targets, source session status
`
)
```

## Integration & Usage

### Command Chain
- **Called By**: `/workflow:test-gen` (Phase 4), `/workflow:test-fix-gen` (Phase 4)
- **Invokes**: `action-planning-agent` for test planning document generation
- **Followed By**: `/workflow:test-cycle-execute` or `/workflow:execute` (user-triggered)

### Usage Examples
```bash
# Standard execution
/workflow:tools:test-task-generate --session WFS-test-auth

# With semantic CLI request (include in task description)
# e.g., "Generate tests, use Codex for implementation and fixes"
```

### CLI Tool Selection
CLI tool usage is determined semantically from user's task description:
- Include "use Codex" for automated fixes
- Include "use Gemini" for analysis
- Default: Agent execution (no `command` field)

### Output
- Test task JSON files in `.task/` directory (minimum 2)
- IMPL_PLAN.md with test strategy and fix cycle specification
- TODO_LIST.md with test phase indicators
- Session ready for test execution
