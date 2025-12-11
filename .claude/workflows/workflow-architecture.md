# Workflow Architecture

## Overview

This document defines the complete workflow system architecture using a **JSON-only data model**, **marker-based session management**, and **unified file structure** with dynamic task decomposition.

## Core Architecture

### JSON-Only Data Model
**JSON files (.task/IMPL-*.json) are the only authoritative source of task state. All markdown documents are read-only generated views.**

- **Task State**: Stored exclusively in JSON files
- **Documents**: Generated on-demand from JSON data
- **No Synchronization**: Eliminates bidirectional sync complexity
- **Performance**: Direct JSON access without parsing overhead

### Key Design Decisions
- **JSON files are the single source of truth** - All markdown documents are read-only generated views
- **Marker files for session tracking** - Ultra-simple active session management
- **Unified file structure definition** - Same structure template for all workflows, created on-demand
- **Dynamic task decomposition** - Subtasks created as needed during execution
- **On-demand file creation** - Directories and files created only when required
- **Agent-agnostic task definitions** - Complete context preserved for autonomous execution

## Session Management

### Directory-Based Session Management
**Simple Location-Based Tracking**: Sessions in `.workflow/active/` directory

```bash
.workflow/
├── active/
│   ├── WFS-oauth-integration/         # Active session directory
│   ├── WFS-user-profile/             # Active session directory
│   └── WFS-bug-fix-123/              # Active session directory
└── archives/
    └── WFS-old-feature/              # Archived session (completed)
```


### Session Operations

#### Detect Active Session(s)
```bash
active_sessions=$(find .workflow/active/ -name "WFS-*" -type d 2>/dev/null)
count=$(echo "$active_sessions" | wc -l)

if [ -z "$active_sessions" ]; then
  echo "No active session"
elif [ "$count" -eq 1 ]; then
  session_name=$(basename "$active_sessions")
  echo "Active session: $session_name"
else
  echo "Multiple sessions found:"
  echo "$active_sessions" | while read session_dir; do
    session=$(basename "$session_dir")
    echo "  - $session"
  done
  echo "Please specify which session to work with"
fi
```

#### Archive Session
```bash
mv .workflow/active/WFS-feature .workflow/archives/WFS-feature
```

### Session State Tracking
Each session directory contains `workflow-session.json`:

```json
{
  "session_id": "WFS-[topic-slug]",
  "project": "feature description",
  "type": "simple|medium|complex",
  "current_phase": "PLAN|IMPLEMENT|REVIEW",
  "status": "active|paused|completed",
  "progress": {
    "completed_phases": ["PLAN"],
    "current_tasks": ["IMPL-1", "IMPL-2"]
  }
}
```

## Task System

### Hierarchical Task Structure
**Maximum Depth**: 2 levels (IMPL-N.M format)

```
IMPL-1              # Main task
IMPL-1.1            # Subtask of IMPL-1 (dynamically created)
IMPL-1.2            # Another subtask of IMPL-1
IMPL-2              # Another main task
IMPL-2.1            # Subtask of IMPL-2 (dynamically created)
```

**Task Status Rules**:
- **Container tasks**: Parent tasks with subtasks (cannot be directly executed)
- **Leaf tasks**: Only these can be executed directly
- **Status inheritance**: Parent status derived from subtask completion

### Enhanced Task JSON Schema
All task files use this unified 6-field schema with optional artifacts enhancement:

```json
{
  "id": "IMPL-1.2",
  "title": "Implement JWT authentication",
  "status": "pending|active|completed|blocked|container",
  "context_package_path": ".workflow/WFS-session/.process/context-package.json",

  "meta": {
    "type": "feature|bugfix|refactor|test-gen|test-fix|docs",
    "agent": "@code-developer|@action-planning-agent|@test-fix-agent|@universal-executor"
  },

  "context": {
    "requirements": ["JWT authentication", "OAuth2 support"],
    "focus_paths": ["src/auth", "tests/auth", "config/auth.json"],
    "acceptance": ["JWT validation works", "OAuth flow complete"],
    "parent": "IMPL-1",
    "depends_on": ["IMPL-1.1"],
    "inherited": {
      "from": "IMPL-1",
      "context": ["Authentication system design completed"]
    },
    "shared_context": {
      "auth_strategy": "JWT with refresh tokens"
    },
    "artifacts": [
      {
        "type": "role_analyses",
        "source": "brainstorm_clarification",
        "path": ".workflow/WFS-session/.brainstorming/*/analysis*.md",
        "priority": "highest",
        "contains": "role_specific_requirements_and_design"
      }
    ]
  },

  "flow_control": {
    "pre_analysis": [
      {
        "step": "check_patterns",
        "action": "Analyze existing patterns",
        "command": "bash(rg 'auth' [focus_paths] | head -10)",
        "output_to": "patterns"
      },
      {
        "step": "analyze_architecture",
        "action": "Review system architecture",
        "command": "gemini \"analyze patterns: [patterns]\"",
        "output_to": "design"
      },
      {
        "step": "check_deps",
        "action": "Check dependencies",
        "command": "bash(echo [depends_on] | xargs cat)",
        "output_to": "context"
      }
    ],
    "implementation_approach": [
      {
        "step": 1,
        "title": "Set up authentication infrastructure",
        "description": "Install JWT library and create auth config following [design] patterns from [parent]",
        "modification_points": [
          "Add JWT library dependencies to package.json",
          "Create auth configuration file using [parent] patterns"
        ],
        "logic_flow": [
          "Install jsonwebtoken library via npm",
          "Configure JWT secret and expiration from [inherited]",
          "Export auth config for use by [jwt_generator]"
        ],
        "depends_on": [],
        "output": "auth_config"
      },
      {
        "step": 2,
        "title": "Implement JWT generation",
        "description": "Create JWT token generation logic using [auth_config] and [inherited] validation patterns",
        "modification_points": [
          "Add JWT generation function in auth service",
          "Implement token signing with [auth_config]"
        ],
        "logic_flow": [
          "User login → validate credentials with [inherited]",
          "Generate JWT payload with user data",
          "Sign JWT using secret from [auth_config]",
          "Return signed token"
        ],
        "depends_on": [1],
        "output": "jwt_generator"
      },
      {
        "step": 3,
        "title": "Implement JWT validation middleware",
        "description": "Create middleware to validate JWT tokens using [auth_config] and [shared] rules",
        "modification_points": [
          "Create validation middleware using [jwt_generator]",
          "Add token verification using [shared] rules",
          "Implement user attachment to request object"
        ],
        "logic_flow": [
          "Protected route → extract JWT from Authorization header",
          "Validate token signature using [auth_config]",
          "Check token expiration and [shared] rules",
          "Decode payload and attach user to request",
          "Call next() or return 401 error"
        ],
        "command": "bash(npm test -- middleware.test.ts)",
        "depends_on": [1, 2],
        "output": "auth_middleware"
      }
    ],
    "target_files": [
      "src/auth/login.ts:handleLogin:75-120",
      "src/middleware/auth.ts:validateToken",
      "src/auth/PasswordReset.ts"
    ]
  }
}
```

### Focus Paths & Context Management

#### Context Package Path (Top-Level Field)
The **context_package_path** field provides the location of the smart context package:
- **Location**: Top-level field (not in `artifacts` array)
- **Path**: `.workflow/WFS-session/.process/context-package.json`
- **Purpose**: References the comprehensive context package containing project structure, dependencies, and brainstorming artifacts catalog
- **Usage**: Loaded in `pre_analysis` steps via `Read({{context_package_path}})`

#### Focus Paths Format
The **focus_paths** field specifies concrete project paths for task implementation:
- **Array of strings**: `["folder1", "folder2", "specific_file.ts"]`
- **Concrete paths**: Use actual directory/file names without wildcards
- **Mixed types**: Can include both directories and specific files
- **Relative paths**: From project root (e.g., `src/auth`, not `./src/auth`)

#### Artifacts Field ⚠️ NEW FIELD
Optional field referencing brainstorming outputs for task execution:

```json
"artifacts": [
  {
    "type": "role_analyses|topic_framework|individual_role_analysis",
    "source": "brainstorm_clarification|brainstorm_framework|brainstorm_roles",
    "path": ".workflow/WFS-session/.brainstorming/document.md",
    "priority": "highest|high|medium|low"
  }
]
```

**Types & Priority**: role_analyses (highest) → topic_framework (medium) → individual_role_analysis (low)

#### Flow Control Configuration
The **flow_control** field manages task execution through structured sequential steps. For complete format specifications and usage guidelines, see [Flow Control Format Guide](#flow-control-format-guide) below.

**Quick Reference**:
- **pre_analysis**: Context gathering steps (supports multiple command types)
- **implementation_approach**: Implementation steps array with dependency management
- **target_files**: Target files for modification (file:function:lines format)
- **Variable references**: Use `[variable_name]` to reference step outputs
- **Tool integration**: Supports Gemini, Codex, Bash commands, and MCP tools

## Flow Control Format Guide

The `[FLOW_CONTROL]` marker indicates that a task or prompt contains flow control steps for sequential execution. There are **two distinct formats** used in different scenarios:

### Format Comparison Matrix

| Aspect | Inline Format | JSON Format |
|--------|--------------|-------------|
| **Used In** | Brainstorm workflows | Implementation tasks |
| **Agent** | conceptual-planning-agent | code-developer, test-fix-agent, doc-generator |
| **Location** | Task() prompt (markdown) | .task/IMPL-*.json file |
| **Persistence** | Temporary (prompt-only) | Persistent (file storage) |
| **Complexity** | Simple (3-5 steps) | Complex (10+ steps) |
| **Dependencies** | None | Full `depends_on` support |
| **Purpose** | Load brainstorming context | Implement task with preparation |

### Inline Format (Brainstorm)

**Marker**: `[FLOW_CONTROL]` written directly in Task() prompt

**Structure**: Markdown list format

**Used By**: Brainstorm commands (`auto-parallel.md`, role commands)

**Agent**: `conceptual-planning-agent`

**Example**:
```markdown
[FLOW_CONTROL]

### Flow Control Steps
**AGENT RESPONSIBILITY**: Execute these pre_analysis steps sequentially with context accumulation:

1. **load_topic_framework**
   - Action: Load structured topic discussion framework
   - Command: Read(.workflow/WFS-{session}/.brainstorming/guidance-specification.md)
   - Output: topic_framework

2. **load_role_template**
   - Action: Load role-specific planning template
   - Command: bash($(cat "~/.claude/workflows/cli-templates/planning-roles/{role}.md"))
   - Output: role_template

3. **load_session_metadata**
   - Action: Load session metadata and topic description
   - Command: bash(cat .workflow/WFS-{session}/workflow-session.json 2>/dev/null || echo '{}')
   - Output: session_metadata
```

**Characteristics**:
- 3-5 simple context loading steps
- Written directly in prompt (not persistent)
- No dependency management between steps
- Used for temporary context preparation
- Variables: `[variable_name]` for output references

### JSON Format (Implementation)

**Marker**: `[FLOW_CONTROL]` used in TodoWrite or documentation to indicate task has flow control

**Structure**: Complete JSON structure in task file

**Used By**: Implementation tasks (IMPL-*.json)

**Agents**: `code-developer`, `test-fix-agent`, `doc-generator`

**Example**:
```json
"flow_control": {
  "pre_analysis": [
    {
      "step": "load_role_analyses",
      "action": "Load role analysis documents from brainstorming",
      "commands": [
        "bash(ls .workflow/WFS-{session}/.brainstorming/*/analysis*.md 2>/dev/null || echo 'not found')",
        "Glob(.workflow/WFS-{session}/.brainstorming/*/analysis*.md)",
        "Read(each discovered role analysis file)"
      ],
      "output_to": "role_analyses",
      "on_error": "skip_optional"
    },
    {
      "step": "local_codebase_exploration",
      "action": "Explore codebase using local search",
      "commands": [
        "bash(rg '^(function|class|interface).*auth' --type ts -n --max-count 15)",
        "bash(find . -name '*auth*' -type f | grep -v node_modules | head -10)"
      ],
      "output_to": "codebase_structure"
    }
  ],
  "implementation_approach": [
    {
      "step": 1,
      "title": "Setup infrastructure",
      "description": "Install JWT library and create config following [role_analyses]",
      "modification_points": [
        "Add JWT library dependencies to package.json",
        "Create auth configuration file"
      ],
      "logic_flow": [
        "Install jsonwebtoken library via npm",
        "Configure JWT secret from [role_analyses]",
        "Export auth config for use by [jwt_generator]"
      ],
      "depends_on": [],
      "output": "auth_config"
    },
    {
      "step": 2,
      "title": "Implement JWT generation",
      "description": "Create JWT token generation logic using [auth_config]",
      "modification_points": [
        "Add JWT generation function in auth service",
        "Implement token signing with [auth_config]"
      ],
      "logic_flow": [
        "User login → validate credentials",
        "Generate JWT payload with user data",
        "Sign JWT using secret from [auth_config]",
        "Return signed token"
      ],
      "depends_on": [1],
      "output": "jwt_generator"
    }
  ],
  "target_files": [
    "src/auth/login.ts:handleLogin:75-120",
    "src/middleware/auth.ts:validateToken"
  ]
}
```

**Characteristics**:
- Persistent storage in .task/IMPL-*.json files
- Complete dependency management (`depends_on` arrays)
- Two-phase structure: `pre_analysis` + `implementation_approach`
- Error handling strategies (`on_error` field)
- Target file specifications
- Variables: `[variable_name]` for cross-step references

### JSON Format Field Specifications

#### pre_analysis Field
**Purpose**: Context gathering phase before implementation

**Structure**: Array of step objects with sequential execution

**Step Fields**:
- **step**: Step identifier (string, e.g., "load_role_analyses")
- **action**: Human-readable description of the step
- **command** or **commands**: Single command string or array of command strings
- **output_to**: Variable name for storing step output
- **on_error**: Error handling strategy (`skip_optional`, `fail`, `retry_once`, `manual_intervention`)

**Command Types Supported**:
- **Bash commands**: `bash(command)` - Any shell command
- **Tool calls**: `Read(file)`, `Glob(pattern)`, `Grep(pattern)`
- **MCP tools**: `mcp__exa__get_code_context_exa()`, `mcp__exa__web_search_exa()`
- **CLI commands**: `gemini`, `qwen`, `codex --full-auto exec`

**Example**:
```json
{
  "step": "load_context",
  "action": "Load project context and patterns",
  "commands": [
    "bash(ccw tool exec get_modules_by_depth '{}')",
    "Read(CLAUDE.md)"
  ],
  "output_to": "project_structure",
  "on_error": "skip_optional"
}
```

#### implementation_approach Field
**Purpose**: Define implementation steps with dependency management

**Structure**: Array of step objects (NOT object format)

**Step Fields (All Required)**:
- **step**: Unique step number (1, 2, 3, ...) - serves as step identifier
- **title**: Brief step title
- **description**: Comprehensive implementation description with context variable references
- **modification_points**: Array of specific code modification targets
- **logic_flow**: Array describing business logic execution sequence
- **depends_on**: Array of step numbers this step depends on (e.g., `[1]`, `[1, 2]`) - empty array `[]` for independent steps
- **output**: Output variable name that can be referenced by subsequent steps via `[output_name]`

**Optional Fields**:
- **command**: Command for step execution (supports any shell command or CLI tool)
  - When omitted: Agent interprets modification_points and logic_flow to execute
  - When specified: Command executes the step directly

**Execution Modes**:
- **Default (without command)**: Agent executes based on modification_points and logic_flow
- **With command**: Specified command handles execution

**Command Field Usage**:
- **Default approach**: Omit command field - let agent execute autonomously
- **CLI tools (codex/gemini/qwen)**: Add ONLY when user explicitly requests CLI tool usage
- **Simple commands**: Can include bash commands, test commands, validation scripts
- **Complex workflows**: Use command for multi-step operations or tool coordination

**Command Format Examples** (only when explicitly needed):
```json
// Simple Bash
"command": "bash(npm install package)"
"command": "bash(npm test)"

// Validation
"command": "bash(test -f config.ts && grep -q 'JWT_SECRET' config.ts)"

// Codex (user requested)
"command": "codex -C path --full-auto exec \"task\" --skip-git-repo-check -s danger-full-access"

// Codex Resume (user requested, maintains context)
"command": "codex --full-auto exec \"task\" resume --last --skip-git-repo-check -s danger-full-access"

// Gemini (user requested)
"command": "gemini \"analyze [context]\""

// Qwen (fallback for Gemini)
"command": "qwen \"analyze [context]\""
```

**Example Step**:
```json
{
  "step": 2,
  "title": "Implement JWT generation",
  "description": "Create JWT token generation logic using [auth_config]",
  "modification_points": [
    "Add JWT generation function in auth service",
    "Implement token signing with [auth_config]"
  ],
  "logic_flow": [
    "User login → validate credentials",
    "Generate JWT payload with user data",
    "Sign JWT using secret from [auth_config]",
    "Return signed token"
  ],
  "depends_on": [1],
  "output": "jwt_generator"
}
```

#### target_files Field
**Purpose**: Specify files to be modified or created

**Format**: Array of strings
- **Existing files**: `"file:function:lines"` (e.g., `"src/auth/login.ts:handleLogin:75-120"`)
- **New files**: `"path/to/NewFile.ts"` (file path only)

### Tool Reference

**Available Command Types**:

**Gemini CLI**:
```bash
gemini "prompt"
gemini --approval-mode yolo "prompt"  # For write mode
```

**Qwen CLI** (Gemini fallback):
```bash
qwen "prompt"
qwen --approval-mode yolo "prompt"  # For write mode
```

**Codex CLI**:
```bash
codex -C directory --full-auto exec "task" --skip-git-repo-check -s danger-full-access
codex --full-auto exec "task" resume --last --skip-git-repo-check -s danger-full-access
```

**Built-in Tools**:
- `Read(file_path)` - Read file contents
- `Glob(pattern)` - Find files by pattern
- `Grep(pattern)` - Search content with regex
- `bash(command)` - Execute bash command

**MCP Tools**:
- `mcp__exa__get_code_context_exa(query="...")` - Get code context from Exa
- `mcp__exa__web_search_exa(query="...")` - Web search via Exa

**Bash Commands**:
```bash
bash(rg 'pattern' src/)
bash(find . -name "*.ts")
bash(npm test)
bash(git log --oneline | head -5)
```

### Variable System & Context Flow

**Variable Reference Syntax**:
Both formats use `[variable_name]` syntax for referencing outputs from previous steps.

**Variable Types**:
- **Step outputs**: `[step_output_name]` - Reference any pre_analysis step output
- **Task properties**: `[task_property]` - Reference any task context field
- **Previous results**: `[analysis_result]` - Reference accumulated context
- **Implementation outputs**: Reference outputs from previous implementation steps

**Examples**:
```json
// Reference pre_analysis output
"description": "Install JWT library following [role_analyses]"

// Reference previous step output
"description": "Create middleware using [auth_config] and [jwt_generator]"

// Reference task context
"command": "bash(cd [focus_paths] && npm test)"
```

**Context Accumulation Process**:
1. **Structure Analysis**: `get_modules_by_depth.sh` → project hierarchy
2. **Pattern Analysis**: Tool-specific commands → existing patterns
3. **Dependency Mapping**: Previous task summaries → inheritance context
4. **Task Context Generation**: Combined analysis → task.context fields

**Context Inheritance Rules**:
- **Parent → Child**: Container tasks pass context via `context.inherited`
- **Dependency → Dependent**: Previous task summaries via `context.depends_on`
- **Session → Task**: Global session context included in all tasks
- **Module → Feature**: Module patterns inform feature implementation

### Agent Processing Rules

**conceptual-planning-agent** (Inline Format):
- Parses markdown list from prompt
- Executes 3-5 simple loading steps
- No dependency resolution needed
- Accumulates context in variables
- Used only in brainstorm workflows

**code-developer, test-fix-agent** (JSON Format):
- Loads complete task JSON from file
- Executes `pre_analysis` steps sequentially
- Processes `implementation_approach` with dependency resolution
- Handles complex variable substitution
- Updates task status in JSON file

### Usage Guidelines

**Use Inline Format When**:
- Running brainstorm workflows
- Need 3-5 simple context loading steps
- No persistence required
- No dependencies between steps
- Temporary context preparation

**Use JSON Format When**:
- Implementing features or tasks
- Need 10+ complex execution steps
- Require dependency management
- Need persistent task definitions
- Complex variable flow between steps
- Error handling strategies needed

### Variable Reference Syntax

Both formats use `[variable_name]` syntax for referencing outputs:

**Inline Format**:
```markdown
2. **analyze_context**
   - Action: Analyze using [topic_framework] and [role_template]
   - Output: analysis_results
```

**JSON Format**:
```json
{
  "step": 2,
  "description": "Implement following [role_analyses] and [codebase_structure]",
  "depends_on": [1],
  "output": "implementation"
}
```

### Task Validation Rules
1. **ID Uniqueness**: All task IDs must be unique
2. **Hierarchical Format**: Must follow IMPL-N[.M] pattern (maximum 2 levels)
3. **Parent References**: All parent IDs must exist as JSON files
4. **Status Consistency**: Status values from defined enumeration
5. **Required Fields**: All 5 core fields must be present (id, title, status, meta, context, flow_control)
6. **Focus Paths Structure**: context.focus_paths must contain concrete paths (no wildcards)
7. **Flow Control Format**: pre_analysis must be array with required fields
8. **Dependency Integrity**: All task-level depends_on references must exist as JSON files
9. **Artifacts Structure**: context.artifacts (optional) must use valid type, priority, and path format
10. **Implementation Steps Array**: implementation_approach must be array of step objects
11. **Step Number Uniqueness**: All step numbers within a task must be unique and sequential (1, 2, 3, ...)
12. **Step Dependencies**: All step-level depends_on numbers must reference valid steps within same task
13. **Step Sequence**: Step numbers should match array order (first item step=1, second item step=2, etc.)
14. **Step Required Fields**: Each step must have step, title, description, modification_points, logic_flow, depends_on, output
15. **Step Optional Fields**: command field is optional - when omitted, agent executes based on modification_points and logic_flow

## Workflow Structure

### Unified File Structure
All workflows use the same file structure definition regardless of complexity. **Directories and files are created on-demand as needed**, not all at once during initialization.

#### Complete Structure Reference
```
.workflow/
├── [.scratchpad/]              # Non-session-specific outputs (created when needed)
│   ├── analyze-*-[timestamp].md        # One-off analysis results
│   ├── chat-*-[timestamp].md           # Standalone chat sessions
│   ├── plan-*-[timestamp].md           # Ad-hoc planning notes
│   ├── bug-index-*-[timestamp].md      # Quick bug analyses
│   ├── code-analysis-*-[timestamp].md  # Standalone code analysis
│   ├── execute-*-[timestamp].md        # Ad-hoc implementation logs
│   └── codex-execute-*-[timestamp].md  # Multi-stage execution logs
│
├── [design-run-*/]             # Standalone UI design outputs (created when needed)
│   └── (timestamped)/          # Timestamped design runs without session
│       ├── .intermediates/     # Intermediate analysis files
│       │   ├── style-analysis/ # Style analysis data
│       │   │   ├── computed-styles.json        # Extracted CSS values
│       │   │   └── design-space-analysis.json  # Design directions
│       │   └── layout-analysis/ # Layout analysis data
│       │       ├── dom-structure-{target}.json # DOM extraction
│       │       └── inspirations/               # Layout research
│       │           └── {target}-layout-ideas.txt
│       ├── style-extraction/   # Final design systems
│       │   ├── style-1/        # design-tokens.json, style-guide.md
│       │   └── style-N/
│       ├── layout-extraction/  # Layout templates
│       │   └── layout-templates.json
│       ├── prototypes/         # Generated HTML/CSS prototypes
│       │   ├── {target}-style-{s}-layout-{l}.html  # Final prototypes
│       │   ├── compare.html    # Interactive matrix view
│       │   └── index.html      # Navigation page
│       └── .run-metadata.json  # Run configuration
│
├── active/                          # Active workflow sessions
│   └── WFS-[topic-slug]/
│       ├── workflow-session.json        # Session metadata and state (REQUIRED)
│       ├── [.brainstorming/]           # Optional brainstorming phase (created when needed)
│       ├── [.chat/]                    # CLI interaction sessions (created when analysis is run)
│       │   ├── chat-*.md              # Saved chat sessions
│       │   └── analysis-*.md          # Analysis results
│       ├── [.process/]                 # Planning analysis results (created by /workflow:plan)
│       │   └── ANALYSIS_RESULTS.md    # Analysis results and planning artifacts
│       ├── IMPL_PLAN.md                # Planning document (REQUIRED)
│       ├── TODO_LIST.md                # Progress tracking (REQUIRED)
│       ├── [.summaries/]               # Task completion summaries (created when tasks complete)
│       │   ├── IMPL-*-summary.md      # Main task summaries
│       │   └── IMPL-*.*-summary.md    # Subtask summaries
│       ├── [.review/]                  # Code review results (created by review commands)
│       │   ├── review-metadata.json    # Review configuration and scope
│       │   ├── review-state.json       # Review state machine
│       │   ├── review-progress.json    # Real-time progress tracking
│       │   ├── dimensions/             # Per-dimension analysis results
│       │   ├── iterations/             # Deep-dive iteration results
│       │   ├── reports/                # Human-readable reports and CLI outputs
│       │   ├── REVIEW-SUMMARY.md       # Final consolidated summary
│       │   └── dashboard.html          # Interactive review dashboard
│       ├── [design-*/]                 # UI design outputs (created by ui-design workflows)
│       │   ├── .intermediates/         # Intermediate analysis files
│       │   │   ├── style-analysis/     # Style analysis data
│       │   │   │   ├── computed-styles.json        # Extracted CSS values
│       │   │   │   └── design-space-analysis.json  # Design directions
│       │   │   └── layout-analysis/    # Layout analysis data
│       │   │       ├── dom-structure-{target}.json # DOM extraction
│       │   │       └── inspirations/               # Layout research
│       │   │           └── {target}-layout-ideas.txt
│       │   ├── style-extraction/       # Final design systems
│       │   │   ├── style-1/            # design-tokens.json, style-guide.md
│       │   │   └── style-N/
│       │   ├── layout-extraction/      # Layout templates
│       │   │   └── layout-templates.json
│       │   ├── prototypes/             # Generated HTML/CSS prototypes
│       │   │   ├── {target}-style-{s}-layout-{l}.html  # Final prototypes
│       │   │   ├── compare.html        # Interactive matrix view
│       │   │   └── index.html          # Navigation page
│       │   └── .run-metadata.json      # Run configuration
│       └── .task/                      # Task definitions (REQUIRED)
│           ├── IMPL-*.json             # Main task definitions
│           └── IMPL-*.*.json           # Subtask definitions (created dynamically)
└── archives/                       # Completed workflow sessions
    └── WFS-[completed-topic]/      # Archived session directories
```

#### Creation Strategy
- **Initial Setup**: Create only `workflow-session.json`, `IMPL_PLAN.md`, `TODO_LIST.md`, and `.task/` directory
- **On-Demand Creation**: Other directories created when first needed
- **Dynamic Files**: Subtask JSON files created during task decomposition
- **Scratchpad Usage**: `.scratchpad/` created when CLI commands run without active session
- **Design Usage**: `design-{timestamp}/` created by UI design workflows in `.workflow/` directly for standalone design runs
- **Review Usage**: `.review/` created by review commands (`/workflow:review-module-cycle`, `/workflow:review-session-cycle`) for comprehensive code quality analysis
- **Intermediate Files**: `.intermediates/` contains analysis data (style/layout) separate from final deliverables
- **Layout Templates**: `layout-extraction/layout-templates.json` contains structural templates for UI assembly

#### Scratchpad Directory (.scratchpad/)
**Purpose**: Centralized location for non-session-specific CLI outputs

**When to Use**:
1. **No Active Session**: CLI analysis/chat commands run without an active workflow session
2. **Unrelated Analysis**: Quick analysis not related to current active session
3. **Exploratory Work**: Ad-hoc investigation before creating formal workflow
4. **One-Off Queries**: Standalone questions or debugging without workflow context

**Output Routing Logic**:
- **IF** active session exists in `.workflow/active/` AND command is session-relevant:
  - Save to `.workflow/active/WFS-[id]/.chat/[command]-[timestamp].md`
- **ELSE** (no session OR one-off analysis):
  - Save to `.workflow/.scratchpad/[command]-[description]-[timestamp].md`

**File Naming Pattern**: `[command-type]-[brief-description]-[timestamp].md`

**Examples**:

*Workflow Commands (lightweight):*
- `/workflow:lite-plan "feature idea"` (exploratory) → `.scratchpad/lite-plan-feature-idea-20250105-143110.md`
- `/workflow:lite-fix "bug description"` (bug fixing) → `.scratchpad/lite-fix-bug-20250105-143130.md`

> **Note**: Direct CLI commands (`/cli:analyze`, `/cli:execute`, etc.) have been replaced by semantic invocation and workflow commands.

**Maintenance**:
- Periodically review and clean up old scratchpad files
- Promote useful analyses to formal workflow sessions if needed
- No automatic cleanup - manual management recommended

### File Naming Conventions

#### Session Identifiers
**Format**: `WFS-[topic-slug]`

**WFS Prefix Meaning**:
- `WFS` = **W**ork**F**low **S**ession
- Identifies directories as workflow session containers
- Distinguishes workflow sessions from other project directories

**Naming Rules**:
- Convert topic to lowercase with hyphens (e.g., "User Auth System" → `WFS-user-auth-system`)
- Add `-NNN` suffix only if conflicts exist (e.g., `WFS-payment-integration-002`)
- Maximum length: 50 characters including WFS- prefix

#### Document Naming
- `workflow-session.json` - Session state (required)
- `IMPL_PLAN.md` - Planning document (required)
- `TODO_LIST.md` - Progress tracking (auto-generated when needed)
- Chat sessions: `chat-analysis-*.md`
- Task summaries: `IMPL-[task-id]-summary.md`

### Document Templates

#### TODO_LIST.md Template
```markdown
# Tasks: [Session Topic]

## Task Progress
▸ **IMPL-001**: [Main Task Group] → [📋](./.task/IMPL-001.json)
  - [ ] **IMPL-001.1**: [Subtask] → [📋](./.task/IMPL-001.1.json)
  - [x] **IMPL-001.2**: [Subtask] → [📋](./.task/IMPL-001.2.json) | [✅](./.summaries/IMPL-001.2-summary.md)

- [x] **IMPL-002**: [Simple Task] → [📋](./.task/IMPL-002.json) | [✅](./.summaries/IMPL-002-summary.md)

## Status Legend
- `▸` = Container task (has subtasks)
- `- [ ]` = Pending leaf task
- `- [x]` = Completed leaf task
- Maximum 2 levels: Main tasks and subtasks only
```

## Operations Guide

### Session Management
```bash
# Create minimal required structure
mkdir -p .workflow/active/WFS-topic-slug/.task
echo '{"session_id":"WFS-topic-slug",...}' > .workflow/active/WFS-topic-slug/workflow-session.json
echo '# Implementation Plan' > .workflow/active/WFS-topic-slug/IMPL_PLAN.md
echo '# Tasks' > .workflow/active/WFS-topic-slug/TODO_LIST.md
```

### Task Operations
```bash
# Create task
echo '{"id":"IMPL-1","title":"New task",...}' > .task/IMPL-1.json

# Update task status
jq '.status = "active"' .task/IMPL-1.json > temp && mv temp .task/IMPL-1.json

# Generate TODO list from JSON state
generate_todo_list_from_json .task/
```

### Directory Creation (On-Demand)
```bash
mkdir -p .brainstorming     # When brainstorming is initiated
mkdir -p .chat              # When analysis commands are run
mkdir -p .summaries         # When first task completes
```

### Session Consistency Checks & Recovery
```bash
# Validate session directory structure
if [ -d ".workflow/active/" ]; then
  for session_dir in .workflow/active/WFS-*; do
    if [ ! -f "$session_dir/workflow-session.json" ]; then
      echo "⚠️ Missing workflow-session.json in $session_dir"
    fi
  done
fi
```

**Recovery Strategies**:
- **Missing Session File**: Recreate workflow-session.json from template
- **Corrupted Session File**: Restore from template with basic metadata
- **Broken Task Hierarchy**: Reconstruct parent-child relationships from task JSON files
- **Orphaned Sessions**: Move incomplete sessions to archives/

## Complexity Classification

### Task Complexity Rules
**Complexity is determined by task count and decomposition needs:**

| Complexity | Task Count | Hierarchy Depth | Decomposition Behavior |
|------------|------------|----------------|----------------------|
| **Simple** | <5 tasks | 1 level (IMPL-N) | Direct execution, minimal decomposition |
| **Medium** | 5-15 tasks | 2 levels (IMPL-N.M) | Moderate decomposition, context coordination |
| **Complex** | >15 tasks | 2 levels (IMPL-N.M) | Frequent decomposition, multi-agent orchestration |

### Workflow Characteristics & Tool Guidance

#### Simple Workflows
- **Examples**: Bug fixes, small feature additions, configuration changes
- **Task Decomposition**: Usually single-level tasks, minimal breakdown needed
- **Agent Coordination**: Direct execution without complex orchestration
- **Tool Strategy**: `bash()` commands, `grep()` for pattern matching

#### Medium Workflows
- **Examples**: New features, API endpoints with integration, database schema changes
- **Task Decomposition**: Two-level hierarchy when decomposition is needed
- **Agent Coordination**: Context coordination between related tasks
- **Tool Strategy**: `gemini` for pattern analysis, `codex --full-auto` for implementation

#### Complex Workflows
- **Examples**: Major features, architecture refactoring, security implementations, multi-service deployments
- **Task Decomposition**: Frequent use of two-level hierarchy with dynamic subtask creation
- **Agent Coordination**: Multi-agent orchestration with deep context analysis
- **Tool Strategy**: `gemini` for architecture analysis, `codex --full-auto` for complex problem solving, `bash()` commands for flexible analysis

### Assessment & Upgrades
- **During Creation**: System evaluates requirements and assigns complexity
- **During Execution**: Can upgrade (Simple→Medium→Complex) but never downgrade
- **Override Allowed**: Users can specify higher complexity manually

## Agent Integration

### Agent Assignment
Based on task type and title keywords:
- **Planning tasks** → @action-planning-agent
- **Implementation** → @code-developer (code + tests)
- **Test execution/fixing** → @test-fix-agent
- **Review** → @universal-executor (optional, only when explicitly requested)

### Execution Context
Agents receive complete task JSON plus workflow context:
```json
{
  "task": { /* complete task JSON */ },
  "workflow": {
    "session": "WFS-user-auth",
    "phase": "IMPLEMENT"
  }
}
```

