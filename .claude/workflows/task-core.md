# Task System Core Reference

## Overview
Task commands provide single-execution workflow capabilities with full context awareness, hierarchical organization, and agent orchestration.

## Task JSON Schema
All task files use this simplified 5-field schema:

```json
{
  "id": "IMPL-1.2",
  "title": "Implement JWT authentication",
  "status": "pending|active|completed|blocked|container",

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
    }
  },

  "flow_control": {
    "pre_analysis": [
      {
        "step": "gather_context",
        "action": "Read dependency summaries",
        "command": "bash(cat .workflow/*/summaries/IMPL-1.1-summary.md)",
        "output_to": "auth_design_context",
        "on_error": "skip_optional"
      }
    ],
    "implementation_approach": [
      {
        "step": 1,
        "title": "Implement JWT authentication system",
        "description": "Implement comprehensive JWT authentication system with token generation, validation, and refresh logic",
        "modification_points": ["Add JWT token generation", "Implement token validation middleware", "Create refresh token logic"],
        "logic_flow": ["User login request → validate credentials", "Generate JWT access and refresh tokens", "Store refresh token securely", "Return tokens to client"],
        "depends_on": [],
        "output": "jwt_implementation"
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

## Field Structure Details

### focus_paths Field (within context)
**Purpose**: Specifies concrete project paths relevant to task implementation

**Format**:
- **Array of strings**: `["folder1", "folder2", "specific_file.ts"]`
- **Concrete paths**: Use actual directory/file names without wildcards
- **Mixed types**: Can include both directories and specific files
- **Relative paths**: From project root (e.g., `src/auth`, not `./src/auth`)

**Examples**:
```json
// Authentication system task
"focus_paths": ["src/auth", "tests/auth", "config/auth.json", "src/middleware/auth.ts"]

// UI component task
"focus_paths": ["src/components/Button", "src/styles", "tests/components"]
```

### flow_control Field Structure
**Purpose**: Universal process manager for task execution

**Components**:
- **pre_analysis**: Array of sequential process steps
- **implementation_approach**: Task execution strategy
- **target_files**: Files to modify/create - existing files in `file:function:lines` format, new files as `file` only

**Step Structure**:
```json
{
  "step": "gather_context",
  "action": "Human-readable description",
  "command": "bash(executable command with [variables])",
  "output_to": "variable_name",
  "on_error": "skip_optional|fail|retry_once|manual_intervention"
}
```

## Hierarchical System

### Task Hierarchy Rules
- **Format**: IMPL-N (main), IMPL-N.M (subtasks) - uppercase required
- **Maximum Depth**: 2 levels only
- **10-Task Limit**: Hard limit enforced across all tasks
- **Container Tasks**: Parents with subtasks (not executable)
- **Leaf Tasks**: No subtasks (executable)
- **File Cohesion**: Related files must stay in same task

### Task Complexity Classifications
- **Simple**: ≤5 tasks, single-level tasks, direct execution
- **Medium**: 6-10 tasks, two-level hierarchy, context coordination
- **Over-scope**: >10 tasks requires project re-scoping into iterations

### Complexity Assessment Rules
- **Creation**: System evaluates and assigns complexity
- **10-task limit**: Hard limit enforced - exceeding requires re-scoping
- **Execution**: Can upgrade (Simple→Medium→Over-scope), triggers re-scoping
- **Override**: Users can manually specify complexity within 10-task limit

### Status Rules
- **pending**: Ready for execution
- **active**: Currently being executed
- **completed**: Successfully finished
- **blocked**: Waiting for dependencies
- **container**: Has subtasks (parent only)

## Session Integration

### Active Session Detection
```bash
# Check for active session in sessions directory
active_session=$(find .workflow/active/ -name 'WFS-*' -type d 2>/dev/null | head -1)
```

### Workflow Context Inheritance
Tasks inherit from:
1. `workflow-session.json` - Session metadata
2. Parent task context (for subtasks)
3. `IMPL_PLAN.md` - Planning document

### File Locations
- **Task JSON**: `.workflow/active/WFS-[topic]/.task/IMPL-*.json` (uppercase required)
- **Session State**: `.workflow/active/WFS-[topic]/workflow-session.json`
- **Planning Doc**: `.workflow/active/WFS-[topic]/IMPL_PLAN.md`
- **Progress**: `.workflow/active/WFS-[topic]/TODO_LIST.md`

## Agent Mapping

### Automatic Agent Selection
- **@code-developer**: Implementation tasks, coding, test writing
- **@action-planning-agent**: Design, architecture planning
- **@test-fix-agent**: Test execution, failure diagnosis, code fixing
- **@universal-executor**: Optional manual review (only when explicitly requested)

### Agent Context Filtering
Each agent receives tailored context:
- **@code-developer**: Complete implementation details, test requirements
- **@action-planning-agent**: High-level requirements, risks, architecture
- **@test-fix-agent**: Test execution, failure diagnosis, code fixing
- **@universal-executor**: Quality standards, security considerations (when requested)

## Deprecated Fields

### Legacy paths Field
**Deprecated**: The semicolon-separated `paths` field has been replaced by `context.focus_paths` array.

**Old Format** (no longer used):
```json
"paths": "src/auth;tests/auth;config/auth.json;src/middleware/auth.ts"
```

**New Format** (use this instead):
```json
"context": {
  "focus_paths": ["src/auth", "tests/auth", "config/auth.json", "src/middleware/auth.ts"]
}
```

## Validation Rules

### Pre-execution Checks
1. Task exists and is valid JSON
2. Task status allows operation
3. Dependencies are met
4. Active workflow session exists
5. All 5 core fields present (id, title, status, meta, context, flow_control)
6. Total task count ≤ 10 (hard limit)
7. File cohesion maintained in focus_paths

### Hierarchy Validation
- Parent-child relationships valid
- Maximum depth not exceeded
- Container tasks have subtasks
- No circular dependencies

## Error Handling Patterns

### Common Errors
- **Task not found**: Check ID format and session
- **Invalid status**: Verify task can be operated on
- **Missing session**: Ensure active workflow exists
- **Max depth exceeded**: Restructure hierarchy
- **Missing implementation**: Complete required fields

### Recovery Strategies
- Session validation with clear guidance
- Automatic ID correction suggestions
- Implementation field completion prompts
- Hierarchy restructuring options