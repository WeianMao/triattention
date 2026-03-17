---
name: breakdown
description: Decompose complex task into subtasks with dependency mapping, creates child task JSONs with parent references and execution order
argument-hint: "task-id"
---

# Task Breakdown Command (/task:breakdown)

## Overview
Breaks down complex tasks into executable subtasks with context inheritance and agent assignment.

## Core Principles
**File Cohesion:** Related files must stay in same task
**10-Task Limit:** Total tasks cannot exceed 10 (triggers re-scoping)

## Core Features

**CRITICAL**: Manual breakdown with safety controls to prevent file conflicts and task limit violations.

### Breakdown Process
1. **Session Check**: Verify active session contains parent task
2. **Task Validation**: Ensure parent is `pending` status
3. **10-Task Limit Check**: Verify breakdown won't exceed total limit
4. **Manual Decomposition**: User defines subtasks with validation
5. **File Conflict Detection**: Warn if same files appear in multiple subtasks
6. **Similar Function Warning**: Alert if subtasks have overlapping functionality
7. **Context Distribution**: Inherit parent requirements and scope
8. **Agent Assignment**: Auto-assign agents based on subtask type
9. **TODO_LIST Update**: Regenerate TODO_LIST.md with new structure

### Breakdown Rules
- Only `pending` tasks can be broken down
- **Manual breakdown only**: Automated breakdown disabled to prevent violations
- Parent becomes `container` status (not executable)
- Subtasks use format: IMPL-N.M (max 2 levels)
- Context flows from parent to subtasks
- All relationships tracked in JSON
- **10-task limit enforced**: Breakdown rejected if total would exceed 10 tasks
- **File cohesion preserved**: Same files cannot be split across subtasks

## Usage

### Basic Breakdown
```bash
/task:breakdown impl-1
```

Interactive process:
```
Task: Build authentication module
Current total tasks: 6/10

MANUAL BREAKDOWN REQUIRED
Define subtasks manually (remaining capacity: 4 tasks):

1. Enter subtask title: User authentication core
   Focus files: models/User.js, routes/auth.js, middleware/auth.js

2. Enter subtask title: OAuth integration
   Focus files: services/OAuthService.js, routes/oauth.js

FILE CONFLICT DETECTED:
   - routes/auth.js appears in multiple subtasks
   - Recommendation: Merge related authentication routes

SIMILAR FUNCTIONALITY WARNING:
   - "User authentication" and "OAuth integration" both handle auth
   - Consider combining into single task

# Use AskUserQuestion for confirmation
AskUserQuestion({
  questions: [{
    question: "File conflicts and/or similar functionality detected. How do you want to proceed?",
    header: "Confirm",
    options: [
      { label: "Proceed with breakdown", description: "Accept the risks and create the subtasks as defined." },
      { label: "Restart breakdown", description: "Discard current subtasks and start over." },
      { label: "Cancel breakdown", description: "Abort the operation and leave the parent task as is." }
    ],
    multiSelect: false
  }]
})

User selected: "Proceed with breakdown"

Task IMPL-1 broken down:
IMPL-1: Build authentication module (container)
  ├── IMPL-1.1: User authentication core -> @code-developer
  └── IMPL-1.2: OAuth integration -> @code-developer

Files updated: .task/IMPL-1.json + 2 subtask files + TODO_LIST.md
```

## Decomposition Logic

### Agent Assignment
- **Design/Planning** → `@planning-agent`
- **Implementation** → `@code-developer`
- **Testing** → `@code-developer` (type: "test-gen")
- **Test Validation** → `@test-fix-agent` (type: "test-fix")
- **Review** → `@universal-executor` (optional)

### Context Inheritance
- Subtasks inherit parent requirements
- Scope refined for specific subtask
- Implementation details distributed appropriately

## Safety Controls

### File Conflict Detection
**Validates file cohesion across subtasks:**
- Scans `focus_paths` in all subtasks
- Warns if same file appears in multiple subtasks
- Suggests merging subtasks with overlapping files
- Blocks breakdown if critical conflicts detected

### Similar Functionality Detection
**Prevents functional overlap:**
- Analyzes subtask titles for similar keywords
- Warns about potential functional redundancy
- Suggests consolidation of related functionality
- Examples: "user auth" + "login system" → merge recommendation

### 10-Task Limit Enforcement
**Hard limit compliance:**
- Counts current total tasks in session
- Calculates breakdown impact on total
- Rejects breakdown if would exceed 10 tasks
- Suggests re-scoping if limit reached

### Manual Control Requirements
**User-driven breakdown only:**
- No automatic subtask generation
- User must define each subtask title and scope
- Real-time validation during input
- Confirmation required before execution

## Implementation Details

- Complete task JSON schema
- Implementation field structure
- Context inheritance rules
- Agent assignment logic

## Validation

### Pre-breakdown Checks
1. Active session exists
2. Task found in session
3. Task status is `pending`
4. Not already broken down
5. **10-task limit compliance**: Total tasks + new subtasks ≤ 10
6. **Manual mode enabled**: No automatic breakdown allowed

### Post-breakdown Actions
1. Update parent to `container` status
2. Create subtask JSON files
3. Update parent subtasks list
4. Update session stats
5. **Regenerate TODO_LIST.md** with new hierarchy
6. Validate file paths in focus_paths
7. Update session task count

## Examples

### Basic Breakdown
```bash
/task:breakdown impl-1

impl-1: Build authentication (container)
  ├── impl-1.1: Design schema -> @planning-agent
  ├── impl-1.2: Implement logic + tests -> @code-developer
  └── impl-1.3: Execute & fix tests -> @test-fix-agent
```

## Error Handling

```bash
# Task not found
Task IMPL-5 not found

# Already broken down
Task IMPL-1 already has subtasks

# Wrong status
Cannot breakdown completed task IMPL-2

# 10-task limit exceeded
Breakdown would exceed 10-task limit (current: 8, proposed: 4)
Suggestion: Re-scope project into smaller iterations

# File conflicts detected
File conflict: routes/auth.js appears in IMPL-1.1 and IMPL-1.2
Recommendation: Merge subtasks or redistribute files

# Similar functionality warning
Similar functions detected: "user login" and "authentication"
Consider consolidating related functionality

# Manual breakdown required
Automatic breakdown disabled. Use manual breakdown process.
```

**System ensures**: Manual breakdown control with file cohesion enforcement, similar functionality detection, and 10-task limit compliance