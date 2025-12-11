---
name: replan
description: Update task JSON with new requirements or batch-update multiple tasks from verification report, tracks changes in task-changes.json
argument-hint: "task-id [\"text\"|file.md] | --batch [verification-report.md]"
allowed-tools: Read(*), Write(*), Edit(*), TodoWrite(*), Glob(*), Bash(*)
---

# Task Replan Command (/task:replan)

> **⚠️ DEPRECATION NOTICE**: This command is maintained for backward compatibility. For new workflows, use `/workflow:replan` which provides:
> - Session-level replanning with comprehensive artifact updates
> - Interactive boundary clarification
> - Updates to IMPL_PLAN.md, TODO_LIST.md, and session metadata
> - Better integration with workflow sessions
>
> **Migration**: Replace `/task:replan IMPL-1 "changes"` with `/workflow:replan IMPL-1 "changes"`

## Overview
Replans individual tasks or batch processes multiple tasks with change tracking and backup management.

**Modes**:
- **Single Task Mode**: Replan one task with specific changes
- **Batch Mode**: Process multiple tasks from action-plan verification report

## Key Features
- **Single/Batch Operations**: Single task or multiple tasks from verification report
- **Multiple Input Sources**: Text, files, or verification report
- **Backup Management**: Automatic backup of previous versions
- **Change Documentation**: Track all modifications
- **Progress Tracking**: TodoWrite integration for batch operations

**CRITICAL**: Validates active session before replanning

## Operation Modes

### Single Task Mode

#### Direct Text (Default)
```bash
/task:replan IMPL-1 "Add OAuth2 authentication support"
```

#### File-based Input
```bash
/task:replan IMPL-1 updated-specs.md
```
Supports: .md, .txt, .json, .yaml

#### Interactive Mode
```bash
/task:replan IMPL-1 --interactive
```
Guided step-by-step modification process with validation

### Batch Mode

#### From Verification Report
```bash
/task:replan --batch ACTION_PLAN_VERIFICATION.md
```

**Workflow**:
1. Parse verification report to extract replan recommendations
2. Create TodoWrite task list for all modifications
3. Process each task sequentially with confirmation
4. Track progress and generate summary report

**Auto-detection**: If input file contains "Action Plan Verification Report" header, automatically enters batch mode

## Replanning Process

### Single Task Process

1. **Load & Validate**: Read task JSON and validate session
2. **Parse Input**: Process changes from input source
3. **Create Backup**: Save previous version to backup folder
4. **Update Task**: Modify JSON structure and relationships
5. **Save Changes**: Write updated task and increment version
6. **Update Session**: Reflect changes in workflow stats

### Batch Process

1. **Parse Verification Report**: Extract all replan recommendations
2. **Initialize TodoWrite**: Create task list for tracking
3. **For Each Task**:
   - Mark todo as in_progress
   - Load and validate task JSON
   - Create backup
   - Apply recommended changes
   - Save updated task
   - Mark todo as completed
4. **Generate Summary**: Report all changes and backup locations

## Backup Management

### Backup Tracking
Tasks maintain backup history:
```json
{
  "id": "IMPL-1",
  "version": "1.2",
  "replan_history": [
    {
      "version": "1.2",
      "reason": "Add OAuth2 support",
      "input_source": "direct_text",
      "backup_location": ".task/backup/IMPL-1-v1.1.json",
      "timestamp": "2025-10-17T10:30:00Z"
    }
  ]
}
```

**Complete schema**: See @~/.claude/workflows/task-core.md

### File Structure
```
.task/
├── IMPL-1.json                    # Current version
├── backup/
│   ├── IMPL-1-v1.0.json          # Original version
│   ├── IMPL-1-v1.1.json          # Previous backup
│   └── IMPL-1-v1.2.json          # Latest backup
└── [new subtasks as needed]
```

**Backup Naming**: `{task-id}-v{version}.json`

## Implementation Updates

### Change Detection
Tracks modifications to:
- Files in implementation.files array
- Dependencies and affected modules
- Risk assessments and performance notes
- Logic flows and code locations

### Analysis Triggers
May require gemini re-analysis when:
- New files need code extraction
- Function locations change
- Dependencies require re-evaluation

## Document Updates

### Planning Document
May update IMPL_PLAN.md sections when task structure changes significantly

### TODO List Sync
If TODO_LIST.md exists, synchronizes:
- New subtasks (with [ ] checkbox)
- Modified tasks (marked as updated)
- Removed subtasks (deleted from list)

## Change Documentation

### Change Summary
Generates brief change log with:
- Version increment (1.1 → 1.2)
- Input source and reason
- Key modifications made
- Files updated/created
- Backup location

## Session Updates

Updates workflow-session.json with:
- Modified task tracking
- Task count changes (if subtasks added/removed)
- Last modification timestamps

## Rollback Support

```bash
/task:replan IMPL-1 --rollback v1.1

Rollback to version 1.1:
- Restore task from backup/.../IMPL-1-v1.1.json
- Remove new subtasks if any
- Update session stats

# Use AskUserQuestion for confirmation
AskUserQuestion({
  questions: [{
    question: "Are you sure you want to roll back this task to a previous version?",
    header: "Confirm",
    options: [
      { label: "Yes, rollback", description: "Restore the task from the selected backup." },
      { label: "No, cancel", description: "Keep the current version of the task." }
    ],
    multiSelect: false
  }]
})

User selected: "Yes, rollback"

Task rolled back to version 1.1
```

## Batch Processing with TodoWrite

### Progress Tracking
When processing multiple tasks, automatically creates TodoWrite task list:

```markdown
**Batch Replan Progress**:
- [x] IMPL-002: Add FR-12 draft saving acceptance criteria
- [x] IMPL-003: Add FR-14 history tracking acceptance criteria
- [ ] IMPL-004: Add FR-09 response surface explicit coverage
- [ ] IMPL-008: Add NFR performance validation steps
```

### Batch Report
After completion, generates summary:
```markdown
## Batch Replan Summary

**Total Tasks**: 4
**Successful**: 3
**Failed**: 1
**Skipped**: 0

### Changes Made
- IMPL-002 v1.0 → v1.1: Added FR-12 acceptance criteria
- IMPL-003 v1.0 → v1.1: Added FR-14 acceptance criteria
- IMPL-004 v1.0 → v1.1: Added FR-09 explicit coverage

### Backups Created
- .task/backup/IMPL-002-v1.0.json
- .task/backup/IMPL-003-v1.0.json
- .task/backup/IMPL-004-v1.0.json

### Errors
- IMPL-008: File not found (task may have been renamed)
```

## Examples

### Single Task - Text Input
```bash
/task:replan IMPL-1 "Add OAuth2 authentication support"

Processing changes...
Proposed updates:
+ Add OAuth2 integration
+ Update authentication flow

# Use AskUserQuestion for confirmation
AskUserQuestion({
  questions: [{
    question: "Do you want to apply these changes to the task?",
    header: "Apply",
    options: [
      { label: "Yes, apply", description: "Create new version with these changes." },
      { label: "No, cancel", description: "Discard changes and keep current version." }
    ],
    multiSelect: false
  }]
})

User selected: "Yes, apply"

Version 1.2 created
Context updated
Backup saved to .task/backup/IMPL-1-v1.1.json
```

### Single Task - File Input
```bash
/task:replan IMPL-2 requirements.md

Loading requirements.md...
Applying specification changes...

Task updated with new requirements
Version 1.1 created
Backup saved to .task/backup/IMPL-2-v1.0.json
```

### Batch Mode - From Verification Report
```bash
/task:replan --batch .workflow/active/WFS-{session}/.process/ACTION_PLAN_VERIFICATION.md

Parsing verification report...
Found 4 tasks requiring replanning:
- IMPL-002: Add FR-12 draft saving acceptance criteria
- IMPL-003: Add FR-14 history tracking acceptance criteria
- IMPL-004: Add FR-09 response surface explicit coverage
- IMPL-008: Add NFR performance validation steps

Creating task tracking list...

Processing IMPL-002...
Backup created: .task/backup/IMPL-002-v1.0.json
Updated to v1.1

Processing IMPL-003...
Backup created: .task/backup/IMPL-003-v1.0.json
Updated to v1.1

Processing IMPL-004...
Backup created: .task/backup/IMPL-004-v1.0.json
Updated to v1.1

Processing IMPL-008...
Backup created: .task/backup/IMPL-008-v1.0.json
Updated to v1.1

Batch replan completed: 4/4 successful
Summary report saved
```

### Batch Mode - Auto-detection
```bash
# If file contains "Action Plan Verification Report", auto-enters batch mode
/task:replan ACTION_PLAN_VERIFICATION.md

Detected verification report format
Entering batch mode...
[same as above]
```

## Error Handling

### Single Task Errors
```bash
# Task not found
Task IMPL-5 not found
Check task ID with /workflow:status

# Task completed
Task IMPL-1 is completed (cannot replan)
Create new task for additional work

# File not found
File requirements.md not found
Check file path

# No input provided
Please specify changes needed
Provide text, file, or verification report
```

### Batch Mode Errors
```bash
# Invalid verification report
File does not contain valid verification report format
Check report structure or use single task mode

# Partial failures
Batch completed with errors: 3/4 successful
Review error details in summary report

# No replan recommendations found
Verification report contains no replan recommendations
Check report content or use /workflow:action-plan-verify first
```

## Batch Mode Integration

### Input Format Expectations
Batch mode parses verification reports looking for:

1. **Required Actions Section**: Commands like `/task:replan IMPL-X "changes"`
2. **Findings Table**: Task IDs with recommendations
3. **Next Actions Section**: Specific replan commands

**Example Patterns**:
```markdown
#### 1. HIGH Priority - Address FR Coverage Gaps
/task:replan IMPL-004 "
Add explicit acceptance criteria:
- FR-09: Response surface 3D visualization
"

#### 2. MEDIUM Priority - Enhance NFR Coverage
/task:replan IMPL-008 "
Add performance testing:
- NFR-01: Load test API endpoints
"
```

### Extraction Logic
1. Scan for `/task:replan` commands in report
2. Extract task ID and change description
3. Group by priority (HIGH, MEDIUM, LOW)
4. Process in priority order with TodoWrite tracking

### Confirmation Behavior
- **Default**: Confirm each task before applying
- **With `--auto-confirm`**: Apply all changes without prompting
  ```bash
  /task:replan --batch report.md --auto-confirm
  ```

## Implementation Details

### Backup Management
```typescript
// Backup file naming convention
const backupPath = `.task/backup/${taskId}-v${previousVersion}.json`;

// Backup metadata in task JSON
{
  "replan_history": [
    {
      "version": "1.2",
      "timestamp": "2025-10-17T10:30:00Z",
      "reason": "Add FR-09 explicit coverage",
      "input_source": "batch_verification_report",
      "backup_location": ".task/backup/IMPL-004-v1.1.json"
    }
  ]
}
```

### TodoWrite Integration
```typescript
// Initialize tracking for batch mode
TodoWrite({
  todos: taskList.map(task => ({
    content: `${task.id}: ${task.changeDescription}`,
    status: "pending",
    activeForm: `Replanning ${task.id}`
  }))
});

// Update progress during processing
TodoWrite({
  todos: updateTaskStatus(taskId, "in_progress")
});

// Mark completed
TodoWrite({
  todos: updateTaskStatus(taskId, "completed")
});
```