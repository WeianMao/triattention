---
name: execute
description: Execute task JSON using appropriate agent (@doc-generator/@implementation-agent/@test-agent) with pre-analysis context loading and status tracking
argument-hint: "task-id"
---

## Command Overview: /task:execute

**Purpose**: Executes tasks using intelligent agent selection, context preparation, and progress tracking.


## Execution Modes

-   **auto (Default)**
    -   Fully autonomous execution with automatic agent selection.
    -   Provides progress updates at each checkpoint.
    -   Automatically completes the task when done.
-   **guided**
    -   Executes step-by-step, requiring user confirmation at each checkpoint.
    -   Allows for dynamic adjustments and manual review during the process.
-   **review**
    -   Optional manual review using `@universal-executor`.
    -   Used only when explicitly requested by user.

## Agent Selection Logic

The system determines the appropriate agent for a task using the following logic.

```pseudo
FUNCTION select_agent(task, agent_override):
    // A manual override always takes precedence.
    // Corresponds to the --agent=<agent-type> flag.
    IF agent_override IS NOT NULL:
        RETURN agent_override

    // If no override, select based on keywords in the task title.
    ELSE:
        CASE task.title:
            WHEN CONTAINS "Build API", "Implement":
                RETURN "@code-developer"
            WHEN CONTAINS "Design schema", "Plan":
                RETURN "@planning-agent"
            WHEN CONTAINS "Write tests", "Generate tests":
                RETURN "@code-developer" // type: test-gen
            WHEN CONTAINS "Execute tests", "Fix tests", "Validate":
                RETURN "@test-fix-agent" // type: test-fix
            WHEN CONTAINS "Review code":
                RETURN "@universal-executor" // Optional manual review
            DEFAULT:
                RETURN "@code-developer" // Default agent
        END CASE
END FUNCTION
```

## Core Execution Protocol

`Pre-Execution` -> `Execution` -> `Post-Execution`

### Pre-Execution Protocol

`Validate Task & Dependencies` **->** `Prepare Execution Context` **->** `Coordinate with TodoWrite`

-   **Validation**: Checks for the task's JSON file in `.task/` and resolves its dependencies.
-   **Context Preparation**: Loads task and workflow context, preparing it for the selected agent.
-   **Session Context Injection**: Provides workflow directory paths to agents for TODO_LIST.md and summary management.
-   **TodoWrite Coordination**: Generates execution Todos and checkpoints, syncing with `TODO_LIST.md`.

### Post-Execution Protocol

`Update Task Status` **->** `Generate Summary` **->** `Save Artifacts` **->** `Sync All Progress` **->** `Validate File Integrity`

-   Updates status in the task's JSON file and `TODO_LIST.md`.
-   Creates a summary in `.summaries/`.
-   Stores outputs and syncs progress across the entire workflow session.

### Task & Subtask Execution Logic

This logic defines how single, multiple, or parent tasks are handled.

```pseudo
FUNCTION execute_task_command(task_id, mode, parallel_flag):
    // Handle parent tasks by executing their subtasks.
    IF is_parent_task(task_id):
        subtasks = get_subtasks(task_id)
        EXECUTE_SUBTASK_BATCH(subtasks, mode)

    // Handle wildcard execution (e.g., IMPL-001.*)
    ELSE IF task_id CONTAINS "*":
        subtasks = find_matching_tasks(task_id)
        IF parallel_flag IS true:
            EXECUTE_IN_PARALLEL(subtasks)
        ELSE:
            FOR each subtask in subtasks:
                EXECUTE_SINGLE_TASK(subtask, mode)
  
    // Default case for a single task ID.
    ELSE:
        EXECUTE_SINGLE_TASK(task_id, mode)
END FUNCTION
```

### Error Handling & Recovery Logic

```pseudo
FUNCTION pre_execution_check(task):
    // Ensure dependencies are met before starting.
    IF task.dependencies ARE NOT MET:
        LOG_ERROR("Cannot execute " + task.id)
        LOG_INFO("Blocked by: " + unmet_dependencies)
        HALT_EXECUTION()

FUNCTION on_execution_failure(checkpoint):
    // Provide user with recovery options upon failure.
    LOG_WARNING("Execution failed at checkpoint " + checkpoint)
    PRESENT_OPTIONS([
        "Retry from checkpoint",
        "Retry from beginning",
        "Switch to guided mode",
        "Abort execution"
    ])
    AWAIT user_input
    // System performs the selected action.
END FUNCTION
```


### Simplified Context Structure (JSON)

This is the simplified data structure loaded to provide context for task execution.

```json
{
  "task": {
    "id": "IMPL-1",
    "title": "Build authentication module",
    "type": "feature",
    "status": "active",
    "agent": "code-developer",
    "context": {
      "requirements": ["JWT authentication", "OAuth2 support"],
      "scope": ["src/auth/*", "tests/auth/*"],
      "acceptance": ["Module handles JWT tokens", "OAuth2 flow implemented"],
      "inherited_from": "WFS-user-auth"
    },
    "relations": {
      "parent": null,
      "subtasks": ["IMPL-1.1", "IMPL-1.2"],
      "dependencies": ["IMPL-0"]
    },
    "implementation": {
      "files": [
        {
          "path": "src/auth/login.ts",
          "location": {
            "function": "authenticateUser",
            "lines": "25-65",
            "description": "Main authentication logic"
          },
          "original_code": "// Code snippet extracted via gemini analysis",
          "modifications": {
            "current_state": "Basic password authentication only",
            "proposed_changes": [
              "Add JWT token generation",
              "Implement OAuth2 callback handling",
              "Add multi-factor authentication support"
            ],
            "logic_flow": [
              "validateCredentials() ───► checkUserExists()",
              "◊─── if password ───► generateJWT() ───► return token",
              "◊─── if OAuth ───► validateOAuthCode() ───► exchangeForToken()",
              "◊─── if MFA ───► sendMFACode() ───► awaitVerification()"
            ],
            "reason": "Support modern authentication standards and security requirements",
            "expected_outcome": "Comprehensive authentication system supporting multiple methods"
          }
        }
      ],
      "context_notes": {
        "dependencies": ["jsonwebtoken", "passport", "speakeasy"],
        "affected_modules": ["user-session", "auth-middleware", "api-routes"],
        "risks": [
          "Breaking changes to existing login endpoints",
          "Token storage and rotation complexity",
          "OAuth provider configuration dependencies"
        ],
        "performance_considerations": "JWT validation adds ~10ms per request, OAuth callbacks may timeout",
        "error_handling": "Ensure sensitive authentication errors don't leak user enumeration data"
      },
      "pre_analysis": [
        {
          "action": "analyze patterns",
          "template": "~/.claude/workflows/cli-templates/prompts/analysis/02-analyze-code-patterns.txt",
          "method": "gemini"
        }
      ]
    }
  },
  "workflow": {
    "session": "WFS-user-auth",
    "phase": "IMPLEMENT",
    "session_context": {
      "workflow_directory": ".workflow/active/WFS-user-auth/",
      "todo_list_location": ".workflow/active/WFS-user-auth/TODO_LIST.md",
      "summaries_directory": ".workflow/active/WFS-user-auth/.summaries/",
      "task_json_location": ".workflow/active/WFS-user-auth/.task/"
    }
  },
  "execution": {
    "agent": "code-developer",
    "mode": "auto",
    "attempts": 0
  }
}
```

### Agent-Specific Context

Different agents receive context tailored to their function, including implementation details:

**`@code-developer`**: 
- Complete implementation.files array with file paths and locations
- original_code snippets and proposed_changes for precise modifications
- logic_flow diagrams for understanding data flow
- Dependencies and affected modules for integration planning
- Performance and error handling considerations

**`@planning-agent`**: 
- High-level requirements, constraints, success criteria
- Implementation risks and mitigation strategies
- Architecture implications from implementation.context_notes

**`@test-fix-agent`**:
- Test files to execute from task.context.focus_paths
- Source files to fix from implementation.files[].path
- Expected behaviors from implementation.modifications.logic_flow
- Error conditions to validate from implementation.context_notes.error_handling
- Performance requirements from implementation.context_notes.performance_considerations

**`@universal-executor`**:
- Used for optional manual reviews when explicitly requested
- Code quality standards and implementation patterns
- Security considerations from implementation.context_notes.risks
- Dependency validation from implementation.context_notes.dependencies
- Architecture compliance checks

### Simplified File Output

-   **Task JSON File (`.task/<task-id>.json`)**: Updated with status and last attempt time only.
-   **Session File (`workflow-session.json`)**: Updated task stats (completed count).
-   **Summary File**: Generated in `.summaries/` upon completion (optional).

### Simplified Summary Template

Optional summary file generated at `.summaries/IMPL-[task-id]-summary.md`.

```markdown
# Task Summary: IMPL-1 Build Authentication Module

## What Was Done
- Created src/auth/login.ts with JWT validation
- Added tests in tests/auth.test.ts

## Execution Results
- **Agent**: code-developer
- **Status**: completed

## Files Modified
- `src/auth/login.ts` (created)
- `tests/auth.test.ts` (created)
```
