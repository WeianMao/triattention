---
name: create
description: Generate task JSON from natural language description with automatic file pattern detection, scope inference, and dependency analysis
argument-hint: "\"task title\""
---

# Task Create Command (/task:create)

## Overview
Creates new implementation tasks with automatic context awareness and ID generation.

## Core Principles
**Task System:** @~/.claude/workflows/task-core.md

## Core Features

### Automatic Behaviors
- **ID Generation**: Auto-generates IMPL-N format (max 2 levels)
- **Context Inheritance**: Inherits from active workflow session
- **JSON Creation**: Creates task JSON in active session
- **Status Setting**: Initial status = "pending"
- **Agent Assignment**: Suggests agent based on task type
- **Session Integration**: Updates workflow session stats

### Context Awareness
- Validates active workflow session exists
- Avoids duplicate task IDs
- Inherits session requirements and scope
- Suggests task relationships

## Usage

### Basic Creation
```bash
/task:create "Build authentication module"
```

Output:
```
Task created: IMPL-1
Title: Build authentication module
Type: feature
Agent: code-developer
Status: pending
```

### Task Types
- `feature` - New functionality (default)
- `bugfix` - Bug fixes
- `refactor` - Code improvements
- `test` - Test implementation
- `docs` - Documentation

## Task Creation Process

1. **Session Validation**: Check active workflow session
2. **ID Generation**: Auto-increment IMPL-N
3. **Context Inheritance**: Load workflow context
4. **Implementation Setup**: Initialize implementation field
5. **Agent Assignment**: Select appropriate agent
6. **File Creation**: Save JSON to .task/ directory
7. **Session Update**: Update workflow stats

**Task Schema**: See @~/.claude/workflows/task-core.md for complete JSON structure

## Implementation Field Setup

### Auto-Population Strategy
- **Detailed info**: Extract from task description and scope
- **Missing info**: Mark `pre_analysis` as multi-step array format for later pre-analysis
- **Basic structure**: Initialize with standard template

### Analysis Triggers
When implementation details incomplete:
```bash
Task requires analysis for implementation details
Suggest running: gemini analysis for file locations and dependencies
```

## File Management

### JSON Task File
- **Location**: `.task/IMPL-[N].json` in active session
- **Content**: Complete task with implementation field
- **Updates**: Session stats only

### Simple Process
1. Validate session and inputs
2. Generate task JSON
3. Update session stats
4. Notify completion

## Context Inheritance

Tasks inherit from:
1. **Active Session** - Requirements and scope from workflow-session.json
2. **Planning Document** - Context from IMPL_PLAN.md
3. **Parent Task** - For subtasks (IMPL-N.M format)

## Agent Assignment

Based on task type and title keywords:
- **Build/Implement** → @code-developer
- **Design/Plan** → @planning-agent
- **Test Generation** → @code-developer (type: "test-gen")
- **Test Execution/Fix** → @test-fix-agent (type: "test-fix")
- **Review/Audit** → @universal-executor (optional, only when explicitly requested)

## Validation Rules

1. **Session Check** - Active workflow session required
2. **Duplicate Check** - Avoid similar task titles
3. **ID Uniqueness** - Auto-increment task IDs
4. **Schema Validation** - Ensure proper JSON structure

## Error Handling

```bash
# No workflow session
No active workflow found
Use: /workflow init "project name"

# Duplicate task
Similar task exists: IMPL-3
Continue anyway? (y/n)

# Max depth exceeded
Cannot create IMPL-1.2.1 (max 2 levels)
Use: IMPL-2 for new main task
```

## Examples

### Feature Task
```bash
/task:create "Implement user authentication"

Created IMPL-1: Implement user authentication
Type: feature
Agent: code-developer
Status: pending
```

### Bug Fix
```bash
/task:create "Fix login validation bug" --type=bugfix

Created IMPL-2: Fix login validation bug
Type: bugfix
Agent: code-developer
Status: pending
```