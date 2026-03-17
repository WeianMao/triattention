# Index Structure Reference

Complete documentation for command index files and their data structures.

## Overview

The command-guide skill uses 5 JSON index files to organize and query 69 commands across the Claude DMS3 workflow system.

## Index Files

### 1. `all-commands.json`

**Purpose**: Complete catalog of all commands with full metadata

**Use Cases**:
- Full-text search across all commands
- Detailed command queries
- Batch operations
- Reference lookup

**Structure**:
```json
[
  {
    "name": "workflow:plan",
    "description": "Orchestrate 5-phase planning workflow with quality gate",
    "arguments": "[--agent] [--cli-execute] \"text description\"|file.md",
    "category": "workflow",
    "subcategory": "core",
    "usage_scenario": "planning",
    "difficulty": "Advanced",
    "file_path": "workflow/plan.md"
  },
  ...
]
```

**Fields**:
- `name` (string): Command name (e.g., "workflow:plan")
- `description` (string): Brief functional description
- `arguments` (string): Parameter specification
- `category` (string): Primary category (workflow/cli/memory/task)
- `subcategory` (string|null): Secondary grouping if applicable
- `usage_scenario` (string): Primary use case (planning/implementation/testing/etc.)
- `difficulty` (string): Skill level (Beginner/Intermediate/Advanced)
- `file_path` (string): Relative path to command file

**Total Records**: 69 commands

---

### 2. `by-category.json`

**Purpose**: Hierarchical organization by category and subcategory

**Use Cases**:
- Browse commands by category
- Category-specific listings
- Hierarchical navigation
- Understanding command organization

**Structure**:
```json
{
  "workflow": {
    "core": [
      {
        "name": "workflow:plan",
        "description": "...",
        ...
      }
    ],
    "brainstorm": [...],
    "session": [...],
    "tools": [...],
    "ui-design": [...]
  },
  "cli": {
    "mode": [...],
    "core": [...]
  },
  "memory": [...],
  "task": [...]
}
```

**Category Breakdown**:
- **workflow** (46 commands):
  - core: 11 commands
  - brainstorm: 12 commands
  - session: 4 commands
  - tools: 9 commands
  - ui-design: 10 commands
- **cli** (9 commands):
  - mode: 3 commands
  - core: 6 commands
- **memory** (8 commands)
- **task** (4 commands)
- **general** (2 commands)

---

### 3. `by-use-case.json`

**Purpose**: Commands organized by practical usage scenarios

**Use Cases**:
- Task-oriented command discovery
- "I want to do X" queries
- Workflow planning
- Learning paths

**Structure**:
```json
{
  "planning": [
    {
      "name": "workflow:plan",
      "description": "...",
      ...
    },
    ...
  ],
  "implementation": [...],
  "testing": [...],
  "documentation": [...],
  "session-management": [...],
  "general": [...]
}
```

**Use Case Categories**:
- **planning**: Architecture, task breakdown, design
- **implementation**: Coding, development, execution
- **testing**: Test generation, TDD, quality assurance
- **documentation**: Docs generation, memory management
- **session-management**: Workflow control, resumption
- **general**: Utilities, versioning, prompt enhancement

---

### 4. `essential-commands.json`

**Purpose**: Curated list of 10-15 most frequently used commands

**Use Cases**:
- Quick reference for beginners
- Onboarding new users
- Common workflow starters
- Cheat sheet

**Structure**:
```json
[
  {
    "name": "enhance-prompt",
    "description": "Context-aware prompt enhancement",
    "arguments": "\"user input to enhance\"",
    "category": "general",
    "subcategory": null,
    "usage_scenario": "general",
    "difficulty": "Intermediate",
    "file_path": "enhance-prompt.md"
  },
  ...
]
```

**Selection Criteria**:
- Frequency of use in common workflows
- Value for beginners
- Core functionality coverage
- Minimal overlap in capabilities

**Current Count**: 14 commands

**List**:
1. `enhance-prompt` - Prompt enhancement
2. `version` - Version info
3. `cli:analyze` - Quick codebase analysis
4. `cli:chat` - Direct CLI interaction
5. `cli:execute` - Auto-execution
6. `cli:mode:plan` - Planning mode
7. `task:breakdown` - Task decomposition
8. `task:create` - Create tasks
9. `task:execute` - Execute tasks
10. `workflow:execute` - Run workflows
11. `workflow:plan` - Plan workflows
12. `workflow:review` - Review implementation
13. `workflow:tdd-plan` - TDD planning
14. `workflow:test-gen` - Test generation

---

### 5. `command-relationships.json`

**Purpose**: Mapping of command dependencies and common sequences

**Use Cases**:
- Next-step recommendations
- Workflow pattern suggestions
- Related command discovery
- Smart navigation

**Structure**:
```json
{
  "workflow:plan": {
    "related_commands": [
      "workflow:execute",
      "workflow:action-plan-verify"
    ],
    "next_steps": ["workflow:execute"],
    "prerequisites": []
  },
  "workflow:execute": {
    "related_commands": [
      "workflow:status",
      "workflow:resume",
      "workflow:review"
    ],
    "next_steps": ["workflow:review", "workflow:status"],
    "prerequisites": ["workflow:plan"]
  },
  ...
}
```

**Fields**:
- `related_commands` (array): Commonly used together
- `next_steps` (array): Typical next commands
- `prerequisites` (array): Usually run before this command

**Relationship Types**:
1. **Sequential**: A → B (plan → execute)
2. **Alternatives**: A | B (execute OR codex-execute)
3. **Built-in**: A includes B (plan auto-includes context-gather)

---

## Query Patterns

### Pattern 1: Keyword Search
```javascript
// Search by keyword in name or description
const results = allCommands.filter(cmd =>
  cmd.name.includes(keyword) ||
  cmd.description.toLowerCase().includes(keyword.toLowerCase())
);
```

### Pattern 2: Category Browse
```javascript
// Get all commands in a category
const workflowCommands = byCategory.workflow;
const coreWorkflow = byCategory.workflow.core;
```

### Pattern 3: Use-Case Lookup
```javascript
// Find commands for specific use case
const planningCommands = byUseCase.planning;
```

### Pattern 4: Related Commands
```javascript
// Get next steps after a command
const nextSteps = commandRelationships["workflow:plan"].next_steps;
```

### Pattern 5: Essential Commands
```javascript
// Get beginner-friendly quick reference
const quickStart = essentialCommands;
```

---

## Maintenance

### Regenerating Indexes

When commands are added/modified/removed:

```bash
bash scripts/update-index.sh
```

The script will:
1. Scan all .md files in commands/
2. Extract metadata from YAML frontmatter
3. Analyze command relationships
4. Identify essential commands
5. Generate all 5 index files

### Validation Checklist

After regeneration, verify:
- [ ] All 5 JSON files are valid (no syntax errors)
- [ ] Total command count matches (currently 69)
- [ ] No missing fields in records
- [ ] Category breakdown correct
- [ ] Essential commands reasonable (10-15)
- [ ] Relationships make logical sense

---

## Performance Considerations

**Index Sizes**:
- `all-commands.json`: ~28KB
- `by-category.json`: ~31KB
- `by-use-case.json`: ~29KB
- `command-relationships.json`: ~7KB
- `essential-commands.json`: ~5KB

**Total**: ~100KB (fast to load)

**Query Speed**:
- In-memory search: < 1ms
- File read + parse: < 50ms
- Recommended: Load indexes once, cache in memory

---

**Last Updated**: 2025-01-06
