---
name: lite-execute
description: Execute tasks based on in-memory plan, prompt description, or file content
argument-hint: "[--in-memory] [\"task description\"|file-path]"
allowed-tools: TodoWrite(*), Task(*), Bash(*)
---

# Workflow Lite-Execute Command (/workflow:lite-execute)

## Overview

Flexible task execution command supporting three input modes: in-memory plan (from lite-plan), direct prompt description, or file content. Handles execution orchestration, progress tracking, and optional code review.

**Core capabilities:**
- Multi-mode input (in-memory plan, prompt description, or file path)
- Execution orchestration (Agent or Codex) with full context
- Live progress tracking via TodoWrite at execution call level
- Optional code review with selected tool (Gemini, Agent, or custom)
- Context continuity across multiple executions
- Intelligent format detection (Enhanced Task JSON vs plain text)

## Usage

### Command Syntax
```bash
/workflow:lite-execute [FLAGS] <INPUT>

# Flags
--in-memory                Use plan from memory (called by lite-plan)

# Arguments
<input>                    Task description string, or path to file (required)
```

## Input Modes

### Mode 1: In-Memory Plan

**Trigger**: Called by lite-plan after Phase 4 approval with `--in-memory` flag

**Input Source**: `executionContext` global variable set by lite-plan

**Content**: Complete execution context (see Data Structures section)

**Behavior**:
- Skip execution method selection (already set by lite-plan)
- Directly proceed to execution with full context
- All planning artifacts available (exploration, clarifications, plan)

### Mode 2: Prompt Description

**Trigger**: User calls with task description string

**Input**: Simple task description (e.g., "Add unit tests for auth module")

**Behavior**:
- Store prompt as `originalUserInput`
- Create simple execution plan from prompt
- AskUserQuestion: Select execution method (Agent/Codex/Auto)
- AskUserQuestion: Select code review tool (Skip/Gemini/Agent/Other)
- Proceed to execution with `originalUserInput` included

**User Interaction**:
```javascript
AskUserQuestion({
  questions: [
    {
      question: "Select execution method:",
      header: "Execution",
      multiSelect: false,
      options: [
        { label: "Agent", description: "@code-developer agent" },
        { label: "Codex", description: "codex CLI tool" },
        { label: "Auto", description: "Auto-select based on complexity" }
      ]
    },
    {
      question: "Enable code review after execution?",
      header: "Code Review",
      multiSelect: false,
      options: [
        { label: "Skip", description: "No review" },
        { label: "Gemini Review", description: "Gemini CLI tool" },
        { label: "Agent Review", description: "Current agent review" }
      ]
    }
  ]
})
```

### Mode 3: File Content

**Trigger**: User calls with file path

**Input**: Path to file containing task description or plan.json

**Step 1: Read and Detect Format**

```javascript
fileContent = Read(filePath)

// Attempt JSON parsing
try {
  jsonData = JSON.parse(fileContent)

  // Check if plan.json from lite-plan session
  if (jsonData.summary && jsonData.approach && jsonData.tasks) {
    planObject = jsonData
    originalUserInput = jsonData.summary
    isPlanJson = true
  } else {
    // Valid JSON but not plan.json - treat as plain text
    originalUserInput = fileContent
    isPlanJson = false
  }
} catch {
  // Not valid JSON - treat as plain text prompt
  originalUserInput = fileContent
  isPlanJson = false
}
```

**Step 2: Create Execution Plan**

If `isPlanJson === true`:
- Use `planObject` directly
- User selects execution method and code review

If `isPlanJson === false`:
- Treat file content as prompt (same behavior as Mode 2)
- Create simple execution plan from content

**Step 3: User Interaction**

- AskUserQuestion: Select execution method (Agent/Codex/Auto)
- AskUserQuestion: Select code review tool
- Proceed to execution with full context

## Execution Process

```
Input Parsing:
   └─ Decision (mode detection):
      ├─ --in-memory flag → Mode 1: Load executionContext → Skip user selection
      ├─ Ends with .md/.json/.txt → Mode 3: Read file → Detect format
      │   ├─ Valid plan.json → Use planObject → User selects method + review
      │   └─ Not plan.json → Treat as prompt → User selects method + review
      └─ Other → Mode 2: Prompt description → User selects method + review

Execution:
   ├─ Step 1: Initialize result tracking (previousExecutionResults = [])
   ├─ Step 2: Task grouping & batch creation
   │   ├─ Extract explicit depends_on (no file/keyword inference)
   │   ├─ Group: independent tasks → single parallel batch (maximize utilization)
   │   ├─ Group: dependent tasks → sequential phases (respect dependencies)
   │   └─ Create TodoWrite list for batches
   ├─ Step 3: Launch execution
   │   ├─ Phase 1: All independent tasks (⚡ single batch, concurrent)
   │   └─ Phase 2+: Dependent tasks by dependency order
   ├─ Step 4: Track progress (TodoWrite updates per batch)
   └─ Step 5: Code review (if codeReviewTool ≠ "Skip")

Output:
   └─ Execution complete with results in previousExecutionResults[]
```

## Detailed Execution Steps

### Step 1: Initialize Execution Tracking

**Operations**:
- Initialize result tracking for multi-execution scenarios
- Set up `previousExecutionResults` array for context continuity

```javascript
// Initialize result tracking
previousExecutionResults = []
```

### Step 2: Task Grouping & Batch Creation

**Dependency Analysis & Grouping Algorithm**:
```javascript
// Use explicit depends_on from plan.json (no inference from file/keywords)
function extractDependencies(tasks) {
  const taskIdToIndex = {}
  tasks.forEach((t, i) => { taskIdToIndex[t.id] = i })

  return tasks.map((task, i) => {
    // Only use explicit depends_on from plan.json
    const deps = (task.depends_on || [])
      .map(depId => taskIdToIndex[depId])
      .filter(idx => idx !== undefined && idx < i)
    return { ...task, taskIndex: i, dependencies: deps }
  })
}

// Group into batches: maximize parallel execution
function createExecutionCalls(tasks, executionMethod) {
  const tasksWithDeps = extractDependencies(tasks)
  const processed = new Set()
  const calls = []

  // Phase 1: All independent tasks → single parallel batch (maximize utilization)
  const independentTasks = tasksWithDeps.filter(t => t.dependencies.length === 0)
  if (independentTasks.length > 0) {
    independentTasks.forEach(t => processed.add(t.taskIndex))
    calls.push({
      method: executionMethod,
      executionType: "parallel",
      groupId: "P1",
      taskSummary: independentTasks.map(t => t.title).join(' | '),
      tasks: independentTasks
    })
  }

  // Phase 2: Dependent tasks → sequential batches (respect dependencies)
  let sequentialIndex = 1
  let remaining = tasksWithDeps.filter(t => !processed.has(t.taskIndex))

  while (remaining.length > 0) {
    // Find tasks whose dependencies are all satisfied
    const ready = remaining.filter(t =>
      t.dependencies.every(d => processed.has(d))
    )

    if (ready.length === 0) {
      console.warn('Circular dependency detected, forcing remaining tasks')
      ready.push(...remaining)
    }

    // Group ready tasks (can run in parallel within this phase)
    ready.forEach(t => processed.add(t.taskIndex))
    calls.push({
      method: executionMethod,
      executionType: ready.length > 1 ? "parallel" : "sequential",
      groupId: ready.length > 1 ? `P${calls.length + 1}` : `S${sequentialIndex++}`,
      taskSummary: ready.map(t => t.title).join(ready.length > 1 ? ' | ' : ' → '),
      tasks: ready
    })

    remaining = remaining.filter(t => !processed.has(t.taskIndex))
  }

  return calls
}

executionCalls = createExecutionCalls(planObject.tasks, executionMethod).map(c => ({ ...c, id: `[${c.groupId}]` }))

TodoWrite({
  todos: executionCalls.map(c => ({
    content: `${c.executionType === "parallel" ? "⚡" : "→"} ${c.id} (${c.tasks.length} tasks)`,
    status: "pending",
    activeForm: `Executing ${c.id}`
  }))
})
```

### Step 3: Launch Execution

**Execution Flow**: Parallel batches concurrently → Sequential batches in order
```javascript
const parallel = executionCalls.filter(c => c.executionType === "parallel")
const sequential = executionCalls.filter(c => c.executionType === "sequential")

// Phase 1: Launch all parallel batches (single message with multiple tool calls)
if (parallel.length > 0) {
  TodoWrite({ todos: executionCalls.map(c => ({ status: c.executionType === "parallel" ? "in_progress" : "pending" })) })
  parallelResults = await Promise.all(parallel.map(c => executeBatch(c)))
  previousExecutionResults.push(...parallelResults)
  TodoWrite({ todos: executionCalls.map(c => ({ status: parallel.includes(c) ? "completed" : "pending" })) })
}

// Phase 2: Execute sequential batches one by one
for (const call of sequential) {
  TodoWrite({ todos: executionCalls.map(c => ({ status: c === call ? "in_progress" : "..." })) })
  result = await executeBatch(call)
  previousExecutionResults.push(result)
  TodoWrite({ todos: executionCalls.map(c => ({ status: "completed" or "pending" })) })
}
```

**Option A: Agent Execution**

When to use:
- `executionMethod = "Agent"`
- `executionMethod = "Auto" AND complexity = "Low"`

**Task Formatting Principle**: Each task is a self-contained checklist. The agent only needs to know what THIS task requires, not its position or relation to other tasks.

Agent call format:
```javascript
// Format single task as self-contained checklist
function formatTaskChecklist(task) {
  return `
## ${task.title}

**Target**: \`${task.file}\`
**Action**: ${task.action}

### What to do
${task.description}

### How to do it
${task.implementation.map(step => `- ${step}`).join('\n')}

### Reference
- Pattern: ${task.reference.pattern}
- Examples: ${task.reference.files.join(', ')}
- Notes: ${task.reference.examples}

### Done when
${task.acceptance.map(c => `- [ ] ${c}`).join('\n')}
`
}

// For batch execution: aggregate tasks without numbering
function formatBatchPrompt(batch) {
  const tasksSection = batch.tasks.map(t => formatTaskChecklist(t)).join('\n---\n')

  return `
${originalUserInput ? `## Goal\n${originalUserInput}\n` : ''}

## Tasks

${tasksSection}

${batch.context ? `## Context\n${batch.context}` : ''}

Complete each task according to its "Done when" checklist.
`
}

Task(
  subagent_type="code-developer",
  description=batch.taskSummary,
  prompt=formatBatchPrompt({
    tasks: batch.tasks,
    context: buildRelevantContext(batch.tasks)
  })
)

// Helper: Build relevant context for batch
// Context serves as REFERENCE ONLY - helps agent understand existing state
function buildRelevantContext(tasks) {
  const sections = []

  // 1. Previous work completion - what's already done (reference for continuity)
  if (previousExecutionResults.length > 0) {
    sections.push(`### Previous Work (Reference)
Use this to understand what's already completed. Avoid duplicating work.

${previousExecutionResults.map(r => `**${r.tasksSummary}**
- Status: ${r.status}
- Outputs: ${r.keyOutputs || 'See git diff'}
${r.notes ? `- Notes: ${r.notes}` : ''}`
    ).join('\n\n')}`)
  }

  // 2. Related files - files that may need to be read/referenced
  const relatedFiles = extractRelatedFiles(tasks)
  if (relatedFiles.length > 0) {
    sections.push(`### Related Files (Reference)
These files may contain patterns, types, or utilities relevant to your tasks:

${relatedFiles.map(f => `- \`${f}\``).join('\n')}`)
  }

  // 3. Clarifications from user
  if (clarificationContext) {
    sections.push(`### User Clarifications
${Object.entries(clarificationContext).map(([q, a]) => `- **${q}**: ${a}`).join('\n')}`)
  }

  // 4. Artifact files (for deeper context if needed)
  if (executionContext?.session?.artifacts?.plan) {
    sections.push(`### Artifacts
For detailed planning context, read: ${executionContext.session.artifacts.plan}`)
  }

  return sections.join('\n\n')
}

// Extract related files from task references
function extractRelatedFiles(tasks) {
  const files = new Set()
  tasks.forEach(task => {
    // Add reference example files
    if (task.reference?.files) {
      task.reference.files.forEach(f => files.add(f))
    }
  })
  return [...files]
}
```

**Result Collection**: After completion, collect result following `executionResult` structure (see Data Structures section)

**Option B: CLI Execution (Codex)**

When to use:
- `executionMethod = "Codex"`
- `executionMethod = "Auto" AND complexity = "Medium" or "High"`

**Task Formatting Principle**: Same as Agent - each task is a self-contained checklist. No task numbering or position awareness.

Command format:
```bash
// Format single task as compact checklist for CLI
function formatTaskForCLI(task) {
  return `
## ${task.title}
File: ${task.file}
Action: ${task.action}

What: ${task.description}

How:
${task.implementation.map(step => `- ${step}`).join('\n')}

Reference: ${task.reference.pattern} (see ${task.reference.files.join(', ')})
Notes: ${task.reference.examples}

Done when:
${task.acceptance.map(c => `- [ ] ${c}`).join('\n')}
`
}

// Build CLI prompt for batch
// Context provides REFERENCE information - not requirements to fulfill
function buildCLIPrompt(batch) {
  const tasksSection = batch.tasks.map(t => formatTaskForCLI(t)).join('\n---\n')

  let prompt = `${originalUserInput ? `## Goal\n${originalUserInput}\n\n` : ''}`
  prompt += `## Tasks\n\n${tasksSection}\n`

  // Context section - reference information only
  const contextSections = []

  // 1. Previous work - what's already completed
  if (previousExecutionResults.length > 0) {
    contextSections.push(`### Previous Work (Reference)
Already completed - avoid duplicating:
${previousExecutionResults.map(r => `- ${r.tasksSummary}: ${r.status}${r.keyOutputs ? ` (${r.keyOutputs})` : ''}`).join('\n')}`)
  }

  // 2. Related files from task references
  const relatedFiles = [...new Set(batch.tasks.flatMap(t => t.reference?.files || []))]
  if (relatedFiles.length > 0) {
    contextSections.push(`### Related Files (Reference)
Patterns and examples to follow:
${relatedFiles.map(f => `- ${f}`).join('\n')}`)
  }

  // 3. User clarifications
  if (clarificationContext) {
    contextSections.push(`### Clarifications
${Object.entries(clarificationContext).map(([q, a]) => `- ${q}: ${a}`).join('\n')}`)
  }

  // 4. Plan artifact for deeper context
  if (executionContext?.session?.artifacts?.plan) {
    contextSections.push(`### Artifacts
Detailed plan: ${executionContext.session.artifacts.plan}`)
  }

  if (contextSections.length > 0) {
    prompt += `\n## Context\n${contextSections.join('\n\n')}\n`
  }

  prompt += `\nComplete each task according to its "Done when" checklist.`

  return prompt
}

codex --full-auto exec "${buildCLIPrompt(batch)}" --skip-git-repo-check -s danger-full-access
```

**Execution with tracking**:
```javascript
// Launch CLI in foreground (NOT background)
// Timeout based on complexity: Low=40min, Medium=60min, High=100min
const timeoutByComplexity = {
  "Low": 2400000,    // 40 minutes
  "Medium": 3600000, // 60 minutes
  "High": 6000000    // 100 minutes
}

bash_result = Bash(
  command=cli_command,
  timeout=timeoutByComplexity[planObject.complexity] || 3600000
)

// Update TodoWrite when execution completes
```

**Result Collection**: After completion, analyze output and collect result following `executionResult` structure

### Step 4: Progress Tracking

Progress tracked at batch level (not individual task level). Icons: ⚡ (parallel, concurrent), → (sequential, one-by-one)

### Step 5: Code Review (Optional)

**Skip Condition**: Only run if `codeReviewTool ≠ "Skip"`

**Review Focus**: Verify implementation against plan acceptance criteria
- Read plan.json for task acceptance criteria
- Check each acceptance criterion is fulfilled
- Validate code quality and identify issues
- Ensure alignment with planned approach

**Operations**:
- Agent Review: Current agent performs direct review
- Gemini Review: Execute gemini CLI with review prompt
- Custom tool: Execute specified CLI tool (qwen, codex, etc.)

**Unified Review Template** (All tools use same standard):

**Review Criteria**:
- **Acceptance Criteria**: Verify each criterion from plan.tasks[].acceptance
- **Code Quality**: Analyze quality, identify issues, suggest improvements
- **Plan Alignment**: Validate implementation matches planned approach

**Shared Prompt Template** (used by all CLI tools):
```
PURPOSE: Code review for implemented changes against plan acceptance criteria
TASK: • Verify plan acceptance criteria fulfillment • Analyze code quality • Identify issues • Suggest improvements • Validate plan adherence
MODE: analysis
CONTEXT: @**/* @{plan.json} [@{exploration.json}] | Memory: Review lite-execute changes against plan requirements
EXPECTED: Quality assessment with acceptance criteria verification, issue identification, and recommendations. Explicitly check each acceptance criterion from plan.json tasks.
RULES: $(cat ~/.claude/workflows/cli-templates/prompts/analysis/02-review-code-quality.txt) | Focus on plan acceptance criteria and plan adherence | analysis=READ-ONLY
```

**Tool-Specific Execution** (Apply shared prompt template above):

```bash
# Method 1: Agent Review (current agent)
# - Read plan.json: ${executionContext.session.artifacts.plan}
# - Apply unified review criteria (see Shared Prompt Template)
# - Report findings directly

# Method 2: Gemini Review (recommended)
gemini -p "[Shared Prompt Template with artifacts]"
# CONTEXT includes: @**/* @${plan.json} [@${exploration.json}]

# Method 3: Qwen Review (alternative)
qwen -p "[Shared Prompt Template with artifacts]"
# Same prompt as Gemini, different execution engine

# Method 4: Codex Review (autonomous)
codex --full-auto exec "[Verify plan acceptance criteria at ${plan.json}]" --skip-git-repo-check -s danger-full-access
```

**Implementation Note**: Replace `[Shared Prompt Template with artifacts]` placeholder with actual template content, substituting:
- `@{plan.json}` → `@${executionContext.session.artifacts.plan}`
- `[@{exploration.json}]` → exploration files from artifacts (if exists)

### Step 6: Update Development Index

**Trigger**: After all executions complete (regardless of code review)

**Skip Condition**: Skip if `.workflow/project.json` does not exist

**Operations**:
```javascript
const projectJsonPath = '.workflow/project.json'
if (!fileExists(projectJsonPath)) return  // Silent skip

const projectJson = JSON.parse(Read(projectJsonPath))

// Initialize if needed
if (!projectJson.development_index) {
  projectJson.development_index = { feature: [], enhancement: [], bugfix: [], refactor: [], docs: [] }
}

// Detect category from keywords
function detectCategory(text) {
  text = text.toLowerCase()
  if (/\b(fix|bug|error|issue|crash)\b/.test(text)) return 'bugfix'
  if (/\b(refactor|cleanup|reorganize)\b/.test(text)) return 'refactor'
  if (/\b(doc|readme|comment)\b/.test(text)) return 'docs'
  if (/\b(add|new|create|implement)\b/.test(text)) return 'feature'
  return 'enhancement'
}

// Detect sub_feature from task file paths
function detectSubFeature(tasks) {
  const dirs = tasks.map(t => t.file?.split('/').slice(-2, -1)[0]).filter(Boolean)
  const counts = dirs.reduce((a, d) => { a[d] = (a[d] || 0) + 1; return a }, {})
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'general'
}

const category = detectCategory(`${planObject.summary} ${planObject.approach}`)
const entry = {
  title: planObject.summary.slice(0, 60),
  sub_feature: detectSubFeature(planObject.tasks),
  date: new Date().toISOString().split('T')[0],
  description: planObject.approach.slice(0, 100),
  status: previousExecutionResults.every(r => r.status === 'completed') ? 'completed' : 'partial',
  session_id: executionContext?.session?.id || null
}

projectJson.development_index[category].push(entry)
projectJson.statistics.last_updated = new Date().toISOString()
Write(projectJsonPath, JSON.stringify(projectJson, null, 2))

console.log(`✓ Development index: [${category}] ${entry.title}`)
```

## Best Practices

**Input Modes**: In-memory (lite-plan), prompt (standalone), file (JSON/text)
**Task Grouping**: Based on explicit depends_on only; independent tasks run in single parallel batch
**Execution**: All independent tasks launch concurrently via single Claude message with multiple tool calls

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| Missing executionContext | --in-memory without context | Error: "No execution context found. Only available when called by lite-plan." |
| File not found | File path doesn't exist | Error: "File not found: {path}. Check file path." |
| Empty file | File exists but no content | Error: "File is empty: {path}. Provide task description." |
| Invalid Enhanced Task JSON | JSON missing required fields | Warning: "Missing required fields. Treating as plain text." |
| Malformed JSON | JSON parsing fails | Treat as plain text (expected for non-JSON files) |
| Execution failure | Agent/Codex crashes | Display error, save partial progress, suggest retry |
| Codex unavailable | Codex not installed | Show installation instructions, offer Agent execution |

## Data Structures

### executionContext (Input - Mode 1)

Passed from lite-plan via global variable:

```javascript
{
  planObject: {
    summary: string,
    approach: string,
    tasks: [...],
    estimated_time: string,
    recommended_execution: string,
    complexity: string
  },
  explorationsContext: {...} | null,       // Multi-angle explorations
  explorationAngles: string[],             // List of exploration angles
  explorationManifest: {...} | null,       // Exploration manifest
  clarificationContext: {...} | null,
  executionMethod: "Agent" | "Codex" | "Auto",
  codeReviewTool: "Skip" | "Gemini Review" | "Agent Review" | string,
  originalUserInput: string,

  // Session artifacts location (saved by lite-plan)
  session: {
    id: string,                        // Session identifier: {taskSlug}-{shortTimestamp}
    folder: string,                    // Session folder path: .workflow/.lite-plan/{session-id}
    artifacts: {
      explorations: [{angle, path}],   // exploration-{angle}.json paths
      explorations_manifest: string,   // explorations-manifest.json path
      plan: string                     // plan.json path (always present)
    }
  }
}
```

**Artifact Usage**:
- Artifact files contain detailed planning context
- Pass artifact paths to CLI tools and agents for enhanced context
- See execution options below for usage examples

### executionResult (Output)

Collected after each execution call completes:

```javascript
{
  executionId: string,                 // e.g., "[Agent-1]", "[Codex-1]"
  status: "completed" | "partial" | "failed",
  tasksSummary: string,                // Brief description of tasks handled
  completionSummary: string,           // What was completed
  keyOutputs: string,                  // Files created/modified, key changes
  notes: string                        // Important context for next execution
}
```

Appended to `previousExecutionResults` array for context continuity in multi-execution scenarios.
