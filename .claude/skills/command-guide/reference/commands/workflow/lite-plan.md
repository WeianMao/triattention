---
name: lite-plan
description: Lightweight interactive planning workflow with in-memory planning, code exploration, and execution dispatch to lite-execute after user confirmation
argument-hint: "[-e|--explore] \"task description\"|file.md"
allowed-tools: TodoWrite(*), Task(*), SlashCommand(*), AskUserQuestion(*)
---

# Workflow Lite-Plan Command (/workflow:lite-plan)

## Overview

Intelligent lightweight planning command with dynamic workflow adaptation based on task complexity. Focuses on planning phases (exploration, clarification, planning, confirmation) and delegates execution to `/workflow:lite-execute`.

**Core capabilities:**
- Intelligent task analysis with automatic exploration detection
- Dynamic code exploration (cli-explore-agent) when codebase understanding needed
- Interactive clarification after exploration to gather missing information
- Adaptive planning strategy (direct Claude vs cli-lite-planning-agent) based on complexity
- Two-step confirmation: plan display → multi-dimensional input collection
- Execution dispatch with complete context handoff to lite-execute

## Usage

```bash
/workflow:lite-plan [FLAGS] <TASK_DESCRIPTION>

# Flags
-e, --explore              Force code exploration phase (overrides auto-detection)

# Arguments
<task-description>         Task description or path to .md file (required)
```

## Execution Process

```
Phase 1: Task Analysis & Exploration
   ├─ Parse input (description or .md file)
   ├─ intelligent complexity assessment (Low/Medium/High)
   ├─ Exploration decision (auto-detect or --explore flag)
   ├─ ⚠️ Context protection: If file reading ≥50k chars → force cli-explore-agent
   └─ Decision:
      ├─ needsExploration=true → Launch parallel cli-explore-agents (1-4 based on complexity)
      └─ needsExploration=false → Skip to Phase 2/3

Phase 2: Clarification (optional, multi-round)
   ├─ Aggregate clarification_needs from all exploration angles
   ├─ Deduplicate similar questions
   └─ Decision:
      ├─ Has clarifications → AskUserQuestion (max 4 questions per round, multiple rounds allowed)
      └─ No clarifications → Skip to Phase 3

Phase 3: Planning (NO CODE EXECUTION - planning only)
   └─ Decision (based on Phase 1 complexity):
      ├─ Low → Load schema: cat ~/.claude/workflows/cli-templates/schemas/plan-json-schema.json → Direct Claude planning (following schema) → plan.json → MUST proceed to Phase 4
      └─ Medium/High → cli-lite-planning-agent → plan.json → MUST proceed to Phase 4

Phase 4: Confirmation & Selection
   ├─ Display plan summary (tasks, complexity, estimated time)
   └─ AskUserQuestion:
      ├─ Confirm: Allow / Modify / Cancel
      ├─ Execution: Agent / Codex / Auto
      └─ Review: Gemini / Agent / Skip

Phase 5: Dispatch
   ├─ Build executionContext (plan + explorations + clarifications + selections)
   └─ SlashCommand("/workflow:lite-execute --in-memory")
```

## Implementation

### Phase 1: Intelligent Multi-Angle Exploration

**Session Setup** (MANDATORY - follow exactly):
```javascript
// Helper: Get UTC+8 (China Standard Time) ISO string
const getUtc8ISOString = () => new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString()

const taskSlug = task_description.toLowerCase().replace(/[^a-z0-9]+/g, '-').substring(0, 40)
const dateStr = getUtc8ISOString().substring(0, 10)  // Format: 2025-11-29

const sessionId = `${taskSlug}-${dateStr}`  // e.g., "implement-jwt-refresh-2025-11-29"
const sessionFolder = `.workflow/.lite-plan/${sessionId}`

bash(`mkdir -p ${sessionFolder} && test -d ${sessionFolder} && echo "SUCCESS: ${sessionFolder}" || echo "FAILED: ${sessionFolder}"`)
```

**Exploration Decision Logic**:
```javascript
needsExploration = (
  flags.includes('--explore') || flags.includes('-e') ||
  task.mentions_specific_files ||
  task.requires_codebase_context ||
  task.needs_architecture_understanding ||
  task.modifies_existing_code
)

if (!needsExploration) {
  // Skip to Phase 2 (Clarification) or Phase 3 (Planning)
  proceed_to_next_phase()
}
```

**⚠️ Context Protection**: File reading ≥50k chars → force `needsExploration=true` (delegate to cli-explore-agent)

**Complexity Assessment** (Intelligent Analysis):
```javascript
// analyzes task complexity based on:
// - Scope: How many systems/modules are affected?
// - Depth: Surface change vs architectural impact?
// - Risk: Potential for breaking existing functionality?
// - Dependencies: How interconnected is the change?

const complexity = analyzeTaskComplexity(task_description)
// Returns: 'Low' | 'Medium' | 'High'
// Low: Single file, isolated change, minimal risk
// Medium: Multiple files, some dependencies, moderate risk
// High: Cross-module, architectural, high risk

// Angle assignment based on task type (orchestrator decides, not agent)
const ANGLE_PRESETS = {
  architecture: ['architecture', 'dependencies', 'modularity', 'integration-points'],
  security: ['security', 'auth-patterns', 'dataflow', 'validation'],
  performance: ['performance', 'bottlenecks', 'caching', 'data-access'],
  bugfix: ['error-handling', 'dataflow', 'state-management', 'edge-cases'],
  feature: ['patterns', 'integration-points', 'testing', 'dependencies']
}

function selectAngles(taskDescription, count) {
  const text = taskDescription.toLowerCase()
  let preset = 'feature' // default

  if (/refactor|architect|restructure|modular/.test(text)) preset = 'architecture'
  else if (/security|auth|permission|access/.test(text)) preset = 'security'
  else if (/performance|slow|optimi|cache/.test(text)) preset = 'performance'
  else if (/fix|bug|error|issue|broken/.test(text)) preset = 'bugfix'

  return ANGLE_PRESETS[preset].slice(0, count)
}

const selectedAngles = selectAngles(task_description, complexity === 'High' ? 4 : (complexity === 'Medium' ? 3 : 1))

console.log(`
## Exploration Plan

Task Complexity: ${complexity}
Selected Angles: ${selectedAngles.join(', ')}

Launching ${selectedAngles.length} parallel explorations...
`)
```

**Launch Parallel Explorations** - Orchestrator assigns angle to each agent:

```javascript
// Launch agents with pre-assigned angles
const explorationTasks = selectedAngles.map((angle, index) =>
  Task(
    subagent_type="cli-explore-agent",
    description=`Explore: ${angle}`,
    prompt=`
## Task Objective
Execute **${angle}** exploration for task planning context. Analyze codebase from this specific angle to discover relevant structure, patterns, and constraints.

## Assigned Context
- **Exploration Angle**: ${angle}
- **Task Description**: ${task_description}
- **Exploration Index**: ${index + 1} of ${selectedAngles.length}
- **Output File**: ${sessionFolder}/exploration-${angle}.json

## MANDATORY FIRST STEPS (Execute by Agent)
**You (cli-explore-agent) MUST execute these steps in order:**
1. Run: ccw tool exec get_modules_by_depth '{}' (project structure)
2. Run: rg -l "{keyword_from_task}" --type ts (locate relevant files)
3. Execute: cat ~/.claude/workflows/cli-templates/schemas/explore-json-schema.json (get output schema reference)

## Exploration Strategy (${angle} focus)

**Step 1: Structural Scan** (Bash)
- get_modules_by_depth.sh → identify modules related to ${angle}
- find/rg → locate files relevant to ${angle} aspect
- Analyze imports/dependencies from ${angle} perspective

**Step 2: Semantic Analysis** (Gemini CLI)
- How does existing code handle ${angle} concerns?
- What patterns are used for ${angle}?
- Where would new code integrate from ${angle} viewpoint?

**Step 3: Write Output**
- Consolidate ${angle} findings into JSON
- Identify ${angle}-specific clarification needs

## Expected Output

**File**: ${sessionFolder}/exploration-${angle}.json

**Schema Reference**: Schema obtained in MANDATORY FIRST STEPS step 3, follow schema exactly

**Required Fields** (all ${angle} focused):
- project_structure: Modules/architecture relevant to ${angle}
- relevant_files: Files affected from ${angle} perspective
  **IMPORTANT**: Use object format with relevance scores for synthesis:
  \`[{path: "src/file.ts", relevance: 0.85, rationale: "Core ${angle} logic"}]\`
  Scores: 0.7+ high priority, 0.5-0.7 medium, <0.5 low
- patterns: ${angle}-related patterns to follow
- dependencies: Dependencies relevant to ${angle}
- integration_points: Where to integrate from ${angle} viewpoint (include file:line locations)
- constraints: ${angle}-specific limitations/conventions
- clarification_needs: ${angle}-related ambiguities (options array + recommended index)
- _metadata.exploration_angle: "${angle}"

## Success Criteria
- [ ] Schema obtained via cat explore-json-schema.json
- [ ] get_modules_by_depth.sh executed
- [ ] At least 3 relevant files identified with ${angle} rationale
- [ ] Patterns are actionable (code examples, not generic advice)
- [ ] Integration points include file:line locations
- [ ] Constraints are project-specific to ${angle}
- [ ] JSON output follows schema exactly
- [ ] clarification_needs includes options + recommended

## Output
Write: ${sessionFolder}/exploration-${angle}.json
Return: 2-3 sentence summary of ${angle} findings
`
  )
)

// Execute all exploration tasks in parallel
```

**Auto-discover Generated Exploration Files**:
```javascript
// After explorations complete, auto-discover all exploration-*.json files
const explorationFiles = bash(`find ${sessionFolder} -name "exploration-*.json" -type f`)
  .split('\n')
  .filter(f => f.trim())

// Read metadata to build manifest
const explorationManifest = {
  session_id: sessionId,
  task_description: task_description,
  timestamp: getUtc8ISOString(),
  complexity: complexity,
  exploration_count: explorationCount,
  explorations: explorationFiles.map(file => {
    const data = JSON.parse(Read(file))
    const filename = path.basename(file)
    return {
      angle: data._metadata.exploration_angle,
      file: filename,
      path: file,
      index: data._metadata.exploration_index
    }
  })
}

Write(`${sessionFolder}/explorations-manifest.json`, JSON.stringify(explorationManifest, null, 2))

console.log(`
## Exploration Complete

Generated exploration files in ${sessionFolder}:
${explorationManifest.explorations.map(e => `- exploration-${e.angle}.json (angle: ${e.angle})`).join('\n')}

Manifest: explorations-manifest.json
Angles explored: ${explorationManifest.explorations.map(e => e.angle).join(', ')}
`)
```

**Output**:
- `${sessionFolder}/exploration-{angle1}.json`
- `${sessionFolder}/exploration-{angle2}.json`
- ... (1-4 files based on complexity)
- `${sessionFolder}/explorations-manifest.json`

---

### Phase 2: Clarification (Optional, Multi-Round)

**Skip if**: No exploration or `clarification_needs` is empty across all explorations

**⚠️ CRITICAL**: AskUserQuestion tool limits max 4 questions per call. **MUST execute multiple rounds** to exhaust all clarification needs - do NOT stop at round 1.

**Aggregate clarification needs from all exploration angles**:
```javascript
// Load manifest and all exploration files
const manifest = JSON.parse(Read(`${sessionFolder}/explorations-manifest.json`))
const explorations = manifest.explorations.map(exp => ({
  angle: exp.angle,
  data: JSON.parse(Read(exp.path))
}))

// Aggregate clarification needs from all explorations
const allClarifications = []
explorations.forEach(exp => {
  if (exp.data.clarification_needs?.length > 0) {
    exp.data.clarification_needs.forEach(need => {
      allClarifications.push({
        ...need,
        source_angle: exp.angle
      })
    })
  }
})

// Deduplicate exact same questions only
const seen = new Set()
const dedupedClarifications = allClarifications.filter(c => {
  const key = c.question.toLowerCase()
  if (seen.has(key)) return false
  seen.add(key)
  return true
})

// Multi-round clarification: batch questions (max 4 per round)
if (dedupedClarifications.length > 0) {
  const BATCH_SIZE = 4
  const totalRounds = Math.ceil(dedupedClarifications.length / BATCH_SIZE)

  for (let i = 0; i < dedupedClarifications.length; i += BATCH_SIZE) {
    const batch = dedupedClarifications.slice(i, i + BATCH_SIZE)
    const currentRound = Math.floor(i / BATCH_SIZE) + 1

    console.log(`### Clarification Round ${currentRound}/${totalRounds}`)

    AskUserQuestion({
      questions: batch.map(need => ({
        question: `[${need.source_angle}] ${need.question}\n\nContext: ${need.context}`,
        header: need.source_angle.substring(0, 12),
        multiSelect: false,
        options: need.options.map((opt, index) => ({
          label: need.recommended === index ? `${opt} ★` : opt,
          description: need.recommended === index ? `Recommended` : `Use ${opt}`
        }))
      }))
    })

    // Store batch responses in clarificationContext before next round
  }
}
```

**Output**: `clarificationContext` (in-memory)

---

### Phase 3: Planning

**Planning Strategy Selection** (based on Phase 1 complexity):

**IMPORTANT**: Phase 3 is **planning only** - NO code execution. All execution happens in Phase 5 via lite-execute.

**Low Complexity** - Direct planning by Claude:
```javascript
// Step 1: Read schema
const schema = Bash(`cat ~/.claude/workflows/cli-templates/schemas/plan-json-schema.json`)

// Step 2: Generate plan following schema (Claude directly, no agent)
const plan = {
  summary: "...",
  approach: "...",
  tasks: [...],  // Each task: { id, title, scope, ..., depends_on, execution_group, complexity }
  estimated_time: "...",
  recommended_execution: "Agent",
  complexity: "Low",
  _metadata: { timestamp: getUtc8ISOString(), source: "direct-planning", planning_mode: "direct" }
}

// Step 3: Write plan to session folder
Write(`${sessionFolder}/plan.json`, JSON.stringify(plan, null, 2))

// Step 4: MUST continue to Phase 4 (Confirmation) - DO NOT execute code here
```

**Medium/High Complexity** - Invoke cli-lite-planning-agent:

```javascript
Task(
  subagent_type="cli-lite-planning-agent",
  description="Generate detailed implementation plan",
  prompt=`
Generate implementation plan and write plan.json.

## Output Schema Reference
Execute: cat ~/.claude/workflows/cli-templates/schemas/plan-json-schema.json (get schema reference before generating plan)

## Task Description
${task_description}

## Multi-Angle Exploration Context

${manifest.explorations.map(exp => `### Exploration: ${exp.angle} (${exp.file})
Path: ${exp.path}

Read this file for detailed ${exp.angle} analysis.`).join('\n\n')}

Total explorations: ${manifest.exploration_count}
Angles covered: ${manifest.explorations.map(e => e.angle).join(', ')}

Manifest: ${sessionFolder}/explorations-manifest.json

## User Clarifications
${JSON.stringify(clarificationContext) || "None"}

## Complexity Level
${complexity}

## Requirements
Generate plan.json with:
- summary: 2-3 sentence overview
- approach: High-level implementation strategy (incorporating insights from all exploration angles)
- tasks: 2-7 structured tasks (**IMPORTANT: group by feature/module, NOT by file**)
  - **Task Granularity Principle**: Each task = one complete feature unit or module
  - title: action verb + target module/feature (e.g., "Implement auth token refresh")
  - scope: module path (src/auth/) or feature name, prefer module-level over single file
  - action, description
  - modification_points: ALL files to modify for this feature (group related changes)
  - implementation (3-7 steps covering all modification_points)
  - reference (pattern, files, examples)
  - acceptance (2-4 criteria for the entire feature)
  - depends_on: task IDs this task depends on (use sparingly, only for true dependencies)
- estimated_time, recommended_execution, complexity
- _metadata:
  - timestamp, source, planning_mode
  - exploration_angles: ${JSON.stringify(manifest.explorations.map(e => e.angle))}

## Task Grouping Rules
1. **Group by feature**: All changes for one feature = one task (even if 3-5 files)
2. **Group by context**: Tasks with similar context or related functional changes can be grouped together
3. **Minimize agent count**: Simple, unrelated tasks can also be grouped to reduce agent execution overhead
4. **Avoid file-per-task**: Do NOT create separate tasks for each file
5. **Substantial tasks**: Each task should represent 15-60 minutes of work
6. **True dependencies only**: Only use depends_on when Task B cannot start without Task A's output
7. **Prefer parallel**: Most tasks should be independent (no depends_on)

## Execution
1. Read ALL exploration files for comprehensive context
2. Execute CLI planning using Gemini (Qwen fallback)
3. Synthesize findings from multiple exploration angles
4. Parse output and structure plan
5. Write JSON: Write('${sessionFolder}/plan.json', jsonContent)
6. Return brief completion summary
`
)
```

**Output**: `${sessionFolder}/plan.json`

---

### Phase 4: Task Confirmation & Execution Selection

**Step 4.1: Display Plan**
```javascript
const plan = JSON.parse(Read(`${sessionFolder}/plan.json`))

console.log(`
## Implementation Plan

**Summary**: ${plan.summary}
**Approach**: ${plan.approach}

**Tasks** (${plan.tasks.length}):
${plan.tasks.map((t, i) => `${i+1}. ${t.title} (${t.file})`).join('\n')}

**Complexity**: ${plan.complexity}
**Estimated Time**: ${plan.estimated_time}
**Recommended**: ${plan.recommended_execution}
`)
```

**Step 4.2: Collect Confirmation**
```javascript
AskUserQuestion({
  questions: [
    {
      question: `Confirm plan? (${plan.tasks.length} tasks, ${plan.complexity})`,
      header: "Confirm",
      multiSelect: true,
      options: [
        { label: "Allow", description: "Proceed as-is" },
        { label: "Modify", description: "Adjust before execution" },
        { label: "Cancel", description: "Abort workflow" }
      ]
    },
    {
      question: "Execution method:",
      header: "Execution",
      multiSelect: false,
      options: [
        { label: "Agent", description: "@code-developer agent" },
        { label: "Codex", description: "codex CLI tool" },
        { label: "Auto", description: `Auto: ${plan.complexity === 'Low' ? 'Agent' : 'Codex'}` }
      ]
    },
    {
      question: "Code review after execution?",
      header: "Review",
      multiSelect: false,
      options: [
        { label: "Gemini Review", description: "Gemini CLI" },
        { label: "Agent Review", description: "@code-reviewer" },
        { label: "Skip", description: "No review" }
      ]
    }
  ]
})
```

---

### Phase 5: Dispatch to Execution

**CRITICAL**: lite-plan NEVER executes code directly. ALL execution MUST go through lite-execute.

**Step 5.1: Build executionContext**

```javascript
// Load manifest and all exploration files
const manifest = JSON.parse(Read(`${sessionFolder}/explorations-manifest.json`))
const explorations = {}

manifest.explorations.forEach(exp => {
  if (file_exists(exp.path)) {
    explorations[exp.angle] = JSON.parse(Read(exp.path))
  }
})

const plan = JSON.parse(Read(`${sessionFolder}/plan.json`))

executionContext = {
  planObject: plan,
  explorationsContext: explorations,
  explorationAngles: manifest.explorations.map(e => e.angle),
  explorationManifest: manifest,
  clarificationContext: clarificationContext || null,
  executionMethod: userSelection.execution_method,
  codeReviewTool: userSelection.code_review_tool,
  originalUserInput: task_description,
  session: {
    id: sessionId,
    folder: sessionFolder,
    artifacts: {
      explorations: manifest.explorations.map(exp => ({
        angle: exp.angle,
        path: exp.path
      })),
      explorations_manifest: `${sessionFolder}/explorations-manifest.json`,
      plan: `${sessionFolder}/plan.json`
    }
  }
}
```

**Step 5.2: Dispatch**

```javascript
SlashCommand(command="/workflow:lite-execute --in-memory")
```

## Session Folder Structure

```
.workflow/.lite-plan/{task-slug}-{YYYY-MM-DD}/
├── exploration-{angle1}.json      # Exploration angle 1
├── exploration-{angle2}.json      # Exploration angle 2
├── exploration-{angle3}.json      # Exploration angle 3 (if applicable)
├── exploration-{angle4}.json      # Exploration angle 4 (if applicable)
├── explorations-manifest.json     # Exploration index
└── plan.json                      # Implementation plan
```

**Example**:
```
.workflow/.lite-plan/implement-jwt-refresh-2025-11-25-14-30-25/
├── exploration-architecture.json
├── exploration-auth-patterns.json
├── exploration-security.json
├── explorations-manifest.json
└── plan.json
```

## Error Handling

| Error | Resolution |
|-------|------------|
| Exploration agent failure | Skip exploration, continue with task description only |
| Planning agent failure | Fallback to direct planning by Claude |
| Clarification timeout | Use exploration findings as-is |
| Confirmation timeout | Save context, display resume instructions |
| Modify loop > 3 times | Suggest breaking task or using /workflow:plan |
