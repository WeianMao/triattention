---
name: lite-fix
description: Lightweight bug diagnosis and fix workflow with intelligent severity assessment and optional hotfix mode for production incidents
argument-hint: "[--hotfix] \"bug description or issue reference\""
allowed-tools: TodoWrite(*), Task(*), SlashCommand(*), AskUserQuestion(*)
---

# Workflow Lite-Fix Command (/workflow:lite-fix)

## Overview

Intelligent lightweight bug fixing command with dynamic workflow adaptation based on severity assessment. Focuses on diagnosis phases (root cause analysis, impact assessment, fix planning, confirmation) and delegates execution to `/workflow:lite-execute`.

**Core capabilities:**
- Intelligent bug analysis with automatic severity detection
- Dynamic code diagnosis (cli-explore-agent) for root cause identification
- Interactive clarification after diagnosis to gather missing information
- Adaptive fix planning strategy (direct Claude vs cli-lite-planning-agent) based on complexity
- Two-step confirmation: fix-plan display -> multi-dimensional input collection
- Execution dispatch with complete context handoff to lite-execute

## Usage

```bash
/workflow:lite-fix [FLAGS] <BUG_DESCRIPTION>

# Flags
--hotfix, -h               Production hotfix mode (minimal diagnosis, fast fix)

# Arguments
<bug-description>          Bug description, error message, or path to .md file (required)
```

## Execution Process

```
Phase 1: Bug Analysis & Diagnosis
   |- Parse input (description, error message, or .md file)
   |- Intelligent severity pre-assessment (Low/Medium/High/Critical)
   |- Diagnosis decision (auto-detect or --hotfix flag)
   |- Context protection: If file reading >=50k chars -> force cli-explore-agent
   +- Decision:
      |- needsDiagnosis=true -> Launch parallel cli-explore-agents (1-4 based on severity)
      +- needsDiagnosis=false (hotfix) -> Skip directly to Phase 3 (Fix Planning)

Phase 2: Clarification (optional, multi-round)
   |- Aggregate clarification_needs from all diagnosis angles
   |- Deduplicate similar questions
   +- Decision:
      |- Has clarifications -> AskUserQuestion (max 4 questions per round, multiple rounds allowed)
      +- No clarifications -> Skip to Phase 3

Phase 3: Fix Planning (NO CODE EXECUTION - planning only)
   +- Decision (based on Phase 1 severity):
      |- Low/Medium -> Load schema: cat ~/.claude/workflows/cli-templates/schemas/fix-plan-json-schema.json -> Direct Claude planning (following schema) -> fix-plan.json -> MUST proceed to Phase 4
      +- High/Critical -> cli-lite-planning-agent -> fix-plan.json -> MUST proceed to Phase 4

Phase 4: Confirmation & Selection
   |- Display fix-plan summary (tasks, severity, estimated time)
   +- AskUserQuestion:
      |- Confirm: Allow / Modify / Cancel
      |- Execution: Agent / Codex / Auto
      +- Review: Gemini / Agent / Skip

Phase 5: Dispatch
   |- Build executionContext (fix-plan + diagnoses + clarifications + selections)
   +- SlashCommand("/workflow:lite-execute --in-memory --mode bugfix")
```

## Implementation

### Phase 1: Intelligent Multi-Angle Diagnosis

**Session Setup** (MANDATORY - follow exactly):
```javascript
// Helper: Get UTC+8 (China Standard Time) ISO string
const getUtc8ISOString = () => new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString()

const bugSlug = bug_description.toLowerCase().replace(/[^a-z0-9]+/g, '-').substring(0, 40)
const dateStr = getUtc8ISOString().substring(0, 10)  // Format: 2025-11-29

const sessionId = `${bugSlug}-${dateStr}`  // e.g., "user-avatar-upload-fails-2025-11-29"
const sessionFolder = `.workflow/.lite-fix/${sessionId}`

bash(`mkdir -p ${sessionFolder} && test -d ${sessionFolder} && echo "SUCCESS: ${sessionFolder}" || echo "FAILED: ${sessionFolder}"`)
```

**Diagnosis Decision Logic**:
```javascript
const hotfixMode = $ARGUMENTS.includes('--hotfix') || $ARGUMENTS.includes('-h')

needsDiagnosis = (
  !hotfixMode &&
  (
    bug.lacks_specific_error_message ||
    bug.requires_codebase_context ||
    bug.needs_execution_tracing ||
    bug.root_cause_unclear
  )
)

if (!needsDiagnosis) {
  // Skip to Phase 2 (Clarification) or Phase 3 (Fix Planning)
  proceed_to_next_phase()
}
```

**Context Protection**: File reading >=50k chars -> force `needsDiagnosis=true` (delegate to cli-explore-agent)

**Severity Pre-Assessment** (Intelligent Analysis):
```javascript
// Analyzes bug severity based on:
// - Symptoms: Error messages, crash reports, user complaints
// - Scope: How many users/features are affected?
// - Urgency: Production down vs minor inconvenience
// - Impact: Data loss, security, business impact

const severity = analyzeBugSeverity(bug_description)
// Returns: 'Low' | 'Medium' | 'High' | 'Critical'
// Low: Minor UI issue, localized, no data impact
// Medium: Multiple users affected, degraded functionality
// High: Significant functionality broken, many users affected
// Critical: Production down, data loss risk, security issue

// Angle assignment based on bug type (orchestrator decides, not agent)
const DIAGNOSIS_ANGLE_PRESETS = {
  runtime_error: ['error-handling', 'dataflow', 'state-management', 'edge-cases'],
  performance: ['performance', 'bottlenecks', 'caching', 'data-access'],
  security: ['security', 'auth-patterns', 'dataflow', 'validation'],
  data_corruption: ['data-integrity', 'state-management', 'transactions', 'validation'],
  ui_bug: ['state-management', 'event-handling', 'rendering', 'data-binding'],
  integration: ['api-contracts', 'error-handling', 'timeouts', 'fallbacks']
}

function selectDiagnosisAngles(bugDescription, count) {
  const text = bugDescription.toLowerCase()
  let preset = 'runtime_error' // default

  if (/slow|timeout|performance|lag|hang/.test(text)) preset = 'performance'
  else if (/security|auth|permission|access|token/.test(text)) preset = 'security'
  else if (/corrupt|data|lost|missing|inconsistent/.test(text)) preset = 'data_corruption'
  else if (/ui|display|render|style|click|button/.test(text)) preset = 'ui_bug'
  else if (/api|integration|connect|request|response/.test(text)) preset = 'integration'

  return DIAGNOSIS_ANGLE_PRESETS[preset].slice(0, count)
}

const selectedAngles = selectDiagnosisAngles(bug_description, severity === 'Critical' ? 4 : (severity === 'High' ? 3 : (severity === 'Medium' ? 2 : 1)))

console.log(`
## Diagnosis Plan

Bug Severity: ${severity}
Selected Angles: ${selectedAngles.join(', ')}

Launching ${selectedAngles.length} parallel diagnoses...
`)
```

**Launch Parallel Diagnoses** - Orchestrator assigns angle to each agent:

```javascript
// Launch agents with pre-assigned diagnosis angles
const diagnosisTasks = selectedAngles.map((angle, index) =>
  Task(
    subagent_type="cli-explore-agent",
    description=`Diagnose: ${angle}`,
    prompt=`
## Task Objective
Execute **${angle}** diagnosis for bug root cause analysis. Analyze codebase from this specific angle to discover root cause, affected paths, and fix hints.

## Assigned Context
- **Diagnosis Angle**: ${angle}
- **Bug Description**: ${bug_description}
- **Diagnosis Index**: ${index + 1} of ${selectedAngles.length}
- **Output File**: ${sessionFolder}/diagnosis-${angle}.json

## MANDATORY FIRST STEPS (Execute by Agent)
**You (cli-explore-agent) MUST execute these steps in order:**
1. Run: ccw tool exec get_modules_by_depth '{}' (project structure)
2. Run: rg -l "{error_keyword_from_bug}" --type ts (locate relevant files)
3. Execute: cat ~/.claude/workflows/cli-templates/schemas/diagnosis-json-schema.json (get output schema reference)

## Diagnosis Strategy (${angle} focus)

**Step 1: Error Tracing** (Bash)
- rg for error messages, stack traces, log patterns
- git log --since='2 weeks ago' for recent changes
- Trace execution path in affected modules

**Step 2: Root Cause Analysis** (Gemini CLI)
- What code paths lead to this ${angle} issue?
- What edge cases are not handled from ${angle} perspective?
- What recent changes might have introduced this bug?

**Step 3: Write Output**
- Consolidate ${angle} findings into JSON
- Identify ${angle}-specific clarification needs
- Provide fix hints based on ${angle} analysis

## Expected Output

**File**: ${sessionFolder}/diagnosis-${angle}.json

**Schema Reference**: Schema obtained in MANDATORY FIRST STEPS step 3, follow schema exactly

**Required Fields** (all ${angle} focused):
- symptom: Bug symptoms and error messages
- root_cause: Root cause hypothesis from ${angle} perspective
  **IMPORTANT**: Use structured format:
  \`{file: "src/module/file.ts", line_range: "45-60", issue: "Description", confidence: 0.85}\`
- affected_files: Files involved from ${angle} perspective
  **IMPORTANT**: Use object format with relevance scores:
  \`[{path: "src/file.ts", relevance: 0.85, rationale: "Contains ${angle} logic"}]\`
- reproduction_steps: Steps to reproduce the bug
- fix_hints: Suggested fix approaches from ${angle} viewpoint
- dependencies: Dependencies relevant to ${angle} diagnosis
- constraints: ${angle}-specific limitations affecting fix
- clarification_needs: ${angle}-related ambiguities (options array + recommended index)
- _metadata.diagnosis_angle: "${angle}"
- _metadata.diagnosis_index: ${index + 1}

## Success Criteria
- [ ] Schema obtained via cat diagnosis-json-schema.json
- [ ] get_modules_by_depth.sh executed
- [ ] Root cause identified with confidence score
- [ ] At least 3 affected files identified with ${angle} rationale
- [ ] Fix hints are actionable (specific code changes, not generic advice)
- [ ] Reproduction steps are verifiable
- [ ] JSON output follows schema exactly
- [ ] clarification_needs includes options + recommended

## Output
Write: ${sessionFolder}/diagnosis-${angle}.json
Return: 2-3 sentence summary of ${angle} diagnosis findings
`
  )
)

// Execute all diagnosis tasks in parallel
```

**Auto-discover Generated Diagnosis Files**:
```javascript
// After diagnoses complete, auto-discover all diagnosis-*.json files
const diagnosisFiles = bash(`find ${sessionFolder} -name "diagnosis-*.json" -type f`)
  .split('\n')
  .filter(f => f.trim())

// Read metadata to build manifest
const diagnosisManifest = {
  session_id: sessionId,
  bug_description: bug_description,
  timestamp: getUtc8ISOString(),
  severity: severity,
  diagnosis_count: diagnosisFiles.length,
  diagnoses: diagnosisFiles.map(file => {
    const data = JSON.parse(Read(file))
    const filename = path.basename(file)
    return {
      angle: data._metadata.diagnosis_angle,
      file: filename,
      path: file,
      index: data._metadata.diagnosis_index
    }
  })
}

Write(`${sessionFolder}/diagnoses-manifest.json`, JSON.stringify(diagnosisManifest, null, 2))

console.log(`
## Diagnosis Complete

Generated diagnosis files in ${sessionFolder}:
${diagnosisManifest.diagnoses.map(d => `- diagnosis-${d.angle}.json (angle: ${d.angle})`).join('\n')}

Manifest: diagnoses-manifest.json
Angles diagnosed: ${diagnosisManifest.diagnoses.map(d => d.angle).join(', ')}
`)
```

**Output**:
- `${sessionFolder}/diagnosis-{angle1}.json`
- `${sessionFolder}/diagnosis-{angle2}.json`
- ... (1-4 files based on severity)
- `${sessionFolder}/diagnoses-manifest.json`

---

### Phase 2: Clarification (Optional, Multi-Round)

**Skip if**: No diagnosis or `clarification_needs` is empty across all diagnoses

**⚠️ CRITICAL**: AskUserQuestion tool limits max 4 questions per call. **MUST execute multiple rounds** to exhaust all clarification needs - do NOT stop at round 1.

**Aggregate clarification needs from all diagnosis angles**:
```javascript
// Load manifest and all diagnosis files
const manifest = JSON.parse(Read(`${sessionFolder}/diagnoses-manifest.json`))
const diagnoses = manifest.diagnoses.map(diag => ({
  angle: diag.angle,
  data: JSON.parse(Read(diag.path))
}))

// Aggregate clarification needs from all diagnoses
const allClarifications = []
diagnoses.forEach(diag => {
  if (diag.data.clarification_needs?.length > 0) {
    diag.data.clarification_needs.forEach(need => {
      allClarifications.push({
        ...need,
        source_angle: diag.angle
      })
    })
  }
})

// Deduplicate by question similarity
function deduplicateClarifications(clarifications) {
  const unique = []
  clarifications.forEach(c => {
    const isDuplicate = unique.some(u =>
      u.question.toLowerCase() === c.question.toLowerCase()
    )
    if (!isDuplicate) unique.push(c)
  })
  return unique
}

const uniqueClarifications = deduplicateClarifications(allClarifications)

// Multi-round clarification: batch questions (max 4 per round)
// ⚠️ MUST execute ALL rounds until uniqueClarifications exhausted
if (uniqueClarifications.length > 0) {
  const BATCH_SIZE = 4
  const totalRounds = Math.ceil(uniqueClarifications.length / BATCH_SIZE)

  for (let i = 0; i < uniqueClarifications.length; i += BATCH_SIZE) {
    const batch = uniqueClarifications.slice(i, i + BATCH_SIZE)
    const currentRound = Math.floor(i / BATCH_SIZE) + 1

    console.log(`### Clarification Round ${currentRound}/${totalRounds}`)

    AskUserQuestion({
      questions: batch.map(need => ({
        question: `[${need.source_angle}] ${need.question}\n\nContext: ${need.context}`,
        header: need.source_angle,
        multiSelect: false,
        options: need.options.map((opt, index) => {
          const isRecommended = need.recommended === index
          return {
            label: isRecommended ? `${opt} ★` : opt,
            description: isRecommended ? `Use ${opt} approach (Recommended)` : `Use ${opt} approach`
          }
        })
      }))
    })

    // Store batch responses in clarificationContext before next round
  }
}
```

**Output**: `clarificationContext` (in-memory)

---

### Phase 3: Fix Planning

**Planning Strategy Selection** (based on Phase 1 severity):

**IMPORTANT**: Phase 3 is **planning only** - NO code execution. All execution happens in Phase 5 via lite-execute.

**Low/Medium Severity** - Direct planning by Claude:
```javascript
// Step 1: Read schema
const schema = Bash(`cat ~/.claude/workflows/cli-templates/schemas/fix-plan-json-schema.json`)

// Step 2: Generate fix-plan following schema (Claude directly, no agent)
const fixPlan = {
  summary: "...",
  root_cause: "...",
  strategy: "immediate_patch|comprehensive_fix|refactor",
  tasks: [...],  // Each task: { id, title, scope, ..., depends_on, complexity }
  estimated_time: "...",
  recommended_execution: "Agent",
  severity: severity,
  risk_level: "...",
  _metadata: { timestamp: getUtc8ISOString(), source: "direct-planning", planning_mode: "direct" }
}

// Step 3: Write fix-plan to session folder
Write(`${sessionFolder}/fix-plan.json`, JSON.stringify(fixPlan, null, 2))

// Step 4: MUST continue to Phase 4 (Confirmation) - DO NOT execute code here
```

**High/Critical Severity** - Invoke cli-lite-planning-agent:

```javascript
Task(
  subagent_type="cli-lite-planning-agent",
  description="Generate detailed fix plan",
  prompt=`
Generate fix plan and write fix-plan.json.

## Output Schema Reference
Execute: cat ~/.claude/workflows/cli-templates/schemas/fix-plan-json-schema.json (get schema reference before generating plan)

## Bug Description
${bug_description}

## Multi-Angle Diagnosis Context

${manifest.diagnoses.map(diag => `### Diagnosis: ${diag.angle} (${diag.file})
Path: ${diag.path}

Read this file for detailed ${diag.angle} analysis.`).join('\n\n')}

Total diagnoses: ${manifest.diagnosis_count}
Angles covered: ${manifest.diagnoses.map(d => d.angle).join(', ')}

Manifest: ${sessionFolder}/diagnoses-manifest.json

## User Clarifications
${JSON.stringify(clarificationContext) || "None"}

## Severity Level
${severity}

## Requirements
Generate fix-plan.json with:
- summary: 2-3 sentence overview of the fix
- root_cause: Consolidated root cause from all diagnoses
- strategy: "immediate_patch" | "comprehensive_fix" | "refactor"
- tasks: 1-5 structured fix tasks (**IMPORTANT: group by fix area, NOT by file**)
  - **Task Granularity Principle**: Each task = one complete fix unit
  - title: action verb + target (e.g., "Fix token validation edge case")
  - scope: module path (src/auth/) or feature name
  - action: "Fix" | "Update" | "Refactor" | "Add" | "Delete"
  - description
  - modification_points: ALL files to modify for this fix (group related changes)
  - implementation (2-5 steps covering all modification_points)
  - verification (test criteria)
  - depends_on: task IDs this task depends on (use sparingly)
- estimated_time, recommended_execution, severity, risk_level
- _metadata:
  - timestamp, source, planning_mode
  - diagnosis_angles: ${JSON.stringify(manifest.diagnoses.map(d => d.angle))}

## Task Grouping Rules
1. **Group by fix area**: All changes for one fix = one task (even if 2-3 files)
2. **Avoid file-per-task**: Do NOT create separate tasks for each file
3. **Substantial tasks**: Each task should represent 10-45 minutes of work
4. **True dependencies only**: Only use depends_on when Task B cannot start without Task A's output
5. **Prefer parallel**: Most tasks should be independent (no depends_on)

## Execution
1. Read ALL diagnosis files for comprehensive context
2. Execute CLI planning using Gemini (Qwen fallback)
3. Synthesize findings from multiple diagnosis angles
4. Parse output and structure fix-plan
5. Write JSON: Write('${sessionFolder}/fix-plan.json', jsonContent)
6. Return brief completion summary
`
)
```

**Output**: `${sessionFolder}/fix-plan.json`

---

### Phase 4: Task Confirmation & Execution Selection

**Step 4.1: Display Fix Plan**
```javascript
const fixPlan = JSON.parse(Read(`${sessionFolder}/fix-plan.json`))

console.log(`
## Fix Plan

**Summary**: ${fixPlan.summary}
**Root Cause**: ${fixPlan.root_cause}
**Strategy**: ${fixPlan.strategy}

**Tasks** (${fixPlan.tasks.length}):
${fixPlan.tasks.map((t, i) => `${i+1}. ${t.title} (${t.scope})`).join('\n')}

**Severity**: ${fixPlan.severity}
**Risk Level**: ${fixPlan.risk_level}
**Estimated Time**: ${fixPlan.estimated_time}
**Recommended**: ${fixPlan.recommended_execution}
`)
```

**Step 4.2: Collect Confirmation**
```javascript
AskUserQuestion({
  questions: [
    {
      question: `Confirm fix plan? (${fixPlan.tasks.length} tasks, ${fixPlan.severity} severity)`,
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
        { label: "Auto", description: `Auto: ${fixPlan.severity === 'Low' ? 'Agent' : 'Codex'}` }
      ]
    },
    {
      question: "Code review after fix?",
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

**CRITICAL**: lite-fix NEVER executes code directly. ALL execution MUST go through lite-execute.

**Step 5.1: Build executionContext**

```javascript
// Load manifest and all diagnosis files
const manifest = JSON.parse(Read(`${sessionFolder}/diagnoses-manifest.json`))
const diagnoses = {}

manifest.diagnoses.forEach(diag => {
  if (file_exists(diag.path)) {
    diagnoses[diag.angle] = JSON.parse(Read(diag.path))
  }
})

const fixPlan = JSON.parse(Read(`${sessionFolder}/fix-plan.json`))

executionContext = {
  mode: "bugfix",
  severity: fixPlan.severity,
  planObject: fixPlan,
  diagnosisContext: diagnoses,
  diagnosisAngles: manifest.diagnoses.map(d => d.angle),
  diagnosisManifest: manifest,
  clarificationContext: clarificationContext || null,
  executionMethod: userSelection.execution_method,
  codeReviewTool: userSelection.code_review_tool,
  originalUserInput: bug_description,
  session: {
    id: sessionId,
    folder: sessionFolder,
    artifacts: {
      diagnoses: manifest.diagnoses.map(diag => ({
        angle: diag.angle,
        path: diag.path
      })),
      diagnoses_manifest: `${sessionFolder}/diagnoses-manifest.json`,
      fix_plan: `${sessionFolder}/fix-plan.json`
    }
  }
}
```

**Step 5.2: Dispatch**

```javascript
SlashCommand(command="/workflow:lite-execute --in-memory --mode bugfix")
```

## Session Folder Structure

```
.workflow/.lite-fix/{bug-slug}-{YYYY-MM-DD}/
|- diagnosis-{angle1}.json      # Diagnosis angle 1
|- diagnosis-{angle2}.json      # Diagnosis angle 2
|- diagnosis-{angle3}.json      # Diagnosis angle 3 (if applicable)
|- diagnosis-{angle4}.json      # Diagnosis angle 4 (if applicable)
|- diagnoses-manifest.json      # Diagnosis index
+- fix-plan.json                # Fix plan
```

**Example**:
```
.workflow/.lite-fix/user-avatar-upload-fails-413-2025-11-25-14-30-25/
|- diagnosis-error-handling.json
|- diagnosis-dataflow.json
|- diagnosis-validation.json
|- diagnoses-manifest.json
+- fix-plan.json
```

## Error Handling

| Error | Resolution |
|-------|------------|
| Diagnosis agent failure | Skip diagnosis, continue with bug description only |
| Planning agent failure | Fallback to direct planning by Claude |
| Clarification timeout | Use diagnosis findings as-is |
| Confirmation timeout | Save context, display resume instructions |
| Modify loop > 3 times | Suggest breaking task or using /workflow:plan |
| Root cause unclear | Extend diagnosis time or use broader angles |
| Too complex for lite-fix | Escalate to /workflow:plan --mode bugfix |


