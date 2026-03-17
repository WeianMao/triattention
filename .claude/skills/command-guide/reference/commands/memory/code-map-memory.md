---
name: code-map-memory
description: 3-phase orchestrator: parse feature keyword → cli-explore-agent analyzes (Deep Scan dual-source) → orchestrator generates Mermaid docs + SKILL package (skips phase 2 if exists)
argument-hint: "\"feature-keyword\" [--regenerate] [--tool <gemini|qwen>]"
allowed-tools: SlashCommand(*), TodoWrite(*), Bash(*), Read(*), Write(*), Task(*)
---

# Code Flow Mapping Generator

## Overview

**Pure Orchestrator with Agent Delegation**: Prepares context paths and delegates code flow analysis to specialized cli-explore-agent. Orchestrator transforms agent's JSON analysis into Mermaid documentation.

**Auto-Continue Workflow**: Runs fully autonomously once triggered. Each phase completes and automatically triggers the next phase.

**Execution Paths**:
- **Full Path**: All 3 phases (no existing codemap OR `--regenerate` specified)
- **Skip Path**: Phase 1 → Phase 3 (existing codemap found AND no `--regenerate` flag)
- **Phase 3 Always Executes**: SKILL index is always generated or updated

**Agent Responsibility** (cli-explore-agent):
- Deep code flow analysis using dual-source strategy (Bash + Gemini CLI)
- Returns structured JSON with architecture, functions, data flow, conditionals, patterns
- NO file writing - analysis only

**Orchestrator Responsibility**:
- Provides feature keyword and analysis scope to agent
- Transforms agent's JSON into Mermaid-enriched markdown documentation
- Writes all files (5 docs + metadata.json + SKILL.md)

## Core Rules

1. **Start Immediately**: First action is TodoWrite initialization, second action is Phase 1 execution
2. **Feature-Specific SKILL**: Each feature creates independent `.claude/skills/codemap-{feature}/` package
3. **Specialized Agent**: Phase 2a uses cli-explore-agent for professional code analysis (Deep Scan mode)
4. **Orchestrator Documentation**: Phase 2b transforms agent JSON into Mermaid markdown files
5. **Auto-Continue**: After completing each phase, update TodoWrite and immediately execute next phase
6. **No User Prompts**: Never ask user questions or wait for input between phases
7. **Track Progress**: Update TodoWrite after EVERY phase completion before starting next phase
8. **Multi-Level Detail**: Generate 4 levels: architecture → function → data → conditional

---

## 3-Phase Execution

### Phase 1: Parse Feature Keyword & Check Existing

**Goal**: Normalize feature keyword, check existing codemap, prepare for analysis

**Step 1: Parse Feature Keyword**
```bash
# Get feature keyword from argument
FEATURE_KEYWORD="$1"

# Normalize: lowercase, spaces to hyphens
normalized_feature=$(echo "$FEATURE_KEYWORD" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr '_' '-')

# Example: "User Authentication" → "user-authentication"
# Example: "支付处理" → "支付处理" (keep non-ASCII)
```

**Step 2: Set Tool Preference**
```bash
# Default to gemini unless --tool specified
TOOL="${tool_flag:-gemini}"
```

**Step 3: Check Existing Codemap**
```bash
# Define codemap directory
CODEMAP_DIR=".claude/skills/codemap-${normalized_feature}"

# Check if codemap exists
bash(test -d "$CODEMAP_DIR" && echo "exists" || echo "not_exists")

# Count existing files
bash(find "$CODEMAP_DIR" -name "*.md" 2>/dev/null | wc -l || echo 0)
```

**Step 4: Skip Decision**
```javascript
if (existing_files > 0 && !regenerate_flag) {
  SKIP_GENERATION = true
  message = "Codemap already exists, skipping Phase 2. Use --regenerate to force regeneration."
} else if (regenerate_flag) {
  bash(rm -rf "$CODEMAP_DIR")
  SKIP_GENERATION = false
  message = "Regenerating codemap from scratch."
} else {
  SKIP_GENERATION = false
  message = "No existing codemap found, generating new code flow analysis."
}
```

**Output Variables**:
- `FEATURE_KEYWORD`: Original feature keyword
- `normalized_feature`: Normalized feature name for directory
- `CODEMAP_DIR`: `.claude/skills/codemap-{feature}`
- `TOOL`: CLI tool to use (gemini or qwen)
- `SKIP_GENERATION`: Boolean - whether to skip Phase 2

**TodoWrite**:
- If skipping: Mark phase 1 completed, phase 2 completed, phase 3 in_progress
- If not skipping: Mark phase 1 completed, phase 2 in_progress

---

### Phase 2: Code Flow Analysis & Documentation Generation

**Skip Condition**: Skipped if `SKIP_GENERATION = true`

**Goal**: Use cli-explore-agent for professional code analysis, then orchestrator generates Mermaid documentation

**Architecture**: Phase 2a (Agent Analysis) → Phase 2b (Orchestrator Documentation)

---

#### Phase 2a: cli-explore-agent Analysis

**Purpose**: Leverage specialized cli-explore-agent for deep code flow analysis

**Agent Task Specification**:

```
Task(
  subagent_type: "cli-explore-agent",
  description: "Analyze code flow: {FEATURE_KEYWORD}",
  prompt: "
Perform Deep Scan analysis for feature: {FEATURE_KEYWORD}

**Analysis Mode**: deep-scan (Dual-source: Bash structural scan + Gemini semantic analysis)

**Analysis Objectives**:
1. **Module Architecture**: Identify high-level module organization, interactions, and entry points
2. **Function Call Chains**: Trace execution paths, call sequences, and parameter flows
3. **Data Transformations**: Map data structure changes and transformation stages
4. **Conditional Paths**: Document decision trees, branches, and error handling strategies
5. **Design Patterns**: Discover architectural patterns and extract design intent

**Scope**:
- Feature: {FEATURE_KEYWORD}
- CLI Tool: {TOOL} (gemini-2.5-pro or qwen coder-model)
- File Discovery: MCP Code Index (preferred) + rg fallback
- Target: 5-15 most relevant files

**MANDATORY FIRST STEP**:
Read: ~/.claude/workflows/cli-templates/schemas/codemap-json-schema.json

**Output**: Return JSON following schema exactly. NO FILE WRITING - return JSON analysis only.

**Critical Requirements**:
- Use Deep Scan mode: Bash (Phase 1 - precise locations) + Gemini CLI (Phase 2 - semantic understanding) + Synthesis (Phase 3 - merge with attribution)
- Focus exclusively on {FEATURE_KEYWORD} feature flow
- Include file:line references for ALL findings
- Extract design intent from code structure and comments
- NO FILE WRITING - return JSON analysis only
- Handle tool failures gracefully (Gemini → Qwen fallback, MCP → rg fallback)
  "
)
```

**Agent Output**: JSON analysis result with architecture, functions, data flow, conditionals, and patterns

---

#### Phase 2b: Orchestrator Documentation Generation

**Purpose**: Transform cli-explore-agent JSON into Mermaid-enriched documentation

**Input**: Agent's JSON analysis result

**Process**:

1. **Parse Agent Analysis**:
   ```javascript
   const analysis = JSON.parse(agentResult)
   const { feature, files_analyzed, architecture, function_calls, data_flow, conditional_logic, design_patterns } = analysis
   ```

2. **Generate Mermaid Diagrams from Structured Data**:

   **a) architecture-flow.md** (~3K tokens):
   ```javascript
   // Convert architecture.modules + architecture.interactions → Mermaid graph TD
   const architectureMermaid = `
   graph TD
   ${architecture.modules.map(m => `    ${m.name}[${m.name}]`).join('\n')}
   ${architecture.interactions.map(i => `    ${i.from} -->|${i.type}| ${i.to}`).join('\n')}
   `

   Write({
     file_path: `${CODEMAP_DIR}/architecture-flow.md`,
     content: `---
feature: ${feature}
level: architecture
detail: high-level module interactions
---
# Architecture Flow: ${feature}

## Overview
${architecture.overview}

## Module Architecture
${architecture.modules.map(m => `### ${m.name}\n- **File**: ${m.file}\n- **Role**: ${m.responsibility}\n- **Dependencies**: ${m.dependencies.join(', ')}`).join('\n\n')}

## Flow Diagram
\`\`\`mermaid
${architectureMermaid}
\`\`\`

## Key Interactions
${architecture.interactions.map(i => `- **${i.from} → ${i.to}**: ${i.description}`).join('\n')}

## Entry Points
${architecture.entry_points.map(e => `- **${e.function}** (${e.file}): ${e.description}`).join('\n')}
`
   })
   ```

   **b) function-calls.md** (~5K tokens):
   ```javascript
   // Convert function_calls.sequences → Mermaid sequenceDiagram
   const sequenceMermaid = `
   sequenceDiagram
   ${function_calls.sequences.map(s => `    ${s.from}->>${s.to}: ${s.method}`).join('\n')}
   `

   Write({
     file_path: `${CODEMAP_DIR}/function-calls.md`,
     content: `---
feature: ${feature}
level: function
detail: function-level call sequences
---
# Function Call Chains: ${feature}

## Call Sequence Diagram
\`\`\`mermaid
${sequenceMermaid}
\`\`\`

## Detailed Call Chains
${function_calls.call_chains.map(chain => `
### Chain ${chain.chain_id}: ${chain.description}
${chain.sequence.map(fn => `- **${fn.function}** (${fn.file})\n  - Calls: ${fn.calls.join(', ')}`).join('\n')}
`).join('\n')}

## Parameters & Returns
${function_calls.sequences.map(s => `- **${s.method}** → Returns: ${s.returns || 'void'}`).join('\n')}
`
   })
   ```

   **c) data-flow.md** (~4K tokens):
   ```javascript
   // Convert data_flow.transformations → Mermaid flowchart LR
   const dataFlowMermaid = `
   flowchart LR
   ${data_flow.transformations.map((t, i) => `    Stage${i}[${t.from}] -->|${t.transformer}| Stage${i+1}[${t.to}]`).join('\n')}
   `

   Write({
     file_path: `${CODEMAP_DIR}/data-flow.md`,
     content: `---
feature: ${feature}
level: data
detail: data structure transformations
---
# Data Flow: ${feature}

## Data Transformation Diagram
\`\`\`mermaid
${dataFlowMermaid}
\`\`\`

## Data Structures
${data_flow.structures.map(s => `### ${s.name} (${s.stage})\n\`\`\`json\n${JSON.stringify(s.shape, null, 2)}\n\`\`\``).join('\n\n')}

## Transformations
${data_flow.transformations.map(t => `- **${t.from} → ${t.to}** via \`${t.transformer}\` (${t.file})`).join('\n')}
`
   })
   ```

   **d) conditional-paths.md** (~4K tokens):
   ```javascript
   // Convert conditional_logic.branches → Mermaid flowchart TD
   const conditionalMermaid = `
   flowchart TD
       Start[Entry Point]
   ${conditional_logic.branches.map((b, i) => `
       Start --> Check${i}{${b.condition}}
       Check${i} -->|Yes| Path${i}A[${b.true_path}]
       Check${i} -->|No| Path${i}B[${b.false_path}]
   `).join('\n')}
   `

   Write({
     file_path: `${CODEMAP_DIR}/conditional-paths.md`,
     content: `---
feature: ${feature}
level: conditional
detail: decision trees and error paths
---
# Conditional Paths: ${feature}

## Decision Tree
\`\`\`mermaid
${conditionalMermaid}
\`\`\`

## Branch Conditions
${conditional_logic.branches.map(b => `- **${b.condition}** (${b.file})\n  - True: ${b.true_path}\n  - False: ${b.false_path}`).join('\n')}

## Error Handling
${conditional_logic.error_handling.map(e => `- **${e.error_type}**: Handler \`${e.handler}\` (${e.file}) - Recovery: ${e.recovery}`).join('\n')}
`
   })
   ```

   **e) complete-flow.md** (~8K tokens):
   ```javascript
   // Integrate all Mermaid diagrams
   Write({
     file_path: `${CODEMAP_DIR}/complete-flow.md`,
     content: `---
feature: ${feature}
level: complete
detail: integrated multi-level view
---
# Complete Flow: ${feature}

## Integrated Flow Diagram
\`\`\`mermaid
graph TB
    subgraph Architecture
    ${architecture.modules.map(m => `    ${m.name}[${m.name}]`).join('\n')}
    end

    subgraph "Function Calls"
    ${function_calls.call_chains[0]?.sequence.map(fn => `    ${fn.function}`).join('\n') || ''}
    end

    subgraph "Data Flow"
    ${data_flow.structures.map(s => `    ${s.name}[${s.name}]`).join('\n')}
    end
\`\`\`

## Complete Trace
[Comprehensive end-to-end documentation combining all analysis layers]

## Design Patterns Identified
${design_patterns.map(p => `- **${p.pattern}** in ${p.location}: ${p.description}`).join('\n')}

## Recommendations
${analysis.recommendations.map(r => `- ${r}`).join('\n')}

## Cross-References
- [Architecture Flow](./architecture-flow.md) - High-level module structure
- [Function Calls](./function-calls.md) - Detailed call chains
- [Data Flow](./data-flow.md) - Data transformation stages
- [Conditional Paths](./conditional-paths.md) - Decision trees and error handling
`
   })
   ```

3. **Write metadata.json**:
   ```javascript
   Write({
     file_path: `${CODEMAP_DIR}/metadata.json`,
     content: JSON.stringify({
       feature: feature,
       normalized_name: normalized_feature,
       generated_at: new Date().toISOString(),
       tool_used: analysis.analysis_metadata.tool_used,
       files_analyzed: files_analyzed.map(f => f.file),
       analysis_summary: {
         total_files: files_analyzed.length,
         modules_traced: architecture.modules.length,
         functions_traced: function_calls.call_chains.reduce((sum, c) => sum + c.sequence.length, 0),
         patterns_discovered: design_patterns.length
       }
     }, null, 2)
   })
   ```

4. **Report Phase 2 Completion**:
   ```
   Phase 2 Complete: Code flow analysis and documentation generated

   - Agent Analysis: cli-explore-agent with {TOOL}
   - Files Analyzed: {count}
   - Documentation Generated: 5 markdown files + metadata.json
   - Location: {CODEMAP_DIR}
   ```

**Completion Criteria**:
- cli-explore-agent task completed successfully with JSON result
- 5 documentation files written with valid Mermaid diagrams
- metadata.json written with analysis summary
- All files properly formatted and cross-referenced

**TodoWrite**: Mark phase 2 completed, phase 3 in_progress

---

### Phase 3: Generate SKILL.md Index

**Note**: This phase **ALWAYS executes** - generates or updates the SKILL index.

**Goal**: Read generated flow documentation and create SKILL.md index with progressive loading

**Steps**:

1. **Verify Generated Files**:
   ```bash
   bash(find "{CODEMAP_DIR}" -name "*.md" -type f | sort)
   ```

2. **Read metadata.json**:
   ```javascript
   Read({CODEMAP_DIR}/metadata.json)
   // Extract: feature, normalized_name, files_analyzed, analysis_summary
   ```

3. **Read File Headers** (optional, first 30 lines):
   ```javascript
   Read({CODEMAP_DIR}/architecture-flow.md, limit: 30)
   Read({CODEMAP_DIR}/function-calls.md, limit: 30)
   // Extract overview and diagram counts
   ```

4. **Generate SKILL.md Index**:

   Template structure:
   ```yaml
   ---
   name: codemap-{normalized_feature}
   description: Code flow mapping for {FEATURE_KEYWORD} feature (located at {project_path}). Load this SKILL when analyzing, tracing, or understanding {FEATURE_KEYWORD} execution flow, especially when no relevant context exists in memory.
   version: 1.0.0
   generated_at: {ISO_TIMESTAMP}
   ---
   # Code Flow Map: {FEATURE_KEYWORD}

   ## Feature: `{FEATURE_KEYWORD}`

   **Analysis Date**: {DATE}
   **Tool Used**: {TOOL}
   **Files Analyzed**: {COUNT}

   ## Progressive Loading

   ### Level 0: Quick Overview (~2K tokens)
   - [Architecture Flow](./architecture-flow.md) - High-level module interactions

   ### Level 1: Core Flows (~10K tokens)
   - [Architecture Flow](./architecture-flow.md) - Module architecture
   - [Function Calls](./function-calls.md) - Function call chains

   ### Level 2: Complete Analysis (~20K tokens)
   - [Architecture Flow](./architecture-flow.md)
   - [Function Calls](./function-calls.md)
   - [Data Flow](./data-flow.md) - Data transformations

   ### Level 3: Deep Dive (~30K tokens)
   - [Architecture Flow](./architecture-flow.md)
   - [Function Calls](./function-calls.md)
   - [Data Flow](./data-flow.md)
   - [Conditional Paths](./conditional-paths.md) - Branches and error handling
   - [Complete Flow](./complete-flow.md) - Integrated comprehensive view

   ## Usage

   Load this SKILL package when:
   - Analyzing {FEATURE_KEYWORD} implementation
   - Tracing execution flow for debugging
   - Understanding code dependencies
   - Planning refactoring or enhancements

   ## Analysis Summary

   - **Modules Traced**: {modules_traced}
   - **Functions Traced**: {functions_traced}
   - **Files Analyzed**: {total_files}

   ## Mermaid Diagrams Included

   - Architecture flow diagram (graph TD)
   - Function call sequence diagram (sequenceDiagram)
   - Data transformation flowchart (flowchart LR)
   - Conditional decision tree (flowchart TD)
   - Complete integrated diagram (graph TB)
   ```

5. **Write SKILL.md**:
   ```javascript
   Write({
     file_path: `{CODEMAP_DIR}/SKILL.md`,
     content: generatedIndexMarkdown
   })
   ```

**Completion Criteria**:
- SKILL.md index written
- All documentation files verified
- Progressive loading levels (0-3) properly structured
- Mermaid diagram references included

**TodoWrite**: Mark phase 3 completed

**Final Report**:
```
Code Flow Mapping Complete

Feature: {FEATURE_KEYWORD}
Location: .claude/skills/codemap-{normalized_feature}/

Files Generated:
- SKILL.md (index)
- architecture-flow.md (with Mermaid diagram)
- function-calls.md (with Mermaid sequence diagram)
- data-flow.md (with Mermaid flowchart)
- conditional-paths.md (with Mermaid decision tree)
- complete-flow.md (with integrated Mermaid diagram)
- metadata.json

Analysis:
- Files analyzed: {count}
- Modules traced: {count}
- Functions traced: {count}

Usage: Skill(command: "codemap-{normalized_feature}")
```

---

## Implementation Details

### TodoWrite Patterns

**Initialization** (Before Phase 1):
```javascript
TodoWrite({todos: [
  {"content": "Parse feature keyword and check existing", "status": "in_progress", "activeForm": "Parsing feature keyword"},
  {"content": "Agent analyzes code flow and generates files", "status": "pending", "activeForm": "Analyzing code flow"},
  {"content": "Generate SKILL.md index", "status": "pending", "activeForm": "Generating SKILL index"}
]})
```

**Full Path** (SKIP_GENERATION = false):
```javascript
// After Phase 1
TodoWrite({todos: [
  {"content": "Parse feature keyword and check existing", "status": "completed", ...},
  {"content": "Agent analyzes code flow and generates files", "status": "in_progress", ...},
  {"content": "Generate SKILL.md index", "status": "pending", ...}
]})

// After Phase 2
TodoWrite({todos: [
  {"content": "Parse feature keyword and check existing", "status": "completed", ...},
  {"content": "Agent analyzes code flow and generates files", "status": "completed", ...},
  {"content": "Generate SKILL.md index", "status": "in_progress", ...}
]})

// After Phase 3
TodoWrite({todos: [
  {"content": "Parse feature keyword and check existing", "status": "completed", ...},
  {"content": "Agent analyzes code flow and generates files", "status": "completed", ...},
  {"content": "Generate SKILL.md index", "status": "completed", ...}
]})
```

**Skip Path** (SKIP_GENERATION = true):
```javascript
// After Phase 1 (skip Phase 2)
TodoWrite({todos: [
  {"content": "Parse feature keyword and check existing", "status": "completed", ...},
  {"content": "Agent analyzes code flow and generates files", "status": "completed", ...},  // Skipped
  {"content": "Generate SKILL.md index", "status": "in_progress", ...}
]})
```

### Execution Flow

**Full Path**:
```
User → TodoWrite Init → Phase 1 (parse) → Phase 2 (agent analyzes) → Phase 3 (write index) → Report
```

**Skip Path**:
```
User → TodoWrite Init → Phase 1 (detect existing) → Phase 3 (update index) → Report
```

### Error Handling

**Phase 1 Errors**:
- Empty feature keyword: Report error, ask user to provide feature description
- Invalid characters: Normalize and continue

**Phase 2 Errors (Agent)**:
- Agent task fails: Retry once, report if fails again
- No files discovered: Warn user, ask for more specific feature keyword
- CLI failures: Agent handles internally with retries
- Invalid Mermaid syntax: Agent validates before writing

**Phase 3 Errors**:
- Write failures: Report which files failed
- Missing files: Note in SKILL.md, suggest regeneration

---

## Parameters

```bash
/memory:code-map-memory "feature-keyword" [--regenerate] [--tool <gemini|qwen>]
```

**Arguments**:
- **"feature-keyword"**: Feature or flow to analyze (required)
  - Examples: `"user authentication"`, `"payment processing"`, `"数据导入流程"`
  - Can be English, Chinese, or mixed
  - Spaces and underscores normalized to hyphens
- **--regenerate**: Force regenerate existing codemap (deletes and recreates)
- **--tool**: CLI tool for analysis (default: gemini)
  - `gemini`: Comprehensive flow analysis with gemini-2.5-pro
  - `qwen`: Alternative with coder-model

---

## Examples

**Generated File Structure** (for all examples):
```
.claude/skills/codemap-{feature}/
├── SKILL.md                    # Index (Phase 3)
├── architecture-flow.md        # Agent (Phase 2) - High-level flow
├── function-calls.md           # Agent (Phase 2) - Function chains
├── data-flow.md                # Agent (Phase 2) - Data transformations
├── conditional-paths.md        # Agent (Phase 2) - Branches & errors
├── complete-flow.md            # Agent (Phase 2) - Integrated view
└── metadata.json               # Agent (Phase 2)
```

### Example 1: User Authentication Flow

```bash
/memory:code-map-memory "user authentication"
```

**Workflow**:
1. Phase 1: Normalizes to "user-authentication", checks existing codemap
2. Phase 2: Agent discovers auth-related files, executes CLI analysis, generates 5 flow docs with Mermaid
3. Phase 3: Generates SKILL.md index with progressive loading

**Output**: `.claude/skills/codemap-user-authentication/` with 6 files + metadata


### Example 3: Regenerate with Qwen

```bash
/memory:code-map-memory "payment processing" --regenerate --tool qwen
```

**Workflow**:
1. Phase 1: Deletes existing codemap due to --regenerate
2. Phase 2: Agent uses qwen with coder-model for fresh analysis
3. Phase 3: Generates updated SKILL.md

---


## Architecture

```
code-map-memory (orchestrator)
  ├─ Phase 1: Parse & Check (bash commands, skip decision)
  ├─ Phase 2: Code Analysis & Documentation (skippable)
  │   ├─ Phase 2a: cli-explore-agent Analysis
  │   │   └─ Deep Scan: Bash structural + Gemini semantic → JSON
  │   └─ Phase 2b: Orchestrator Documentation
  │       └─ Transform JSON → 5 Mermaid markdown files + metadata.json
  └─ Phase 3: Write SKILL.md (index generation, always runs)

Output: .claude/skills/codemap-{feature}/
```
