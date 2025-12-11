---
name: tech-research
description: 3-phase orchestrator: extract tech stack from session/name → delegate to agent for Exa research and module generation → generate SKILL.md index (skips phase 2 if exists)
argument-hint: "[session-id | tech-stack-name] [--regenerate] [--tool <gemini|qwen>]"
allowed-tools: SlashCommand(*), TodoWrite(*), Bash(*), Read(*), Write(*), Task(*)
---

# Tech Stack Research SKILL Generator

## Overview

**Pure Orchestrator with Agent Delegation**: Prepares context paths and delegates ALL work to agent. Agent produces files directly.

**Auto-Continue Workflow**: Runs fully autonomously once triggered. Each phase completes and automatically triggers the next phase.

**Execution Paths**:
- **Full Path**: All 3 phases (no existing SKILL OR `--regenerate` specified)
- **Skip Path**: Phase 1 → Phase 3 (existing SKILL found AND no `--regenerate` flag)
- **Phase 3 Always Executes**: SKILL index is always generated or updated

**Agent Responsibility**:
- Agent does ALL the work: context reading, Exa research, content synthesis, file writing
- Orchestrator only provides context paths and waits for completion

## Core Rules

1. **Start Immediately**: First action is TodoWrite initialization, second action is Phase 1 execution
2. **Context Path Delegation**: Pass session directory or tech stack name to agent, let agent do discovery
3. **Agent Produces Files**: Agent directly writes all module files, orchestrator does NOT parse agent output
4. **Auto-Continue**: After completing each phase, update TodoWrite and immediately execute next phase
5. **No User Prompts**: Never ask user questions or wait for input between phases
6. **Track Progress**: Update TodoWrite after EVERY phase completion before starting next phase
7. **Lightweight Index**: Phase 3 only generates SKILL.md index by reading existing files

---

## 3-Phase Execution

### Phase 1: Prepare Context Paths

**Goal**: Detect input mode, prepare context paths for agent, check existing SKILL

**Input Mode Detection**:
```bash
# Get input parameter
input="$1"

# Detect mode
if [[ "$input" == WFS-* ]]; then
  MODE="session"
  SESSION_ID="$input"
  CONTEXT_PATH=".workflow/${SESSION_ID}"
else
  MODE="direct"
  TECH_STACK_NAME="$input"
  CONTEXT_PATH="$input"  # Pass tech stack name as context
fi
```

**Check Existing SKILL**:
```bash
# For session mode, peek at session to get tech stack name
if [[ "$MODE" == "session" ]]; then
  bash(test -f ".workflow/${SESSION_ID}/workflow-session.json")
  Read(.workflow/${SESSION_ID}/workflow-session.json)
  # Extract tech_stack_name (minimal extraction)
fi

# Normalize and check
normalized_name=$(echo "$TECH_STACK_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
bash(test -d ".claude/skills/${normalized_name}" && echo "exists" || echo "not_exists")
bash(find ".claude/skills/${normalized_name}" -name "*.md" 2>/dev/null | wc -l || echo 0)
```

**Skip Decision**:
```javascript
if (existing_files > 0 && !regenerate_flag) {
  SKIP_GENERATION = true
  message = "Tech stack SKILL already exists, skipping Phase 2. Use --regenerate to force regeneration."
} else if (regenerate_flag) {
  bash(rm -rf ".claude/skills/${normalized_name}")
  SKIP_GENERATION = false
  message = "Regenerating tech stack SKILL from scratch."
} else {
  SKIP_GENERATION = false
  message = "No existing SKILL found, generating new tech stack documentation."
}
```

**Output Variables**:
- `MODE`: `session` or `direct`
- `SESSION_ID`: Session ID (if session mode)
- `CONTEXT_PATH`: Path to session directory OR tech stack name
- `TECH_STACK_NAME`: Extracted or provided tech stack name
- `SKIP_GENERATION`: Boolean - whether to skip Phase 2

**TodoWrite**:
- If skipping: Mark phase 1 completed, phase 2 completed, phase 3 in_progress
- If not skipping: Mark phase 1 completed, phase 2 in_progress

---

### Phase 2: Agent Produces All Files

**Skip Condition**: Skipped if `SKIP_GENERATION = true`

**Goal**: Delegate EVERYTHING to agent - context reading, Exa research, content synthesis, and file writing

**Agent Task Specification**:

```
Task(
  subagent_type: "general-purpose",
  description: "Generate tech stack SKILL: {CONTEXT_PATH}",
  prompt: "
Generate a complete tech stack SKILL package with Exa research.

**Context Provided**:
- Mode: {MODE}
- Context Path: {CONTEXT_PATH}

**Templates Available**:
- Module Format: ~/.claude/workflows/cli-templates/prompts/tech/tech-module-format.txt
- SKILL Index: ~/.claude/workflows/cli-templates/prompts/tech/tech-skill-index.txt

**Your Responsibilities**:

1. **Extract Tech Stack Information**:

   IF MODE == 'session':
     - Read `.workflow/active/{session_id}/workflow-session.json`
     - Read `.workflow/active/{session_id}/.process/context-package.json`
     - Extract tech_stack: {language, frameworks, libraries}
     - Build tech stack name: \"{language}-{framework1}-{framework2}\"
     - Example: \"typescript-react-nextjs\"

   IF MODE == 'direct':
     - Tech stack name = CONTEXT_PATH
     - Parse composite: split by '-' delimiter
     - Example: \"typescript-react-nextjs\" → [\"typescript\", \"react\", \"nextjs\"]

2. **Execute Exa Research** (4-6 parallel queries):

   Base Queries (always execute):
   - mcp__exa__get_code_context_exa(query: \"{tech} core principles best practices 2025\", tokensNum: 8000)
   - mcp__exa__get_code_context_exa(query: \"{tech} common patterns architecture examples\", tokensNum: 7000)
   - mcp__exa__web_search_exa(query: \"{tech} configuration setup tooling 2025\", numResults: 5)
   - mcp__exa__get_code_context_exa(query: \"{tech} testing strategies\", tokensNum: 5000)

   Component Queries (if composite):
   - For each additional component:
     mcp__exa__get_code_context_exa(query: \"{main_tech} {component} integration\", tokensNum: 5000)

3. **Read Module Format Template**:

   Read template for structure guidance:
   ```bash
   Read(~/.claude/workflows/cli-templates/prompts/tech/tech-module-format.txt)
   ```

4. **Synthesize Content into 6 Modules**:

   Follow template structure from tech-module-format.txt:
   - **principles.md** - Core concepts, philosophies (~3K tokens)
   - **patterns.md** - Implementation patterns with code examples (~5K tokens)
   - **practices.md** - Best practices, anti-patterns, pitfalls (~4K tokens)
   - **testing.md** - Testing strategies, frameworks (~3K tokens)
   - **config.md** - Setup, configuration, tooling (~3K tokens)
   - **frameworks.md** - Framework integration (only if composite, ~4K tokens)

   Each module follows template format:
   - Frontmatter (YAML)
   - Main sections with clear headings
   - Code examples from Exa research
   - Best practices sections
   - References to Exa sources

5. **Write Files Directly**:

   ```javascript
   // Create directory
   bash(mkdir -p \".claude/skills/{tech_stack_name}\")

   // Write each module file using Write tool
   Write({ file_path: \".claude/skills/{tech_stack_name}/principles.md\", content: ... })
   Write({ file_path: \".claude/skills/{tech_stack_name}/patterns.md\", content: ... })
   Write({ file_path: \".claude/skills/{tech_stack_name}/practices.md\", content: ... })
   Write({ file_path: \".claude/skills/{tech_stack_name}/testing.md\", content: ... })
   Write({ file_path: \".claude/skills/{tech_stack_name}/config.md\", content: ... })
   // Write frameworks.md only if composite

   // Write metadata.json
   Write({
     file_path: \".claude/skills/{tech_stack_name}/metadata.json\",
     content: JSON.stringify({
       tech_stack_name,
       components,
       is_composite,
       generated_at: timestamp,
       source: \"exa-research\",
       research_summary: { total_queries, total_sources }
     })
   })
   ```

6. **Report Completion**:

   Provide summary:
   - Tech stack name
   - Files created (count)
   - Exa queries executed
   - Sources consulted

**CRITICAL**:
- MUST read external template files before generating content (step 3 for modules, step 4 for index)
- You have FULL autonomy - read files, execute Exa, synthesize content, write files
- Do NOT return JSON or structured data - produce actual .md files
- Handle errors gracefully (Exa failures, missing files, template read failures)
- If tech stack cannot be determined, ask orchestrator to clarify
  "
)
```

**Completion Criteria**:
- Agent task executed successfully
- 5-6 modular files written to `.claude/skills/{tech_stack_name}/`
- metadata.json written
- Agent reports completion

**TodoWrite**: Mark phase 2 completed, phase 3 in_progress

---

### Phase 3: Generate SKILL.md Index

**Note**: This phase **ALWAYS executes** - generates or updates the SKILL index.

**Goal**: Read generated module files and create SKILL.md index with loading recommendations

**Steps**:

1. **Verify Generated Files**:
   ```bash
   bash(find ".claude/skills/${TECH_STACK_NAME}" -name "*.md" -type f | sort)
   ```

2. **Read metadata.json**:
   ```javascript
   Read(.claude/skills/${TECH_STACK_NAME}/metadata.json)
   // Extract: tech_stack_name, components, is_composite, research_summary
   ```

3. **Read Module Headers** (optional, first 20 lines):
   ```javascript
   Read(.claude/skills/${TECH_STACK_NAME}/principles.md, limit: 20)
   // Repeat for other modules
   ```

4. **Read SKILL Index Template**:

   ```javascript
   Read(~/.claude/workflows/cli-templates/prompts/tech/tech-skill-index.txt)
   ```

5. **Generate SKILL.md Index**:

   Follow template from tech-skill-index.txt with variable substitutions:
   - `{TECH_STACK_NAME}`: From metadata.json
   - `{MAIN_TECH}`: Primary technology
   - `{ISO_TIMESTAMP}`: Current timestamp
   - `{QUERY_COUNT}`: From research_summary
   - `{SOURCE_COUNT}`: From research_summary
   - Conditional sections for composite tech stacks

   Template provides structure for:
   - Frontmatter with metadata
   - Overview and tech stack description
   - Module organization (Core/Practical/Config sections)
   - Loading recommendations (Quick/Implementation/Complete)
   - Usage guidelines and auto-trigger keywords
   - Research metadata and version history

6. **Write SKILL.md**:
   ```javascript
   Write({
     file_path: `.claude/skills/${TECH_STACK_NAME}/SKILL.md`,
     content: generatedIndexMarkdown
   })
   ```

**Completion Criteria**:
- SKILL.md index written
- All module files verified
- Loading recommendations included

**TodoWrite**: Mark phase 3 completed

**Final Report**:
```
Tech Stack SKILL Package Complete

Tech Stack: {TECH_STACK_NAME}
Location: .claude/skills/{TECH_STACK_NAME}/

Files: SKILL.md + 5-6 modules + metadata.json
Exa Research: {queries} queries, {sources} sources

Usage: Skill(command: "{TECH_STACK_NAME}")
```

---

## Implementation Details

### TodoWrite Patterns

**Initialization** (Before Phase 1):
```javascript
TodoWrite({todos: [
  {"content": "Prepare context paths", "status": "in_progress", "activeForm": "Preparing context paths"},
  {"content": "Agent produces all module files", "status": "pending", "activeForm": "Agent producing files"},
  {"content": "Generate SKILL.md index", "status": "pending", "activeForm": "Generating SKILL index"}
]})
```

**Full Path** (SKIP_GENERATION = false):
```javascript
// After Phase 1
TodoWrite({todos: [
  {"content": "Prepare context paths", "status": "completed", ...},
  {"content": "Agent produces all module files", "status": "in_progress", ...},
  {"content": "Generate SKILL.md index", "status": "pending", ...}
]})

// After Phase 2
TodoWrite({todos: [
  {"content": "Prepare context paths", "status": "completed", ...},
  {"content": "Agent produces all module files", "status": "completed", ...},
  {"content": "Generate SKILL.md index", "status": "in_progress", ...}
]})

// After Phase 3
TodoWrite({todos: [
  {"content": "Prepare context paths", "status": "completed", ...},
  {"content": "Agent produces all module files", "status": "completed", ...},
  {"content": "Generate SKILL.md index", "status": "completed", ...}
]})
```

**Skip Path** (SKIP_GENERATION = true):
```javascript
// After Phase 1 (skip Phase 2)
TodoWrite({todos: [
  {"content": "Prepare context paths", "status": "completed", ...},
  {"content": "Agent produces all module files", "status": "completed", ...},  // Skipped
  {"content": "Generate SKILL.md index", "status": "in_progress", ...}
]})
```

### Execution Flow

**Full Path**:
```
User → TodoWrite Init → Phase 1 (prepare) → Phase 2 (agent writes files) → Phase 3 (write index) → Report
```

**Skip Path**:
```
User → TodoWrite Init → Phase 1 (detect existing) → Phase 3 (update index) → Report
```

### Error Handling

**Phase 1 Errors**:
- Invalid session ID: Report error, verify session exists
- Missing context-package: Warn, fall back to direct mode
- No tech stack detected: Ask user to specify tech stack name

**Phase 2 Errors (Agent)**:
- Agent task fails: Retry once, report if fails again
- Exa API failures: Agent handles internally with retries
- Incomplete results: Warn user, proceed with partial data if minimum sections available

**Phase 3 Errors**:
- Write failures: Report which files failed
- Missing files: Note in SKILL.md, suggest regeneration

---

## Parameters

```bash
/memory:tech-research [session-id | "tech-stack-name"] [--regenerate] [--tool <gemini|qwen>]
```

**Arguments**:
- **session-id | tech-stack-name**: Input source (auto-detected by WFS- prefix)
  - Session mode: `WFS-user-auth-v2` - Extract tech stack from workflow
  - Direct mode: `"typescript"`, `"typescript-react-nextjs"` - User specifies
- **--regenerate**: Force regenerate existing SKILL (deletes and recreates)
- **--tool**: Reserved for future CLI integration (default: gemini)

---

## Examples

**Generated File Structure** (for all examples):
```
.claude/skills/{tech-stack}/
├── SKILL.md           # Index (Phase 3)
├── principles.md      # Agent (Phase 2)
├── patterns.md        # Agent
├── practices.md       # Agent
├── testing.md         # Agent
├── config.md          # Agent
├── frameworks.md      # Agent (if composite)
└── metadata.json      # Agent
```

### Direct Mode - Single Stack

```bash
/memory:tech-research "typescript"
```

**Workflow**:
1. Phase 1: Detects direct mode, checks existing SKILL
2. Phase 2: Agent executes 4 Exa queries, writes 5 modules
3. Phase 3: Generates SKILL.md index

### Direct Mode - Composite Stack

```bash
/memory:tech-research "typescript-react-nextjs"
```

**Workflow**:
1. Phase 1: Decomposes into ["typescript", "react", "nextjs"]
2. Phase 2: Agent executes 6 Exa queries (4 base + 2 components), writes 6 modules (adds frameworks.md)
3. Phase 3: Generates SKILL.md index with framework integration

### Session Mode - Extract from Workflow

```bash
/memory:tech-research WFS-user-auth-20251104
```

**Workflow**:
1. Phase 1: Reads session, extracts tech stack: `python-fastapi-sqlalchemy`
2. Phase 2: Agent researches Python + FastAPI + SQLAlchemy, writes 6 modules
3. Phase 3: Generates SKILL.md index

### Regenerate Existing

```bash
/memory:tech-research "react" --regenerate
```

**Workflow**:
1. Phase 1: Deletes existing SKILL due to --regenerate
2. Phase 2: Agent executes fresh Exa research (latest 2025 practices)
3. Phase 3: Generates updated SKILL.md

### Skip Path - Fast Update

```bash
/memory:tech-research "python"
```

**Scenario**: SKILL already exists with 7 files

**Workflow**:
1. Phase 1: Detects existing SKILL, sets SKIP_GENERATION = true
2. Phase 2: **SKIPPED**
3. Phase 3: Updates SKILL.md index only (5-10x faster)


