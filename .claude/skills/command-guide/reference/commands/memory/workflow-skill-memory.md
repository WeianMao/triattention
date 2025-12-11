---
name: workflow-skill-memory
description: Process WFS-* archived sessions using universal-executor agents with Gemini analysis to generate workflow-progress SKILL package (sessions-timeline, lessons, conflicts)
argument-hint: "session <session-id> | all"
allowed-tools: Task(*), TodoWrite(*), Bash(*), Read(*), Write(*)
---

# Workflow SKILL Memory Generator

## Overview

Generate SKILL package from archived workflow sessions using agent-driven analysis. Supports single-session incremental updates or parallel processing of all sessions.

**Scope**: Only processes WFS-* workflow sessions. Other session types (e.g., doc sessions) are automatically ignored.

## Usage

```bash
/memory:workflow-skill-memory session WFS-<session-id>   # Process single WFS session
/memory:workflow-skill-memory all                        # Process all WFS sessions in parallel
```

## Execution Modes

### Mode 1: Single Session (`session <session-id>`)

**Purpose**: Incremental update - process one archived session and merge into existing SKILL package

**Workflow**:
1. **Validate session**: Check if session exists in `.workflow/.archives/{session-id}/`
2. **Invoke agent**: Call `universal-executor` to analyze session and update SKILL documents
3. **Agent tasks**:
   - Read session data from `.workflow/.archives/{session-id}/`
   - Extract lessons, conflicts, and outcomes
   - Use Gemini for intelligent aggregation (optional)
   - Update or create SKILL documents using templates
   - Regenerate SKILL.md index

**Command Example**:
```bash
/memory:workflow-skill-memory session WFS-user-auth
```

**Expected Output**:
```
Session WFS-user-auth processed
Updated:
- sessions-timeline.md (1 session added)
- lessons-learned.md (3 lessons merged)
- conflict-patterns.md (1 conflict added)
- SKILL.md (index regenerated)
```

---

### Mode 2: All Sessions (`all`)

**Purpose**: Full regeneration - process all archived sessions in parallel for complete SKILL package

**Workflow**:
1. **List sessions**: Read manifest.json to get all archived session IDs
2. **Parallel invocation**: Launch multiple `universal-executor` agents in parallel (one per session)
3. **Agent coordination**:
   - Each agent processes one session independently
   - Agents use Gemini for analysis
   - Agents collect data into JSON (no direct file writes)
   - Final aggregator agent merges results and generates SKILL documents

**Command Example**:
```bash
/memory:workflow-skill-memory all
```

**Expected Output**:
```
All sessions processed in parallel
Sessions: 8 total
Updated:
- sessions-timeline.md (8 sessions)
- lessons-learned.md (24 lessons aggregated)
- conflict-patterns.md (12 conflicts documented)
- SKILL.md (index regenerated)
```

---

## Implementation Flow

### Phase 1: Validation and Setup

**Step 1.1: Parse Command Arguments**

Extract mode and session ID:
```javascript
if (args === "all") {
  mode = "all"
} else if (args.startsWith("session ")) {
  mode = "session"
  session_id = args.replace("session ", "").trim()
} else {
  ERROR = "Invalid arguments. Usage: session <session-id> | all"
  EXIT
}
```

**Step 1.2: Validate Archive Directory**
```bash
bash(test -d .workflow/.archives && echo "exists" || echo "missing")
```

If missing, report error and exit.

**Step 1.3: Mode-Specific Validation**

**Single Session Mode**:
```bash
# Validate session ID format (must start with WFS-)
if [[ ! "$session_id" =~ ^WFS- ]]; then
  ERROR = "Invalid session ID format. Only WFS-* sessions are supported"
  EXIT
fi

# Check if session exists
bash(test -d .workflow/.archives/{session_id} && echo "exists" || echo "missing")
```

If missing, report error: "Session {session_id} not found in archives"

**All Sessions Mode**:
```bash
# Read manifest and filter only WFS- sessions
bash(cat .workflow/.archives/manifest.json | jq -r '.archives[].session_id | select(startswith("WFS-"))')
```

Store filtered session IDs in array. Ignore doc sessions and other non-WFS sessions.

**Step 1.4: TodoWrite Initialization**

**Single Session Mode**:
```javascript
TodoWrite({todos: [
  {"content": "Validate session existence", "status": "completed", "activeForm": "Validating session"},
  {"content": "Invoke agent to process session", "status": "in_progress", "activeForm": "Invoking agent"},
  {"content": "Verify SKILL package updated", "status": "pending", "activeForm": "Verifying update"}
]})
```

**All Sessions Mode**:
```javascript
TodoWrite({todos: [
  {"content": "Read manifest and list sessions", "status": "completed", "activeForm": "Reading manifest"},
  {"content": "Invoke agents in parallel", "status": "in_progress", "activeForm": "Invoking agents"},
  {"content": "Verify SKILL package regenerated", "status": "pending", "activeForm": "Verifying regeneration"}
]})
```

---

### Phase 2: Agent Invocation

#### Single Session Mode - Agent Task

Invoke `universal-executor` with session-specific task:

**Agent Prompt Structure**:
```
Task: Process Workflow Session for SKILL Package

Context:
- Session ID: {session_id}
- Session Path: .workflow/.archives/{session_id}/
- Mode: Incremental update

Objectives:

1. Read session data:
   - workflow-session.json (metadata)
   - IMPL_PLAN.md (implementation summary)
   - TODO_LIST.md (if exists)
   - manifest.json entry for lessons

2. Extract key information:
   - Description, tags, metrics
   - Lessons (successes, challenges, watch_patterns)
   - Context package path (reference only)
   - Key outcomes from IMPL_PLAN

3. Use Gemini for aggregation (optional):
   Command pattern:
   cd .workflow/.archives/{session_id} && gemini -p "
   PURPOSE: Extract lessons and conflicts from workflow session
   TASK:
   • Analyze IMPL_PLAN and lessons from manifest
   • Identify success patterns and challenges
   • Extract conflict patterns with resolutions
   • Categorize by functional domain
   MODE: analysis
   CONTEXT: @IMPL_PLAN.md @workflow-session.json
   EXPECTED: Structured lessons and conflicts in JSON format
   RULES: Template reference from skill-aggregation.txt
   "

3.5. **Generate SKILL.md Description** (CRITICAL for auto-loading):

   Read skill-index.txt template Section: "Description Field Generation"

   Execute command to get project root:
   ```bash
   git rev-parse --show-toplevel  # Example output: /d/Claude_dms3
   ```

   Apply description format:
   ```
   Progressive workflow development history (located at {project_root}).
   Load this SKILL when continuing development, analyzing past implementations,
   or learning from workflow history, especially when no relevant context exists in memory.
   ```

   **Validation**:
   - [ ] Path uses forward slashes (not backslashes)
   - [ ] All three use cases present
   - [ ] Trigger optimization phrase included
   - [ ] Path is absolute (starts with / or drive letter)

4. Read templates for formatting guidance:
   - ~/.claude/workflows/cli-templates/prompts/workflow/skill-sessions-timeline.txt
   - ~/.claude/workflows/cli-templates/prompts/workflow/skill-lessons-learned.txt
   - ~/.claude/workflows/cli-templates/prompts/workflow/skill-conflict-patterns.txt
   - ~/.claude/workflows/cli-templates/prompts/workflow/skill-index.txt

   **CRITICAL**: From skill-index.txt, read these sections:
   - "Description Field Generation" - Rules for generating description
   - "Variable Substitution Guide" - All required variables
   - "Generation Instructions" - Step-by-step generation process
   - "Validation Checklist" - Final validation steps

5. Update SKILL documents:
   - sessions-timeline.md: Append new session, update domain grouping
   - lessons-learned.md: Merge lessons into categories, update frequencies
   - conflict-patterns.md: Add conflicts, update recurring pattern frequencies
   - SKILL.md: Regenerate index with updated counts

   **For SKILL.md generation**:
   - Follow "Generation Instructions" from skill-index.txt (Steps 1-7)
   - Use git command for project_root: `git rev-parse --show-toplevel`
   - Apply "Description Field Generation" rules
   - Validate using "Validation Checklist"
   - Increment version (patch level)

6. Return result JSON:
   {
     "status": "success",
     "session_id": "{session_id}",
     "updates": {
       "sessions_added": 1,
       "lessons_merged": count,
       "conflicts_added": count
     }
   }
```

---

#### All Sessions Mode - Parallel Agent Tasks

**Step 2.1: Launch parallel session analyzers**

Invoke multiple agents in parallel (one message with multiple Task calls):

**Per-Session Agent Prompt**:
```
Task: Extract Session Data for SKILL Package

Context:
- Session ID: {session_id}
- Mode: Parallel analysis (no direct file writes)

Objectives:

1. Read session data (same as single mode)

2. Extract key information (same as single mode)

3. Use Gemini for analysis (same as single mode)

4. Return structured data JSON:
   {
     "status": "success",
     "session_id": "{session_id}",
     "data": {
       "metadata": {
         "description": "...",
         "archived_at": "...",
         "tags": [...],
         "metrics": {...}
       },
       "lessons": {
         "successes": [...],
         "challenges": [...],
         "watch_patterns": [...]
       },
       "conflicts": [
         {
           "type": "architecture|dependencies|testing|performance",
           "pattern": "...",
           "resolution": "...",
           "code_impact": [...]
         }
       ],
       "impl_summary": "First 200 chars of IMPL_PLAN",
       "context_package_path": "..."
     }
   }
```

**Step 2.2: Aggregate results**

After all session agents complete, invoke aggregator agent:

**Aggregator Agent Prompt**:
```
Task: Aggregate Session Results and Generate SKILL Package

Context:
- Mode: Full regeneration
- Input: JSON results from {session_count} session agents

Objectives:

1. Aggregate all session data:
   - Collect metadata from all sessions
   - Merge lessons by category
   - Group conflicts by type
   - Sort sessions by date

2. Use Gemini for final aggregation:
   gemini -p "
   PURPOSE: Aggregate lessons and conflicts from all workflow sessions
   TASK:
   • Group successes by functional domain
   • Categorize challenges by severity (HIGH/MEDIUM/LOW)
   • Identify recurring conflict patterns
   • Calculate frequencies and prioritize
   MODE: analysis
   CONTEXT: [Provide aggregated JSON data]
   EXPECTED: Final aggregated structure for SKILL documents
   RULES: Template reference from skill-aggregation.txt
   "

3. Read templates for formatting (same 4 templates as single mode)

4. Generate all SKILL documents:
   - sessions-timeline.md (all sessions, sorted by date)
   - lessons-learned.md (aggregated lessons with frequencies)
   - conflict-patterns.md (recurring patterns with resolutions)
   - SKILL.md (index with progressive loading)

5. Write files to .claude/skills/workflow-progress/

6. Return result JSON:
   {
     "status": "success",
     "sessions_processed": count,
     "files_generated": ["SKILL.md", "sessions-timeline.md", ...],
     "summary": {
       "total_sessions": count,
       "functional_domains": [...],
       "date_range": "...",
       "lessons_count": count,
       "conflicts_count": count
     }
   }
```

---

### Phase 3: Verification

**Step 3.1: Check SKILL Package Files**
```bash
bash(ls -lh .claude/skills/workflow-progress/)
```

Verify all 4 files exist:
- SKILL.md
- sessions-timeline.md
- lessons-learned.md
- conflict-patterns.md

**Step 3.2: TodoWrite Completion**

Mark all tasks as completed.

**Step 3.3: Display Summary**

**Single Session Mode**:
```
Session {session_id} processed successfully

Updated:
- sessions-timeline.md
- lessons-learned.md
- conflict-patterns.md
- SKILL.md

SKILL Location: .claude/skills/workflow-progress/SKILL.md
```

**All Sessions Mode**:
```
All sessions processed in parallel

Sessions: {count} total
Functional Domains: {domain_list}
Date Range: {earliest} - {latest}

Generated:
- sessions-timeline.md ({count} sessions)
- lessons-learned.md ({lessons_count} lessons)
- conflict-patterns.md ({conflicts_count} conflicts)
- SKILL.md (4-level progressive loading)

SKILL Location: .claude/skills/workflow-progress/SKILL.md

Usage:
- Level 0: Quick refresh (~2K tokens)
- Level 1: Recent history (~8K tokens)
- Level 2: Complete analysis (~25K tokens)
- Level 3: Deep dive (~40K tokens)
```

---

## Agent Guidelines

### Agent Capabilities

**universal-executor agents can**:
- Read files from `.workflow/.archives/`
- Execute bash commands
- Call Gemini CLI for intelligent analysis
- Read template files for formatting guidance
- Write SKILL package files (single mode) or return JSON (parallel mode)
- Return structured results

### Gemini Usage Pattern

**When to use Gemini**:
- Aggregating lessons from multiple sources
- Identifying recurring patterns
- Classifying conflicts by type and severity
- Extracting structured data from IMPL_PLAN

**Fallback Strategy**: If Gemini fails or times out, use direct file parsing with structured extraction logic.

---

## Template System

### Template Files

All templates located in: `~/.claude/workflows/cli-templates/prompts/workflow/`

1. **skill-sessions-timeline.txt**: Format for sessions-timeline.md
2. **skill-lessons-learned.txt**: Format for lessons-learned.md
3. **skill-conflict-patterns.txt**: Format for conflict-patterns.md
4. **skill-index.txt**: Format for SKILL.md index
5. **skill-aggregation.txt**: Rules for Gemini aggregation (existing)

### Template Usage in Agent

**Agents read templates to understand**:
- File structure and markdown format
- Data sources (which files to read)
- Update strategy (incremental vs full)
- Formatting rules and conventions
- Aggregation logic (for Gemini)

**Templates are NOT shown in this command documentation** - agents read them directly as needed.

---

## Error Handling

### Validation Errors
- **No archives directory**: "Error: No workflow archives found at .workflow/.archives/"
- **Invalid session ID format**: "Error: Invalid session ID format. Only WFS-* sessions are supported"
- **Session not found**: "Error: Session {session_id} not found in archives"
- **No WFS sessions in manifest**: "Error: No WFS-* workflow sessions found in manifest.json"

### Agent Errors
- If agent fails, report error message from agent result
- If Gemini times out, agents use fallback direct parsing
- If template read fails, agents use inline format

### Recovery
- Single session mode: Can be retried without affecting other sessions
- All sessions mode: If one agent fails, others continue; retry failed sessions individually



## Integration

### Called by `/workflow:session:complete`

Automatically invoked after session archival:
```bash
SlashCommand(command="/memory:workflow-skill-memory session {session_id}")
```

### Manual Invocation

Users can manually process sessions:
```bash
/memory:workflow-skill-memory session WFS-custom-feature  # Single session
/memory:workflow-skill-memory all                         # Full regeneration
```
