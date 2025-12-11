---
name: complete
description: Mark active workflow session as complete, archive with lessons learned, update manifest, remove active flag
examples:
  - /workflow:session:complete
  - /workflow:session:complete --detailed
---

# Complete Workflow Session (/workflow:session:complete)

## Overview
Mark the currently active workflow session as complete, analyze it for lessons learned, move it to the archive directory, and remove the active flag marker.

## Usage
```bash
/workflow:session:complete           # Complete current active session
/workflow:session:complete --detailed # Show detailed completion summary
```

## Implementation Flow

### Phase 1: Pre-Archival Preparation (Transactional Setup)

**Purpose**: Find active session, create archiving marker to prevent concurrent operations. Session remains in active location for agent processing.

#### Step 1.1: Find Active Session and Get Name
```bash
# Find active session directory
bash(find .workflow/active/ -name "WFS-*" -type d | head -1)

# Extract session name from directory path
bash(basename .workflow/active/WFS-session-name)
```
**Output**: Session name `WFS-session-name`

#### Step 1.2: Check for Existing Archiving Marker (Resume Detection)
```bash
# Check if session is already being archived
bash(test -f .workflow/active/WFS-session-name/.archiving && echo "RESUMING" || echo "NEW")
```

**If RESUMING**:
- Previous archival attempt was interrupted
- Skip to Phase 2 to resume agent analysis

**If NEW**:
- Continue to Step 1.3

#### Step 1.3: Create Archiving Marker
```bash
# Mark session as "archiving in progress"
bash(touch .workflow/active/WFS-session-name/.archiving)
```
**Purpose**:
- Prevents concurrent operations on this session
- Enables recovery if archival fails
- Session remains in `.workflow/active/` for agent analysis

**Result**: Session still at `.workflow/active/WFS-session-name/` with `.archiving` marker

### Phase 2: Agent Analysis (In-Place Processing)

**Purpose**: Agent analyzes session WHILE STILL IN ACTIVE LOCATION. Generates metadata but does NOT move files or update manifest.

#### Agent Invocation

Invoke `universal-executor` agent to analyze session and prepare archive metadata.

**Agent Task**:
```
Task(
  subagent_type="universal-executor",
  description="Analyze session for archival",
  prompt=`
Analyze workflow session for archival preparation. Session is STILL in active location.

## Context
- Session: .workflow/active/WFS-session-name/
- Status: Marked as archiving (.archiving marker present)
- Location: Active sessions directory (NOT archived yet)

## Tasks

1. **Extract session data** from workflow-session.json
   - session_id, description/topic, started_at, completed_at, status
   - If status != "completed", update it with timestamp

2. **Count files**: tasks (.task/*.json) and summaries (.summaries/*.md)

3. **Extract review data** (if .review/ exists):
   - Count dimension results: .review/dimensions/*.json
   - Count deep-dive results: .review/iterations/*.json
   - Extract findings summary from dimension JSONs (total, critical, high, medium, low)
   - Check fix results if .review/fixes/ exists (fixed_count, failed_count)
   - Build review_metrics: {dimensions_analyzed, total_findings, severity_distribution, fix_success_rate}

4. **Generate lessons**: Use gemini with ~/.claude/workflows/cli-templates/prompts/archive/analysis-simple.txt
   - Return: {successes, challenges, watch_patterns}
   - If review data exists, include review-specific lessons (common issue patterns, effective fixes)

5. **Build archive entry**:
   - Calculate: duration_hours, success_rate, tags (3-5 keywords)
   - Construct complete JSON with session_id, description, archived_at, metrics, tags, lessons
   - Include archive_path: ".workflow/archives/WFS-session-name" (future location)
   - If review data exists, include review_metrics in metrics object

6. **Extract feature metadata** (for Phase 4):
   - Parse IMPL_PLAN.md for title (first # heading)
   - Extract description (first paragraph, max 200 chars)
   - Generate feature tags (3-5 keywords from content)

7. **Return result**: Complete metadata package for atomic commit
   {
     "status": "success",
     "session_id": "WFS-session-name",
     "archive_entry": {
       "session_id": "...",
       "description": "...",
       "archived_at": "...",
       "archive_path": ".workflow/archives/WFS-session-name",
       "metrics": {
         "duration_hours": 2.5,
         "tasks_completed": 5,
         "summaries_generated": 3,
         "review_metrics": {                    // Optional, only if .review/ exists
           "dimensions_analyzed": 4,
           "total_findings": 15,
           "severity_distribution": {"critical": 1, "high": 3, "medium": 8, "low": 3},
           "fix_success_rate": 0.87             // Optional, only if .review/fixes/ exists
         }
       },
       "tags": [...],
       "lessons": {...}
     },
     "feature_metadata": {
       "title": "...",
       "description": "...",
       "tags": [...]
     }
   }

## Important Constraints
- DO NOT move or delete any files
- DO NOT update manifest.json yet
- Session remains in .workflow/active/ during analysis
- Return complete metadata package for orchestrator to commit atomically

## Error Handling
- On failure: return {"status": "error", "task": "...", "message": "..."}
- Do NOT modify any files on error
  `
)
```

**Expected Output**:
- Agent returns complete metadata package
- Session remains in `.workflow/active/` with `.archiving` marker
- No files moved or manifests updated yet

### Phase 3: Atomic Commit (Transactional File Operations)

**Purpose**: Atomically commit all changes. Only execute if Phase 2 succeeds.

#### Step 3.1: Create Archive Directory
```bash
bash(mkdir -p .workflow/archives/)
```

#### Step 3.2: Move Session to Archive
```bash
bash(mv .workflow/active/WFS-session-name .workflow/archives/WFS-session-name)
```
**Result**: Session now at `.workflow/archives/WFS-session-name/`

#### Step 3.3: Update Manifest
```bash
# Read current manifest (or create empty array if not exists)
bash(test -f .workflow/archives/manifest.json && cat .workflow/archives/manifest.json || echo "[]")
```

**JSON Update Logic**:
```javascript
// Read agent result from Phase 2
const agentResult = JSON.parse(agentOutput);
const archiveEntry = agentResult.archive_entry;

// Read existing manifest
let manifest = [];
try {
  const manifestContent = Read('.workflow/archives/manifest.json');
  manifest = JSON.parse(manifestContent);
} catch {
  manifest = []; // Initialize if not exists
}

// Append new entry
manifest.push(archiveEntry);

// Write back
Write('.workflow/archives/manifest.json', JSON.stringify(manifest, null, 2));
```

#### Step 3.4: Remove Archiving Marker
```bash
bash(rm .workflow/archives/WFS-session-name/.archiving)
```
**Result**: Clean archived session without temporary markers

**Output Confirmation**:
```
✓ Session "${sessionId}" archived successfully
  Location: .workflow/archives/WFS-session-name/
  Lessons: ${archiveEntry.lessons.successes.length} successes, ${archiveEntry.lessons.challenges.length} challenges
  Manifest: Updated with ${manifest.length} total sessions
  ${reviewMetrics ? `Review: ${reviewMetrics.total_findings} findings across ${reviewMetrics.dimensions_analyzed} dimensions, ${Math.round(reviewMetrics.fix_success_rate * 100)}% fixed` : ''}
```

### Phase 4: Update Project Feature Registry

**Purpose**: Record completed session as a project feature in `.workflow/project.json`.

**Execution**: Uses feature metadata from Phase 2 agent result to update project registry.

#### Step 4.1: Check Project State Exists
```bash
bash(test -f .workflow/project.json && echo "EXISTS" || echo "SKIP")
```

**If SKIP**: Output warning and skip Phase 4
```
WARNING: No project.json found. Run /workflow:session:start to initialize.
```

#### Step 4.2: Extract Feature Information from Agent Result

**Data Processing** (Uses Phase 2 agent output):
```javascript
// Extract feature metadata from agent result
const agentResult = JSON.parse(agentOutput);
const featureMeta = agentResult.feature_metadata;

// Data already prepared by agent:
const title = featureMeta.title;
const description = featureMeta.description;
const tags = featureMeta.tags;

// Create feature ID (lowercase slug)
const featureId = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').substring(0, 50);
```

#### Step 4.3: Update project.json

```bash
# Read current project state
bash(cat .workflow/project.json)
```

**JSON Update Logic**:
```javascript
// Read existing project.json (created by /workflow:init)
// Note: overview field is managed by /workflow:init, not modified here
const projectMeta = JSON.parse(Read('.workflow/project.json'));
const currentTimestamp = new Date().toISOString();
const currentDate = currentTimestamp.split('T')[0]; // YYYY-MM-DD

// Extract tags from IMPL_PLAN.md (simple keyword extraction)
const tags = extractTags(planContent); // e.g., ["auth", "security"]

// Build feature object with complete metadata
const newFeature = {
  id: featureId,
  title: title,
  description: description,
  status: "completed",
  tags: tags,
  timeline: {
    created_at: currentTimestamp,
    implemented_at: currentDate,
    updated_at: currentTimestamp
  },
  traceability: {
    session_id: sessionId,
    archive_path: archivePath, // e.g., ".workflow/archives/WFS-auth-system"
    commit_hash: getLatestCommitHash() || "" // Optional: git rev-parse HEAD
  },
  docs: [],      // Placeholder for future doc links
  relations: []  // Placeholder for feature dependencies
};

// Add new feature to array
projectMeta.features.push(newFeature);

// Update statistics
projectMeta.statistics.total_features = projectMeta.features.length;
projectMeta.statistics.total_sessions += 1;
projectMeta.statistics.last_updated = currentTimestamp;

// Write back
Write('.workflow/project.json', JSON.stringify(projectMeta, null, 2));
```

**Helper Functions**:
```javascript
// Extract tags from IMPL_PLAN.md content
function extractTags(planContent) {
  const tags = [];

  // Look for common keywords
  const keywords = {
    'auth': /authentication|login|oauth|jwt/i,
    'security': /security|encrypt|hash|token/i,
    'api': /api|endpoint|rest|graphql/i,
    'ui': /component|page|interface|frontend/i,
    'database': /database|schema|migration|sql/i,
    'test': /test|testing|spec|coverage/i
  };

  for (const [tag, pattern] of Object.entries(keywords)) {
    if (pattern.test(planContent)) {
      tags.push(tag);
    }
  }

  return tags.slice(0, 5); // Max 5 tags
}

// Get latest git commit hash (optional)
function getLatestCommitHash() {
  try {
    const result = Bash({
      command: "git rev-parse --short HEAD 2>/dev/null",
      description: "Get latest commit hash"
    });
    return result.trim();
  } catch {
    return "";
  }
}
```

#### Step 4.4: Output Confirmation

```
✓ Feature "${title}" added to project registry
  ID: ${featureId}
  Session: ${sessionId}
  Location: .workflow/project.json
```

**Error Handling**:
- If project.json malformed: Output error, skip update
- If feature_metadata missing from agent result: Skip Phase 4
- If extraction fails: Use minimal defaults

**Phase 4 Total Commands**: 1 bash read + JSON manipulation

## Error Recovery

### If Agent Fails (Phase 2)

**Symptoms**:
- Agent returns `{"status": "error", ...}`
- Agent crashes or times out
- Analysis incomplete

**Recovery Steps**:
```bash
# Session still in .workflow/active/WFS-session-name
# Remove archiving marker
bash(rm .workflow/active/WFS-session-name/.archiving)
```

**User Notification**:
```
ERROR: Session archival failed during analysis phase
Reason: [error message from agent]
Session remains active in: .workflow/active/WFS-session-name

Recovery:
1. Fix any issues identified in error message
2. Retry: /workflow:session:complete

Session state: SAFE (no changes committed)
```

### If Move Fails (Phase 3)

**Symptoms**:
- `mv` command fails
- Permission denied
- Disk full

**Recovery Steps**:
```bash
# Archiving marker still present
# Session still in .workflow/active/ (move failed)
# No manifest updated yet
```

**User Notification**:
```
ERROR: Session archival failed during move operation
Reason: [mv error message]
Session remains in: .workflow/active/WFS-session-name

Recovery:
1. Fix filesystem issues (permissions, disk space)
2. Retry: /workflow:session:complete
   - System will detect .archiving marker
   - Will resume from Phase 2 (agent analysis)

Session state: SAFE (analysis complete, ready to retry move)
```

### If Manifest Update Fails (Phase 3)

**Symptoms**:
- JSON parsing error
- Write permission denied
- Session moved but manifest not updated

**Recovery Steps**:
```bash
# Session moved to .workflow/archives/WFS-session-name
# Manifest NOT updated
# Archiving marker still present in archived location
```

**User Notification**:
```
ERROR: Session archived but manifest update failed
Reason: [error message]
Session location: .workflow/archives/WFS-session-name

Recovery:
1. Fix manifest.json issues (syntax, permissions)
2. Manual manifest update:
   - Add archive entry from agent output
   - Remove .archiving marker: rm .workflow/archives/WFS-session-name/.archiving

Session state: PARTIALLY COMPLETE (session archived, manifest needs update)
```

## Workflow Execution Strategy

### Transactional Four-Phase Approach

**Phase 1: Pre-Archival Preparation** (Marker creation)
- Find active session and extract name
- Check for existing `.archiving` marker (resume detection)
- Create `.archiving` marker if new
- **No data processing** - just state tracking
- **Total**: 2-3 bash commands (find + marker check/create)

**Phase 2: Agent Analysis** (Read-only data processing)
- Extract all session data from active location
- Count tasks and summaries
- Extract review data if .review/ exists (dimension results, findings, fix results)
- Generate lessons learned analysis (including review-specific lessons if applicable)
- Extract feature metadata from IMPL_PLAN.md
- Build complete archive + feature metadata package (with review_metrics if applicable)
- **No file modifications** - pure analysis
- **Total**: 1 agent invocation

**Phase 3: Atomic Commit** (Transactional file operations)
- Create archive directory
- Move session to archive location
- Update manifest.json with archive entry
- Remove `.archiving` marker
- **All-or-nothing**: Either all succeed or session remains in safe state
- **Total**: 4 bash commands + JSON manipulation

**Phase 4: Project Registry Update** (Optional feature tracking)
- Check project.json exists
- Use feature metadata from Phase 2 agent result
- Build feature object with complete traceability
- Update project statistics
- **Independent**: Can fail without affecting archival
- **Total**: 1 bash read + JSON manipulation

### Transactional Guarantees

**State Consistency**:
- Session NEVER in inconsistent state
- `.archiving` marker enables safe resume
- Agent failure leaves session in recoverable state
- Move/manifest operations grouped in Phase 3

**Failure Isolation**:
- Phase 1 failure: No changes made
- Phase 2 failure: Session still active, can retry
- Phase 3 failure: Clear error state, manual recovery documented
- Phase 4 failure: Does not affect archival success

**Resume Capability**:
- Detect interrupted archival via `.archiving` marker
- Resume from Phase 2 (skip marker creation)
- Idempotent operations (safe to retry)


