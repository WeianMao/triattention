---
name: resume
description: Resume the most recently paused workflow session with automatic session discovery and status update
---

# Resume Workflow Session (/workflow:session:resume)

## Overview
Resume the most recently paused workflow session, restoring all context and state.

## Usage
```bash
/workflow:session:resume     # Resume most recent paused session
```

## Implementation Flow

### Step 1: Find Paused Sessions
```bash
ls .workflow/active/WFS-* 2>/dev/null
```

### Step 2: Check Session Status
```bash
jq -r '.status' .workflow/active/WFS-session/workflow-session.json
```

### Step 3: Find Most Recent Paused
```bash
ls -t .workflow/active/WFS-*/workflow-session.json | head -1
```

### Step 4: Update Session Status
```bash
jq '.status = "active"' .workflow/active/WFS-session/workflow-session.json > temp.json
mv temp.json .workflow/active/WFS-session/workflow-session.json
```

### Step 5: Add Resume Timestamp
```bash
jq '.resumed_at = "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"' .workflow/active/WFS-session/workflow-session.json > temp.json
mv temp.json .workflow/active/WFS-session/workflow-session.json
```

## Simple Bash Commands

### Basic Operations
- **List sessions**: `ls .workflow/active/WFS-*`
- **Check status**: `jq -r '.status' session.json`
- **Find recent**: `ls -t .workflow/active/*/workflow-session.json | head -1`
- **Update status**: `jq '.status = "active"' session.json > temp.json`
- **Add timestamp**: `jq '.resumed_at = "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"'`

### Resume Result
```
Session WFS-user-auth resumed
- Status: active
- Paused at: 2025-09-15T14:30:00Z
- Resumed at: 2025-09-15T15:45:00Z
- Ready for: /workflow:execute
```