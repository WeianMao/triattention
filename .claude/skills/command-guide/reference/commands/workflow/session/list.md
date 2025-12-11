---
name: list
description: List all workflow sessions with status filtering, shows session metadata and progress information
examples:
  - /workflow:session:list
---

# List Workflow Sessions (/workflow:session:list)

## Overview
Display all workflow sessions with their current status, progress, and metadata.

## Usage
```bash
/workflow:session:list       # Show all sessions with status
```

## Implementation Flow

### Step 1: Find All Sessions
```bash
ls .workflow/active/WFS-* 2>/dev/null
```

### Step 2: Check Active Session
```bash
find .workflow/active/ -name "WFS-*" -type d 2>/dev/null | head -1
```

### Step 3: Read Session Metadata
```bash
jq -r '.session_id, .status, .project' .workflow/active/WFS-session/workflow-session.json
```

### Step 4: Count Task Progress
```bash
find .workflow/active/WFS-session/.task/ -name "*.json" -type f 2>/dev/null | wc -l
find .workflow/active/WFS-session/.summaries/ -name "*.md" -type f 2>/dev/null | wc -l
```

### Step 5: Get Creation Time
```bash
jq -r '.created_at // "unknown"' .workflow/active/WFS-session/workflow-session.json
```

## Simple Bash Commands

### Basic Operations
- **List sessions**: `find .workflow/active/ -name "WFS-*" -type d`
- **Find active**: `find .workflow/active/ -name "WFS-*" -type d`
- **Read session data**: `jq -r '.session_id, .status' session.json`
- **Count tasks**: `find .task/ -name "*.json" -type f | wc -l`
- **Count completed**: `find .summaries/ -name "*.md" -type f 2>/dev/null | wc -l`
- **Get timestamp**: `jq -r '.created_at' session.json`

## Simple Output Format

### Session List Display
```
Workflow Sessions:

[ACTIVE] WFS-oauth-integration
   Project: OAuth2 authentication system
   Status: active
   Progress: 3/8 tasks completed
   Created: 2025-09-15T10:30:00Z

[PAUSED] WFS-user-profile
   Project: User profile management
   Status: paused
   Progress: 1/5 tasks completed
   Created: 2025-09-14T14:15:00Z

[COMPLETED] WFS-database-migration
   Project: Database schema migration
   Status: completed
   Progress: 4/4 tasks completed
   Created: 2025-09-13T09:00:00Z

Total: 3 sessions (1 active, 1 paused, 1 completed)
```

### Status Indicators
- **[ACTIVE]**: Active session
- **[PAUSED]**: Paused session
- **[COMPLETED]**: Completed session
- **[ERROR]**: Error/corrupted session

### Quick Commands
```bash
# Count all sessions
ls .workflow/active/WFS-* | wc -l

# Show recent sessions
ls -t .workflow/active/WFS-*/workflow-session.json | head -3
```