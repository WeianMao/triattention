---
name: version
description: Display Claude Code version information and check for updates
allowed-tools: Bash(*)
---

# Version Command (/version)

## Purpose
Display local and global installation versions, check for the latest updates from GitHub, and provide upgrade recommendations.

## Execution Flow
1. **Local Version Check**: Read version information from `./.claude/version.json` if it exists.
2. **Global Version Check**: Read version information from `~/.claude/version.json` if it exists.
3. **Fetch Remote Versions**: Use GitHub API to get the latest stable release tag and the latest commit hash from the main branch.
4. **Compare & Suggest**: Compare installed versions with the latest remote versions and provide upgrade suggestions if applicable.

## Step 1: Check Local Version

### Check if local version.json exists
```bash
bash(test -f ./.claude/version.json && echo "found" || echo "not_found")
```

### Read local version (if exists)
```bash
bash(cat ./.claude/version.json)
```

### Extract version with jq (preferred)
```bash
bash(cat ./.claude/version.json | grep -o '"version": *"[^"]*"' | cut -d'"' -f4)
```

### Extract installation date
```bash
bash(cat ./.claude/version.json | grep -o '"installation_date_utc": *"[^"]*"' | cut -d'"' -f4)
```

**Output Format**:
```
Local Version: 3.2.1
Installed: 2025-10-03T12:00:00Z
```

## Step 2: Check Global Version

### Check if global version.json exists
```bash
bash(test -f ~/.claude/version.json && echo "found" || echo "not_found")
```

### Read global version
```bash
bash(cat ~/.claude/version.json)
```

### Extract version
```bash
bash(cat ~/.claude/version.json | grep -o '"version": *"[^"]*"' | cut -d'"' -f4)
```

### Extract installation date
```bash
bash(cat ~/.claude/version.json | grep -o '"installation_date_utc": *"[^"]*"' | cut -d'"' -f4)
```

**Output Format**:
```
Global Version: 3.2.1
Installed: 2025-10-03T12:00:00Z
```

## Step 3: Fetch Latest Stable Release

### Call GitHub API for latest release (with timeout)
```bash
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/releases/latest" 2>/dev/null, timeout: 30000)
```

### Extract tag name (version)
```bash
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/releases/latest" 2>/dev/null | grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4, timeout: 30000)
```

### Extract release name
```bash
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/releases/latest" 2>/dev/null | grep -o '"name": *"[^"]*"' | head -1 | cut -d'"' -f4, timeout: 30000)
```

### Extract published date
```bash
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/releases/latest" 2>/dev/null | grep -o '"published_at": *"[^"]*"' | cut -d'"' -f4, timeout: 30000)
```

**Output Format**:
```
Latest Stable: v3.2.2
Release: v3.2.2: Independent Test-Gen Workflow with Cross-Session Context
Published: 2025-10-03T04:10:08Z
```

## Step 4: Fetch Latest Main Branch

### Call GitHub API for latest commit on main (with timeout)
```bash
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/commits/main" 2>/dev/null, timeout: 30000)
```

### Extract commit SHA (short)
```bash
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/commits/main" 2>/dev/null | grep -o '"sha": *"[^"]*"' | head -1 | cut -d'"' -f4 | cut -c1-7, timeout: 30000)
```

### Extract commit message (first line only)
```bash
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/commits/main" 2>/dev/null | grep '"message":' | cut -d'"' -f4 | cut -d'\' -f1, timeout: 30000)
```

### Extract commit date
```bash
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/commits/main" 2>/dev/null | grep -o '"date": *"[^"]*"' | head -1 | cut -d'"' -f4, timeout: 30000)
```

**Output Format**:
```
Latest Dev: a03415b
Message: feat: Add version tracking and upgrade check system
Date: 2025-10-03T04:46:44Z
```

## Step 5: Compare Versions and Suggest Upgrade

### Normalize versions (remove 'v' prefix)
```bash
bash(echo "v3.2.1" | sed 's/^v//')
```

### Compare two versions
```bash
bash(printf "%s\n%s" "3.2.1" "3.2.2" | sort -V | tail -n 1)
```

### Check if versions are equal
```bash
# If equal: Up to date
# If remote newer: Upgrade available
# If local newer: Development version
```

**Output Scenarios**:

**Scenario 1: Up to date**
```
You are on the latest stable version (3.2.1)
```

**Scenario 2: Upgrade available**
```
A newer stable version is available: v3.2.2
Your version: 3.2.1

To upgrade:
PowerShell: iex (iwr -useb https://raw.githubusercontent.com/catlog22/Claude-Code-Workflow/main/install-remote.ps1)
Bash: bash <(curl -fsSL https://raw.githubusercontent.com/catlog22/Claude-Code-Workflow/main/install-remote.sh)
```

**Scenario 3: Development version**
```
You are running a development version (3.4.0-dev)
This is newer than the latest stable release (v3.3.0)
```

## Simple Bash Commands

### Basic Operations
```bash
# Check local version file
bash(test -f ./.claude/version.json && cat ./.claude/version.json)

# Check global version file
bash(test -f ~/.claude/version.json && cat ~/.claude/version.json)

# Extract version from JSON
bash(cat version.json | grep -o '"version": *"[^"]*"' | cut -d'"' -f4)

# Extract date from JSON
bash(cat version.json | grep -o '"installation_date_utc": *"[^"]*"' | cut -d'"' -f4)

# Fetch latest release (with timeout)
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/releases/latest" 2>/dev/null, timeout: 30000)

# Extract tag name
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/releases/latest" 2>/dev/null | grep -o '"tag_name": *"[^"]*"' | cut -d'"' -f4, timeout: 30000)

# Extract release name
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/releases/latest" 2>/dev/null | grep -o '"name": *"[^"]*"' | head -1 | cut -d'"' -f4, timeout: 30000)

# Fetch latest commit (with timeout)
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/commits/main" 2>/dev/null, timeout: 30000)

# Extract commit SHA
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/commits/main" 2>/dev/null | grep -o '"sha": *"[^"]*"' | head -1 | cut -d'"' -f4 | cut -c1-7, timeout: 30000)

# Extract commit message (first line)
bash(curl -fsSL "https://api.github.com/repos/catlog22/Claude-Code-Workflow/commits/main" 2>/dev/null | grep '"message":' | cut -d'"' -f4 | cut -d'\' -f1, timeout: 30000)

# Compare versions
bash(printf "%s\n%s" "3.2.1" "3.2.2" | sort -V | tail -n 1)

# Remove 'v' prefix
bash(echo "v3.2.1" | sed 's/^v//')
```

## Error Handling

### No installation found
```
WARNING: Claude Code Workflow not installed
Install using:
PowerShell: iex (iwr -useb https://raw.githubusercontent.com/catlog22/Claude-Code-Workflow/main/install-remote.ps1)
```

### Network error
```
ERROR: Could not fetch latest version from GitHub
Check your network connection
```

### Invalid version.json
```
ERROR: version.json is invalid or corrupted
```

## Design Notes

- Uses simple, direct bash commands instead of complex functions
- Each step is independent and can be executed separately
- Fallback to grep/sed for JSON parsing (no jq dependency required)
- Network calls use curl with error suppression and 30-second timeout
- Version comparison uses `sort -V` for accurate semantic versioning
- Use `/commits/main` API instead of `/branches/main` for more reliable commit info
- Extract first line of commit message using `cut -d'\' -f1` to handle JSON escape sequences

## API Endpoints

### GitHub API Used
- **Latest Release**: `https://api.github.com/repos/catlog22/Claude-Code-Workflow/releases/latest`
  - Fields: `tag_name`, `name`, `published_at`
- **Latest Commit**: `https://api.github.com/repos/catlog22/Claude-Code-Workflow/commits/main`
  - Fields: `sha`, `commit.message`, `commit.author.date`

### Timeout Configuration
All network calls should use `timeout: 30000` (30 seconds) to handle slow connections.
