---
name: cli-init
description: Generate .gemini/ and .qwen/ config directories with settings.json and ignore files based on workspace technology detection
argument-hint: "[--tool gemini|qwen|all] [--output path] [--preview]"
allowed-tools: Bash(*), Read(*), Write(*), Glob(*)
---

# CLI Initialization Command (/cli:cli-init)

## Overview
Initializes CLI tool configurations for the workspace by:
1. Analyzing current workspace using `get_modules_by_depth.sh` to identify technology stacks
2. Generating ignore files (`.geminiignore` and `.qwenignore`) with filtering rules optimized for detected technologies
3. Creating configuration directories (`.gemini/` and `.qwen/`) with settings.json files

**Supported Tools**: gemini, qwen, all (default: all)

## Core Functionality

### Configuration Generation
1. **Workspace Analysis**: Runs `get_modules_by_depth.sh` to analyze project structure
2. **Technology Stack Detection**: Identifies tech stacks based on file extensions, directories, and configuration files
3. **Config Creation**: Generates tool-specific configuration directories and settings files
4. **Ignore Rules Generation**: Creates ignore files with filtering patterns for detected technologies

### Generated Files

#### Configuration Directories
Creates tool-specific configuration directories:

**For Gemini** (`.gemini/`):
- `.gemini/settings.json`:
```json
{
  "contextfilename": ["CLAUDE.md","GEMINI.md"]
}
```

**For Qwen** (`.qwen/`):
- `.qwen/settings.json`:
```json
{
  "contextfilename": ["CLAUDE.md","QWEN.md"]
}
```

#### Ignore Files
Uses gitignore syntax to filter files from CLI tool analysis:
- `.geminiignore` - For Gemini CLI
- `.qwenignore` - For Qwen CLI

Both files have identical content based on detected technologies.

### Supported Technology Stacks

#### Frontend Technologies
- **React/Next.js**: Ignores build artifacts, .next/, node_modules
- **Vue/Nuxt**: Ignores .nuxt/, dist/, .cache/
- **Angular**: Ignores dist/, .angular/, node_modules
- **Webpack/Vite**: Ignores build outputs, cache directories

#### Backend Technologies
- **Node.js**: Ignores node_modules, package-lock.json, npm-debug.log
- **Python**: Ignores __pycache__, .venv, *.pyc, .pytest_cache
- **Java**: Ignores target/, .gradle/, *.class, .mvn/
- **Go**: Ignores vendor/, *.exe, go.sum (when appropriate)
- **C#/.NET**: Ignores bin/, obj/, *.dll, *.pdb

#### Database & Infrastructure
- **Docker**: Ignores .dockerignore, docker-compose.override.yml
- **Kubernetes**: Ignores *.secret.yaml, helm charts temp files
- **Database**: Ignores *.db, *.sqlite, database dumps

### Generated Rules Structure

#### Base Rules (Always Included)
```
# Version Control
.git/
.svn/
.hg/

# OS Files
.DS_Store
Thumbs.db
*.tmp
*.swp

# IDE Files
.vscode/
.idea/
.vs/

# Logs
*.log
logs/
```

#### Technology-Specific Rules
Rules are added based on detected technologies:

**Node.js Projects** (package.json detected):
```
# Node.js
node_modules/
npm-debug.log*
.npm/
.yarn/
package-lock.json
yarn.lock
.pnpm-store/
```

**Python Projects** (requirements.txt, setup.py, pyproject.toml detected):
```
# Python
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.coverage
htmlcov/
```

**Java Projects** (pom.xml, build.gradle detected):
```
# Java
target/
.gradle/
*.class
*.jar
*.war
.mvn/
```

## Command Options

### Tool Selection

**Initialize All Tools (default)**:
```bash
/cli:cli-init
```
- Creates `.gemini/`, `.qwen/` directories with settings.json
- Creates `.geminiignore` and `.qwenignore` files
- Sets contextfilename to "CLAUDE.md" for both

**Initialize Gemini Only**:
```bash
/cli:cli-init --tool gemini
```
- Creates only `.gemini/` directory and `.geminiignore` file

**Initialize Qwen Only**:
```bash
/cli:cli-init --tool qwen
```
- Creates only `.qwen/` directory and `.qwenignore` file

### Preview Mode
```bash
/cli:cli-init --preview
```
- Shows what would be generated without creating files
- Displays detected technologies, configuration, and ignore rules

### Custom Output Path
```bash
/cli:cli-init --output=.config/
```
- Generates files in specified directory
- Creates directories if they don't exist

### Combined Options
```bash
/cli:cli-init --tool qwen --preview
/cli:cli-init --tool all --output=.config/
```

## EXECUTION INSTRUCTIONS - START HERE

**When this command is triggered, follow these exact steps:**

### Step 1: Parse Tool Selection
```bash
# Extract --tool flag (default: all)
# Options: gemini, qwen, all
```

### Step 2: Workspace Analysis (MANDATORY FIRST)
```bash
# Analyze workspace structure
bash(ccw tool exec get_modules_by_depth '{"format":"json"}')
```

### Step 3: Technology Detection
```bash
# Check for common tech stack indicators
bash(find . -name "package.json" -not -path "*/node_modules/*" | head -1)
bash(find . -name "requirements.txt" -o -name "setup.py" -o -name "pyproject.toml" | head -1)
bash(find . -name "pom.xml" -o -name "build.gradle" | head -1)
bash(find . -name "Dockerfile" | head -1)
```

### Step 4: Generate Configuration Files

**For Gemini** (if --tool is gemini or all):
```bash
# Create .gemini/ directory and settings.json
mkdir -p .gemini
Write({file_path: '.gemini/settings.json', content: '{"contextfilename": "CLAUDE.md"}'})

# Create .geminiignore file with detected technology rules
# Backup existing files if present
```

**For Qwen** (if --tool is qwen or all):
```bash
# Create .qwen/ directory and settings.json
mkdir -p .qwen
Write({file_path: '.qwen/settings.json', content: '{"contextfilename": "CLAUDE.md"}'})

# Create .qwenignore file with detected technology rules
# Backup existing files if present
```

### Step 5: Validation
```bash
# Verify generated files are valid
bash(ls -la .gemini* .qwen* 2>/dev/null || echo "Configuration files created")
```

## Implementation Process (Technical Details)

### Phase 1: Tool Selection
1. Parse `--tool` flag from command arguments
2. Determine which configurations to generate:
   - `gemini`: Generate .gemini/ and .geminiignore only
   - `qwen`: Generate .qwen/ and .qwenignore only
   - `all` (default): Generate both sets of files

### Phase 2: Workspace Analysis
1. Execute `get_modules_by_depth.sh json` to get structured project data
2. Parse JSON output to identify directories and files
3. Scan for technology indicators:
   - Configuration files (package.json, requirements.txt, etc.)
   - Directory patterns (src/, tests/, etc.)
   - File extensions (.js, .py, .java, etc.)
4. Detect project name from directory name or package.json

### Phase 3: Technology Detection
```bash
# Technology detection logic
detect_nodejs() {
    [ -f "package.json" ] || find . -name "package.json" -not -path "*/node_modules/*" | head -1
}

detect_python() {
    [ -f "requirements.txt" ] || [ -f "setup.py" ] || [ -f "pyproject.toml" ] || \
    find . -name "*.py" -not -path "*/__pycache__/*" | head -1
}

detect_java() {
    [ -f "pom.xml" ] || [ -f "build.gradle" ] || \
    find . -name "*.java" | head -1
}
```

### Phase 4: Configuration Generation
**For each selected tool**, create:

1. **Config Directory**:
   - Create `.gemini/` or `.qwen/` directory if it doesn't exist
   - Generate `settings.json` with contextfilename setting
   - Set contextfilename to "CLAUDE.md" by default

2. **Settings.json Format** (identical for both tools):
```json
{
  "contextfilename": "CLAUDE.md"
}
```

### Phase 5: Ignore Rules Generation
1. Start with base rules (always included)
2. Add technology-specific rules based on detection
3. Add workspace-specific patterns if found
4. Sort and deduplicate rules
5. Generate identical content for both `.geminiignore` and `.qwenignore`

### Phase 6: File Creation
1. **Generate config directories**: Create `.gemini/` and/or `.qwen/` directories with settings.json
2. **Generate ignore files**: Create organized ignore files with sections
3. **Create backups**: Backup existing files if present
4. **Validate**: Check generated files are valid

## Generated File Format

### Configuration Files
```json
// .gemini/settings.json or .qwen/settings.json
{
  "contextfilename": "CLAUDE.md"
}
```

### Ignore Files
```
# .geminiignore / .qwenignore
# Generated by Claude Code /cli:cli-init command
# Creation date: 2024-01-15 10:30:00
# Detected technologies: Node.js, Python, Docker
#
# This file uses gitignore syntax to filter files for CLI tool analysis
# Edit this file to customize filtering rules for your project

# ============================================================================
# Base Rules (Always Applied)
# ============================================================================

# Version Control
.git/
.svn/
.hg/

# ============================================================================
# Node.js (Detected: package.json found)
# ============================================================================

node_modules/
npm-debug.log*
.npm/
yarn-error.log
package-lock.json

# ============================================================================
# Python (Detected: requirements.txt, *.py files found)
# ============================================================================

__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.coverage

# ============================================================================
# Docker (Detected: Dockerfile found)
# ============================================================================

.dockerignore
docker-compose.override.yml

# ============================================================================
# Custom Rules (Add your project-specific rules below)
# ============================================================================

```

## Error Handling

### Missing Dependencies
- If `get_modules_by_depth.sh` not found, show error with path to script
- Gracefully handle cases where script fails

### Write Permissions
- Check write permissions before attempting file creation
- Show clear error message if cannot write to target location

### Backup Existing Files
- If `.gemini/` directory exists, create backup as `.gemini.backup/`
- If `.qwen/` directory exists, create backup as `.qwen.backup/`
- If `.geminiignore` exists, create backup as `.geminiignore.backup`
- If `.qwenignore` exists, create backup as `.qwenignore.backup`
- Include timestamp in backup filename

## Integration Points

### Workflow Commands
- **After `/cli:plan`**: Suggest running cli-init for better analysis
- **Before analysis**: Recommend updating ignore patterns for cleaner results

### CLI Tool Integration
- Automatically update when new technologies detected
- Integrate with `intelligent-tools-strategy.md` recommendations

## Usage Examples

### Basic Project Setup
```bash
# Initialize all CLI tools (Gemini + Qwen)
/cli:cli-init

# Initialize only Gemini
/cli:cli-init --tool gemini

# Initialize only Qwen
/cli:cli-init --tool qwen

# Preview what would be generated
/cli:cli-init --preview

# Generate in subdirectory
/cli:cli-init --output=.config/
```

### Technology Migration
```bash
# After adding new tech stack (e.g., Docker)
/cli:cli-init  # Regenerates all config and ignore files with new rules

# Check what changed
/cli:cli-init --preview  # Compare with existing configuration

# Update only Qwen configuration
/cli:cli-init --tool qwen
```

### Tool-Specific Initialization
```bash
# Setup for Gemini-only workflow
/cli:cli-init --tool gemini

# Setup for Qwen-only workflow
/cli:cli-init --tool qwen

# Setup both with preview
/cli:cli-init --tool all --preview
```


## Tool Selection Guide

| Scenario | Command | Result |
|----------|---------|--------|
| **New project, using both tools** | `/cli:cli-init` | Creates .gemini/, .qwen/, .geminiignore, .qwenignore |
| **Gemini-only workflow** | `/cli:cli-init --tool gemini` | Creates .gemini/ and .geminiignore only |
| **Qwen-only workflow** | `/cli:cli-init --tool qwen` | Creates .qwen/ and .qwenignore only |
| **Preview before commit** | `/cli:cli-init --preview` | Shows what would be generated |
| **Update configurations** | `/cli:cli-init` | Regenerates all files with backups |
