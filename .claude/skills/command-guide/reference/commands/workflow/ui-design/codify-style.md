---
name: workflow:ui-design:codify-style
description: Orchestrator to extract styles from code and generate shareable reference package with preview (automatic file discovery)
argument-hint: "<path> [--package-name <name>] [--output-dir <path>] [--overwrite]"
allowed-tools: SlashCommand,Bash,Read,TodoWrite
auto-continue: true
---

# UI Design: Codify Style (Orchestrator)

## Overview & Execution Model

**Fully autonomous orchestrator**: Coordinates style extraction from codebase and generates shareable reference packages.

**Pure Orchestrator Pattern**: Does NOT directly execute agent tasks. Delegates to specialized commands:
1. `/workflow:ui-design:import-from-code` - Extract styles from source code
2. `/workflow:ui-design:reference-page-generator` - Generate versioned reference package with interactive preview

**Output**: Shareable, versioned style reference package at `.workflow/reference_style/{package-name}/`

**Autonomous Flow** (⚠️ CONTINUOUS EXECUTION - DO NOT STOP):
1. User triggers: `/workflow:ui-design:codify-style <path> --package-name <name>`
2. Phase 0: Parameter validation & preparation → **IMMEDIATELY triggers Phase 1**
3. Phase 1 (import-from-code) → **Attach 4 tasks → Execute tasks → Collapse** → Auto-continues to Phase 2
4. Phase 2 (reference-page-generator) → **Attach 4 tasks → Execute tasks → Collapse** → Auto-continues to Phase 3
5. Phase 3 (cleanup & verification) → **Execute orchestrator task** → Reports completion

**Phase Transition Mechanism**:
- **Phase 0 (Validation)**: Validate parameters, prepare workspace → IMMEDIATELY triggers Phase 1
- **Phase 1-2 (Task Attachment)**: `SlashCommand` invocation **ATTACHES** tasks to current workflow. Orchestrator **EXECUTES** these tasks itself.
- **Task Execution**: Orchestrator runs attached tasks sequentially, updating TodoWrite as each completes
- **Task Collapse**: After all attached tasks complete, collapse them into phase summary
- **Phase Transition**: Automatically execute next phase after collapsing completed tasks
- No user interaction required after initial command

**Auto-Continue Mechanism**: TodoWrite tracks phase status with dynamic task attachment/collapse. After executing all attached tasks, you MUST immediately collapse them, restore phase summary, and execute the next phase. No user intervention required. The workflow is NOT complete until reaching Phase 3.

**Task Attachment Model**: SlashCommand invocation is NOT delegation - it's task expansion. The orchestrator executes these attached tasks itself, not waiting for external completion.

## Core Rules

1. **Start Immediately**: TodoWrite initialization → Phase 0 validation → Phase 1 execution
2. **No Task JSON**: This command does not create task JSON files - pure orchestrator pattern
3. **Parse & Pass**: Extract required data from each command output (design run path, metadata)
4. **Intelligent Validation**: Smart parameter validation with user-friendly error messages
5. **Safety First**: Package overwrite protection, existence checks, fallback error handling
6. **Track Progress**: Update TodoWrite dynamically with task attachment/collapse pattern
7. **⚠️ CRITICAL: Task Attachment Model** - SlashCommand invocation **ATTACHES** tasks to current workflow. Orchestrator **EXECUTES** these attached tasks itself, not waiting for external completion. This is NOT delegation - it's task expansion.
8. **⚠️ CRITICAL: DO NOT STOP** - This is a continuous multi-phase workflow. After executing all attached tasks, you MUST immediately collapse them and execute the next phase. Workflow is NOT complete until Phase 3.

---

## Usage

### Command Syntax

```bash
/workflow:ui-design:codify-style <path> [OPTIONS]

# Required
<path>                  Source code directory to analyze

# Optional
--package-name <name>   Custom name for the style reference package
                        (default: auto-generated from directory name)
--output-dir <path>     Output directory (default: .workflow/reference_style)
--overwrite             Overwrite existing package without prompting
```

**Note**: File discovery is fully automatic. The command will scan the source directory and find all style-related files (CSS, SCSS, JS, HTML) automatically.

---

## 4-Phase Execution

### Phase 0: Intelligent Parameter Validation & Session Preparation

**Goal**: Validate parameters, check safety constraints, prepare session, and get user confirmation

**TodoWrite** (First Action):
```json
[
  {"content": "Phase 0: Validate parameters and prepare session", "status": "in_progress", "activeForm": "Validating parameters"},
  {"content": "Phase 1: Style extraction from source code (import-from-code)", "status": "pending", "activeForm": "Extracting styles"},
  {"content": "Phase 2: Reference package generation (reference-page-generator)", "status": "pending", "activeForm": "Generating package"},
  {"content": "Phase 3: Cleanup and verify package", "status": "pending", "activeForm": "Cleanup and verification"}
]
```

**Note**: Orchestrator tracks only high-level phases. Sub-command details shown when executed.

**Step 0a: Parse and Validate Required Parameters**

```bash
# Parse positional path parameter (first non-flag argument)
source_path = FIRST_POSITIONAL_ARG

# Validate source path
IF NOT source_path:
    REPORT: "❌ ERROR: Missing required parameter: <path>"
    REPORT: "USAGE: /workflow:ui-design:codify-style <path> [OPTIONS]"
    REPORT: "EXAMPLE: /workflow:ui-design:codify-style ./src"
    REPORT: "EXAMPLE: /workflow:ui-design:codify-style ./app --package-name design-system-v2"
    EXIT 1

# Validate source path existence
TRY:
    source_exists = Bash(test -d "${source_path}" && echo "exists" || echo "not_exists")
    IF source_exists != "exists":
        REPORT: "❌ ERROR: Source directory not found: ${source_path}"
        REPORT: "Please provide a valid directory path."
        EXIT 1
CATCH error:
    REPORT: "❌ ERROR: Cannot validate source path: ${error}"
    EXIT 1

source = source_path
STORE: source

# Auto-generate package name if not provided
IF NOT --package-name:
    # Extract directory name from path
    dir_name = Bash(basename "${source}")
    # Normalize to package name format (lowercase, replace special chars with hyphens)
    normalized_name = dir_name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
    # Add version suffix
    package_name = "${normalized_name}-style-v1"

    ELSE:
    package_name = --package-name

    # Validate custom package name format (lowercase, alphanumeric, hyphens only)
    IF NOT package_name MATCHES /^[a-z0-9][a-z0-9-]*$/:
        REPORT: "❌ ERROR: Invalid package name format: ${package_name}"
        REPORT: "Requirements:"
        REPORT: "  • Must start with lowercase letter or number"
        REPORT: "  • Only lowercase letters, numbers, and hyphens allowed"
        REPORT: "  • No spaces or special characters"
        REPORT: "EXAMPLES: main-app-style-v1, design-system-v2, component-lib-v1"
        EXIT 1

STORE: package_name, output_dir (default: ".workflow/reference_style"), overwrite_flag
```

**Step 0b: Intelligent Package Safety Check**

```bash
# Set default output directory
output_dir = --output-dir OR ".workflow/reference_style"
package_path = "${output_dir}/${package_name}"

TRY:
    package_exists = Bash(test -d "${package_path}" && echo "exists" || echo "not_exists")

    IF package_exists == "exists":
        IF NOT --overwrite:
            REPORT: "❌ ERROR: Package '${package_name}' already exists at ${package_path}/"
            REPORT: "Use --overwrite flag to replace, or choose a different package name"
            EXIT 1
        ELSE:
            REPORT: "⚠️  Overwriting existing package: ${package_name}"

CATCH error:
    REPORT: "⚠️  Warning: Cannot check package existence: ${error}"
    REPORT: "Continuing with package creation..."
```

**Step 0c: Session Preparation**

```bash
# Create temporary workspace for processing
TRY:
    # Step 1: Ensure .workflow directory exists and generate unique ID
    Bash(mkdir -p .workflow)
    temp_id = Bash(echo "codify-temp-$(date +%Y%m%d-%H%M%S)")

    # Step 2: Create temporary directory
    Bash(mkdir -p ".workflow/${temp_id}")

    # Step 3: Get absolute path using bash
    design_run_path = Bash(cd ".workflow/${temp_id}" && pwd)

CATCH error:
    REPORT: "❌ ERROR: Failed to create temporary workspace: ${error}"
    EXIT 1

STORE: temp_id, design_run_path
```

**Summary Variables**:
- `SOURCE`: Validated source directory path
- `PACKAGE_NAME`: Validated package name (lowercase, alphanumeric, hyphens)
- `PACKAGE_PATH`: Full output path `${output_dir}/${package_name}`
- `OUTPUT_DIR`: `.workflow/reference_style` (default) or user-specified
- `OVERWRITE`: `true` if --overwrite flag present
- `CSS/SCSS/JS/HTML/STYLE_FILES`: Optional glob patterns
- `TEMP_ID`: `codify-temp-{timestamp}` (temporary workspace identifier)
- `DESIGN_RUN_PATH`: Absolute path to temporary workspace

<!-- TodoWrite: Update Phase 0 → completed, Phase 1 → in_progress, INSERT import-from-code tasks -->

**TodoWrite Update (Phase 1 SlashCommand invoked - tasks attached)**:
```json
[
  {"content": "Phase 0: Validate parameters and prepare session", "status": "completed", "activeForm": "Validating parameters"},
  {"content": "Phase 1.0: Discover and categorize code files (import-from-code)", "status": "in_progress", "activeForm": "Discovering code files"},
  {"content": "Phase 1.1: Style Agent extraction (import-from-code)", "status": "pending", "activeForm": "Extracting style tokens"},
  {"content": "Phase 1.2: Animation Agent extraction (import-from-code)", "status": "pending", "activeForm": "Extracting animation tokens"},
  {"content": "Phase 1.3: Layout Agent extraction (import-from-code)", "status": "pending", "activeForm": "Extracting layout patterns"},
  {"content": "Phase 2: Reference package generation (reference-page-generator)", "status": "pending", "activeForm": "Generating package"},
  {"content": "Phase 3: Cleanup and verify package", "status": "pending", "activeForm": "Cleanup and verification"}
]
```

**Note**: SlashCommand invocation **attaches** import-from-code's 4 tasks to current workflow. Orchestrator **executes** these tasks itself.

**Next Action**: Tasks attached → **Execute Phase 1.0-1.3** sequentially

---

### Phase 1: Style Extraction from Source Code

**Goal**: Extract design tokens, style patterns, and component styles from codebase

**Command Construction**:

```bash
# Build command with required parameters only
# Use temp_id as design-id since it's the workspace directory name
command = "/workflow:ui-design:import-from-code" +
          " --design-id \"${temp_id}\"" +
          " --source \"${source}\""
```

**Execute Command (Task Attachment Pattern)**:

```bash
TRY:
    # SlashCommand invocation ATTACHES import-from-code's 4 tasks to current workflow
    # Orchestrator will EXECUTE these attached tasks itself:
    #   1. Phase 1.0: Discover and categorize code files
    #   2. Phase 1.1: Style Agent extraction
    #   3. Phase 1.2: Animation Agent extraction
    #   4. Phase 1.3: Layout Agent extraction
    SlashCommand(command)

    # After executing all attached tasks, verify extraction outputs
    tokens_path = "${design_run_path}/style-extraction/style-1/design-tokens.json"
    guide_path = "${design_run_path}/style-extraction/style-1/style-guide.md"

    tokens_exists = Bash(test -f "${tokens_path}" && echo "exists" || echo "missing")
    guide_exists = Bash(test -f "${guide_path}" && echo "exists" || echo "missing")

    IF tokens_exists != "exists" OR guide_exists != "exists":
        REPORT: "⚠️  WARNING: Expected extraction files not found"
        REPORT: "Continuing with available outputs..."

CATCH error:
    REPORT: "❌ ERROR: Style extraction failed"
    REPORT: "Error: ${error}"
    REPORT: "Possible cause: Source directory contains no style files"
    Bash(rm -rf .workflow/${temp_id})
    EXIT 1
```

**Example Command**:
```bash
# Automatic file discovery
/workflow:ui-design:import-from-code --design-id codify-temp-20250111-123456 --source ./src
```

**Completion Criteria**:
- ✅ `import-from-code` command executed successfully
- ✅ Design run created at `${design_run_path}`
- ✅ Required files exist:
  - `design-tokens.json` - Complete design token system
  - `style-guide.md` - Style documentation
- ⭕ Optional files:
  - `animation-tokens.json` - Animation specifications
  - `component-patterns.json` - Component catalog

<!-- TodoWrite: REMOVE Phase 1.0-1.3 tasks, INSERT reference-page-generator tasks -->

**TodoWrite Update (Phase 2 SlashCommand invoked - tasks attached)**:
```json
[
  {"content": "Phase 0: Validate parameters and prepare session", "status": "completed", "activeForm": "Validating parameters"},
  {"content": "Phase 1: Style extraction from source code (import-from-code)", "status": "completed", "activeForm": "Extracting styles"},
  {"content": "Phase 2.0: Setup and validation (reference-page-generator)", "status": "in_progress", "activeForm": "Validating parameters"},
  {"content": "Phase 2.1: Prepare component data (reference-page-generator)", "status": "pending", "activeForm": "Copying layout templates"},
  {"content": "Phase 2.2: Generate preview pages (reference-page-generator)", "status": "pending", "activeForm": "Generating preview"},
  {"content": "Phase 3: Cleanup and verify package", "status": "pending", "activeForm": "Cleanup and verification"}
]
```

**Note**: Phase 1 tasks completed and collapsed. SlashCommand invocation **attaches** reference-page-generator's 3 tasks. Orchestrator **executes** these tasks itself.

**Next Action**: Tasks attached → **Execute Phase 2.0-2.2** sequentially

---

### Phase 2: Reference Package Generation

**Goal**: Generate shareable reference package with interactive preview and documentation

**Command Construction**:

```bash
command = "/workflow:ui-design:reference-page-generator " +
          "--design-run \"${design_run_path}\" " +
          "--package-name \"${package_name}\" " +
          "--output-dir \"${output_dir}\""
```

**Execute Command (Task Attachment Pattern)**:

```bash
TRY:
    # SlashCommand invocation ATTACHES reference-page-generator's 3 tasks to current workflow
    # Orchestrator will EXECUTE these attached tasks itself:
    #   1. Phase 2.0: Setup and validation
    #   2. Phase 2.1: Prepare component data
    #   3. Phase 2.2: Generate preview pages
    SlashCommand(command)

    # After executing all attached tasks, verify package outputs
    required_files = [
        "layout-templates.json",
        "design-tokens.json",
        "preview.html",
        "preview.css"
    ]

    missing_files = []
    FOR file IN required_files:
        file_path = "${package_path}/${file}"
        exists = Bash(test -f "${file_path}" && echo "exists" || echo "missing")
        IF exists != "exists":
            missing_files.append(file)

    IF missing_files.length > 0:
        REPORT: "⚠️  WARNING: Some expected files are missing"
        REPORT: "Package may be incomplete. Continuing with cleanup..."

CATCH error:
    REPORT: "❌ ERROR: Reference package generation failed"
    REPORT: "Error: ${error}"
    Bash(rm -rf .workflow/${temp_id})
    EXIT 1
```

**Example Command**:
```bash
/workflow:ui-design:reference-page-generator \
  --design-run .workflow/codify-temp-20250111-123456 \
  --package-name main-app-style-v1 \
  --output-dir .workflow/reference_style
```

**Completion Criteria**:
- ✅ `reference-page-generator` executed successfully
- ✅ Reference package created at `${package_path}/`
- ✅ All required files present:
  - `layout-templates.json` - Layout templates from design run
  - `design-tokens.json` - Complete design token system
  - `preview.html` - Interactive multi-component showcase
  - `preview.css` - Showcase styling
- ⭕ Optional files:
  - `animation-tokens.json` - Animation specifications (if available from extraction)

<!-- TodoWrite: REMOVE Phase 2.0-2.2 tasks, restore to orchestrator view -->

**TodoWrite Update (Phase 2 completed - tasks collapsed)**:
```json
[
  {"content": "Phase 0: Validate parameters and prepare session", "status": "completed", "activeForm": "Validating parameters"},
  {"content": "Phase 1: Style extraction from source code (import-from-code)", "status": "completed", "activeForm": "Extracting styles"},
  {"content": "Phase 2: Reference package generation (reference-page-generator)", "status": "completed", "activeForm": "Generating package"},
  {"content": "Phase 3: Cleanup and verify package", "status": "in_progress", "activeForm": "Cleanup and verification"}
]
```

**Note**: Phase 2 tasks completed and collapsed to summary.

**Next Action**: TodoWrite restored → **Execute Phase 3** (orchestrator's own task)

---

### Phase 3: Cleanup & Verification

**Goal**: Clean up temporary workspace and report completion

**Operations**:

```bash
# Cleanup temporary workspace
TRY:
    Bash(rm -rf .workflow/${temp_id})
CATCH error:
    # Silent failure - not critical

# Quick verification
package_exists = Bash(test -d "${package_path}" && echo "exists" || echo "missing")

IF package_exists != "exists":
    REPORT: "❌ ERROR: Package generation failed - directory not found"
    EXIT 1

# Get absolute path and component count for final report
absolute_package_path = Bash(cd "${package_path}" && pwd 2>/dev/null || echo "${package_path}")
component_count = Bash(jq -r '.layout_templates | length // "unknown"' "${package_path}/layout-templates.json" 2>/dev/null || echo "unknown")
anim_exists = Bash(test -f "${package_path}/animation-tokens.json" && echo "✓" || echo "○")
```

<!-- TodoWrite: Update Phase 3 → completed -->

**Final Action**: Display completion summary to user

---

## Completion Message

```
✅ Style reference package generated successfully

📦 Package: {package_name}
📂 Location: {absolute_package_path}/
📄 Source: {source}
📊 Components: {component_count}

Files: layout-templates.json, design-tokens.json, animation-tokens.json (optional), preview.html, preview.css

Preview: file://{absolute_package_path}/preview.html

Next: /memory:style-skill-memory {package_name}
```

---

## TodoWrite Pattern

```javascript
// Initialize IMMEDIATELY at the start to track orchestrator workflow (4 high-level tasks)
TodoWrite({todos: [
  {"content": "Phase 0: Validate parameters and prepare session", "status": "in_progress", "activeForm": "Validating parameters"},
  {"content": "Phase 1: Style extraction from source code (import-from-code)", "status": "pending", "activeForm": "Extracting styles"},
  {"content": "Phase 2: Reference package generation (reference-page-generator)", "status": "pending", "activeForm": "Generating package"},
  {"content": "Phase 3: Cleanup and verify package", "status": "pending", "activeForm": "Cleanup and verification"}
]})

// ⚠️ CRITICAL: Dynamic TodoWrite task attachment strategy:
//
// **Key Concept**: SlashCommand invocation ATTACHES tasks to current workflow.
// Orchestrator EXECUTES these attached tasks itself, not waiting for external completion.
//
// 1. INITIAL STATE: 4 orchestrator-level tasks only
//
// 2. PHASE 1 SlashCommand INVOCATION:
//    - SlashCommand(/workflow:ui-design:import-from-code) ATTACHES 4 tasks
//    - TodoWrite expands to: Phase 0 (completed) + 4 import-from-code tasks + Phase 2 + Phase 3
//    - Orchestrator EXECUTES these 4 tasks sequentially (Phase 1.0 → 1.1 → 1.2 → 1.3)
//    - First attached task marked as in_progress
//
// 3. PHASE 1 TASKS COMPLETED:
//    - All 4 import-from-code tasks executed and completed
//    - COLLAPSE completed tasks into Phase 1 summary
//    - TodoWrite becomes: Phase 0-1 (completed) + Phase 2 + Phase 3
//
// 4. PHASE 2 SlashCommand INVOCATION:
//    - SlashCommand(/workflow:ui-design:reference-page-generator) ATTACHES 3 tasks
//    - TodoWrite expands to: Phase 0-1 (completed) + 3 reference-page-generator tasks + Phase 3
//    - Orchestrator EXECUTES these 3 tasks sequentially (Phase 2.0 → 2.1 → 2.2)
//
// 5. PHASE 2 TASKS COMPLETED:
//    - All 3 reference-page-generator tasks executed and completed
//    - COLLAPSE completed tasks into Phase 2 summary
//    - TodoWrite returns to: Phase 0-2 (completed) + Phase 3 (in_progress)
//
// 6. PHASE 3 EXECUTION:
//    - Orchestrator's own task (no SlashCommand attachment)
//    - Mark Phase 3 as completed
//    - Final state: All 4 orchestrator tasks completed

// ✓ Dynamic attachment/collapse maintains clarity
```

---

## Execution Flow Diagram

```
User triggers: /workflow:ui-design:codify-style ./src --package-name my-style-v1
  ↓
[TodoWrite Init] 4 orchestrator-level tasks
  ├─ Phase 0: Validate parameters and prepare session (in_progress)
  ├─ Phase 1: Style extraction from source code (pending)
  ├─ Phase 2: Reference package generation (pending)
  └─ Phase 3: Cleanup and verify package (pending)
  ↓
[Phase 0] Parameter validation & preparation
  ├─ Parse positional path parameter
  ├─ Validate source directory exists
  ├─ Auto-generate or validate package name
  ├─ Check package overwrite protection
  ├─ Create temporary workspace
  └─ Display configuration summary
  ↓
[Phase 0 Complete] → TodoWrite: Phase 0 → completed
  ↓
[Phase 1 Invoke] → SlashCommand(/workflow:ui-design:import-from-code) ATTACHES 4 tasks
  ├─ Phase 0 (completed)
  ├─ Phase 1.0: Discover and categorize code files (in_progress)  ← ATTACHED
  ├─ Phase 1.1: Style Agent extraction (pending)                   ← ATTACHED
  ├─ Phase 1.2: Animation Agent extraction (pending)               ← ATTACHED
  ├─ Phase 1.3: Layout Agent extraction (pending)                  ← ATTACHED
  ├─ Phase 2: Reference package generation (pending)
  └─ Phase 3: Cleanup and verify package (pending)
  ↓
[Execute Phase 1.0] → Discover files (orchestrator executes this)
  ↓
[Execute Phase 1.1-1.3] → Run 3 agents in parallel (orchestrator executes these)
  └─ Outputs: design-tokens.json, style-guide.md, animation-tokens.json, layout-templates.json
  ↓
[Phase 1 Complete] → TodoWrite: COLLAPSE Phase 1.0-1.3 into Phase 1 summary
  ↓
[Phase 2 Invoke] → SlashCommand(/workflow:ui-design:reference-page-generator) ATTACHES 3 tasks
  ├─ Phase 0 (completed)
  ├─ Phase 1: Style extraction from source code (completed)        ← COLLAPSED
  ├─ Phase 2.0: Setup and validation (in_progress)                 ← ATTACHED
  ├─ Phase 2.1: Prepare component data (pending)                   ← ATTACHED
  ├─ Phase 2.2: Generate preview pages (pending)                   ← ATTACHED
  └─ Phase 3: Cleanup and verify package (pending)
  ↓
[Execute Phase 2.0] → Setup and validation (orchestrator executes this)
  ↓
[Execute Phase 2.1] → Prepare component data (orchestrator executes this)
  ↓
[Execute Phase 2.2] → Generate preview pages (orchestrator executes this)
  └─ Outputs: layout-templates.json, design-tokens.json, animation-tokens.json (optional), preview.html, preview.css
  ↓
[Phase 2 Complete] → TodoWrite: COLLAPSE Phase 2.0-2.2 into Phase 2 summary
  ├─ Phase 0 (completed)
  ├─ Phase 1: Style extraction from source code (completed)
  ├─ Phase 2: Reference package generation (completed)             ← COLLAPSED
  └─ Phase 3: Cleanup and verify package (in_progress)
  ↓
[Execute Phase 3] → Orchestrator's own task (no attachment needed)
  ├─ Remove temporary workspace (.workflow/codify-temp-{timestamp}/)
  ├─ Verify package directory
  ├─ Extract component count
  └─ Display completion summary
  ↓
[Phase 3 Complete] → TodoWrite: Phase 3 → completed
  ├─ Phase 0 (completed)
  ├─ Phase 1 (completed)
  ├─ Phase 2 (completed)
  └─ Phase 3 (completed)
```

---

## Error Handling

### Common Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| Missing --source or --package-name | Required parameters not provided | Provide both flags |
| Invalid package name | Contains uppercase, special chars | Use lowercase, alphanumeric, hyphens only |
| import-from-code failed | Source path invalid or no files found | Verify source path, check glob patterns |
| reference-page-generator failed | Design run incomplete | Check import-from-code output, verify extraction files |
| Package verification failed | Output directory creation failed | Check file permissions |

### Error Recovery

- If Phase 2 fails: Cleanup temporary session and report error
- If Phase 3 fails: Keep design run for debugging, report error
- User can manually inspect `${design_run_path}` if needed

---

## Implementation Details

### Critical Rules

1. **No User Prompts Between Phases**: Never ask user questions or wait for input between phases
2. **Immediate Phase Transition**: After TodoWrite update, immediately execute next phase command
3. **Status-Driven Execution**: Check TodoList status after each phase
4. **Phase Completion Pattern**:
   ```
   Phase N completes → Update TodoWrite (N=completed, N+1=in_progress) → Execute Phase N+1
   ```

### Parameter Pass-Through

Only essential parameters are passed to `import-from-code`:
- `--design-id` → temporary design run ID (auto-generated)
- `--source` → user-specified source directory

File discovery is fully automatic - no glob patterns needed.

### Output Directory Structure

```
.workflow/
├── reference_style/              # Default output directory
│   └── {package-name}/
│       ├── layout-templates.json
│       ├── design-tokens.json
│       ├── animation-tokens.json (optional)
│       ├── preview.html
│       └── preview.css
│
└── codify-temp-{timestamp}/      # Temporary workspace (cleaned up after completion)
    ├── style-extraction/
    ├── animation-extraction/
    └── layout-extraction/
```

---



## Architecture

```
codify-style (orchestrator - simplified interface)
  ├─ Phase 0: Intelligent Validation
  │   ├─ Parse positional path parameter
  │   ├─ Auto-generate package name (if not provided)
  │   ├─ Safety checks (overwrite protection)
  │   └─ User confirmation
  ├─ Phase 1: /workflow:ui-design:import-from-code (style extraction)
  │   ├─ Extract design tokens from source code
  │   ├─ Generate style guide
  │   └─ Extract component patterns
  ├─ Phase 2: /workflow:ui-design:reference-page-generator (reference package)
  │   ├─ Generate shareable package
  │   ├─ Create interactive preview
  │   └─ Generate documentation
  └─ Phase 3: Cleanup & Verification
      ├─ Remove temporary workspace
      ├─ Verify package integrity
      └─ Report completion

Design Principles:
✓ No task JSON created by this command
✓ All extraction delegated to import-from-code
✓ All packaging delegated to reference-page-generator
✓ Pure orchestration with intelligent defaults
✓ Single path parameter for maximum simplicity
```
