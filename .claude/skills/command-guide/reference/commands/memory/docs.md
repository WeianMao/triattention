---
name: docs
description: Plan documentation workflow with dynamic grouping (≤10 docs/task), generates IMPL tasks for parallel module trees, README, ARCHITECTURE, and HTTP API docs
argument-hint: "[path] [--tool <gemini|qwen|codex>] [--mode <full|partial>] [--cli-execute]"
---

# Documentation Workflow (/memory:docs)

## Overview
Lightweight planner that analyzes project structure, decomposes documentation work into tasks, and generates execution plans. Does NOT generate documentation content itself - delegates to doc-generator agent.

**Execution Strategy**:
- **Dynamic Task Grouping**: Level 1 tasks grouped by top-level directories with document count limit
  - **Primary constraint**: Each task generates ≤10 documents (API.md + README.md count)
  - **Optimization goal**: Prefer grouping 2 top-level directories per task for context sharing
  - **Conflict resolution**: If 2 dirs exceed 10 docs, reduce to 1 dir/task; if 1 dir exceeds 10 docs, split by subdirectories
  - **Context benefit**: Same-task directories analyzed together via single Gemini call
- **Parallel Execution**: Multiple Level 1 tasks execute concurrently for faster completion
- **Pre-computed Analysis**: Phase 2 performs unified analysis once, stored in `.process/` for reuse
- **Efficient Data Loading**: All existing docs loaded once in Phase 2, shared across tasks

**Path Mirroring**: Documentation structure mirrors source code under `.workflow/docs/{project_name}/`
- Example: `my_app/src/core/` → `.workflow/docs/my_app/src/core/API.md`

**Two Execution Modes**:
- **Default (Agent Mode)**: CLI analyzes in `pre_analysis` (MODE=analysis), agent writes docs
- **--cli-execute (CLI Mode)**: CLI generates docs in `implementation_approach` (MODE=write), agent executes CLI commands

## Path Mirroring Strategy

**Principle**: Documentation structure **mirrors** source code structure under project-specific directory.

| Source Path | Project Name | Documentation Path |
|------------|--------------|-------------------|
| `my_app/src/core/` | `my_app` | `.workflow/docs/my_app/src/core/API.md` |
| `my_app/src/modules/auth/` | `my_app` | `.workflow/docs/my_app/src/modules/auth/API.md` |
| `another_project/lib/utils/` | `another_project` | `.workflow/docs/another_project/lib/utils/API.md` |


## Parameters

```bash
/memory:docs [path] [--tool <gemini|qwen|codex>] [--mode <full|partial>] [--cli-execute]
```

- **path**: Source directory to analyze (default: current directory)
  - Specifies the source code directory to be documented
  - Documentation is generated in a separate `.workflow/docs/{project_name}/` directory at the workspace root, **not** within the source `path` itself
  - The source path's structure is mirrored within the project-specific documentation folder
  - Example: analyzing `src/modules` produces documentation at `.workflow/docs/{project_name}/src/modules/`
- **--mode**: Documentation generation mode (default: full)
  - `full`: Complete documentation (modules + README + ARCHITECTURE + EXAMPLES + HTTP API)
  - `partial`: Module documentation only (API.md + README.md)
- **--tool**: CLI tool selection (default: gemini)
  - `gemini`: Comprehensive documentation, pattern recognition
  - `qwen`: Architecture analysis, system design focus
  - `codex`: Implementation validation, code quality
- **--cli-execute**: Enable CLI-based documentation generation (optional)

## Planning Workflow

### Phase 1: Initialize Session

```bash
# Get target path, project name, and root
bash(pwd && basename "$(pwd)" && git rev-parse --show-toplevel 2>/dev/null || pwd && date +%Y%m%d-%H%M%S)
```

```javascript
// Create docs session (type: docs)
SlashCommand(command="/workflow:session:start --type docs --new \"{project_name}-docs-{timestamp}\"")
// Parse output to get sessionId
```

```bash
# Update workflow-session.json with docs-specific fields
bash(jq '. + {"target_path":"{target_path}","project_root":"{project_root}","project_name":"{project_name}","mode":"full","tool":"gemini","cli_execute":false}' .workflow/active/{sessionId}/workflow-session.json > tmp.json && mv tmp.json .workflow/active/{sessionId}/workflow-session.json)
```

### Phase 2: Analyze Structure

**Smart filter**: Auto-detect and skip tests/build/config/vendor based on project tech stack.

**Commands** (collect data with simple bash):

```bash
# 1. Run folder analysis
bash(ccw tool exec get_modules_by_depth '{}' | ccw tool exec classify_folders '{}')

# 2. Get top-level directories (first 2 path levels)
bash(ccw tool exec get_modules_by_depth '{}' | ccw tool exec classify_folders '{}' | awk -F'|' '{print $1}' | sed 's|^\./||' | awk -F'/' '{if(NF>=2) print $1"/"$2; else if(NF==1) print $1}' | sort -u)

# 3. Find existing docs (if directory exists)
bash(if [ -d .workflow/docs/\${project_name} ]; then find .workflow/docs/\${project_name} -type f -name "*.md" ! -path "*/README.md" ! -path "*/ARCHITECTURE.md" ! -path "*/EXAMPLES.md" ! -path "*/api/*" 2>/dev/null; fi)

# 4. Read existing docs content (if files exist)
bash(if [ -d .workflow/docs/\${project_name} ]; then find .workflow/docs/\${project_name} -type f -name "*.md" ! -path "*/README.md" ! -path "*/ARCHITECTURE.md" ! -path "*/EXAMPLES.md" ! -path "*/api/*" 2>/dev/null | xargs cat 2>/dev/null; fi)
```

**Data Processing**: Parse bash outputs, calculate statistics, use **Write tool** to create `${session_dir}/.process/doc-planning-data.json` with structure:

```json
{
  "metadata": {
    "generated_at": "2025-11-03T16:57:30.469669",
    "project_name": "project_name",
    "project_root": "/path/to/project"
  },
  "folder_analysis": [
    {"path": "./src/core", "type": "code", "code_count": 5, "dirs_count": 2}
  ],
  "top_level_dirs": ["src/modules", "lib/core"],
  "existing_docs": {
    "file_list": [".workflow/docs/project/src/core/API.md"],
    "content": "... existing docs content ..."
  },
  "unified_analysis": [],
  "statistics": {
    "total": 15,
    "code": 8,
    "navigation": 7,
    "top_level": 3
  }
}
```

**Then** use **Edit tool** to update `workflow-session.json` adding analysis field.

**Output**: Single `doc-planning-data.json` with all analysis data (no temp files or Python scripts).

**Auto-skipped**: Tests (`**/test/**`, `**/*.test.*`), Build (`**/node_modules/**`, `**/dist/**`), Config (root-level files), Vendor directories.

### Phase 3: Detect Update Mode

**Commands**:

```bash
# Count existing docs from doc-planning-data.json
bash(cat .workflow/active/WFS-docs-{timestamp}/.process/doc-planning-data.json | jq '.existing_docs.file_list | length')
```

**Data Processing**: Use count result, then use **Edit tool** to update `workflow-session.json`:
- Add `"update_mode": "update"` if count > 0, else `"create"`
- Add `"existing_docs": <count>`

### Phase 4: Decompose Tasks

**Task Hierarchy** (Dynamic based on document count):

```
Small Projects (total ≤10 docs):
  Level 1: IMPL-001 (all directories in single task, shared context)
  Level 2: IMPL-002 (README, full mode only)
  Level 3: IMPL-003 (ARCHITECTURE+EXAMPLES), IMPL-004 (HTTP API, optional)

Medium Projects (Example: 7 top-level dirs, 18 total docs):
  Step 1: Count docs per top-level dir
    ├─ dir1: 3 docs, dir2: 4 docs → Group 1 (7 docs)
    ├─ dir3: 5 docs, dir4: 3 docs → Group 2 (8 docs)
    ├─ dir5: 2 docs → Group 3 (2 docs, can add more)

  Step 2: Create tasks with ≤10 docs constraint
  Level 1: IMPL-001 to IMPL-003 (parallel groups)
    ├─ IMPL-001: Group 1 (dir1 + dir2, 7 docs, shared context)
    ├─ IMPL-002: Group 2 (dir3 + dir4, 8 docs, shared context)
    └─ IMPL-003: Group 3 (remaining dirs, ≤10 docs)
  Level 2: IMPL-004 (README, depends on Level 1, full mode only)
  Level 3: IMPL-005 (ARCHITECTURE+EXAMPLES), IMPL-006 (HTTP API, optional)

Large Projects (single dir >10 docs):
  Step 1: Detect oversized directory
    └─ src/modules/: 15 subdirs → 30 docs (exceeds limit)

  Step 2: Split by subdirectories
  Level 1: IMPL-001 to IMPL-003 (split oversized dir)
    ├─ IMPL-001: src/modules/ subdirs 1-5 (10 docs)
    ├─ IMPL-002: src/modules/ subdirs 6-10 (10 docs)
    └─ IMPL-003: src/modules/ subdirs 11-15 (10 docs)
```

**Grouping Algorithm**:
1. Count total docs for each top-level directory
2. Try grouping 2 directories (optimization for context sharing)
3. If group exceeds 10 docs, split to 1 dir/task
4. If single dir exceeds 10 docs, split by subdirectories
5. Create parallel Level 1 tasks with ≤10 docs each


**Commands**:

```bash
# 1. Get top-level directories from doc-planning-data.json
bash(cat .workflow/active/WFS-docs-{timestamp}/.process/doc-planning-data.json | jq -r '.top_level_dirs[]')

# 2. Get mode from workflow-session.json
bash(cat .workflow/active/WFS-docs-{timestamp}/workflow-session.json | jq -r '.mode // "full"')

# 3. Check for HTTP API
bash(grep -r "router\.|@Get\|@Post" src/ 2>/dev/null && echo "API_FOUND" || echo "NO_API")
```

**Data Processing**:
1. Count documents for each top-level directory (from folder_analysis):
   - Code folders: 2 docs each (API.md + README.md)
   - Navigation folders: 1 doc each (README.md only)
2. Apply grouping algorithm with ≤10 docs constraint:
   - Try grouping 2 directories, calculate total docs
   - If total ≤10 docs: create group
   - If total >10 docs: split to 1 dir/group or subdivide
   - If single dir >10 docs: split by subdirectories
3. Use **Edit tool** to update `doc-planning-data.json` adding groups field:
   ```json
   "groups": {
     "count": 3,
     "assignments": [
       {"group_id": "001", "directories": ["src/modules", "src/utils"], "doc_count": 5},
       {"group_id": "002", "directories": ["lib/core"], "doc_count": 6},
       {"group_id": "003", "directories": ["lib/helpers"], "doc_count": 3}
     ]
   }
   ```

**Task ID Calculation**:
```bash
group_count=$(jq '.groups.count' .workflow/active/WFS-docs-{timestamp}/.process/doc-planning-data.json)
readme_id=$((group_count + 1))   # Next ID after groups
arch_id=$((group_count + 2))
api_id=$((group_count + 3))
```

### Phase 5: Generate Task JSONs

**CLI Strategy**:

| Mode | cli_execute | Placement | CLI MODE | Approval Flag | Agent Role |
|------|-------------|-----------|----------|---------------|------------|
| **Agent** | false | pre_analysis | analysis | (none) | Generate docs in implementation_approach |
| **CLI** | true | implementation_approach | write | --approval-mode yolo | Execute CLI commands, validate output |

**Command Patterns**:
- Gemini/Qwen: `cd dir && gemini -p "..."`
- CLI Mode: `cd dir && gemini --approval-mode yolo -p "..."`
- Codex: `codex -C dir --full-auto exec "..." --skip-git-repo-check -s danger-full-access`

**Generation Process**:
1. Read configuration values (tool, cli_execute, mode) from workflow-session.json
2. Read group assignments from doc-planning-data.json
3. Generate Level 1 tasks (IMPL-001 to IMPL-N, one per group)
4. Generate Level 2+ tasks if mode=full (README, ARCHITECTURE, HTTP API)

## Task Templates

### Level 1: Module Trees Group Task (Unified)

**Execution Model**: Each task processes assigned directory group (max 2 directories) using pre-analyzed data from Phase 2.

```json
{
  "id": "IMPL-${group_number}",
  "title": "Document Module Trees Group ${group_number}",
  "status": "pending",
  "meta": {
    "type": "docs-tree-group",
    "agent": "@doc-generator",
    "tool": "gemini",
    "cli_execute": false,
    "group_number": "${group_number}",
    "total_groups": "${total_groups}"
  },
  "context": {
    "requirements": [
      "Process directories from group ${group_number} in doc-planning-data.json",
      "Generate docs to .workflow/docs/${project_name}/ (mirrored structure)",
      "Code folders: API.md + README.md; Navigation folders: README.md only",
      "Use pre-analyzed data from Phase 2 (no redundant analysis)"
    ],
    "focus_paths": ["${group_dirs_from_json}"],
    "precomputed_data": {
      "phase2_analysis": "${session_dir}/.process/doc-planning-data.json"
    }
  },
  "flow_control": {
    "pre_analysis": [
      {
        "step": "load_precomputed_data",
        "action": "Load Phase 2 analysis and extract group directories",
        "commands": [
          "bash(cat ${session_dir}/.process/doc-planning-data.json)",
          "bash(jq '.groups.assignments[] | select(.group_id == \"${group_number}\") | .directories' ${session_dir}/.process/doc-planning-data.json)"
        ],
        "output_to": "phase2_context",
        "note": "Single JSON file contains all Phase 2 analysis results"
      }
    ],
    "implementation_approach": [
      {
        "step": 1,
        "title": "Generate documentation for assigned directory group",
        "description": "Process directories in Group ${group_number} using pre-analyzed data",
        "modification_points": [
          "Read group directories from [phase2_context].groups.assignments[${group_number}].directories",
          "For each directory: parse folder types from folder_analysis, parse structure from unified_analysis",
          "Map source_path to .workflow/docs/${project_name}/{path}",
          "Generate API.md for code folders, README.md for all folders",
          "Preserve user modifications from [phase2_context].existing_docs.content"
        ],
        "logic_flow": [
          "phase2 = parse([phase2_context])",
          "dirs = phase2.groups.assignments[${group_number}].directories",
          "for dir in dirs:",
          "  folder_info = find(dir, phase2.folder_analysis)",
          "  outline = find(dir, phase2.unified_analysis)",
          "  if folder_info.type == 'code': generate API.md + README.md",
          "  elif folder_info.type == 'navigation': generate README.md only",
          "  write to .workflow/docs/${project_name}/{dir}/"
        ],
        "depends_on": [],
        "output": "group_module_docs"
      }
    ],
    "target_files": [
      ".workflow/docs/${project_name}/*/API.md",
      ".workflow/docs/${project_name}/*/README.md"
    ]
  }
}
```

**CLI Execute Mode Note**: When `cli_execute=true`, add Step 2 in `implementation_approach`:
```json
{
  "step": 2,
  "title": "Batch generate documentation via CLI",
  "command": "bash(dirs=$(jq -r '.groups.assignments[] | select(.group_id == \"${group_number}\") | .directories[]' ${session_dir}/.process/doc-planning-data.json); for dir in $dirs; do cd \"$dir\" && gemini --approval-mode yolo -p \"PURPOSE: Generate module docs\\nTASK: Create documentation\\nMODE: write\\nCONTEXT: @**/* [phase2_context]\\nEXPECTED: API.md and README.md\\nRULES: Mirror structure\" || echo \"Failed: $dir\"; cd -; done)",
  "depends_on": [1],
  "output": "generated_docs"
}
```

### Level 2: Project README Task

**Task ID**: `IMPL-${readme_id}` (where `readme_id = group_count + 1`)
**Dependencies**: Depends on all Level 1 tasks completing.

```json
{
  "id": "IMPL-${readme_id}",
  "title": "Generate Project README",
  "status": "pending",
  "depends_on": ["IMPL-001", "...", "IMPL-${group_count}"],
  "meta": {"type": "docs", "agent": "@doc-generator", "tool": "gemini", "cli_execute": false},
  "flow_control": {
    "pre_analysis": [
      {
        "step": "load_existing_readme",
        "command": "bash(cat .workflow/docs/${project_name}/README.md 2>/dev/null || echo 'No existing README')",
        "output_to": "existing_readme"
      },
      {
        "step": "load_module_docs",
        "command": "bash(find .workflow/docs/${project_name} -type f -name '*.md' ! -path '.workflow/docs/${project_name}/README.md' ! -path '.workflow/docs/${project_name}/ARCHITECTURE.md' ! -path '.workflow/docs/${project_name}/EXAMPLES.md' ! -path '.workflow/docs/${project_name}/api/*' | xargs cat)",
        "output_to": "all_module_docs"
      },
      {
        "step": "analyze_project",
        "command": "bash(gemini \"PURPOSE: Analyze project structure\\nTASK: Extract overview from modules\\nMODE: analysis\\nCONTEXT: [all_module_docs]\\nEXPECTED: Project outline\")",
        "output_to": "project_outline"
      }
    ],
    "implementation_approach": [
      {
        "step": 1,
        "title": "Generate project README",
        "description": "Generate project README with navigation links while preserving user modifications",
        "modification_points": [
          "Parse [project_outline] and [all_module_docs]",
          "Generate README structure with navigation links",
          "Preserve [existing_readme] user modifications"
        ],
        "logic_flow": ["Parse data", "Generate README with navigation", "Preserve modifications"],
        "depends_on": [],
        "output": "project_readme"
      }
    ],
    "target_files": [".workflow/docs/${project_name}/README.md"]
  }
}
```

### Level 3: Architecture & Examples Documentation Task

**Task ID**: `IMPL-${arch_id}` (where `arch_id = group_count + 2`)
**Dependencies**: Depends on Level 2 (Project README).

```json
{
  "id": "IMPL-${arch_id}",
  "title": "Generate Architecture & Examples Documentation",
  "status": "pending",
  "depends_on": ["IMPL-${readme_id}"],
  "meta": {"type": "docs", "agent": "@doc-generator", "tool": "gemini", "cli_execute": false},
  "flow_control": {
    "pre_analysis": [
      {"step": "load_existing_docs", "command": "bash(cat .workflow/docs/${project_name}/{ARCHITECTURE,EXAMPLES}.md 2>/dev/null || echo 'No existing docs')", "output_to": "existing_arch_examples"},
      {"step": "load_all_docs", "command": "bash(cat .workflow/docs/${project_name}/README.md && find .workflow/docs/${project_name} -type f -name '*.md' ! -path '*/README.md' ! -path '*/ARCHITECTURE.md' ! -path '*/EXAMPLES.md' ! -path '*/api/*' | xargs cat)", "output_to": "all_docs"},
      {"step": "analyze_architecture", "command": "bash(gemini \"PURPOSE: Analyze system architecture\\nTASK: Synthesize architectural overview and examples\\nMODE: analysis\\nCONTEXT: [all_docs]\\nEXPECTED: Architecture + Examples outline\")", "output_to": "arch_examples_outline"}
    ],
    "implementation_approach": [
      {
        "step": 1,
        "title": "Generate architecture and examples documentation",
        "modification_points": [
          "Parse [arch_examples_outline] and [all_docs]",
          "Generate ARCHITECTURE.md (system design, patterns)",
          "Generate EXAMPLES.md (code snippets, usage)",
          "Preserve [existing_arch_examples] modifications"
        ],
        "depends_on": [],
        "output": "arch_examples_docs"
      }
    ],
    "target_files": [".workflow/docs/${project_name}/ARCHITECTURE.md", ".workflow/docs/${project_name}/EXAMPLES.md"]
  }
}
```

### Level 4: HTTP API Documentation Task (Optional)

**Task ID**: `IMPL-${api_id}` (where `api_id = group_count + 3`)
**Dependencies**: Depends on Level 3.

```json
{
  "id": "IMPL-${api_id}",
  "title": "Generate HTTP API Documentation",
  "status": "pending",
  "depends_on": ["IMPL-${arch_id}"],
  "meta": {"type": "docs", "agent": "@doc-generator", "tool": "gemini", "cli_execute": false},
  "flow_control": {
    "pre_analysis": [
      {"step": "discover_api", "command": "bash(rg 'router\\.| @(Get|Post)' -g '*.{ts,js}')", "output_to": "endpoint_discovery"},
      {"step": "load_existing_api", "command": "bash(cat .workflow/docs/${project_name}/api/README.md 2>/dev/null || echo 'No existing API docs')", "output_to": "existing_api_docs"},
      {"step": "analyze_api", "command": "bash(gemini \"PURPOSE: Document HTTP API\\nTASK: Analyze endpoints\\nMODE: analysis\\nCONTEXT: @src/api/**/* [endpoint_discovery]\\nEXPECTED: API outline\")", "output_to": "api_outline"}
    ],
    "implementation_approach": [
      {
        "step": 1,
        "title": "Generate HTTP API documentation",
        "modification_points": [
          "Parse [api_outline] and [endpoint_discovery]",
          "Document endpoints, request/response formats",
          "Preserve [existing_api_docs] modifications"
        ],
        "depends_on": [],
        "output": "api_docs"
      }
    ],
    "target_files": [".workflow/docs/${project_name}/api/README.md"]
  }
}
```

## Session Structure

**Unified Structure** (single JSON replaces multiple text files):

```
.workflow/active/
└── WFS-docs-{timestamp}/
    ├── workflow-session.json            # Session metadata
    ├── IMPL_PLAN.md
    ├── TODO_LIST.md
    ├── .process/
    │   └── doc-planning-data.json       # All Phase 2 analysis data (replaces 7+ files)
    └── .task/
        ├── IMPL-001.json                # Small: all modules | Large: group 1
        ├── IMPL-00N.json                # (Large only: groups 2-N)
        ├── IMPL-{N+1}.json              # README (full mode)
        ├── IMPL-{N+2}.json              # ARCHITECTURE+EXAMPLES (full mode)
        └── IMPL-{N+3}.json              # HTTP API (optional)
```

**doc-planning-data.json Structure**:
```json
{
  "metadata": {
    "generated_at": "2025-11-03T16:41:06+08:00",
    "project_name": "Claude_dms3",
    "project_root": "/d/Claude_dms3"
  },
  "folder_analysis": [
    {"path": "./src/core", "type": "code", "code_count": 5, "dirs_count": 2},
    {"path": "./src/utils", "type": "navigation", "code_count": 0, "dirs_count": 4}
  ],
  "top_level_dirs": ["src/modules", "src/utils", "lib/core"],
  "existing_docs": {
    "file_list": [".workflow/docs/project/src/core/API.md"],
    "content": "... concatenated existing docs ..."
  },
  "unified_analysis": [
    {"module_path": "./src/core", "outline_summary": "Core functionality"}
  ],
  "groups": {
    "count": 4,
    "assignments": [
      {"group_id": "001", "directories": ["src/modules", "src/utils"], "doc_count": 6},
      {"group_id": "002", "directories": ["lib/core", "lib/helpers"], "doc_count": 7}
    ]
  },
  "statistics": {
    "total": 15,
    "code": 8,
    "navigation": 7,
    "top_level": 3
  }
}
```

**Workflow Session Structure** (workflow-session.json):
```json
{
  "session_id": "WFS-docs-{timestamp}",
  "project": "{project_name} documentation",
  "status": "planning",
  "timestamp": "2024-01-20T14:30:22+08:00",
  "path": ".",
  "target_path": "/path/to/project",
  "project_root": "/path/to/project",
  "project_name": "{project_name}",
  "mode": "full",
  "tool": "gemini",
  "cli_execute": false,
  "update_mode": "update",
  "existing_docs": 5,
  "analysis": {
    "total": "15",
    "code": "8",
    "navigation": "7",
    "top_level": "3"
  }
}
```

## Generated Documentation

**Structure mirrors project source directories under project-specific folder**:

```
.workflow/docs/
└── {project_name}/                    # Project-specific root
    ├── src/                           # Mirrors src/ directory
    │   ├── modules/
    │   │   ├── README.md              # Navigation
    │   │   ├── auth/
    │   │   │   ├── API.md             # API signatures
    │   │   │   ├── README.md          # Module docs
    │   │   │   └── middleware/
    │   │   │       ├── API.md
    │   │   │       └── README.md
    │   │   └── api/
    │   │       ├── API.md
    │   │       └── README.md
    │   └── utils/
    │       └── README.md
    ├── lib/                           # Mirrors lib/ directory
    │   └── core/
    │       ├── API.md
    │       └── README.md
    ├── README.md                      # Project root
    ├── ARCHITECTURE.md                # System design
    ├── EXAMPLES.md                    # Usage examples
    └── api/                           # Optional
        └── README.md                  # HTTP API reference
```

## Execution Commands

```bash
# Execute entire workflow (auto-discovers active session)
/workflow:execute

# Or specify session
/workflow:execute --resume-session="WFS-docs-yyyymmdd-hhmmss"

# Individual task execution
/task:execute IMPL-001
```

## Template Reference

**Available Templates** (`~/.claude/workflows/cli-templates/prompts/documentation/`):
- `api.txt`: Code API (Part A) + HTTP API (Part B)
- `module-readme.txt`: Module purpose, usage, dependencies
- `folder-navigation.txt`: Navigation README for folders with subdirectories
- `project-readme.txt`: Project overview, getting started, navigation
- `project-architecture.txt`: System structure, module map, design patterns
- `project-examples.txt`: End-to-end usage examples

## Execution Mode Summary

| Mode | CLI Placement | CLI MODE | Approval Flag | Agent Role |
|------|---------------|----------|---------------|------------|
| **Agent (default)** | pre_analysis | analysis | (none) | Generates documentation content |
| **CLI (--cli-execute)** | implementation_approach | write | --approval-mode yolo | Executes CLI commands, validates output |

**Execution Flow**:
- **Phase 2**: Unified analysis once, results in `.process/`
- **Phase 4**: Dynamic grouping (max 2 dirs per group)
- **Level 1**: Parallel processing for module tree groups
- **Level 2+**: Sequential execution for project-level docs

## Related Commands
- `/workflow:execute` - Execute documentation tasks
- `/workflow:status` - View task progress
- `/workflow:session:complete` - Mark session complete
