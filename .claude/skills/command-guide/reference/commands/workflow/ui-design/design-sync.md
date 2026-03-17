---
name: design-sync
description: Synchronize finalized design system references to brainstorming artifacts, preparing them for /workflow:plan consumption
argument-hint: --session <session_id> [--selected-prototypes "<list>"]
allowed-tools: Read(*), Write(*), Edit(*), TodoWrite(*), Glob(*), Bash(*)
---

# Design Sync Command

## Overview

Synchronize finalized design system references to brainstorming artifacts, preparing them for `/workflow:plan` consumption. This command updates **references only** (via @ notation), not content duplication.

## Core Philosophy

- **Reference-Only Updates**: Use @ references, no content duplication
- **Main Claude Execution**: Direct updates by main Claude (no Agent handoff)
- **Synthesis Alignment**: Update role analysis documents UI/UX Guidelines section
- **Plan-Ready Output**: Ensure design artifacts discoverable by task-generate
- **Minimal Reading**: Verify file existence, don't read design content

## Execution Process

```
Input Parsing:
   ├─ Parse flags: --session, --selected-prototypes
   └─ Validation: session_id REQUIRED

Phase 1: Session & Artifact Validation
   ├─ Step 1: Validate session exists
   ├─ Step 2: Find latest design run
   ├─ Step 3: Detect design system structure
   └─ Step 4: Select prototypes (--selected-prototypes OR all)

Phase 1.1: Memory Check (Conditional)
   └─ Decision (current design run in synthesis):
      ├─ Already updated → Skip Phase 2-5, EXIT
      └─ Not found → Continue to Phase 2

Phase 2: Load Target Artifacts
   ├─ Read role analysis documents (files to update)
   ├─ Read ui-designer/analysis.md (if exists)
   └─ Read prototype notes (minimal context)

Phase 3: Update Synthesis Specification
   └─ Edit role analysis documents with UI/UX Guidelines section

Phase 4A: Update Relevant Role Analysis Documents
   ├─ ui-designer/analysis.md (always)
   ├─ ux-expert/analysis.md (if animations exist)
   ├─ system-architect/analysis.md (if layouts exist)
   └─ product-manager/analysis.md (if prototypes)

Phase 4B: Create UI Designer Design System Reference
   └─ Write ui-designer/design-system-reference.md

Phase 5: Update Context Package
   └─ Update context-package.json with design system references

Phase 6: Completion
   └─ Report updated artifacts
```

## Execution Protocol

### Phase 1: Session & Artifact Validation

```bash
# Validate session
CHECK: find .workflow/active/ -name "WFS-*" -type d; VALIDATE: session_id matches active session

# Verify design artifacts in latest design run
latest_design = find_latest_path_matching(".workflow/active/WFS-{session}/design-run-*")

# Detect design system structure
IF exists({latest_design}/style-extraction/style-1/design-tokens.json):
    design_system_mode = "separate"; design_tokens_path = "style-extraction/style-1/design-tokens.json"; style_guide_path = "style-extraction/style-1/style-guide.md"
ELSE:
    ERROR: "No design tokens found. Run /workflow:ui-design:style-extract first"

VERIFY: {latest_design}/{design_tokens_path}, {latest_design}/{style_guide_path}, {latest_design}/prototypes/*.html

REPORT: "📋 Design system mode: {design_system_mode} | Tokens: {design_tokens_path}"

# Prototype selection
selected_list = --selected-prototypes ? parse_comma_separated(--selected-prototypes) : Glob({latest_design}/prototypes/*.html)
VALIDATE: Specified prototypes exist IF --selected-prototypes

REPORT: "Found {count} design artifacts, {prototype_count} prototypes"
```

### Phase 1.1: Memory Check (Skip if Already Updated)

```bash
# Check if role analysis documents contains current design run reference
synthesis_spec_path = ".workflow/active/WFS-{session}/.brainstorming/role analysis documents"
current_design_run = basename(latest_design)  # e.g., "design-run-20250109-143022"

IF exists(synthesis_spec_path):
    synthesis_content = Read(synthesis_spec_path)
    IF "## UI/UX Guidelines" in synthesis_content AND current_design_run in synthesis_content:
        REPORT: "✅ Design system references already updated (found in memory)"
        REPORT: "   Skipping: Phase 2-5 (Load Target Artifacts, Update Synthesis, Update UI Designer Guide, Completion)"
        EXIT 0
```

### Phase 2: Load Target Artifacts Only

**What to Load**: Only the files we need to **update**, not the design files we're referencing.

```bash
# Load target brainstorming artifacts (files to be updated)
Read(.workflow/active/WFS-{session}/.brainstorming/role analysis documents)
IF exists(.workflow/active/WFS-{session}/.brainstorming/ui-designer/analysis.md): Read(analysis.md)

# Optional: Read prototype notes for descriptions (minimal context)
FOR each selected_prototype IN selected_list:
    Read({latest_design}/prototypes/{selected_prototype}-notes.md)  # Extract: layout_strategy, page_name only

# Note: Do NOT read design-tokens.json, style-guide.md, or prototype HTML. Only verify existence and generate @ references.
```

### Phase 3: Update Synthesis Specification

Update `.brainstorming/role analysis documents` with design system references.

**Target Section**: `## UI/UX Guidelines`

**Content Template**:
```markdown
## UI/UX Guidelines

### Design System Reference
**Finalized Design Tokens**: @../{design_id}/{design_tokens_path}
**Style Guide**: @../{design_id}/{style_guide_path}
**Design System Mode**: {design_system_mode}

### Implementation Requirements
**Token Adherence**: All UI implementations MUST use design token CSS custom properties
**Accessibility**: WCAG AA compliance validated in design-tokens.json
**Responsive**: Mobile-first design using token-based breakpoints
**Component Patterns**: Follow patterns documented in style-guide.md

### Reference Prototypes
{FOR each selected_prototype:
- **{page_name}**: @../{design_id}/prototypes/{prototype}.html | Layout: {layout_strategy from notes}
}

### Design System Assets
```json
{"design_tokens": "{design_id}/{design_tokens_path}", "style_guide": "{design_id}/{style_guide_path}", "design_system_mode": "{design_system_mode}", "prototypes": [{FOR each: "{design_id}/prototypes/{prototype}.html"}]}
```
```

**Implementation**:
```bash
# Option 1: Edit existing section
Edit(file_path=".workflow/active/WFS-{session}/.brainstorming/role analysis documents",
     old_string="## UI/UX Guidelines\n[existing content]",
     new_string="## UI/UX Guidelines\n\n[new design reference content]")

# Option 2: Append if section doesn't exist
IF section not found:
    Edit(file_path="...", old_string="[end of document]", new_string="\n\n## UI/UX Guidelines\n\n[new design reference content]")
```

### Phase 4A: Update Relevant Role Analysis Documents

**Discovery**: Find role analysis.md files affected by design outputs

```bash
# Always update ui-designer
ui_designer_files = Glob(".workflow/active/WFS-{session}/.brainstorming/ui-designer/analysis*.md")

# Conditionally update other roles
has_animations = exists({latest_design}/animation-extraction/animation-tokens.json)
has_layouts = exists({latest_design}/layout-extraction/layout-templates.json)

IF has_animations: ux_expert_files = Glob(".workflow/active/WFS-{session}/.brainstorming/ux-expert/analysis*.md")
IF has_layouts: architect_files = Glob(".workflow/active/WFS-{session}/.brainstorming/system-architect/analysis*.md")
IF selected_list: pm_files = Glob(".workflow/active/WFS-{session}/.brainstorming/product-manager/analysis*.md")
```

**Content Templates**:

**ui-designer/analysis.md** (append if not exists):
```markdown
## Design System Implementation Reference

**Design Tokens**: @../../{design_id}/{design_tokens_path}
**Style Guide**: @../../{design_id}/{style_guide_path}
**Prototypes**: {FOR each: @../../{design_id}/prototypes/{prototype}.html}

*Reference added by /workflow:ui-design:update*
```

**ux-expert/analysis.md** (if animations):
```markdown
## Animation & Interaction Reference

**Animations**: @../../{design_id}/animation-extraction/animation-tokens.json
**Prototypes**: {FOR each: @../../{design_id}/prototypes/{prototype}.html}

*Reference added by /workflow:ui-design:update*
```

**system-architect/analysis.md** (if layouts):
```markdown
## Layout Structure Reference

**Layout Templates**: @../../{design_id}/layout-extraction/layout-templates.json

*Reference added by /workflow:ui-design:update*
```

**product-manager/analysis.md** (if prototypes):
```markdown
## Prototype Validation Reference

**Prototypes**: {FOR each: @../../{design_id}/prototypes/{prototype}.html}

*Reference added by /workflow:ui-design:update*
```

**Implementation**:
```bash
FOR file IN [ui_designer_files, ux_expert_files, architect_files, pm_files]:
  IF file exists AND section_not_exists(file):
    Edit(file, old_string="[end of document]", new_string="\n\n{role-specific section}")
```

### Phase 4B: Create UI Designer Design System Reference

Create or update `.brainstorming/ui-designer/design-system-reference.md`:

```markdown
# UI Designer Design System Reference

## Design System Integration
This style guide references the finalized design system from the design refinement phase.

**Design Tokens**: @../../{design_id}/{design_tokens_path}
**Style Guide**: @../../{design_id}/{style_guide_path}
**Design System Mode**: {design_system_mode}

## Implementation Guidelines
1. **Use CSS Custom Properties**: All styles reference design tokens
2. **Follow Semantic HTML**: Use HTML5 semantic elements
3. **Maintain Accessibility**: WCAG AA compliance required
4. **Responsive Design**: Mobile-first with token-based breakpoints

## Reference Prototypes
{FOR each selected_prototype:
- **{page_name}**: @../../{design_id}/prototypes/{prototype}.html
}

## Token System
For complete token definitions and usage examples, see:
- Design Tokens: @../../{design_id}/{design_tokens_path}
- Style Guide: @../../{design_id}/{style_guide_path}

---
*Auto-generated by /workflow:ui-design:update | Last updated: {timestamp}*
```

**Implementation**:
```bash
Write(file_path=".workflow/active/WFS-{session}/.brainstorming/ui-designer/design-system-reference.md",
      content="[generated content with @ references]")
```

### Phase 5: Update Context Package

**Purpose**: Sync design system references to context-package.json

**Operations**:
```bash
context_pkg_path = ".workflow/active/WFS-{session}/.process/context-package.json"

# 1. Read existing package
context_pkg = Read(context_pkg_path)

# 2. Update brainstorm_artifacts (role analyses now contain @ design references)
brainstorm_dir = ".workflow/active/WFS-{session}/.brainstorming"
role_analysis_files = Glob({brainstorm_dir}/*/analysis*.md)

context_pkg.brainstorm_artifacts.role_analyses = []
FOR file IN role_analysis_files:
    role_name = extract_role_from_path(file)
    relative_path = file.replace({brainstorm_dir}/, "")

    context_pkg.brainstorm_artifacts.role_analyses.push({
        "role": role_name,
        "files": [{
            "path": relative_path,
            "type": "primary",
            "content": Read(file),  # Contains @ design system references
            "updated_at": NOW()
        }]
    })

# 3. Add design_system_references field
context_pkg.design_system_references = {
    "design_run_id": design_id,
    "tokens": `${design_id}/${design_tokens_path}`,
    "style_guide": `${design_id}/${style_guide_path}`,
    "prototypes": selected_list.map(p => `${design_id}/prototypes/${p}.html`),
    "updated_at": NOW()
}

# 4. Optional: Add animations and layouts if they exist
IF exists({latest_design}/animation-extraction/animation-tokens.json):
    context_pkg.design_system_references.animations = `${design_id}/animation-extraction/animation-tokens.json`

IF exists({latest_design}/layout-extraction/layout-templates.json):
    context_pkg.design_system_references.layouts = `${design_id}/layout-extraction/layout-templates.json`

# 5. Update metadata
context_pkg.metadata.updated_at = NOW()
context_pkg.metadata.design_sync_timestamp = NOW()

# 6. Write back
Write(context_pkg_path, JSON.stringify(context_pkg, indent=2))

REPORT: "✅ Updated context-package.json with design system references"
```

**TodoWrite Update**:
```json
{"content": "Update context package with design references", "status": "completed", "activeForm": "Updating context package"}
```

### Phase 6: Completion

```javascript
TodoWrite({todos: [
  {content: "Validate session and design system artifacts", status: "completed", activeForm: "Validating artifacts"},
  {content: "Load target brainstorming artifacts", status: "completed", activeForm: "Loading target files"},
  {content: "Update role analysis documents with design references", status: "completed", activeForm: "Updating synthesis spec"},
  {content: "Update relevant role analysis.md documents", status: "completed", activeForm: "Updating role analysis files"},
  {content: "Create/update ui-designer/design-system-reference.md", status: "completed", activeForm: "Creating design system reference"}
]});
```

**Completion Message**:
```
✅ Design system references updated for session: WFS-{session}

Updated artifacts:
✓ role analysis documents - UI/UX Guidelines section with @ references
✓ {role_count} role analysis.md files - Design system references
✓ ui-designer/design-system-reference.md - Design system reference guide

Design system assets ready for /workflow:plan:
- design-tokens.json | style-guide.md | {prototype_count} reference prototypes

Next: /workflow:plan [--agent] "<task description>"
      The plan phase will automatically discover and utilize the design system.
```

## Output Structure

**Updated Files**:
```
.workflow/active/WFS-{session}/.brainstorming/
├── role analysis documents              # Updated with UI/UX Guidelines section
├── ui-designer/
│   ├── analysis*.md                     # Updated with design system references
│   └── design-system-reference.md       # New or updated design reference guide
├── ux-expert/analysis*.md               # Updated if animations exist
├── product-manager/analysis*.md         # Updated if prototypes exist
└── system-architect/analysis*.md        # Updated if layouts exist
```

**@ Reference Format** (role analysis documents):
```
@../{design_id}/style-extraction/style-1/design-tokens.json
@../{design_id}/style-extraction/style-1/style-guide.md
@../{design_id}/prototypes/{prototype}.html
```

**@ Reference Format** (ui-designer/design-system-reference.md):
```
@../../{design_id}/style-extraction/style-1/design-tokens.json
@../../{design_id}/style-extraction/style-1/style-guide.md
@../../{design_id}/prototypes/{prototype}.html
```

**@ Reference Format** (role analysis.md files):
```
@../../{design_id}/style-extraction/style-1/design-tokens.json
@../../{design_id}/animation-extraction/animation-tokens.json
@../../{design_id}/layout-extraction/layout-templates.json
@../../{design_id}/prototypes/{prototype}.html
```

## Integration with /workflow:plan

After this update, `/workflow:plan` will discover design assets through:

**Phase 3: Intelligent Analysis** (`/workflow:tools:concept-enhanced`)
- Reads role analysis documents → Discovers @ references → Includes design system context in ANALYSIS_RESULTS.md

**Phase 4: Task Generation** (`/workflow:tools:task-generate`)
- Reads ANALYSIS_RESULTS.md → Discovers design assets → Includes design system paths in task JSON files

**Example Task JSON** (generated by task-generate):
```json
{
  "task_id": "IMPL-001",
  "context": {
    "design_system": {
      "tokens": "{design_id}/style-extraction/style-1/design-tokens.json",
      "style_guide": "{design_id}/style-extraction/style-1/style-guide.md",
      "prototypes": ["{design_id}/prototypes/dashboard-variant-1.html"]
    }
  }
}
```

## Error Handling

- **Missing design artifacts**: Error with message "Run /workflow:ui-design:style-extract and /workflow:ui-design:generate first"
- **role analysis documents not found**: Warning, create minimal version with just UI/UX Guidelines
- **ui-designer/ directory missing**: Create directory and file
- **Edit conflicts**: Preserve existing content, append or replace only UI/UX Guidelines section
- **Invalid prototype names**: Skip invalid entries, continue with valid ones

## Validation Checks

After update, verify:
- [ ] role analysis documents contains UI/UX Guidelines section
- [ ] UI/UX Guidelines include @ references (not content duplication)
- [ ] ui-designer/analysis*.md updated with design system references
- [ ] ui-designer/design-system-reference.md created or updated
- [ ] Relevant role analysis.md files updated (ux-expert, product-manager, system-architect)
- [ ] All @ referenced files exist and are accessible
- [ ] @ reference paths are relative and correct

## Key Features

1. **Reference-Only Updates**: Uses @ notation for file references, no content duplication, lightweight and maintainable
2. **Main Claude Direct Execution**: No Agent handoff (preserves context), simple reference generation, reliable path resolution
3. **Plan-Ready Output**: `/workflow:plan` Phase 3 can discover design system, task generation includes design asset paths, clear integration points
4. **Minimal Reading**: Only reads target files to update, verifies design file existence (no content reading), optional prototype notes for descriptions
5. **Flexible Prototype Selection**: Auto-select all prototypes (default), manual selection via --selected-prototypes parameter, validates existence

## Integration Points

- **Input**: Design system artifacts from `/workflow:ui-design:style-extract` and `/workflow:ui-design:generate`
- **Output**: Updated role analysis documents, role analysis.md files, ui-designer/design-system-reference.md with @ references
- **Next Phase**: `/workflow:plan` discovers and utilizes design system through @ references
- **Auto Integration**: Automatically triggered by `/workflow:ui-design:auto` workflow

