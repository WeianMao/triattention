---
name: workflow:ui-design:reference-page-generator
description: Generate multi-component reference pages and documentation from design run extraction
argument-hint: "[--design-run <path>] [--package-name <name>] [--output-dir <path>]"
allowed-tools: Read,Write,Bash,Task,TodoWrite
auto-continue: true
---

# UI Design: Reference Page Generator

## Overview

Converts design run extraction results into shareable reference package with:
- Interactive multi-component preview (preview.html + preview.css)
- Component layout templates (layout-templates.json)

**Role**: Takes existing design run (from `import-from-code` or other extraction commands) and generates preview pages for easy reference.

## Usage

### Command Syntax

```bash
/workflow:ui-design:reference-page-generator [FLAGS]

# Flags
--design-run <path>      Design run directory path (required)
--package-name <name>    Package name for reference (required)
--output-dir <path>      Output directory (default: .workflow/reference_style)
```

---

## Execution Process

```
Input Parsing:
   ├─ Parse flags: --design-run, --package-name, --output-dir
   └─ Validation:
      ├─ --design-run and --package-name REQUIRED
      └─ Package name format: lowercase, alphanumeric, hyphens only

Phase 0: Setup & Validation
   ├─ Step 1: Validate required parameters
   ├─ Step 2: Validate package name format
   ├─ Step 3: Validate design run exists
   ├─ Step 4: Check required extraction files (design-tokens.json, layout-templates.json)
   └─ Step 5: Setup output directory

Phase 1: Prepare Component Data
   ├─ Step 1: Copy layout templates
   ├─ Step 2: Copy design tokens
   └─ Step 3: Copy animation tokens (optional)

Phase 2: Preview Generation (Agent)
   └─ Generate preview.html + preview.css via ui-design-agent
```

### Phase 0: Setup & Validation

**Purpose**: Validate inputs, prepare output directory

**Operations**:

```bash
# 1. Validate required parameters
if [ -z "$design_run" ] || [ -z "$package_name" ]; then
  echo "ERROR: --design-run and --package-name are required"
  echo "USAGE: /workflow:ui-design:reference-page-generator --design-run <path> --package-name <name>"
  exit 1
fi

# 2. Validate package name format (lowercase, alphanumeric, hyphens only)
if ! [[ "$package_name" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "ERROR: Invalid package name. Use lowercase, alphanumeric, and hyphens only."
  echo "EXAMPLE: main-app-style-v1"
  exit 1
fi

# 3. Validate design run exists
if [ ! -d "$design_run" ]; then
  echo "ERROR: Design run not found: $design_run"
  echo "HINT: Run '/workflow:ui-design:import-from-code' first to create design run"
  exit 1
fi

# 4. Check required extraction files exist
required_files=(
  "$design_run/style-extraction/style-1/design-tokens.json"
  "$design_run/layout-extraction/layout-templates.json"
)

for file in "${required_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "ERROR: Required file not found: $file"
    echo "HINT: Ensure design run has style and layout extraction results"
    exit 1
  fi
done

# 5. Setup output directory and validate
output_dir="${output_dir:-.workflow/reference_style}"
package_dir="${output_dir}/${package_name}"

# Check if package directory exists and is not empty
if [ -d "$package_dir" ] && [ "$(ls -A $package_dir 2>/dev/null)" ]; then
  # Directory exists - check if it's a valid package or just a directory
  if [ -f "$package_dir/metadata.json" ]; then
    # Valid package - safe to overwrite
    existing_version=$(jq -r '.version // "unknown"' "$package_dir/metadata.json" 2>/dev/null || echo "unknown")
    echo "INFO: Overwriting existing package '$package_name' (version: $existing_version)"
  else
    # Directory exists but not a valid package
    echo "ERROR: Directory '$package_dir' exists but is not a valid package"
    echo "Use a different package name or remove the directory manually"
    exit 1
  fi
fi

mkdir -p "$package_dir"

echo "[Phase 0] Setup Complete"
echo "  Design Run: $design_run"
echo "  Package: $package_name"
echo "  Output: $package_dir"
```

<!-- TodoWrite: Initialize todo list -->

**TodoWrite**:
```json
[
  {"content": "Phase 0: 验证和准备", "status": "completed", "activeForm": "验证参数"},
  {"content": "Phase 1: 准备组件数据", "status": "in_progress", "activeForm": "复制布局模板"},
  {"content": "Phase 2: 生成预览页面", "status": "pending", "activeForm": "生成预览"}
]
```

---

### Phase 1: Prepare Component Data

**Purpose**: Copy required files from design run to package directory

**Operations**:

```bash
echo "[Phase 1] Preparing component data from design run"

# 1. Copy layout templates as component patterns
cp "${design_run}/layout-extraction/layout-templates.json" "${package_dir}/layout-templates.json"

if [ ! -f "${package_dir}/layout-templates.json" ]; then
  echo "ERROR: Failed to copy layout templates"
  exit 1
fi

# Count components from layout templates
component_count=$(jq -r '.layout_templates | length // 0' "${package_dir}/layout-templates.json" 2>/dev/null || echo 0)
echo "  ✓ Layout templates copied (${component_count} components)"

# 2. Copy design tokens (required for preview generation)
cp "${design_run}/style-extraction/style-1/design-tokens.json" "${package_dir}/design-tokens.json"

if [ ! -f "${package_dir}/design-tokens.json" ]; then
  echo "ERROR: Failed to copy design tokens"
  exit 1
fi
echo "  ✓ Design tokens copied"

# 3. Copy animation tokens if exists (optional)
if [ -f "${design_run}/animation-extraction/animation-tokens.json" ]; then
  cp "${design_run}/animation-extraction/animation-tokens.json" "${package_dir}/animation-tokens.json"
  echo "  ✓ Animation tokens copied"
else
  echo "  ○ Animation tokens not found (optional)"
fi

echo "[Phase 1] Component data preparation complete"
```

<!-- TodoWrite: Mark Phase 1 complete, start Phase 2 -->

**TodoWrite**:
```json
[
  {"content": "Phase 1: 准备组件数据", "status": "completed", "activeForm": "复制布局模板"},
  {"content": "Phase 2: 生成预览页面", "status": "in_progress", "activeForm": "生成预览"}
]
```

---

### Phase 2: Preview Generation (Final Phase)

**Purpose**: Generate interactive multi-component preview (preview.html + preview.css)

**Agent Task**:

```javascript
Task(ui-design-agent): `
  [PREVIEW_SHOWCASE_GENERATION]
  Generate interactive multi-component showcase panel for reference package

  PACKAGE_DIR: ${package_dir} | PACKAGE_NAME: ${package_name}

  ## Input Files (MUST READ ALL)

  1. ${package_dir}/layout-templates.json (component layout patterns - REQUIRED)
  2. ${package_dir}/design-tokens.json (design tokens - REQUIRED)
  3. ${package_dir}/animation-tokens.json (optional, if exists)

  ## Generation Task

  Create interactive showcase with these sections:

  ### Section 1: Colors
  - Display all color categories as color swatches
  - Show hex/rgb values
  - Group by: brand, semantic, surface, text, border

  ### Section 2: Typography
  - Display typography scale (font sizes, weights)
  - Show typography combinations if available
  - Include font family examples
  - **Display usage recommendations** (from design-tokens.json _metadata.usage_recommendations.typography):
    * Common sizes table (small_text, body_text, heading)
    * Common combinations with use cases

  ### Section 3: Components
  - Render all components from layout-templates.json (use layout_templates field)
  - **Universal Components**: Display reusable multi-component showcases (buttons, inputs, cards, etc.)
    * **Display usage_guide** (from layout-templates.json):
      - Common sizes table with dimensions and use cases
      - Variant recommendations (when to use primary/secondary/etc)
      - Usage context list (typical scenarios)
      - Accessibility tips checklist
  - **Specialized Components**: Display module-specific components from code (feature-specific layouts, custom widgets)
  - Display all variants side-by-side
  - Show DOM structure with proper styling
  - Include usage code snippets in <details> tags
  - Clearly label component types (universal vs specialized)

  ### Section 4: Spacing & Layout
  - Visual spacing scale
  - Border radius examples
  - Shadow depth examples
  - **Display spacing recommendations** (from design-tokens.json _metadata.usage_recommendations.spacing):
    * Size guide table (tight/normal/loose categories)
    * Common patterns with use cases and pixel values

  ### Section 5: Animations (if available)
  - Animation duration examples
  - Easing function demonstrations

  ## Output Requirements

  Generate 2 files:
  1. ${package_dir}/preview.html
  2. ${package_dir}/preview.css

  ### preview.html Structure:
  - Complete standalone HTML file
  - Responsive design with mobile-first approach
  - Sticky navigation for sections
  - Interactive component demonstrations
  - Code snippets in collapsible <details> elements
  - Footer with package metadata

  ### preview.css Structure:
  - CSS Custom Properties from design-tokens.json
  - Typography combination classes
  - Component classes from layout-templates.json
  - Preview page layout styles
  - Interactive demo styles

  ## Critical Requirements
  - ✅ Read ALL input files (layout-templates.json, design-tokens.json, animation-tokens.json if exists)
  - ✅ Generate complete, interactive showcase HTML
  - ✅ All CSS uses var() references to design tokens
  - ✅ Display ALL components from layout-templates.json
  - ✅ **Separate universal components from specialized components** in the showcase
  - ✅ Display component DOM structures with proper styling
  - ✅ Include usage code snippets
  - ✅ Label each component type clearly (Universal / Specialized)
  - ✅ **Display usage recommendations** when available:
    - Typography: common_sizes, common_combinations (from _metadata.usage_recommendations)
    - Components: usage_guide for universal components (from layout-templates)
    - Spacing: size_guide, common_patterns (from _metadata.usage_recommendations)
  - ✅ Gracefully handle missing usage data (display sections only if data exists)
  - ✅ Use Write() to save both files:
    - ${package_dir}/preview.html
    - ${package_dir}/preview.css
  - ❌ NO external research or MCP calls
`
```

<!-- TodoWrite: Mark all complete -->

**TodoWrite**:
```json
[
  {"content": "Phase 0: 验证和准备", "status": "completed", "activeForm": "验证参数"},
  {"content": "Phase 1: 准备组件数据", "status": "completed", "activeForm": "复制布局模板"},
  {"content": "Phase 2: 生成预览页面", "status": "completed", "activeForm": "生成预览"}
]
```

---

## Output Structure

```
${output_dir}/
└── ${package_name}/
    ├── layout-templates.json    # Layout templates (copied from design run)
    ├── design-tokens.json       # Design tokens (copied from design run)
    ├── animation-tokens.json    # Animation tokens (copied from design run, optional)
    ├── preview.html             # Interactive showcase (NEW)
    └── preview.css              # Showcase styling (NEW)
```

## Completion Message

```
✅ Preview package generated!

Package: {package_name}
Location: {package_dir}

Files:
✓ layout-templates.json    {component_count} components
✓ design-tokens.json       Design tokens
✓ animation-tokens.json    Animation tokens {if exists: "✓" else: "○ (not found)"}
✓ preview.html             Interactive showcase
✓ preview.css              Showcase styling

Open preview:
  file://{absolute_path_to_package_dir}/preview.html
```

## Error Handling

### Common Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| Missing --design-run or --package-name | Required parameters not provided | Provide both flags |
| Invalid package name | Contains uppercase, special chars | Use lowercase, alphanumeric, hyphens only |
| Design run not found | Incorrect path or design run doesn't exist | Verify design run path, run import-from-code first |
| Missing extraction files | Design run incomplete | Ensure design run has style-extraction and layout-extraction results |
| Layout templates copy failed | layout-templates.json not found | Run import-from-code with Layout Agent first |
| Preview generation failed | Invalid design tokens | Check design-tokens.json format |

---

