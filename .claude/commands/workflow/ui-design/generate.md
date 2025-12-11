---
name: generate
description: Assemble UI prototypes by combining layout templates with design tokens (default animation support), pure assembler without new content generation
argument-hint: [--design-id <id>] [--session <id>]
allowed-tools: TodoWrite(*), Read(*), Write(*), Task(ui-design-agent), Bash(*)
---

# Generate UI Prototypes (/workflow:ui-design:generate)

## Overview
Pure assembler that combines pre-extracted layout templates with design tokens to generate UI prototypes (`style × layout × targets`). No layout design logic - purely combines existing components.

**Strategy**: Pure Assembly
- **Input**: `layout-*.json` files + `design-tokens.json` (+ reference images if available)
- **Process**: Combine structure (DOM) with style (tokens)
- **Output**: Complete HTML/CSS prototypes
- **No Design Logic**: All layout and style decisions already made
- **Automatic Image Reference**: If source images exist in layout templates, they're automatically used for visual context

**Prerequisite Commands**:
- `/workflow:ui-design:style-extract` → Complete design systems (design-tokens.json + style-guide.md)
- `/workflow:ui-design:layout-extract` → Layout structure

## Execution Process

```
Input Parsing:
   ├─ Parse flags: --design-id, --session
   └─ Decision (base path resolution):
      ├─ --design-id provided → Exact match by design ID
      ├─ --session provided → Latest in session
      └─ No flags → Latest globally

Phase 1: Setup & Validation
   ├─ Step 1: Resolve base path & parse configuration
   ├─ Step 2: Load layout templates
   ├─ Step 3: Validate design tokens
   └─ Step 4: Load animation tokens (optional)

Phase 2: Assembly (Agent)
   ├─ Step 1: Calculate agent grouping plan
   │  └─ Grouping rules:
   │     ├─ Style isolation: Each agent processes ONE style
   │     ├─ Balanced distribution: Layouts evenly split
   │     └─ Max 10 layouts per agent, max 6 concurrent agents
   ├─ Step 2: Launch batched assembly tasks (parallel)
   └─ Step 3: Verify generated files

Phase 3: Generate Preview Files
   ├─ Step 1: Run preview generation script
   └─ Step 2: Verify preview files
```

## Phase 1: Setup & Validation

### Step 1: Resolve Base Path & Parse Configuration
```bash
# Determine base path with priority: --design-id > --session > auto-detect
if [ -n "$DESIGN_ID" ]; then
  # Exact match by design ID
  relative_path=$(find .workflow -name "${DESIGN_ID}" -type d -print -quit)
elif [ -n "$SESSION_ID" ]; then
  # Latest in session
  relative_path=$(find .workflow/active/WFS-$SESSION_ID -name "design-run-*" -type d -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2)
else
  # Latest globally
  relative_path=$(find .workflow -name "design-run-*" -type d -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2)
fi

# Validate and convert to absolute path
if [ -z "$relative_path" ] || [ ! -d "$relative_path" ]; then
  echo "❌ ERROR: Design run not found"
  echo "💡 HINT: Run '/workflow:ui-design:list' to see available design runs"
  exit 1
fi

base_path=$(cd "$relative_path" && pwd)
bash(echo "✓ Base path: $base_path")

# Get style count
bash(ls "$base_path"/style-extraction/style-* -d | wc -l)

# Image reference auto-detected from layout template source_image_path
```

### Step 2: Load Layout Templates
```bash
# Check layout templates exist (multi-file pattern)
bash(find {base_path}/layout-extraction -name "layout-*.json" -print -quit | grep -q . && echo "exists")

# Get list of all layout files
bash(ls {base_path}/layout-extraction/layout-*.json 2>/dev/null)

# Load each layout template file
FOR each layout_file in layout_files:
    template_data = Read(layout_file)
    # Extract: target, variant_id, device_type, dom_structure, css_layout_rules

# Aggregate: targets[], layout_variants count, device_type, all template structures
```

**Output**: `base_path`, `style_variants`, `layout_templates[]`, `targets[]`, `device_type`

### Step 3: Validate Design Tokens
```bash
# Check design tokens exist for all styles
bash(test -f {base_path}/style-extraction/style-1/design-tokens.json && echo "valid")

# For each style variant: Load design tokens
Read({base_path}/style-extraction/style-{id}/design-tokens.json)
```

**Output**: `design_tokens[]` for all style variants

### Step 4: Load Animation Tokens (Optional)
```bash
# Check if animation tokens exist
bash(test -f {base_path}/animation-extraction/animation-tokens.json && echo "exists")

# Load animation tokens if available
IF exists({base_path}/animation-extraction/animation-tokens.json):
    animation_tokens = Read({base_path}/animation-extraction/animation-tokens.json)
    has_animations = true
ELSE:
    has_animations = false
```

**Output**: `animation_tokens` (optional), `has_animations` flag

## Phase 2: Assembly (Agent)

**Executor**: `Task(ui-design-agent)` grouped by `target × style` (max 10 layouts per agent, max 6 concurrent agents)

**⚠️ Core Principle**: **Each agent processes ONLY ONE style** (but can process multiple layouts for that style)

### Agent Grouping Strategy

**Grouping Rules**:
1. **Style Isolation**: Each agent processes ONLY ONE style (never mixed)
2. **Balanced Distribution**: Layouts evenly split (e.g., 12→6+6, not 10+2)
3. **Target Separation**: Different targets use different agents

**Distribution Formula**:
```
agents_needed = ceil(layout_count / MAX_LAYOUTS_PER_AGENT)
base_count = floor(layout_count / agents_needed)
remainder = layout_count % agents_needed
# First 'remainder' agents get (base_count + 1), others get base_count
```

**Examples** (MAX=10):

| Scenario | Result | Explanation |
|----------|--------|-------------|
| 3 styles × 3 layouts | 3 agents | Each style: 1 agent (3 layouts) |
| 3 styles × 12 layouts | 6 agents | Each style: 2 agents (6+6 layouts) |
| 2 styles × 5 layouts × 2 targets | 4 agents | Each (target, style): 1 agent (5 layouts) |

### Step 1: Calculate Agent Grouping Plan
```bash
bash(mkdir -p {base_path}/prototypes)

MAX_LAYOUTS_PER_AGENT = 10
MAX_PARALLEL = 6

agent_groups = []
FOR each target in targets:
  FOR each style_id in [1..S]:
    layouts_for_this_target_style = filter layouts by current target
    layout_count = len(layouts_for_this_target_style)

    # Balanced distribution (e.g., 12 layouts → 6+6)
    agents_needed = ceil(layout_count / MAX_LAYOUTS_PER_AGENT)
    base_count = floor(layout_count / agents_needed)
    remainder = layout_count % agents_needed

    layout_chunks = []
    start_idx = 0
    FOR i in range(agents_needed):
      chunk_size = base_count + 1 if i < remainder else base_count
      layout_chunks.append(layouts[start_idx : start_idx + chunk_size])
      start_idx += chunk_size

    FOR each chunk in layout_chunks:
      agent_groups.append({
        target: target,           # Single target
        style_id: style_id,       # Single style
        layout_ids: chunk         # Balanced layouts (≤10)
      })

total_agents = len(agent_groups)
total_batches = ceil(total_agents / MAX_PARALLEL)

TodoWrite({todos: [
  {content: "Setup and validation", status: "completed", activeForm: "Loading design systems"},
  {content: "Batch 1/{total_batches}: Assemble up to 6 agent groups", status: "in_progress", activeForm: "Assembling batch 1"},
  {content: "Batch 2/{total_batches}: Assemble up to 6 agent groups", status: "pending", activeForm: "Assembling batch 2"},
  ... (continue for all batches)
]})
```

### Step 2: Launch Batched Assembly Tasks

For each batch (up to 6 parallel agents per batch):
For each agent group `{target, style_id, layout_ids[]}` in current batch:
```javascript
Task(ui-design-agent): `
  [LAYOUT_STYLE_ASSEMBLY]
  🎯 {target} × Style-{style_id} × Layouts-{layout_ids}
  ⚠️ CONSTRAINT: Use ONLY style-{style_id}/design-tokens.json (never mix styles)

  TARGET: {target} | STYLE: {style_id} | LAYOUTS: {layout_ids} (max 10)
  BASE_PATH: {base_path}

  ## Inputs (READ ONLY - NO DESIGN DECISIONS)
  1. Layout Templates (LOOP THROUGH):
     FOR each layout_id in layout_ids:
       Read("{base_path}/layout-extraction/layout-{target}-{layout_id}.json")
       This file contains the specific layout template for this target and variant.
       Extract: dom_structure, css_layout_rules, device_type, source_image_path (from template field)

  2. Design Tokens (SHARED - READ ONCE):
     Read("{base_path}/style-extraction/style-{style_id}/design-tokens.json")
     Extract: ALL token values including:
       * colors, typography (with combinations), spacing, opacity
       * border_radius, shadows, breakpoints
       * component_styles (button, card, input variants)
     Note: typography.combinations, opacity, and component_styles fields contain preset configurations using var() references

  3. Animation Tokens (OPTIONAL):
     IF exists("{base_path}/animation-extraction/animation-tokens.json"):
       Read("{base_path}/animation-extraction/animation-tokens.json")
       Extract: duration, easing, transitions, keyframes, interactions
       has_animations = true
     ELSE:
       has_animations = false

  4. Reference Image (AUTO-DETECTED):
     IF template.source_image_path exists:
       Read(template.source_image_path)
       Purpose: Additional visual context for better placeholder content generation
       Note: This is for reference only - layout and style decisions already made
     ELSE:
       Use generic placeholder content

  ## Assembly Process (LOOP FOR EACH LAYOUT)
  FOR each layout_id in layout_ids:

    1. Build HTML: {base_path}/prototypes/{target}-style-{style_id}-layout-{layout_id}.html
       - Recursively build from template.dom_structure
       - Add: <!DOCTYPE html>, <head>, <meta viewport>
       - CSS link: <link href="{target}-style-{style_id}-layout-{layout_id}.css">
       - Inject placeholder content:
         * Default: Use Lorem ipsum, generic sample data
         * If reference image available: Generate more contextually appropriate placeholders
           (e.g., realistic headings, meaningful text snippets that match the visual context)
       - Preserve all attributes from dom_structure

    2. Build CSS: {base_path}/prototypes/{target}-style-{style_id}-layout-{layout_id}.css
       - Start with template.css_layout_rules
       - Replace ALL var(--*) with actual token values from design-tokens.json
         Example: var(--spacing-4) → 1rem (from tokens.spacing.4)
         Example: var(--breakpoint-md) → 768px (from tokens.breakpoints.md)
         Example: var(--opacity-80) → 0.8 (from tokens.opacity.80)
       - Add visual styling using design tokens:
         * Colors: tokens.colors.*
         * Typography: tokens.typography.* (including combinations)
         * Opacity: tokens.opacity.*
         * Shadows: tokens.shadows.*
         * Border radius: tokens.border_radius.*
       - IF tokens.component_styles exists: Add component style classes
         * Generate classes for button variants (.btn-primary, .btn-secondary)
         * Generate classes for card variants (.card-default, .card-interactive)
         * Generate classes for input variants (.input-default, .input-focus, .input-error)
         * Use var() references that resolve to actual token values
       - IF tokens.typography.combinations exists: Add typography preset classes
         * Generate classes for typography presets (.text-heading-primary, .text-body-regular, .text-caption)
         * Use var() references for family, size, weight, line-height, letter-spacing
       - IF has_animations == true: Inject animation tokens (ONCE, shared across layouts)
         * Add CSS Custom Properties for animations at :root level:
           --duration-instant, --duration-fast, --duration-normal, etc.
           --easing-linear, --easing-ease-out, etc.
         * Add @keyframes rules from animation_tokens.keyframes
         * Add interaction classes (.button-hover, .card-hover) from animation_tokens.interactions
         * Add utility classes (.transition-color, .transition-transform) from animation_tokens.transitions
         * Include prefers-reduced-motion media query for accessibility
       - Device-optimized for template.device_type

    3. Write files IMMEDIATELY after each layout completes

  ## Assembly Rules
  - ✅ Pure assembly: Combine pre-extracted structure + tokens
  - ❌ NO design decisions (layout/style pre-defined)
  - ✅ Read tokens ONCE, apply to all layouts in this batch
  - ✅ Replace var() with actual values
  - ✅ CSS filename MUST match HTML <link href="...">

  ## Output
  - Files: {len(layout_ids) × 2} (HTML + CSS pairs)
  - Each layout generates 2 files independently
`

# After each batch completes
TodoWrite: Mark current batch completed, next batch in_progress
```

### Step 3: Verify Generated Files
```bash
# Count expected vs found (should equal S × L × T)
bash(ls {base_path}/prototypes/{target}-style-*-layout-*.html | wc -l)

# Validate samples
Read({base_path}/prototypes/{target}-style-{style_id}-layout-{layout_id}.html)
# Check: <!DOCTYPE html>, correct CSS href, sufficient CSS length
```

**Output**: `total_files = S × L × T × 2` files verified (HTML + CSS pairs)

## Phase 3: Generate Preview Files

### Step 1: Run Preview Generation Script
```bash
bash(ccw tool exec ui_generate_preview '{"prototypesDir":"{base_path}/prototypes"}')
```

**Script generates**:
- `compare.html` (interactive matrix)
- `index.html` (navigation)
- `PREVIEW.md` (instructions)

### Step 2: Verify Preview Files
```bash
bash(ls {base_path}/prototypes/compare.html {base_path}/prototypes/index.html {base_path}/prototypes/PREVIEW.md)
```

**Output**: 3 preview files

## Completion

### Todo Update
```javascript
TodoWrite({todos: [
  {content: "Setup and validation", status: "completed", activeForm: "Loading design systems"},
  {content: "Batch 1/{total_batches}: Assemble 6 tasks", status: "completed", activeForm: "Assembling batch 1"},
  {content: "Batch 2/{total_batches}: Assemble 6 tasks", status: "completed", activeForm: "Assembling batch 2"},
  ... (all batches completed)
  {content: "Verify files & generate previews", status: "completed", activeForm: "Creating previews"}
]});
```

### Output Message
```
✅ UI prototype assembly complete!

Configuration:
- Style Variants: {style_variants}
- Layout Variants: {layout_variants} (from layout-*.json files)
- Device Type: {device_type}
- Targets: {targets}
- Total Prototypes: {S × L × T}
- Image Reference: Auto-detected (uses source images when available in layout templates)
- Animation Support: {has_animations ? 'Enabled (animation-tokens.json loaded)' : 'Not available'}

Assembly Process:
- Pure assembly: Combined pre-extracted layouts + design tokens
- Agent grouping: target × style (max 10 layouts per agent)
- Balanced distribution: Layouts evenly split (e.g., 12 → 6+6, not 10+2)

Batch Execution:
- Total agents: {total_agents} (each processes ONE style only)
- Batches: {total_batches} (max 6 agents parallel)
- Token efficiency: Read once per agent, apply to all layouts

Quality:
- Structure: From layout-extract (DOM, CSS layout rules)
- Style: From style-extract (design tokens)
- CSS: Token values directly applied (var() replaced)
- Device-optimized: Layouts match device_type from templates
- Animations: {has_animations ? 'CSS custom properties and @keyframes injected' : 'Static styles only'}

Generated Files:
{base_path}/prototypes/
├── {target}-style-{s}-layout-{l}.html ({S×L×T} prototypes)
├── {target}-style-{s}-layout-{l}.css
├── compare.html (interactive matrix)
├── index.html (navigation)
└── PREVIEW.md (instructions)

Input Files (from layout-extraction/):
├── layout-{target}-{variant}.json (multiple files, one per target-variant combination)

Preview:
1. Open compare.html (recommended)
2. Open index.html
3. Read PREVIEW.md

Next: /workflow:ui-design:update
```

## Simple Bash Commands

### Path Operations
```bash
# Find design directory
bash(find .workflow -type d -name "design-run-*" | head -1)

# Count style variants
bash(ls {base_path}/style-extraction/style-* -d | wc -l)
```

### Validation Commands
```bash
# Check layout templates exist (multi-file pattern)
bash(find {base_path}/layout-extraction -name "layout-*.json" -print -quit | grep -q . && echo "exists")

# Count layout files
bash(ls {base_path}/layout-extraction/layout-*.json 2>/dev/null | wc -l)

# Check design tokens exist
bash(test -f {base_path}/style-extraction/style-1/design-tokens.json && echo "valid")

# Count generated files
bash(ls {base_path}/prototypes/{target}-style-*-layout-*.html | wc -l)

# Verify preview
bash(test -f {base_path}/prototypes/compare.html && echo "exists")
```

### File Operations
```bash
# Create directories
bash(mkdir -p {base_path}/prototypes)

# Run preview script
bash(ccw tool exec ui_generate_preview '{"prototypesDir":"{base_path}/prototypes"}')
```

## Output Structure

```
{base_path}/
├── layout-extraction/
│   └── layout-{target}-{variant}.json  # Input (multiple files from layout-extract)
├── style-extraction/
│   └── style-{s}/
│       ├── design-tokens.json          # Input (from style-extract)
│       └── style-guide.md
└── prototypes/
    ├── {target}-style-{s}-layout-{l}.html  # Assembled prototypes
    ├── {target}-style-{s}-layout-{l}.css
    ├── compare.html
    ├── index.html
    └── PREVIEW.md
```

## Error Handling

### Common Errors
```
ERROR: Layout templates not found
→ Run /workflow:ui-design:layout-extract first

ERROR: Design tokens not found
→ Run /workflow:ui-design:style-extract first

ERROR: Agent assembly failed
→ Check inputs exist, validate JSON structure

ERROR: Script permission denied
→ Verify ccw tool is available: ccw tool list
```

### Recovery Strategies
- **Partial success**: Keep successful assembly combinations
- **Invalid template structure**: Validate layout-*.json files
- **Invalid tokens**: Validate design-tokens.json structure

## Quality Checklist

- [ ] CSS uses direct token values (var() replaced)
- [ ] HTML structure matches layout template exactly
- [ ] Semantic HTML5 structure preserved
- [ ] ARIA attributes from template present
- [ ] Device-specific optimizations applied
- [ ] All token references resolved
- [ ] compare.html works

## Key Features

- **Pure Assembly**: No design decisions, only combination
- **Token Resolution**: var() → actual values
- **Efficient Grouping**: target × style (max 10 layouts/agent, balanced split)
- **Style Isolation**: Each agent processes ONE style only
- **Production-Ready**: Semantic, accessible, token-driven

## Integration

**Prerequisites**:
- `/workflow:ui-design:style-extract` → `design-tokens.json` + `style-guide.md`
- `/workflow:ui-design:layout-extract` → `layout-{target}-{variant}.json` files

**Input**: `layout-*.json` files + `design-tokens.json`
**Output**: S×L×T prototypes for `/workflow:ui-design:update`
**Called by**: `/workflow:ui-design:explore-auto`, `/workflow:ui-design:imitate-auto`
