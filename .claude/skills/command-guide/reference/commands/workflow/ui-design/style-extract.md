---
name: style-extract
description: Extract design style from reference images or text prompts using Claude analysis with variant generation or refinement mode
argument-hint: "[--design-id <id>] [--session <id>] [--images "<glob>"] [--prompt "<desc>"] [--variants <count>] [--interactive] [--refine]"
allowed-tools: TodoWrite(*), Read(*), Write(*), Glob(*), AskUserQuestion(*)
---

# Style Extraction Command

## Overview
Extract design style from reference images or text prompts using Claude's built-in analysis. Supports two modes:
1. **Exploration Mode** (default): Generate multiple contrasting design variants
2. **Refinement Mode** (`--refine`): Refine a single existing design through detailed adjustments

**Strategy**: AI-Driven Design Space Exploration
- **Claude-Native**: 100% Claude analysis, no external tools
- **Direct Output**: Complete design systems (design-tokens.json)
- **Flexible Input**: Images, text prompts, or both (hybrid mode)
- **Dual Mode**: Exploration (multiple contrasting variants) or Refinement (single design fine-tuning)
- **Production-Ready**: WCAG AA compliant, OKLCH colors, semantic naming

## Execution Process

```
Input Parsing:
   ├─ Parse flags: --design-id, --session, --images, --prompt, --variants, --interactive, --refine
   └─ Decision (mode detection):
      ├─ --refine flag → Refinement Mode (variants_count = 1)
      └─ No --refine → Exploration Mode (variants_count = --variants OR 3)

Phase 0: Setup & Input Validation
   ├─ Step 1: Detect input mode, extraction mode & base path
   ├─ Step 2: Load inputs
   └─ Step 3: Memory check (skip if exists)

Phase 1: Design Direction/Refinement Options Generation
   ├─ Step 1: Load project context
   ├─ Step 2: Generate options (Agent Task 1)
   │  └─ Decision:
   │     ├─ Exploration Mode → Generate contrasting design directions
   │     └─ Refinement Mode → Generate refinement options
   └─ Step 3: Verify options file created

Phase 1.5: User Confirmation (Optional)
   └─ Decision (--interactive flag):
      ├─ --interactive present → Present options, capture selection
      └─ No --interactive → Skip to Phase 2

Phase 2: Design System Generation
   ├─ Step 1: Load user selection or default to all
   ├─ Step 2: Create output directories
   └─ Step 3: Launch agent tasks (parallel)

Phase 3: Verify Output
   ├─ Step 1: Check files created
   └─ Step 2: Verify file sizes
```

## Phase 0: Setup & Input Validation

### Step 1: Detect Input Mode, Extraction Mode & Base Path
```bash
# Detect input source
# Priority: --images + --prompt → hybrid | --images → image | --prompt → text

# Detect refinement mode
refine_mode = --refine OR false

# Set variants count
# Refinement mode: Force variants_count = 1 (ignore user-provided --variants)
# Exploration mode: Use --variants or default to 3 (range: 1-5)
IF refine_mode:
    variants_count = 1
    REPORT: "🔧 Refinement mode enabled: Will generate 1 refined design system"
ELSE:
    variants_count = --variants OR 3
    VALIDATE: 1 <= variants_count <= 5
    REPORT: "🔍 Exploration mode: Will generate {variants_count} contrasting design directions"

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
```

### Step 2: Load Inputs
```bash
# For image mode
bash(ls {images_pattern})  # Expand glob pattern
Read({image_path})  # Load each image

# For text mode
# Validate --prompt is non-empty

# Create output directory
bash(mkdir -p {base_path}/style-extraction/)
```

### Step 3: Memory Check
```bash
# 1. Check if inputs cached in session memory
IF session_has_inputs: SKIP Step 2 file reading

# 2. Check if output already exists
bash(test -f {base_path}/style-extraction/style-1/design-tokens.json && echo "exists")
IF exists: SKIP to completion
```

---

**Phase 0 Output**: `input_mode`, `base_path`, `extraction_mode`, `variants_count`, `loaded_images[]` or `prompt_guidance`

## Phase 1: Design Direction or Refinement Options Generation

### Step 1: Load Project Context
```bash
# Load brainstorming context if available
bash(test -f {base_path}/.brainstorming/role analysis documents && cat it)

# Load existing design system if refinement mode
IF refine_mode:
    existing_tokens = Read({base_path}/style-extraction/style-1/design-tokens.json)
```

### Step 2: Generate Options (Agent Task 1 - Mode-Specific)
**Executor**: `Task(ui-design-agent)`

**Exploration Mode** (default): Generate contrasting design directions
**Refinement Mode** (`--refine`): Generate refinement options for existing design

```javascript
// Conditional agent task based on refine_mode
IF NOT refine_mode:
    // EXPLORATION MODE
    Task(ui-design-agent): `
      [DESIGN_DIRECTION_GENERATION_TASK]
      Generate {variants_count} maximally contrasting design directions with visual previews

      SESSION: {session_id} | MODE: explore | BASE_PATH: {base_path}

      ## Input Analysis
      - User prompt: {prompt_guidance}
      - Visual references: {loaded_images if available}
      - Project context: {brainstorming_context if available}

      ## Analysis Rules
      - Analyze 6D attribute space: color saturation, visual weight, formality, organic/geometric, innovation, density
      - Generate {variants_count} directions with MAXIMUM contrast
      - Each direction must be distinctly different (min distance score: 0.7)

      ## Generate for EACH Direction
      1. **Core Philosophy**:
         - philosophy_name (2-3 words, e.g., "Minimalist & Airy")
         - design_attributes (6D scores 0-1)
         - search_keywords (3-5 keywords)
         - anti_keywords (2-3 keywords to avoid)
         - rationale (why this is distinct from others)

      2. **Visual Preview Elements**:
         - primary_color (OKLCH format)
         - secondary_color (OKLCH format)
         - accent_color (OKLCH format)
         - font_family_heading (specific font name)
         - font_family_body (specific font name)
         - border_radius_base (e.g., "0.5rem")
         - mood_description (1-2 sentences describing the feel)

      ## Output
      Write single JSON file: {base_path}/.intermediates/style-analysis/analysis-options.json

      Use schema from INTERACTIVE-DATA-SPEC.md (Style Extract: analysis-options.json)

      CRITICAL: Use Write() tool immediately after generating complete JSON
    `
ELSE:
    // REFINEMENT MODE
    Task(ui-design-agent): `
      [DESIGN_REFINEMENT_OPTIONS_TASK]
      Generate refinement options for existing design system

      SESSION: {session_id} | MODE: refine | BASE_PATH: {base_path}

      ## Existing Design System
      - design-tokens.json: {existing_tokens}

      ## Input Guidance
      - User prompt: {prompt_guidance}
      - Visual references: {loaded_images if available}

      ## Refinement Categories
      Generate 8-12 refinement options across these categories:

      1. **Intensity/Degree Adjustments** (2-3 options):
         - Color intensity: more vibrant ↔ more muted
         - Visual weight: bolder ↔ lighter
         - Density: more compact ↔ more spacious

      2. **Specific Attribute Tuning** (3-4 options):
         - Border radius: sharper corners ↔ rounder edges
         - Shadow depth: subtler shadows ↔ deeper elevation
         - Typography scale: tighter scale ↔ more contrast
         - Spacing scale: tighter rhythm ↔ more breathing room

      3. **Context-Specific Variations** (2-3 options):
         - Dark mode optimization
         - Mobile-specific adjustments
         - High-contrast accessibility mode

      4. **Component-Level Customization** (1-2 options):
         - Button styling emphasis
         - Card/container treatment
         - Form input refinements

      ## Output Format
      Each option:
      - category: "intensity|attribute|context|component"
      - option_id: unique identifier
      - label: Short descriptive name (e.g., "More Vibrant Colors")
      - description: What changes (2-3 sentences)
      - preview_changes: Key token adjustments preview
      - impact_scope: Which tokens affected

      ## Output
      Write single JSON file: {base_path}/.intermediates/style-analysis/analysis-options.json

      Use refinement schema:
      {
        "mode": "refinement",
        "base_design": "style-1",
        "refinement_options": [array of refinement options]
      }

      CRITICAL: Use Write() tool immediately after generating complete JSON
    `
```

### Step 3: Verify Options File Created
```bash
bash(test -f {base_path}/.intermediates/style-analysis/analysis-options.json && echo "created")

# Quick validation
bash(cat {base_path}/.intermediates/style-analysis/analysis-options.json | grep -q "design_directions" && echo "valid")
```

**Output**: `analysis-options.json` with design direction options

---

## Phase 1.5: User Confirmation (Optional - Triggered by --interactive)

**Purpose**:
- **Exploration Mode**: Allow user to select preferred design direction(s)
- **Refinement Mode**: Allow user to select refinement options to apply

**Trigger Condition**: Execute this phase ONLY if `--interactive` flag is present

### Step 1: Check Interactive Flag
```bash
# Skip this entire phase if --interactive flag is not present
IF NOT --interactive:
    SKIP to Phase 2
    IF refine_mode:
        REPORT: "ℹ️ Non-interactive refinement mode: Will apply all suggested refinements"
    ELSE:
        REPORT: "ℹ️ Non-interactive mode: Will generate all {variants_count} variants"

REPORT: "🎯 Interactive mode enabled: User selection required"
```

### Step 2: Load and Present Options (Mode-Specific)
```bash
# Read options file
options = Read({base_path}/.intermediates/style-analysis/analysis-options.json)

# Branch based on mode
IF NOT refine_mode:
    # EXPLORATION MODE
    design_directions = options.design_directions
ELSE:
    # REFINEMENT MODE
    refinement_options = options.refinement_options
```

### Step 3: Present Options to User (Mode-Specific)

**Exploration Mode**:
```
📋 Design Direction Options

We've generated {variants_count} contrasting design directions for your review.
Please select the direction(s) you'd like to develop into complete design systems.

{FOR each direction in design_directions:
  ═══════════════════════════════════════════════════
  Option {direction.index}: {direction.philosophy_name}
  ═══════════════════════════════════════════════════

  Philosophy: {direction.rationale}

  Visual Preview:
  • Colors: {direction.preview.primary_color} (primary), {direction.preview.accent_color} (accent)
  • Typography: {direction.preview.font_family_heading} (headings), {direction.preview.font_family_body} (body)
  • Border Radius: {direction.preview.border_radius_base}
  • Mood: {direction.preview.mood_description}

  Design Attributes:
  • Color Saturation: {direction.design_attributes.color_saturation * 100}%
  • Visual Weight: {direction.design_attributes.visual_weight * 100}%
  • Formality: {direction.design_attributes.formality * 100}%
  • Innovation: {direction.design_attributes.innovation * 100}%

  Keywords: {join(direction.search_keywords, ", ")}
  Avoiding: {join(direction.anti_keywords, ", ")}
}

═══════════════════════════════════════════════════
```

**Refinement Mode**:
```
📋 Design Refinement Options

We've analyzed your existing design system and generated refinement options.
Please select the refinement(s) you'd like to apply.

Base Design: style-1

{FOR each option in refinement_options grouped by category:
  ━━━ {category_name} ━━━

  {FOR each refinement in category_options:
    [{refinement.option_id}] {refinement.label}
    {refinement.description}
    Preview: {refinement.preview_changes}
    Affects: {refinement.impact_scope}
  }
}

═══════════════════════════════════════════════════
```

### Step 4: Capture User Selection and Update Analysis JSON

**Exploration Mode Interaction**:
```javascript
// Use AskUserQuestion tool for multi-selection
AskUserQuestion({
  questions: [{
    question: "Which design direction(s) would you like to develop into complete design systems?",
    header: "Style Choice",
    multiSelect: true,  // Multi-selection enabled
    options: [
      {FOR each direction:
        label: "Option {direction.index}: {direction.philosophy_name}",
        description: "{direction.mood_description}"
      }
    ]
  }]
})

// Parse user response (array of selections)
selected_options = user_answer

// Check for user cancellation
IF selected_options == null OR selected_options.length == 0:
    REPORT: "⚠️ User canceled selection. Workflow terminated."
    EXIT workflow

// Extract option indices
selected_indices = []
FOR each selected_option_text IN selected_options:
    match = selected_option_text.match(/Option (\d+):/)
    IF match:
        selected_indices.push(parseInt(match[1]))

REPORT: "✅ Selected {selected_indices.length} design direction(s)"

// Update analysis-options.json
options_file = Read({base_path}/.intermediates/style-analysis/analysis-options.json)
options_file.user_selection = {
  "selected_at": NOW(),
  "selected_indices": selected_indices,
  "selection_count": selected_indices.length
}
Write({base_path}/.intermediates/style-analysis/analysis-options.json, JSON.stringify(options_file, indent=2))
```

**Refinement Mode Interaction**:
```javascript
// Use AskUserQuestion tool for multi-selection of refinements
AskUserQuestion({
  questions: [{
    question: "Which refinement(s) would you like to apply to your design system?",
    header: "Refinements",
    multiSelect: true,  // Multi-selection enabled
    options: [
      {FOR each refinement:
        label: "{refinement.label}",
        description: "{refinement.description} (Affects: {refinement.impact_scope})"
      }
    ]
  }]
})

// Parse user response
selected_refinements = user_answer

// Check for user cancellation
IF selected_refinements == null OR selected_refinements.length == 0:
    REPORT: "⚠️ User canceled selection. Workflow terminated."
    EXIT workflow

// Extract refinement option_ids
selected_option_ids = []
FOR each selected_text IN selected_refinements:
    # Match against refinement labels to find option_ids
    FOR each refinement IN refinement_options:
        IF refinement.label IN selected_text:
            selected_option_ids.push(refinement.option_id)

REPORT: "✅ Selected {selected_option_ids.length} refinement(s)"

// Update analysis-options.json
options_file = Read({base_path}/.intermediates/style-analysis/analysis-options.json)
options_file.user_selection = {
  "selected_at": NOW(),
  "selected_option_ids": selected_option_ids,
  "selection_count": selected_option_ids.length
}
Write({base_path}/.intermediates/style-analysis/analysis-options.json, JSON.stringify(options_file, indent=2))
```

### Step 5: Confirmation Message (Mode-Specific)

**Exploration Mode**:
```
✅ Selection recorded!

You selected {selected_indices.length} design direction(s):
{FOR each index IN selected_indices:
  • Option {index} - {design_directions[index-1].philosophy_name}
}

Proceeding to generate {selected_indices.length} complete design system(s)...
```

**Refinement Mode**:
```
✅ Selection recorded!

You selected {selected_option_ids.length} refinement(s):
{FOR each id IN selected_option_ids:
  • {refinement_by_id[id].label} ({refinement_by_id[id].category})
}

Proceeding to apply refinements and generate updated design system...
```

**Output**: Updated `analysis-options.json` with user's selection embedded

## Phase 2: Design System Generation (Agent Task 2)

**Executor**: `Task(ui-design-agent)` for selected variant(s)

### Step 1: Load User Selection or Default to All
```bash
# Read analysis-options.json which may contain user_selection
options = Read({base_path}/.intermediates/style-analysis/analysis-options.json)

# Check if user_selection field exists (interactive mode)
IF options.user_selection AND options.user_selection.selected_indices:
    # Interactive mode: Use user-selected variants
    selected_indices = options.user_selection.selected_indices  # Array of selected indices (e.g., [1, 3])

    REPORT: "🎯 Interactive mode: Using {selected_indices.length} user-selected variant(s)"
ELSE:
    # Non-interactive mode: Generate ALL variants (default behavior)
    selected_indices = [1, 2, ..., variants_count]  # All indices from 1 to variants_count

    REPORT: "ℹ️ Non-interactive mode: Generating all {variants_count} variant(s)"

# Extract the selected direction details from options
selected_directions = [options.design_directions[i-1] for i in selected_indices]  # 0-indexed array

actual_variants_count = selected_indices.length
REPORT: "📦 Generating {actual_variants_count} design system(s)..."
```

### Step 2: Create Output Directories
```bash
# Create directories for each selected variant
FOR index IN 1..actual_variants_count:
    bash(mkdir -p {base_path}/style-extraction/style-{index})
```

### Step 3: Launch Agent Tasks (Parallel)
Generate design systems for ALL selected directions in parallel (max 5 concurrent):

```javascript
// Launch parallel tasks, one for each selected direction
FOR variant_index IN 1..actual_variants_count:
    selected_direction = selected_directions[variant_index - 1]  // 0-indexed

    Task(ui-design-agent): `
      [DESIGN_SYSTEM_GENERATION_TASK #{variant_index}/{actual_variants_count}]
      Generate production-ready design system based on user-selected direction

      SESSION: {session_id} | VARIANT: {variant_index}/{actual_variants_count} | BASE_PATH: {base_path}

      USER SELECTION:
      - Selected Direction: ${selected_direction.philosophy_name}
      - Design Attributes: ${JSON.stringify(selected_direction.design_attributes)}
      - Search Keywords: ${selected_direction.search_keywords.join(", ")}
      - Anti-keywords: ${selected_direction.anti_keywords.join(", ")}
      - Rationale: ${selected_direction.rationale}
      - Preview Colors: Primary=${selected_direction.preview.primary_color}, Accent=${selected_direction.preview.accent_color}
      - Preview Typography: Heading=${selected_direction.preview.font_family_heading}, Body=${selected_direction.preview.font_family_body}
      - Preview Border Radius: ${selected_direction.preview.border_radius_base}

      ## Input Analysis
      - Input mode: {input_mode} (image/text/hybrid)
      - Visual references: {loaded_images OR prompt_guidance}

      ## Generation Rules
      - Develop the selected design direction into a complete design system
      - Use preview elements as foundation and expand with full token coverage
      - Apply design_attributes to all token values:
        * color_saturation → OKLCH chroma values
        * visual_weight → font weights, shadow depths
        * density → spacing scale compression/expansion
        * formality → typography choices, border radius
        * organic_geometric → border radius, shape patterns
        * innovation → token naming, experimental values
      - Honor search_keywords for design inspiration
      - Avoid anti_keywords patterns
      - All colors in OKLCH format
      - WCAG AA compliance: 4.5:1 text contrast, 3:1 UI contrast

      ## Generate
      Create complete design system in {base_path}/style-extraction/style-{variant_index}/

  1. **design-tokens.json**:
     - Complete token structure with ALL fields:
       * colors (brand, surface, semantic, text, border) - OKLCH format
       * typography (families, sizes, weights, line heights, letter spacing, combinations)
       * typography.combinations: Predefined typography presets (heading-primary, heading-secondary, body-regular, body-emphasis, caption, label) using var() references
       * spacing (0-24 scale)
       * opacity (0, 10, 20, 40, 60, 80, 90, 100)
       * border_radius (none to full)
       * shadows (sm to xl)
       * component_styles (button, card, input variants) - component presets using var() references
       * breakpoints (sm to 2xl)
     - All colors in OKLCH format
     ${extraction_mode == "explore" ? "- Start from preview colors and expand to full palette" : ""}
     ${extraction_mode == "explore" && refinements.enabled ? "- Apply user refinements where specified" : ""}
     - Common Tailwind CSS usage patterns in project (if extracting from existing project)

  ## Critical Requirements
  - ✅ Use Write() tool immediately for each file
  - ✅ Write to style-{variant_index}/ directory
  - ✅ Can use Exa MCP to research modern design patterns and obtain code examples (Explore/Text mode)
  - ✅ Maintain consistency with user-selected direction
    `
```

**Output**: {actual_variants_count} parallel agent tasks generate design-tokens.json for each variant

## Phase 3: Verify Output

### Step 1: Check Files Created
```bash
# Verify all design systems created
bash(ls {base_path}/style-extraction/style-*/design-tokens.json | wc -l)

# Validate structure
bash(cat {base_path}/style-extraction/style-1/design-tokens.json | grep -q "colors" && echo "valid")
```

### Step 2: Verify File Sizes
```bash
bash(ls -lh {base_path}/style-extraction/style-1/)
```

**Output**: `variants_count × 2` files verified

## Completion

### Todo Update
```javascript
TodoWrite({todos: [
  {content: "Setup and input validation", status: "completed", activeForm: "Validating inputs"},
  {content: "Design space analysis (explore mode)", status: "completed", activeForm: "Analyzing design space"},
  {content: "Design system generation (agent)", status: "completed", activeForm: "Generating design systems"},
  {content: "Verify output files", status: "completed", activeForm: "Verifying files"}
]});
```

### Output Message
```
✅ Style extraction complete!

Configuration:
- Session: {session_id}
- Extraction Mode: {extraction_mode} (imitate/explore)
- Input Mode: {input_mode} (image/text/hybrid)
- Variants: {variants_count}
- Production-Ready: Complete design systems generated

{IF extraction_mode == "explore":
Design Direction Selection:
- You selected: Option {selected_index} - {selected_direction.philosophy_name}
- Generated from {variants_count} contrasting design direction options
}

Generated Files:
{base_path}/style-extraction/
└── style-1/design-tokens.json
{IF extraction_mode == "explore":
{base_path}/.intermediates/style-analysis/analysis-options.json (design direction options + user selection)
}

Next: /workflow:ui-design:layout-extract --session {session_id} --targets "..."
  OR: /workflow:ui-design:generate --session {session_id}
```

## Simple Bash Commands

### Path Operations
```bash
# Find design directory
bash(find .workflow -type d -name "design-run-*" | head -1)

# Expand image pattern
bash(ls {images_pattern})

# Create output directory
bash(mkdir -p {base_path}/style-extraction/)
```

### Validation Commands
```bash
# Check if already extracted
bash(test -f {base_path}/style-extraction/style-1/design-tokens.json && echo "exists")

# Count variants
bash(ls {base_path}/style-extraction/style-* -d | wc -l)

# Validate JSON structure
bash(cat {base_path}/style-extraction/style-1/design-tokens.json | grep -q "colors" && echo "valid")
```

### File Operations
```bash
# Load brainstorming context
bash(test -f .brainstorming/role analysis documents && cat it)

# Create directories (example for multiple variants)
bash(mkdir -p {base_path}/style-extraction/style-1)
bash(mkdir -p {base_path}/style-extraction/style-2)
bash(mkdir -p {base_path}/style-extraction/style-3)

# Verify output
bash(ls {base_path}/style-extraction/style-1/)
bash(test -f {base_path}/.intermediates/style-analysis/analysis-options.json && echo "saved")
```

## Output Structure

```
{base_path}/
├── .intermediates/                  # Intermediate analysis files
│   └── style-analysis/
│       ├── computed-styles.json     # Extracted CSS values from DOM (if URL available)
│       └── analysis-options.json    # Design direction options + user selection (explore mode only)
└── style-extraction/                # Final design system
    └── style-1/
        └── design-tokens.json       # Production-ready design tokens
```

## design-tokens.json Format

```json
{
  "colors": {
    "brand": {"primary": "oklch(...)", "secondary": "oklch(...)", "accent": "oklch(...)"},
    "surface": {"background": "oklch(...)", "elevated": "oklch(...)", "overlay": "oklch(...)"},
    "semantic": {"success": "oklch(...)", "warning": "oklch(...)", "error": "oklch(...)", "info": "oklch(...)"},
    "text": {"primary": "oklch(...)", "secondary": "oklch(...)", "tertiary": "oklch(...)", "inverse": "oklch(...)"},
    "border": {"default": "oklch(...)", "strong": "oklch(...)", "subtle": "oklch(...)"}
  },
  "typography": {
    "font_family": {...},
    "font_size": {...},
    "font_weight": {...},
    "line_height": {...},
    "letter_spacing": {...},
    "combinations": {
      "heading-primary": {"family": "var(--font-family-heading)", "size": "var(--font-size-3xl)", "weight": "var(--font-weight-bold)", "line_height": "var(--line-height-tight)", "letter_spacing": "var(--letter-spacing-tight)"},
      "heading-secondary": {...},
      "body-regular": {...},
      "body-emphasis": {...},
      "caption": {...},
      "label": {...}
    }
  },
  "spacing": {"0": "0", "1": "0.25rem", ..., "24": "6rem"},
  "opacity": {"0": "0", "10": "0.1", "20": "0.2", "40": "0.4", "60": "0.6", "80": "0.8", "90": "0.9", "100": "1"},
  "border_radius": {"none": "0", "sm": "0.25rem", ..., "full": "9999px"},
  "shadows": {"sm": "...", "md": "...", "lg": "...", "xl": "..."},
  "component_styles": {
    "button": {
      "primary": {"background": "var(--color-brand-primary)", "color": "var(--color-text-inverse)", "padding": "var(--spacing-3) var(--spacing-6)", "border_radius": "var(--border-radius-md)", "font_weight": "var(--font-weight-semibold)"},
      "secondary": {...},
      "tertiary": {...}
    },
    "card": {
      "default": {"background": "var(--color-surface-elevated)", "padding": "var(--spacing-6)", "border_radius": "var(--border-radius-lg)", "shadow": "var(--shadow-md)"},
      "interactive": {...}
    },
    "input": {
      "default": {"border": "1px solid var(--color-border-default)", "padding": "var(--spacing-3)", "border_radius": "var(--border-radius-md)", "background": "var(--color-surface-background)"},
      "focus": {...},
      "error": {...}
    }
  },
  "breakpoints": {"sm": "640px", ..., "2xl": "1536px"}
}
```

**Requirements**: OKLCH colors, complete coverage, semantic naming, WCAG AA compliance, typography combinations, component style presets, opacity scale

## Error Handling

### Common Errors
```
ERROR: No images found
→ Check glob pattern

ERROR: Invalid prompt
→ Provide non-empty string

ERROR: Claude JSON parsing error
→ Retry with stricter format
```

## Key Features

- **Direct Design System Generation** - Complete design-tokens.json + style-guide.md in one step
- **AI-Driven Design Space Exploration** - 6D attribute space analysis for maximum contrast
- **Variant-Specific Directions** - Each variant has unique philosophy, keywords, anti-patterns
- **Maximum Contrast Guarantee** - Variants maximally distant in attribute space
- **Flexible Input** - Images, text, or hybrid mode
- **Production-Ready** - OKLCH colors, WCAG AA compliance, semantic naming
- **Agent-Driven** - Autonomous multi-file generation with ui-design-agent


