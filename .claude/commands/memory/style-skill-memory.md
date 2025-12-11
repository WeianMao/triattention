---
name: style-skill-memory
description: Generate SKILL memory package from style reference for easy loading and consistent design system usage
argument-hint: "[package-name] [--regenerate]"
allowed-tools: Bash,Read,Write,TodoWrite
auto-continue: true
---

# Memory: Style SKILL Memory Generator

## Overview

**Purpose**: Convert style reference package into SKILL memory for easy loading and context management.

**Input**: Style reference package at `.workflow/reference_style/{package-name}/`

**Output**: SKILL memory index at `.claude/skills/style-{package-name}/SKILL.md`

**Use Case**: Load design system context when working with UI components, analyzing design patterns, or implementing style guidelines.

**Key Features**:
- Extracts primary design references (colors, typography, spacing, etc.)
- Provides dynamic adjustment guidelines for design tokens
- Includes prerequisites and tooling requirements (browsers, PostCSS, dark mode)
- Progressive loading structure for efficient token usage
- Complete implementation examples with React components
- Interactive preview showcase

---

## Quick Reference

### Command Syntax

```bash
/memory:style-skill-memory [package-name] [--regenerate]

# Arguments
package-name    Style reference package name (required)
--regenerate    Force regenerate SKILL.md even if it exists (optional)
```

### Usage Examples

```bash
# Generate SKILL memory for package
/memory:style-skill-memory main-app-style-v1

# Regenerate SKILL memory
/memory:style-skill-memory main-app-style-v1 --regenerate

# Package name from current directory or default
/memory:style-skill-memory
```

### Key Variables

**Input Variables**:
- `PACKAGE_NAME`: Style reference package name
- `PACKAGE_DIR`: `.workflow/reference_style/${package_name}`
- `SKILL_DIR`: `.claude/skills/style-${package_name}`
- `REGENERATE`: `true` if --regenerate flag, `false` otherwise

**Data Sources** (Phase 2):
- `DESIGN_TOKENS_DATA`: Complete design-tokens.json content (from Read)
- `LAYOUT_TEMPLATES_DATA`: Complete layout-templates.json content (from Read)
- `ANIMATION_TOKENS_DATA`: Complete animation-tokens.json content (from Read, if exists)

**Metadata** (Phase 2):
- `COMPONENT_COUNT`: Total components
- `UNIVERSAL_COUNT`: Universal components count
- `SPECIALIZED_COUNT`: Specialized components count
- `UNIVERSAL_COMPONENTS`: Universal component names (first 5)
- `HAS_ANIMATIONS`: Whether animation-tokens.json exists

**Analysis Output** (`DESIGN_ANALYSIS` - Phase 2):
- `has_colors`: Colors exist
- `color_semantic`: Has semantic naming (primary/secondary/accent)
- `uses_oklch`: Uses modern color spaces (oklch, lab, etc.)
- `has_dark_mode`: Has separate light/dark mode color tokens
- `spacing_pattern`: Pattern type ("linear", "geometric", "custom")
- `spacing_scale`: Actual scale values (e.g., [4, 8, 16, 32, 64])
- `has_typography`: Typography system exists
- `typography_hierarchy`: Has size scale for hierarchy
- `uses_calc`: Uses calc() expressions in token values
- `has_radius`: Border radius exists
- `radius_style`: Style characteristic ("sharp" <4px, "moderate" 4-8px, "rounded" >8px)
- `has_shadows`: Shadow system exists
- `shadow_pattern`: Elevation naming pattern
- `has_animations`: Animation tokens exist
- `animation_range`: Duration range (fast to slow)
- `easing_variety`: Types of easing functions

### Common Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| Package not found | Invalid package name or doesn't exist | Run `/workflow:ui-design:codify-style` first |
| SKILL already exists | SKILL.md already generated | Use `--regenerate` flag |
| Missing layout-templates.json | Incomplete package | Verify package integrity, re-run codify-style |
| Invalid JSON format | Corrupted package files | Regenerate package with codify-style |

---

## Execution Process

### Phase 1: Validate Package

**TodoWrite** (First Action):
```json
[
  {
    "content": "Validate package exists and check SKILL status",
    "activeForm": "Validating package and SKILL status",
    "status": "in_progress"
  },
  {
    "content": "Read package data and analyze design system",
    "activeForm": "Reading package data and analyzing design system",
    "status": "pending"
  },
  {
    "content": "Generate SKILL.md with design principles and token values",
    "activeForm": "Generating SKILL.md with design principles and token values",
    "status": "pending"
  }
]
```

**Step 1: Parse Package Name**

```bash
# Get package name from argument or auto-detect
bash(echo "${package_name}" || basename "$(pwd)" | sed 's/^style-//')
```

**Step 2: Validate Package Exists**

```bash
bash(test -d .workflow/reference_style/${package_name} && echo "exists" || echo "missing")
```

**Error Handling**:
```javascript
if (package_not_exists) {
  error("ERROR: Style reference package not found: ${package_name}")
  error("HINT: Run '/workflow:ui-design:codify-style' first to create package")
  error("Available packages:")
  bash(ls -1 .workflow/reference_style/ 2>/dev/null || echo "  (none)")
  exit(1)
}
```

**Step 3: Check SKILL Already Exists**

```bash
bash(test -f .claude/skills/style-${package_name}/SKILL.md && echo "exists" || echo "missing")
```

**Decision Logic**:
```javascript
if (skill_exists && !regenerate_flag) {
  echo("SKILL memory already exists for: ${package_name}")
  echo("Use --regenerate to force regeneration")
  exit(0)
}

if (regenerate_flag && skill_exists) {
  echo("Regenerating SKILL memory for: ${package_name}")
}
```

**TodoWrite Update**: Mark "Validate" as completed, "Read package data" as in_progress

---

### Phase 2: Read Package Data & Analyze Design System

**Step 1: Read All JSON Files**

```bash
# Read layout templates
Read(file_path=".workflow/reference_style/${package_name}/layout-templates.json")

# Read design tokens
Read(file_path=".workflow/reference_style/${package_name}/design-tokens.json")

# Read animation tokens (if exists)
bash(test -f .workflow/reference_style/${package_name}/animation-tokens.json && echo "exists" || echo "missing")
Read(file_path=".workflow/reference_style/${package_name}/animation-tokens.json")  # if exists
```

**Step 2: Extract Metadata for Description**

```bash
# Count components and classify by type
bash(jq '.layout_templates | length' layout-templates.json)
bash(jq '[.layout_templates[] | select(.component_type == "universal")] | length' layout-templates.json)
bash(jq '[.layout_templates[] | select(.component_type == "specialized")] | length' layout-templates.json)
bash(jq -r '.layout_templates | to_entries[] | select(.value.component_type == "universal") | .key' layout-templates.json | head -5)
```

Store results in metadata variables (see [Key Variables](#key-variables))

**Step 3: Analyze Design System for Dynamic Principles**

Analyze design-tokens.json to extract characteristics and patterns:

```bash
# Color system characteristics
bash(jq '.colors | keys' design-tokens.json)
bash(jq '.colors | to_entries[0:2] | map(.value)' design-tokens.json)
# Check for modern color spaces
bash(jq '.colors | to_entries[] | .value | test("oklch|lab|lch")' design-tokens.json)
# Check for dark mode variants
bash(jq '.colors | keys | map(select(contains("dark") or contains("light")))' design-tokens.json)
# → Store: has_colors, color_semantic, uses_oklch, has_dark_mode

# Spacing pattern detection
bash(jq '.spacing | to_entries | map(.value) | map(gsub("[^0-9.]"; "") | tonumber)' design-tokens.json)
# Analyze pattern: linear (4-8-12-16) vs geometric (4-8-16-32) vs custom
# → Store: spacing_pattern, spacing_scale

# Typography characteristics
bash(jq '.typography | keys | map(select(contains("family") or contains("weight")))' design-tokens.json)
bash(jq '.typography | to_entries | map(select(.key | contains("size"))) | .[].value' design-tokens.json)
# Check for calc() usage
bash(jq '. | tostring | test("calc\\(")' design-tokens.json)
# → Store: has_typography, typography_hierarchy, uses_calc

# Border radius style
bash(jq '.border_radius | to_entries | map(.value)' design-tokens.json)
# Check range: small (sharp <4px) vs moderate (4-8px) vs large (rounded >8px)
# → Store: has_radius, radius_style

# Shadow characteristics
bash(jq '.shadows | keys' design-tokens.json)
bash(jq '.shadows | to_entries[0].value' design-tokens.json)
# → Store: has_shadows, shadow_pattern

# Animations (if available)
bash(jq '.duration | to_entries | map(.value)' animation-tokens.json)
bash(jq '.easing | keys' animation-tokens.json)
# → Store: has_animations, animation_range, easing_variety
```

Store analysis results in `DESIGN_ANALYSIS` (see [Key Variables](#key-variables))

**Note**: Analysis focuses on characteristics and patterns, not counts. Include technical feature detection (oklch, calc, dark mode) for Prerequisites section.

**TodoWrite Update**: Mark "Read package data" as completed, "Generate SKILL.md" as in_progress

---

### Phase 3: Generate SKILL.md

**Step 1: Create SKILL Directory**

```bash
bash(mkdir -p .claude/skills/style-${package_name})
```

**Step 2: Generate Intelligent Description**

**Format**:
```
{package_name} project-independent design system with {universal_count} universal layout templates and interactive preview (located at .workflow/reference_style/{package_name}). Load when working with reusable UI components, design tokens, layout patterns, or implementing visual consistency. Excludes {specialized_count} project-specific components.
```

**Step 3: Load and Process SKILL.md Template**

**⚠️ CRITICAL - Execute First**:
```bash
bash(cat ~/.claude/workflows/cli-templates/memory/style-skill-memory/skill-md-template.md)
```

**Template Processing**:
1. **Replace variables**: Substitute all `{variable}` placeholders with actual values from Phase 2
2. **Generate dynamic sections**:
   - **Prerequisites & Tooling**: Generate based on `DESIGN_ANALYSIS` technical features (oklch, calc, dark mode)
   - **Design Principles**: Generate based on `DESIGN_ANALYSIS` characteristics
   - **Complete Implementation Example**: Include React component example with token adaptation
   - **Design Token Values**: Iterate `DESIGN_TOKENS_DATA`, `ANIMATION_TOKENS_DATA` and display all key-value pairs with DEFAULT annotations
3. **Write to file**: Use Write tool to save to `.claude/skills/style-{package_name}/SKILL.md`

**Variable Replacement Map**:
- `{package_name}` → PACKAGE_NAME
- `{intelligent_description}` → Generated description from Step 2
- `{component_count}` → COMPONENT_COUNT
- `{universal_count}` → UNIVERSAL_COUNT
- `{specialized_count}` → SPECIALIZED_COUNT
- `{universal_components_list}` → UNIVERSAL_COMPONENTS (comma-separated)
- `{has_animations}` → HAS_ANIMATIONS

**Dynamic Content Generation**:

See template file for complete structure. Key dynamic sections:

1. **Prerequisites & Tooling** (based on DESIGN_ANALYSIS technical features):
   - IF uses_oklch → Include PostCSS plugin requirement (`postcss-oklab-function`)
   - IF uses_calc → Include preprocessor requirement for calc() expressions
   - IF has_dark_mode → Include dark mode implementation mechanism (class or media query)
   - ALWAYS include browser support, jq installation, and local server setup

2. **Design Principles** (based on DESIGN_ANALYSIS):
   - IF has_colors → Include "Color System" principle with semantic pattern
   - IF spacing_pattern detected → Include "Spatial Rhythm" with unified scale description (actual token values)
   - IF has_typography_hierarchy → Include "Typographic System" with scale examples
   - IF has_radius → Include "Shape Language" with style characteristic
   - IF has_shadows → Include "Depth & Elevation" with elevation pattern
   - IF has_animations → Include "Motion & Timing" with duration range
   - ALWAYS include "Accessibility First" principle

3. **Design Token Values** (iterate from read data):
   - Colors: Iterate `DESIGN_TOKENS_DATA.colors`
   - Typography: Iterate `DESIGN_TOKENS_DATA.typography`
   - Spacing: Iterate `DESIGN_TOKENS_DATA.spacing`
   - Border Radius: Iterate `DESIGN_TOKENS_DATA.border_radius` with calc() explanations
   - Shadows: Iterate `DESIGN_TOKENS_DATA.shadows` with DEFAULT token annotations
   - Animations (if available): Iterate `ANIMATION_TOKENS_DATA.duration` and `ANIMATION_TOKENS_DATA.easing`

**Step 4: Verify SKILL.md Created**

```bash
bash(test -f .claude/skills/style-${package_name}/SKILL.md && echo "success" || echo "failed")
```

**TodoWrite Update**: Mark all todos as completed

---

### Completion Message

Display a simple completion message with key information:

```
✅ SKILL memory generated for style package: {package_name}

📁 Location: .claude/skills/style-{package_name}/SKILL.md

📊 Package Summary:
   - {component_count} components ({universal_count} universal, {specialized_count} specialized)
   - Design tokens: colors, typography, spacing, shadows{animations_note}

💡 Usage: /memory:load-skill-memory style-{package_name} "your task description"
```

Variables: `{package_name}`, `{component_count}`, `{universal_count}`, `{specialized_count}`, `{animations_note}` (", animations" if exists)

---

## Implementation Details

### Critical Rules

1. **Check Before Generate**: Verify package exists before attempting SKILL generation
2. **Respect Existing SKILL**: Don't overwrite unless --regenerate flag provided
3. **Load Templates via cat**: Use `cat ~/.claude/workflows/cli-templates/memory/style-skill-memory/{template}` to load templates
4. **Variable Substitution**: Replace all `{variable}` placeholders with actual values
5. **Technical Feature Detection**: Analyze tokens for modern features (oklch, calc, dark mode) and generate appropriate Prerequisites section
6. **Dynamic Content Generation**: Generate sections based on DESIGN_ANALYSIS characteristics
7. **Unified Spacing Scale**: Use actual token values as primary scale reference, avoid contradictory pattern descriptions
8. **Direct Iteration**: Iterate data structures (DESIGN_TOKENS_DATA, etc.) for token values
9. **Annotate Special Tokens**: Add comments for DEFAULT tokens and calc() expressions
10. **Embed jq Commands**: Include bash/jq commands in SKILL.md for dynamic loading
11. **Progressive Loading**: Include all 3 levels (0-2) with specific jq commands
12. **Complete Examples**: Include end-to-end implementation examples (React components)
13. **Intelligent Description**: Extract component count and key features from metadata
14. **Emphasize Flexibility**: Strongly warn against rigid copying - values are references for creative adaptation

### Template Files Location


```
Phase 1: Validate
  ├─ Parse package_name
  ├─ Check PACKAGE_DIR exists
  └─ Check SKILL_DIR exists (skip if exists and no --regenerate)

Phase 2: Read & Analyze
  ├─ Read design-tokens.json → DESIGN_TOKENS_DATA
  ├─ Read layout-templates.json → LAYOUT_TEMPLATES_DATA
  ├─ Read animation-tokens.json → ANIMATION_TOKENS_DATA (if exists)
  ├─ Extract Metadata → COMPONENT_COUNT, UNIVERSAL_COUNT, etc.
  └─ Analyze Design System → DESIGN_ANALYSIS (characteristics)

Phase 3: Generate
  ├─ Create SKILL directory
  ├─ Generate intelligent description
  ├─ Load SKILL.md template (cat command)
  ├─ Replace variables and generate dynamic content
  ├─ Write SKILL.md
  ├─ Verify creation
  ├─ Load completion message template (cat command)
  └─ Display completion message
```
