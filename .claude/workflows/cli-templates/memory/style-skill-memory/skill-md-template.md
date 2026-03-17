# SKILL.md Template for Style Memory Package

---
name: style-{package_name}
description: {intelligent_description}
---

# {Package Name} Style SKILL Package

## 🔍 Quick Index

### Available JSON Fields

**High-level structure overview for quick understanding**

#### design-tokens.json
```
.colors             # Color palette (brand, semantic, surface, text, border)
.typography         # Font families, sizes, weights, line heights
.spacing            # Spacing scale (xs, sm, md, lg, xl, etc.)
.border_radius      # Border radius tokens (sm, md, lg, etc.)
.shadows            # Shadow definitions (elevation levels)
._metadata          # Usage recommendations and guidelines
  ├─ .usage_recommendations.typography
  ├─ .usage_recommendations.spacing
  └─ ...
```

#### layout-templates.json
```
.layout_templates                    # Component layout patterns
  ├─ .<component_name>
  │   ├─ .component_type            # "universal" or "specialized"
  │   ├─ .variants                  # Component variants array
  │   ├─ .usage_guide               # Usage guidelines
  │   │   ├─ .common_sizes
  │   │   ├─ .variant_recommendations
  │   │   ├─ .usage_context
  │   │   └─ .accessibility_tips
  │   └─ ...
```

#### animation-tokens.json (if available)
```
.duration           # Animation duration tokens
.easing            # Easing function tokens
```

---


### Progressive jq Usage Guide

#### 📦 Package Overview

**Base Location**: `.workflow/reference_style/{package_name}/`

**JSON Files**:
- **Design Tokens**: `.workflow/reference_style/{package_name}/design-tokens.json`
- **Layout Templates**: `.workflow/reference_style/{package_name}/layout-templates.json`
- **Animation Tokens**: `.workflow/reference_style/{package_name}/animation-tokens.json` {has_animations ? "(available)" : "(not available)"}

**⚠️ Usage Note**: All jq commands below should be executed with directory context. Use the pattern:
```bash
cd .workflow/reference_style/{package_name} && jq '<query>' <file>.json
```

#### 🔰 Level 0: Basic Queries (~5K tokens)

```bash
# View entire file
jq '.' <file>.json

# List top-level keys
jq 'keys' <file>.json

# Extract specific field
jq '.<field_name>' <file>.json
```

**Use when:** Quick reference, first-time exploration

---

#### 🎯 Level 1: Filter & Extract (~12K tokens)

```bash
# Count items
jq '.<field> | length' <file>.json

# Filter by condition
jq '[.<field>[] | select(.<key> == "<value>")]' <file>.json

# Extract names
jq -r '.<field> | to_entries[] | select(<condition>) | .key' <file>.json

# Formatted output
jq -r '.<field> | to_entries[] | "\(.key): \(.value)"' <file>.json
```

**Universal components filter:** `select(.component_type == "universal")`

**Use when:** Building components, filtering data

---

#### 🚀 Level 2: Combine & Transform (~20K tokens)

```bash
# Pattern search
jq '.<field> | keys[] | select(. | contains("<pattern>"))' <file>.json

# Regex match
jq -r '.<field> | to_entries[] | select(.key | test("<regex>"; "i"))' <file>.json

# Multi-file query
jq '.' file1.json && jq '.' file2.json

# Nested extraction
jq '.<field>["<name>"].<nested_field>' <file>.json

# Preview server (requires directory context)
cd .workflow/reference_style/{package_name} && python -m http.server 8080
```

**Use when:** Complex queries, comprehensive analysis

---

### Quick Reference: Consistency vs Adaptation

| Aspect | Prioritize Consistency | Prioritize Adaptation |
|--------|----------------------|---------------------|
| **UI Components** | Standard buttons, inputs, cards | Hero sections, landing pages |
| **Spacing** | Component internal padding | Page layout, section gaps |
| **Typography** | Body text, headings (h1-h6) | Display text, metric numbers |
| **Colors** | Brand colors, semantic states | Marketing accents, data viz |
| **Radius** | Interactive elements | Feature cards, containers |

---

## 📖 How to Use This SKILL

### Quick Access Pattern

**This SKILL provides design references, NOT executable code.** To use the design system:

1. **Query JSON files with jq commands** 
2. **Extract relevant tokens** for your implementation
3. **Adapt values** based on your specific design needs


### Progressive Loading

- **Level 0** (~5K tokens): Quick token reference with `jq '.colors'`, `jq '.typography'`
- **Level 1** (~12K tokens): Component filtering with `select(.component_type == "universal")`
- **Level 2** (~20K tokens): Complex queries, animations, and preview

---
## ⚡ Core Rules

1. **Reference, Not Requirements**: This is a design reference system for inspiration and guidance. **DO NOT rigidly copy values.** Analyze the patterns and principles, then adapt creatively to build the optimal design for your specific requirements. Your project's unique needs, brand identity, and user context should drive the final design decisions.

2. **Universal Components Only**: When using `layout-templates.json`, **ONLY** reference components where `component_type: "universal"`. **IGNORE** `component_type: "specialized"`.

3. **Token Adaptation**: Adjust design tokens (colors, spacing, typography, shadows, etc.) based on:
   - Brand requirements and identity
   - Accessibility standards (WCAG compliance, readability)
   - Platform conventions (mobile/desktop, iOS/Android/Web)
   - Context needs (light/dark mode, responsive breakpoints)
   - User experience goals and interaction patterns
   - Performance and technical constraints

---

## ⚖️ Token Adaptation Strategy

### Balancing Consistency and Flexibility

**The Core Problem**: Reference tokens may not have suitable sizes or styles for your new interface needs. Blindly copying leads to poor, rigid design. But ignoring tokens breaks visual consistency.

**The Solution**: Use tokens as a **pattern foundation**, not a value prison.

---

### Decision Framework

**When to Use Existing Tokens (Maintain Consistency)**:
- ✅ Common UI elements (buttons, cards, inputs, typography hierarchy)
- ✅ Repeated patterns across multiple pages
- ✅ Standard interactions (hover, focus, active states)
- ✅ When existing token serves the purpose adequately

**When to Extend Tokens (Adapt for New Needs)**:
- ✅ New component type not in reference system
- ✅ Existing token creates visually awkward result
- ✅ Special context requires unique treatment (hero sections, landing pages, data visualization)
- ✅ Target device/platform has different constraints (mobile vs desktop)



### Integration Guidelines

**1. Document Your Extensions**
When you create new token values:
- Note the pattern/method used (interpolation, extrapolation, etc.)
- Document the specific use case
- Consider if it should be added to the design system

**2. Maintain Visual Consistency**
- Use extended values sparingly (don't create 50 spacing tokens)
- Prefer using existing tokens when "close enough" (90% fit is often acceptable)
- Group extended tokens by context (e.g., "hero-section-padding", "dashboard-metric-size")

**3. Respect Core Patterns**
- **Color**: Preserve hue and semantic meaning, adjust lightness/saturation
- **Spacing**: Follow progression pattern (linear/geometric)
- **Typography**: Maintain scale ratio and line-height relationships
- **Radius**: Continue shape language (sharp vs rounded personality)

**4. Know When to Break Rules**
Sometimes you need unique values for special contexts:
- Landing page hero sections
- Marketing/promotional components
- Data visualization (charts, graphs)
- Platform-specific requirements

**For these cases**: Create standalone tokens with clear naming (`hero-title-size`, `chart-accent-color`) rather than forcing into standard scale.

---

## 🎨 Style Understanding & Design References

**IMPORTANT**: These are reference values and patterns extracted from codebase for inspiration. **DO NOT copy rigidly.** Use them as a starting point to understand design patterns, then creatively adapt to build the optimal solution for your specific project requirements, user needs, and brand identity.

### Design Principles

**Dynamically generated based on design token characteristics:**

{GENERATE_PRINCIPLES_FROM_DESIGN_ANALYSIS}

{IF DESIGN_ANALYSIS.has_colors:
**Color System**
- Semantic naming: {DESIGN_ANALYSIS.color_semantic ? "primary/secondary/accent hierarchy" : "descriptive names"}
- Use color intentionally to guide attention and convey meaning
- Maintain consistent color relationships for brand identity
- Ensure sufficient contrast ratios (WCAG AA/AAA) for accessibility
}

{IF DESIGN_ANALYSIS.spacing_pattern detected:
**Spatial Rhythm**
- Primary scale pattern: {DESIGN_ANALYSIS.spacing_scale} (derived from analysis, represents actual token values)
- Pattern characteristics: {DESIGN_ANALYSIS.spacing_pattern} (e.g., "primarily geometric with practical refinements")
- {DESIGN_ANALYSIS.spacing_pattern == "geometric" ? "Geometric progression provides clear hierarchy with exponential growth" : "Linear progression offers subtle gradations for fine control"}
- Apply systematically: smaller values (4-12px) for compact elements, medium (16-32px) for standard spacing, larger (48px+) for section breathing room
- Note: Practical scales may deviate slightly from pure mathematical patterns to optimize for real-world UI needs
}

{IF DESIGN_ANALYSIS.has_typography with typography_hierarchy:
**Typographic System**
- Type scale establishes content hierarchy and readability
- Size progression: {DESIGN_ANALYSIS.typography_scale_example} (e.g., "12px→14px→16px→20px→24px")
- Use scale consistently: body text at base, headings at larger sizes
- Maintain adequate line-height for readability (1.4-1.6 for body text)
}

{IF DESIGN_ANALYSIS.has_radius:
**Shape Language**
- Radius style: {DESIGN_ANALYSIS.radius_style} (e.g., "sharp <4px: modern, technical" or "rounded >8px: friendly, approachable")
- Creates visual personality: sharp = precision, rounded = warmth
- Apply consistently across similar elements (all cards, all buttons)
- Match to brand tone: corporate/technical = sharper, consumer/friendly = rounder
}

{IF DESIGN_ANALYSIS.has_shadows:
**Depth & Elevation**
- Shadow pattern: {DESIGN_ANALYSIS.shadow_pattern} (e.g., "elevation-based: subtle→moderate→prominent")
- Use shadows to indicate interactivity and component importance
- Consistent application reinforces spatial relationships
- Subtle for static cards, prominent for floating/interactive elements
}

{IF DESIGN_ANALYSIS.has_animations:
**Motion & Timing**
- Duration range: {DESIGN_ANALYSIS.animation_range} (e.g., "100ms (fast feedback) to 300ms (content transitions)")
- Easing variety: {DESIGN_ANALYSIS.easing_variety} (e.g., "ease-in-out for natural motion, ease-out for UI responses")
- Fast durations for immediate feedback, slower for spatial changes
- Consistent timing creates predictable, polished experience
}

{ALWAYS include:
**Accessibility First**
- Minimum 4.5:1 contrast for text, 3:1 for UI components (WCAG AA)
- Touch targets ≥44px for mobile interaction
- Clear focus states for keyboard navigation
- Test with screen readers and keyboard-only navigation
}

---
