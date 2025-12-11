#!/bin/bash
# ⚠️ DEPRECATED: This script is deprecated.
# Please use: ccw tool exec ui_instantiate_prototypes '{"designPath":"design-run-1","outputDir":"output"}'
# This file will be removed in a future version.


# UI Prototype Instantiation Script with Preview Generation (v3.0 - Auto-detect)
# Purpose: Generate S × L × P final prototypes from templates + interactive preview files
# Usage:
#   Simple: ui-instantiate-prototypes.sh <prototypes_dir>
#   Full:   ui-instantiate-prototypes.sh <base_path> <pages> <style_variants> <layout_variants> [options]

# Use safer error handling
set -o pipefail

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo "$1"
}

log_success() {
    echo "✅ $1"
}

log_error() {
    echo "❌ $1"
}

log_warning() {
    echo "⚠️  $1"
}

# Auto-detect pages from templates directory
auto_detect_pages() {
    local templates_dir="$1/_templates"

    if [ ! -d "$templates_dir" ]; then
        log_error "Templates directory not found: $templates_dir"
        return 1
    fi

    # Find unique page names from template files (e.g., login-layout-1.html -> login)
    local pages=$(find "$templates_dir" -name "*-layout-*.html" -type f | \
                  sed 's|.*/||' | \
                  sed 's|-layout-[0-9]*\.html||' | \
                  sort -u | \
                  tr '\n' ',' | \
                  sed 's/,$//')

    echo "$pages"
}

# Auto-detect style variants count
auto_detect_style_variants() {
    local base_path="$1"
    local style_dir="$base_path/../style-extraction"

    if [ ! -d "$style_dir" ]; then
        log_warning "Style consolidation directory not found: $style_dir"
        echo "3"  # Default
        return
    fi

    # Count style-* directories
    local count=$(find "$style_dir" -maxdepth 1 -type d -name "style-*" | wc -l)

    if [ "$count" -eq 0 ]; then
        echo "3"  # Default
    else
        echo "$count"
    fi
}

# Auto-detect layout variants count
auto_detect_layout_variants() {
    local templates_dir="$1/_templates"

    if [ ! -d "$templates_dir" ]; then
        echo "3"  # Default
        return
    fi

    # Find the first page and count its layouts
    local first_page=$(find "$templates_dir" -name "*-layout-1.html" -type f | head -1 | sed 's|.*/||' | sed 's|-layout-1\.html||')

    if [ -z "$first_page" ]; then
        echo "3"  # Default
        return
    fi

    # Count layout files for this page
    local count=$(find "$templates_dir" -name "${first_page}-layout-*.html" -type f | wc -l)

    if [ "$count" -eq 0 ]; then
        echo "3"  # Default
    else
        echo "$count"
    fi
}

# ============================================================================
# Parse Arguments
# ============================================================================

show_usage() {
    cat <<'EOF'
Usage:
  Simple (auto-detect): ui-instantiate-prototypes.sh <prototypes_dir> [options]
  Full:                 ui-instantiate-prototypes.sh <base_path> <pages> <style_variants> <layout_variants> [options]

Simple Mode (Recommended):
  prototypes_dir    Path to prototypes directory (auto-detects everything)

Full Mode:
  base_path         Base path to prototypes directory
  pages             Comma-separated list of pages/components
  style_variants    Number of style variants (1-5)
  layout_variants   Number of layout variants (1-5)

Options:
  --run-id <id>         Run ID (default: auto-generated)
  --session-id <id>     Session ID (default: standalone)
  --mode <page|component>  Exploration mode (default: page)
  --template <path>     Path to compare.html template (default: ~/.claude/workflows/_template-compare-matrix.html)
  --no-preview          Skip preview file generation
  --help                Show this help message

Examples:
  # Simple usage (auto-detect everything)
  ui-instantiate-prototypes.sh .workflow/design-run-*/prototypes

  # With options
  ui-instantiate-prototypes.sh .workflow/design-run-*/prototypes --session-id WFS-auth

  # Full manual mode
  ui-instantiate-prototypes.sh .workflow/design-run-*/prototypes "login,dashboard" 3 3 --session-id WFS-auth
EOF
}

# Default values
BASE_PATH=""
PAGES=""
STYLE_VARIANTS=""
LAYOUT_VARIANTS=""
RUN_ID="run-$(date +%Y%m%d-%H%M%S)"
SESSION_ID="standalone"
MODE="page"
TEMPLATE_PATH="$HOME/.claude/workflows/_template-compare-matrix.html"
GENERATE_PREVIEW=true
AUTO_DETECT=false

# Parse arguments
if [ $# -lt 1 ]; then
    log_error "Missing required arguments"
    show_usage
    exit 1
fi

# Check if using simple mode (only 1 positional arg before options)
if [ $# -eq 1 ] || [[ "$2" == --* ]]; then
    # Simple mode - auto-detect
    AUTO_DETECT=true
    BASE_PATH="$1"
    shift 1
else
    # Full mode - manual parameters
    if [ $# -lt 4 ]; then
        log_error "Full mode requires 4 positional arguments"
        show_usage
        exit 1
    fi

    BASE_PATH="$1"
    PAGES="$2"
    STYLE_VARIANTS="$3"
    LAYOUT_VARIANTS="$4"
    shift 4
fi

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --session-id)
            SESSION_ID="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --template)
            TEMPLATE_PATH="$2"
            shift 2
            ;;
        --no-preview)
            GENERATE_PREVIEW=false
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Auto-detection (if enabled)
# ============================================================================

if [ "$AUTO_DETECT" = true ]; then
    log_info "🔍 Auto-detecting configuration from directory..."

    # Detect pages
    PAGES=$(auto_detect_pages "$BASE_PATH")
    if [ -z "$PAGES" ]; then
        log_error "Could not auto-detect pages from templates"
        exit 1
    fi
    log_info "   Pages: $PAGES"

    # Detect style variants
    STYLE_VARIANTS=$(auto_detect_style_variants "$BASE_PATH")
    log_info "   Style variants: $STYLE_VARIANTS"

    # Detect layout variants
    LAYOUT_VARIANTS=$(auto_detect_layout_variants "$BASE_PATH")
    log_info "   Layout variants: $LAYOUT_VARIANTS"

    echo ""
fi

# ============================================================================
# Validation
# ============================================================================

# Validate base path
if [ ! -d "$BASE_PATH" ]; then
    log_error "Base path not found: $BASE_PATH"
    exit 1
fi

# Validate style and layout variants
if [ "$STYLE_VARIANTS" -lt 1 ] || [ "$STYLE_VARIANTS" -gt 5 ]; then
    log_error "Style variants must be between 1 and 5 (got: $STYLE_VARIANTS)"
    exit 1
fi

if [ "$LAYOUT_VARIANTS" -lt 1 ] || [ "$LAYOUT_VARIANTS" -gt 5 ]; then
    log_error "Layout variants must be between 1 and 5 (got: $LAYOUT_VARIANTS)"
    exit 1
fi

# Validate STYLE_VARIANTS against actual style directories
if [ "$STYLE_VARIANTS" -gt 0 ]; then
    style_dir="$BASE_PATH/../style-extraction"

    if [ ! -d "$style_dir" ]; then
        log_error "Style consolidation directory not found: $style_dir"
        log_info "Run /workflow:ui-design:consolidate first"
        exit 1
    fi

    actual_styles=$(find "$style_dir" -maxdepth 1 -type d -name "style-*" 2>/dev/null | wc -l)

    if [ "$actual_styles" -eq 0 ]; then
        log_error "No style directories found in: $style_dir"
        log_info "Run /workflow:ui-design:consolidate first to generate style design systems"
        exit 1
    fi

    if [ "$STYLE_VARIANTS" -gt "$actual_styles" ]; then
        log_warning "Requested $STYLE_VARIANTS style variants, but only found $actual_styles directories"
        log_info "Available style directories:"
        find "$style_dir" -maxdepth 1 -type d -name "style-*" 2>/dev/null | sed 's|.*/||' | sort
        log_info "Auto-correcting to $actual_styles style variants"
        STYLE_VARIANTS=$actual_styles
    fi
fi

# Parse pages into array
IFS=',' read -ra PAGE_ARRAY <<< "$PAGES"

if [ ${#PAGE_ARRAY[@]} -eq 0 ]; then
    log_error "No pages found"
    exit 1
fi

# ============================================================================
# Header Output
# ============================================================================

echo "========================================="
echo "UI Prototype Instantiation & Preview"
if [ "$AUTO_DETECT" = true ]; then
    echo "(Auto-detected configuration)"
fi
echo "========================================="
echo "Base Path: $BASE_PATH"
echo "Mode: $MODE"
echo "Pages/Components: $PAGES"
echo "Style Variants: $STYLE_VARIANTS"
echo "Layout Variants: $LAYOUT_VARIANTS"
echo "Run ID: $RUN_ID"
echo "Session ID: $SESSION_ID"
echo "========================================="
echo ""

# Change to base path
cd "$BASE_PATH" || exit 1

# ============================================================================
# Phase 1: Instantiate Prototypes
# ============================================================================

log_info "🚀 Phase 1: Instantiating prototypes from templates..."
echo ""

total_generated=0
total_failed=0

for page in "${PAGE_ARRAY[@]}"; do
    # Trim whitespace
    page=$(echo "$page" | xargs)

    log_info "Processing page/component: $page"

    for s in $(seq 1 "$STYLE_VARIANTS"); do
        for l in $(seq 1 "$LAYOUT_VARIANTS"); do
            # Define file paths
            TEMPLATE_HTML="_templates/${page}-layout-${l}.html"
            STRUCTURAL_CSS="_templates/${page}-layout-${l}.css"
            TOKEN_CSS="../style-extraction/style-${s}/tokens.css"
            OUTPUT_HTML="${page}-style-${s}-layout-${l}.html"

            # Copy template and replace placeholders
            if [ -f "$TEMPLATE_HTML" ]; then
                cp "$TEMPLATE_HTML" "$OUTPUT_HTML" || {
                    log_error "Failed to copy template: $TEMPLATE_HTML"
                    ((total_failed++))
                    continue
                }

                # Replace CSS placeholders (Windows-compatible sed syntax)
                sed -i "s|{{STRUCTURAL_CSS}}|${STRUCTURAL_CSS}|g" "$OUTPUT_HTML" || true
                sed -i "s|{{TOKEN_CSS}}|${TOKEN_CSS}|g" "$OUTPUT_HTML" || true

                log_success "Created: $OUTPUT_HTML"
                ((total_generated++))

                # Create implementation notes (simplified)
                NOTES_FILE="${page}-style-${s}-layout-${l}-notes.md"

                # Generate notes with simple heredoc
                cat > "$NOTES_FILE" <<NOTESEOF
# Implementation Notes: ${page}-style-${s}-layout-${l}

## Generation Details
- **Template**: ${TEMPLATE_HTML}
- **Structural CSS**: ${STRUCTURAL_CSS}
- **Style Tokens**: ${TOKEN_CSS}
- **Layout Strategy**: Layout ${l}
- **Style Variant**: Style ${s}
- **Mode**: ${MODE}

## Template Reuse
This prototype was generated from a shared layout template to ensure consistency
across all style variants. The HTML structure is identical for all ${page}-layout-${l}
prototypes, with only the design tokens (colors, fonts, spacing) varying.

## Design System Reference
Refer to \`../style-extraction/style-${s}/style-guide.md\` for:
- Design philosophy
- Token usage guidelines
- Component patterns
- Accessibility requirements

## Customization
To modify this prototype:
1. Edit the layout template: \`${TEMPLATE_HTML}\` (affects all styles)
2. Edit the structural CSS: \`${STRUCTURAL_CSS}\` (affects all styles)
3. Edit design tokens: \`${TOKEN_CSS}\` (affects only this style variant)

## Run Information
- **Run ID**: ${RUN_ID}
- **Session ID**: ${SESSION_ID}
- **Generated**: $(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +%Y-%m-%d)
NOTESEOF

            else
                log_error "Template not found: $TEMPLATE_HTML"
                ((total_failed++))
            fi
        done
    done
done

echo ""
log_success "Phase 1 complete: Generated ${total_generated} prototypes"
if [ $total_failed -gt 0 ]; then
    log_warning "Failed: ${total_failed} prototypes"
fi
echo ""

# ============================================================================
# Phase 2: Generate Preview Files (if enabled)
# ============================================================================

if [ "$GENERATE_PREVIEW" = false ]; then
    log_info "⏭️  Skipping preview generation (--no-preview flag)"
    exit 0
fi

log_info "🎨 Phase 2: Generating preview files..."
echo ""

# ============================================================================
# 2a. Generate compare.html from template
# ============================================================================

if [ ! -f "$TEMPLATE_PATH" ]; then
    log_warning "Template not found: $TEMPLATE_PATH"
    log_info "   Skipping compare.html generation"
else
    log_info "📄 Generating compare.html from template..."

    # Convert page array to JSON format
    PAGES_JSON="["
    for i in "${!PAGE_ARRAY[@]}"; do
        page=$(echo "${PAGE_ARRAY[$i]}" | xargs)
        PAGES_JSON+="\"$page\""
        if [ $i -lt $((${#PAGE_ARRAY[@]} - 1)) ]; then
            PAGES_JSON+=", "
        fi
    done
    PAGES_JSON+="]"

    TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +%Y-%m-%d)

    # Read template and replace placeholders
    cat "$TEMPLATE_PATH" | \
        sed "s|{{run_id}}|${RUN_ID}|g" | \
        sed "s|{{session_id}}|${SESSION_ID}|g" | \
        sed "s|{{timestamp}}|${TIMESTAMP}|g" | \
        sed "s|{{style_variants}}|${STYLE_VARIANTS}|g" | \
        sed "s|{{layout_variants}}|${LAYOUT_VARIANTS}|g" | \
        sed "s|{{pages_json}}|${PAGES_JSON}|g" \
        > compare.html

    log_success "Generated: compare.html"
fi

# ============================================================================
# 2b. Generate index.html
# ============================================================================

log_info "📄 Generating index.html..."

# Calculate total prototypes
TOTAL_PROTOTYPES=$((STYLE_VARIANTS * LAYOUT_VARIANTS * ${#PAGE_ARRAY[@]}))

# Generate index.html with simple heredoc
cat > index.html <<'INDEXEOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UI Prototypes - __MODE__ Mode - __RUN_ID__</title>
  <style>
    body {
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 900px;
      margin: 2rem auto;
      padding: 0 2rem;
      background: #f9fafb;
    }
    .header {
      background: white;
      padding: 2rem;
      border-radius: 0.75rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      margin-bottom: 2rem;
    }
    h1 {
      color: #2563eb;
      margin-bottom: 0.5rem;
      font-size: 2rem;
    }
    .meta {
      color: #6b7280;
      font-size: 0.875rem;
      margin-top: 0.5rem;
    }
    .info {
      background: #f3f4f6;
      padding: 1.5rem;
      border-radius: 0.5rem;
      margin: 1.5rem 0;
      border-left: 4px solid #2563eb;
    }
    .cta {
      display: inline-block;
      background: #2563eb;
      color: white;
      padding: 1rem 2rem;
      border-radius: 0.5rem;
      text-decoration: none;
      font-weight: 600;
      margin: 1rem 0;
      transition: background 0.2s;
    }
    .cta:hover {
      background: #1d4ed8;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1rem;
      margin: 1.5rem 0;
    }
    .stat {
      background: white;
      border: 1px solid #e5e7eb;
      padding: 1.5rem;
      border-radius: 0.5rem;
      text-align: center;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stat-value {
      font-size: 2.5rem;
      font-weight: bold;
      color: #2563eb;
      margin-bottom: 0.25rem;
    }
    .stat-label {
      color: #6b7280;
      font-size: 0.875rem;
    }
    .section {
      background: white;
      padding: 2rem;
      border-radius: 0.75rem;
      margin-bottom: 2rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    h2 {
      color: #1f2937;
      margin-bottom: 1rem;
      font-size: 1.5rem;
    }
    ul {
      line-height: 1.8;
      color: #374151;
    }
    .pages-list {
      list-style: none;
      padding: 0;
    }
    .pages-list li {
      background: #f9fafb;
      padding: 0.75rem 1rem;
      margin: 0.5rem 0;
      border-radius: 0.375rem;
      border-left: 3px solid #2563eb;
    }
    .badge {
      display: inline-block;
      background: #dbeafe;
      color: #1e40af;
      padding: 0.25rem 0.75rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      font-weight: 600;
      margin-left: 0.5rem;
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>🎨 UI Prototype __MODE__ Mode</h1>
    <div class="meta">
      <strong>Run ID:</strong> __RUN_ID__ |
      <strong>Session:</strong> __SESSION_ID__ |
      <strong>Generated:</strong> __TIMESTAMP__
    </div>
  </div>

  <div class="info">
    <p><strong>Matrix Configuration:</strong> __STYLE_VARIANTS__ styles × __LAYOUT_VARIANTS__ layouts × __PAGE_COUNT__ __MODE__s</p>
    <p><strong>Total Prototypes:</strong> __TOTAL_PROTOTYPES__ interactive HTML files</p>
  </div>

  <a href="compare.html" class="cta">🔍 Open Interactive Matrix Comparison →</a>

  <div class="stats">
    <div class="stat">
      <div class="stat-value">__STYLE_VARIANTS__</div>
      <div class="stat-label">Style Variants</div>
    </div>
    <div class="stat">
      <div class="stat-value">__LAYOUT_VARIANTS__</div>
      <div class="stat-label">Layout Options</div>
    </div>
    <div class="stat">
      <div class="stat-value">__PAGE_COUNT__</div>
      <div class="stat-label">__MODE__s</div>
    </div>
    <div class="stat">
      <div class="stat-value">__TOTAL_PROTOTYPES__</div>
      <div class="stat-label">Total Prototypes</div>
    </div>
  </div>

  <div class="section">
    <h2>🌟 Features</h2>
    <ul>
      <li><strong>Interactive Matrix View:</strong> __STYLE_VARIANTS__×__LAYOUT_VARIANTS__ grid with synchronized scrolling</li>
      <li><strong>Flexible Zoom:</strong> 25%, 50%, 75%, 100% viewport scaling</li>
      <li><strong>Fullscreen Mode:</strong> Detailed view for individual prototypes</li>
      <li><strong>Selection System:</strong> Mark favorites with export to JSON</li>
      <li><strong>__MODE__ Switcher:</strong> Compare different __MODE__s side-by-side</li>
      <li><strong>Persistent State:</strong> Selections saved in localStorage</li>
    </ul>
  </div>

  <div class="section">
    <h2>📄 Generated __MODE__s</h2>
    <ul class="pages-list">
__PAGES_LIST__
    </ul>
  </div>

  <div class="section">
    <h2>📚 Next Steps</h2>
    <ol>
      <li>Open <code>compare.html</code> to explore all variants in matrix view</li>
      <li>Use zoom and sync scroll controls to compare details</li>
      <li>Select your preferred style×layout combinations</li>
      <li>Export selections as JSON for implementation planning</li>
      <li>Review implementation notes in <code>*-notes.md</code> files</li>
    </ol>
  </div>
</body>
</html>
INDEXEOF

# Build pages list HTML
PAGES_LIST_HTML=""
for page in "${PAGE_ARRAY[@]}"; do
    page=$(echo "$page" | xargs)
    VARIANT_COUNT=$((STYLE_VARIANTS * LAYOUT_VARIANTS))
    PAGES_LIST_HTML+="      <li>\n"
    PAGES_LIST_HTML+="        <strong>${page}</strong>\n"
    PAGES_LIST_HTML+="        <span class=\"badge\">${STYLE_VARIANTS}×${LAYOUT_VARIANTS} = ${VARIANT_COUNT} variants</span>\n"
    PAGES_LIST_HTML+="      </li>\n"
done

# Replace all placeholders in index.html
MODE_UPPER=$(echo "$MODE" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')
sed -i "s|__RUN_ID__|${RUN_ID}|g" index.html
sed -i "s|__SESSION_ID__|${SESSION_ID}|g" index.html
sed -i "s|__TIMESTAMP__|${TIMESTAMP}|g" index.html
sed -i "s|__MODE__|${MODE_UPPER}|g" index.html
sed -i "s|__STYLE_VARIANTS__|${STYLE_VARIANTS}|g" index.html
sed -i "s|__LAYOUT_VARIANTS__|${LAYOUT_VARIANTS}|g" index.html
sed -i "s|__PAGE_COUNT__|${#PAGE_ARRAY[@]}|g" index.html
sed -i "s|__TOTAL_PROTOTYPES__|${TOTAL_PROTOTYPES}|g" index.html
sed -i "s|__PAGES_LIST__|${PAGES_LIST_HTML}|g" index.html

log_success "Generated: index.html"

# ============================================================================
# 2c. Generate PREVIEW.md
# ============================================================================

log_info "📄 Generating PREVIEW.md..."

cat > PREVIEW.md <<PREVIEWEOF
# UI Prototype Preview Guide

## Quick Start
1. Open \`index.html\` for overview and navigation
2. Open \`compare.html\` for interactive matrix comparison
3. Use browser developer tools to inspect responsive behavior

## Configuration

- **Exploration Mode:** ${MODE_UPPER}
- **Run ID:** ${RUN_ID}
- **Session ID:** ${SESSION_ID}
- **Style Variants:** ${STYLE_VARIANTS}
- **Layout Options:** ${LAYOUT_VARIANTS}
- **${MODE_UPPER}s:** ${PAGES}
- **Total Prototypes:** ${TOTAL_PROTOTYPES}
- **Generated:** ${TIMESTAMP}

## File Naming Convention

\`\`\`
{${MODE}}-style-{s}-layout-{l}.html
\`\`\`

**Example:** \`dashboard-style-1-layout-2.html\`
- ${MODE_UPPER}: dashboard
- Style: Design system 1
- Layout: Layout variant 2

## Interactive Features (compare.html)

### Matrix View
- **Grid Layout:** ${STYLE_VARIANTS}×${LAYOUT_VARIANTS} table with all prototypes visible
- **Synchronized Scroll:** All iframes scroll together (toggle with button)
- **Zoom Controls:** Adjust viewport scale (25%, 50%, 75%, 100%)
- **${MODE_UPPER} Selector:** Switch between different ${MODE}s instantly

### Prototype Actions
- **⭐ Selection:** Click star icon to mark favorites
- **⛶ Fullscreen:** View prototype in fullscreen overlay
- **↗ New Tab:** Open prototype in dedicated browser tab

### Selection Export
1. Select preferred prototypes using star icons
2. Click "Export Selection" button
3. Downloads JSON file: \`selection-${RUN_ID}.json\`
4. Use exported file for implementation planning

## Design System References

Each prototype references a specific style design system:
PREVIEWEOF

# Add style references
for s in $(seq 1 "$STYLE_VARIANTS"); do
    cat >> PREVIEW.md <<STYLEEOF

### Style ${s}
- **Tokens:** \`../style-extraction/style-${s}/design-tokens.json\`
- **CSS Variables:** \`../style-extraction/style-${s}/tokens.css\`
- **Style Guide:** \`../style-extraction/style-${s}/style-guide.md\`
STYLEEOF
done

cat >> PREVIEW.md <<'FOOTEREOF'

## Responsive Testing

All prototypes are mobile-first responsive. Test at these breakpoints:

- **Mobile:** 375px - 767px
- **Tablet:** 768px - 1023px
- **Desktop:** 1024px+

Use browser DevTools responsive mode for testing.

## Accessibility Features

- Semantic HTML5 structure
- ARIA attributes for screen readers
- Keyboard navigation support
- Proper heading hierarchy
- Focus indicators

## Next Steps

1. **Review:** Open `compare.html` and explore all variants
2. **Select:** Mark preferred prototypes using star icons
3. **Export:** Download selection JSON for implementation
4. **Implement:** Use `/workflow:ui-design:update` to integrate selected designs
5. **Plan:** Run `/workflow:plan` to generate implementation tasks

---

**Generated by:** `ui-instantiate-prototypes.sh`
**Version:** 3.0 (auto-detect mode)
FOOTEREOF

log_success "Generated: PREVIEW.md"

# ============================================================================
# Completion Summary
# ============================================================================

echo ""
echo "========================================="
echo "✅ Generation Complete!"
echo "========================================="
echo ""
echo "📊 Summary:"
echo "   Prototypes: ${total_generated} generated"
if [ $total_failed -gt 0 ]; then
    echo "   Failed: ${total_failed}"
fi
echo "   Preview Files: compare.html, index.html, PREVIEW.md"
echo "   Matrix: ${STYLE_VARIANTS}×${LAYOUT_VARIANTS} (${#PAGE_ARRAY[@]} ${MODE}s)"
echo "   Total Files: ${TOTAL_PROTOTYPES} prototypes + preview files"
echo ""
echo "🌐 Next Steps:"
echo "   1. Open: ${BASE_PATH}/index.html"
echo "   2. Explore: ${BASE_PATH}/compare.html"
echo "   3. Review: ${BASE_PATH}/PREVIEW.md"
echo ""
echo "Performance: Template-based approach with ${STYLE_VARIANTS}× speedup"
echo "========================================="
