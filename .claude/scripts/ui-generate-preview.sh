#!/bin/bash
# ⚠️ DEPRECATED: This script is deprecated.
# Please use: ccw tool exec ui_generate_preview '{"designPath":"design-run-1","outputDir":"preview"}'
# This file will be removed in a future version.

#
# UI Generate Preview v2.0 - Template-Based Preview Generation
# Purpose: Generate compare.html and index.html using template substitution
# Template: ~/.claude/workflows/_template-compare-matrix.html
#
# Usage: ui-generate-preview.sh <prototypes_dir> [--template <path>]
#

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default template path
TEMPLATE_PATH="$HOME/.claude/workflows/_template-compare-matrix.html"

# Parse arguments
prototypes_dir="${1:-.}"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --template)
            TEMPLATE_PATH="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

if [[ ! -d "$prototypes_dir" ]]; then
    echo -e "${RED}Error: Directory not found: $prototypes_dir${NC}"
    exit 1
fi

cd "$prototypes_dir" || exit 1

echo -e "${GREEN}📊 Auto-detecting matrix dimensions...${NC}"

# Auto-detect styles, layouts, targets from file patterns
# Pattern: {target}-style-{s}-layout-{l}.html
styles=$(find . -maxdepth 1 -name "*-style-*-layout-*.html" | \
         sed 's/.*-style-\([0-9]\+\)-.*/\1/' | sort -un)
layouts=$(find . -maxdepth 1 -name "*-style-*-layout-*.html" | \
          sed 's/.*-layout-\([0-9]\+\)\.html/\1/' | sort -un)
targets=$(find . -maxdepth 1 -name "*-style-*-layout-*.html" | \
          sed 's/\.\///; s/-style-.*//' | sort -u)

S=$(echo "$styles" | wc -l)
L=$(echo "$layouts" | wc -l)
T=$(echo "$targets" | wc -l)

echo -e "   Detected: ${GREEN}${S}${NC} styles × ${GREEN}${L}${NC} layouts × ${GREEN}${T}${NC} targets"

if [[ $S -eq 0 ]] || [[ $L -eq 0 ]] || [[ $T -eq 0 ]]; then
    echo -e "${RED}Error: No prototype files found matching pattern {target}-style-{s}-layout-{l}.html${NC}"
    exit 1
fi

# ============================================================================
# Generate compare.html from template
# ============================================================================

echo -e "${YELLOW}🎨 Generating compare.html from template...${NC}"

if [[ ! -f "$TEMPLATE_PATH" ]]; then
    echo -e "${RED}Error: Template not found: $TEMPLATE_PATH${NC}"
    exit 1
fi

# Build pages/targets JSON array
PAGES_JSON="["
first=true
for target in $targets; do
    if [[ "$first" == true ]]; then
        first=false
    else
        PAGES_JSON+=", "
    fi
    PAGES_JSON+="\"$target\""
done
PAGES_JSON+="]"

# Generate metadata
RUN_ID="run-$(date +%Y%m%d-%H%M%S)"
SESSION_ID="standalone"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +"%Y-%m-%d")

# Replace placeholders in template
cat "$TEMPLATE_PATH" | \
    sed "s|{{run_id}}|${RUN_ID}|g" | \
    sed "s|{{session_id}}|${SESSION_ID}|g" | \
    sed "s|{{timestamp}}|${TIMESTAMP}|g" | \
    sed "s|{{style_variants}}|${S}|g" | \
    sed "s|{{layout_variants}}|${L}|g" | \
    sed "s|{{pages_json}}|${PAGES_JSON}|g" \
    > compare.html

echo -e "${GREEN}   ✓ Generated compare.html from template${NC}"

# ============================================================================
# Generate index.html
# ============================================================================

echo -e "${YELLOW}📋 Generating index.html...${NC}"

cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UI Prototypes Index</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f5f5f5;
        }
        h1 { margin-bottom: 10px; color: #333; }
        .subtitle { color: #666; margin-bottom: 30px; }
        .cta {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .cta h2 { margin-bottom: 10px; }
        .cta a {
            display: inline-block;
            background: white;
            color: #667eea;
            padding: 10px 20px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
            margin-top: 10px;
        }
        .cta a:hover { background: #f8f9fa; }
        .style-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .style-section h2 {
            color: #495057;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }
        .target-group {
            margin-bottom: 20px;
        }
        .target-group h3 {
            color: #6c757d;
            font-size: 16px;
            margin-bottom: 10px;
        }
        .link-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }
        .prototype-link {
            padding: 12px 16px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            text-decoration: none;
            color: #495057;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s;
        }
        .prototype-link:hover {
            background: #e9ecef;
            border-color: #667eea;
            transform: translateX(2px);
        }
        .prototype-link .label { font-weight: 500; }
        .prototype-link .icon { color: #667eea; }
    </style>
</head>
<body>
    <h1>🎨 UI Prototypes Index</h1>
    <p class="subtitle">Generated __S__×__L__×__T__ = __TOTAL__ prototypes</p>

    <div class="cta">
        <h2>📊 Interactive Comparison</h2>
        <p>View all styles and layouts side-by-side in an interactive matrix</p>
        <a href="compare.html">Open Matrix View →</a>
    </div>

    <h2>📂 All Prototypes</h2>
__CONTENT__
</body>
</html>
EOF

# Build content HTML
CONTENT=""
for style in $styles; do
    CONTENT+="<div class='style-section'>"$'\n'
    CONTENT+="<h2>Style ${style}</h2>"$'\n'

    for target in $targets; do
        target_capitalized="$(echo ${target:0:1} | tr '[:lower:]' '[:upper:]')${target:1}"
        CONTENT+="<div class='target-group'>"$'\n'
        CONTENT+="<h3>${target_capitalized}</h3>"$'\n'
        CONTENT+="<div class='link-grid'>"$'\n'

        for layout in $layouts; do
            html_file="${target}-style-${style}-layout-${layout}.html"
            if [[ -f "$html_file" ]]; then
                CONTENT+="<a href='${html_file}' class='prototype-link' target='_blank'>"$'\n'
                CONTENT+="<span class='label'>Layout ${layout}</span>"$'\n'
                CONTENT+="<span class='icon'>↗</span>"$'\n'
                CONTENT+="</a>"$'\n'
            fi
        done

        CONTENT+="</div></div>"$'\n'
    done

    CONTENT+="</div>"$'\n'
done

# Calculate total
TOTAL_PROTOTYPES=$((S * L * T))

# Replace placeholders (using a temp file for complex replacement)
{
    echo "$CONTENT" > /tmp/content_tmp.txt
    sed "s|__S__|${S}|g" index.html | \
        sed "s|__L__|${L}|g" | \
        sed "s|__T__|${T}|g" | \
        sed "s|__TOTAL__|${TOTAL_PROTOTYPES}|g" | \
        sed -e "/__CONTENT__/r /tmp/content_tmp.txt" -e "/__CONTENT__/d" > /tmp/index_tmp.html
    mv /tmp/index_tmp.html index.html
    rm -f /tmp/content_tmp.txt
}

echo -e "${GREEN}   ✓ Generated index.html${NC}"

# ============================================================================
# Generate PREVIEW.md
# ============================================================================

echo -e "${YELLOW}📝 Generating PREVIEW.md...${NC}"

cat > PREVIEW.md << EOF
# UI Prototypes Preview Guide

Generated: $(date +"%Y-%m-%d %H:%M:%S")

## 📊 Matrix Dimensions

- **Styles**: ${S}
- **Layouts**: ${L}
- **Targets**: ${T}
- **Total Prototypes**: $((S*L*T))

## 🌐 How to View

### Option 1: Interactive Matrix (Recommended)

Open \`compare.html\` in your browser to see all prototypes in an interactive matrix view.

**Features**:
- Side-by-side comparison of all styles and layouts
- Switch between targets using the dropdown
- Adjust grid columns for better viewing
- Direct links to full-page views
- Selection system with export to JSON
- Fullscreen mode for detailed inspection

### Option 2: Simple Index

Open \`index.html\` for a simple list of all prototypes with direct links.

### Option 3: Direct File Access

Each prototype can be opened directly:
- Pattern: \`{target}-style-{s}-layout-{l}.html\`
- Example: \`dashboard-style-1-layout-1.html\`

## 📁 File Structure

\`\`\`
prototypes/
├── compare.html           # Interactive matrix view
├── index.html             # Simple navigation index
├── PREVIEW.md             # This file
EOF

for style in $styles; do
    for target in $targets; do
        for layout in $layouts; do
            echo "├── ${target}-style-${style}-layout-${layout}.html" >> PREVIEW.md
            echo "├── ${target}-style-${style}-layout-${layout}.css" >> PREVIEW.md
        done
    done
done

cat >> PREVIEW.md << 'EOF2'
```

## 🎨 Style Variants

EOF2

for style in $styles; do
    cat >> PREVIEW.md << EOF3
### Style ${style}

EOF3
    style_guide="../style-extraction/style-${style}/style-guide.md"
    if [[ -f "$style_guide" ]]; then
        head -n 10 "$style_guide" | tail -n +2 >> PREVIEW.md 2>/dev/null || echo "Design philosophy and tokens" >> PREVIEW.md
    else
        echo "Design system ${style}" >> PREVIEW.md
    fi
    echo "" >> PREVIEW.md
done

cat >> PREVIEW.md << 'EOF4'

## 🎯 Targets

EOF4

for target in $targets; do
    target_capitalized="$(echo ${target:0:1} | tr '[:lower:]' '[:upper:]')${target:1}"
    echo "- **${target_capitalized}**: ${L} layouts × ${S} styles = $((L*S)) variations" >> PREVIEW.md
done

cat >> PREVIEW.md << 'EOF5'

## 💡 Tips

1. **Comparison**: Use compare.html to see how different styles affect the same layout
2. **Navigation**: Use index.html for quick access to specific prototypes
3. **Selection**: Mark favorites in compare.html using star icons
4. **Export**: Download selection JSON for implementation planning
5. **Inspection**: Open browser DevTools to inspect HTML structure and CSS
6. **Sharing**: All files are standalone - can be shared or deployed directly

## 📝 Next Steps

1. Review prototypes in compare.html
2. Select preferred style × layout combinations
3. Export selections as JSON
4. Provide feedback for refinement
5. Use selected designs for implementation

---

Generated by /workflow:ui-design:generate-v2 (Style-Centric Architecture)
EOF5

echo -e "${GREEN}   ✓ Generated PREVIEW.md${NC}"

# ============================================================================
# Completion Summary
# ============================================================================

echo ""
echo -e "${GREEN}✅ Preview generation complete!${NC}"
echo -e "   Files created: compare.html, index.html, PREVIEW.md"
echo -e "   Matrix: ${S} styles × ${L} layouts × ${T} targets = $((S*L*T)) prototypes"
echo ""
echo -e "${YELLOW}🌐 Next Steps:${NC}"
echo -e "   1. Open compare.html for interactive matrix view"
echo -e "   2. Open index.html for simple navigation"
echo -e "   3. Read PREVIEW.md for detailed usage guide"
echo ""
