#!/bin/bash
# ⚠️ DEPRECATED: This script is deprecated.
# Please use: ccw tool exec convert_tokens_to_css '{"inputPath":"design-tokens.json","outputPath":"tokens.css"}'
# This file will be removed in a future version.

# Convert design-tokens.json to tokens.css with Google Fonts import and global font rules
# Usage: cat design-tokens.json | ./convert_tokens_to_css.sh > tokens.css
# Or: ./convert_tokens_to_css.sh < design-tokens.json > tokens.css

# Read JSON from stdin
json_input=$(cat)

# Extract metadata for header comment
style_name=$(echo "$json_input" | jq -r '.meta.name // "Unknown Style"' 2>/dev/null || echo "Design Tokens")

# Generate header
cat <<EOF
/* ========================================
   Design Tokens: ${style_name}
   Auto-generated from design-tokens.json
   ======================================== */

EOF

# ========================================
# Google Fonts Import Generation
# ========================================
# Extract font families and generate Google Fonts import URL
fonts=$(echo "$json_input" | jq -r '
  .typography.font_family | to_entries[] | .value
' 2>/dev/null | sed "s/'//g" | cut -d',' -f1 | sort -u)

# Build Google Fonts URL
google_fonts_url="https://fonts.googleapis.com/css2?"
font_params=""

while IFS= read -r font; do
  # Skip system fonts and empty lines
  if [[ -z "$font" ]] || [[ "$font" =~ ^(system-ui|sans-serif|serif|monospace|cursive|fantasy)$ ]]; then
    continue
  fi

  # Special handling for common web fonts with weights
  case "$font" in
    "Comic Neue")
      font_params+="family=Comic+Neue:wght@300;400;700&"
      ;;
    "Patrick Hand"|"Caveat"|"Dancing Script"|"Architects Daughter"|"Indie Flower"|"Shadows Into Light"|"Permanent Marker")
      # URL-encode font name and add common weights
      encoded_font=$(echo "$font" | sed 's/ /+/g')
      font_params+="family=${encoded_font}:wght@400;700&"
      ;;
    "Segoe Print"|"Bradley Hand"|"Chilanka")
      # These are system fonts, skip
      ;;
    *)
      # Generic font: add with default weights
      encoded_font=$(echo "$font" | sed 's/ /+/g')
      font_params+="family=${encoded_font}:wght@400;500;600;700&"
      ;;
  esac
done <<< "$fonts"

# Generate @import if we have fonts
if [[ -n "$font_params" ]]; then
  # Remove trailing &
  font_params="${font_params%&}"
  echo "/* Import Web Fonts */"
  echo "@import url('${google_fonts_url}${font_params}&display=swap');"
  echo ""
fi

# ========================================
# CSS Custom Properties Generation
# ========================================
echo ":root {"

# Colors - Brand
echo "  /* Colors - Brand */"
echo "$json_input" | jq -r '
  .colors.brand | to_entries[] |
  "  --color-brand-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Colors - Surface
echo "  /* Colors - Surface */"
echo "$json_input" | jq -r '
  .colors.surface | to_entries[] |
  "  --color-surface-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Colors - Semantic
echo "  /* Colors - Semantic */"
echo "$json_input" | jq -r '
  .colors.semantic | to_entries[] |
  "  --color-semantic-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Colors - Text
echo "  /* Colors - Text */"
echo "$json_input" | jq -r '
  .colors.text | to_entries[] |
  "  --color-text-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Colors - Border
echo "  /* Colors - Border */"
echo "$json_input" | jq -r '
  .colors.border | to_entries[] |
  "  --color-border-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Typography - Font Family
echo "  /* Typography - Font Family */"
echo "$json_input" | jq -r '
  .typography.font_family | to_entries[] |
  "  --font-family-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Typography - Font Size
echo "  /* Typography - Font Size */"
echo "$json_input" | jq -r '
  .typography.font_size | to_entries[] |
  "  --font-size-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Typography - Font Weight
echo "  /* Typography - Font Weight */"
echo "$json_input" | jq -r '
  .typography.font_weight | to_entries[] |
  "  --font-weight-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Typography - Line Height
echo "  /* Typography - Line Height */"
echo "$json_input" | jq -r '
  .typography.line_height | to_entries[] |
  "  --line-height-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Typography - Letter Spacing
echo "  /* Typography - Letter Spacing */"
echo "$json_input" | jq -r '
  .typography.letter_spacing | to_entries[] |
  "  --letter-spacing-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Spacing
echo "  /* Spacing */"
echo "$json_input" | jq -r '
  .spacing | to_entries[] |
  "  --spacing-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Border Radius
echo "  /* Border Radius */"
echo "$json_input" | jq -r '
  .border_radius | to_entries[] |
  "  --border-radius-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Shadows
echo "  /* Shadows */"
echo "$json_input" | jq -r '
  .shadows | to_entries[] |
  "  --shadow-\(.key): \(.value);"
' 2>/dev/null

echo ""

# Breakpoints
echo "  /* Breakpoints */"
echo "$json_input" | jq -r '
  .breakpoints | to_entries[] |
  "  --breakpoint-\(.key): \(.value);"
' 2>/dev/null

echo "}"
echo ""

# ========================================
# Global Font Application
# ========================================
echo "/* ========================================"
echo "   Global Font Application"
echo "   ======================================== */"
echo ""
echo "body {"
echo "  font-family: var(--font-family-body);"
echo "  font-size: var(--font-size-base);"
echo "  line-height: var(--line-height-normal);"
echo "  color: var(--color-text-primary);"
echo "  background-color: var(--color-surface-background);"
echo "}"
echo ""
echo "h1, h2, h3, h4, h5, h6, legend {"
echo "  font-family: var(--font-family-heading);"
echo "}"
echo ""
echo "/* Reset default margins for better control */"
echo "* {"
echo "  margin: 0;"
echo "  padding: 0;"
echo "  box-sizing: border-box;"
echo "}"
