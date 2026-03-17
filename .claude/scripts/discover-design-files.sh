#!/usr/bin/env bash
# ⚠️ DEPRECATED: This script is deprecated.
# Please use: ccw tool exec discover_design_files '{"sourceDir":".","outputPath":"output.json"}'
# This file will be removed in a future version.

# discover-design-files.sh - Discover design-related files and output JSON
# Usage: discover-design-files.sh <source_dir> <output_json>

set -euo pipefail

source_dir="${1:-.}"
output_json="${2:-discovered-files.json}"

# Function to find and format files as JSON array
find_files() {
  local pattern="$1"
  local files
  files=$(eval "find \"$source_dir\" -type f $pattern \
    ! -path \"*/node_modules/*\" \
    ! -path \"*/dist/*\" \
    ! -path \"*/.git/*\" \
    ! -path \"*/build/*\" \
    ! -path \"*/coverage/*\" \
    2>/dev/null | sort || true")

  local count
  if [ -z "$files" ]; then
    count=0
  else
    count=$(echo "$files" | grep -c . || echo 0)
  fi
  local json_files=""

  if [ "$count" -gt 0 ]; then
    json_files=$(echo "$files" | awk '{printf "\"%s\"%s\n", $0, (NR<'$count'?",":"")}' | tr '\n' ' ')
  fi

  echo "$count|$json_files"
}

# Discover CSS/SCSS files
css_result=$(find_files '\( -name "*.css" -o -name "*.scss" \)')
css_count=${css_result%%|*}
css_files=${css_result#*|}

# Discover JS/TS files (all framework files)
js_result=$(find_files '\( -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" -o -name "*.mjs" -o -name "*.cjs" -o -name "*.vue" -o -name "*.svelte" \)')
js_count=${js_result%%|*}
js_files=${js_result#*|}

# Discover HTML files
html_result=$(find_files '-name "*.html"')
html_count=${html_result%%|*}
html_files=${html_result#*|}

# Calculate total
total_count=$((css_count + js_count + html_count))

# Generate JSON
cat > "$output_json" << EOF
{
  "discovery_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source_directory": "$(cd "$source_dir" && pwd)",
  "file_types": {
    "css": {
      "count": $css_count,
      "files": [${css_files}]
    },
    "js": {
      "count": $js_count,
      "files": [${js_files}]
    },
    "html": {
      "count": $html_count,
      "files": [${html_files}]
    }
  },
  "total_files": $total_count
}
EOF

# Ensure file is fully written and synchronized to disk
# This prevents race conditions when the file is immediately read by another process
sync "$output_json" 2>/dev/null || sync  # Sync specific file, fallback to full sync
sleep 0.1  # Additional safety: 100ms delay for filesystem metadata update

echo "Discovered: CSS=$css_count, JS=$js_count, HTML=$html_count (Total: $total_count)" >&2
