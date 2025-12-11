#!/bin/bash
# ⚠️ DEPRECATED: This script is deprecated.
# Please use: ccw tool exec classify_folders '{"path":".","outputFormat":"json"}'
# This file will be removed in a future version.

# Classify folders by type for documentation generation
# Usage: get_modules_by_depth.sh | classify-folders.sh
# Output: folder_path|folder_type|code:N|dirs:N

while IFS='|' read -r depth_info path_info files_info types_info claude_info; do
  # Extract folder path from format "path:./src/modules"
  folder_path=$(echo "$path_info" | cut -d':' -f2-)

  # Skip if path extraction failed
  [[ -z "$folder_path" || ! -d "$folder_path" ]] && continue

  # Count code files (maxdepth 1)
  code_files=$(find "$folder_path" -maxdepth 1 -type f \
    \( -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.jsx" \
       -o -name "*.py" -o -name "*.go" -o -name "*.java" -o -name "*.rs" \
       -o -name "*.c" -o -name "*.cpp" -o -name "*.cs" \) \
    2>/dev/null | wc -l)

  # Count subdirectories
  subfolders=$(find "$folder_path" -maxdepth 1 -type d \
    -not -path "$folder_path" 2>/dev/null | wc -l)

  # Determine folder type
  if [[ $code_files -gt 0 ]]; then
    folder_type="code"  # API.md + README.md
  elif [[ $subfolders -gt 0 ]]; then
    folder_type="navigation"  # README.md only
  else
    folder_type="skip"  # Empty or no relevant content
  fi

  # Output classification result
  echo "${folder_path}|${folder_type}|code:${code_files}|dirs:${subfolders}"
done
