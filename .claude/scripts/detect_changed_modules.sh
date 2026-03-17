#!/bin/bash
# ⚠️ DEPRECATED: This script is deprecated.
# Please use: ccw tool exec detect_changed_modules '{"baseBranch":"main","format":"list"}'
# This file will be removed in a future version.

# Detect modules affected by git changes or recent modifications
# Usage: detect_changed_modules.sh [format]
#   format: list|grouped|paths (default: paths)
#
# Features:
# - Respects .gitignore patterns (current directory or git root)
# - Detects git changes (staged, unstaged, or last commit)
# - Falls back to recently modified files (last 24 hours)

# Build exclusion filters from .gitignore
build_exclusion_filters() {
    local filters=""

    # Common system/cache directories to exclude
    local system_excludes=(
        ".git" "__pycache__" "node_modules" ".venv" "venv" "env"
        "dist" "build" ".cache" ".pytest_cache" ".mypy_cache"
        "coverage" ".nyc_output" "logs" "tmp" "temp"
    )

    for exclude in "${system_excludes[@]}"; do
        filters+=" -not -path '*/$exclude' -not -path '*/$exclude/*'"
    done

    # Find and parse .gitignore (current dir first, then git root)
    local gitignore_file=""

    # Check current directory first
    if [ -f ".gitignore" ]; then
        gitignore_file=".gitignore"
    else
        # Try to find git root and check for .gitignore there
        local git_root=$(git rev-parse --show-toplevel 2>/dev/null)
        if [ -n "$git_root" ] && [ -f "$git_root/.gitignore" ]; then
            gitignore_file="$git_root/.gitignore"
        fi
    fi

    # Parse .gitignore if found
    if [ -n "$gitignore_file" ]; then
        while IFS= read -r line; do
            # Skip empty lines and comments
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

            # Remove trailing slash and whitespace
            line=$(echo "$line" | sed 's|/$||' | xargs)

            # Skip wildcards patterns (too complex for simple find)
            [[ "$line" =~ \* ]] && continue

            # Add to filters
            filters+=" -not -path '*/$line' -not -path '*/$line/*'"
        done < "$gitignore_file"
    fi

    echo "$filters"
}

detect_changed_modules() {
    local format="${1:-paths}"
    local changed_files=""
    local affected_dirs=""
    local exclusion_filters=$(build_exclusion_filters)

    # Step 1: Try to get git changes (staged + unstaged)
    if git rev-parse --git-dir > /dev/null 2>&1; then
        changed_files=$(git diff --name-only HEAD 2>/dev/null; git diff --name-only --cached 2>/dev/null)

        # If no changes in working directory, check last commit
        if [ -z "$changed_files" ]; then
            changed_files=$(git diff --name-only HEAD~1 HEAD 2>/dev/null)
        fi
    fi

    # Step 2: If no git changes, find recently modified source files (last 24 hours)
    # Apply exclusion filters from .gitignore
    if [ -z "$changed_files" ]; then
        changed_files=$(eval "find . -type f \( \
            -name '*.md' -o \
            -name '*.js' -o -name '*.ts' -o -name '*.jsx' -o -name '*.tsx' -o \
            -name '*.py' -o -name '*.go' -o -name '*.rs' -o \
            -name '*.java' -o -name '*.cpp' -o -name '*.c' -o -name '*.h' -o \
            -name '*.sh' -o -name '*.ps1' -o \
            -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
        \) $exclusion_filters -mtime -1 2>/dev/null")
    fi
    
    # Step 3: Extract unique parent directories
    if [ -n "$changed_files" ]; then
        affected_dirs=$(echo "$changed_files" | \
            sed 's|/[^/]*$||' | \
            grep -v '^\.$' | \
            sort -u)
        
        # Add current directory if files are in root
        if echo "$changed_files" | grep -q '^[^/]*$'; then
            affected_dirs=$(echo -e ".\n$affected_dirs" | sort -u)
        fi
    fi
    
    # Step 4: Output in requested format
    case "$format" in
        "list")
            if [ -n "$affected_dirs" ]; then
                echo "$affected_dirs" | while read dir; do
                    if [ -d "$dir" ]; then
                        local file_count=$(find "$dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
                        local depth=$(echo "$dir" | tr -cd '/' | wc -c)
                        if [ "$dir" = "." ]; then depth=0; fi
                        
                        local types=$(find "$dir" -maxdepth 1 -type f -name "*.*" 2>/dev/null | \
                                    grep -E '\.[^/]*$' | sed 's/.*\.//' | sort -u | tr '\n' ',' | sed 's/,$//')
                        local has_claude="no"
                        [ -f "$dir/CLAUDE.md" ] && has_claude="yes"
                        echo "depth:$depth|path:$dir|files:$file_count|types:[$types]|has_claude:$has_claude|status:changed"
                    fi
                done
            fi
            ;;
            
        "grouped")
            if [ -n "$affected_dirs" ]; then
                echo "📊 Affected modules by changes:"
                # Group by depth
                echo "$affected_dirs" | while read dir; do
                    if [ -d "$dir" ]; then
                        local depth=$(echo "$dir" | tr -cd '/' | wc -c)
                        if [ "$dir" = "." ]; then depth=0; fi
                        local claude_indicator=""
                        [ -f "$dir/CLAUDE.md" ] && claude_indicator=" [✓]"
                        echo "$depth:$dir$claude_indicator"
                    fi
                done | sort -n | awk -F: '
                    {
                        if ($1 != prev_depth) {
                            if (prev_depth != "") print ""
                            print "  📁 Depth " $1 ":"
                            prev_depth = $1
                        }
                        print "    - " $2 " (changed)"
                    }'
            else
                echo "📊 No recent changes detected"
            fi
            ;;
            
        "paths"|*)
            echo "$affected_dirs"
            ;;
    esac
}

# Execute function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    detect_changed_modules "$@"
fi
