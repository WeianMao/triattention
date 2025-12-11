#!/bin/bash
# ⚠️ DEPRECATED: This script is deprecated.
# Please use: ccw tool exec get_modules_by_depth '{"format":"list","path":"."}' OR ccw tool exec get_modules_by_depth '{}'
# This file will be removed in a future version.

# Get modules organized by directory depth (deepest first)
# Usage: get_modules_by_depth.sh [format]
#   format: list|grouped|json (default: list)

# Parse .gitignore patterns and build exclusion filters
build_exclusion_filters() {
    local filters=""
    
    # Always exclude these system/cache directories and common web dev packages
    local system_excludes=(
        # Version control and IDE
        ".git" ".gitignore" ".gitmodules" ".gitattributes"
        ".svn" ".hg" ".bzr"
        ".history" ".vscode" ".idea" ".vs" ".vscode-test"
        ".sublime-text" ".atom"

        # Python
        "__pycache__" ".pytest_cache" ".mypy_cache" ".tox"
        ".coverage" "htmlcov" ".nox" ".venv" "venv" "env"
        ".egg-info" "*.egg-info" ".eggs" ".wheel"
        "site-packages" ".python-version" ".pyc"

        # Node.js/JavaScript
        "node_modules" ".npm" ".yarn" ".pnpm" "yarn-error.log"
        ".nyc_output" "coverage" ".next" ".nuxt"
        ".cache" ".parcel-cache" ".vite" "dist" "build"
        ".turbo" ".vercel" ".netlify"

        # Package managers
        ".pnpm-store" "pnpm-lock.yaml" "yarn.lock" "package-lock.json"
        ".bundle" "vendor/bundle" "Gemfile.lock"
        ".gradle" "gradle" "gradlew" "gradlew.bat"
        ".mvn" "target" ".m2"

        # Build/compile outputs
        "dist" "build" "out" "output" "_site" "public"
        ".output" ".generated" "generated" "gen"
        "bin" "obj" "Debug" "Release"

        # Testing
        ".pytest_cache" ".coverage" "htmlcov" "test-results"
        ".nyc_output" "junit.xml" "test_results"
        "cypress/screenshots" "cypress/videos"
        "playwright-report" ".playwright"

        # Logs and temp files
        "logs" "*.log" "log" "tmp" "temp" ".tmp" ".temp"
        ".env" ".env.local" ".env.*.local"
        ".DS_Store" "Thumbs.db" "*.tmp" "*.swp" "*.swo"

        # Documentation build outputs
        "_book" "_site" "docs/_build" "site" "gh-pages"
        ".docusaurus" ".vuepress" ".gitbook"

        # Database files
        "*.sqlite" "*.sqlite3" "*.db" "data.db"

        # OS and editor files
        ".DS_Store" "Thumbs.db" "desktop.ini"
        "*.stackdump" "*.core"

        # Cloud and deployment
        ".serverless" ".terraform" "terraform.tfstate"
        ".aws" ".azure" ".gcp"

        # Mobile development
        ".gradle" "build" ".expo" ".metro"
        "android/app/build" "ios/build" "DerivedData"

        # Game development
        "Library" "Temp" "ProjectSettings"
        "Logs" "MemoryCaptures" "UserSettings"
    )
    
    for exclude in "${system_excludes[@]}"; do
        filters+=" -not -path '*/$exclude' -not -path '*/$exclude/*'"
    done
    
    # Parse .gitignore if it exists
    if [ -f ".gitignore" ]; then
        while IFS= read -r line; do
            # Skip empty lines and comments
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
            
            # Remove trailing slash and whitespace
            line=$(echo "$line" | sed 's|/$||' | xargs)
            
            # Add to filters
            filters+=" -not -path '*/$line' -not -path '*/$line/*'"
        done < .gitignore
    fi
    
    echo "$filters"
}

get_modules_by_depth() {
    local format="${1:-list}"
    local exclusion_filters=$(build_exclusion_filters)
    local max_depth=$(eval "find . -type d $exclusion_filters 2>/dev/null" | awk -F/ '{print NF-1}' | sort -n | tail -1)
    
    case "$format" in
        "grouped")
            echo "📊 Modules by depth (deepest first):"
            for depth in $(seq $max_depth -1 0); do
                local dirs=$(eval "find . -mindepth $depth -maxdepth $depth -type d $exclusion_filters 2>/dev/null" | \
                           while read dir; do
                               if [ $(find "$dir" -maxdepth 1 -type f 2>/dev/null | wc -l) -gt 0 ]; then
                                   local claude_indicator=""
                                   [ -f "$dir/CLAUDE.md" ] && claude_indicator=" [✓]"
                                   echo "$dir$claude_indicator"
                               fi
                           done)
                if [ -n "$dirs" ]; then
                    echo "  📁 Depth $depth:"
                    echo "$dirs" | sed 's/^/    - /'
                fi
            done
            ;;
            
        "json")
            echo "{"
            echo "  \"max_depth\": $max_depth,"
            echo "  \"modules\": {"
            for depth in $(seq $max_depth -1 0); do
                local dirs=$(eval "find . -mindepth $depth -maxdepth $depth -type d $exclusion_filters 2>/dev/null" | \
                           while read dir; do
                               if [ $(find "$dir" -maxdepth 1 -type f 2>/dev/null | wc -l) -gt 0 ]; then
                                   local has_claude="false"
                                   [ -f "$dir/CLAUDE.md" ] && has_claude="true"
                                   echo "{\"path\":\"$dir\",\"has_claude\":$has_claude}"
                               fi
                           done | tr '\n' ',')
                if [ -n "$dirs" ]; then
                    dirs=${dirs%,}  # Remove trailing comma
                    echo "    \"$depth\": [$dirs]"
                    [ $depth -gt 0 ] && echo ","
                fi
            done
            echo "  }"
            echo "}"
            ;;
            
        "list"|*)
            # Simple list format (deepest first)
            for depth in $(seq $max_depth -1 0); do
                eval "find . -mindepth $depth -maxdepth $depth -type d $exclusion_filters 2>/dev/null" | \
                while read dir; do
                    if [ $(find "$dir" -maxdepth 1 -type f 2>/dev/null | wc -l) -gt 0 ]; then
                        local file_count=$(find "$dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
                        local types=$(find "$dir" -maxdepth 1 -type f -name "*.*" 2>/dev/null | \
                                    grep -E '\.[^/]*$' | sed 's/.*\.//' | sort -u | tr '\n' ',' | sed 's/,$//')
                        local has_claude="no"
                        [ -f "$dir/CLAUDE.md" ] && has_claude="yes"
                        echo "depth:$depth|path:$dir|files:$file_count|types:[$types]|has_claude:$has_claude"
                    fi
                done
            done
            ;;
    esac
}

# Execute function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    get_modules_by_depth "$@"
fi
