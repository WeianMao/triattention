#!/bin/bash
# ⚠️ DEPRECATED: This script is deprecated.
# Please use: ccw tool exec update_module_claude '{"strategy":"single-layer","path":".","tool":"gemini"}'
# This file will be removed in a future version.

# Update CLAUDE.md for modules with two strategies
# Usage: update_module_claude.sh <strategy> <module_path> [tool] [model]
#   strategy: single-layer|multi-layer
#   module_path: Path to the module directory
#   tool: gemini|qwen|codex (default: gemini)
#   model: Model name (optional, uses tool defaults)
#
# Default Models:
#   gemini: gemini-2.5-flash
#   qwen: coder-model
#   codex: gpt5-codex
#
# Strategies:
#   single-layer: Upward aggregation
#     - Read: Current directory code + child CLAUDE.md files
#     - Generate: Single ./CLAUDE.md in current directory
#     - Use: Large projects, incremental bottom-up updates
#
#   multi-layer: Downward distribution
#     - Read: All files in current and subdirectories
#     - Generate: CLAUDE.md for each directory containing files
#     - Use: Small projects, full documentation generation
#
# Features:
#   - Minimal prompts based on unified template
#   - Respects .gitignore patterns
#   - Path-focused processing (script only cares about paths)
#   - Template-driven generation

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

# Scan directory structure and generate structured information
scan_directory_structure() {
    local target_path="$1"
    local strategy="$2"

    if [ ! -d "$target_path" ]; then
        echo "Directory not found: $target_path"
        return 1
    fi

    local exclusion_filters=$(build_exclusion_filters)
    local structure_info=""

    # Get basic directory info
    local dir_name=$(basename "$target_path")
    local total_files=$(eval "find \"$target_path\" -type f $exclusion_filters 2>/dev/null" | wc -l)
    local total_dirs=$(eval "find \"$target_path\" -type d $exclusion_filters 2>/dev/null" | wc -l)

    structure_info+="Directory: $dir_name\n"
    structure_info+="Total files: $total_files\n"
    structure_info+="Total directories: $total_dirs\n\n"

    if [ "$strategy" = "multi-layer" ]; then
        # For multi-layer: show all subdirectories with file counts
        structure_info+="Subdirectories with files:\n"
        while IFS= read -r dir; do
            if [ -n "$dir" ] && [ "$dir" != "$target_path" ]; then
                local rel_path=${dir#$target_path/}
                local file_count=$(eval "find \"$dir\" -maxdepth 1 -type f $exclusion_filters 2>/dev/null" | wc -l)
                if [ $file_count -gt 0 ]; then
                    structure_info+="  - $rel_path/ ($file_count files)\n"
                fi
            fi
        done < <(eval "find \"$target_path\" -type d $exclusion_filters 2>/dev/null")
    else
        # For single-layer: show direct children only
        structure_info+="Direct subdirectories:\n"
        while IFS= read -r dir; do
            if [ -n "$dir" ]; then
                local dir_name=$(basename "$dir")
                local file_count=$(eval "find \"$dir\" -maxdepth 1 -type f $exclusion_filters 2>/dev/null" | wc -l)
                local has_claude=$([ -f "$dir/CLAUDE.md" ] && echo " [has CLAUDE.md]" || echo "")
                structure_info+="  - $dir_name/ ($file_count files)$has_claude\n"
            fi
        done < <(eval "find \"$target_path\" -maxdepth 1 -type d $exclusion_filters 2>/dev/null" | grep -v "^$target_path$")
    fi

    # Show main file types in current directory
    structure_info+="\nCurrent directory files:\n"
    local code_files=$(eval "find \"$target_path\" -maxdepth 1 -type f \\( -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.py' -o -name '*.sh' \\) $exclusion_filters 2>/dev/null" | wc -l)
    local config_files=$(eval "find \"$target_path\" -maxdepth 1 -type f \\( -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.toml' \\) $exclusion_filters 2>/dev/null" | wc -l)
    local doc_files=$(eval "find \"$target_path\" -maxdepth 1 -type f -name '*.md' $exclusion_filters 2>/dev/null" | wc -l)

    structure_info+="  - Code files: $code_files\n"
    structure_info+="  - Config files: $config_files\n"
    structure_info+="  - Documentation: $doc_files\n"

    printf "%b" "$structure_info"
}

update_module_claude() {
    local strategy="$1"
    local module_path="$2"
    local tool="${3:-gemini}"
    local model="$4"

    # Validate parameters
    if [ -z "$strategy" ] || [ -z "$module_path" ]; then
        echo "❌ Error: Strategy and module path are required"
        echo "Usage: update_module_claude.sh <strategy> <module_path> [tool] [model]"
        echo "Strategies: single-layer|multi-layer"
        return 1
    fi

    # Validate strategy
    if [ "$strategy" != "single-layer" ] && [ "$strategy" != "multi-layer" ]; then
        echo "❌ Error: Invalid strategy '$strategy'"
        echo "Valid strategies: single-layer, multi-layer"
        return 1
    fi

    if [ ! -d "$module_path" ]; then
        echo "❌ Error: Directory '$module_path' does not exist"
        return 1
    fi

    # Set default models if not specified
    if [ -z "$model" ]; then
        case "$tool" in
            gemini)
                model="gemini-2.5-flash"
                ;;
            qwen)
                model="coder-model"
                ;;
            codex)
                model="gpt5-codex"
                ;;
            *)
                model=""
                ;;
        esac
    fi

    # Build exclusion filters from .gitignore
    local exclusion_filters=$(build_exclusion_filters)

    # Check if directory has files (excluding gitignored paths)
    local file_count=$(eval "find \"$module_path\" -maxdepth 1 -type f $exclusion_filters 2>/dev/null" | wc -l)
    if [ $file_count -eq 0 ]; then
        echo "⚠️  Skipping '$module_path' - no files found (after .gitignore filtering)"
        return 0
    fi

    # Use unified template for all modules
    local template_path="$HOME/.claude/workflows/cli-templates/prompts/memory/02-document-module-structure.txt"

    # Read template content directly
    local template_content=""
    if [ -f "$template_path" ]; then
        template_content=$(cat "$template_path")
        echo "   📋 Loaded template: $(wc -l < "$template_path") lines"
    else
        echo "   ⚠️  Template not found: $template_path"
        echo "   Using fallback template..."
        template_content="Create comprehensive CLAUDE.md documentation following standard structure with Purpose, Structure, Components, Dependencies, Integration, and Implementation sections."
    fi

    # Scan directory structure first
    echo "   🔍 Scanning directory structure..."
    local structure_info=$(scan_directory_structure "$module_path" "$strategy")

    # Prepare logging info
    local module_name=$(basename "$module_path")

    echo "⚡ Updating: $module_path"
    echo "   Strategy: $strategy | Tool: $tool | Model: $model | Files: $file_count"
    echo "   Template: $(basename "$template_path") ($(echo "$template_content" | wc -l) lines)"
    echo "   Structure: Scanned $(echo "$structure_info" | wc -l) lines of structure info"

    # Build minimal strategy-specific prompt with explicit paths and structure info
    local final_prompt=""

    if [ "$strategy" = "multi-layer" ]; then
        # multi-layer strategy: read all, generate for each directory
        final_prompt="Directory Structure Analysis:
$structure_info

Read: @**/*

Generate CLAUDE.md files:
- Primary: ./CLAUDE.md (current directory)
- Additional: CLAUDE.md in each subdirectory containing files

Template Guidelines:
$template_content

Instructions:
- Work bottom-up: deepest directories first
- Parent directories reference children
- Each CLAUDE.md file must be in its respective directory
- Follow the template guidelines above for consistent structure
- Use the structure analysis to understand directory hierarchy"
    else
        # single-layer strategy: read current + child CLAUDE.md, generate current only
        final_prompt="Directory Structure Analysis:
$structure_info

Read: @*/CLAUDE.md @*.ts @*.tsx @*.js @*.jsx @*.py @*.sh @*.md @*.json @*.yaml @*.yml

Generate single file: ./CLAUDE.md

Template Guidelines:
$template_content

Instructions:
- Create exactly one CLAUDE.md file in the current directory
- Reference child CLAUDE.md files, do not duplicate their content
- Follow the template guidelines above for consistent structure
- Use the structure analysis to understand the current directory context"
    fi

    # Execute update
    local start_time=$(date +%s)
    echo "   🔄 Starting update..."

    if cd "$module_path" 2>/dev/null; then
        local tool_result=0

        # Execute with selected tool
        # NOTE: Model parameter (-m) is placed AFTER the prompt
        case "$tool" in
            qwen)
                if [ "$model" = "coder-model" ]; then
                    # coder-model is default, -m is optional
                    qwen -p "$final_prompt" --yolo 2>&1
                else
                    qwen -p "$final_prompt" -m "$model" --yolo 2>&1
                fi
                tool_result=$?
                ;;
            codex)
                codex --full-auto exec "$final_prompt" -m "$model" --skip-git-repo-check -s danger-full-access 2>&1
                tool_result=$?
                ;;
            gemini)
                gemini -p "$final_prompt" -m "$model" --yolo 2>&1
                tool_result=$?
                ;;
            *)
                echo "   ⚠️  Unknown tool: $tool, defaulting to gemini"
                gemini -p "$final_prompt" -m "$model" --yolo 2>&1
                tool_result=$?
                ;;
        esac

        if [ $tool_result -eq 0 ]; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            echo "   ✅ Completed in ${duration}s"
            cd - > /dev/null
            return 0
        else
            echo "   ❌ Update failed for $module_path"
            cd - > /dev/null
            return 1
        fi
    else
        echo "   ❌ Cannot access directory: $module_path"
        return 1
    fi
}

# Execute function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Show help if no arguments or help requested
    if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        echo "Usage: update_module_claude.sh <strategy> <module_path> [tool] [model]"
        echo ""
        echo "Strategies:"
        echo "  single-layer    - Read current dir code + child CLAUDE.md, generate ./CLAUDE.md"
        echo "  multi-layer     - Read all files, generate CLAUDE.md for each directory"
        echo ""
        echo "Tools: gemini (default), qwen, codex"
        echo "Models: Use tool defaults if not specified"
        echo ""
        echo "Examples:"
        echo "  ./update_module_claude.sh single-layer ./src/auth"
        echo "  ./update_module_claude.sh multi-layer ./components gemini gemini-2.5-flash"
        exit 0
    fi

    update_module_claude "$@"
fi
