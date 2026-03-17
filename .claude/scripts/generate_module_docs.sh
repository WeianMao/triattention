#!/bin/bash
# ⚠️ DEPRECATED: This script is deprecated.
# Please use: ccw tool exec generate_module_docs '{"path":".","strategy":"single-layer","tool":"gemini"}'
# This file will be removed in a future version.

# Generate documentation for modules and projects with multiple strategies
# Usage: generate_module_docs.sh <strategy> <source_path> <project_name> [tool] [model]
#   strategy: full|single|project-readme|project-architecture|http-api
#   source_path: Path to the source module directory (or project root for project-level docs)
#   project_name: Project name for output path (e.g., "myproject")
#   tool: gemini|qwen|codex (default: gemini)
#   model: Model name (optional, uses tool defaults)
#
# Default Models:
#   gemini: gemini-2.5-flash
#   qwen: coder-model
#   codex: gpt5-codex
#
# Module-Level Strategies:
#   full: Full documentation generation
#     - Read: All files in current and subdirectories (@**/*)
#     - Generate: API.md + README.md for each directory containing code files
#     - Use: Deep directories (Layer 3), comprehensive documentation
#
#   single: Single-layer documentation
#     - Read: Current directory code + child API.md/README.md files
#     - Generate: API.md + README.md only in current directory
#     - Use: Upper layers (Layer 1-2), incremental updates
#
# Project-Level Strategies:
#   project-readme: Project overview documentation
#     - Read: All module API.md and README.md files
#     - Generate: README.md (project root)
#     - Use: After all module docs are generated
#
#   project-architecture: System design documentation
#     - Read: All module docs + project README
#     - Generate: ARCHITECTURE.md + EXAMPLES.md
#     - Use: After project README is generated
#
#   http-api: HTTP API documentation
#     - Read: API route files + existing docs
#     - Generate: api/README.md
#     - Use: For projects with HTTP APIs
#
# Output Structure:
#   Module docs: .workflow/docs/{project_name}/{source_path}/API.md
#   Module docs: .workflow/docs/{project_name}/{source_path}/README.md
#   Project docs: .workflow/docs/{project_name}/README.md
#   Project docs: .workflow/docs/{project_name}/ARCHITECTURE.md
#   Project docs: .workflow/docs/{project_name}/EXAMPLES.md
#   API docs: .workflow/docs/{project_name}/api/README.md
#
# Features:
#   - Path mirroring: source structure → docs structure
#   - Template-driven generation
#   - Respects .gitignore patterns
#   - Detects code vs navigation folders
#   - Tool fallback support

# Build exclusion filters from .gitignore
build_exclusion_filters() {
    local filters=""

    # Common system/cache directories to exclude
    local system_excludes=(
        ".git" "__pycache__" "node_modules" ".venv" "venv" "env"
        "dist" "build" ".cache" ".pytest_cache" ".mypy_cache"
        "coverage" ".nyc_output" "logs" "tmp" "temp" ".workflow"
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

# Detect folder type (code vs navigation)
detect_folder_type() {
    local target_path="$1"
    local exclusion_filters="$2"

    # Count code files (primary indicators)
    local code_count=$(eval "find \"$target_path\" -maxdepth 1 -type f \\( -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.py' -o -name '*.sh' -o -name '*.go' -o -name '*.rs' \\) $exclusion_filters 2>/dev/null" | wc -l)

    if [ $code_count -gt 0 ]; then
        echo "code"
    else
        echo "navigation"
    fi
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
    local folder_type=$(detect_folder_type "$target_path" "$exclusion_filters")

    structure_info+="Directory: $dir_name\n"
    structure_info+="Total files: $total_files\n"
    structure_info+="Total directories: $total_dirs\n"
    structure_info+="Folder type: $folder_type\n\n"

    if [ "$strategy" = "full" ]; then
        # For full: show all subdirectories with file counts
        structure_info+="Subdirectories with files:\n"
        while IFS= read -r dir; do
            if [ -n "$dir" ] && [ "$dir" != "$target_path" ]; then
                local rel_path=${dir#$target_path/}
                local file_count=$(eval "find \"$dir\" -maxdepth 1 -type f $exclusion_filters 2>/dev/null" | wc -l)
                if [ $file_count -gt 0 ]; then
                    local subdir_type=$(detect_folder_type "$dir" "$exclusion_filters")
                    structure_info+="  - $rel_path/ ($file_count files, type: $subdir_type)\n"
                fi
            fi
        done < <(eval "find \"$target_path\" -type d $exclusion_filters 2>/dev/null")
    else
        # For single: show direct children only
        structure_info+="Direct subdirectories:\n"
        while IFS= read -r dir; do
            if [ -n "$dir" ]; then
                local dir_name=$(basename "$dir")
                local file_count=$(eval "find \"$dir\" -maxdepth 1 -type f $exclusion_filters 2>/dev/null" | wc -l)
                local has_api=$([ -f "$dir/API.md" ] && echo " [has API.md]" || echo "")
                local has_readme=$([ -f "$dir/README.md" ] && echo " [has README.md]" || echo "")
                structure_info+="  - $dir_name/ ($file_count files)$has_api$has_readme\n"
            fi
        done < <(eval "find \"$target_path\" -maxdepth 1 -type d $exclusion_filters 2>/dev/null" | grep -v "^$target_path$")
    fi

    # Show main file types in current directory
    structure_info+="\nCurrent directory files:\n"
    local code_files=$(eval "find \"$target_path\" -maxdepth 1 -type f \\( -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.py' -o -name '*.sh' -o -name '*.go' -o -name '*.rs' \\) $exclusion_filters 2>/dev/null" | wc -l)
    local config_files=$(eval "find \"$target_path\" -maxdepth 1 -type f \\( -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.toml' \\) $exclusion_filters 2>/dev/null" | wc -l)
    local doc_files=$(eval "find \"$target_path\" -maxdepth 1 -type f -name '*.md' $exclusion_filters 2>/dev/null" | wc -l)

    structure_info+="  - Code files: $code_files\n"
    structure_info+="  - Config files: $config_files\n"
    structure_info+="  - Documentation: $doc_files\n"

    printf "%b" "$structure_info"
}

# Calculate output path based on source path and project name
calculate_output_path() {
    local source_path="$1"
    local project_name="$2"
    local project_root="$3"

    # Get absolute path of source (normalize to Unix-style path)
    local abs_source=$(cd "$source_path" && pwd)

    # Normalize project root to same format
    local norm_project_root=$(cd "$project_root" && pwd)

    # Calculate relative path from project root
    local rel_path="${abs_source#$norm_project_root}"

    # Remove leading slash if present
    rel_path="${rel_path#/}"

    # If source is project root, use project name directly
    if [ "$abs_source" = "$norm_project_root" ] || [ -z "$rel_path" ]; then
        echo "$norm_project_root/.workflow/docs/$project_name"
    else
        echo "$norm_project_root/.workflow/docs/$project_name/$rel_path"
    fi
}

generate_module_docs() {
    local strategy="$1"
    local source_path="$2"
    local project_name="$3"
    local tool="${4:-gemini}"
    local model="$5"

    # Validate parameters
    if [ -z "$strategy" ] || [ -z "$source_path" ] || [ -z "$project_name" ]; then
        echo "❌ Error: Strategy, source path, and project name are required"
        echo "Usage: generate_module_docs.sh <strategy> <source_path> <project_name> [tool] [model]"
        echo "Module strategies: full, single"
        echo "Project strategies: project-readme, project-architecture, http-api"
        return 1
    fi

    # Validate strategy
    local valid_strategies=("full" "single" "project-readme" "project-architecture" "http-api")
    local strategy_valid=false
    for valid_strategy in "${valid_strategies[@]}"; do
        if [ "$strategy" = "$valid_strategy" ]; then
            strategy_valid=true
            break
        fi
    done

    if [ "$strategy_valid" = false ]; then
        echo "❌ Error: Invalid strategy '$strategy'"
        echo "Valid module strategies: full, single"
        echo "Valid project strategies: project-readme, project-architecture, http-api"
        return 1
    fi

    if [ ! -d "$source_path" ]; then
        echo "❌ Error: Source directory '$source_path' does not exist"
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

    # Build exclusion filters
    local exclusion_filters=$(build_exclusion_filters)

    # Get project root
    local project_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

    # Determine if this is a project-level strategy
    local is_project_level=false
    if [[ "$strategy" =~ ^project- ]] || [ "$strategy" = "http-api" ]; then
        is_project_level=true
    fi

    # Calculate output path
    local output_path
    if [ "$is_project_level" = true ]; then
        # Project-level docs go to project root
        if [ "$strategy" = "http-api" ]; then
            output_path="$project_root/.workflow/docs/$project_name/api"
        else
            output_path="$project_root/.workflow/docs/$project_name"
        fi
    else
        output_path=$(calculate_output_path "$source_path" "$project_name" "$project_root")
    fi

    # Create output directory
    mkdir -p "$output_path"

    # Detect folder type (only for module-level strategies)
    local folder_type=""
    if [ "$is_project_level" = false ]; then
        folder_type=$(detect_folder_type "$source_path" "$exclusion_filters")
    fi

    # Load templates based on strategy
    local api_template=""
    local readme_template=""
    local template_content=""

    if [ "$is_project_level" = true ]; then
        # Project-level templates
        case "$strategy" in
            project-readme)
                local proj_readme_path="$HOME/.claude/workflows/cli-templates/prompts/documentation/project-readme.txt"
                if [ -f "$proj_readme_path" ]; then
                    template_content=$(cat "$proj_readme_path")
                    echo "   📋 Loaded Project README template: $(wc -l < "$proj_readme_path") lines"
                fi
                ;;
            project-architecture)
                local arch_path="$HOME/.claude/workflows/cli-templates/prompts/documentation/project-architecture.txt"
                local examples_path="$HOME/.claude/workflows/cli-templates/prompts/documentation/project-examples.txt"
                if [ -f "$arch_path" ]; then
                    template_content=$(cat "$arch_path")
                    echo "   📋 Loaded Architecture template: $(wc -l < "$arch_path") lines"
                fi
                if [ -f "$examples_path" ]; then
                    template_content="$template_content

EXAMPLES TEMPLATE:
$(cat "$examples_path")"
                    echo "   📋 Loaded Examples template: $(wc -l < "$examples_path") lines"
                fi
                ;;
            http-api)
                local api_path="$HOME/.claude/workflows/cli-templates/prompts/documentation/api.txt"
                if [ -f "$api_path" ]; then
                    template_content=$(cat "$api_path")
                    echo "   📋 Loaded HTTP API template: $(wc -l < "$api_path") lines"
                fi
                ;;
        esac
    else
        # Module-level templates
        local api_template_path="$HOME/.claude/workflows/cli-templates/prompts/documentation/api.txt"
        local readme_template_path="$HOME/.claude/workflows/cli-templates/prompts/documentation/module-readme.txt"
        local nav_template_path="$HOME/.claude/workflows/cli-templates/prompts/documentation/folder-navigation.txt"

        if [ "$folder_type" = "code" ]; then
            if [ -f "$api_template_path" ]; then
                api_template=$(cat "$api_template_path")
                echo "   📋 Loaded API template: $(wc -l < "$api_template_path") lines"
            fi
            if [ -f "$readme_template_path" ]; then
                readme_template=$(cat "$readme_template_path")
                echo "   📋 Loaded README template: $(wc -l < "$readme_template_path") lines"
            fi
        else
            # Navigation folder uses navigation template
            if [ -f "$nav_template_path" ]; then
                readme_template=$(cat "$nav_template_path")
                echo "   📋 Loaded Navigation template: $(wc -l < "$nav_template_path") lines"
            fi
        fi
    fi

    # Scan directory structure (only for module-level strategies)
    local structure_info=""
    if [ "$is_project_level" = false ]; then
        echo "   🔍 Scanning directory structure..."
        structure_info=$(scan_directory_structure "$source_path" "$strategy")
    fi

    # Prepare logging info
    local module_name=$(basename "$source_path")

    echo "⚡ Generating docs: $source_path → $output_path"
    echo "   Strategy: $strategy | Tool: $tool | Model: $model | Type: $folder_type"
    echo "   Output: $output_path"

    # Build strategy-specific prompt
    local final_prompt=""

    # Project-level strategies
    if [ "$strategy" = "project-readme" ]; then
        final_prompt="PURPOSE: Generate comprehensive project overview documentation

PROJECT: $project_name
OUTPUT: Current directory (file will be moved to final location)

Read: @.workflow/docs/$project_name/**/*.md

Context: All module documentation files from the project

Generate ONE documentation file in current directory:
- README.md - Project root documentation

Template:
$template_content

Instructions:
- Create README.md in CURRENT DIRECTORY
- Synthesize information from all module docs
- Include project overview, getting started, and navigation
- Create clear module navigation with links
- Follow template structure exactly"

    elif [ "$strategy" = "project-architecture" ]; then
        final_prompt="PURPOSE: Generate system design and usage examples documentation

PROJECT: $project_name
OUTPUT: Current directory (files will be moved to final location)

Read: @.workflow/docs/$project_name/**/*.md

Context: All project documentation including module docs and project README

Generate TWO documentation files in current directory:
1. ARCHITECTURE.md - System architecture and design patterns
2. EXAMPLES.md - End-to-end usage examples

Template:
$template_content

Instructions:
- Create both ARCHITECTURE.md and EXAMPLES.md in CURRENT DIRECTORY
- Synthesize architectural patterns from module documentation
- Document system structure, module relationships, and design decisions
- Provide practical code examples and usage scenarios
- Follow template structure for both files"

    elif [ "$strategy" = "http-api" ]; then
        final_prompt="PURPOSE: Generate HTTP API reference documentation

PROJECT: $project_name
OUTPUT: Current directory (file will be moved to final location)

Read: @**/*.{ts,js,py,go,rs} @.workflow/docs/$project_name/**/*.md

Context: API route files and existing documentation

Generate ONE documentation file in current directory:
- README.md - HTTP API documentation (in api/ subdirectory)

Template:
$template_content

Instructions:
- Create README.md in CURRENT DIRECTORY
- Document all HTTP endpoints (routes, methods, parameters, responses)
- Include authentication requirements and error codes
- Provide request/response examples
- Follow template structure (Part B: HTTP API documentation)"

    # Module-level strategies
    elif [ "$strategy" = "full" ]; then
        # Full strategy: read all files, generate for each directory
        if [ "$folder_type" = "code" ]; then
            final_prompt="PURPOSE: Generate comprehensive API and module documentation

Directory Structure Analysis:
$structure_info

SOURCE: $source_path
OUTPUT: Current directory (files will be moved to final location)

Read: @**/*

Generate TWO documentation files in current directory:
1. API.md - Code API documentation (functions, classes, interfaces)
   Template:
$api_template

2. README.md - Module overview documentation
   Template:
$readme_template

Instructions:
- Generate both API.md and README.md in CURRENT DIRECTORY
- If subdirectories contain code files, generate their docs too (recursive)
- Work bottom-up: deepest directories first
- Follow template structure exactly
- Use structure analysis for context"
        else
            # Navigation folder - README only
            final_prompt="PURPOSE: Generate navigation documentation for folder structure

Directory Structure Analysis:
$structure_info

SOURCE: $source_path
OUTPUT: Current directory (file will be moved to final location)

Read: @**/*

Generate ONE documentation file in current directory:
- README.md - Navigation and folder overview

Template:
$readme_template

Instructions:
- Create README.md in CURRENT DIRECTORY
- Focus on folder structure and navigation
- Link to subdirectory documentation
- Use structure analysis for context"
        fi
    else
        # Single strategy: read current + child docs only
        if [ "$folder_type" = "code" ]; then
            final_prompt="PURPOSE: Generate API and module documentation for current directory

Directory Structure Analysis:
$structure_info

SOURCE: $source_path
OUTPUT: Current directory (files will be moved to final location)

Read: @*/API.md @*/README.md @*.ts @*.tsx @*.js @*.jsx @*.py @*.sh @*.go @*.rs @*.md @*.json @*.yaml @*.yml

Generate TWO documentation files in current directory:
1. API.md - Code API documentation
   Template:
$api_template

2. README.md - Module overview
   Template:
$readme_template

Instructions:
- Generate both API.md and README.md in CURRENT DIRECTORY
- Reference child documentation, do not duplicate
- Follow template structure
- Use structure analysis for current directory context"
        else
            # Navigation folder - README only
            final_prompt="PURPOSE: Generate navigation documentation

Directory Structure Analysis:
$structure_info

SOURCE: $source_path
OUTPUT: Current directory (file will be moved to final location)

Read: @*/API.md @*/README.md @*.md

Generate ONE documentation file in current directory:
- README.md - Navigation and overview

Template:
$readme_template

Instructions:
- Create README.md in CURRENT DIRECTORY
- Link to child documentation
- Use structure analysis for navigation context"
        fi
    fi

    # Execute documentation generation
    local start_time=$(date +%s)
    echo "   🔄 Starting documentation generation..."

    if cd "$source_path" 2>/dev/null; then
        local tool_result=0

        # Store current output path for CLI context
        export DOC_OUTPUT_PATH="$output_path"

        # Record git HEAD before CLI execution (to detect unwanted auto-commits)
        local git_head_before=""
        if git rev-parse --git-dir >/dev/null 2>&1; then
            git_head_before=$(git rev-parse HEAD 2>/dev/null)
        fi

        # Execute with selected tool
        case "$tool" in
            qwen)
                if [ "$model" = "coder-model" ]; then
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

        # Move generated files to output directory
        local docs_created=0
        local moved_files=""

        if [ $tool_result -eq 0 ]; then
            if [ "$is_project_level" = true ]; then
                # Project-level documentation files
                case "$strategy" in
                    project-readme)
                        if [ -f "README.md" ]; then
                            mv "README.md" "$output_path/README.md" 2>/dev/null && {
                                docs_created=$((docs_created + 1))
                                moved_files+="README.md "
                            }
                        fi
                        ;;
                    project-architecture)
                        if [ -f "ARCHITECTURE.md" ]; then
                            mv "ARCHITECTURE.md" "$output_path/ARCHITECTURE.md" 2>/dev/null && {
                                docs_created=$((docs_created + 1))
                                moved_files+="ARCHITECTURE.md "
                            }
                        fi
                        if [ -f "EXAMPLES.md" ]; then
                            mv "EXAMPLES.md" "$output_path/EXAMPLES.md" 2>/dev/null && {
                                docs_created=$((docs_created + 1))
                                moved_files+="EXAMPLES.md "
                            }
                        fi
                        ;;
                    http-api)
                        if [ -f "README.md" ]; then
                            mv "README.md" "$output_path/README.md" 2>/dev/null && {
                                docs_created=$((docs_created + 1))
                                moved_files+="api/README.md "
                            }
                        fi
                        ;;
                esac
            else
                # Module-level documentation files
                # Check and move API.md if it exists
                if [ "$folder_type" = "code" ] && [ -f "API.md" ]; then
                    mv "API.md" "$output_path/API.md" 2>/dev/null && {
                        docs_created=$((docs_created + 1))
                        moved_files+="API.md "
                    }
                fi

                # Check and move README.md if it exists
                if [ -f "README.md" ]; then
                    mv "README.md" "$output_path/README.md" 2>/dev/null && {
                        docs_created=$((docs_created + 1))
                        moved_files+="README.md "
                    }
                fi
            fi
        fi

        # Check if CLI tool auto-committed (and revert if needed)
        if [ -n "$git_head_before" ]; then
            local git_head_after=$(git rev-parse HEAD 2>/dev/null)
            if [ "$git_head_before" != "$git_head_after" ]; then
                echo "   ⚠️  Detected unwanted auto-commit by CLI tool, reverting..."
                git reset --soft "$git_head_before" 2>/dev/null
                echo "   ✅ Auto-commit reverted (files remain staged)"
            fi
        fi

        if [ $docs_created -gt 0 ]; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            echo "   ✅ Generated $docs_created doc(s) in ${duration}s: $moved_files"
            cd - > /dev/null
            return 0
        else
            echo "   ❌ Documentation generation failed for $source_path"
            cd - > /dev/null
            return 1
        fi
    else
        echo "   ❌ Cannot access directory: $source_path"
        return 1
    fi
}

# Execute function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Show help if no arguments or help requested
    if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        echo "Usage: generate_module_docs.sh <strategy> <source_path> <project_name> [tool] [model]"
        echo ""
        echo "Module-Level Strategies:"
        echo "  full                   - Generate docs for all subdirectories with code"
        echo "  single                 - Generate docs only for current directory"
        echo ""
        echo "Project-Level Strategies:"
        echo "  project-readme         - Generate project root README.md"
        echo "  project-architecture   - Generate ARCHITECTURE.md + EXAMPLES.md"
        echo "  http-api               - Generate HTTP API documentation (api/README.md)"
        echo ""
        echo "Tools: gemini (default), qwen, codex"
        echo "Models: Use tool defaults if not specified"
        echo ""
        echo "Module Examples:"
        echo "  ./generate_module_docs.sh full ./src/auth myproject"
        echo "  ./generate_module_docs.sh single ./components myproject gemini"
        echo ""
        echo "Project Examples:"
        echo "  ./generate_module_docs.sh project-readme . myproject"
        echo "  ./generate_module_docs.sh project-architecture . myproject qwen"
        echo "  ./generate_module_docs.sh http-api . myproject"
        exit 0
    fi

    generate_module_docs "$@"
fi
