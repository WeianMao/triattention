---
name: workflow:ui-design:import-from-code
description: Import design system from code files (CSS/JS/HTML/SCSS) with automatic file discovery and parallel agent analysis
argument-hint: "[--design-id <id>] [--session <id>] [--source <path>]"
allowed-tools: Read,Write,Bash,Glob,Grep,Task,TodoWrite
auto-continue: true
---

# UI Design: Import from Code

## Overview

Extract design system tokens from source code files (CSS/SCSS/JS/TS/HTML) using parallel agent analysis. Each agent can reference any file type for cross-source token extraction, and directly generates completeness reports with findings and gaps.

**Key Characteristics**:
- Executes parallel agent analysis (3 agents: Style, Animation, Layout)
- Each agent can read ALL file types (CSS/SCSS/JS/TS/HTML) for cross-reference
- Direct completeness reporting without synthesis phase
- Graceful failure handling with detailed missing content analysis
- Returns concrete analysis results with recommendations

## Core Functionality

- **File Discovery**: Auto-discover or target specific CSS/SCSS/JS/HTML files
- **Parallel Analysis**: 3 agents extract tokens simultaneously with cross-file-type support
- **Completeness Reporting**: Each agent reports found tokens, missing content, and recommendations
- **Cross-Source Extraction**: Agents can reference any file type (e.g., Style agent can read JS theme configs)

## Usage

### Command Syntax

```bash
/workflow:ui-design:import-from-code [FLAGS]

# Flags
--design-id <id>        Design run ID to import into (must exist)
--session <id>          Session ID (uses latest design run in session)
--source <path>         Source code directory to analyze (required)
```

**Note**: All file discovery is automatic. The command will scan the source directory and find all relevant style files (CSS, SCSS, JS, HTML) automatically.

## Execution Process

```
Input Parsing:
   ├─ Parse flags: --design-id, --session, --source
   └─ Decision (base path resolution):
      ├─ --design-id provided → Exact match by design ID
      ├─ --session provided → Latest design run in session
      └─ Neither → ERROR: Must provide --design-id or --session

Phase 0: Setup & File Discovery
   ├─ Step 1: Resolve base path
   ├─ Step 2: Initialize directories
   └─ Step 3: Discover files using script

Phase 1: Parallel Agent Analysis (3 agents)
   ├─ Style Agent → design-tokens.json + code_snippets
   ├─ Animation Agent → animation-tokens.json + code_snippets
   └─ Layout Agent → layout-templates.json + code_snippets
```

### Step 1: Setup & File Discovery

**Purpose**: Initialize session, discover and categorize code files

**Operations**:

```bash
# 1. Determine base path with priority: --design-id > --session > error
if [ -n "$DESIGN_ID" ]; then
  # Exact match by design ID
  relative_path=$(find .workflow -name "${DESIGN_ID}" -type d -print -quit)
  if [ -z "$relative_path" ]; then
    echo "ERROR: Design run not found: $DESIGN_ID"
    echo "HINT: Run '/workflow:ui-design:list' to see available design runs"
    exit 1
  fi
elif [ -n "$SESSION_ID" ]; then
  # Latest in session
  relative_path=$(find .workflow/active/WFS-$SESSION_ID -name "design-run-*" -type d -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2)
  if [ -z "$relative_path" ]; then
    echo "ERROR: No design run found in session: $SESSION_ID"
    echo "HINT: Create a design run first or provide --design-id"
    exit 1
  fi
else
  echo "ERROR: Must provide --design-id or --session parameter"
  exit 1
fi

base_path=$(cd "$relative_path" && pwd)
design_id=$(basename "$base_path")

# 2. Initialize directories
source="${source:-.}"
intermediates_dir="${base_path}/.intermediates/import-analysis"
mkdir -p "$intermediates_dir"

echo "[Phase 0] File Discovery Started"
echo "  Design ID: $design_id"
echo "  Source: $source"
echo "  Output: $base_path"

# 3. Discover files using script
discovery_file="${intermediates_dir}/discovered-files.json"
ccw tool exec discover_design_files '{"sourceDir":"'"$source"'","outputPath":"'"$discovery_file"'"}'

echo "  Output: $discovery_file"
```

<!-- TodoWrite: Initialize todo list -->

**TodoWrite**:
```json
[
  {"content": "Phase 0: 发现和分类代码文件", "status": "in_progress", "activeForm": "发现代码文件"},
  {"content": "Phase 1.1: Style Agent - 提取视觉token及代码片段 (design-tokens.json + code_snippets)", "status": "pending", "activeForm": "提取视觉token"},
  {"content": "Phase 1.2: Animation Agent - 提取动画token及代码片段 (animation-tokens.json + code_snippets)", "status": "pending", "activeForm": "提取动画token"},
  {"content": "Phase 1.3: Layout Agent - 提取布局模式及代码片段 (layout-templates.json + code_snippets)", "status": "pending", "activeForm": "提取布局模式"}
]
```

**File Discovery Behavior**:

- **Automatic discovery**: Intelligently scans source directory for all style-related files
- **Supported file types**: CSS, SCSS, JavaScript, TypeScript, HTML
- **Smart filtering**: Finds theme-related JS/TS files (e.g., tailwind.config.js, theme.js, styled-components)
- **Exclusions**: Automatically excludes `node_modules/`, `dist/`, `.git/`, and build directories
- **Output**: Single JSON file `discovered-files.json` in `.intermediates/import-analysis/`
  - Structure: `{ "css": [...], "js": [...], "html": [...], "counts": {...}, "discovery_time": "..." }`
  - Generated via bash commands using `find` + JSON formatting

<!-- TodoWrite: Update Phase 0 → completed, Phase 1.1-1.3 → in_progress (all 3 agents in parallel) -->

---

### Step 2: Parallel Agent Analysis

**Purpose**: Three agents analyze all file types in parallel, each producing completeness-report.json

**Operations**:
- **Style Agent**: Extracts visual tokens (colors, typography, spacing) from ALL files (CSS/SCSS/JS/HTML)
- **Animation Agent**: Extracts animations/transitions from ALL files
- **Layout Agent**: Extracts layout patterns/component structures from ALL files

**Validation**:
- Each agent can reference any file type (not restricted to single type)
- Direct output: Each agent generates completeness-report.json with findings + missing content
- No synthesis needed: Agents produce final output directly

```bash
echo "[Phase 1] Starting parallel agent analysis (3 agents)"
```

#### Style Agent Task (design-tokens.json, style-guide.md)

**Agent Task**:

```javascript
Task(subagent_type="ui-design-agent",
     prompt="[STYLE_TOKENS_EXTRACTION]
  Extract visual design tokens from code files using code import extraction pattern.

  MODE: style-extraction | SOURCE: ${source} | BASE_PATH: ${base_path}

  ## Input Files

  **Discovered Files**: ${intermediates_dir}/discovered-files.json
  $(cat \"${intermediates_dir}/discovered-files.json\" 2>/dev/null | grep -E '(count|files)' | head -30)

  ## Code Import Extraction Strategy

  **Step 0: Fast Conflict Detection** (Use Bash/Grep for quick global scan)
  - Quick scan: \`rg --color=never -n "^\\s*--primary:|^\\s*--secondary:|^\\s*--accent:" --type css ${source}\` to find core color definitions with line numbers
  - Semantic search: \`rg --color=never -B3 -A1 "^\\s*--primary:" --type css ${source}\` to capture surrounding context and comments
  - Core token scan: Search for --primary, --secondary, --accent, --background patterns to detect all theme-critical definitions
  - Pattern: rg → Extract values → Compare → If different → Read full context with comments → Record conflict
  - Alternative (if many files): Execute CLI analysis for comprehensive report:
    \`\`\`bash
    cd ${source} && gemini -p \"
    PURPOSE: Detect color token conflicts across all CSS/SCSS/JS files
    TASK: • Scan all files for color definitions • Identify conflicting values • Extract semantic comments
    MODE: analysis
    CONTEXT: @**/*.css @**/*.scss @**/*.js @**/*.ts
    EXPECTED: JSON report listing conflicts with file:line, values, semantic context
    RULES: Focus on core tokens | Report ALL variants | analysis=READ-ONLY
    \"
    \`\`\`

  **Step 1: Load file list**
  - Read(${intermediates_dir}/discovered-files.json)
  - Extract: file_types.css.files, file_types.js.files, file_types.html.files

  **Step 2: Cross-source token extraction**
  - CSS/SCSS: Colors, typography, spacing, shadows, borders
  - JavaScript/TypeScript: Theme configs (Tailwind, styled-components, CSS-in-JS)
  - HTML: Inline styles, usage patterns

  **Step 3: Validation and Conflict Detection**
  - Report missing tokens WITHOUT inference (mark as "missing" in _metadata.completeness)
  - Detect and report inconsistent values across files (list ALL variants with file:line sources)
  - Report missing categories WITHOUT auto-filling (document gaps for manual review)
  - CRITICAL: Verify core tokens (primary, secondary, accent) against semantic comments in source code

  ## Output Files

  **Target Directory**: ${base_path}/style-extraction/style-1/

  **Files to Generate**:
  1. **design-tokens.json**
     - Follow [DESIGN_SYSTEM_GENERATION_TASK] standard token structure
     - Add \"_metadata.extraction_source\": \"code_import\"
     - Add \"_metadata.files_analyzed\": {css, js, html file lists}
     - Add \"_metadata.completeness\": {status, missing_categories, recommendations}
     - Add \"_metadata.conflicts\": Array of conflicting definitions (MANDATORY if conflicts exist)
     - Add \"_metadata.code_snippets\": Map of code snippets (see below)
     - Add \"_metadata.usage_recommendations\": Usage patterns from code (see below)
     - Include \"source\" field for each token (e.g., \"file.css:23\")

  **Code Snippet Recording**:
  - For each extracted token, record the actual code snippet in `_metadata.code_snippets`
  - Structure:
    ```json
    \"code_snippets\": {
      \"file.css:23\": {
        \"lines\": \"23-27\",
        \"snippet\": \":root {\\n  --color-primary: oklch(0.5555 0.15 270);\\n  /* Primary brand color */\\n  --color-primary-hover: oklch(0.6 0.15 270);\\n}\",
        \"context\": \"css-variable\"
      }
    }
    ```
  - Context types: \"css-variable\" | \"css-class\" | \"js-object\" | \"js-theme-config\" | \"inline-style\"
  - Record complete code blocks with all dependencies and relevant comments
  - Typical ranges: Simple declarations (1-5 lines), Utility classes (5-15 lines), Complete configs (15-50 lines)
  - Preserve original formatting and indentation

  **Conflict Detection and Reporting**:
  - When the same token is defined differently across multiple files, record in `_metadata.conflicts`
  - Follow Agent schema for conflicts array structure (see ui-design-agent.md)
  - Each conflict MUST include: token_name, category, all definitions with context, selected_value, selection_reason
  - Selection priority:
    1. Definitions with semantic comments explaining intent (/* Blue theme */, /* Primary brand color */)
    2. Definitions that align with overall color scheme described in comments
    3. When in doubt, report ALL variants and flag for manual review in completeness.recommendations

  **Usage Recommendations Generation**:
  - Analyze code usage patterns to extract `_metadata.usage_recommendations` (see ui-design-agent.md schema)
  - **Typography recommendations**:
    * `common_sizes`: Identify most frequent font size usage (e.g., \"body_text\": \"base (1rem)\")
    * `common_combinations`: Extract heading+body pairings from actual usage (e.g., h1 with p tags)
  - **Spacing recommendations**:
    * `size_guide`: Categorize spacing values into tight/normal/loose based on frequency
    * `common_patterns`: Extract frequent padding/margin combinations from components
  - Analysis method: Scan code for class/style usage frequency, extract patterns from component implementations
  - Optional: If insufficient usage data, mark fields as empty arrays/objects with note in completeness.recommendations

  ## Code Import Specific Requirements
  - ✅ Read discovered-files.json FIRST to get file paths
  - ✅ Track extraction source for each token (file:line)
  - ✅ Record complete code snippets in _metadata.code_snippets (complete blocks with dependencies/comments)
  - ✅ Include completeness assessment in _metadata
  - ✅ Report inconsistent values with ALL source locations in _metadata.conflicts (DO NOT auto-normalize or choose)
  - ✅ CRITICAL: Verify core theme tokens (primary, secondary, accent) match source code semantic intent
  - ✅ When conflicts exist, prefer definitions with semantic comments explaining intent
  - ❌ NO inference, NO smart filling, NO automatic conflict resolution
  - ❌ NO external research or web searches (code-only extraction)
")
```

#### Animation Agent Task (animation-tokens.json, animation-guide.md)

**Agent Task**:

```javascript
Task(subagent_type="ui-design-agent",
     prompt="[ANIMATION_TOKEN_GENERATION_TASK]
  Extract animation tokens from code files using code import extraction pattern.

  MODE: animation-extraction | SOURCE: ${source} | BASE_PATH: ${base_path}

  ## Input Files

  **Discovered Files**: ${intermediates_dir}/discovered-files.json
  $(cat \"${intermediates_dir}/discovered-files.json\" 2>/dev/null | grep -E '(count|files)' | head -30)

  ## Code Import Extraction Strategy

  **Step 0: Fast Animation Discovery** (Use Bash/Grep for quick pattern detection)
  - Quick scan: \`rg --color=never -n "@keyframes|animation:|transition:" --type css ${source}\` to find animation definitions with line numbers
  - Framework detection: \`rg --color=never "framer-motion|gsap|@react-spring|react-spring" --type js --type ts ${source}\` to detect animation frameworks
  - Pattern categorization: \`rg --color=never -B2 -A5 "@keyframes" --type css ${source}\` to extract keyframe animations with context
  - Pattern: rg → Identify animation types → Map framework usage → Prioritize extraction targets
  - Alternative (if complex framework mix): Execute CLI analysis for comprehensive report:
    \`\`\`bash
    cd ${source} && gemini -p \"
    PURPOSE: Detect animation frameworks and patterns
    TASK: • Identify frameworks • Map animation patterns • Categorize by complexity
    MODE: analysis
    CONTEXT: @**/*.css @**/*.scss @**/*.js @**/*.ts
    EXPECTED: JSON report listing frameworks, animation types, file locations
    RULES: Focus on framework consistency | Map all animations | analysis=READ-ONLY
    \"
    \`\`\`

  **Step 1: Load file list**
  - Read(${intermediates_dir}/discovered-files.json)
  - Extract: file_types.css.files, file_types.js.files, file_types.html.files

  **Step 2: Cross-source animation extraction**
  - CSS/SCSS: @keyframes, transitions, animation properties
  - JavaScript/TypeScript: Animation frameworks (Framer Motion, GSAP), CSS-in-JS
  - HTML: Inline styles, data-animation attributes

  **Step 3: Framework detection & normalization**
  - Detect animation frameworks used (css-animations | framer-motion | gsap | none)
  - Normalize into semantic token system
  - Cross-reference CSS animations with JS configs

  ## Output Files

  **Target Directory**: ${base_path}/animation-extraction/

  **Files to Generate**:
  1. **animation-tokens.json**
     - Follow [ANIMATION_TOKEN_GENERATION_TASK] standard structure
     - Add \"_metadata.framework_detected\"
     - Add \"_metadata.files_analyzed\"
     - Add \"_metadata.completeness\"
     - Add \"_metadata.code_snippets\": Map of code snippets (same format as Style Agent)
     - Include \"source\" field for each token

  **Code Snippet Recording**:
  - Record actual animation/transition code in `_metadata.code_snippets`
  - Context types: \"css-keyframes\" | \"css-transition\" | \"js-animation\" | \"framer-motion\" | \"gsap\"
  - Record complete blocks: @keyframes animations (10-30 lines), transition configs (5-15 lines), JS animation objects (15-50 lines)
  - Include all animation steps, timing functions, and related comments
  - Preserve original formatting and framework-specific syntax

  ## Code Import Specific Requirements
  - ✅ Read discovered-files.json FIRST to get file paths
  - ✅ Detect animation framework if present
  - ✅ Track extraction source for each token (file:line)
  - ✅ Record complete code snippets in _metadata.code_snippets (complete animation blocks with all steps/timing)
  - ✅ Normalize framework-specific syntax into standard tokens
  - ❌ NO external research or web searches (code-only extraction)
")
```

#### Layout Agent Task (layout-templates.json, layout-guide.md)

**Agent Task**:

```javascript
Task(subagent_type="ui-design-agent",
     prompt="[LAYOUT_TEMPLATE_GENERATION_TASK]
  Extract layout patterns from code files using code import extraction pattern.

  MODE: layout-extraction | SOURCE: ${source} | BASE_PATH: ${base_path}

  ## Input Files

  **Discovered Files**: ${intermediates_dir}/discovered-files.json
  $(cat \"${intermediates_dir}/discovered-files.json\" 2>/dev/null | grep -E '(count|files)' | head -30)

  ## Code Import Extraction Strategy

  **Step 0: Fast Component Discovery** (Use Bash/Grep for quick component scan)
  - Layout pattern scan: \`rg --color=never -n "display:\\s*(grid|flex)|grid-template" --type css ${source}\` to find layout systems
  - Component class scan: \`rg --color=never "class.*=.*\\"[^\"]*\\b(btn|button|card|input|modal|dialog|dropdown)" --type html --type js --type ts ${source}\` to identify UI components
  - Universal component heuristic: Components appearing in 3+ files = universal, <3 files = specialized
  - Pattern: rg → Count occurrences → Classify by frequency → Prioritize universal components
  - Alternative (if large codebase): Execute CLI analysis for comprehensive categorization:
    \`\`\`bash
    cd ${source} && gemini -p \"
    PURPOSE: Classify components as universal vs specialized
    TASK: • Identify UI components • Classify reusability • Map layout systems
    MODE: analysis
    CONTEXT: @**/*.css @**/*.scss @**/*.js @**/*.ts @**/*.html
    EXPECTED: JSON report categorizing components, layout patterns, naming conventions
    RULES: Focus on component reusability | Identify layout systems | analysis=READ-ONLY
    \"
    \`\`\`

  **Step 1: Load file list**
  - Read(${intermediates_dir}/discovered-files.json)
  - Extract: file_types.css.files, file_types.js.files, file_types.html.files

  **Step 2: Cross-source layout extraction**
  - CSS/SCSS: Grid systems, flexbox utilities, layout classes, media queries
  - JavaScript/TypeScript: Layout components (React/Vue), grid configs
  - HTML: Semantic structure, component hierarchies

  **Component Classification** (MUST annotate in extraction):
  - **Universal Components**: Reusable multi-component templates (buttons, inputs, cards, modals, etc.)
  - **Specialized Components**: Module-specific components from code (feature-specific layouts, custom widgets, domain components)

  **Step 3: System identification**
  - Detect naming convention (BEM | SMACSS | utility-first | css-modules)
  - Identify layout system (12-column | flexbox | css-grid | custom)
  - Extract responsive strategy and breakpoints

  ## Output Files

  **Target Directory**: ${base_path}/layout-extraction/

  **Files to Generate**:

  1. **layout-templates.json**
     - Follow [LAYOUT_TEMPLATE_GENERATION_TASK] standard structure
     - Add \"extraction_metadata\" section:
       * extraction_source: \"code_import\"
       * naming_convention: detected convention
       * layout_system: {type, confidence, source_files}
       * responsive: {breakpoints, mobile_first, source}
       * completeness: {status, missing_items, recommendations}
       * code_snippets: Map of code snippets (same format as Style Agent)
     - For each component in \"layout_templates\":
       * Include \"source\" field (file:line)
       * **Include \"component_type\" field: \"universal\" | \"specialized\"**
       * dom_structure with semantic HTML5
       * css_layout_rules using var() placeholders
       * Add \"description\" field explaining component purpose and classification rationale
       * **Add \"usage_guide\" field for universal components** (see ui-design-agent.md schema):
         - common_sizes: Extract size variants (small/medium/large) from code
         - variant_recommendations: Document when to use each variant (primary/secondary/etc)
         - usage_context: List typical usage scenarios from actual implementation
         - accessibility_tips: Extract ARIA patterns and a11y notes from code

  **Code Snippet Recording**:
  - Record actual layout/component code in `extraction_metadata.code_snippets`
  - Context types: \"css-grid\" | \"css-flexbox\" | \"css-utility\" | \"html-structure\" | \"react-component\"
  - Record complete blocks: Utility classes (5-15 lines), HTML structures (10-30 lines), React components (20-100 lines)
  - For components: include HTML structure + associated CSS rules + component logic
  - Preserve original formatting and framework-specific syntax

  ## Code Import Specific Requirements
  - ✅ Read discovered-files.json FIRST to get file paths
  - ✅ Detect and document naming conventions
  - ✅ Identify layout system with confidence level
  - ✅ Extract component variants and states from usage patterns
  - ✅ **Classify each component as \"universal\" or \"specialized\"** based on:
    * Universal: Reusable across multiple features (buttons, inputs, cards, modals)
    * Specialized: Feature-specific or domain-specific (checkout form, dashboard widget)
  - ✅ Record complete code snippets in extraction_metadata.code_snippets (complete components/structures)
  - ✅ **Document classification rationale** in component description
  - ✅ **Generate usage_guide for universal components** (REQUIRED):
    * Analyze code to extract size variants (scan for size-related classes/props)
    * Document variant usage from code comments and implementation patterns
    * List usage contexts from component instances in codebase
    * Extract accessibility patterns from ARIA attributes and a11y comments
    * If insufficient data, populate with minimal valid structure and note in completeness
  - ❌ NO external research or web searches (code-only extraction)
")
```

**Wait for All Agents**:

```bash
# Note: Agents run in parallel and write separate completeness reports
# Each agent generates its own completeness-report.json directly
# No synthesis phase needed
echo "[Phase 1] Parallel agent analysis complete"
```

<!-- TodoWrite: Update Phase 1.1-1.3 → completed (all 3 agents complete together) -->

---

## Output Files

### Generated Files

**Location**: `${base_path}/`

**Directory Structure**:
```
${base_path}/
├── style-extraction/
│   └── style-1/
│       └── design-tokens.json       # Production-ready design tokens with code snippets
├── animation-extraction/
│   └── animation-tokens.json        # Animation/transition tokens with code snippets
├── layout-extraction/
│   └── layout-templates.json        # Layout patterns with code snippets
└── .intermediates/
    └── import-analysis/
        └── discovered-files.json    # All discovered files (JSON format)
```

**Files**:
1. **style-extraction/style-1/design-tokens.json**
   - Production-ready design tokens
   - Categories: colors, typography, spacing, opacity, border_radius, shadows, breakpoints
   - Metadata: extraction_source, files_analyzed, completeness assessment, **code_snippets**
   - **Code snippets**: Complete code blocks from source files (CSS variables, theme configs, inline styles)

2. **animation-extraction/animation-tokens.json**
   - Animation tokens: duration, easing, transitions, keyframes, interactions
   - Framework detection: css-animations, framer-motion, gsap, etc.
   - Metadata: extraction_source, completeness assessment, **code_snippets**
   - **Code snippets**: Complete animation blocks (@keyframes, transition configs, JS animations)

3. **layout-extraction/layout-templates.json**
   - Layout templates for each discovered component
   - Extraction metadata: naming_convention, layout_system, responsive strategy, **code_snippets**
   - Component patterns with DOM structure and CSS rules
   - **Code snippets**: Complete component/structure code (HTML, CSS utilities, React components)

**Intermediate Files**: `.intermediates/import-analysis/`
- `discovered-files.json` - All discovered files in JSON format with counts and metadata

---

## Error Handling

### Common Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| No files discovered | Wrong --source path or no style files in directory | Verify --source parameter points to correct directory with style files |
| Agent reports "failed" status | No tokens found in any file | Review file content, check if files contain design tokens |
| Empty completeness reports | Files exist but contain no extractable tokens | Manually verify token definitions in source files |
| Missing file type | File extensions not recognized | Check that files use standard extensions (.css, .scss, .js, .ts, .html) |

---

## Best Practices

1. **Point to the right directory**: Use `--source` to specify the directory containing your style files (e.g., `./src`, `./app`, `./styles`)
2. **Let automatic discovery work**: The command will find all relevant files - no need to specify patterns
3. **Specify target design run**: Use `--design-id` for existing design run or `--session` to use session's latest design run (one of these is required)
4. **Cross-reference agent reports**: Compare all three completeness reports (style, animation, layout) to identify gaps
5. **Review missing content**: Check `_metadata.completeness` field in reports for actionable improvements
6. **Verify file discovery**: Check `${base_path}/.intermediates/import-analysis/discovered-files.json` if agents report no data
