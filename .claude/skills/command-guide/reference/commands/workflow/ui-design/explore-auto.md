---
name: explore-auto
description: Interactive exploratory UI design workflow with style-centric batch generation, creates design variants from prompts/images with parallel execution and user selection
argument-hint: "[--input "<value>"] [--targets "<list>"] [--target-type "page|component"] [--session <id>] [--style-variants <count>] [--layout-variants <count>]"
allowed-tools: SlashCommand(*), TodoWrite(*), Read(*), Bash(*), Glob(*), Write(*), Task(conceptual-planning-agent)
---

# UI Design Auto Workflow Command

## Overview & Execution Model

**Fully autonomous orchestrator**: Executes all design phases sequentially from style extraction to design integration, with optional batch planning.

**Unified Target System**: Generates `style_variants × layout_variants × targets` prototypes, where targets can be:
- **Pages** (full-page layouts): home, dashboard, settings, etc.
- **Components** (isolated UI elements): navbar, card, hero, form, etc.
- **Mixed**: Can combine both in a single workflow

**Autonomous Flow** (⚠️ CONTINUOUS EXECUTION - DO NOT STOP):
1. User triggers: `/workflow:ui-design:explore-auto [params]`
2. Phase 5: Target confirmation → User confirms → **IMMEDIATELY triggers Phase 7**
3. Phase 7 (style-extract) → **Attach tasks → Execute → Collapse** → Auto-continues to Phase 8
4. Phase 8 (animation-extract, conditional):
   - **IF should_extract_animation**: **Attach tasks → Execute → Collapse** → Auto-continues to Phase 9
   - **ELSE**: Skip (use code import) → Auto-continues to Phase 9
5. Phase 9 (layout-extract) → **Attach tasks → Execute → Collapse** → Auto-continues to Phase 10
6. **Phase 10 (ui-assembly)** → **Attach tasks → Execute → Collapse** → Workflow complete

**Phase Transition Mechanism**:
- **Phase 5 (User Interaction)**: User confirms targets → IMMEDIATELY dispatches Phase 7
- **Phase 7-10 (Autonomous)**: SlashCommand dispatch **ATTACHES** tasks to current workflow
- **Task Execution**: Orchestrator **EXECUTES** these attached tasks itself
- **Task Collapse**: After tasks complete, collapse them into phase summary
- **Phase Transition**: Automatically execute next phase after collapsing
- No additional user interaction after Phase 5 confirmation

**Auto-Continue Mechanism**: TodoWrite tracks phase status with dynamic task attachment/collapse. After executing all attached tasks, you MUST immediately collapse them, restore phase summary, and execute the next phase. No user intervention required. The workflow is NOT complete until Phase 10 (UI assembly) finishes.

**Task Attachment Model**: SlashCommand dispatch is NOT delegation - it's task expansion. The orchestrator executes these attached tasks itself, not waiting for external completion.

**Target Type Detection**: Automatically inferred from prompt/targets, or explicitly set via `--target-type`.

## Execution Process

```
Input Parsing:
   ├─ Parse flags: --input, --targets, --target-type, --device-type, --session, --style-variants, --layout-variants
   └─ Decision (input detection):
      ├─ Contains * or glob matches → images_input (visual)
      ├─ File/directory exists → code import source
      └─ Pure text → design prompt

Phase 1-4: Parameter Parsing & Initialization
   ├─ Phase 1: Normalize parameters (legacy deprecation warning)
   ├─ Phase 2: Intelligent prompt parsing (extract variant counts)
   ├─ Phase 3: Device type inference (explicit > keywords > target_type > default)
   └─ Phase 4: Run initialization and directory setup

Phase 5: Unified Target Inference
   ├─ Priority: --pages/--components (legacy) → --targets → prompt analysis → synthesis → default
   ├─ Display confirmation with modification options
   └─ User confirms → IMMEDIATELY triggers Phase 7

Phase 6: Code Import (Conditional)
   └─ Decision (design_source):
      ├─ code_only | hybrid → Dispatch /workflow:ui-design:import-from-code
      └─ visual_only → Skip to Phase 7

Phase 7: Style Extraction
   └─ Decision (needs_visual_supplement):
      ├─ visual_only OR supplement needed → Dispatch /workflow:ui-design:style-extract
      └─ code_only AND style_complete → Use code import

Phase 8: Animation Extraction
   └─ Decision (should_extract_animation):
      ├─ visual_only OR incomplete OR regenerate → Dispatch /workflow:ui-design:animation-extract
      └─ code_only AND animation_complete → Use code import

Phase 9: Layout Extraction
   └─ Decision (needs_visual_supplement OR NOT layout_complete):
      ├─ True → Dispatch /workflow:ui-design:layout-extract
      └─ False → Use code import

Phase 10: UI Assembly
   └─ Dispatch /workflow:ui-design:generate → Workflow complete
```

## Core Rules

1. **Start Immediately**: TodoWrite initialization → Phase 7 execution
2. **No Preliminary Validation**: Sub-commands handle their own validation
3. **Parse & Pass**: Extract data from each output for next phase
4. **Default to All**: When selecting variants/prototypes, use ALL generated items
5. **Track Progress**: Update TodoWrite dynamically with task attachment/collapse pattern
6. **⚠️ CRITICAL: Task Attachment Model** - SlashCommand dispatch **ATTACHES** tasks to current workflow. Orchestrator **EXECUTES** these attached tasks itself, not waiting for external completion. This is NOT delegation - it's task expansion.
7. **⚠️ CRITICAL: DO NOT STOP** - This is a continuous multi-phase workflow. After executing all attached tasks, you MUST immediately collapse them and execute the next phase. Workflow is NOT complete until Phase 10 (UI assembly) finishes.

## Parameter Requirements

**Recommended Parameter**:
- `--input "<value>"`: Unified input source (auto-detects type)
  - **Glob pattern** (images): `"design-refs/*"`, `"screenshots/*.png"`
  - **File/directory path** (code): `"./src/components"`, `"/path/to/styles"`
  - **Text description** (prompt): `"modern dashboard with 3 styles"`, `"minimalist design"`
  - **Combination**: `"design-refs/* modern dashboard"` (glob + description)
  - Multiple inputs: Separate with `|` → `"design-refs/*|modern style"`

**Detection Logic**:
- Contains `*` or matches existing files → **glob pattern** (images)
- Existing file/directory path → **code import**
- Pure text without paths → **design prompt**
- Contains `|` separator → **multiple inputs** (glob|prompt or path|prompt)

**Legacy Parameters** (deprecated, use `--input` instead):
- `--images "<glob>"`: Reference image paths (shows deprecation warning)
- `--prompt "<description>"`: Design description (shows deprecation warning)

**Optional Parameters** (all have smart defaults):
- `--targets "<list>"`: Comma-separated targets (pages/components) to generate (inferred from prompt/session if omitted)
- `--target-type "page|component|auto"`: Explicitly set target type (default: `auto` - intelligent detection)
- `--device-type "desktop|mobile|tablet|responsive|auto"`: Device type for layout optimization (default: `auto` - intelligent detection)
  - **Desktop**: 1920×1080px - Mouse-driven, spacious layouts
  - **Mobile**: 375×812px - Touch-friendly, compact layouts
  - **Tablet**: 768×1024px - Hybrid touch/mouse layouts
  - **Responsive**: 1920×1080px base with mobile-first breakpoints
- `--session <id>`: Workflow session ID (standalone mode if omitted)
- `--style-variants <count>`: Style variants (default: inferred from prompt or 3, range: 1-5)
- `--layout-variants <count>`: Layout variants per style (default: inferred or 3, range: 1-5)

**Legacy Target Parameters** (maintained for backward compatibility):
- `--pages "<list>"`: Alias for `--targets` with `--target-type page`
- `--components "<list>"`: Alias for `--targets` with `--target-type component`

**Input Rules**:
- Must provide: `--input` OR (legacy: `--images`/`--prompt`) OR `--targets`
- `--input` can combine multiple input types
- If `--targets` not provided, intelligently inferred from prompt/session

**Supported Target Types**:
- **Pages** (full layouts): home, dashboard, settings, profile, login, etc.
- **Components** (UI elements):
  - Navigation: navbar, header, menu, breadcrumb, tabs, sidebar
  - Content: hero, card, list, table, grid, timeline
  - Input: form, search, filter, input-group
  - Feedback: modal, alert, toast, badge, progress
  - Media: gallery, carousel, video-player, image-card
  - Other: footer, pagination, dropdown, tooltip, avatar

**Intelligent Prompt Parsing**: Extracts variant counts from natural language:
- "Generate **3 style variants**" → `--style-variants 3`
- "**2 layout options**" → `--layout-variants 2`
- "Create **4 styles** with **2 layouts each**" → `--style-variants 4 --layout-variants 2`
- Explicit flags override prompt inference

## Execution Modes

**Matrix Mode** (style-centric):
- Generates `style_variants × layout_variants × targets` prototypes
- **Phase 1**: `style_variants` complete design systems (extract)
- **Phase 2**: Layout templates extraction (layout-extract)
- **Phase 3**: Style-centric batch generation (generate)
  - Sub-phase 1: `targets × layout_variants` target-specific layout plans
  - **Sub-phase 2**: `S` style-centric agents (each handles `L×T` combinations)
  - Sub-phase 3: `style_variants × layout_variants × targets` final prototypes
  - Performance: Efficient parallel execution with S agents
  - Quality: HTML structure adapts to design_attributes
  - Pages: Full-page layouts with complete structure
  - Components: Isolated elements with minimal wrapper

**Integrated vs. Standalone**:
- `--session` flag determines session integration or standalone execution

## 10-Phase Execution

### Phase 1: Parameter Parsing & Input Detection

**Unified Principle**: Detect → Classify → Store (avoid string concatenation and escaping)

**Step 1: Parameter Normalization**
```bash
# Legacy parameters (deprecated)
IF --images OR --prompt:
    WARN: "⚠️ --images/--prompt deprecated. Use --input"
    images_input = --images; prompt_text = --prompt

# Unified --input (split by "|")
ELSE IF --input:
    FOR part IN split(--input, "|"):
        IF "*" IN part OR glob_exists(part): images_input = part
        ELSE IF path_exists(part): prompt_text += part
        ELSE: prompt_text += part
```

**Step 2: Design Source Detection**
```bash
code_base_path = extract_first_valid_path(prompt_text)
has_visual_input = (images_input AND glob_exists(images_input))

design_source = classify_source(code_base_path, has_visual_input):
    • code + visual → "hybrid"
    • code only → "code_only"
    • visual/prompt → "visual_only"
    • none → ERROR
```

**Stored Variables**: `design_source`, `code_base_path`, `has_visual_input`, `images_input`, `prompt_text`

---

### Phase 2: Intelligent Prompt Parsing

**Unified Principle**: explicit > inferred > default

```bash
# Variant counts (priority chain)
style_variants = --style-variants OR extract_number(prompt_text, "style") OR 3
layout_variants = --layout-variants OR extract_number(prompt_text, "layout") OR 3

VALIDATE: 1 ≤ variants ≤ 5
```

**Stored Variables**: `style_variants`, `layout_variants`

---

### Phase 3: Device Type Inference

**Unified Principle**: explicit > prompt keywords > target_type > default

```bash
# Device type (priority chain)
device_type = --device-type (if != "auto")
           OR detect_keywords(prompt_text, ["mobile", "desktop", "tablet", "responsive"])
           OR infer_from_target(target_type)  # component→desktop, page→responsive
           OR "responsive"

device_source = track_detection_source()
```

**Detection Keywords**: mobile, phone, smartphone → mobile | desktop, web, laptop → desktop | tablet, ipad → tablet | responsive, adaptive → responsive

**Device Presets**: Desktop (1920×1080) | Mobile (375×812) | Tablet (768×1024) | Responsive (1920×1080 + breakpoints)

**Stored Variables**: `device_type`, `device_source`

### Phase 4: Run Initialization & Directory Setup
```bash
design_id = "design-run-$(date +%Y%m%d)-$RANDOM"
relative_base_path = --session ? ".workflow/active/WFS-{session}/${design_id}" : ".workflow/${design_id}"

# Create directory and convert to absolute path
Bash(mkdir -p "${relative_base_path}/style-extraction")
Bash(mkdir -p "${relative_base_path}/prototypes")
base_path=$(cd "${relative_base_path}" && pwd)

Write({base_path}/.run-metadata.json): {
  "design_id": "${design_id}", "session_id": "${session_id}", "timestamp": "...",
  "workflow": "ui-design:auto",
  "architecture": "style-centric-batch-generation",
  "parameters": { "style_variants": ${style_variants}, "layout_variants": ${layout_variants},
                  "targets": "${inferred_target_list}", "target_type": "${target_type}",
                  "prompt": "${prompt_text}", "images": "${images_input}",
                  "input": "${--input}",
                  "device_type": "${device_type}", "device_source": "${device_source}" },
  "status": "in_progress",
  "performance_mode": "optimized"
}

# Initialize default flags for animation extraction logic
animation_complete = false  # Default: always extract animations unless code import proves complete
needs_visual_supplement = false  # Will be set to true in hybrid mode
skip_animation_extraction = false  # User preference for code import scenario
```

### Phase 5: Unified Target Inference with Intelligent Type Detection
```bash
# Priority: --pages/--components (legacy) → --targets → --prompt analysis → synthesis → default
target_list = []; target_type = "auto"; target_source = "none"

# Step 1-2: Explicit parameters (legacy or unified)
IF --pages: target_list = split(--pages); target_type = "page"; target_source = "explicit_legacy"
ELSE IF --components: target_list = split(--components); target_type = "component"; target_source = "explicit_legacy"
ELSE IF --targets:
    target_list = split(--targets); target_source = "explicit"
    target_type = --target-type != "auto" ? --target-type : detect_target_type(target_list)

# Step 3: Prompt analysis (Claude internal analysis)
ELSE IF prompt_text:
    analysis_result = analyze_prompt(prompt_text)  # Extract targets, types, purpose
    target_list = analysis_result.targets
    target_type = analysis_result.primary_type OR detect_target_type(target_list)
    target_source = "prompt_analysis"

# Step 4: Session synthesis
ELSE IF --session AND exists(role analysis documents):
    target_list = extract_targets_from_synthesis(); target_type = "page"; target_source = "synthesis"

# Step 5: Fallback
IF NOT target_list: target_list = ["home"]; target_type = "page"; target_source = "default"

# Validate and clean
validated_targets = [normalize(t) for t in target_list if is_valid(t)]
IF NOT validated_targets: validated_targets = ["home"]; target_type = "page"
IF --target-type != "auto": target_type = --target-type

# Interactive confirmation
DISPLAY_CONFIRMATION(target_type, target_source, validated_targets, device_type, device_source):
  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  "{emoji} {LABEL} CONFIRMATION (Style-Centric)"
  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  "Type: {target_type} | Source: {target_source}"
  "Targets ({count}): {', '.join(validated_targets)}"
  "Device: {device_type} | Source: {device_source}"
  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  "Performance: {style_variants} agent calls"
  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  "Modification Options:"
  "  • 'continue/yes/ok' - Proceed with current configuration"
  "  • 'targets: a,b,c' - Replace target list"
  "  • 'skip: x,y' - Remove specific targets"
  "  • 'add: z' - Add new targets"
  "  • 'type: page|component' - Change target type"
  "  • 'device: desktop|mobile|tablet|responsive' - Change device type"
  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

user_input = WAIT_FOR_USER_INPUT()

# Process user modifications
MATCH user_input:
  "continue|yes|ok" → proceed
  "targets: ..." → validated_targets = parse_new_list()
  "skip: ..." → validated_targets = remove_items()
  "add: ..." → validated_targets = add_items()
  "type: ..." → target_type = extract_type()
  "device: ..." → device_type = extract_device()
  default → proceed with current list

STORE: inferred_target_list, target_type, target_inference_source

# ⚠️ CRITICAL: User confirmation complete, IMMEDIATELY initialize TodoWrite and execute Phase 7
# This is the only user interaction point in the workflow
# After this point, all subsequent phases execute automatically without user intervention
```

**Helper Function: detect_target_type()**
```bash
detect_target_type(target_list):
    page_keywords = ["home", "dashboard", "settings", "profile", "login", "signup", "auth", ...]
    component_keywords = ["navbar", "header", "footer", "hero", "card", "button", "form", ...]

    page_matches = count_matches(target_list, page_keywords + ["page", "screen", "view"])
    component_matches = count_matches(target_list, component_keywords + ["component", "widget"])

    RETURN "component" IF component_matches > page_matches ELSE "page"
```

### Phase 6: Code Import & Completeness Assessment (Conditional)

**Step 6.1: Dispatch** - Import design system from code files

```javascript
IF design_source IN ["code_only", "hybrid"]:
    REPORT: "🔍 Phase 6: Code Import ({design_source})"
    command = "/workflow:ui-design:import-from-code --design-id \"{design_id}\" --source \"{code_base_path}\""

    TRY:
        # SlashCommand dispatch ATTACHES import-from-code's tasks to current workflow
        # Orchestrator will EXECUTE these attached tasks itself:
        #   - Phase 0: Discover and categorize code files
        #   - Phase 1.1-1.3: Style/Animation/Layout Agent extraction
        SlashCommand(command)
    CATCH error:
        WARN: "⚠️ Code import failed: {error}"
        WARN: "Cleaning up incomplete import directories"
        Bash(rm -rf "{base_path}/style-extraction" "{base_path}/animation-extraction" "{base_path}/layout-extraction" 2>/dev/null)

        IF design_source == "code_only":
            REPORT: "Cannot proceed with code-only mode after import failure"
            EXIT 1
        ELSE:  # hybrid mode
            WARN: "Continuing with visual-only mode"
            design_source = "visual_only"

    # Check file existence and assess completeness
    style_exists = exists("{base_path}/style-extraction/style-1/design-tokens.json")
    animation_exists = exists("{base_path}/animation-extraction/animation-tokens.json")
    layout_count = bash(ls {base_path}/layout-extraction/layout-*.json 2>/dev/null | wc -l)
    layout_exists = (layout_count > 0)

    style_complete = false
    animation_complete = false
    layout_complete = false
    missing_categories = []

    # Style completeness check
    IF style_exists:
        tokens = Read("{base_path}/style-extraction/style-1/design-tokens.json")
        style_complete = (
            tokens.colors?.brand && tokens.colors?.surface &&
            tokens.typography?.font_family && tokens.spacing &&
            Object.keys(tokens.colors.brand || {}).length >= 3 &&
            Object.keys(tokens.spacing || {}).length >= 8
        )
        IF NOT style_complete AND tokens._metadata?.completeness?.missing_categories:
            missing_categories.extend(tokens._metadata.completeness.missing_categories)
    ELSE:
        missing_categories.push("style tokens")

    # Animation completeness check
    IF animation_exists:
        anim = Read("{base_path}/animation-extraction/animation-tokens.json")
        animation_complete = (
            anim.duration && anim.easing &&
            Object.keys(anim.duration || {}).length >= 3 &&
            Object.keys(anim.easing || {}).length >= 3
        )
        IF NOT animation_complete AND anim._metadata?.completeness?.missing_items:
            missing_categories.extend(anim._metadata.completeness.missing_items)
    ELSE:
        missing_categories.push("animation tokens")

    # Layout completeness check
    IF layout_exists:
        # Read first layout file to verify structure
        first_layout = bash(ls {base_path}/layout-extraction/layout-*.json 2>/dev/null | head -1)
        layout_data = Read(first_layout)
        layout_complete = (
            layout_count >= 1 &&
            layout_data.template?.dom_structure &&
            layout_data.template?.css_layout_rules
        )
        IF NOT layout_complete:
            missing_categories.push("complete layout structure")
    ELSE:
        missing_categories.push("layout templates")

    needs_visual_supplement = false

    IF design_source == "code_only" AND NOT (style_complete AND layout_complete):
        REPORT: "⚠️  Missing: {', '.join(missing_categories)}"
        REPORT: "Options: 'continue' | 'supplement: <images>' | 'cancel'"
        user_response = WAIT_FOR_USER_INPUT()
        MATCH user_response:
            "continue" → needs_visual_supplement = false
            "supplement: ..." → needs_visual_supplement = true; --images = extract_path(user_response)
            "cancel" → EXIT 0
            default → needs_visual_supplement = false
    ELSE IF design_source == "hybrid":
        needs_visual_supplement = true

    # Animation reuse confirmation (code import with complete animations)
    IF design_source == "code_only" AND animation_complete:
        REPORT: "✅ Complete animation system detected (from code import)"
        REPORT: "   Duration scales: {duration_count} | Easing functions: {easing_count}"
        REPORT: ""
        REPORT: "Options:"
        REPORT: "  • 'reuse' (default) - Reuse existing animation system"
        REPORT: "  • 'regenerate' - Regenerate animation system (interactive)"
        REPORT: "  • 'cancel' - Cancel workflow"
        user_response = WAIT_FOR_USER_INPUT()
        MATCH user_response:
            "reuse" → skip_animation_extraction = true
            "regenerate" → skip_animation_extraction = false
            "cancel" → EXIT 0
            default → skip_animation_extraction = true  # Default: reuse

    STORE: needs_visual_supplement, style_complete, animation_complete, layout_complete, skip_animation_extraction
```

### Phase 7: Style Extraction

**Step 7.1: Dispatch** - Extract style design systems

```javascript
IF design_source == "visual_only" OR needs_visual_supplement:
    REPORT: "🎨 Phase 7: Style Extraction (variants: {style_variants})"
    command = "/workflow:ui-design:style-extract --design-id \"{design_id}\" " +
              (images_input ? "--images \"{images_input}\" " : "") +
              (prompt_text ? "--prompt \"{prompt_text}\" " : "") +
              "--variants {style_variants} --interactive"

    # SlashCommand dispatch ATTACHES style-extract's tasks to current workflow
    # Orchestrator will EXECUTE these attached tasks itself
    SlashCommand(command)

    # After executing all attached tasks, collapse them into phase summary
ELSE:
    REPORT: "✅ Phase 7: Style (Using Code Import)"
```

### Phase 8: Animation Extraction

**Step 8.1: Dispatch** - Extract animation patterns

```javascript
# Determine if animation extraction is needed
should_extract_animation = false

IF (design_source == "visual_only" OR needs_visual_supplement):
    # Pure visual input or hybrid mode requiring visual supplement
    should_extract_animation = true
ELSE IF NOT animation_complete:
    # Code import but animations are incomplete
    should_extract_animation = true
ELSE IF design_source == "code_only" AND animation_complete AND NOT skip_animation_extraction:
    # Code import with complete animations, but user chose to regenerate
    should_extract_animation = true

IF should_extract_animation:
    REPORT: "🚀 Phase 8: Animation Extraction"

    # Build command with available inputs
    command_parts = [f"/workflow:ui-design:animation-extract --design-id \"{design_id}\""]

    IF images_input:
        command_parts.append(f"--images \"{images_input}\"")

    IF prompt_text:
        command_parts.append(f"--prompt \"{prompt_text}\"")

    command_parts.append("--interactive")

    command = " ".join(command_parts)

    # SlashCommand dispatch ATTACHES animation-extract's tasks to current workflow
    # Orchestrator will EXECUTE these attached tasks itself
    SlashCommand(command)

    # After executing all attached tasks, collapse them into phase summary
ELSE:
    REPORT: "✅ Phase 8: Animation (Using Code Import)"

# Output: animation-tokens.json + animation-guide.md
# When phase finishes, IMMEDIATELY execute Phase 9 (auto-continue)
```

### Phase 9: Layout Extraction

**Step 9.1: Dispatch** - Extract layout templates

```javascript
targets_string = ",".join(inferred_target_list)

IF (design_source == "visual_only" OR needs_visual_supplement) OR (NOT layout_complete):
    REPORT: "🚀 Phase 9: Layout Extraction ({targets_string}, variants: {layout_variants}, device: {device_type})"
    command = "/workflow:ui-design:layout-extract --design-id \"{design_id}\" " +
              (images_input ? "--images \"{images_input}\" " : "") +
              (prompt_text ? "--prompt \"{prompt_text}\" " : "") +
              "--targets \"{targets_string}\" --variants {layout_variants} --device-type \"{device_type}\" --interactive"

    # SlashCommand dispatch ATTACHES layout-extract's tasks to current workflow
    # Orchestrator will EXECUTE these attached tasks itself
    SlashCommand(command)

    # After executing all attached tasks, collapse them into phase summary
ELSE:
    REPORT: "✅ Phase 9: Layout (Using Code Import)"
```

### Phase 10: UI Assembly

**Step 10.1: Dispatch** - Assemble UI prototypes from design tokens and layout templates

```javascript
command = "/workflow:ui-design:generate --design-id \"{design_id}\"" + (--session ? " --session {session_id}" : "")

total = style_variants × layout_variants × len(inferred_target_list)

REPORT: "🚀 Phase 10: UI Assembly | Matrix: {s}×{l}×{n} = {total} prototypes"
REPORT: "   → Pure assembly: Combining layout templates + design tokens"
REPORT: "   → Device: {device_type} (from layout templates)"
REPORT: "   → Assembly tasks: {total} combinations"

# SlashCommand dispatch ATTACHES generate's tasks to current workflow
# Orchestrator will EXECUTE these attached tasks itself
SlashCommand(command)

# After executing all attached tasks, collapse them into phase summary
# Workflow complete - generate command handles preview file generation (compare.html, PREVIEW.md)
# Output (generated by generate command):
# - {target}-style-{s}-layout-{l}.html (assembled prototypes)
# - {target}-style-{s}-layout-{l}.css
# - compare.html (interactive matrix view)
# - PREVIEW.md (usage instructions)
```

## TodoWrite Pattern
```javascript
// Initialize IMMEDIATELY after Phase 5 user confirmation to track multi-phase execution (4 orchestrator-level tasks)
TodoWrite({todos: [
  {"content": "Phase 7: Style Extraction", "status": "in_progress", "activeForm": "Executing style extraction"},
  {"content": "Phase 8: Animation Extraction", "status": "pending", "activeForm": "Executing animation extraction"},
  {"content": "Phase 9: Layout Extraction", "status": "pending", "activeForm": "Executing layout extraction"},
  {"content": "Phase 10: UI Assembly", "status": "pending", "activeForm": "Executing UI assembly"}
]})

// ⚠️ CRITICAL: Dynamic TodoWrite task attachment strategy:
//
// **Key Concept**: SlashCommand dispatch ATTACHES tasks to current workflow.
// Orchestrator EXECUTES these attached tasks itself, not waiting for external completion.
//
// Phase 7-10 SlashCommand Dispatch Pattern (when tasks are attached):
// Example - Phase 7 with sub-tasks:
// [
//   {"content": "Phase 7: Style Extraction", "status": "in_progress", "activeForm": "Executing style extraction"},
//   {"content": "  → Analyze style references", "status": "in_progress", "activeForm": "Analyzing style references"},
//   {"content": "  → Generate style variants", "status": "pending", "activeForm": "Generating style variants"},
//   {"content": "  → Create design tokens", "status": "pending", "activeForm": "Creating design tokens"},
//   {"content": "Phase 8: Animation Extraction", "status": "pending", "activeForm": "Executing animation extraction"},
//   ...
// ]
//
// After sub-tasks complete, COLLAPSE back to:
// [
//   {"content": "Phase 7: Style Extraction", "status": "completed", "activeForm": "Executing style extraction"},
//   {"content": "Phase 8: Animation Extraction", "status": "in_progress", "activeForm": "Executing animation extraction"},
//   ...
// ]
//
```

## Completion Output
```
✅ UI Design Explore-Auto Workflow Complete!

Architecture: Style-Centric Batch Generation
Run ID: {run_id} | Session: {session_id or "standalone"}
Type: {icon} {target_type} | Device: {device_type} | Matrix: {s}×{l}×{n} = {total} prototypes

Phase 7: {s} complete design systems (style-extract with multi-select)
Phase 9: {n×l} layout templates (layout-extract with multi-select)
  - Device: {device_type} layouts
  - {n} targets × {l} layout variants = {n×l} structural templates
  - User-selected concepts generated in parallel
Phase 10: UI Assembly (generate)
  - Pure assembly: layout templates + design tokens
  - {s}×{l}×{n} = {total} final prototypes
  - Preview files: compare.html, PREVIEW.md (auto-generated by generate command)

Assembly Process:
✅ Separation of Concerns: Layout (structure) + Style (tokens) kept separate
✅ Layout Extraction: {n×l} reusable structural templates
✅ Multi-Selection Workflow: User selects multiple variants from generated options
✅ Pure Assembly: No design decisions in generate phase
✅ Device-Optimized: Layouts designed for {device_type}

Design Quality:
✅ Token-Driven Styling: 100% var() usage
✅ Structural Variety: {l} distinct layouts per target (user-selected)
✅ Style Variety: {s} independent design systems (user-selected)
✅ Device-Optimized: Layouts designed for {device_type}

📂 {base_path}/
  ├── .intermediates/          (Intermediate analysis files)
  │   ├── style-analysis/      (analysis-options.json with embedded user_selection, computed-styles.json if URL mode)
  │   ├── animation-analysis/  (analysis-options.json with embedded user_selection, animations-*.json if URL mode)
  │   └── layout-analysis/     (analysis-options.json with embedded user_selection, dom-structure-*.json if URL mode)
  ├── style-extraction/        ({s} complete design systems)
  ├── animation-extraction/    (animation-tokens.json, animation-guide.md)
  ├── layout-extraction/       ({n×l} layout template files: layout-{target}-{variant}.json)
  ├── prototypes/              ({total} assembled prototypes)
  └── .run-metadata.json       (includes device type)

🌐 Preview: {base_path}/prototypes/compare.html
  - Interactive {s}×{l} matrix view
  - Side-by-side comparison
  - Target-specific layouts with style-aware structure
  - Toggle between {n} targets

{icon} Targets: {', '.join(targets)} (type: {target_type})
  - Each target has {l} custom-designed layouts
  - Each style × target × layout has unique HTML structure (not just CSS!)
  - Layout plans stored as structured JSON
  - Optimized for {device_type} viewing

Next: Open compare.html to preview all design variants
```

