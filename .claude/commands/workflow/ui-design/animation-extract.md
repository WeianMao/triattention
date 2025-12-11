---
name: animation-extract
description: Extract animation and transition patterns from prompt inference and image references for design system documentation
argument-hint: "[--design-id <id>] [--session <id>] [--images "<glob>"] [--focus "<types>"] [--interactive] [--refine]"
allowed-tools: TodoWrite(*), Read(*), Write(*), Glob(*), Bash(*), AskUserQuestion(*), Task(ui-design-agent)
---

# Animation Extraction Command

## Overview

Extract animation and transition patterns from prompt inference and image references using AI analysis. Directly generates production-ready animation systems with complete `animation-tokens.json`.

**Strategy**: AI-Driven Animation Specification with Visual Previews

- **Dual Modes**: Exploration mode (generate from scratch) or Refinement mode (fine-tune existing)
- **Prompt Inference**: AI analyzes design intent from textual descriptions and image references
- **Question Generation**: Agent generates context-aware specification questions with visual previews
- **Refinement Options**: Fine-tune timing, easing, context variations, and interaction intensity
- **Visual Previews**: Timeline representations, easing curve ASCII art, and animation sequence diagrams
- **Flexible Input**: Image references and prompts for animation specification
- **Optional Interaction**: User answers questions only when `--interactive` flag present
- **Production-Ready**: CSS var() format, WCAG-compliant, semantic naming
- **Default Behavior**: Non-interactive mode uses inferred patterns + best practices

## Execution Process

```
Input Parsing:
   ├─ Parse flags: --design-id, --session, --images, --focus, --interactive, --refine
   └─ Decision (mode detection):
      ├─ --refine flag → Refinement Mode
      └─ No --refine → Exploration Mode

Phase 0: Setup & Input Validation
   ├─ Step 1: Detect input mode & base path
   ├─ Step 2: Prepare image references (if available)
   ├─ Step 3: Load design tokens context
   └─ Step 4: Memory check (skip if exists)

Phase 1: Animation Specification Generation
   ├─ Step 1: Load project context
   ├─ Step 2: Generate animation specification options (Agent Task 1)
   │  └─ Decision:
   │     ├─ Exploration Mode → Generate specification questions
   │     └─ Refinement Mode → Generate refinement options
   └─ Step 3: Verify options file created

Phase 1.5: User Confirmation (Optional)
   └─ Decision (--interactive flag):
      ├─ --interactive present → Present options, capture selection
      └─ No --interactive → Skip to Phase 2

Phase 2: Animation System Generation
   ├─ Step 1: Load user selection or use defaults
   ├─ Step 2: Create output directory
   └─ Step 3: Launch animation generation task (Agent Task 2)

Phase 3: Verify Output
   ├─ Step 1: Check files created
   └─ Step 2: Verify file sizes
```

## Phase 0: Setup & Input Validation

### Step 1: Detect Input Mode & Base Path

```bash
# Detect input source
# Priority: --images → visual references available | no --images → prompt-only mode

# Parse images if provided (glob pattern)
IF --images:
    # Check if glob pattern matches any files
    image_files = bash(find . -path "{--images}" -type f 2>/dev/null | head -10)
    IF image_files:
        has_images = true
        image_count = bash(find . -path "{--images}" -type f 2>/dev/null | wc -l)
    ELSE:
        has_images = false
        REPORT: "⚠️ No image files found matching pattern: {--images}"
ELSE:
    has_images = false

# Parse animation focus (if provided)
IF --focus:
    focus_types = split(--focus, ",")  # e.g., "transitions,hover,scroll"
ELSE:
    focus_types = ["all"]  # Extract all animation types

# Check interactive mode flag
interactive_mode = --interactive OR false

# Check refinement mode flag
refine_mode = --refine OR false

IF refine_mode:
    REPORT: "🔧 Refinement mode enabled: Will refine existing animation system"
ELSE:
    REPORT: "✨ Exploration mode: Will generate animation system from scratch"

# Determine base path with priority: --design-id > --session > auto-detect
if [ -n "$DESIGN_ID" ]; then
  # Exact match by design ID
  relative_path=$(find .workflow -name "${DESIGN_ID}" -type d -print -quit)
elif [ -n "$SESSION_ID" ]; then
  # Latest in session
  relative_path=$(find .workflow/active/WFS-$SESSION_ID -name "design-run-*" -type d -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2)
else
  # Latest globally
  relative_path=$(find .workflow -name "design-run-*" -type d -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2)
fi

# Validate and convert to absolute path
if [ -z "$relative_path" ] || [ ! -d "$relative_path" ]; then
  echo "❌ ERROR: Design run not found"
  echo "💡 HINT: Run '/workflow:ui-design:list' to see available design runs"
  exit 1
fi

base_path=$(cd "$relative_path" && pwd)
bash(echo "✓ Base path: $base_path")
```

### Step 2: Prepare Image References (If Available)

```bash
# Load image references if provided
IF has_images:
    REPORT: "🔍 Loading image references for animation analysis"
    REPORT: "   Pattern: {--images}"
    REPORT: "   Found: {image_count} image(s)"

    bash(mkdir -p {base_path}/.intermediates/animation-analysis)

    # Store image paths for agent reference
    image_list = []
    FOR image_file IN image_files:
        image_list.append(image_file)
        REPORT: "   • {image_file}"

    # Save image references metadata
    image_metadata = {
        "pattern": --images,
        "count": image_count,
        "files": image_list,
        "timestamp": current_timestamp()
    }
    Write({base_path}/.intermediates/animation-analysis/image-references.json, JSON.stringify(image_metadata, indent=2))

    REPORT: "   ✅ Image references prepared for AI analysis"
ELSE:
    REPORT: "ℹ️ No image references provided - using prompt-only mode"
```

**Image Analysis Strategy**:
- Agent analyzes visual motion cues from reference images
- Infers animation patterns from UI element positioning and design style
- Generates context-aware animation specifications based on visual analysis



### Step 3: Load Design Tokens Context

```bash
# Load existing design tokens for duration/easing alignment
IF exists({base_path}/style-extraction/style-1/design-tokens.json):
    design_tokens = Read({base_path}/style-extraction/style-1/design-tokens.json)
    has_design_context = true
ELSE:
    has_design_context = false
    REPORT: "ℹ️ No design tokens found - animation tokens will use standalone values"

# Create output directory
bash(mkdir -p {base_path}/animation-extraction)
```

### Step 4: Memory Check

```bash
# Check if output already exists
bash(test -f {base_path}/animation-extraction/animation-tokens.json && echo "exists")
IF exists: SKIP to completion
```

---

**Phase 0 Output**: `input_mode`, `base_path`, `has_images`, `image_list[]`, `focus_types[]`, `has_design_context`, `interactive_mode`, `refine_mode`

## Phase 1: Animation Specification Generation

### Step 1: Load Project Context

```bash
# Load brainstorming context if available
bash(test -f {base_path}/.brainstorming/role-analysis.md && cat it)

# Load image references if available
IF has_images:
    image_references = Read({base_path}/.intermediates/animation-analysis/image-references.json)
    REPORT: "📸 Image references loaded: {image_references.count} file(s)"
```

### Step 2: Generate Animation Specification Options (Agent Task 1)

**Executor**: `Task(ui-design-agent)`

**Conditional Logic**: Branch based on `refine_mode` flag

```javascript
IF NOT refine_mode:
    // EXPLORATION MODE (default)
    Task(ui-design-agent): `
      [ANIMATION_SPECIFICATION_GENERATION_TASK]
      Generate context-aware animation specification questions

      SESSION: {session_id} | MODE: explore | BASE_PATH: {base_path}

  ## Input Analysis
  - Focus types: {focus_types.join(", ")}
  - Design context: {has_design_context ? "Available" : "None"}
  - Image references: {has_images ? "Available (" + image_count + " files)" : "None"}
  ${has_images ? "- Image Data: Read from .intermediates/animation-analysis/image-references.json" : ""}

  ## Analysis Rules
  - Analyze image references (if available) to infer animation patterns from visual cues
  - Generate questions covering timing, easing, interactions, and motion patterns
  - Based on focus_types, include relevant categories:
    * "all" or "transitions": timing_scale, easing_philosophy
    * "all" or "interactions" or "hover": button_interactions, card_interactions, input_interactions
    * "all" or "page": page_transitions
    * "all" or "loading": loading_states
    * "all" or "scroll": scroll_animations

  ## Generate Questions
  For each applicable category, create question with:
  1. **Category ID** (e.g., "timing_scale", "button_interactions")
  2. **Question text** (in Chinese, clear and concise)
  3. **Options** (2-5 options per question):
     - Option key (a, b, c, d, e)
     - Option label (brief description)
     - Option details (detailed explanation with technical specs)
     - Technical specs (duration values, easing curves, transform values)
     - Visual preview (timeline representation or easing curve ASCII art)

  ## Output
  Write single JSON file: {base_path}/.intermediates/animation-analysis/analysis-options.json

  Use schema:
  {
    "metadata": {
      "generated_at": "<timestamp>",
      "focus_types": [...],
      "total_questions": <count>,
      "has_css_data": <boolean>
    },
    "specification_options": [
      {
        "id": 1,
        "category": "timing_scale",
        "question": "您的设计需要什么样的过渡速度？",
        "options": [
          {
            "key": "a",
            "label": "快速敏捷",
            "details": "100-200ms 过渡，适合工具型应用和即时反馈场景",
            "duration_values": {"fast": "100ms", "normal": "150ms", "slow": "200ms"},
            "visual_preview": {
              "timeline": "0ms ━━━━━━━━━━ 150ms",
              "description": "快速完成，几乎瞬时反馈"
            }
          },
          ...
        ]
      },
      {
        "id": 2,
        "category": "easing_philosophy",
        "question": "您偏好什么样的动画缓动曲线？",
        "options": [
          {
            "key": "a",
            "label": "自然缓动",
            "details": "标准 ease-out，模拟自然减速",
            "easing_curves": {
              "ease-in": "cubic-bezier(0.4, 0, 1, 1)",
              "ease-out": "cubic-bezier(0, 0, 0.2, 1)",
              "ease-in-out": "cubic-bezier(0.4, 0, 0.2, 1)"
            },
            "visual_preview": {
              "curve_art": "│      ╱─\n│    ╱\n│  ╱\n│╱\n└─────",
              "description": "快速启动，平滑减速"
            }
          },
          ...
        ]
      },
      ...
    ]
  }

  CRITICAL: Use Write() tool immediately after generating complete JSON
    `

ELSE:
    // REFINEMENT MODE
    Task(ui-design-agent): `
      [ANIMATION_REFINEMENT_OPTIONS_TASK]
      Generate refinement options for existing animation system

      SESSION: {session_id} | MODE: refine | BASE_PATH: {base_path}

      ## Load Existing Animation System
      - Existing tokens: Read from {base_path}/animation-extraction/animation-tokens.json
      - Focus types: {focus_types.join(", ")}
      - Design context: {has_design_context ? "Available" : "None"}
      ${has_images ? "- Image Data: Read from .intermediates/animation-analysis/image-references.json" : ""}

      ## Refinement Categories
      Generate 8-12 refinement options across these categories:

      1. **Timing Adjustments** (2-3 options):
         - Duration scale: Faster timing across the board ↔ Slower, more deliberate timing
         - Specific categories: Accelerate interactions only ↔ Extend page transitions
         - Micro-timing: Adjust stagger delays ↔ Sequential animation gaps

      2. **Easing Fine-Tuning** (2-3 options):
         - Curve intensity: Sharper, snappier curves ↔ Softer, smoother curves
         - Category-specific: Bouncier interactions ↔ Linear state changes
         - Spring physics: Adjust bounce/damping parameters

      3. **Context-Specific Variations** (2-3 options):
         - Reduced motion: Adjust reduced-motion fallbacks
         - Mobile optimization: Shorter durations for touch interactions
         - Component-specific: Different hover styles for buttons vs cards

      4. **Interaction Intensity** (1-2 options):
         - Transform magnitude: Subtle movements (2-4px) ↔ Dramatic movements (8-12px)
         - Scale adjustments: Minimal scale changes ↔ Bold scale emphasis
         - Opacity ranges: Partial fades ↔ Full visibility transitions

      ## Generate Refinement Options
      For each category, create option with:
      1. **Option ID** (sequential number)
      2. **Category** (timing_adjustments, easing_tuning, context_variations, interaction_intensity)
      3. **Label** (brief Chinese description, e.g., "加快整体节奏")
      4. **Description** (detailed explanation of changes)
      5. **Impact Scope** (which tokens will be modified)
      6. **Technical Changes** (specific value adjustments)
      7. **Before/After Preview** (show current vs proposed values)

      ## Output
      Write single JSON file: {base_path}/.intermediates/animation-analysis/refinement-options.json

      Use schema:
      {
        "metadata": {
          "generated_at": "<timestamp>",
          "mode": "refinement",
          "existing_tokens_loaded": true,
          "total_refinements": <count>
        },
        "current_animation_system": {
          // Copy from animation-tokens.json for reference
        },
        "refinement_options": [
          {
            "id": 1,
            "category": "timing_adjustments",
            "label": "加快整体动画节奏",
            "description": "将所有 duration 值减少 30%，使界面响应更快速",
            "impact_scope": "duration.fast, duration.normal, duration.slow",
            "technical_changes": {
              "duration.fast": {"from": "150ms", "to": "105ms"},
              "duration.normal": {"from": "300ms", "to": "210ms"},
              "duration.slow": {"from": "500ms", "to": "350ms"}
            },
            "preview": {
              "before": "Normal button hover: 150ms",
              "after": "Faster button hover: 105ms"
            }
          },
          ...
        ]
      }

      CRITICAL: Use Write() tool immediately after generating complete JSON
    `
```

### Step 3: Verify Options File Created

```bash
IF NOT refine_mode:
    # Exploration mode: Check for analysis-options.json
    bash(test -f {base_path}/.intermediates/animation-analysis/analysis-options.json && echo "created")
    bash(cat {base_path}/.intermediates/animation-analysis/analysis-options.json | grep -q "specification_options" && echo "valid")
ELSE:
    # Refinement mode: Check for refinement-options.json
    bash(test -f {base_path}/.intermediates/animation-analysis/refinement-options.json && echo "created")
    bash(cat {base_path}/.intermediates/animation-analysis/refinement-options.json | grep -q "refinement_options" && echo "valid")
```

**Output**:
- Exploration mode: `analysis-options.json` with animation specification questions
- Refinement mode: `refinement-options.json` with refinement options

---

**Phase 1 Output**:
- Exploration mode: `analysis-options.json` with generated specification questions
- Refinement mode: `refinement-options.json` with refinement options

## Phase 1.5: User Confirmation (Optional - Triggered by --interactive)

**Purpose**: Allow user to answer animation specification questions (exploration) or select refinement options (refinement) before generating tokens

**Trigger Condition**: Execute this phase ONLY if `--interactive` flag is present

### Step 1: Check Interactive Flag

```bash
# Skip this entire phase if --interactive flag is not present
IF NOT --interactive:
    SKIP to Phase 2
    REPORT: "ℹ️ Non-interactive mode: Using CSS extraction + default animation preferences"

REPORT: "🎯 Interactive mode enabled: User answers required"
```

### Step 2: Load and Present Options

```bash
# Read options file based on mode
IF NOT refine_mode:
    # Exploration mode
    options = Read({base_path}/.intermediates/animation-analysis/analysis-options.json)
    specification_options = options.specification_options
ELSE:
    # Refinement mode
    options = Read({base_path}/.intermediates/animation-analysis/refinement-options.json)
    refinement_options = options.refinement_options
```

### Step 3: Present Options to User

**Conditional Display**: Branch based on `refine_mode` flag

```
IF NOT refine_mode:
    // EXPLORATION MODE
    📋 Animation Specification Questions

    We've generated {options.metadata.total_questions} questions to define your animation system.
    Please answer each question to customize the animation behavior.

    {FOR each question in specification_options:
      ═══════════════════════════════════════════════════
      Question {question.id}: {question.question}
      Category: {question.category}
      ═══════════════════════════════════════════════════

      {FOR each option in question.options:
        {option.key}) {option.label}
           {option.details}

           ${option.visual_preview ? "Preview:\n       " + option.visual_preview.timeline || option.visual_preview.curve_art || option.visual_preview.animation_sequence : ""}
           ${option.visual_preview ? "       " + option.visual_preview.description : ""}

           ${option.duration_values ? "Durations: " + JSON.stringify(option.duration_values) : ""}
           ${option.easing_curves ? "Easing: " + JSON.stringify(option.easing_curves) : ""}
           ${option.transform_value ? "Transform: " + option.transform_value : ""}
      }

      ═══════════════════════════════════════════════════
    }

ELSE:
    // REFINEMENT MODE
    🔧 Animation System Refinement Options

    We've generated {options.metadata.total_refinements} refinement options to fine-tune your animation system.
    Select which refinements to apply (can select multiple).

    {FOR each refinement in refinement_options:
      ═══════════════════════════════════════════════════
      Option {refinement.id}: {refinement.label}
      Category: {refinement.category}
      ═══════════════════════════════════════════════════

      Description: {refinement.description}
      Impact Scope: {refinement.impact_scope}

      Technical Changes:
      {FOR each token, changes IN refinement.technical_changes:
        • {token}:
          Before: {changes.from}
          After:  {changes.to}
      }

      Preview:
      {refinement.preview.before} → {refinement.preview.after}

      ═══════════════════════════════════════════════════
    }
```

### Step 4: Capture User Selection

**Conditional Interaction**: Branch based on `refine_mode` flag

```javascript
IF NOT refine_mode:
    // EXPLORATION MODE - Single selection per question
    user_answers = {}

    FOR each question IN specification_options:
      AskUserQuestion({
        questions: [{
          question: question.question,
          header: question.category,
          multiSelect: false,  // Single selection per question
          options: [
            {FOR each option IN question.options:
              label: "{option.key}) {option.label}",
              description: option.details
            }
          ]
        }]
      })

      // Parse user response (single selection, e.g., "a) Fast & Snappy")
      selected_option_text = user_answer

      // Check for user cancellation
      IF selected_option_text == null:
          REPORT: "⚠️ User canceled selection. Using default animation preferences."
          EXIT Phase 1.5

      // Extract option key from selection text
      match = selected_option_text.match(/^([a-e])\)/)
      IF match:
          selected_key = match[1]
          user_answers[question.category] = selected_key
          REPORT: "✅ {question.category}: Selected option {selected_key}"
      ELSE:
          ERROR: "Invalid selection format. Expected 'a) ...' format"
          EXIT workflow

    REPORT: "✅ Collected {Object.keys(user_answers).length} animation preferences"

ELSE:
    // REFINEMENT MODE - Multi-selection of refinements
    AskUserQuestion({
      questions: [{
        question: "Which refinement(s) would you like to apply to your animation system?",
        header: "Refinements",
        multiSelect: true,  // Can select multiple refinements
        options: [
          {FOR each refinement IN refinement_options:
            label: "{refinement.id}. {refinement.label}",
            description: "{refinement.description} (Affects: {refinement.impact_scope})"
          }
        ]
      }]
    })

    // Parse user response (multi-selection)
    selected_refinements = user_answer

    // Check for user cancellation
    IF selected_refinements == null:
        REPORT: "⚠️ User canceled selection. No refinements will be applied."
        EXIT Phase 1.5

    // Extract refinement IDs
    selected_ids = []
    FOR each selection IN selected_refinements:
        match = selection.match(/^(\d+)\./)
        IF match:
            selected_ids.push(parseInt(match[1]))

    REPORT: "✅ Selected {selected_ids.length} refinement(s) to apply"
```

### Step 5: Update Options File with User Selection

```bash
IF NOT refine_mode:
    # EXPLORATION MODE - Update analysis-options.json
    options.user_selection = {
      "selected_at": "{current_timestamp}",
      "session_id": "{session_id}",
      "answers": user_answers  // {category: selected_key}
    }

    # Write updated file back
    Write({base_path}/.intermediates/animation-analysis/analysis-options.json, JSON.stringify(options, indent=2))

    # Verify
    bash(test -f {base_path}/.intermediates/animation-analysis/analysis-options.json && echo "saved")

ELSE:
    # REFINEMENT MODE - Update refinement-options.json
    options.user_selection = {
      "selected_at": "{current_timestamp}",
      "session_id": "{session_id}",
      "selected_refinements": selected_ids  // Array of refinement IDs
    }

    # Write updated file back
    Write({base_path}/.intermediates/animation-analysis/refinement-options.json, JSON.stringify(options, indent=2))

    # Verify
    bash(test -f {base_path}/.intermediates/animation-analysis/refinement-options.json && echo "saved")
```

### Step 6: Confirmation Message

```
IF NOT refine_mode:
    // EXPLORATION MODE
    ✅ Animation preferences recorded!

    You selected:
    {FOR each category, selected_key IN user_answers:
        question = find(specification_options, q => q.category == category)
        option = find(question.options, o => o.key == selected_key)
        • {category}: {option.label}
          ({option.details})
    }

    Proceeding to generate animation system with your preferences...

ELSE:
    // REFINEMENT MODE
    ✅ Refinement selections recorded!

    You selected {selected_ids.length} refinement(s):
    {FOR each id IN selected_ids:
        refinement = find(refinement_options, r => r.id == id)
        • {refinement.label} ({refinement.category})
          Impact: {refinement.impact_scope}
    }

    Proceeding to apply refinements to animation system...
```

**Output**:
- Exploration mode: Updated `analysis-options.json` with embedded `user_selection` field
- Refinement mode: Updated `refinement-options.json` with `user_selection.selected_refinements` array

## Phase 2: Animation System Generation (Agent Task 2)

**Executor**: `Task(ui-design-agent)` for animation token generation

### Step 1: Load User Selection or Use Defaults

```bash
IF NOT refine_mode:
    # EXPLORATION MODE - Read analysis-options.json
    options = Read({base_path}/.intermediates/animation-analysis/analysis-options.json)
    specification_options = options.specification_options

    # Check if user_selection field exists (interactive mode)
    IF options.user_selection AND options.user_selection.answers:
        # Interactive mode: Use user-selected preferences
        user_answers = options.user_selection.answers
        REPORT: "🎯 Interactive mode: Using user-selected animation preferences"
    ELSE:
        # Non-interactive mode: Use defaults (first option for each question)
        user_answers = null
        REPORT: "ℹ️ Non-interactive mode: Using default animation preferences"

ELSE:
    # REFINEMENT MODE - Read refinement-options.json
    options = Read({base_path}/.intermediates/animation-analysis/refinement-options.json)
    refinement_options = options.refinement_options

    # Check if user_selection field exists (interactive mode)
    IF options.user_selection AND options.user_selection.selected_refinements:
        # Interactive mode: Use user-selected refinements
        selected_refinements = options.user_selection.selected_refinements
        REPORT: "🎯 Interactive mode: Applying {selected_refinements.length} selected refinement(s)"
    ELSE:
        # Non-interactive mode: Apply all refinements
        selected_refinements = null
        REPORT: "ℹ️ Non-interactive mode: Applying all refinements"

# Load image references if available for agent context
image_context = null
IF has_images:
    IF exists({base_path}/.intermediates/animation-analysis/image-references.json):
        image_context = Read({base_path}/.intermediates/animation-analysis/image-references.json)
        REPORT: "📸 Using {image_context.count} image reference(s) for animation inference"
```

### Step 2: Create Output Directory

```bash
# Create directory for animation system
bash(mkdir -p {base_path}/animation-extraction)
```

### Step 3: Launch Animation Generation Task

**Conditional Task**: Branch based on `refine_mode` flag

```javascript
IF NOT refine_mode:
    // EXPLORATION MODE
    Task(ui-design-agent): `
      [ANIMATION_SYSTEM_GENERATION_TASK]
      Generate production-ready animation system based on user preferences and CSS extraction

      SESSION: {session_id} | MODE: explore | BASE_PATH: {base_path}

  USER PREFERENCES:
  ${user_answers ? "- User Selection: " + JSON.stringify(user_answers) : "- Using Defaults: First option for each category"}
  ${user_answers ? "- Specification Options: Read from .intermediates/animation-analysis/analysis-options.json for detailed specs" : ""}

  ## Input Analysis
  - Interactive mode: {user_answers ? "Yes (user preferences available)" : "No (using defaults)"}
  - Image references: {image_context ? "Available (" + image_context.count + " files)" : "None"}
  ${image_context ? "- Image Data: " + JSON.stringify(image_context) : ""}
  - Design context: {has_design_context ? "Available" : "None"}
  ${has_design_context ? "- Design Tokens: Read from style-extraction/style-1/design-tokens.json" : ""}

  ## Generation Rules
  ${user_answers ? `
  - Read analysis-options.json to get user_selection.answers
  - For each category in user_selection.answers, find the selected option
  - Use the selected option's technical specs (duration_values, easing_curves, transform_value, etc.)
  - Apply these specs to generate animation tokens
  ` : `
  - Use first option (key "a") from each question in specification_options as default
  - Extract technical specs from default options
  `}
  - Infer animation patterns from image references (if available)
  - Align with design tokens (spacing, colors) if available
  - All tokens use CSS Custom Property format: var(--duration-fast)
  - WCAG-compliant: Respect prefers-reduced-motion
  - Semantic naming for all animation values

  ## Synthesis Priority
  1. User answers from analysis-options.json user_selection field (highest priority)
  2. Inferred patterns from image references (medium priority)
  3. Industry best practices (fallback)

  ## Duration Normalization
  - IF user_selection.answers.timing_scale EXISTS:
      Find selected option in specification_options
      Use option's duration_values for token generation
  - ELSE IF image references available:
      Infer timing patterns from visual design style (minimalist → faster, ornate → slower)
  - ELSE:
      Use standard scale (instant:0ms, fast:150ms, normal:300ms, slow:500ms, very-slow:800ms)

  ## Easing Standardization
  - IF user_selection.answers.easing_philosophy EXISTS:
      Find selected option in specification_options
      Use option's easing_curves for token generation
  - ELSE IF image references available:
      Infer easing preferences from visual style (sharp edges → snappy, soft curves → smooth)
  - ELSE:
      Use standard easings (linear, ease-in, ease-out, ease-in-out, spring)

  ## Animation Categorization
  Organize into:
  - **duration**: Timing scale (instant, fast, normal, slow, very-slow)
  - **easing**: Easing functions (linear, ease-in, ease-out, ease-in-out, spring)
  - **transitions**: Property-specific transitions (color, transform, opacity, etc.)
  - **keyframes**: Named @keyframe animations (fadeIn, slideInUp, pulse, etc.)
  - **interactions**: Interaction-specific presets (button-hover, card-hover, input-focus, etc.)
  - **page_transitions**: Route/view change animations (if user enabled)
  - **scroll_animations**: Scroll-triggered animations (if user enabled)

  ## Generate Files

  ### 1. animation-tokens.json
  Complete animation token structure using var() references:

  {
    "duration": {
      "instant": "0ms",
      "fast": "150ms",      # From user_selection or CSS extraction or default
      "normal": "300ms",
      "slow": "500ms",
      "very-slow": "800ms"
    },
    "easing": {
      "linear": "linear",
      "ease-in": "cubic-bezier(0.4, 0, 1, 1)",
      "ease-out": "cubic-bezier(0, 0, 0.2, 1)",      # From user_selection or CSS extraction or default
      "ease-in-out": "cubic-bezier(0.4, 0, 0.2, 1)",
      "spring": "cubic-bezier(0.34, 1.56, 0.64, 1)"
    },
    "transitions": {
      "color": {
        "property": "color, background-color, border-color",
        "duration": "var(--duration-fast)",
        "easing": "var(--easing-ease-out)"
      },
      "transform": {
        "property": "transform",
        "duration": "var(--duration-normal)",
        "easing": "var(--easing-ease-out)"
      },
      "opacity": {
        "property": "opacity",
        "duration": "var(--duration-normal)",
        "easing": "var(--easing-ease-in-out)"
      }
    },
    "keyframes": {
      "fadeIn": {"0%": {"opacity": "0"}, "100%": {"opacity": "1"}},
      "slideInUp": {"0%": {"transform": "translateY(20px)", "opacity": "0"}, "100%": {"transform": "translateY(0)", "opacity": "1"}},
      "pulse": {"0%, 100%": {"opacity": "1"}, "50%": {"opacity": "0.7"}}
    },
    "interactions": {
      "button-hover": {
        # From user_selection.answers.button_interactions or CSS extraction or default
        "properties": ["background-color", "transform"],
        "duration": "var(--duration-fast)",
        "easing": "var(--easing-ease-out)",
        "transform": "scale(1.02)"
      },
      "card-hover": {
        # From user_selection.answers.card_interactions or CSS extraction or default
        "properties": ["box-shadow", "transform"],
        "duration": "var(--duration-normal)",
        "easing": "var(--easing-ease-out)",
        "transform": "translateY(-4px)"
      }
    },
    "page_transitions": {
      # IF user_selection.answers.page_transitions enabled
      "fade": {
        "duration": "var(--duration-normal)",
        "enter": "fadeIn",
        "exit": "fadeOut"
      }
    },
    "scroll_animations": {
      # IF user_selection.answers.scroll_animations enabled
      "default": {
        "animation": "fadeIn",
        "duration": "var(--duration-slow)",
        "easing": "var(--easing-ease-out)",
        "threshold": "0.1"
      }
    }
  }

  ## Output File Paths
  - animation-tokens.json: {base_path}/animation-extraction/animation-tokens.json

  ## Critical Requirements
  - ✅ Use Write() tool immediately to generate JSON file
  - ✅ All tokens use CSS Custom Property format: var(--duration-fast)
  - ✅ Include prefers-reduced-motion media query guidance
  - ✅ Validate all cubic-bezier values are valid (4 numbers between 0-1)
  - ${user_answers ? "✅ READ analysis-options.json for user_selection field" : "✅ Use first option from each question as default"}
  - ❌ NO user questions or interaction in this phase
  - ✅ Can use Exa MCP to research modern animation patterns and obtain code examples (Explore/Text mode)
    `

ELSE:
    // REFINEMENT MODE
    Task(ui-design-agent): `
      [ANIMATION_SYSTEM_REFINEMENT_TASK]
      Apply selected refinements to existing animation system

      SESSION: {session_id} | MODE: refine | BASE_PATH: {base_path}

      ## Load Existing Animation System
      - Current tokens: Read from {base_path}/animation-extraction/animation-tokens.json
      - Refinement options: Read from .intermediates/animation-analysis/refinement-options.json

      REFINEMENT SELECTION:
      ${selected_refinements ? `
      - Interactive mode: Apply selected refinements
      - Selected IDs: ${JSON.stringify(selected_refinements)}
      - For each ID in selected_refinements:
          * Find refinement in refinement_options by id
          * Apply technical_changes to corresponding tokens
      ` : `
      - Non-interactive mode: Apply ALL refinements
      - For each refinement in refinement_options:
          * Apply technical_changes to corresponding tokens
      `}

      ## Input Analysis
      - Image references: {image_context ? "Available (" + image_context.count + " files)" : "None"}
      ${image_context ? "- Image Data: " + JSON.stringify(image_context) : ""}
      - Design context: {has_design_context ? "Available" : "None"}
      ${has_design_context ? "- Design Tokens: Read from style-extraction/style-1/design-tokens.json" : ""}

      ## Refinement Application Rules
      ${selected_refinements ? `
      - ONLY apply refinements with IDs in selected_refinements array
      - Skip refinements not selected by user
      ` : `
      - Apply ALL refinements from refinement_options
      - Combine multiple refinements that affect same token
      `}
      - Load current animation-tokens.json
      - For each applicable refinement:
          * Parse technical_changes field
          * Apply "to" values to replace "from" values in tokens
          * Preserve structure and var() references
      - If multiple refinements affect same token, apply in sequence
      - Maintain WCAG compliance and semantic naming
      - All tokens use CSS Custom Property format: var(--duration-fast)

      ## Conflict Resolution
      - If multiple selected refinements modify same token:
          * Apply refinements in ID order (lowest first)
          * Later refinements override earlier ones

      ## Generate Updated Files

      ### 1. animation-tokens.json
      Updated animation token structure with refinements applied:
      - Load existing structure
      - Apply technical_changes from selected/all refinements
      - Maintain var() references and semantic naming
      - Validate all cubic-bezier values

      ## Output File Paths
      - animation-tokens.json: {base_path}/animation-extraction/animation-tokens.json (OVERWRITE)

      ## Critical Requirements
      - ✅ Use Write() tool immediately to generate JSON file
      - ✅ OVERWRITE existing animation-tokens.json with refined version
      - ✅ All tokens use CSS Custom Property format: var(--duration-fast)
      - ✅ Include prefers-reduced-motion media query guidance
      - ✅ Validate all cubic-bezier values are valid (4 numbers between 0-1)
      - ${selected_refinements ? "✅ READ refinement-options.json for user_selection.selected_refinements" : "✅ Apply ALL refinements from refinement_options"}
      - ❌ NO user questions or interaction in this phase
      - ✅ Can use Exa MCP to research modern animation patterns and obtain code examples (Explore/Text mode)
    `
```

**Output**: Agent generates/updates animation-tokens.json

## Phase 3: Verify Output

### Step 1: Check Files Created

```bash
# Verify animation system created
bash(test -f {base_path}/animation-extraction/animation-tokens.json && echo "exists")

# Validate structure
bash(cat {base_path}/animation-extraction/animation-tokens.json | grep -q "duration" && echo "valid")
bash(cat {base_path}/animation-extraction/animation-tokens.json | grep -q "easing" && echo "valid")
```

### Step 2: Verify File Sizes

```bash
bash(ls -lh {base_path}/animation-extraction/)
```

**Output**: animation-tokens.json verified

## Completion

### Todo Update

```javascript
TodoWrite({todos: [
  {content: "Setup and input validation", status: "completed", activeForm: "Validating inputs"},
  {content: "CSS animation extraction (Phase 1)", status: "completed", activeForm: "Extracting from CSS"},
  {content: "Specification generation (Phase 1 - Agent)", status: "completed", activeForm: "Generating questions"},
  {content: "User confirmation (Phase 1.5 - Optional)", status: "completed", activeForm: "Collecting user answers"},
  {content: "Animation system generation (Phase 2 - Agent)", status: "completed", activeForm: "Generating animation system"},
  {content: "Verify output files (Phase 3)", status: "completed", activeForm: "Verifying files"}
]});
```

### Output Message

```
✅ Animation extraction complete!

Configuration:
- Session: {session_id}
- Interactive Mode: {interactive_mode ? "Enabled (user preferences collected)" : "Disabled (default preferences)"}
- Input Sources:
  {IF has_images:
  - ✅ Image references analyzed ({image_count} file(s))
  }
  {IF interactive_mode AND options.user_selection:
  - ✅ User preferences collected via interactive mode
  }
  {IF NOT interactive_mode:
  - ℹ️ Using default animation preferences (no user interaction)
  }
  {IF has_design_context:
  - ✅ Aligned with existing design tokens
  }

Generated Files:
{base_path}/animation-extraction/
└── animation-tokens.json      # Production-ready animation tokens

{IF has_images OR options.user_selection:
Intermediate Analysis:
{base_path}/.intermediates/animation-analysis/
{IF has_images:
├── image-references.json       # Image reference metadata ({image_count} files)
}
├── analysis-options.json       # Generated questions{options.user_selection ? " + user answers" : ""}
}

Extracted Data Summary:
- Duration scales: {duration_count} values
- Easing functions: {easing_count} types
- Interaction presets: {interaction_count} patterns
- Keyframe animations: {keyframe_count} animations

Next: Animation tokens ready for integration
  • style-extract/layout-extract can reference animation tokens
  • generate command will include animation CSS
  • Tokens use var() format for easy customization
```

## Simple Bash Commands

### Path Operations

```bash
# Find design directory
bash(find .workflow -type d -name "design-run-*" | head -1)

# Create output directories
bash(mkdir -p {base_path}/animation-extraction)
bash(mkdir -p {base_path}/.intermediates/animation-analysis)
```

### Validation Commands

```bash
# Check if already extracted
bash(test -f {base_path}/animation-extraction/animation-tokens.json && echo "exists")

# Validate JSON structure
bash(cat {base_path}/animation-extraction/animation-tokens.json | grep -q "duration" && echo "valid")

# Count animation types
bash(cat animation-tokens.json | grep -c "\"keyframes\":")
```

### File Operations

```bash
# Load design tokens context
bash(test -f {base_path}/style-extraction/style-1/design-tokens.json && cat it)

# Verify output
bash(ls {base_path}/animation-extraction/)
```

## Output Structure

```
{base_path}/
├── .intermediates/                  # Intermediate analysis files
│   └── animation-analysis/
│       ├── animations-{target}.json      # Extracted CSS (URL mode only)
│       └── analysis-options.json         # Generated questions + user answers (embedded)
└── animation-extraction/            # Final animation system
    └── animation-tokens.json        # Production-ready animation tokens
```

## animation-tokens.json Format

```json
{
  "duration": {
    "instant": "0ms",
    "fast": "150ms",
    "normal": "300ms",
    "slow": "500ms",
    "very-slow": "800ms"
  },
  "easing": {
    "linear": "linear",
    "ease-in": "cubic-bezier(0.4, 0, 1, 1)",
    "ease-out": "cubic-bezier(0, 0, 0.2, 1)",
    "ease-in-out": "cubic-bezier(0.4, 0, 0.2, 1)",
    "spring": "cubic-bezier(0.34, 1.56, 0.64, 1)"
  },
  "transitions": {
    "color": {"property": "...", "duration": "var(--duration-fast)", "easing": "..."},
    "transform": {"property": "...", "duration": "...", "easing": "..."}
  },
  "keyframes": {
    "fadeIn": {"0%": {...}, "100%": {...}},
    "slideInUp": {...}
  },
  "interactions": {
    "button-hover": {"properties": [...], "duration": "...", "transform": "..."},
    "card-hover": {...}
  },
  "page_transitions": {...},
  "scroll_animations": {...}
}
```

**Requirements**: CSS var() format, valid cubic-bezier values, prefers-reduced-motion support

## Error Handling

### Common Errors

```
ERROR: No image references found
→ Provide valid --images glob pattern or proceed with prompt-only mode

ERROR: Invalid image format
→ Skips unsupported files, continues with valid images

ERROR: Invalid cubic-bezier values
→ Validates and corrects to nearest standard easing
```

### Recovery Strategies

- **Image loading failure**: Falls back to prompt-only specification mode
- **Partial image set**: Supplements with default values and best practices
- **Invalid data**: Validates and uses fallback values

## Key Features

- **Prompt & Image Inference** - Analyzes design intent from textual descriptions and visual references (Phase 0)
- **Agent-Generated Questions** - Context-aware specification questions with visual previews (Phase 1)
- **Visual Previews** - Timeline representations, easing curve ASCII art, and animation sequences for each option
- **Optional User Interaction** - User answers questions only when `--interactive` flag present (Phase 1.5)
- **Non-Interactive Mode** - Default behavior uses inferred patterns + best practices (no user questions)
- **Hybrid Strategy** - Combines image analysis with user preferences (when interactive)
- **No MCP Dependencies** - Pure AI-driven inference from visual and textual inputs
- **Context-Aware** - Aligns with existing design tokens
- **Production-Ready** - CSS var() format, accessibility support
- **Comprehensive Coverage** - Transitions, keyframes, interactions, scroll animations
- **Clear Phase Separation** - Question generation (Agent) → User confirmation (Optional) → Token synthesis (Agent)


