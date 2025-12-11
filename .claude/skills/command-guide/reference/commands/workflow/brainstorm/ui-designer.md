---
name: ui-designer
description: Generate or update ui-designer/analysis.md addressing guidance-specification discussion points for UI design perspective
argument-hint: "optional topic - uses existing framework if available"
allowed-tools: Task(conceptual-planning-agent), TodoWrite(*), Read(*), Write(*)
---

## 🎨 **UI Designer Analysis Generator**

### Purpose
**Specialized command for generating ui-designer/analysis.md** that addresses guidance-specification.md discussion points from UI/UX design perspective. Creates or updates role-specific analysis with framework references.

### Core Function
- **Framework-based Analysis**: Address each discussion point in guidance-specification.md
- **UI/UX Focus**: User experience, interface design, and accessibility perspective
- **Update Mechanism**: Create new or update existing analysis.md
- **Agent Delegation**: Use conceptual-planning-agent for analysis generation

### Analysis Scope
- **Visual Design**: Color palettes, typography, spacing, and visual hierarchy implementation
- **High-Fidelity Mockups**: Polished, pixel-perfect interface designs
- **Design System Implementation**: Component libraries, design tokens, and style guides
- **Micro-Interactions & Animations**: Transition effects, loading states, and interactive feedback
- **Responsive Design**: Layout adaptations for different screen sizes and devices

### Role Boundaries & Responsibilities

#### **What This Role OWNS (Concrete Visual Interface Implementation)**
- **Visual Design Language**: Colors, typography, iconography, spacing, and overall aesthetic
- **High-Fidelity Mockups**: Polished designs showing exactly how the interface will look
- **Design System Components**: Building and documenting reusable UI components (buttons, inputs, cards, etc.)
- **Design Tokens**: Defining variables for colors, spacing, typography that can be used in code
- **Micro-Interactions**: Hover states, transitions, animations, and interactive feedback details
- **Responsive Layouts**: Adapting designs for mobile, tablet, and desktop breakpoints

#### **What This Role DOES NOT Own (Defers to Other Roles)**
- **User Research & Personas**: User behavior analysis and needs assessment → Defers to **UX Expert**
- **Information Architecture**: Content structure and navigation strategy → Defers to **UX Expert**
- **Low-Fidelity Wireframes**: Structural layouts without visual design → Defers to **UX Expert**

#### **Handoff Points**
- **FROM UX Expert**: Receives wireframes, user flows, and information architecture as the foundation for visual design
- **TO Frontend Developers**: Provides design specifications, component libraries, and design tokens for implementation
- **WITH API Designer**: Coordinates on data presentation and form validation feedback (visual aspects only)

## ⚙️ **Execution Protocol**

### Phase 1: Session & Framework Detection
```bash
# Check active session and framework
CHECK: find .workflow/active/ -name "WFS-*" -type d
IF active_session EXISTS:
    session_id = get_active_session()
    brainstorm_dir = .workflow/active/WFS-{session}/.brainstorming/

    CHECK: brainstorm_dir/guidance-specification.md
    IF EXISTS:
        framework_mode = true
        load_framework = true
    ELSE:
        IF topic_provided:
            framework_mode = false  # Create analysis without framework
        ELSE:
            ERROR: "No framework found and no topic provided"
```

### Phase 2: Analysis Mode Detection
```bash
# Determine execution mode
IF framework_mode == true:
    mode = "framework_based_analysis"
    topic_ref = load_framework_topic()
    discussion_points = extract_framework_points()
ELSE:
    mode = "standalone_analysis"
    topic_ref = provided_topic
    discussion_points = generate_basic_structure()
```

### Phase 3: Agent Execution with Flow Control
**Framework-Based Analysis Generation**

```bash
Task(conceptual-planning-agent): "
[FLOW_CONTROL]

Execute ui-designer analysis for existing topic framework

## Context Loading
ASSIGNED_ROLE: ui-designer
OUTPUT_LOCATION: .workflow/active/WFS-{session}/.brainstorming/ui-designer/
ANALYSIS_MODE: {framework_mode ? "framework_based" : "standalone"}

## Flow Control Steps
1. **load_topic_framework**
   - Action: Load structured topic discussion framework
   - Command: Read(.workflow/active/WFS-{session}/.brainstorming/guidance-specification.md)
   - Output: topic_framework_content

2. **load_role_template**
   - Action: Load ui-designer planning template
   - Command: bash($(cat ~/.claude/workflows/cli-templates/planning-roles/ui-designer.md))
   - Output: role_template_guidelines

3. **load_session_metadata**
   - Action: Load session metadata and existing context
   - Command: Read(.workflow/active/WFS-{session}/workflow-session.json)
   - Output: session_context

## Analysis Requirements
**Framework Reference**: Address all discussion points in guidance-specification.md from UI/UX perspective
**Role Focus**: User experience design, interface optimization, accessibility compliance
**Structured Approach**: Create analysis.md addressing framework discussion points
**Template Integration**: Apply role template guidelines within framework structure

## Expected Deliverables
1. **analysis.md**: Comprehensive UI/UX analysis addressing all framework discussion points
2. **Framework Reference**: Include @../guidance-specification.md reference in analysis

## Completion Criteria
- Address each discussion point from guidance-specification.md with UI/UX design expertise
- Provide actionable design recommendations and interface solutions
- Include accessibility considerations and WCAG compliance planning
- Reference framework document using @ notation for integration
"
```

## 📋 **TodoWrite Integration**

### Workflow Progress Tracking
```javascript
TodoWrite({
  todos: [
    {
      content: "Detect active session and locate topic framework",
      status: "in_progress",
      activeForm: "Detecting session and framework"
    },
    {
      content: "Load guidance-specification.md and session metadata for context",
      status: "pending",
      activeForm: "Loading framework and session context"
    },
    {
      content: "Execute ui-designer analysis using conceptual-planning-agent with FLOW_CONTROL",
      status: "pending",
      activeForm: "Executing ui-designer framework analysis"
    },
    {
      content: "Generate analysis.md addressing all framework discussion points",
      status: "pending",
      activeForm: "Generating structured ui-designer analysis"
    },
    {
      content: "Update workflow-session.json with ui-designer completion status",
      status: "pending",
      activeForm: "Updating session metadata"
    }
  ]
});
```

## 📊 **Output Structure**

### Framework-Based Analysis
```
.workflow/active/WFS-{session}/.brainstorming/ui-designer/
└── analysis.md    # Structured analysis addressing guidance-specification.md discussion points
```

### Analysis Document Structure
```markdown
# UI Designer Analysis: [Topic from Framework]

## Framework Reference
**Topic Framework**: @../guidance-specification.md
**Role Focus**: UI/UX Design perspective

## Discussion Points Analysis
[Address each point from guidance-specification.md with UI/UX expertise]

### Core Requirements (from framework)
[UI/UX perspective on requirements]

### Technical Considerations (from framework)
[Interface and design system considerations]

### User Experience Factors (from framework)
[Detailed UX analysis and recommendations]

### Implementation Challenges (from framework)
[Design implementation and accessibility considerations]

### Success Metrics (from framework)
[UX metrics and usability success criteria]

## UI/UX Specific Recommendations
[Role-specific design recommendations and solutions]

---
*Generated by ui-designer analysis addressing structured framework*
```

## 🔄 **Session Integration**

### Completion Status Update
```json
{
  "ui_designer": {
    "status": "completed",
    "framework_addressed": true,
    "output_location": ".workflow/active/WFS-{session}/.brainstorming/ui-designer/analysis.md",
    "framework_reference": "@../guidance-specification.md"
  }
}
```

### Integration Points
- **Framework Reference**: @../guidance-specification.md for structured discussion points
- **Cross-Role Synthesis**: UI/UX insights available for synthesis-report.md integration
- **Agent Autonomy**: Independent execution with framework guidance
