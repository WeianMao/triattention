---
name: product-owner
description: Generate or update product-owner/analysis.md addressing guidance-specification discussion points for product ownership perspective
argument-hint: "optional topic - uses existing framework if available"
allowed-tools: Task(conceptual-planning-agent), TodoWrite(*), Read(*), Write(*)
---

## 🎯 **Product Owner Analysis Generator**

### Purpose
**Specialized command for generating product-owner/analysis.md** that addresses guidance-specification.md discussion points from product backlog and feature prioritization perspective. Creates or updates role-specific analysis with framework references.

### Core Function
- **Framework-based Analysis**: Address each discussion point in guidance-specification.md
- **Product Backlog Focus**: Feature prioritization, user stories, and acceptance criteria
- **Update Mechanism**: Create new or update existing analysis.md
- **Agent Delegation**: Use conceptual-planning-agent for analysis generation

### Analysis Scope
- **Backlog Management**: User story creation, refinement, and prioritization
- **Stakeholder Alignment**: Requirements gathering, value definition, and expectation management
- **Feature Prioritization**: ROI analysis, MoSCoW method, and value-driven delivery
- **Acceptance Criteria**: Definition of Done, acceptance testing, and quality standards

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

Execute product-owner analysis for existing topic framework

## Context Loading
ASSIGNED_ROLE: product-owner
OUTPUT_LOCATION: .workflow/active/WFS-{session}/.brainstorming/product-owner/
ANALYSIS_MODE: {framework_mode ? "framework_based" : "standalone"}

## Flow Control Steps
1. **load_topic_framework**
   - Action: Load structured topic discussion framework
   - Command: Read(.workflow/active/WFS-{session}/.brainstorming/guidance-specification.md)
   - Output: topic_framework_content

2. **load_role_template**
   - Action: Load product-owner planning template
   - Command: bash($(cat ~/.claude/workflows/cli-templates/planning-roles/product-owner.md))
   - Output: role_template_guidelines

3. **load_session_metadata**
   - Action: Load session metadata and existing context
   - Command: Read(.workflow/active/WFS-{session}/workflow-session.json)
   - Output: session_context

## Analysis Requirements
**Framework Reference**: Address all discussion points in guidance-specification.md from product backlog and feature prioritization perspective
**Role Focus**: Backlog management, stakeholder alignment, feature prioritization, acceptance criteria
**Structured Approach**: Create analysis.md addressing framework discussion points
**Template Integration**: Apply role template guidelines within framework structure

## Expected Deliverables
1. **analysis.md**: Comprehensive product ownership analysis addressing all framework discussion points
2. **Framework Reference**: Include @../guidance-specification.md reference in analysis

## Completion Criteria
- Address each discussion point from guidance-specification.md with product ownership expertise
- Provide actionable user stories and acceptance criteria definitions
- Include feature prioritization and stakeholder alignment strategies
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
      content: "Execute product-owner analysis using conceptual-planning-agent with FLOW_CONTROL",
      status: "pending",
      activeForm: "Executing product-owner framework analysis"
    },
    {
      content: "Generate analysis.md addressing all framework discussion points",
      status: "pending",
      activeForm: "Generating structured product-owner analysis"
    },
    {
      content: "Update workflow-session.json with product-owner completion status",
      status: "pending",
      activeForm: "Updating session metadata"
    }
  ]
});
```

## 📊 **Output Structure**

### Framework-Based Analysis
```
.workflow/active/WFS-{session}/.brainstorming/product-owner/
└── analysis.md    # Structured analysis addressing guidance-specification.md discussion points
```

### Analysis Document Structure
```markdown
# Product Owner Analysis: [Topic from Framework]

## Framework Reference
**Topic Framework**: @../guidance-specification.md
**Role Focus**: Product Backlog & Feature Prioritization perspective

## Discussion Points Analysis
[Address each point from guidance-specification.md with product ownership expertise]

### Core Requirements (from framework)
[User story formulation and backlog refinement perspective]

### Technical Considerations (from framework)
[Technical feasibility and implementation sequencing considerations]

### User Experience Factors (from framework)
[User value definition and acceptance criteria analysis]

### Implementation Challenges (from framework)
[Sprint planning, dependency management, and delivery strategies]

### Success Metrics (from framework)
[Feature adoption, value delivery metrics, and stakeholder satisfaction indicators]

## Product Owner Specific Recommendations
[Role-specific backlog management and feature prioritization strategies]

---
*Generated by product-owner analysis addressing structured framework*
```

## 🔄 **Session Integration**

### Completion Status Update
```json
{
  "product_owner": {
    "status": "completed",
    "framework_addressed": true,
    "output_location": ".workflow/active/WFS-{session}/.brainstorming/product-owner/analysis.md",
    "framework_reference": "@../guidance-specification.md"
  }
}
```

### Integration Points
- **Framework Reference**: @../guidance-specification.md for structured discussion points
- **Cross-Role Synthesis**: Product ownership insights available for synthesis-report.md integration
- **Agent Autonomy**: Independent execution with framework guidance
