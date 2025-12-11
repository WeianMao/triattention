---
name: product-manager
description: Generate or update product-manager/analysis.md addressing guidance-specification discussion points for product management perspective
argument-hint: "optional topic - uses existing framework if available"
allowed-tools: Task(conceptual-planning-agent), TodoWrite(*), Read(*), Write(*)
---

## 🎯 **Product Manager Analysis Generator**

### Purpose
**Specialized command for generating product-manager/analysis.md** that addresses guidance-specification.md discussion points from product strategy perspective. Creates or updates role-specific analysis with framework references.

### Core Function
- **Framework-based Analysis**: Address each discussion point in guidance-specification.md
- **Product Strategy Focus**: User needs, business value, and market positioning
- **Update Mechanism**: Create new or update existing analysis.md
- **Agent Delegation**: Use conceptual-planning-agent for analysis generation

### Analysis Scope
- **User Needs Analysis**: Target users, problems, and value propositions
- **Business Impact Assessment**: ROI, metrics, and commercial outcomes
- **Market Positioning**: Competitive analysis and differentiation
- **Product Strategy**: Roadmaps, priorities, and go-to-market approaches

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

Execute product-manager analysis for existing topic framework

## Context Loading
ASSIGNED_ROLE: product-manager
OUTPUT_LOCATION: .workflow/active/WFS-{session}/.brainstorming/product-manager/
ANALYSIS_MODE: {framework_mode ? "framework_based" : "standalone"}

## Flow Control Steps
1. **load_topic_framework**
   - Action: Load structured topic discussion framework
   - Command: Read(.workflow/active/WFS-{session}/.brainstorming/guidance-specification.md)
   - Output: topic_framework_content

2. **load_role_template**
   - Action: Load product-manager planning template
   - Command: bash($(cat ~/.claude/workflows/cli-templates/planning-roles/product-manager.md))
   - Output: role_template_guidelines

3. **load_session_metadata**
   - Action: Load session metadata and existing context
   - Command: Read(.workflow/active/WFS-{session}/workflow-session.json)
   - Output: session_context

## Analysis Requirements
**Framework Reference**: Address all discussion points in guidance-specification.md from product strategy perspective
**Role Focus**: User value, business impact, market positioning, product strategy
**Structured Approach**: Create analysis.md addressing framework discussion points
**Template Integration**: Apply role template guidelines within framework structure

## Expected Deliverables
1. **analysis.md**: Comprehensive product strategy analysis addressing all framework discussion points
2. **Framework Reference**: Include @../guidance-specification.md reference in analysis

## Completion Criteria
- Address each discussion point from guidance-specification.md with product management expertise
- Provide actionable business strategies and user value propositions
- Include market analysis and competitive positioning insights
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
      content: "Execute product-manager analysis using conceptual-planning-agent with FLOW_CONTROL",
      status: "pending",
      activeForm: "Executing product-manager framework analysis"
    },
    {
      content: "Generate analysis.md addressing all framework discussion points",
      status: "pending",
      activeForm: "Generating structured product-manager analysis"
    },
    {
      content: "Update workflow-session.json with product-manager completion status",
      status: "pending",
      activeForm: "Updating session metadata"
    }
  ]
});
```

## 📊 **Output Structure**

### Framework-Based Analysis
```
.workflow/active/WFS-{session}/.brainstorming/product-manager/
└── analysis.md    # Structured analysis addressing guidance-specification.md discussion points
```

### Analysis Document Structure
```markdown
# Product Manager Analysis: [Topic from Framework]

## Framework Reference
**Topic Framework**: @../guidance-specification.md
**Role Focus**: Product Strategy perspective

## Discussion Points Analysis
[Address each point from guidance-specification.md with product management expertise]

### Core Requirements (from framework)
[Product strategy perspective on user needs and requirements]

### Technical Considerations (from framework)
[Business and technical feasibility considerations]

### User Experience Factors (from framework)
[User value proposition and market positioning analysis]

### Implementation Challenges (from framework)
[Business execution and go-to-market considerations]

### Success Metrics (from framework)
[Product success metrics and business KPIs]

## Product Strategy Specific Recommendations
[Role-specific product management strategies and business solutions]

---
*Generated by product-manager analysis addressing structured framework*
```

## 🔄 **Session Integration**

### Completion Status Update
```json
{
  "product_manager": {
    "status": "completed",
    "framework_addressed": true,
    "output_location": ".workflow/active/WFS-{session}/.brainstorming/product-manager/analysis.md",
    "framework_reference": "@../guidance-specification.md"
  }
}
```

### Integration Points
- **Framework Reference**: @../guidance-specification.md for structured discussion points
- **Cross-Role Synthesis**: Product strategy insights available for synthesis-report.md integration
- **Agent Autonomy**: Independent execution with framework guidance
