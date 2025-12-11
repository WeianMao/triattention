---
name: scrum-master
description: Generate or update scrum-master/analysis.md addressing guidance-specification discussion points for Agile process perspective
argument-hint: "optional topic - uses existing framework if available"
allowed-tools: Task(conceptual-planning-agent), TodoWrite(*), Read(*), Write(*)
---

## 🎯 **Scrum Master Analysis Generator**

### Purpose
**Specialized command for generating scrum-master/analysis.md** that addresses guidance-specification.md discussion points from agile process and team collaboration perspective. Creates or updates role-specific analysis with framework references.

### Core Function
- **Framework-based Analysis**: Address each discussion point in guidance-specification.md
- **Agile Process Focus**: Sprint planning, team dynamics, and delivery optimization
- **Update Mechanism**: Create new or update existing analysis.md
- **Agent Delegation**: Use conceptual-planning-agent for analysis generation

### Analysis Scope
- **Sprint Planning**: Task breakdown, estimation, and iteration planning
- **Team Collaboration**: Communication patterns, impediment removal, and facilitation
- **Process Optimization**: Agile ceremonies, retrospectives, and continuous improvement
- **Delivery Management**: Velocity tracking, burndown analysis, and release planning

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

Execute scrum-master analysis for existing topic framework

## Context Loading
ASSIGNED_ROLE: scrum-master
OUTPUT_LOCATION: .workflow/active/WFS-{session}/.brainstorming/scrum-master/
ANALYSIS_MODE: {framework_mode ? "framework_based" : "standalone"}

## Flow Control Steps
1. **load_topic_framework**
   - Action: Load structured topic discussion framework
   - Command: Read(.workflow/active/WFS-{session}/.brainstorming/guidance-specification.md)
   - Output: topic_framework_content

2. **load_role_template**
   - Action: Load scrum-master planning template
   - Command: bash($(cat ~/.claude/workflows/cli-templates/planning-roles/scrum-master.md))
   - Output: role_template_guidelines

3. **load_session_metadata**
   - Action: Load session metadata and existing context
   - Command: Read(.workflow/active/WFS-{session}/workflow-session.json)
   - Output: session_context

## Analysis Requirements
**Framework Reference**: Address all discussion points in guidance-specification.md from agile process and team collaboration perspective
**Role Focus**: Sprint planning, team dynamics, process optimization, delivery management
**Structured Approach**: Create analysis.md addressing framework discussion points
**Template Integration**: Apply role template guidelines within framework structure

## Expected Deliverables
1. **analysis.md**: Comprehensive agile process analysis addressing all framework discussion points
2. **Framework Reference**: Include @../guidance-specification.md reference in analysis

## Completion Criteria
- Address each discussion point from guidance-specification.md with scrum mastery expertise
- Provide actionable sprint planning and team facilitation strategies
- Include process optimization and impediment removal insights
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
      content: "Execute scrum-master analysis using conceptual-planning-agent with FLOW_CONTROL",
      status: "pending",
      activeForm: "Executing scrum-master framework analysis"
    },
    {
      content: "Generate analysis.md addressing all framework discussion points",
      status: "pending",
      activeForm: "Generating structured scrum-master analysis"
    },
    {
      content: "Update workflow-session.json with scrum-master completion status",
      status: "pending",
      activeForm: "Updating session metadata"
    }
  ]
});
```

## 📊 **Output Structure**

### Framework-Based Analysis
```
.workflow/active/WFS-{session}/.brainstorming/scrum-master/
└── analysis.md    # Structured analysis addressing guidance-specification.md discussion points
```

### Analysis Document Structure
```markdown
# Scrum Master Analysis: [Topic from Framework]

## Framework Reference
**Topic Framework**: @../guidance-specification.md
**Role Focus**: Agile Process & Team Collaboration perspective

## Discussion Points Analysis
[Address each point from guidance-specification.md with scrum mastery expertise]

### Core Requirements (from framework)
[Sprint planning and iteration breakdown perspective]

### Technical Considerations (from framework)
[Technical debt management and process considerations]

### User Experience Factors (from framework)
[User story refinement and acceptance criteria analysis]

### Implementation Challenges (from framework)
[Impediment identification and removal strategies]

### Success Metrics (from framework)
[Velocity tracking, burndown metrics, and team performance indicators]

## Scrum Master Specific Recommendations
[Role-specific agile process optimization and team facilitation strategies]

---
*Generated by scrum-master analysis addressing structured framework*
```

## 🔄 **Session Integration**

### Completion Status Update
```json
{
  "scrum_master": {
    "status": "completed",
    "framework_addressed": true,
    "output_location": ".workflow/active/WFS-{session}/.brainstorming/scrum-master/analysis.md",
    "framework_reference": "@../guidance-specification.md"
  }
}
```

### Integration Points
- **Framework Reference**: @../guidance-specification.md for structured discussion points
- **Cross-Role Synthesis**: Agile process insights available for synthesis-report.md integration
- **Agent Autonomy**: Independent execution with framework guidance
