---
name: subject-matter-expert
description: Generate or update subject-matter-expert/analysis.md addressing guidance-specification discussion points for domain expertise perspective
argument-hint: "optional topic - uses existing framework if available"
allowed-tools: Task(conceptual-planning-agent), TodoWrite(*), Read(*), Write(*)
---

## 🎯 **Subject Matter Expert Analysis Generator**

### Purpose
**Specialized command for generating subject-matter-expert/analysis.md** that addresses guidance-specification.md discussion points from domain knowledge and technical expertise perspective. Creates or updates role-specific analysis with framework references.

### Core Function
- **Framework-based Analysis**: Address each discussion point in guidance-specification.md
- **Domain Expertise Focus**: Deep technical knowledge, industry standards, and best practices
- **Update Mechanism**: Create new or update existing analysis.md
- **Agent Delegation**: Use conceptual-planning-agent for analysis generation

### Analysis Scope
- **Domain Knowledge**: Industry-specific expertise, regulatory requirements, and compliance
- **Technical Standards**: Best practices, design patterns, and architectural guidelines
- **Risk Assessment**: Technical debt, scalability concerns, and maintenance implications
- **Knowledge Transfer**: Documentation strategies, training requirements, and expertise sharing

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

Execute subject-matter-expert analysis for existing topic framework

## Context Loading
ASSIGNED_ROLE: subject-matter-expert
OUTPUT_LOCATION: .workflow/active/WFS-{session}/.brainstorming/subject-matter-expert/
ANALYSIS_MODE: {framework_mode ? "framework_based" : "standalone"}

## Flow Control Steps
1. **load_topic_framework**
   - Action: Load structured topic discussion framework
   - Command: Read(.workflow/active/WFS-{session}/.brainstorming/guidance-specification.md)
   - Output: topic_framework_content

2. **load_role_template**
   - Action: Load subject-matter-expert planning template
   - Command: bash($(cat ~/.claude/workflows/cli-templates/planning-roles/subject-matter-expert.md))
   - Output: role_template_guidelines

3. **load_session_metadata**
   - Action: Load session metadata and existing context
   - Command: Read(.workflow/active/WFS-{session}/workflow-session.json)
   - Output: session_context

## Analysis Requirements
**Framework Reference**: Address all discussion points in guidance-specification.md from domain expertise and technical standards perspective
**Role Focus**: Domain knowledge, technical standards, risk assessment, knowledge transfer
**Structured Approach**: Create analysis.md addressing framework discussion points
**Template Integration**: Apply role template guidelines within framework structure

## Expected Deliverables
1. **analysis.md**: Comprehensive domain expertise analysis addressing all framework discussion points
2. **Framework Reference**: Include @../guidance-specification.md reference in analysis

## Completion Criteria
- Address each discussion point from guidance-specification.md with subject matter expertise
- Provide actionable technical standards and best practices recommendations
- Include risk assessment and compliance considerations
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
      content: "Execute subject-matter-expert analysis using conceptual-planning-agent with FLOW_CONTROL",
      status: "pending",
      activeForm: "Executing subject-matter-expert framework analysis"
    },
    {
      content: "Generate analysis.md addressing all framework discussion points",
      status: "pending",
      activeForm: "Generating structured subject-matter-expert analysis"
    },
    {
      content: "Update workflow-session.json with subject-matter-expert completion status",
      status: "pending",
      activeForm: "Updating session metadata"
    }
  ]
});
```

## 📊 **Output Structure**

### Framework-Based Analysis
```
.workflow/active/WFS-{session}/.brainstorming/subject-matter-expert/
└── analysis.md    # Structured analysis addressing guidance-specification.md discussion points
```

### Analysis Document Structure
```markdown
# Subject Matter Expert Analysis: [Topic from Framework]

## Framework Reference
**Topic Framework**: @../guidance-specification.md
**Role Focus**: Domain Expertise & Technical Standards perspective

## Discussion Points Analysis
[Address each point from guidance-specification.md with subject matter expertise]

### Core Requirements (from framework)
[Domain-specific requirements and industry standards perspective]

### Technical Considerations (from framework)
[Deep technical analysis, architectural patterns, and best practices]

### User Experience Factors (from framework)
[Domain-specific usability standards and industry conventions]

### Implementation Challenges (from framework)
[Technical risks, scalability concerns, and maintenance implications]

### Success Metrics (from framework)
[Domain-specific KPIs, compliance metrics, and quality standards]

## Subject Matter Expert Specific Recommendations
[Role-specific technical expertise and industry best practices]

---
*Generated by subject-matter-expert analysis addressing structured framework*
```

## 🔄 **Session Integration**

### Completion Status Update
```json
{
  "subject_matter_expert": {
    "status": "completed",
    "framework_addressed": true,
    "output_location": ".workflow/active/WFS-{session}/.brainstorming/subject-matter-expert/analysis.md",
    "framework_reference": "@../guidance-specification.md"
  }
}
```

### Integration Points
- **Framework Reference**: @../guidance-specification.md for structured discussion points
- **Cross-Role Synthesis**: Domain expertise insights available for synthesis-report.md integration
- **Agent Autonomy**: Independent execution with framework guidance
