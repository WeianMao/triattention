---
name: data-architect
description: Generate or update data-architect/analysis.md addressing guidance-specification discussion points for data architecture perspective
argument-hint: "optional topic - uses existing framework if available"
allowed-tools: Task(conceptual-planning-agent), TodoWrite(*), Read(*), Write(*)
---

## 📊 **Data Architect Analysis Generator**

### Purpose
**Specialized command for generating data-architect/analysis.md** that addresses guidance-specification.md discussion points from data architecture perspective. Creates or updates role-specific analysis with framework references.

### Core Function
- **Framework-based Analysis**: Address each discussion point in guidance-specification.md
- **Data Architecture Focus**: Data models, pipelines, governance, and analytics perspective
- **Update Mechanism**: Create new or update existing analysis.md
- **Agent Delegation**: Use conceptual-planning-agent for analysis generation

### Analysis Scope
- **Data Model Design**: Efficient and scalable data models and schemas
- **Data Flow Design**: Data collection, processing, and storage workflows
- **Data Quality Management**: Data accuracy, completeness, and consistency
- **Analytics and Insights**: Data analysis and business intelligence solutions

### Role Boundaries & Responsibilities

#### **What This Role OWNS (Canonical Data Model - Source of Truth)**
- **Canonical Data Model**: The authoritative, system-wide data schema representing domain entities and relationships
- **Entity-Relationship Design**: Defining entities, attributes, relationships, and constraints
- **Data Normalization & Optimization**: Ensuring data integrity, reducing redundancy, and optimizing storage
- **Database Schema Design**: Physical database structures, indexes, partitioning strategies
- **Data Pipeline Architecture**: ETL/ELT processes, data warehousing, and analytics pipelines
- **Data Governance**: Data quality standards, retention policies, and compliance requirements

#### **What This Role DOES NOT Own (Defers to Other Roles)**
- **API Data Contracts**: Public-facing request/response schemas exposed by APIs → Defers to **API Designer**
- **System Integration Patterns**: How services communicate at the macro level → Defers to **System Architect**
- **UI Data Presentation**: How data is displayed to users → Defers to **UI Designer**

#### **Handoff Points**
- **TO API Designer**: Provides canonical data model that API Designer translates into public API data contracts (as projection/view)
- **TO System Architect**: Provides data flow requirements and storage constraints to inform system design
- **FROM System Architect**: Receives system-level integration requirements and scalability constraints

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

Execute data-architect analysis for existing topic framework

## Context Loading
ASSIGNED_ROLE: data-architect
OUTPUT_LOCATION: .workflow/active/WFS-{session}/.brainstorming/data-architect/
ANALYSIS_MODE: {framework_mode ? "framework_based" : "standalone"}

## Flow Control Steps
1. **load_topic_framework**
   - Action: Load structured topic discussion framework
   - Command: Read(.workflow/active/WFS-{session}/.brainstorming/guidance-specification.md)
   - Output: topic_framework_content

2. **load_role_template**
   - Action: Load data-architect planning template
   - Command: bash($(cat ~/.claude/workflows/cli-templates/planning-roles/data-architect.md))
   - Output: role_template_guidelines

3. **load_session_metadata**
   - Action: Load session metadata and existing context
   - Command: Read(.workflow/active/WFS-{session}/workflow-session.json)
   - Output: session_context

## Analysis Requirements
**Framework Reference**: Address all discussion points in guidance-specification.md from data architecture perspective
**Role Focus**: Data models, pipelines, governance, analytics platforms
**Structured Approach**: Create analysis.md addressing framework discussion points
**Template Integration**: Apply role template guidelines within framework structure

## Expected Deliverables
1. **analysis.md**: Comprehensive data architecture analysis addressing all framework discussion points
2. **Framework Reference**: Include @../guidance-specification.md reference in analysis

## Completion Criteria
- Address each discussion point from guidance-specification.md with data architecture expertise
- Provide data model designs, pipeline architectures, and governance strategies
- Include scalability, performance, and quality considerations
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
      content: "Execute data-architect analysis using conceptual-planning-agent with FLOW_CONTROL",
      status: "pending",
      activeForm: "Executing data-architect framework analysis"
    },
    {
      content: "Generate analysis.md addressing all framework discussion points",
      status: "pending",
      activeForm: "Generating structured data-architect analysis"
    },
    {
      content: "Update workflow-session.json with data-architect completion status",
      status: "pending",
      activeForm: "Updating session metadata"
    }
  ]
});
```

## 📊 **Output Structure**

### Framework-Based Analysis
```
.workflow/active/WFS-{session}/.brainstorming/data-architect/
└── analysis.md    # Structured analysis addressing guidance-specification.md discussion points
```

### Analysis Document Structure
```markdown
# Data Architect Analysis: [Topic from Framework]

## Framework Reference
**Topic Framework**: @../guidance-specification.md
**Role Focus**: Data Architecture perspective

## Discussion Points Analysis
[Address each point from guidance-specification.md with data architecture expertise]

### Core Requirements (from framework)
[Data architecture perspective on requirements]

### Technical Considerations (from framework)
[Data model, pipeline, and storage considerations]

### User Experience Factors (from framework)
[Data access patterns and analytics user experience]

### Implementation Challenges (from framework)
[Data migration, quality, and governance challenges]

### Success Metrics (from framework)
[Data quality metrics and analytics success criteria]

## Data Architecture Specific Recommendations
[Role-specific data architecture recommendations and solutions]

---
*Generated by data-architect analysis addressing structured framework*
```

## 🔄 **Session Integration**

### Completion Status Update
```json
{
  "data_architect": {
    "status": "completed",
    "framework_addressed": true,
    "output_location": ".workflow/active/WFS-{session}/.brainstorming/data-architect/analysis.md",
    "framework_reference": "@../guidance-specification.md"
  }
}
```

### Integration Points
- **Framework Reference**: @../guidance-specification.md for structured discussion points
- **Cross-Role Synthesis**: Data architecture insights available for synthesis-report.md integration
- **Agent Autonomy**: Independent execution with framework guidance
