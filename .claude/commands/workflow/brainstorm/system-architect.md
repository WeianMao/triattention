---
name: system-architect
description: Generate or update system-architect/analysis.md addressing guidance-specification discussion points for system architecture perspective
argument-hint: "optional topic - uses existing framework if available"
allowed-tools: Task(conceptual-planning-agent), TodoWrite(*), Read(*), Write(*)
---

## 🏗️ **System Architect Analysis Generator**

### Purpose
**Specialized command for generating system-architect/analysis.md** that addresses guidance-specification.md discussion points from system architecture perspective. Creates or updates role-specific analysis with framework references.

### Core Function
- **Framework-based Analysis**: Address each discussion point in guidance-specification.md
- **Architecture Focus**: Technical architecture, scalability, and system design perspective
- **Update Mechanism**: Create new or update existing analysis.md
- **Agent Delegation**: Use conceptual-planning-agent for analysis generation

### Analysis Scope
- **Technical Architecture**: Scalable and maintainable system design
- **Technology Selection**: Stack evaluation and architectural decisions
- **Performance & Scalability**: Capacity planning and optimization strategies
- **Integration Patterns**: System communication and data flow design

### Role Boundaries & Responsibilities

#### **What This Role OWNS (Macro-Architecture)**
- **System-Level Architecture**: Service boundaries, deployment topology, and system composition
- **Cross-Service Communication Patterns**: Choosing between microservices/monolithic, event-driven/request-response, sync/async patterns
- **Technology Stack Decisions**: Language, framework, database, and infrastructure choices
- **Non-Functional Requirements**: Scalability, performance, availability, disaster recovery, and monitoring strategies
- **Integration Planning**: How systems and services connect at the macro level (not specific API contracts)

#### **What This Role DOES NOT Own (Defers to Other Roles)**
- **API Contract Details**: Specific endpoint definitions, URL structures, HTTP methods → Defers to **API Designer**
- **Data Schemas**: Detailed data model design and entity relationships → Defers to **Data Architect**
- **UI/UX Design**: Interface design and user experience → Defers to **UX Expert** and **UI Designer**

#### **Handoff Points**
- **TO API Designer**: Provides architectural constraints (REST vs GraphQL, sync vs async) that define the API design space
- **TO Data Architect**: Provides system-level data flow requirements and integration patterns
- **FROM Data Architect**: Receives canonical data model to inform system integration design

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
# Check existing analysis
CHECK: brainstorm_dir/system-architect/analysis.md
IF EXISTS:
    SHOW existing analysis summary
    ASK: "Analysis exists. Do you want to:"
    OPTIONS:
      1. "Update with new insights" → Update existing
      2. "Replace completely" → Generate new
      3. "Cancel" → Exit without changes
ELSE:
    CREATE new analysis
```

### Phase 3: Agent Task Generation
**Framework-Based Analysis** (when guidance-specification.md exists):
```bash
Task(subagent_type="conceptual-planning-agent",
     prompt="Generate system architect analysis addressing topic framework

     ## Framework Integration Required
     **MANDATORY**: Load and address guidance-specification.md discussion points
     **Framework Reference**: @{session.brainstorm_dir}/guidance-specification.md
     **Output Location**: {session.brainstorm_dir}/system-architect/analysis.md

     ## Analysis Requirements
     1. **Load Topic Framework**: Read guidance-specification.md completely
     2. **Address Each Discussion Point**: Respond to all 5 framework sections from system architecture perspective
     3. **Include Framework Reference**: Start analysis.md with @../guidance-specification.md
     4. **Technical Focus**: Emphasize scalability, architecture patterns, technology decisions
     5. **Structured Response**: Use framework structure for analysis organization

     ## Framework Sections to Address
     - Core Requirements (from architecture perspective)
     - Technical Considerations (detailed architectural analysis)
     - User Experience Factors (technical UX considerations)
     - Implementation Challenges (architecture risks and solutions)
     - Success Metrics (technical metrics and monitoring)

     ## Output Structure Required
     ```markdown
     # System Architect Analysis: [Topic]

     **Framework Reference**: @../guidance-specification.md
     **Role Focus**: System Architecture and Technical Design

     ## Core Requirements Analysis
     [Address framework requirements from architecture perspective]

     ## Technical Considerations
     [Detailed architectural analysis]

     ## User Experience Factors
     [Technical aspects of UX implementation]

     ## Implementation Challenges
     [Architecture risks and mitigation strategies]

     ## Success Metrics
     [Technical metrics and system monitoring]

     ## Architecture-Specific Recommendations
     [Detailed technical recommendations]
     ```",
     description="Generate system architect framework-based analysis")
```

### Phase 4: Update Mechanism
**Analysis Update Process**:
```bash
# For existing analysis updates
IF update_mode = "incremental":
    Task(subagent_type="conceptual-planning-agent",
         prompt="Update existing system architect analysis

         ## Current Analysis Context
         **Existing Analysis**: @{session.brainstorm_dir}/system-architect/analysis.md
         **Framework Reference**: @{session.brainstorm_dir}/guidance-specification.md

         ## Update Requirements
         1. **Preserve Structure**: Maintain existing analysis structure
         2. **Add New Insights**: Integrate new technical insights and recommendations
         3. **Framework Alignment**: Ensure continued alignment with topic framework
         4. **Technical Updates**: Add new architecture patterns, technology considerations
         5. **Maintain References**: Keep @../guidance-specification.md reference

         ## Update Instructions
         - Read existing analysis completely
         - Identify areas for enhancement or new insights
         - Add technical depth while preserving original structure
         - Update recommendations with new architectural approaches
         - Maintain framework discussion point addressing",
         description="Update system architect analysis incrementally")
```

## Document Structure

### Output Files
```
.workflow/active/WFS-[topic]/.brainstorming/
├── guidance-specification.md          # Input: Framework (if exists)
└── system-architect/
    └── analysis.md            # ★ OUTPUT: Framework-based analysis
```

### Analysis Structure
**Required Elements**:
- **Framework Reference**: @../guidance-specification.md (if framework exists)
- **Role Focus**: System Architecture and Technical Design perspective
- **5 Framework Sections**: Address each framework discussion point
- **Technical Recommendations**: Architecture-specific insights and solutions
- How should we design APIs and manage versioning?

**4. Performance and Scalability**
- Where are the current system performance bottlenecks?
- How should we handle traffic growth and scaling demands?
- What database scaling and optimization strategies are needed?

## ⚡ **Two-Step Execution Flow**

### ⚠️ Session Management - FIRST STEP
Session detection and selection:
```bash
# Check for existing sessions
existing_sessions=$(find .workflow/active/ -name "WFS-*" -type d 2>/dev/null)
if [ multiple_sessions ]; then
  prompt_user_to_select_session()
else
  use_existing_or_create_new()
fi
```

### Step 1: Context Gathering Phase
**System Architect Perspective Questioning**

Before agent assignment, gather comprehensive system architecture context:

#### 📋 Role-Specific Questions
1. **Scale & Performance Requirements**
   - Expected user load and traffic patterns?
   - Performance requirements (latency, throughput)?
   - Data volume and growth projections?

2. **Technical Constraints & Environment**
   - Existing technology stack and constraints?
   - Integration requirements with external systems?
   - Infrastructure and deployment environment?

3. **Architecture Complexity & Patterns**
   - Microservices vs monolithic considerations?
   - Data consistency and transaction requirements?
   - Event-driven vs request-response patterns?

4. **Non-Functional Requirements**
   - High availability and disaster recovery needs?
   - Security and compliance requirements?
   - Monitoring and observability expectations?

#### Context Validation
- **Minimum Response**: Each answer must be ≥50 characters
- **Re-prompting**: Insufficient detail triggers follow-up questions
- **Context Storage**: Save responses to `.brainstorming/system-architect-context.md`

### Step 2: Agent Assignment with Flow Control
**Dedicated Agent Execution**

```bash
Task(conceptual-planning-agent): "
[FLOW_CONTROL]

Execute dedicated system-architect conceptual analysis for: {topic}

ASSIGNED_ROLE: system-architect
OUTPUT_LOCATION: .brainstorming/system-architect/
USER_CONTEXT: {validated_responses_from_context_gathering}

Flow Control Steps:
[
  {
    \"step\": \"load_role_template\",
    \"action\": \"Load system-architect planning template\",
    \"command\": \"bash($(cat ~/.claude/workflows/cli-templates/planning-roles/system-architect.md))\",
    \"output_to\": \"role_template\"
  }
]

Conceptual Analysis Requirements:
- Apply system-architect perspective to topic analysis
- Focus on architectural patterns, scalability, and integration points
- Use loaded role template framework for analysis structure
- Generate role-specific deliverables in designated output location
- Address all user context from questioning phase

Deliverables:
- analysis.md: Main system architecture analysis
- recommendations.md: Architecture recommendations
- deliverables/: Architecture-specific outputs as defined in role template

Embody system-architect role expertise for comprehensive conceptual planning."
```

### Progress Tracking
TodoWrite tracking for two-step process:
```json
[
  {"content": "Gather system architect context through role-specific questioning", "status": "in_progress", "activeForm": "Gathering context"},
  {"content": "Validate context responses and save to system-architect-context.md", "status": "pending", "activeForm": "Validating context"},
  {"content": "Load system-architect planning template via flow control", "status": "pending", "activeForm": "Loading template"},
  {"content": "Execute dedicated conceptual-planning-agent for system-architect role", "status": "pending", "activeForm": "Executing agent"}
]
```

## 📊 **Output Specification**

### Output Location
```
.workflow/active/WFS-{topic-slug}/.brainstorming/system-architect/
├── analysis.md                 # Primary architecture analysis
├── architecture-design.md      # Detailed system design and diagrams
├── technology-stack.md         # Technology stack recommendations and justifications
└── integration-plan.md         # System integration and API strategies
```

### Document Templates

#### analysis.md Structure
```markdown
# System Architecture Analysis: {Topic}
*Generated: {timestamp}*

## Executive Summary
[Key architectural findings and recommendations overview]

## Current State Assessment
### Existing Architecture Overview
### Technical Stack Analysis
### Performance Bottlenecks
### Technical Debt Assessment

## Requirements Analysis
### Functional Requirements
### Non-Functional Requirements
- Performance: [Response time, throughput requirements]
- Scalability: [User growth, data volume expectations]
- Availability: [Uptime requirements]
- Security: [Security requirements]

## Proposed Architecture
### High-Level Architecture Design
### Component Breakdown
### Data Flow Diagrams
### Technology Stack Recommendations

## Implementation Strategy
### Migration Planning
### Risk Mitigation
### Performance Optimization
### Security Considerations

## Scalability and Maintenance
### Horizontal Scaling Strategy
### Monitoring and Observability
### Deployment Strategy
### Long-term Maintenance Plan
```

## 🔄 **Session Integration**

### Status Synchronization
Upon completion, update `workflow-session.json`:
```json
{
  "phases": {
    "BRAINSTORM": {
      "system_architect": {
        "status": "completed",
        "completed_at": "timestamp",
        "output_directory": ".workflow/active/WFS-{topic}/.brainstorming/system-architect/",
        "key_insights": ["scalability_bottleneck", "architecture_pattern", "technology_recommendation"]
      }
    }
  }
}
```

### Cross-Role Collaboration
System architect perspective provides:
- **Technical Constraints and Possibilities** → Product Manager
- **Architecture Requirements and Limitations** → UI Designer
- **Data Architecture Requirements** → Data Architect
- **Security Architecture Framework** → Security Expert
- **Technical Implementation Framework** → Feature Planner

## ✅ **Quality Assurance**

### Required Analysis Elements
- [ ] Clear architecture diagrams and component designs
- [ ] Detailed technology stack evaluation and recommendations
- [ ] Scalability and performance analysis with metrics
- [ ] System integration and API design specifications
- [ ] Comprehensive risk assessment and mitigation strategies

### Architecture Design Principles
- [ ] **Scalability**: System can handle growth in users and data
- [ ] **Maintainability**: Clear code structure, easy to modify and extend
- [ ] **Reliability**: Built-in fault tolerance and recovery mechanisms
- [ ] **Security**: Integrated security controls and protection measures
- [ ] **Performance**: Meets response time and throughput requirements

### Technical Decision Validation
- [ ] Technology choices have thorough justification and comparison analysis
- [ ] Architectural patterns align with business requirements and constraints
- [ ] Integration solutions consider compatibility and maintenance costs
- [ ] Deployment strategies are feasible with acceptable risk levels
- [ ] Monitoring and operations strategies are comprehensive and actionable

### Implementation Readiness
- [ ] **Technical Feasibility**: All proposed solutions are technically achievable
- [ ] **Resource Planning**: Resource requirements clearly defined and realistic
- [ ] **Risk Management**: Technical risks identified with mitigation plans
- [ ] **Performance Validation**: Architecture can meet performance requirements
- [ ] **Evolution Strategy**: Design allows for future growth and changes
