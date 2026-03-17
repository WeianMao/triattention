---
name: api-designer
description: Generate or update api-designer/analysis.md addressing guidance-specification discussion points for API design perspective
argument-hint: "optional topic - uses existing framework if available"
allowed-tools: Task(conceptual-planning-agent), TodoWrite(*), Read(*), Write(*)
---

## 🔌 **API Designer Analysis Generator**

### Purpose
**Specialized command for generating api-designer/analysis.md** that addresses guidance-specification.md discussion points from backend API design perspective. Creates or updates role-specific analysis with framework references.

### Core Function
- **Framework-based Analysis**: Address each discussion point in guidance-specification.md
- **API Design Focus**: RESTful/GraphQL API design, endpoint structure, and contract definition
- **Update Mechanism**: Create new or update existing analysis.md
- **Agent Delegation**: Use conceptual-planning-agent for analysis generation

### Analysis Scope
- **API Architecture**: RESTful/GraphQL design patterns and best practices
- **Endpoint Design**: Resource modeling, URL structure, and HTTP method selection
- **Data Contracts**: Request/response schemas, validation rules, and data transformation
- **API Documentation**: OpenAPI/Swagger specifications and developer experience

### Role Boundaries & Responsibilities

#### **What This Role OWNS (API Contract Within Architectural Framework)**
- **API Contract Definition**: Specific endpoint paths, HTTP methods, and status codes
- **Resource Modeling**: Mapping domain entities to RESTful resources or GraphQL types
- **Request/Response Schemas**: Detailed data contracts, validation rules, and error formats
- **API Versioning Strategy**: Version management, deprecation policies, and migration paths
- **Developer Experience**: API documentation (OpenAPI/Swagger), code examples, and SDKs

#### **What This Role DOES NOT Own (Defers to Other Roles)**
- **System Architecture Decisions**: Microservices vs monolithic, overall communication patterns → Defers to **System Architect**
- **Canonical Data Model**: Underlying data schemas and entity relationships → Defers to **Data Architect**
- **UI/Frontend Integration**: How clients consume the API → Defers to **UI Designer**

#### **Handoff Points**
- **FROM System Architect**: Receives architectural constraints (REST vs GraphQL, sync vs async) that define the design space
- **FROM Data Architect**: Receives canonical data model and translates it into public API data contracts (as projection/view)
- **TO Frontend Teams**: Provides complete API specifications, documentation, and integration guides

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
CHECK: brainstorm_dir/api-designer/analysis.md
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
     prompt="Generate API designer analysis addressing topic framework

     ## Framework Integration Required
     **MANDATORY**: Load and address guidance-specification.md discussion points
     **Framework Reference**: @{session.brainstorm_dir}/guidance-specification.md
     **Output Location**: {session.brainstorm_dir}/api-designer/analysis.md

     ## Analysis Requirements
     1. **Load Topic Framework**: Read guidance-specification.md completely
     2. **Address Each Discussion Point**: Respond to all 5 framework sections from API design perspective
     3. **Include Framework Reference**: Start analysis.md with @../guidance-specification.md
     4. **API Design Focus**: Emphasize endpoint structure, data contracts, versioning strategies
     5. **Structured Response**: Use framework structure for analysis organization

     ## Framework Sections to Address
     - Core Requirements (from API design perspective)
     - Technical Considerations (detailed API architecture analysis)
     - User Experience Factors (developer experience and API usability)
     - Implementation Challenges (API design risks and solutions)
     - Success Metrics (API performance metrics and adoption tracking)

     ## Output Structure Required
     ```markdown
     # API Designer Analysis: [Topic]

     **Framework Reference**: @../guidance-specification.md
     **Role Focus**: Backend API Design and Contract Definition

     ## Core Requirements Analysis
     [Address framework requirements from API design perspective]

     ## Technical Considerations
     [Detailed API architecture and endpoint design analysis]

     ## Developer Experience Factors
     [API usability, documentation, and integration ease]

     ## Implementation Challenges
     [API design risks and mitigation strategies]

     ## Success Metrics
     [API performance metrics, adoption rates, and developer satisfaction]

     ## API Design-Specific Recommendations
     [Detailed API design recommendations and best practices]
     ```",
     description="Generate API designer framework-based analysis")
```

### Phase 4: Update Mechanism
**Analysis Update Process**:
```bash
# For existing analysis updates
IF update_mode = "incremental":
    Task(subagent_type="conceptual-planning-agent",
         prompt="Update existing API designer analysis

         ## Current Analysis Context
         **Existing Analysis**: @{session.brainstorm_dir}/api-designer/analysis.md
         **Framework Reference**: @{session.brainstorm_dir}/guidance-specification.md

         ## Update Requirements
         1. **Preserve Structure**: Maintain existing analysis structure
         2. **Add New Insights**: Integrate new API design insights and recommendations
         3. **Framework Alignment**: Ensure continued alignment with topic framework
         4. **API Updates**: Add new endpoint patterns, versioning strategies, documentation improvements
         5. **Maintain References**: Keep @../guidance-specification.md reference

         ## Update Instructions
         - Read existing analysis completely
         - Identify areas for enhancement or new insights
         - Add API design depth while preserving original structure
         - Update recommendations with new API design patterns and approaches
         - Maintain framework discussion point addressing",
         description="Update API designer analysis incrementally")
```

## Document Structure

### Output Files
```
.workflow/active/WFS-[topic]/.brainstorming/
├── guidance-specification.md          # Input: Framework (if exists)
└── api-designer/
    └── analysis.md            # ★ OUTPUT: Framework-based analysis
```

### Analysis Structure
**Required Elements**:
- **Framework Reference**: @../guidance-specification.md (if framework exists)
- **Role Focus**: Backend API Design and Contract Definition perspective
- **5 Framework Sections**: Address each framework discussion point
- **API Design Recommendations**: Endpoint-specific insights and solutions

## ⚡ **Two-Step Execution Flow**

### ⚠️ Session Management - FIRST STEP
Session detection and selection:
```bash
# Check for active sessions
active_sessions=$(find .workflow/active/ -name "WFS-*" -type d 2>/dev/null)
if [ multiple_sessions ]; then
  prompt_user_to_select_session()
else
  use_existing_or_create_new()
fi
```

### Step 1: Context Gathering Phase
**API Designer Perspective Questioning**

Before agent assignment, gather comprehensive API design context:

#### 📋 Role-Specific Questions
1. **API Type & Architecture**
   - RESTful, GraphQL, or hybrid API approach?
   - Synchronous vs asynchronous communication patterns?
   - Real-time requirements (WebSocket, Server-Sent Events)?

2. **Resource Modeling & Endpoints**
   - What are the core domain resources/entities?
   - Expected CRUD operations for each resource?
   - Complex query requirements (filtering, sorting, pagination)?

3. **Data Contracts & Validation**
   - Request/response data format requirements (JSON, XML, Protocol Buffers)?
   - Input validation and sanitization requirements?
   - Data transformation and mapping needs?

4. **API Management & Governance**
   - API versioning strategy requirements?
   - Authentication and authorization mechanisms?
   - Rate limiting and throttling requirements?
   - API documentation and developer portal needs?

5. **Integration & Compatibility**
   - Client platforms consuming the API (web, mobile, third-party)?
   - Backward compatibility requirements?
   - External API integrations needed?

#### Context Validation
- **Minimum Response**: Each answer must be ≥50 characters
- **Re-prompting**: Insufficient detail triggers follow-up questions
- **Context Storage**: Save responses to `.brainstorming/api-designer-context.md`

### Step 2: Agent Assignment with Flow Control
**Dedicated Agent Execution**

```bash
Task(conceptual-planning-agent): "
[FLOW_CONTROL]

Execute dedicated api-designer conceptual analysis for: {topic}

ASSIGNED_ROLE: api-designer
OUTPUT_LOCATION: .brainstorming/api-designer/
USER_CONTEXT: {validated_responses_from_context_gathering}

Flow Control Steps:
[
  {
    \"step\": \"load_role_template\",
    \"action\": \"Load api-designer planning template\",
    \"command\": \"bash($(cat ~/.claude/workflows/cli-templates/planning-roles/api-designer.md))\",
    \"output_to\": \"role_template\"
  }
]

Conceptual Analysis Requirements:
- Apply api-designer perspective to topic analysis
- Focus on endpoint design, data contracts, and API governance
- Use loaded role template framework for analysis structure
- Generate role-specific deliverables in designated output location
- Address all user context from questioning phase

Deliverables:
- analysis.md: Main API design analysis
- api-specification.md: Detailed endpoint specifications
- data-contracts.md: Request/response schemas and validation rules
- api-documentation.md: API documentation strategy and templates

Embody api-designer role expertise for comprehensive conceptual planning."
```

### Progress Tracking
TodoWrite tracking for two-step process:
```json
[
  {"content": "Gather API designer context through role-specific questioning", "status": "in_progress", "activeForm": "Gathering context"},
  {"content": "Validate context responses and save to api-designer-context.md", "status": "pending", "activeForm": "Validating context"},
  {"content": "Load api-designer planning template via flow control", "status": "pending", "activeForm": "Loading template"},
  {"content": "Execute dedicated conceptual-planning-agent for api-designer role", "status": "pending", "activeForm": "Executing agent"}
]
```

## 📊 **Output Specification**

### Output Location
```
.workflow/active/WFS-{topic-slug}/.brainstorming/api-designer/
├── analysis.md                 # Primary API design analysis
├── api-specification.md        # Detailed endpoint specifications (OpenAPI/Swagger)
├── data-contracts.md           # Request/response schemas and validation rules
├── versioning-strategy.md      # API versioning and backward compatibility plan
└── developer-guide.md          # API usage documentation and integration examples
```

### Document Templates

#### analysis.md Structure
```markdown
# API Design Analysis: {Topic}
*Generated: {timestamp}*

## Executive Summary
[Key API design findings and recommendations overview]

## API Architecture Overview
### API Type Selection (REST/GraphQL/Hybrid)
### Communication Patterns
### Authentication & Authorization Strategy

## Resource Modeling
### Core Domain Resources
### Resource Relationships
### URL Structure and Naming Conventions

## Endpoint Design
### Resource Endpoints
- GET /api/v1/resources
- POST /api/v1/resources
- GET /api/v1/resources/{id}
- PUT /api/v1/resources/{id}
- DELETE /api/v1/resources/{id}

### Query Parameters
- Filtering: ?filter[field]=value
- Sorting: ?sort=field,-field2
- Pagination: ?page=1&limit=20

### HTTP Methods and Status Codes
- Success responses (2xx)
- Client errors (4xx)
- Server errors (5xx)

## Data Contracts
### Request Schemas
[JSON Schema or OpenAPI definitions]

### Response Schemas
[JSON Schema or OpenAPI definitions]

### Validation Rules
- Required fields
- Data types and formats
- Business logic constraints

## API Versioning Strategy
### Versioning Approach (URL/Header/Accept)
### Version Lifecycle Management
### Deprecation Policy
### Migration Paths

## Security & Governance
### Authentication Mechanisms
- OAuth 2.0 / JWT / API Keys
### Authorization Patterns
- RBAC / ABAC / Resource-based
### Rate Limiting & Throttling
### CORS and Security Headers

## Error Handling
### Standard Error Response Format
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": [],
    "trace_id": "uuid"
  }
}
```

### Error Code Taxonomy
### Validation Error Responses

## API Documentation
### OpenAPI/Swagger Specification
### Developer Portal Requirements
### Code Examples and SDKs
### Changelog and Migration Guides

## Performance Optimization
### Response Caching Strategies
### Compression (gzip, brotli)
### Field Selection (sparse fieldsets)
### Bulk Operations and Batch Endpoints

## Monitoring & Observability
### API Metrics
- Request count, latency, error rates
- Endpoint usage analytics
### Logging Strategy
### Distributed Tracing

## Developer Experience
### API Usability Assessment
### Integration Complexity
### SDK and Client Library Needs
### Sandbox and Testing Environments
```

#### api-specification.md Structure
```markdown
# API Specification: {Topic}
*OpenAPI 3.0 Specification*

## API Information
- **Title**: {API Name}
- **Version**: 1.0.0
- **Base URL**: https://api.example.com/v1
- **Contact**: api-team@example.com

## Endpoints

### Users API

#### GET /users
**Description**: Retrieve a list of users

**Parameters**:
- `page` (query, integer): Page number (default: 1)
- `limit` (query, integer): Items per page (default: 20, max: 100)
- `sort` (query, string): Sort field (e.g., "created_at", "-updated_at")
- `filter[status]` (query, string): Filter by user status

**Response 200**:
```json
{
  "data": [
    {
      "id": "uuid",
      "username": "string",
      "email": "string",
      "created_at": "2025-10-15T00:00:00Z"
    }
  ],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 100
  },
  "links": {
    "self": "/users?page=1",
    "next": "/users?page=2",
    "prev": null
  }
}
```

#### POST /users
**Description**: Create a new user

**Request Body**:
```json
{
  "username": "string (required, 3-50 chars)",
  "email": "string (required, valid email)",
  "password": "string (required, min 8 chars)",
  "profile": {
    "first_name": "string (optional)",
    "last_name": "string (optional)"
  }
}
```

**Response 201**:
```json
{
  "data": {
    "id": "uuid",
    "username": "string",
    "email": "string",
    "created_at": "2025-10-15T00:00:00Z"
  }
}
```

**Response 400** (Validation Error):
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ]
  }
}
```

[Continue for all endpoints...]

## Authentication

### OAuth 2.0 Flow
1. Client requests authorization
2. User grants permission
3. Client receives access token
4. Client uses token in requests

**Header Format**:
```
Authorization: Bearer {access_token}
```

## Rate Limiting

**Headers**:
- `X-RateLimit-Limit`: 1000
- `X-RateLimit-Remaining`: 999
- `X-RateLimit-Reset`: 1634270400

**Response 429** (Too Many Requests):
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "API rate limit exceeded",
    "retry_after": 3600
  }
}
```
```

## 🔄 **Session Integration**

### Status Synchronization
Upon completion, update `workflow-session.json`:
```json
{
  "phases": {
    "BRAINSTORM": {
      "api_designer": {
        "status": "completed",
        "completed_at": "timestamp",
        "output_directory": ".workflow/active/WFS-{topic}/.brainstorming/api-designer/",
        "key_insights": ["endpoint_design", "versioning_strategy", "data_contracts"]
      }
    }
  }
}
```

### Cross-Role Collaboration
API designer perspective provides:
- **API Contract Specifications** → Frontend Developer
- **Data Schema Requirements** → Data Architect
- **Security Requirements** → Security Expert
- **Integration Endpoints** → System Architect
- **Performance Constraints** → DevOps Engineer

## ✅ **Quality Assurance**

### Required Analysis Elements
- [ ] Complete endpoint inventory with HTTP methods and paths
- [ ] Detailed request/response schemas with validation rules
- [ ] Clear versioning strategy and backward compatibility plan
- [ ] Comprehensive error handling and status code usage
- [ ] API documentation strategy (OpenAPI/Swagger)

### API Design Principles
- [ ] **Consistency**: Uniform naming conventions and patterns across all endpoints
- [ ] **Simplicity**: Intuitive resource modeling and URL structures
- [ ] **Flexibility**: Support for filtering, sorting, pagination, and field selection
- [ ] **Security**: Proper authentication, authorization, and input validation
- [ ] **Performance**: Caching strategies, compression, and efficient data structures

### Developer Experience Validation
- [ ] API is self-documenting with clear endpoint descriptions
- [ ] Error messages are actionable and helpful for debugging
- [ ] Response formats are consistent and predictable
- [ ] Code examples and integration guides are provided
- [ ] Sandbox environment available for testing

### Technical Completeness
- [ ] **Resource Modeling**: All domain entities mapped to API resources
- [ ] **CRUD Coverage**: Complete create, read, update, delete operations
- [ ] **Query Capabilities**: Advanced filtering, sorting, and search functionality
- [ ] **Versioning**: Clear version management and migration paths
- [ ] **Monitoring**: API metrics, logging, and tracing strategies defined

### Integration Readiness
- [ ] **Client Compatibility**: API works with all target client platforms
- [ ] **External Integration**: Third-party API dependencies identified
- [ ] **Backward Compatibility**: Changes don't break existing clients
- [ ] **Migration Path**: Clear upgrade paths for API consumers
- [ ] **SDK Support**: Client libraries and code generation considered
