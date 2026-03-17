---
name: review
description: Post-implementation review with specialized types (security/architecture/action-items/quality) using analysis agents and Gemini
argument-hint: "[--type=security|architecture|action-items|quality] [optional: session-id]"
---

## Command Overview: /workflow:review

**Optional specialized review** for completed implementations. In the standard workflow, **passing tests = approved code**. Use this command only when specialized review is required (security, architecture, compliance, docs).

## Philosophy: "Tests Are the Review"

- **Default**: All tests pass -> Code approved
- **Optional**: Specialized reviews for:
  - Security audits (vulnerabilities, auth/authz)
  - Architecture compliance (patterns, technical debt)
  - Action items verification (requirements met, acceptance criteria)

## Review Types

| Type | Focus | Use Case |
|------|-------|----------|
| `quality` | Code quality, best practices, maintainability | Default general review |
| `security` | Security vulnerabilities, data handling, access control | Security audits |
| `architecture` | Architectural patterns, technical debt, design decisions | Architecture compliance |
| `action-items` | Requirements met, acceptance criteria verified, action items completed | Pre-deployment verification |

**Notes**:
- For documentation generation, use `/workflow:tools:docs`
- For CLAUDE.md updates, use `/update-memory-related`

## Execution Process

```
Input Parsing:
   ├─ Parse --type flag (default: quality)
   └─ Parse session-id argument (optional)

Step 1: Session Resolution
   └─ Decision:
      ├─ session-id provided → Use provided session
      └─ Not provided → Auto-detect from .workflow/active/

Step 2: Validation
   ├─ Check session directory exists
   └─ Check for completed implementation (.summaries/IMPL-*.md exists)

Step 3: Type Check
   └─ Decision:
      ├─ type=docs → Redirect to /workflow:tools:docs
      └─ Other types → Continue to analysis

Step 4: Model Analysis Phase
   ├─ Load context (summaries, test results, changed files)
   └─ Perform specialized review by type:
      ├─ security → Security patterns + Gemini analysis
      ├─ architecture → Qwen architecture analysis
      ├─ quality → Gemini code quality analysis
      └─ action-items → Requirements verification

Step 5: Generate Report
   └─ Output: REVIEW-{type}.md
```

## Execution Template

```bash
#!/bin/bash
# Optional specialized review for completed implementation

# Step 1: Session ID resolution
if [ -n "$SESSION_ARG" ]; then
    sessionId="$SESSION_ARG"
else
    sessionId=$(find .workflow/active/ -name "WFS-*" -type d | head -1 | xargs basename)
fi

# Step 2: Validation
if [ ! -d ".workflow/active/${sessionId}" ]; then
    echo "Session ${sessionId} not found"
    exit 1
fi

# Check for completed tasks
if [ ! -d ".workflow/active/${sessionId}/.summaries" ] || [ -z "$(find .workflow/active/${sessionId}/.summaries/ -name "IMPL-*.md" -type f 2>/dev/null)" ]; then
    echo "No completed implementation found. Complete implementation first"
    exit 1
fi

# Step 3: Determine review type (default: quality)
review_type="${TYPE_ARG:-quality}"

# Redirect docs review to specialized command
if [ "$review_type" = "docs" ]; then
    echo "For documentation generation, please use:"
    echo "   /workflow:tools:docs"
    echo ""
    echo "The docs command provides:"
    echo "  - Hierarchical architecture documentation"
    echo "  - API documentation generation"
    echo "  - Documentation structure analysis"
    exit 0
fi

# Step 4: Analysis handover → Model takes control
# BASH_EXECUTION_STOPS → MODEL_ANALYSIS_BEGINS
```

### Model Analysis Phase

After bash validation, the model takes control to:

1. **Load Context**: Read completed task summaries and changed files
   ```bash
   # Load implementation summaries
   cat .workflow/active/${sessionId}/.summaries/IMPL-*.md

   # Load test results (if available)
   cat .workflow/active/${sessionId}/.summaries/TEST-FIX-*.md 2>/dev/null

   # Get changed files
   git log --since="$(cat .workflow/active/${sessionId}/workflow-session.json | jq -r .created_at)" --name-only --pretty=format: | sort -u
   ```

2. **Perform Specialized Review**: Based on `review_type`

   **Security Review** (`--type=security`):
   - Use ripgrep for security patterns:
     ```bash
     rg "password|token|secret|auth" -g "*.{ts,js,py}"
     rg "eval|exec|innerHTML|dangerouslySetInnerHTML" -g "*.{ts,js,tsx}"
     ```
   - Use Gemini for security analysis:
     ```bash
     cd .workflow/active/${sessionId} && gemini -p "
     PURPOSE: Security audit of completed implementation
     TASK: Review code for security vulnerabilities, insecure patterns, auth/authz issues
     CONTEXT: @.summaries/IMPL-*.md,../.. @../../CLAUDE.md
     EXPECTED: Security findings report with severity levels
     RULES: Focus on OWASP Top 10, authentication, authorization, data validation, injection risks
     " --approval-mode yolo
     ```

   **Architecture Review** (`--type=architecture`):
   - Use Qwen for architecture analysis:
     ```bash
     cd .workflow/active/${sessionId} && qwen -p "
     PURPOSE: Architecture compliance review
     TASK: Evaluate adherence to architectural patterns, identify technical debt, review design decisions
     CONTEXT: @.summaries/IMPL-*.md,../.. @../../CLAUDE.md
     EXPECTED: Architecture assessment with recommendations
     RULES: Check for patterns, separation of concerns, modularity, scalability
     " --approval-mode yolo
     ```

   **Quality Review** (`--type=quality`):
   - Use Gemini for code quality:
     ```bash
     cd .workflow/active/${sessionId} && gemini -p "
     PURPOSE: Code quality and best practices review
     TASK: Assess code readability, maintainability, adherence to best practices
     CONTEXT: @.summaries/IMPL-*.md,../.. @../../CLAUDE.md
     EXPECTED: Quality assessment with improvement suggestions
     RULES: Check for code smells, duplication, complexity, naming conventions
     " --approval-mode yolo
     ```

   **Action Items Review** (`--type=action-items`):
   - Verify all requirements and acceptance criteria met:
     ```bash
     # Load task requirements and acceptance criteria
     find .workflow/active/${sessionId}/.task -name "IMPL-*.json" -exec jq -r '
       "Task: " + .id + "\n" +
       "Requirements: " + (.context.requirements | join(", ")) + "\n" +
       "Acceptance: " + (.context.acceptance | join(", "))
     ' {} \;

     # Check implementation summaries against requirements
     cd .workflow/active/${sessionId} && gemini -p "
     PURPOSE: Verify all requirements and acceptance criteria are met
     TASK: Cross-check implementation summaries against original requirements
     CONTEXT: @.task/IMPL-*.json,.summaries/IMPL-*.md,../.. @../../CLAUDE.md
     EXPECTED:
     - Requirements coverage matrix
     - Acceptance criteria verification
     - Missing/incomplete action items
     - Pre-deployment readiness assessment
     RULES:
     - Check each requirement has corresponding implementation
     - Verify all acceptance criteria are met
     - Flag any incomplete or missing action items
     - Assess deployment readiness
     " --approval-mode yolo
     ```


3. **Generate Review Report**: Create structured report
   ```markdown
   # Review Report: ${review_type}

   **Session**: ${sessionId}
   **Date**: $(date)
   **Type**: ${review_type}

   ## Summary
   - Tasks Reviewed: [count IMPL tasks]
   - Files Changed: [count files]
   - Severity: [High/Medium/Low]

   ## Findings

   ### Critical Issues
   - [Issue 1 with file:line reference]
   - [Issue 2 with file:line reference]

   ### Recommendations
   - [Recommendation 1]
   - [Recommendation 2]

   ### Positive Observations
   - [Good pattern observed]

   ## Action Items
   - [ ] [Action 1]
   - [ ] [Action 2]
   ```

4. **Output Files**:
   ```bash
   # Save review report
   Write(.workflow/active/${sessionId}/REVIEW-${review_type}.md)

   # Update session metadata
   # (optional) Update workflow-session.json with review status
   ```

5. **Optional: Update Memory** (if docs review or significant findings):
   ```bash
   # If architecture or quality issues found, suggest memory update
   if [ "$review_type" = "architecture" ] || [ "$review_type" = "quality" ]; then
       echo "Consider updating project documentation:"
       echo "   /update-memory-related"
   fi
   ```

## Usage Examples

```bash
# General quality review after implementation
/workflow:review

# Security audit before deployment
/workflow:review --type=security

# Architecture review for specific session
/workflow:review --type=architecture WFS-payment-integration

# Documentation review
/workflow:review --type=docs
```

## Features

- **Simple Validation**: Check session exists and has completed tasks
- **No Complex Orchestration**: Direct analysis, no multi-phase pipeline
- **Specialized Reviews**: Different prompts and tools for different review types
- **MCP Integration**: Fast code search for security and architecture patterns
- **CLI Tool Integration**: Gemini for analysis, Qwen for architecture
- **Structured Output**: Markdown reports with severity levels and action items
- **Optional Memory Update**: Suggests documentation updates for significant findings

## Integration with Workflow

```
Standard Workflow:
  plan -> execute -> test-gen -> execute (complete)

Optional Review (when needed):
  plan -> execute -> test-gen -> execute -> review (security/architecture/docs)
```

**When to Use**:
- Before production deployment (security review + action-items review)
- After major feature (architecture review)
- Before code freeze (quality review)
- Pre-deployment verification (action-items review)

**When NOT to Use**:
- Regular development (tests are sufficient)
- Simple bug fixes (test-fix-agent handles it)
- Minor changes (update-memory-related is enough)
