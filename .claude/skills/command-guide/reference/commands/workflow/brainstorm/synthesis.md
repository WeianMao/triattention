---
name: synthesis
description: Clarify and refine role analyses through intelligent Q&A and targeted updates with synthesis agent
argument-hint: "[optional: --session session-id]"
allowed-tools: Task(conceptual-planning-agent), TodoWrite(*), Read(*), Write(*), Edit(*), Glob(*), AskUserQuestion(*)
---

## Overview

Six-phase workflow to eliminate ambiguities and enhance conceptual depth in role analyses:

**Phase 1-2**: Session detection → File discovery → Path preparation
**Phase 3A**: Cross-role analysis agent → Generate recommendations
**Phase 4**: User selects enhancements → User answers clarifications (via AskUserQuestion)
**Phase 5**: Parallel update agents (one per role)
**Phase 6**: Context package update → Metadata update → Completion report

All user interactions use AskUserQuestion tool (max 4 questions per call, multi-round).

**Document Flow**:
- Input: `[role]/analysis*.md`, `guidance-specification.md`, session metadata
- Output: Updated `[role]/analysis*.md` with Enhancements + Clarifications sections

---

## Quick Reference

### Phase Summary

| Phase | Goal | Executor | Output |
|-------|------|----------|--------|
| 1 | Session detection | Main flow | session_id, brainstorm_dir |
| 2 | File discovery | Main flow | role_analysis_paths |
| 3A | Cross-role analysis | Agent | enhancement_recommendations |
| 4 | User interaction | Main flow + AskUserQuestion | update_plan |
| 5 | Document updates | Parallel agents | Updated analysis*.md |
| 6 | Finalization | Main flow | context-package.json, report |

### AskUserQuestion Pattern

```javascript
// Enhancement selection (multi-select)
AskUserQuestion({
  questions: [{
    question: "请选择要应用的改进建议",
    header: "改进选择",
    multiSelect: true,
    options: [
      { label: "EP-001: API Contract", description: "添加详细的请求/响应 schema 定义" },
      { label: "EP-002: User Intent", description: "明确用户需求优先级和验收标准" }
    ]
  }]
})

// Clarification questions (single-select, multi-round)
AskUserQuestion({
  questions: [
    {
      question: "MVP 阶段的核心目标是什么？",
      header: "用户意图",
      multiSelect: false,
      options: [
        { label: "快速验证", description: "最小功能集，快速上线获取反馈" },
        { label: "技术壁垒", description: "完善架构，为长期发展打基础" },
        { label: "功能完整", description: "覆盖所有规划功能，延迟上线" }
      ]
    }
  ]
})
```

---

## Task Tracking

```json
[
  {"content": "Detect session and validate analyses", "status": "pending", "activeForm": "Detecting session"},
  {"content": "Discover role analysis file paths", "status": "pending", "activeForm": "Discovering paths"},
  {"content": "Execute analysis agent (cross-role analysis)", "status": "pending", "activeForm": "Executing analysis"},
  {"content": "Present enhancements via AskUserQuestion", "status": "pending", "activeForm": "Selecting enhancements"},
  {"content": "Clarification questions via AskUserQuestion", "status": "pending", "activeForm": "Clarifying"},
  {"content": "Execute parallel update agents", "status": "pending", "activeForm": "Updating documents"},
  {"content": "Update context package and metadata", "status": "pending", "activeForm": "Finalizing"}
]
```

---

## Execution Phases

### Phase 1: Discovery & Validation

1. **Detect Session**: Use `--session` parameter or find `.workflow/active/WFS-*`
2. **Validate Files**:
   - `guidance-specification.md` (optional, warn if missing)
   - `*/analysis*.md` (required, error if empty)
3. **Load User Intent**: Extract from `workflow-session.json`

### Phase 2: Role Discovery & Path Preparation

**Main flow prepares file paths for Agent**:

1. **Discover Analysis Files**:
   - Glob: `.workflow/active/WFS-{session}/.brainstorming/*/analysis*.md`
   - Supports: analysis.md + analysis-{slug}.md (max 5)

2. **Extract Role Information**:
   - `role_analysis_paths`: Relative paths
   - `participating_roles`: Role names from directories

3. **Pass to Agent**: session_id, brainstorm_dir, role_analysis_paths, participating_roles

### Phase 3A: Analysis & Enhancement Agent

**Agent executes cross-role analysis**:

```javascript
Task(conceptual-planning-agent, `
## Agent Mission
Analyze role documents, identify conflicts/gaps, generate enhancement recommendations

## Input
- brainstorm_dir: ${brainstorm_dir}
- role_analysis_paths: ${role_analysis_paths}
- participating_roles: ${participating_roles}

## Flow Control Steps
1. load_session_metadata → Read workflow-session.json
2. load_role_analyses → Read all analysis files
3. cross_role_analysis → Identify consensus, conflicts, gaps, ambiguities
4. generate_recommendations → Format as EP-001, EP-002, ...

## Output Format
[
  {
    "id": "EP-001",
    "title": "API Contract Specification",
    "affected_roles": ["system-architect", "api-designer"],
    "category": "Architecture",
    "current_state": "High-level API descriptions",
    "enhancement": "Add detailed contract definitions",
    "rationale": "Enables precise implementation",
    "priority": "High"
  }
]
`)
```

### Phase 4: User Interaction

**All interactions via AskUserQuestion (Chinese questions)**

#### Step 1: Enhancement Selection

```javascript
// If enhancements > 4, split into multiple rounds
const enhancements = [...]; // from Phase 3A
const BATCH_SIZE = 4;

for (let i = 0; i < enhancements.length; i += BATCH_SIZE) {
  const batch = enhancements.slice(i, i + BATCH_SIZE);

  AskUserQuestion({
    questions: [{
      question: `请选择要应用的改进建议 (第${Math.floor(i/BATCH_SIZE)+1}轮)`,
      header: "改进选择",
      multiSelect: true,
      options: batch.map(ep => ({
        label: `${ep.id}: ${ep.title}`,
        description: `影响: ${ep.affected_roles.join(', ')} | ${ep.enhancement}`
      }))
    }]
  })

  // Store selections before next round
}

// User can also skip: provide "跳过" option
```

#### Step 2: Clarification Questions

```javascript
// Generate questions based on 9-category taxonomy scan
// Categories: User Intent, Requirements, Architecture, UX, Feasibility, Risk, Process, Decisions, Terminology

const clarifications = [...]; // from analysis
const BATCH_SIZE = 4;

for (let i = 0; i < clarifications.length; i += BATCH_SIZE) {
  const batch = clarifications.slice(i, i + BATCH_SIZE);
  const currentRound = Math.floor(i / BATCH_SIZE) + 1;
  const totalRounds = Math.ceil(clarifications.length / BATCH_SIZE);

  AskUserQuestion({
    questions: batch.map(q => ({
      question: q.question,
      header: q.category.substring(0, 12),
      multiSelect: false,
      options: q.options.map(opt => ({
        label: opt.label,
        description: opt.description
      }))
    }))
  })

  // Store answers before next round
}
```

### Question Guidelines

**Target**: 开发者（理解技术但需要从用户需求出发）

**Question Structure**: `[跨角色分析发现] + [需要澄清的决策点]`
**Option Structure**: `标签：[具体方案] + 说明：[业务影响] + [技术权衡]`

**9-Category Taxonomy**:

| Category | Focus | Example Question Pattern |
|----------|-------|--------------------------|
| User Intent | 用户目标 | "MVP阶段核心目标？" + 验证/壁垒/完整性 |
| Requirements | 需求细化 | "功能优先级如何排序？" + 核心/增强/可选 |
| Architecture | 架构决策 | "技术栈选择考量？" + 熟悉度/先进性/成熟度 |
| UX | 用户体验 | "交互复杂度取舍？" + 简洁/丰富/渐进 |
| Feasibility | 可行性 | "资源约束下的范围？" + 最小/标准/完整 |
| Risk | 风险管理 | "风险容忍度？" + 保守/平衡/激进 |
| Process | 流程规范 | "迭代节奏？" + 快速/稳定/灵活 |
| Decisions | 决策确认 | "冲突解决方案？" + 方案A/方案B/折中 |
| Terminology | 术语统一 | "统一使用哪个术语？" + 术语A/术语B |

**Quality Rules**:

**MUST Include**:
- ✅ All questions in Chinese (用中文提问)
- ✅ 基于跨角色分析的具体发现
- ✅ 选项包含业务影响说明
- ✅ 解决实际的模糊点或冲突

**MUST Avoid**:
- ❌ 与角色分析无关的通用问题
- ❌ 重复已在 artifacts 阶段确认的内容
- ❌ 过于细节的实现级问题

#### Step 3: Build Update Plan

```javascript
update_plan = {
  "role1": {
    "enhancements": ["EP-001", "EP-003"],
    "clarifications": [
      {"question": "...", "answer": "...", "category": "..."}
    ]
  },
  "role2": {
    "enhancements": ["EP-002"],
    "clarifications": [...]
  }
}
```

### Phase 5: Parallel Document Update Agents

**Execute in parallel** (one agent per role):

```javascript
// Single message with multiple Task calls for parallelism
Task(conceptual-planning-agent, `
## Agent Mission
Apply enhancements and clarifications to ${role} analysis

## Input
- role: ${role}
- analysis_path: ${brainstorm_dir}/${role}/analysis.md
- enhancements: ${role_enhancements}
- clarifications: ${role_clarifications}
- original_user_intent: ${intent}

## Flow Control Steps
1. load_current_analysis → Read analysis file
2. add_clarifications_section → Insert Q&A section
3. apply_enhancements → Integrate into relevant sections
4. resolve_contradictions → Remove conflicts
5. enforce_terminology → Align terminology
6. validate_intent → Verify alignment with user intent
7. write_updated_file → Save changes

## Output
Updated ${role}/analysis.md
`)
```

**Agent Characteristics**:
- **Isolation**: Each agent updates exactly ONE role (parallel safe)
- **Dependencies**: Zero cross-agent dependencies
- **Validation**: All updates must align with original_user_intent

### Phase 6: Finalization

#### Step 1: Update Context Package

```javascript
// Sync updated analyses to context-package.json
const context_pkg = Read(".workflow/active/WFS-{session}/.process/context-package.json")

// Update guidance-specification if exists
// Update synthesis-specification if exists
// Re-read all role analysis files
// Update metadata timestamps

Write(context_pkg_path, JSON.stringify(context_pkg))
```

#### Step 2: Update Session Metadata

```json
{
  "phases": {
    "BRAINSTORM": {
      "status": "clarification_completed",
      "clarification_completed": true,
      "completed_at": "timestamp",
      "participating_roles": [...],
      "clarification_results": {
        "enhancements_applied": ["EP-001", "EP-002"],
        "questions_asked": 3,
        "categories_clarified": ["Architecture", "UX"],
        "roles_updated": ["role1", "role2"]
      },
      "quality_metrics": {
        "user_intent_alignment": "validated",
        "ambiguity_resolution": "complete",
        "terminology_consistency": "enforced"
      }
    }
  }
}
```

#### Step 3: Completion Report

```markdown
## ✅ Clarification Complete

**Enhancements Applied**: EP-001, EP-002, EP-003
**Questions Answered**: 3/5
**Roles Updated**: role1, role2, role3

### Next Steps
✅ PROCEED: `/workflow:plan --session WFS-{session-id}`
```

---

## Output

**Location**: `.workflow/active/WFS-{session}/.brainstorming/[role]/analysis*.md`

**Updated Structure**:
```markdown
## Clarifications
### Session {date}
- **Q**: {question} (Category: {category})
  **A**: {answer}

## {Existing Sections}
{Refined content based on clarifications}
```

**Changes**:
- User intent validated/corrected
- Requirements more specific/measurable
- Architecture with rationale
- Ambiguities resolved, placeholders removed
- Consistent terminology

---

## Quality Checklist

**Content**:
- ✅ All role analyses loaded/analyzed
- ✅ Cross-role analysis (consensus, conflicts, gaps)
- ✅ 9-category ambiguity scan
- ✅ Questions prioritized

**Analysis**:
- ✅ User intent validated
- ✅ Cross-role synthesis complete
- ✅ Ambiguities resolved
- ✅ Terminology consistent

**Documents**:
- ✅ Clarifications section formatted
- ✅ Sections reflect answers
- ✅ No placeholders (TODO/TBD)
- ✅ Valid Markdown
