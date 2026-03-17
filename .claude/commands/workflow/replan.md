---
name: replan
description: Interactive workflow replanning with session-level artifact updates and boundary clarification through guided questioning
argument-hint: "[--session session-id] [task-id] \"requirements\"|file.md [--interactive]"
allowed-tools: Read(*), Write(*), Edit(*), TodoWrite(*), Glob(*), Bash(*)
---

# Workflow Replan Command

## Overview
Intelligently replans workflow sessions or individual tasks with interactive boundary clarification and comprehensive artifact updates.

**Core Capabilities**:
- **Session Replan**: Updates multiple artifacts (IMPL_PLAN.md, TODO_LIST.md, task JSONs)
- **Task Replan**: Focused updates within session context
- **Interactive Clarification**: Guided questioning to define modification boundaries
- **Impact Analysis**: Automatic detection of affected files and dependencies
- **Backup Management**: Preserves previous versions with restore capability

## Operation Modes

### Session Replan Mode

```bash
# Auto-detect active session
/workflow:replan "添加双因素认证支持"

# Explicit session
/workflow:replan --session WFS-oauth "添加双因素认证支持"

# File-based input
/workflow:replan --session WFS-oauth requirements-update.md

# Interactive mode
/workflow:replan --interactive
```

### Task Replan Mode

```bash
# Direct task update
/workflow:replan IMPL-1 "修改为使用 OAuth2.0 标准"

# Task with explicit session
/workflow:replan --session WFS-oauth IMPL-2 "增加单元测试覆盖率到 90%"

# Interactive mode
/workflow:replan IMPL-1 --interactive
```

## Execution Process

```
Input Parsing:
   ├─ Parse flags: --session, --interactive
   └─ Detect mode: task-id present → Task mode | Otherwise → Session mode

Phase 1: Mode Detection & Session Discovery
   ├─ Detect operation mode (Task vs Session)
   ├─ Discover/validate session (--session flag or auto-detect)
   └─ Load session context (workflow-session.json, IMPL_PLAN.md, TODO_LIST.md)

Phase 2: Interactive Requirement Clarification
   └─ Decision (by mode):
      ├─ Session mode → 3-4 questions (scope, modules, changes, dependencies)
      └─ Task mode → 2 questions (update type, ripple effect)

Phase 3: Impact Analysis & Planning
   ├─ Analyze required changes
   ├─ Generate modification plan
   └─ User confirmation (Execute / Adjust / Cancel)

Phase 4: Backup Creation
   └─ Backup all affected files with manifest

Phase 5: Apply Modifications
   ├─ Update IMPL_PLAN.md (if needed)
   ├─ Update TODO_LIST.md (if needed)
   ├─ Update/Create/Delete task JSONs
   └─ Update session metadata

Phase 6: Verification & Summary
   ├─ Validate consistency (JSON validity, task limits, acyclic dependencies)
   └─ Generate change summary
```

## Execution Lifecycle

### Input Parsing

**Parse flags**:
```javascript
const sessionFlag = $ARGUMENTS.match(/--session\s+(\S+)/)?.[1]
const interactive = $ARGUMENTS.includes('--interactive')
const taskIdMatch = $ARGUMENTS.match(/\b(IMPL-\d+(?:\.\d+)?)\b/)
const taskId = taskIdMatch?.[1]
```

### Phase 1: Mode Detection & Session Discovery

**Process**:
1. **Detect Operation Mode**:
   - Check if task ID provided (IMPL-N or IMPL-N.M format) → Task mode
   - Otherwise → Session mode

2. **Discover/Validate Session**:
   - Use `--session` flag if provided
   - Otherwise auto-detect from `.workflow/active/`
   - Validate session exists

3. **Load Session Context**:
   - Read `workflow-session.json`
   - List existing tasks
   - Read `IMPL_PLAN.md` and `TODO_LIST.md`

**Output**: Session validated, context loaded, mode determined

---

### Phase 2: Interactive Requirement Clarification

**Purpose**: Define modification scope through guided questioning

#### Session Mode Questions

**Q1: Modification Scope**
```javascript
Options:
- 仅更新任务细节 (tasks_only)
- 修改规划方案 (plan_update)
- 重构任务结构 (task_restructure)
- 全面重规划 (comprehensive)
```

**Q2: Affected Modules** (if scope >= plan_update)
```javascript
Options: Dynamically generated from existing tasks' focus_paths
- 认证模块 (src/auth)
- 用户管理 (src/user)
- 全部模块
```

**Q3: Task Changes** (if scope >= task_restructure)
```javascript
Options:
- 添加/删除任务 (add_remove)
- 合并/拆分任务 (merge_split)
- 仅更新内容 (update_only)
// Note: Max 4 options for AskUserQuestion
```

**Q4: Dependency Changes**
```javascript
Options:
- 是,需要重新梳理依赖
- 否,保持现有依赖
```

#### Task Mode Questions

**Q1: Update Type**
```javascript
Options:
- 需求和验收标准 (requirements & acceptance)
- 实现方案 (implementation_approach)
- 文件范围 (focus_paths)
- 依赖关系 (depends_on)
- 全部更新
```

**Q2: Ripple Effect**
```javascript
Options:
- 是,需要同步更新依赖任务
- 否,仅影响当前任务
- 不确定,请帮我分析
```

**Output**: User selections stored, modification boundaries defined

---

### Phase 3: Impact Analysis & Planning

**Step 3.1: Analyze Required Changes**

Determine affected files based on clarification:

```typescript
interface ImpactAnalysis {
  affected_files: {
    impl_plan: boolean;
    todo_list: boolean;
    session_meta: boolean;
    tasks: string[];
  };

  operations: {
    type: 'create' | 'update' | 'delete' | 'merge' | 'split';
    target: string;
    reason: string;
  }[];

  backup_strategy: {
    timestamp: string;
    files: string[];
  };
}
```

**Step 3.2: Generate Modification Plan**

```markdown
## 修改计划

### 影响范围
- [ ] IMPL_PLAN.md: 更新技术方案第 3 节
- [ ] TODO_LIST.md: 添加 2 个新任务,删除 1 个废弃任务
- [ ] IMPL-001.json: 更新实现方案
- [ ] workflow-session.json: 更新任务计数

### 变更操作
1. **创建**: IMPL-004.json (双因素认证实现)
2. **更新**: IMPL-001.json (添加 2FA 准备工作)
3. **删除**: IMPL-003.json (已被新方案替代)
```

**Step 3.3: User Confirmation**

```javascript
Options:
- 确认执行: 开始应用所有修改
- 调整计划: 重新回答问题调整范围
- 取消操作: 放弃本次重规划
```

**Output**: Modification plan confirmed or adjusted

---

### Phase 4: Backup Creation

**Process**:

1. **Create Backup Directory**:
```bash
timestamp=$(date -u +"%Y-%m-%dT%H-%M-%S")
backup_dir=".workflow/active/$SESSION_ID/.process/backup/replan-$timestamp"
mkdir -p "$backup_dir"
```

2. **Backup All Affected Files**:
   - IMPL_PLAN.md
   - TODO_LIST.md
   - workflow-session.json
   - Affected task JSONs

3. **Create Backup Manifest**:
```markdown
# Replan Backup Manifest

**Timestamp**: {timestamp}
**Reason**: {replan_reason}
**Scope**: {modification_scope}

## Restoration Command
cp {backup_dir}/* .workflow/active/{session}/
```

**Output**: All files safely backed up with manifest

---

### Phase 5: Apply Modifications

**Step 5.1: Update IMPL_PLAN.md** (if needed)

Use Edit tool to modify specific sections:
- Update affected technical sections
- Update modification date

**Step 5.2: Update TODO_LIST.md** (if needed)

- Add new tasks with `[ ]` checkbox
- Mark deleted tasks as `[x] ~~task~~ (已废弃)`
- Update modified task descriptions

**Step 5.3: Update Task JSONs**

For each affected task:
```typescript
const updated_task = {
  ...task,
  context: {
    ...task.context,
    requirements: [...updated_requirements],
    acceptance: [...updated_acceptance]
  },
  flow_control: {
    ...task.flow_control,
    implementation_approach: [...updated_steps]
  }
};

Write({
  file_path: `.workflow/active/${SESSION_ID}/.task/${task_id}.json`,
  content: JSON.stringify(updated_task, null, 2)
});
```

**Step 5.4: Create New Tasks** (if needed)

Generate complete task JSON with all required fields:
- id, title, status
- meta (type, agent)
- context (requirements, focus_paths, acceptance)
- flow_control (pre_analysis, implementation_approach, target_files)

**Step 5.5: Delete Obsolete Tasks** (if needed)

Move to backup instead of hard delete:
```bash
mv ".workflow/active/$SESSION_ID/.task/{task-id}.json" "$backup_dir/"
```

**Step 5.6: Update Session Metadata**

Update workflow-session.json:
- progress.current_tasks
- progress.last_replan
- replan_history array

**Output**: All modifications applied, artifacts updated

---

### Phase 6: Verification & Summary

**Step 6.1: Verify Consistency**

1. Validate all task JSONs are valid JSON
2. Check task count within limits (max 10)
3. Verify dependency graph is acyclic

**Step 6.2: Generate Change Summary**

```markdown
## 重规划完成

### 会话信息
- **Session**: {session-id}
- **时间**: {timestamp}
- **备份**: {backup-path}

### 变更摘要
**范围**: {scope}
**原因**: {reason}

### 修改的文件
- ✓ IMPL_PLAN.md: {changes}
- ✓ TODO_LIST.md: {changes}
- ✓ Task JSONs: {count} files updated

### 任务变更
- **新增**: {task-ids}
- **删除**: {task-ids}
- **更新**: {task-ids}

### 回滚方法
cp {backup-path}/* .workflow/active/{session}/
```

**Output**: Summary displayed, replan complete

---

## TodoWrite Progress Tracking

### Session Mode Progress

```json
[
  {"content": "检测模式和发现会话", "status": "completed", "activeForm": "检测模式和发现会话"},
  {"content": "交互式需求明确", "status": "completed", "activeForm": "交互式需求明确"},
  {"content": "影响分析和计划生成", "status": "completed", "activeForm": "影响分析和计划生成"},
  {"content": "创建备份", "status": "completed", "activeForm": "创建备份"},
  {"content": "更新会话产出文件", "status": "completed", "activeForm": "更新会话产出文件"},
  {"content": "验证一致性", "status": "completed", "activeForm": "验证一致性"}
]
```

### Task Mode Progress

```json
[
  {"content": "检测会话和加载任务", "status": "completed", "activeForm": "检测会话和加载任务"},
  {"content": "交互式更新确认", "status": "completed", "activeForm": "交互式更新确认"},
  {"content": "应用任务修改", "status": "completed", "activeForm": "应用任务修改"}
]
```

## Error Handling

### Session Errors

```bash
# No active session found
ERROR: No active session found
Run /workflow:session:start to create a session

# Session not found
ERROR: Session WFS-invalid not found
Available sessions: [list]

# No changes specified
WARNING: No modifications specified
Use --interactive mode or provide requirements
```

### Task Errors

```bash
# Task not found
ERROR: Task IMPL-999 not found in session
Available tasks: [list]

# Task completed
WARNING: Task IMPL-001 is completed
Consider creating new task for additional work

# Circular dependency
ERROR: Circular dependency detected
Resolve dependency conflicts before proceeding
```

### Validation Errors

```bash
# Task limit exceeded
ERROR: Replan would create 12 tasks (limit: 10)
Consider: combining tasks, splitting sessions, or removing tasks

# Invalid JSON
ERROR: Generated invalid JSON
Backup preserved, rolling back changes
```

## File Structure

```
.workflow/active/WFS-session-name/
├── workflow-session.json
├── IMPL_PLAN.md
├── TODO_LIST.md
├── .task/
│   ├── IMPL-001.json
│   ├── IMPL-002.json
│   └── IMPL-003.json
└── .process/
    ├── context-package.json
    └── backup/
        └── replan-{timestamp}/
            ├── MANIFEST.md
            ├── IMPL_PLAN.md
            ├── TODO_LIST.md
            ├── workflow-session.json
            └── IMPL-*.json
```

## Examples

### Session Replan - Add Feature

```bash
/workflow:replan "添加双因素认证支持"

# Interactive clarification
Q: 修改范围?
A: 全面重规划

Q: 受影响模块?
A: 认证模块, API接口

Q: 任务变更?
A: 添加新任务, 更新内容

# Execution
✓ 创建备份
✓ 更新 IMPL_PLAN.md
✓ 更新 TODO_LIST.md
✓ 创建 IMPL-004.json
✓ 更新 IMPL-001.json, IMPL-002.json

重规划完成! 新增 1 任务,更新 2 任务
```

### Task Replan - Update Requirements

```bash
/workflow:replan IMPL-001 "支持 OAuth2.0 标准"

# Interactive clarification
Q: 更新部分?
A: 需求和验收标准, 实现方案

Q: 影响其他任务?
A: 是,需要同步更新依赖任务

# Execution
✓ 创建备份
✓ 更新 IMPL-001.json
✓ 更新 IMPL-002.json (依赖任务)

任务重规划完成! 更新 2 个任务
```
