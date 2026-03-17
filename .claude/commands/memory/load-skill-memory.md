---
name: load-skill-memory
description: Activate SKILL package (auto-detect from paths/keywords or manual) and intelligently load documentation based on task intent keywords
argument-hint: "[skill_name] \"task intent description\""
allowed-tools: Bash(*), Read(*), Skill(*)
---

# Memory Load SKILL Command (/memory:load-skill-memory)

## 1. Overview

The `memory:load-skill-memory` command **activates SKILL package** (auto-detect from task or manual specification) and intelligently loads documentation based on user's task intent. The system automatically determines which documentation files to read based on the intent description.

**Core Philosophy**:
- **Flexible Activation**: Auto-detect skill from task description/paths, or user explicitly specifies
- **Intent-Driven Loading**: System analyzes task intent to determine documentation scope
- **Intelligent Selection**: Automatically chooses appropriate documentation level and modules
- **Direct Context Loading**: Loads selected documentation into conversation memory

**When to Use**:
- Manually activate a known SKILL package for a specific task
- Load SKILL context when system hasn't auto-triggered it
- Force reload SKILL documentation with specific intent focus

**Note**: Normal SKILL activation happens automatically via description triggers or path mentions (system extracts skill name from file paths for intelligent triggering). Use this command only when manual activation is needed.

## 2. Parameters

- `[skill_name]` (Optional): Name of SKILL package to activate
  - If omitted: System auto-detects from task description or file paths
  - If specified: Direct activation of named SKILL package
  - Example: `my_project`, `api_service`
  - Must match directory name under `.claude/skills/`

- `"task intent description"` (Required): Description of what you want to do
  - Used for both: auto-detection (if skill_name omitted) and documentation scope selection
  - **Analysis tasks**: "分析builder pattern实现", "理解参数系统架构"
  - **Modification tasks**: "修改workflow逻辑", "增强thermal template功能"
  - **Learning tasks**: "学习接口设计模式", "了解测试框架使用"
  - **With paths**: "修改D:\projects\my_project\src\auth.py的认证逻辑" (auto-extracts `my_project`)

## 3. Execution Flow

### Step 1: Determine SKILL Name (if not provided)

**Auto-Detection Strategy** (when skill_name parameter is omitted):
1. **Path Extraction**: Scan task description for file paths
   - Extract potential project names from path segments
   - Example: `"修改D:\projects\my_project\src\auth.py"` → extracts `my_project`
2. **Keyword Matching**: Match task keywords against SKILL descriptions
   - Search for project-specific terms, domain keywords
3. **Validation**: Check if extracted name matches `.claude/skills/{skill_name}/`

**Result**: Either uses provided skill_name or auto-detected name for activation

### Step 2: Activate SKILL and Analyze Intent

**Activate SKILL Package**:
```javascript
Skill(command: "${skill_name}")  // Uses provided or auto-detected name
```

**What Happens After Activation**:
1. If SKILL exists in memory: System reads `.claude/skills/${skill_name}/SKILL.md`
2. If SKILL not found in memory: Error - SKILL package doesn't exist
3. SKILL description triggers are loaded into memory
4. Progressive loading mechanism becomes available
5. Documentation structure is now accessible

**Intent Analysis**:
Based on task intent description, system determines:
- **Action type**: analyzing, modifying, learning
- **Scope**: specific module, architecture overview, complete system
- **Depth**: quick reference, detailed API, full documentation

### Step 3: Intelligent Documentation Loading

**Loading Strategy**:

The system automatically selects documentation based on intent keywords:

1. **Quick Understanding** ("了解", "快速理解", "什么是"):
   - Load: Level 0 (README.md only, ~2K tokens)
   - Use case: Quick overview of capabilities

2. **Specific Module Analysis** ("分析XXX模块", "理解XXX实现"):
   - Load: Module-specific README.md + API.md (~5K tokens)
   - Use case: Deep dive into specific component

3. **Architecture Review** ("架构", "设计模式", "整体结构"):
   - Load: README.md + ARCHITECTURE.md (~10K tokens)
   - Use case: System design understanding

4. **Implementation/Modification** ("修改", "增强", "实现"):
   - Load: Relevant module docs + EXAMPLES.md (~15K tokens)
   - Use case: Code modification with examples

5. **Comprehensive Learning** ("学习", "完整了解", "深入理解"):
   - Load: Level 3 (All documentation, ~40K tokens)
   - Use case: Complete system mastery

**Documentation Loaded into Memory**:
After loading, the selected documentation content is available in conversation memory for subsequent operations.

## 4. Usage Examples

### Example 1: Manual Specification

**User Command**:
```bash
/memory:load-skill-memory my_project "修改认证模块增加OAuth支持"
```

**Execution**:
```javascript
// Step 1: Use provided skill_name
skill_name = "my_project"  // Directly from parameter

// Step 2: Activate SKILL
Skill(command: "my_project")

// Step 3: Intent Analysis
Keywords: ["修改", "认证模块", "增加", "OAuth"]
Action: modifying (implementation)
Scope: auth module + examples

// Load documentation based on intent
Read(.workflow/docs/my_project/auth/README.md)
Read(.workflow/docs/my_project/auth/API.md)
Read(.workflow/docs/my_project/EXAMPLES.md)
```

### Example 2: Auto-Detection from Path

**User Command**:
```bash
/memory:load-skill-memory "修改D:\projects\my_project\src\services\api.py的接口逻辑"
```

**Execution**:
```javascript
// Step 1: Auto-detect skill_name from path
Path detected: "D:\projects\my_project\src\services\api.py"
Extracted: "my_project"
Validated: .claude/skills/my_project/ exists ✓
skill_name = "my_project"

// Step 2: Activate SKILL
Skill(command: "my_project")

// Step 3: Intent Analysis
Keywords: ["修改", "services", "接口逻辑"]
Action: modifying (implementation)
Scope: services module + examples

// Load documentation based on intent
Read(.workflow/docs/my_project/services/README.md)
Read(.workflow/docs/my_project/services/API.md)
Read(.workflow/docs/my_project/EXAMPLES.md)
```

## 5. Intent Keyword Mapping

**Quick Reference**:
- **Triggers**: "了解", "快速", "什么是", "简介"
- **Loads**: README.md only (~2K)

**Module-Specific**:
- **Triggers**: "XXX模块", "XXX组件", "分析XXX"
- **Loads**: Module README + API (~5K)

**Architecture**:
- **Triggers**: "架构", "设计", "整体结构", "系统设计"
- **Loads**: README + ARCHITECTURE (~10K)

**Implementation**:
- **Triggers**: "修改", "增强", "实现", "开发", "集成"
- **Loads**: Relevant module + EXAMPLES (~15K)

**Comprehensive**:
- **Triggers**: "完整", "深入", "全面", "学习整个"
- **Loads**: All documentation (~40K)
