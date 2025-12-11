# Command Guide Update Guideline

## 📋 Purpose

This document defines a **standardized, repeatable process** for updating command-guide documentation when command changes are detected. Use this guideline every time you need to update command-guide SKILL documentation to ensure consistency and completeness.

---

## 🎯 Update Trigger Conditions

Execute this update process when ANY of the following conditions are met:

1. **New commands added** to `.claude/commands/`
2. **Command parameters changed** (new flags, modified behavior)
3. **Command architecture refactored** (workflow reorganization)
4. **Agent implementations updated** in `.claude/agents/`
5. **User explicitly requests** command-guide update

---

## 📊 Phase 1: Analysis & Discovery

### Step 1.1: Identify Changed Files

**Objective**: Discover what has changed since last update

**Actions**:
```bash
# Find recent commits affecting commands/agents
git log --oneline --since="<last-update-date>" --grep="command\|agent\|workflow" -20

# Show detailed changes
git diff <last-commit>..<current-commit> --stat .claude/commands/ .claude/agents/

# Identify modified command files
git diff <last-commit>..<current-commit> --name-only .claude/commands/
```

**Output**: List of changed files and commit messages

**Document**:
- Changed command files
- Changed agent files
- Key commit messages
- Change patterns (new features, refactoring, fixes)

---

### Step 1.2: Analyze Change Scope

**Objective**: Understand the nature and impact of changes

**Questions to Answer**:
1. **What changed?** (parameters, workflow, architecture, behavior)
2. **Why changed?** (new feature, optimization, bug fix, refactoring)
3. **Impact scope?** (single command, workflow pattern, system-wide)
4. **User-facing?** (affects user commands, internal only)

**Analysis Matrix**:

| Change Type | Detection Method | Impact Level |
|-------------|--------------------|--------------|
| **New Parameter** | Diff `argument-hint` field | Medium |
| **Workflow Reorganization** | Multiple command changes | High |
| **Architecture Change** | Agent file changes + command changes | High |
| **Bug Fix** | Single file, small change | Low |
| **New Command** | New file in `.claude/commands/` | Medium-High |

**Output**: Change classification and impact assessment

---

### Step 1.3: Map Affected Documentation

**Objective**: Identify which documentation files need updates

**Mapping Rules**:

**Command Changes** → Affects:
- `reference/commands/<category>/<command-name>.md` (copy from source)
- `index/all-commands.json` (regenerate)
- `index/by-category.json` (if new category)
- `guides/ui-design-workflow-guide.md` (if UI workflow affected)
- `guides/workflow-patterns.md` (if workflow pattern changed)

**Agent Changes** → Affects:
- `reference/agents/<agent-name>.md` (copy from source)
- `guides/implementation-details.md` (if agent behavior changed)

**Workflow Reorganization** → Affects:
- All related command references
- Workflow guides
- Examples in guides

**Output**: Checklist of files to update

---

## 🔧 Phase 2: Content Preparation

### Step 2.1: Extract Key Information

**Objective**: Gather information needed for documentation updates

**Extract from Git Commits**:
```bash
# Get commit details
git show <commit-hash> --stat

# Extract commit message
git log --format=%B -n 1 <commit-hash>
```

**Information to Extract**:
1. **Feature Name** (from commit message)
2. **Change Description** (what was added/modified/removed)
3. **Rationale** (why the change was made)
4. **New Parameters** (from diff)
5. **Breaking Changes** (backward compatibility impact)
6. **Usage Examples** (from commit or command file)

**Output**: Structured data for documentation

---

### Step 2.2: Categorize Changes

**Objective**: Organize changes into logical categories

**Categories**:

1. **Major Features**
   - New commands
   - New workflows
   - Architecture changes
   - User-facing feature additions

2. **Enhancements**
   - New parameters
   - Improved behavior
   - Performance optimizations
   - Better error handling

3. **Refactoring**
   - Code reorganization (no user impact)
   - Internal structure changes
   - Consistency improvements

4. **Bug Fixes**
   - Corrected behavior
   - Fixed edge cases
   - Parameter validation fixes

5. **Documentation**
   - Updated descriptions
   - New examples
   - Clarified usage

**Output**: Changes grouped by category with priority

---

### Step 2.3: Analyze User Impact

**Objective**: Determine what users need to know

**User Impact Questions**:
1. **Do existing workflows break?** → Migration guide needed
2. **Are new features optional?** → Enhancement documentation
3. **Is behavior significantly different?** → Usage pattern updates
4. **Do examples need updates?** → Example refresh required

**Impact Levels**:
- **Critical** (Breaking changes, migration required)
- **Important** (New features users should adopt)
- **Nice-to-have** (Enhancements, optional)
- **Internal** (No user action needed)

**Output**: User-facing change summary with impact levels

---

## 📝 Phase 3: Documentation Updates

### Step 3.1: Update Reference Documentation

**Objective**: Sync reference docs with source command files

**Actions**:

1. **Run Python Script to Sync & Rebuild**:
   ```bash
   cd /d/Claude_dms3/.claude/skills/command-guide
   python scripts/analyze_commands.py
   ```

   This script automatically:
   - Deletes existing `reference/` directory
   - Copies all agent files from `.claude/agents/` to `reference/agents/`
   - Copies all command files from `.claude/commands/` to `reference/commands/`
   - Regenerates all 5 index files with updated metadata

2. **Verify Completeness**:
   - Check sync output for file counts (11 agents + 70 commands)
   - Verify all 5 index files regenerated successfully
   - Ensure YAML frontmatter integrity in copied files

**Output**: Updated reference documentation matching source + regenerated indexes

---

### Step 3.2: Update Workflow Guides

**Objective**: Reflect changes in user-facing workflow guides

**Workflow Guide Update Pattern**:

**IF** (UI workflow commands changed):
1. Open `guides/ui-design-workflow-guide.md`
2. Locate affected workflow pattern sections
3. Update examples to use new parameters/behavior
4. Add "New!" badges for new features
5. Update performance metrics if applicable
6. Add troubleshooting entries for new issues

**IF** (General workflow patterns changed):
1. Open `guides/workflow-patterns.md`
2. Update affected workflow examples
3. Add new pattern sections if applicable

**Update Template for New Features**:
```markdown
### [Feature Name] (New!)

**Purpose**: [What this feature enables]

**Usage**:
```bash
[Example command with new feature]
```

**Benefits**:
- [Benefit 1]
- [Benefit 2]

**When to Use**:
- [Use case 1]
- [Use case 2]
```

**Output**: Updated workflow guides with new features documented

---

### Step 3.3: Update Examples and Best Practices

**Objective**: Ensure examples reflect current best practices

**Example Update Checklist**:
- [ ] Remove deprecated parameter usage
- [ ] Add examples for new parameters
- [ ] Update command syntax if changed
- [ ] Verify all examples are runnable
- [ ] Add "Note" sections for common pitfalls

**Best Practices Update**:
- [ ] Add recommendations for new features
- [ ] Update "When to Use" guidelines
- [ ] Revise performance optimization tips
- [ ] Update troubleshooting entries

**Output**: Current, runnable examples

---

### Step 3.4: Update SKILL.md Metadata

**Objective**: Keep SKILL.md current without version-specific details

**Update Sections**:

1. **Supporting Guides** (if new guide added):
   ```markdown
   - **[New Guide Name](guides/new-guide.md)** - Description
   ```

2. **System Statistics** (if counts changed):
   ```markdown
   - **Total Commands**: <new-count>
   - **Total Agents**: <new-count>
   ```

3. **Remove Old Changelog Entries**:
   - Keep only last 3 changelog entries
   - Archive older entries to separate file if needed

**DO NOT**:
- Add version numbers
- Add specific dates
- Create time-based changelog entries

**Output**: Updated SKILL.md metadata

---

## 🧪 Phase 4: Validation

### Step 4.1: Consistency Check

**Objective**: Ensure documentation is internally consistent

**Checklist**:
- [ ] All command references use correct names
- [ ] Parameter descriptions match command files
- [ ] Examples use valid parameter combinations
- [ ] Links between documents are not broken
- [ ] Index files reflect current command count

**Validation Commands**:
```bash
# Check for broken internal links
grep -r "\[.*\](.*\.md)" guides/ reference/ | grep -v "http"

# Verify command count consistency
actual=$(find ../../commands -name "*.md" | wc -l)
indexed=$(jq '.commands | length' index/all-commands.json)
echo "Actual: $actual, Indexed: $indexed"
```

**Output**: Validation report

---

### Step 4.2: Example Testing

**Objective**: Verify all examples are runnable

**Test Cases**:
- [ ] Copy example commands from guides
- [ ] Run in test environment
- [ ] Verify expected output
- [ ] Document any prerequisites

**Note**: Some examples may be illustrative only; mark these clearly

**Output**: Tested examples

---

### Step 4.3: Peer Review Checklist

**Objective**: Prepare documentation for review

**Review Points**:
- [ ] Is the change clearly explained?
- [ ] Are examples helpful and clear?
- [ ] Is migration guidance complete (if breaking)?
- [ ] Are troubleshooting tips adequate?
- [ ] Is the documentation easy to scan?

**Output**: Review-ready documentation

---

## 📤 Phase 5: Commit & Distribution

### Step 5.1: Git Commit Structure

**Objective**: Create clear, traceable commits

**Commit Pattern**:
```bash
git add .claude/skills/command-guide/

# Commit message format
git commit -m "docs(command-guide): update for <feature-name> changes

- Update reference docs for <changed-commands>
- Enhance <guide-name> with <feature> documentation
- Regenerate indexes (new count: <count>)
- Add troubleshooting for <new-issues>

Refs: <commit-hashes-of-source-changes>
"
```

**Commit Message Rules**:
- **Type**: `docs(command-guide)`
- **Scope**: Always `command-guide`
- **Summary**: Concise, imperative mood
- **Body**: Bullet points for each change type
- **Refs**: Link to source change commits

**Output**: Clean commit history

---

### Step 5.2: Update Distribution

**Objective**: Make updates available to users

**Actions**:
```bash
# Push to remote
git push origin main

# Verify GitHub reflects changes
# Check: https://github.com/<org>/<repo>/tree/main/.claude/skills/command-guide
```

**User Notification** (if breaking changes):
- Update project README
- Add note to main documentation
- Consider announcement in team channels

**Output**: Published updates

---

## 🔄 Phase 6: Iteration & Improvement

### Step 6.1: Gather Feedback

**Objective**: Improve documentation based on usage

**Feedback Sources**:
- User questions about changed commands
- Confusion points in examples
- Missing information requests
- Error reports

**Track**:
- Common questions → Add to troubleshooting
- Confusing examples → Simplify or expand
- Missing use cases → Add to guides

**Output**: Improvement backlog

---

### Step 6.2: Continuous Refinement

**Objective**: Keep documentation evolving

**Regular Tasks**:
- [ ] Review index statistics monthly
- [ ] Update examples with real-world usage
- [ ] Consolidate redundant sections
- [ ] Expand troubleshooting based on issues
- [ ] Refresh screenshots/outputs if UI changed

**Output**: Living documentation

---

## 📐 Update Decision Matrix

Use this matrix to determine update depth:

| Change Scope | Reference Docs | Workflow Guides | Examples | Indexes | Migration Guide |
|--------------|----------------|-----------------|----------|---------|-----------------|
| **New Parameter** | Update command file | Add parameter note | Add usage example | Regenerate | No |
| **Workflow Refactor** | Update all affected | Major revision | Update all examples | Regenerate | If breaking |
| **New Command** | Copy new file | Add workflow pattern | Add examples | Regenerate | No |
| **Architecture Change** | Update all affected | Major revision | Comprehensive update | Regenerate | Yes |
| **Bug Fix** | Update description | Add note if user-visible | Fix incorrect examples | No change | No |
| **New Feature** | Update affected files | Add feature section | Add feature examples | Regenerate | No |

---

## 🎯 Quality Gates

Before considering documentation update complete, verify:

### Gate 1: Completeness
- [ ] All changed commands have updated reference docs
- [ ] All new features are documented in guides
- [ ] All examples are current and correct
- [ ] Indexes reflect current state

### Gate 2: Clarity
- [ ] Non-expert can understand changes
- [ ] Examples demonstrate key use cases
- [ ] Migration path is clear (if breaking)
- [ ] Troubleshooting covers common issues

### Gate 3: Consistency
- [ ] Terminology is consistent across docs
- [ ] Parameter descriptions match everywhere
- [ ] Cross-references are accurate
- [ ] Formatting follows established patterns

### Gate 4: Accessibility
- [ ] Table of contents is current
- [ ] Search/navigation works
- [ ] Related docs are linked
- [ ] Issue templates reference new content

---

## 🚀 Quick Start Template

When updates are needed, follow this abbreviated workflow:

```bash
# 1. ANALYZE (5 min)
git log --oneline --since="<last-update>" --grep="command\|agent" -20
# → Identify what changed

# 2. EXTRACT (10 min)
git show <commit-hash> --stat
git diff <commit>..HEAD --stat .claude/commands/
# → Understand changes

# 3. UPDATE (30 min)
# - Update affected guide sections (ui-design-workflow-guide.md, etc.)
# - Add examples for new features
# - Document parameter changes

# 4. SYNC & REGENERATE (2 min)
cd /d/Claude_dms3/.claude/skills/command-guide
python scripts/analyze_commands.py
# → Syncs reference docs + regenerates all 5 indexes

# 5. VALIDATE (10 min)
# - Test examples
# - Check consistency
# - Verify links

# 6. COMMIT (5 min)
git add .claude/skills/command-guide/
git commit -m "docs(command-guide): update for <feature> changes"
git push origin main
```

**Total Time**: ~1 hour for typical update

---

## 🔗 Related Resources

- **Python Index Script**: `.claude/skills/command-guide/scripts/analyze_commands.py`
- **Issue Templates**: `.claude/skills/command-guide/templates/`
- **SKILL Entry Point**: `.claude/skills/command-guide/SKILL.md`
- **Reference Source**: `.claude/commands/` and `.claude/agents/`

---

## 📌 Appendix: Common Patterns

### Pattern 1: New Parameter Addition

**Example**: `--interactive` flag added to `explore-auto`

**Update Sequence**:
1. Update `guides/ui-design-workflow-guide.md` with interactive examples
2. Add "When to Use" guidance
3. Run Python script to sync reference docs and regenerate indexes
4. Update argument-hint in examples

---

### Pattern 2: Workflow Reorganization

**Example**: Layout extraction split into concept generation + selection

**Update Sequence**:
1. Major revision of workflow guide section
2. Update all workflow examples
3. Add migration notes for existing users
4. Update troubleshooting
5. Run Python script to sync and regenerate indexes

---

### Pattern 3: Architecture Change

**Example**: Agent execution model changed

**Update Sequence**:
1. Update `guides/implementation-details.md`
2. Revise all workflow patterns using affected agents
3. Create migration guide
4. Update examples comprehensively
5. Run Python script to sync and regenerate indexes
6. Add extensive troubleshooting

---

**End of Update Guideline**

This guideline is a living document. Improve it based on update experience.
