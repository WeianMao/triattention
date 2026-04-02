# Development Standards for TriAttention Release Execution

> **Audience**: Every agent working on any step of the execution plan.
> **Authority**: This document is binding. Deviations require explicit user approval.

## 0. Controller 上岗必读（硬性要求）

**Controller agent 在开始调度任何任务之前，必须先通读以下全部文档**，了解项目全貌：

1. `release_doc/CURRENT_STATUS.md` — 当前状态
2. `release_doc/plan/execution_plan.md` — 完整执行计划
3. `release_doc/plan/dev_standards.md` — 本文件（开发规范）
4. `release_doc/plan/checkpoint_protocol.md` — 检查点协议
5. `release_doc/plan/execution_log.md` — 执行日志（了解已完成的工作）
6. `release_doc/tracking/14_open_items.md` — 所有决策记录
7. `release_doc/execution/15_checklist.md` — 完整待办清单
8. `release_doc/scope/01_overview.md` — 项目概览
9. `release_doc/code_cleanup/05_repo_structure.md` — 目标 repo 结构
10. `release_doc/execution/12_environment.md` — 环境信息

**不读完这些文档就开始调度 = 盲人指路。** 这不是建议，是硬性要求。

Executor agent 至少需要读 1-5 + 与自己步骤相关的具体文档。

---

## 1. Ambiguity 处理与未确认决策

### 风险分级处理

| 风险等级 | 判断标准 | 处理方式 |
|---------|---------|---------|
| **低** | 纯代码风格（缩进、注释措辞等） | 直接按代码库现有风格处理，不记录 |
| **中** | 有多种合理方案，但不影响功能正确性（如函数名选择） | 选最符合已有决策风格的方案执行，**记录到未确认决策日志** |
| **高** | 可能影响功能正确性、学术合规性、或用户体验 | **停止该项**，记录为 BLOCKING，继续做其他不受影响的工作 |

### 未确认决策日志

文件：`release_doc/plan/unconfirmed_decisions.md`

Agent 遇到中/高风险 ambiguity 时，必须记录到此文件。用户事后可让 agent 批量审查。

### 决策风格参考

Agent 做判断时应参考的用户已有决策风格：
- **信息最小化** — 不暴露不必要的内部信息
- **用户体验优先** — 开箱即用 > 灵活配置
- **保守安全** — 宁可多删不该有的，不可漏掉敏感信息
- **命名统一** — TriAttention 体系，不混用内部名
- **不过度工程** — 简单直接，不搞花哨抽象

### 常见 Edge Cases

| Case | 处理方式 |
|------|---------|
| 文件在 release 范围但 release_doc 中没提到 | 检查内容，明显属于已确认类别就按规则处理，否则记录为中风险 UD |
| 代码改名后发现有隐式依赖没文档化 | 追踪依赖链修复，记录为 UD |
| Flag 在删除清单但有代码路径还在用 | 删除 flag + 对应代码路径。不确定范围则记录为高风险 UD 并暂停 |
| Stats .pt 文件内容和文档描述不一致 | 以实际文件为准，更新文档，记录为 UD |
| 需要判断某段代码是社区还是原创 | 保守处理：加 attribution，记录为低风险 UD |
| Config 中发现未知参数 | 保留原值不动，记录为中风险 UD |

---

## 2. Working Directory Rules (原 dc1/ 不动)

### Sacred Rule: Never Modify `dc1/`

All code changes happen in `dc1-release/` (the worktree on `release/public` branch). The original `dc1/` directory is read-only reference material.

**Exception**: Running the original code in `dc1/` for head-to-head comparison (Phase 5, Step 5.4) is read-only usage, not modification.

### Path Convention

- Always use absolute paths in agent commands: `/data/rbg/users/weian/project/rl/dc1-release/`
- Never `cd` into a directory without recording where you are
- When referencing files in documentation, use paths relative to repo root (e.g., `triattention/triattention.py`)

---

## 3. Commit Conventions

### Commit Message Format

Use conventional commits with a `release:` scope:

```
release(<component>): <description>

<body if needed>
```

Components: `cleanup`, `rename`, `import`, `flag`, `path`, `docs`, `test`, `calibration`, `dfs`, `structure`

### Examples

```
release(cleanup): remove excluded directories and files

Deleted weian_development/, paper_visualizations/, experiments/,
process masking files, and obsolete implementations.
```

```
release(rename): unify speckv -> triattention naming

Renamed all classes, functions, CLI flags, config keys, and
method identifiers per code_cleanup/04_naming.md mapping table.
```

```
release(import): rewrite weian_development.* imports to new packages

Rewrote 15+ files with weian_development.* imports to use
triattention/, kv_compress/, integration/ package paths.
Removed all sys.path.insert() hacks.
```

### Commit Frequency

- **One commit per logical unit of work** within a step
- For large steps (2.3, 2.4, 2.5), commit after each sub-action
- Never leave uncommitted work at the end of an agent session
- Every commit must leave the code in a compilable state (`python -m compileall . -q` passes)

### Branch Hygiene

- All work on `release/public` branch
- No merge commits -- linear history only
- If a checkpoint fails and requires re-work, new commits fix forward (do not rewrite history)

---

## 4. Naming Conventions During Rename

### Reference Document

The authoritative mapping is in `release_doc/code_cleanup/04_naming.md`. Do not deviate from it.

### Key Mappings (Quick Reference)

| Internal | Release |
|----------|---------|
| `speckv` | `triattention` |
| `SpeckVRKVStyle` | `TriAttention` |
| `rkv` (method name) | `r1kv` |
| `rkv/compression/` | `kv_compress/` |
| `--rkv-style-compression` | `--attention-layer-compression` |
| `--rkv-style-slack-trigger` | `--slack-budget-trigger` |
| `--sparse-*` (TriAttention-specific) | `--triattention-*` |
| `--sparse-*` (generic) | descriptive name without prefix |

### Rename Execution Order

1. **Directory/file renames first** (Step 2.3)
2. **Import paths second** (Step 2.4)
3. **String content last** (Step 2.5)

This order prevents broken imports during intermediate states.

### Python Variable/Attribute Names

When renaming Python identifiers:
- `speckv_budget` -> `triattention_budget`
- `rkv_style_compression` -> `attention_layer_compression`
- Internal variable names that don't appear in public API can keep short forms if unambiguous

### YAML Config Keys

- `method: speckv` -> `method: triattention`
- `method: rkv` -> `method: r1kv`
- `rkv_style_compression: true` -> `attention_layer_compression: true`
- Config file names: `deepseek_r1_qwen3_8b_64trace.yaml` -> keep descriptive, remove internal references

---

## 5. Import Restructuring Conventions

### Before (Internal)

```python
from weian_development.speckv.speckv_rkv_style import SpeckVRKVStyle
from weian_development.speckv.round_pruning_utils import SparseRoundPruner
import sys; sys.path.insert(0, "/data/rbg/users/weian/project/rl/dc1/R-KV/HuggingFace")
from rkv.compression.r1_kv import R1KVCompressor
```

### After (Release)

```python
from triattention.triattention import TriAttention
from triattention.pruning_utils import SparseRoundPruner  # class name may also be renamed
from kv_compress.r1_kv import R1KVCompressor
```

### Rules

1. **No sys.path hacks** -- rely on `pip install -e .` for all imports
2. **No relative imports crossing packages** -- use absolute imports between `triattention/`, `kv_compress/`, `integration/`
3. **Relative imports within a package are OK** -- `from .pruning_utils import ...` inside `triattention/`
4. **Every `__init__.py` exports the public API** -- users should be able to `from triattention import TriAttention`

---

## 6. Handling Files Not Released But Not Deleted

### Principle

Some files exist in `dc1/` (source) but must not appear in `dc1-release/` (release). These are handled differently based on category:

| Category | Source Action | Release Action | Example |
|----------|-------------|---------------|---------|
| Excluded directory | Keep in `dc1/` | Delete from `dc1-release/` | `weian_development/` |
| Excluded file in retained dir | Keep in `dc1/` | Delete from `dc1-release/` | `rkv/compression/speckv.py` |
| Internal-only script | Keep in `dc1/` | Never copy to `dc1-release/` | `rkv_sparse_round_calibrate.py` |
| Runtime artifacts | Keep in `dc1/` | Excluded by `.gitignore` | `R-KV/logs/`, `R-KV/outputs/` |

### Critical Check

When restructuring (`kv_compress/__init__.py`), verify it does NOT import `speckv.py` content. The generate-wrapper code path is not part of the release.

---

## 7. Edge Case Handling

### When You Discover Something Not in the Docs

1. **First**: Check all release_doc files (especially `tracking/14_open_items.md`, `execution/15_checklist.md`)
2. **If documented**: Follow the documented decision
3. **If not documented and low-risk** (e.g., a typo in a comment): Fix it, note in commit message
4. **If not documented and medium-risk** (e.g., unclear which import path to use): Make the conservative choice, document your decision in the commit message, and flag it for the next checkpoint agent
5. **If not documented and high-risk** (e.g., a function behaves differently than expected): STOP. Record the finding in `tracking/14_open_items.md`. Do not proceed past this point for the affected code. Continue with other independent work.

### Ambiguity Resolution Hierarchy

1. Existing release_doc documentation (highest authority)
2. Existing code behavior in `dc1/` (ground truth for what the code does)
3. Paper content (ground truth for what the code should do)
4. Conservative default (preserve existing behavior, flag for review)

### When Code Does Not Compile After a Change

1. Do NOT skip the error with try/except or comment-out
2. Trace the import chain to find the missing dependency
3. If it's a file that was deleted in Step 2.1, the import needs to be rewritten (Step 2.4) or the code path needs to be removed (Step 2.6)
4. If you cannot resolve it, commit what you have with a clear `# TODO: resolve import` comment and document in the checkpoint

---

## 8. Sensitive Content Rules

### Zero-Tolerance Keywords

These must NEVER appear in any file in `dc1-release/`:

| Keyword | Why |
|---------|-----|
| `weian` | Personal username |
| `/data/rbg` | Internal server path |
| `PD-L1` | Process masking |
| `mask_process` | Process masking |
| `csail` | Internal infrastructure |
| `gpu_occupier` | Internal tool |
| `/home/linxi` | Collaborator path |
| `/tmp/kewan` | Internal temp path |
| `CHANGELOG_weian` | Internal log |

### Context-Sensitive Keywords

| Keyword | Allowed Context | Forbidden Context |
|---------|----------------|-------------------|
| `aime` | Dataset name in user-facing code (`"aime24"`, `load_data("aime24")`) | Calibration references, stats file names, config comments mentioning calibration source |
| `linxi` | Git attribution/acknowledgements (if agreed) | Code paths, hardcoded directories |
| `rkv` | Method name `r1kv`, `R1-KV` in docs | Package names, import paths (must be `kv_compress`) |

### Stats File Metadata

Every `.pt` file must be verified to not contain:
- `trace_root`
- `dataset` (when referring to calibration dataset)
- `model_path` (when containing absolute paths)
- Any string matching zero-tolerance keywords above

---

## 9. Testing Standards

### Every Code Change Must Compile

After any modification, run:
```bash
python -m compileall /path/to/dc1-release/ -q
```
This is non-negotiable. Do not commit code that fails compilation.

### Test Hierarchy

| Level | What | When | GPU | Time |
|-------|------|------|-----|------|
| Compile check | `python -m compileall` | After every commit | No | <1s |
| Level 1 | Scoring function equivalence | Phase 4 | No | ~1s |
| Level 2 | Pruner + stats equivalence | Phase 4 | Minimal | ~5s |
| Level 3 | Full model head-to-head | Phase 5 | ~16GB | ~60s/model |

### Equivalence Testing Principles

- Original code in `dc1/` is the **ground truth**
- Release code must produce **numerically identical** results for:
  - Per-head frequency domain scores
  - Keep/evict token indices
  - KV cache peak size (budget + divide_length = ~2176 for budget=2048)
- Tolerance: `atol=1e-5` for floating point comparisons

---

## 10. Agent 角色与中断恢复

### 两种角色

| 角色 | 职责 | 日志更新频率 |
|------|------|-------------|
| **Controller（调度）** | 分配步骤、启动 executor、监控进度、决定是否纠偏 | 每个 executor 完成后**立即**更新 |
| **Executor（执行）** | 执行具体开发步骤 | 任务完成后**详细**记录执行状态 |

### 中断恢复日志

**所有 agent（不论角色）都必须维护执行日志**，因为任何 agent 都可能中途断掉。

日志位置：`release_doc/plan/execution_log.md`

#### Controller 日志要求

每次 executor 完成任务后，立即记录：
- 哪个步骤完成了
- 结果是否符合预期
- 下一步是什么
- 是否需要纠偏

#### Executor 日志要求

任务完成后详细记录：
- 做了什么（具体改了哪些文件）
- 有什么已知问题（没完全解决的、需要后续注意的）
- 哪些地方没有完全符合目标（及原因）
- 供后续 agent 复盘的信息

#### 日志格式

```markdown
## Step X.Y: [步骤名]
- **Agent**: controller / executor
- **时间**: YYYY-MM-DD
- **状态**: 完成 / 部分完成 / 失败
- **执行内容**: [具体做了什么]
- **已知问题**: [如有]
- **不符合目标之处**: [如有，附原因]
- **下一步**: [建议]
```

### Controller 的自主纠偏能力

Controller 不是机械执行计划的调度器。它有以下权力：

1. **启动调查 agent**：发现开发可能出现偏差时，启动新的 agent 调查当前状况
2. **执行重规划**：调查后如果确认需要调整，可以修改后续步骤的计划
3. **发起计划外检查**：不需要等到 checkpoint 才能检查，可以随时启动检查 agent 抽查开发质量
4. **不盲从计划**：如果实际情况和计划冲突，以实际情况为准，更新计划后继续

### 计划是蓝图，不是教条

**每个 agent 执行步骤前，必须：**
1. 读完该步骤涉及的所有 release_doc 文档
2. **读相关的实际代码**，理解具体情况
3. 根据实际情况执行，而非盲从步骤清单

计划提供方向和参考，但具体的代码改动、边界情况处理，必须基于 agent 对代码的实际理解来决定。如果代码的实际情况和计划描述不一致，以代码为准，记录差异。

---

## 11. Documentation During Execution

### What to Update

| Event | Update Where |
|-------|-------------|
| Complete a step | `CURRENT_STATUS.md` (mark step done) + `plan/execution_log.md` (详细记录) |
| Discover new issue | `tracking/14_open_items.md` (add entry) + `plan/execution_log.md` |
| Make a non-trivial decision | Commit message + relevant doc file + `plan/execution_log.md` |
| Encounter ambiguity | Note in commit message + flag for checkpoint + `plan/execution_log.md` |
| Finish an agent session | `CURRENT_STATUS.md` + `plan/execution_log.md` must reflect current state |

### CURRENT_STATUS.md Update Cadence

**Every 2-3 conversation turns** or after completing any sub-step, whichever comes first. This is a hard requirement from `guidelines/agent_workflow.md`.

---

## 12. Recovery Procedures

### If a Step Fails

1. Identify the specific failure point
2. Fix forward -- add new commits, do not rewrite history
3. Re-run the step's verification checks
4. If the failure cascades to later steps, re-run those steps too

### If a Checkpoint Fails

1. The checkpoint agent documents exactly what failed and why
2. The responsible step is re-run by a dev agent
3. After the fix, the checkpoint is re-run (not skipped)
4. Downstream steps that may be affected are identified and re-verified

### Nuclear Option

If the release branch is badly broken:
```bash
git worktree remove ../dc1-release
git branch -D release/public
# Start over from Step 1.1
```
This is safe because `dc1/` (main) is never modified.
