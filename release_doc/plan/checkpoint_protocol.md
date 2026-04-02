# Checkpoint Protocol

> **Checkpoint agents are reviewers, not developers.** They inspect, verify, and report. They never modify code. If something is wrong, they document it and specify which step must re-run.

---

## General Rules

1. A checkpoint agent reads the execution plan, dev standards, and all relevant release docs before starting
2. Checkpoints produce a **structured report** (pass/fail per item, with evidence)
3. A checkpoint passes only if ALL items pass
4. Failed checkpoints block downstream work until resolved
5. Checkpoint agents commit their report to `release_doc/plan/checkpoint_reports/` (create dir if needed)

## Report Template

Every checkpoint report follows this format:

```markdown
# Checkpoint C{N} Report

**Date**: YYYY-MM-DD
**Agent**: [identifier]
**Verdict**: PASS / FAIL

## Summary
[1-2 sentence overall assessment]

## Checklist

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | [description] | PASS/FAIL | [command output or file reference] |
| 2 | ... | ... | ... |

## Failures (if any)

### Failure 1: [title]
- **What**: [description of the failure]
- **Where**: [file path and line number]
- **Expected**: [what should have been true]
- **Actual**: [what was found]
- **Remediation**: Re-run Step X.Y with specific focus on [area]
- **Severity**: BLOCKING / NON-BLOCKING

## Notes
[Any observations, warnings, or suggestions for downstream steps]
```

---

## Checkpoint C1: Foundation Verify

**When**: After Phase 1 (Steps 1.1 + 1.2)
**Purpose**: Confirm the workspace and environment are correctly set up before any code changes begin.

### Checks

| # | Check | How to Verify | Pass Criteria |
|---|-------|--------------|---------------|
| 1 | Worktree exists | `git worktree list` | Shows `dc1-release` on `release/public` |
| 2 | Branch is correct | `cd ../dc1-release && git branch --show-current` | Returns `release/public` |
| 3 | Source tree intact | `ls ../dc1-release/R-KV/HuggingFace/rkv/` | Directory exists with expected files |
| 4 | Conda env exists | `conda activate triattention && python --version` | Python 3.10.x |
| 5 | PyTorch works | `python -c "import torch; assert torch.cuda.is_available()"` | No error |
| 6 | flash-attn works | `python -c "from flash_attn import flash_attn_func"` | No error |
| 7 | transformers version | `python -c "import transformers; assert transformers.__version__ >= '4.48.1'"` | No error |
| 8 | Original dc1 untouched | `cd /data/rbg/users/weian/project/rl/dc1 && git status` | No unexpected changes |

### Failure Handling

| Failed Check | Action |
|-------------|--------|
| 1-3 | Re-run Step 1.1 |
| 4-7 | Re-run Step 1.2 |
| 8 | CRITICAL: Investigate immediately -- someone modified the source |

---

## Checkpoint C2: Structure Verify

**When**: After Steps 2.1 + 2.2 (directory restructure + old dir cleanup complete)
**Purpose**: Confirm the repo structure matches the target before beginning content modifications.

### Checks

| # | Check | How to Verify | Pass Criteria |
|---|-------|--------------|---------------|
| 1 | Target dirs exist | `for d in triattention kv_compress integration evaluation scripts configs calibration data tests; do test -d ../dc1-release/$d \|\| echo "MISSING: $d"; done` | No MISSING output |
| 2 | Old dirs removed | `test ! -d ../dc1-release/R-KV && test ! -d ../dc1-release/speckv_experiments && test ! -d ../dc1-release/weian_development` | All pass |
| 3 | Excluded dirs gone | `for d in paper_visualizations experiments deepconf .claude .workflow; do test ! -d ../dc1-release/$d \|\| echo "PRESENT: $d"; done` | No PRESENT output |
| 4 | Key files in place | Spot-check: `test -f ../dc1-release/triattention/triattention.py && test -f ../dc1-release/kv_compress/r1_kv.py && test -f ../dc1-release/integration/modeling.py && test -f ../dc1-release/scripts/cli.py` | All exist |
| 5 | __init__.py files | `for d in triattention kv_compress integration evaluation; do test -f ../dc1-release/$d/__init__.py \|\| echo "MISSING: $d/__init__.py"; done` | No MISSING |
| 6 | DFS benchmark present | `test -d ../dc1-release/benchmarks/dfs` | Exists |
| 7 | No pycache/DS_Store | `find ../dc1-release -name "__pycache__" -o -name ".DS_Store" -o -name "*.pyc" \| wc -l` | Returns 0 |
| 8 | speckv.py excluded | `test ! -f ../dc1-release/kv_compress/speckv.py` | Not present |
| 9 | Process masking gone | `grep -rl "PD-L1_binder\|mask_process_command" ../dc1-release/ \| wc -l` | Returns 0 |
| 10 | setup.py exists | `test -f ../dc1-release/setup.py -o -f ../dc1-release/pyproject.toml` | At least one exists |

### Failure Handling

| Failed Check | Action |
|-------------|--------|
| 1-5, 10 | Re-run Step 2.1 |
| 2-3 | Re-run Step 2.2 |
| 6 | Re-run Step 2.2 (DFS sub-step) |
| 7 | Quick fix: `find ../dc1-release -name "__pycache__" -exec rm -rf {} +` then re-verify |
| 8-9 | Re-run Step 2.2 (deletion was incomplete) |

---

## Checkpoint C3: Code Cleanup Verify

**When**: After Steps 2.3 + 2.4 + 2.5 + 2.6 (all code cleanup complete)
**Purpose**: This is the most critical checkpoint. Verify that ALL naming, imports, flags, paths, and sensitive content are correctly handled.

### Checks

| # | Check | How to Verify | Pass Criteria |
|---|-------|--------------|---------------|
| **Compilation** | | | |
| 1 | All Python compiles | `python -m compileall ../dc1-release/ -q` | Exit code 0 |
| **Import Integrity** | | | |
| 2 | No weian_development imports | `grep -rn "weian_development" ../dc1-release/ --include="*.py" \| wc -l` | 0 |
| 3 | No sys.path hacks | `grep -rn "sys.path.insert\|sys.path.append" ../dc1-release/ --include="*.py" \| wc -l` | 0 |
| 4 | No old rkv imports | `grep -rn "from rkv\.\|import rkv\." ../dc1-release/ --include="*.py" \| wc -l` | 0 |
| 5 | Package imports work | `cd ../dc1-release && pip install -e . && python -c "from triattention.triattention import *; from kv_compress.r1_kv import *; from integration.modeling import *"` | No error |
| **Naming** | | | |
| 6 | No speckv in code | `grep -rni "speckv" ../dc1-release/ --include="*.py" --include="*.yaml" --include="*.sh" \| wc -l` | 0 |
| 7 | No rkv_style in code | `grep -rni "rkv_style" ../dc1-release/ --include="*.py" --include="*.yaml" \| wc -l` | 0 |
| 8 | No rkv_sharded in code | `grep -rni "rkv_sharded" ../dc1-release/ --include="*.py" --include="*.yaml" --include="*.sh" \| wc -l` | 0 |
| 9 | CLI --help works | `cd ../dc1-release && python scripts/cli.py --help` | Prints usage, no speckv/rkv_style in output |
| **Flags** | | | |
| 10 | Deleted flags gone | Check each of 14 deleted flags (list in execution_plan.md Step 2.6) | 0 matches each |
| 11 | Retained flags present | Spot-check: `--attention-layer-compression`, `--triattention-normalize-scores`, `--pruning-seed` appear in argparse | Found |
| 12 | Budget defaults correct | `grep -n "budget.*512\|budget.*2048" ../dc1-release/configs/*.yaml` | DS-Llama-8B = 512, others = 2048 |
| **Paths + Sensitive** | | | |
| 13 | No internal paths | `grep -rn "/data/rbg\|/home/weian\|/home/linxi\|/tmp/kewan" ../dc1-release/ --include="*.py" --include="*.yaml" --include="*.sh" \| wc -l` | 0 |
| 14 | No process masking | `grep -rn "PD-L1\|mask_process\|binder\|gpu_occupier" ../dc1-release/ \| wc -l` | 0 |
| 15 | No csail references | `grep -rn "csail" ../dc1-release/ \| wc -l` | 0 |
| 16 | No weian references | `grep -rn "weian" ../dc1-release/ \| wc -l` | 0 |
| 17 | AIME only in dataset context | `grep -rni "aime" ../dc1-release/ --include="*.py" --include="*.yaml"` | Manual review: only in dataset names, not calibration |
| **Attribution** | | | |
| 18 | 6 eval files have headers | Check `parser.py`, `examples.py`, `python_executor.py`, `data_loader.py`, `utils.py`, `evaluate.py` for ToRA/DeepSeek-Math attribution | Present with license |
| 19 | model_utils.py attribution | Check for Apache 2.0 license declaration | Present |
| **Structural Integrity** | | | |
| 20 | validate_stats_metadata check | Verify `validate_stats_metadata()` still works correctly with renamed template strings | Function exists and no byte-mismatch risk from rename |

### Failure Handling

| Failed Check | Action |
|-------------|--------|
| 1 | Critical -- identify which file(s) fail, trace to the step that broke them |
| 2-4 | Re-run Step 2.3 |
| 5 | Check __init__.py exports, re-run Step 2.1 if packages missing |
| 6-9 | Re-run Step 2.4 |
| 10-12 | Re-run Step 2.5 |
| 13-17 | Re-run Step 2.6 |
| 18-19 | Re-run Step 2.6 (attribution sub-step) |
| 20 | Investigate carefully -- may need manual fix to avoid breaking stats validation |

### Critical Note on Check 17

The word "aime" will appear in dataset names (e.g., `"aime24"` in `DATASET_SOURCES`, `dataset2key`, file names like `run_triattention_aime24.sh`). This is expected and correct -- AIME is a public benchmark. What must NOT appear is AIME referenced as a **calibration data source** (e.g., in stats file metadata, calibration script comments, or config files linking calibration to AIME).

---

## Checkpoint C4: Content Verify

**When**: After Phase 3 (all new content created)
**Purpose**: Verify new files are correct and complete.

### Checks

| # | Check | How to Verify | Pass Criteria |
|---|-------|--------------|---------------|
| 1 | Calibration script exists | `test -f ../dc1-release/scripts/calibrate.py` | Exists |
| 2 | Calibrate script compiles | `python -m py_compile ../dc1-release/scripts/calibrate.py` | No error |
| 3 | Calibrate takes raw text | Review script: no AIME template, no dataset-specific format | Confirmed |
| 4 | Stats files exist | `ls ../dc1-release/calibration/*.pt` | 3 files: qwen3_8b, dsqwen_7b, dsllama_8b |
| 5 | Stats metadata clean | Load each .pt, check no trace_root/dataset/model_path/aime/weian/data_rbg | All clean |
| 6 | setup.py correct | `cd ../dc1-release && pip install -e . && python -c "import triattention"` | Works |
| 7 | requirements.txt exists | `test -f ../dc1-release/requirements.txt` | Exists |
| 8 | requirements.txt complete | Cross-reference with `execution/12_environment.md` dependency list | All deps listed |
| 9 | Data loader works | `cd ../dc1-release && python -c "from evaluation.data_loader import load_data"` | Imports OK |
| 10 | README exists | `test -f ../dc1-release/README.md` | Exists |
| 11 | README has all sections | Check for 15 sections from `components/readme_outline.md` | All present |
| 12 | README placeholders marked | `grep "TODO" ../dc1-release/README.md` | Placeholders clearly marked |
| 13 | LICENSE exists | `test -f ../dc1-release/LICENSE` | Apache 2.0 full text |
| 14 | .gitignore exists | `test -f ../dc1-release/.gitignore` | Exists, uses relative paths |

### Failure Handling

| Failed Check | Action |
|-------------|--------|
| 1-3 | Re-run Step 3.1 |
| 4-5 | Re-run Step 3.2 |
| 6-8 | Re-run Step 3.3 |
| 9 | Re-run Step 3.4 |
| 10-14 | Re-run Step 3.5 |

---

## Checkpoint C5: Pre-Release Gate

**When**: After Phase 4 (all testing complete)
**Purpose**: Final go/no-go decision before creating the clean-room repo.

### This is the most comprehensive checkpoint. It re-verifies everything.

### Checks

| # | Check | How to Verify | Pass Criteria |
|---|-------|--------------|---------------|
| **Tests** | | | |
| 1 | Level 1 tests pass | `cd ../dc1-release && python -m pytest tests/test_triattention.py -v` | All pass |
| 2 | Level 2 tests pass | `cd ../dc1-release && python -m pytest tests/ -v` | All pass |
| 3 | Full compilation | `python -m compileall ../dc1-release/ -q` | Exit 0 |
| **Exhaustive Sensitive Scan** | | | |
| 4 | Zero-tolerance keywords | Scan ALL files (not just .py) for all keywords in dev_standards.md Section 7 | 0 hits |
| 5 | .pt file scan | Load all .pt files, serialize to string, scan for keywords | 0 hits |
| 6 | No absolute paths | `grep -rn "^/[a-z]" ../dc1-release/ --include="*.py" --include="*.yaml" --include="*.sh" \| grep -v "#!/\|# /\|http"` | 0 hits (excluding shebangs, comments, URLs) |
| **Completeness** | | | |
| 7 | All target dirs exist | Full list from `code_cleanup/05_repo_structure.md` | All present |
| 8 | All experiment configs | Cross-reference with `scope/experiment_settings.md` -- at minimum: 3 models x 3 datasets x triattention + baselines | Configs exist |
| 9 | DFS benchmark complete | `python -m py_compile ../dc1-release/benchmarks/dfs/*.py` | All compile |
| 10 | Calibration stats for all models | 3 .pt files in calibration/ | Present |
| **User Experience** | | | |
| 11 | Install from scratch | In clean env: `pip install -e .` | Works |
| 12 | CLI works | `python scripts/cli.py --help` | Prints clean help |
| 13 | README readable | Manual: open README.md, read through, no broken links to internal files | Passes review |
| **Checklist Cross-Reference** | | | |
| 14 | All 15_checklist.md items | Review `execution/15_checklist.md` -- every checked item verified | All addressed |

### Verdict

- **PASS**: Proceed to Phase 5 (Release)
- **FAIL with BLOCKING items**: Fix and re-checkpoint
- **FAIL with NON-BLOCKING items only**: Document exceptions, proceed with user approval

### Failure Handling

For any failure, trace back to the specific step that should have caught it:
- Test failures -> Phase 4 steps
- Sensitive content -> Step 2.6
- Missing files -> Step 2.3 or Phase 3
- Compilation failures -> trace the broken import/reference chain

---

## Checkpoint Scheduling Summary

```
Phase 1 ──> C1 (foundation)
Phase 2 (2.1-2.2) ──> C2 (structure)
Phase 2 (2.3-2.6) ──> C3 (cleanup) *** MOST CRITICAL ***
Phase 3 ──> C4 (content)
Phase 4 ──> C5 (pre-release gate) *** GO/NO-GO ***
```

Total: 5 checkpoints across the entire plan. Each requires a dedicated agent session focused solely on review.

---

## Checkpoint Agent Behavior Rules

1. **Read all relevant docs first** -- execution plan, dev standards, and the specific release_doc files for the area being checked
2. **Run every verification command** -- do not skip any check, even if "it probably works"
3. **Record exact command output** -- the report must include actual output, not just "passed"
4. **Be skeptical** -- look for edge cases the dev agent might have missed
5. **Check the spirit, not just the letter** -- if a renamed function still has `speckv` in its docstring, that's a fail even if the function name is correct
6. **Never fix code** -- only document what's wrong and which step should fix it
7. **Commit the report** -- save to `release_doc/plan/checkpoint_reports/C{N}_report.md` and commit
