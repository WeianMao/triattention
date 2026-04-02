# TriAttention Release Execution Plan

> **Replaces**: `stages/README.md` (old 14-step plan)
> **Status**: Active
> **Last updated**: 2026-04-01

## Design Principles

1. **Each step = one agent's workload** -- self-contained, with explicit inputs and outputs
2. **Checkpoints are separate agents** -- they only review, never modify code
3. **Parallelism is explicit** -- steps that can run concurrently are marked
4. **Recovery is granular** -- failures re-run individual steps, not whole phases
5. **The worktree provides natural rollback** -- if everything fails, delete and re-create

## Prerequisites

Before Phase 1 begins, confirm:
- All open items in `tracking/14_open_items.md` are marked confirmed (except GPT-OSS env info and execution ordering)
- The `main` branch is in a clean, committable state
- The operator has read this plan end-to-end

---

## Phase 1: Foundation Setup

**Goal**: Create the isolated release workspace and environment.

### Step 1.1: Create Worktree and Branch

- **Scope**: Git operations only
- **Input**: Clean `main` branch
- **Actions**:
  1. `git checkout main && git pull`
  2. `git branch release/public` (from current HEAD)
  3. `git worktree add ../dc1-release release/public`
  4. Verify: `ls ../dc1-release/R-KV/` shows expected content
- **Output**: `dc1-release/` directory on `release/public` branch
- **Verification**: `git worktree list` shows both worktrees; `cd ../dc1-release && git branch --show-current` returns `release/public`
- **Estimated workload**: Small
- **Commit convention**: No commits yet -- just setup

### Step 1.2: Create `triattention` Conda Environment

- **Scope**: Conda/pip operations (no code changes)
- **Input**: Access to conda, pip, CUDA toolkit
- **Actions**:
  1. `conda create -n triattention python=3.10 -y`
  2. Install PyTorch (match local CUDA -- see `execution/12_environment.md` for version matrix)
  3. Install flash-attn>=2.5.8
  4. Install transformers>=4.48.1, datasets, huggingface-hub, accelerate
  5. Install eval pipeline deps: pebble, sympy, latex2sympy2, word2number, antlr4-python3-runtime==4.7.2
  6. Install utility deps: numpy, pyyaml, tqdm, matplotlib, scipy, einops, sentencepiece, regex
  7. Verify imports:
     ```bash
     conda activate triattention
     python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
     python -c "from flash_attn import flash_attn_func; print('OK')"
     python -c "import transformers; print(transformers.__version__)"
     ```
- **Output**: Working `triattention` conda environment
- **Verification**: All 3 verify commands succeed
- **Estimated workload**: Small (but flash-attn compilation may take time)

> Steps 1.1 and 1.2 can run in **parallel**.

### CHECKPOINT 1: Foundation Verify

See `checkpoint_protocol.md` -- Checkpoint C1.

---

## Phase 2: Code Cleanup (in `dc1-release/`)

**Goal**: Transform internal codebase into release-ready code. All work happens in `dc1-release/`.

**Critical rule**: Every step in Phase 2 operates on `dc1-release/`. The original `dc1/` is never modified.

### Step 2.1: Delete Excluded Content

- **Scope**: File/directory deletion only -- no content editing
- **Input**: `dc1-release/` with full source tree
- **Actions** (all paths relative to `dc1-release/`):
  1. Delete excluded directories:
     - `weian_development/` (personal dev tools)
     - `paper_visualizations/` (not released)
     - `experiments/` (not Phase 1)
     - `scripts/gpu_occupier.py`, `scripts/test_gpu_occupier.py`, `scripts/test_gpu_workload.py`
     - `.claude/`, `.workflow/`
     - `deepconf/` (historical, unrelated)
     - `TriAttention/` (empty directory)
     - `R-KV/logs/`, `R-KV/outputs/`, `R-KV/vLLM/`, `R-KV/SGLang/` (symlinks to runtime artifacts)
     - `repository_archive/`, `R-KV-backup-*`
     - `release_doc/` (meta-docs, not part of release)
     - `docs/` (if present, internal docs)
  2. Delete excluded files within retained directories:
     - `R-KV/HuggingFace/rkv/compression/speckv.py` (generate-wrapper path, not released -- but see note below)
     - `evaluation/length_eval.py`
     - `evaluation/CHANGELOG_weian.md`
     - `R-KV/weian_development/` (if nested refs remain)
     - All `__pycache__/`, `.DS_Store`, `*.pyc`
  3. Delete process masking files:
     - `rkv_sharded_runner.py` (entire file)
     - Any `PD-L1_binder` related scripts
  4. Delete internal development logs:
     - Any `PROGRESS_SUMMARY.md` files
  5. Delete calibration scripts (not released -- new one will be written):
     - `rkv_sparse_round_calibrate.py`
     - `capture_qk_distributed.py`
  6. Delete obsolete implementations:
     - `sparse_round_pruner_prefill_keep.py`
     - `rkv_speckv_generate.py`
     - `analysiskv.py`
- **Output**: Slimmed directory tree with only releasable content
- **Verification**:
  ```bash
  # These must return zero results:
  find ../dc1-release -name "*.pyc" -o -name "__pycache__" -o -name ".DS_Store" | wc -l
  grep -rl "gpu_occupier" ../dc1-release/ | wc -l
  grep -rl "PD-L1_binder" ../dc1-release/ | wc -l
  test ! -d ../dc1-release/weian_development
  test ! -d ../dc1-release/paper_visualizations
  ```
- **Estimated workload**: Small
- **Note on `speckv.py`**: Do NOT delete from `dc1/` source. Only ensure it is absent from `dc1-release/`.

### Step 2.2: DFS Benchmark Code Integration

- **Scope**: Copy files from `linxi-dev` branch + apply 5 fixes
- **Input**: `dc1-release/` after Step 2.1; access to `linxi-dev` branch
- **Actions**:
  1. From `linxi-dev` branch, manually copy DFS benchmark files to `dc1-release/`:
     - `R-KV/linxi_development/AQA-Bench/dfs_state_query/` contents
     - Place in appropriate location under `dc1-release/` (e.g., `benchmarks/dfs/`)
  2. Apply 5 fixes (per `execution/15_checklist.md`):
     - Replace hardcoded `/home/linxi/...` paths (3 files) with relative paths or HF hub names
     - Deduplicate `build_prompt` function
     - Change bare `except:` to `except Exception:`
     - Translate or delete Chinese documentation
     - Delete `PROGRESS_SUMMARY.md`
- **Output**: DFS benchmark code integrated and cleaned
- **Verification**:
  ```bash
  grep -r "/home/linxi" ../dc1-release/benchmarks/dfs/ | wc -l  # must be 0
  grep -rn "except:" ../dc1-release/benchmarks/dfs/ | grep -v "except " | wc -l  # must be 0
  python -m py_compile ../dc1-release/benchmarks/dfs/*.py  # all must compile
  ```
- **Estimated workload**: Small-Medium

### Step 2.3: Directory Restructure + Package Reorganization

- **Scope**: Move and rename directories/files to match target repo structure (see `code_cleanup/05_repo_structure.md`)
- **Input**: `dc1-release/` after Steps 2.1 and 2.2
- **Actions**:
  1. Create target directory structure:
     ```
     triattention/       (our method -- from R-KV/weian_development/speckv/ core files)
     kv_compress/        (baselines -- from R-KV/HuggingFace/rkv/)
     integration/        (HF integration -- from R-KV/HuggingFace/rkv/modeling.py etc.)
     evaluation/         (eval pipeline -- from existing evaluation/)
     scripts/            (entry points -- from speckv_experiments/)
     configs/            (YAML configs -- from speckv_experiments/configs/)
     calibration/        (stats .pt files -- to be populated)
     data/               (auto-download, initially empty)
     tests/              (unit tests -- to be written)
     benchmarks/dfs/     (DFS benchmark -- from Step 2.2)
     ```
  2. Move files according to the mapping:
     - `R-KV/weian_development/speckv/speckv_rkv_style.py` -> `triattention/triattention.py`
     - `R-KV/weian_development/speckv/round_pruning_utils.py` -> `triattention/pruning_utils.py`
     - `R-KV/weian_development/speckv/prompt_utils.py` -> `triattention/prompt_utils.py`
     - `R-KV/HuggingFace/rkv/compression/r1_kv.py` -> `kv_compress/r1_kv.py`
     - `R-KV/HuggingFace/rkv/compression/snapkv.py` -> `kv_compress/snapkv.py`
     - `R-KV/HuggingFace/rkv/compression/h2o.py` -> `kv_compress/h2o.py`
     - `R-KV/HuggingFace/rkv/compression/streamingllm.py` -> `kv_compress/streamingllm.py`
     - `R-KV/HuggingFace/rkv/modeling.py` -> `integration/modeling.py`
     - `R-KV/HuggingFace/rkv/monkeypatch.py` -> `integration/monkeypatch.py`
     - Eval files -> `evaluation/`
     - `speckv_experiments/run_math.py` -> `scripts/run_math.py`
     - `speckv_experiments/rkv_sharded_dispatch.py` -> `scripts/dispatch.py`
     - `speckv_experiments/rkv_sharded_eval.py` -> `scripts/worker.py`
     - `speckv_experiments/merge_rkv_shards.py` -> `scripts/merge_shards.py`
     - `speckv_experiments/speckv_experiments_cli_v2.py` -> `scripts/cli.py`
     - `speckv_experiments/process_utils.py` -> `scripts/process_utils.py`
     - `speckv_experiments/rkv_cache_utils.py` -> `scripts/cache_utils.py`
  3. Create `__init__.py` for all packages: `triattention/`, `kv_compress/`, `integration/`, `evaluation/`
  4. Create stub `setup.py` / `pyproject.toml` with `find_packages()` for editable install
  5. Delete the now-empty original directories (`R-KV/`, `speckv_experiments/` etc.)
- **Output**: Clean repo structure matching `code_cleanup/05_repo_structure.md`
- **Verification**:
  ```bash
  # Target dirs exist
  for d in triattention kv_compress integration evaluation scripts configs calibration data tests; do
    test -d ../dc1-release/$d || echo "MISSING: $d"
  done
  # Old dirs gone
  test ! -d ../dc1-release/R-KV
  test ! -d ../dc1-release/speckv_experiments
  # Python packages importable (basic check)
  cd ../dc1-release && python -c "import triattention; import kv_compress; import integration"
  ```
- **Estimated workload**: Large (most complex step in the plan)

> Steps 2.1 and 2.2 can run in **parallel**. Step 2.3 depends on both.

### CHECKPOINT 2: Structure Verify

See `checkpoint_protocol.md` -- Checkpoint C2.

---

### Step 2.4: Systematic Import Rewrite

- **Scope**: Python import statements across all files in `dc1-release/`
- **Input**: `dc1-release/` after Step 2.3 (new directory structure)
- **Actions**:
  1. Rewrite all `weian_development.*` imports (15+ files identified in checklist):
     - Map each old import to new package path
     - `from weian_development.speckv.speckv_rkv_style import ...` -> `from triattention.triattention import ...`
     - `from weian_development.speckv.round_pruning_utils import ...` -> `from triattention.pruning_utils import ...`
     - etc.
  2. Rewrite all `rkv.compression.*` imports -> `kv_compress.*`
  3. Rewrite all `rkv.modeling` / `rkv.monkeypatch` imports -> `integration.*`
  4. Fix `latex2sympy` relative import hacks (18 locations, 3 copies x 7 each):
     - Add proper `__init__.py`
     - Replace `sys.path.insert()` with standard relative imports
  5. Remove ALL `sys.path.insert()` / `sys.path.append()` hacks in release scope (~60 locations):
     - CRITICAL: `R-KV/SGLang/eval.py` hardcoded `/tmp/kewan/...` -- delete line
     - HIGH: 16 project root hacks in integration/ and kv_compress/ -- delete, rely on `pip install -e .`
     - MEDIUM: latex2sympy hacks (handled in sub-step 4)
  6. Update `scripts/worker.py` PYTHONPATH logic -- remove hardcoded internal paths
  7. Verify every `.py` file compiles:
     ```bash
     find ../dc1-release -name "*.py" -exec python -m py_compile {} +
     ```
- **Output**: All imports use new package names; no sys.path hacks remain
- **Verification**:
  ```bash
  grep -rn "weian_development" ../dc1-release/ | wc -l    # must be 0
  grep -rn "sys.path.insert\|sys.path.append" ../dc1-release/ --include="*.py" | wc -l  # must be 0 (in release scope)
  grep -rn "from rkv\." ../dc1-release/ --include="*.py" | wc -l  # must be 0
  python -m compileall ../dc1-release/ -q  # must exit 0
  ```
- **Estimated workload**: Large

### Step 2.5: Naming Unification (speckv -> triattention)

- **Scope**: String replacements across all files in `dc1-release/`
- **Input**: `dc1-release/` after Step 2.4
- **Actions** (reference: `code_cleanup/04_naming.md`):
  1. **Class/function renames**:
     - `SpeckVRKVStyle` -> `TriAttention`
     - `apply_speckv_rkv_style_patch()` -> `apply_triattention_patch()`
     - All `speckv_*` functions -> `triattention_*`
  2. **Method identifier renames**:
     - `method: speckv` -> `method: triattention` (in YAML, Python, CLI)
     - `method: rkv` -> `method: r1kv` (in YAML, Python, CLI)
  3. **Flag renames** (per `code_cleanup/04_naming.md` and `code_cleanup/flag_cleanup.md`):
     - `--rkv-style-compression` -> `--attention-layer-compression`
     - `--rkv-style-slack-trigger` -> `--slack-budget-trigger`
     - `--sparse-normalize-scores` -> `--triattention-normalize-scores`
     - `--sparse-score-aggregation` -> `--triattention-score-aggregation`
     - `--sparse-offset-max-length` -> `--triattention-frequency-window`
     - `--sparse-stats-path` -> `--triattention-stats-file`
     - `--sparse-seed` -> `--pruning-seed`
     - `--sparse-round-window` -> `--round-window`
     - `--include-prefill-in-budget` -> `--count-prompt-tokens`
  4. **CLI entry point** (`scripts/cli.py`, 660 lines, 11 categories from `components/08_launcher.md`):
     - Docstring: `"R-KV SpeckV experiments wrapper"` -> TriAttention description
     - MODES list: `speckv` -> `triattention`
     - Root paths: adapt to release repo layout
     - Experiment name prefix: remove `speckv_` prefix
     - Subprocess paths: `weian_development/rkv_sharded_dispatch.py` -> `scripts/dispatch.py`
     - PD-L1_binder env vars: delete entirely
     - Help text: replace all `speckv`/`SpeckV` occurrences
     - Error messages: remove `_v2.sh` suffix references
     - `rkv_style_*` config keys: rename to match new flag names
     - Control flow: `if mode == "speckv":` -> `if mode == "triattention":`
     - `runner_defaults.yaml`: update all paths
  5. **Config YAML files**: update method names, flag names, paths
  6. **Remaining string occurrences**: grep for `speckv`, `SpeckV`, `SPECKV` and address each
- **Output**: No internal naming remains in any file
- **Verification**:
  ```bash
  grep -rni "speckv" ../dc1-release/ --include="*.py" --include="*.yaml" --include="*.sh" --include="*.md" | wc -l  # must be 0
  grep -rni "rkv_style" ../dc1-release/ --include="*.py" --include="*.yaml" | wc -l  # must be 0
  python -m compileall ../dc1-release/ -q  # must still compile
  ```
- **Estimated workload**: Large

### Step 2.6: Flag Cleanup (Delete Experimental Flags)

- **Scope**: Python argparse definitions + all code paths referencing deleted flags
- **Input**: `dc1-release/` after Step 2.5
- **Actions** (reference: `code_cleanup/flag_cleanup.md`):
  1. Delete 14 flags and ALL associated code paths:
     - `--simulate-bug-phase-offset`
     - `--simulate-attention-position-offset`
     - `--output-name` (dead code)
     - `--sparse-use-similarity`
     - `--sparse-similarity-mix-lambda`
     - `--use-rank-similarity-combination`
     - `--use-rank-aggregation`
     - `--disable-top-n-high-freq`
     - `--rkv-aligned-budget`
     - `--per-layer-pruning`
     - `--per-layer-aggregation`
     - `--sparse-head-limit` (always -1)
     - `--reset-cache-each-batch` (always False)
     - `--position-offset-patch` related: delete flag AND `position_offset_patch.py` import at worker L664
  2. Delete `budgets.yaml` entry for 1536 (not in paper Figure 5)
  3. Fix DS-Llama-8B default budget: ensure scripts pass `--budget 512` explicitly (paper Table 1)
  4. Verify `rkv_cache_utils.py` / `reset_model_cache`: if tied to `--reset-cache-each-batch`, delete together
  5. Investigate KV cache state reset bug (per `flag_cleanup.md`):
     - Check `integration/modeling.py` state variables (`self.length`, `self.config.compression`)
     - Check `triattention/triattention.py` pruner internal state
     - If bug exists: add proper reset logic (not a flag, just correct code)
- **Output**: No experimental flags remain; budget defaults are correct
- **Verification**:
  ```bash
  # Check none of the deleted flags appear
  for flag in simulate-bug-phase-offset simulate-attention-position-offset sparse-use-similarity \
    sparse-similarity-mix-lambda use-rank-similarity-combination use-rank-aggregation \
    disable-top-n-high-freq rkv-aligned-budget per-layer-pruning per-layer-aggregation \
    sparse-head-limit reset-cache-each-batch; do
    grep -rn "$flag" ../dc1-release/ --include="*.py" && echo "FAIL: $flag still present"
  done
  python -m compileall ../dc1-release/ -q
  ```
- **Estimated workload**: Medium

### Step 2.7: Path Cleanup + Sensitive Content Removal

- **Scope**: All hardcoded paths, sensitive strings, and metadata
- **Input**: `dc1-release/` after Step 2.6
- **Actions** (reference: `code_cleanup/06_path_cleanup.md`):
  1. **Model paths** -> HuggingFace hub names:
     - All YAML configs: `/data/rbg/.../DeepSeek-R1-0528-Qwen3-8B` -> `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
     - Shell scripts: same pattern
     - Python files: same pattern
  2. **Dataset paths** -> relative paths:
     - YAML: absolute paths -> `./data/{dataset}.jsonl`
  3. **Cache paths** -> env vars:
     - `VLLM_TORCH_COMPILE_CACHE` -> `${VLLM_CACHE_DIR:-.cache/vllm}`
  4. **Process masking removal**:
     - Delete `mask_process_command()` from `scripts/process_utils.py`
     - Delete all `PD-L1_binder` references
     - Delete all `VLLM_PROCESS_NAME_PREFIX` env var settings
  5. **AIME reference sanitization**:
     - `parser.py`, `utils.py`: generalize AIME-specific references
     - Config/script file names: no `aime` in names (already renamed in Step 2.5)
  6. **Attribution headers**: Add to 6 eval files per `components/07_evaluation.md`:
     - `parser.py`, `examples.py`, `python_executor.py`, `data_loader.py`, `utils.py`, `evaluate.py` (ToRA/DeepSeek-Math, MIT)
     - `model_utils.py`: supplement existing attribution with license declaration (Apache 2.0)
  7. **Eval cleanup**:
     - `run_math.py`: rewrite `import weian_development.*` (should be done in 2.4, verify here)
     - `rm_maj_eval.py`: clean `__main__` hardcoded paths
     - `model_utils.py` L448: clean `../models/codellama_7b/v1-16k` reference
  8. **Rewrite `.gitignore`**: fresh file with relative paths only
  9. **Sensitive keyword full scan** (keywords from `scope/03_scope_exclude.md`):
     - `aime` (calibration refs only -- dataset name in user-facing code is OK)
     - `weian`
     - `/data/rbg`
     - `PD-L1`
     - `mask_process`
     - `binder`
     - `csail`
     - `gpu_occupier`
     - `linxi` (except in attribution/acknowledgements if appropriate)
- **Output**: No sensitive content, no internal paths, proper attribution
- **Verification**:
  ```bash
  # Full sensitive keyword scan
  for kw in "weian" "/data/rbg" "PD-L1" "mask_process" "binder" "csail" "gpu_occupier" "/home/linxi" "/tmp/kewan"; do
    count=$(grep -rn "$kw" ../dc1-release/ --include="*.py" --include="*.yaml" --include="*.sh" --include="*.md" --include="*.txt" | wc -l)
    if [ "$count" -gt 0 ]; then echo "FAIL: '$kw' found $count times"; fi
  done
  # AIME check (only in dataset-name context, not calibration refs)
  grep -rni "aime" ../dc1-release/ --include="*.py" --include="*.yaml" | grep -vi "dataset\|data_loader\|load_data\|DATASET_SOURCES" && echo "WARN: aime in non-dataset context"
  ```
- **Estimated workload**: Large

> Steps 2.4, 2.5, 2.6 must be **sequential** (each modifies code the next depends on).
> Step 2.7 depends on 2.6.

### CHECKPOINT 3: Code Cleanup Verify

See `checkpoint_protocol.md` -- Checkpoint C3. This is the most critical checkpoint.

---

## Phase 3: Content Creation

**Goal**: Create new files that don't exist in the source repo.

### Step 3.1: Write Calibration Script (NEW)

- **Scope**: Create `scripts/calibrate.py` (new file, not a copy of internal scripts)
- **Input**: Understanding of calibration flow from `tracking/14_open_items.md` "Calibration Stats" section
- **Actions**:
  1. Write a new calibration script that:
     - Takes raw text input (no dataset template, no AIME format)
     - Runs fullkv inference to generate traces
     - Extracts Q/K frequency domain statistics per attention head
     - Saves as `.pt` file with clean metadata (no `trace_root`, no `dataset`, no `model_path`)
  2. The script interface:
     ```bash
     python scripts/calibrate.py \
       --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
       --input calibration_text.txt \
       --output calibration/qwen7b_stats.pt
     ```
  3. Strip any metadata that could reveal calibration data source
- **Output**: `scripts/calibrate.py` -- clean, documented calibration script
- **Verification**: `python -m py_compile scripts/calibrate.py`; manual review of output .pt metadata fields
- **Estimated workload**: Medium

### Step 3.2: Pre-generate Calibration Stats Files

- **Scope**: Generate `.pt` files for 3 non-GPT-OSS models (GPU required)
- **Input**: Working calibration pipeline (internal tools, NOT the new script); model weights
- **Actions**:
  1. For each model (DS-Qwen-7B, DS-Llama-8B, Qwen3-8B):
     - Run fullkv inference with AIME data (internal process, not public)
     - Extract stats using existing internal calibration pipeline
     - Save with neutral names: `qwen3_8b_stats.pt`, `dsqwen_7b_stats.pt`, `dsllama_8b_stats.pt`
  2. Strip metadata from each .pt file:
     - Remove `trace_root`, `dataset`, `model_path` fields
     - Keep only: model config, RoPE params, computation precision, per-head statistics
  3. Place in `dc1-release/calibration/`
- **Output**: 3 clean `.pt` files in `calibration/`
- **Verification**:
  ```python
  import torch
  for f in ["qwen3_8b_stats.pt", "dsqwen_7b_stats.pt", "dsllama_8b_stats.pt"]:
      data = torch.load(f"calibration/{f}")
      metadata = data.get("metadata", {})
      assert "trace_root" not in metadata
      assert "dataset" not in metadata
      assert "model_path" not in metadata
      # Check no string contains "aime" or "/data/rbg"
      for k, v in metadata.items():
          if isinstance(v, str):
              assert "aime" not in v.lower()
              assert "/data/rbg" not in v
  ```
- **Estimated workload**: Medium (GPU time)

### Step 3.3: Write setup.py / pyproject.toml

- **Scope**: Package configuration files
- **Input**: Target repo structure from Step 2.3; dependency list from `execution/12_environment.md`
- **Actions**:
  1. Write `setup.py` (or `pyproject.toml`) with:
     - `name="triattention"`, `version="0.1.0"`
     - `packages=find_packages()` covering `triattention/`, `kv_compress/`, `integration/`, `evaluation/`
     - Dependencies matching the `triattention` conda env
     - `python_requires=">=3.10"`
  2. Write `requirements.txt` aligned with the conda env (see version table in `execution/12_environment.md`)
     - `torch` with comment about CUDA-specific install
     - Pin `antlr4-python3-runtime==4.7.2`
     - All other deps with minimum versions
  3. Verify editable install:
     ```bash
     cd ../dc1-release && pip install -e . && python -c "import triattention; import kv_compress"
     ```
- **Output**: `setup.py`, `requirements.txt`
- **Verification**: `pip install -e .` succeeds; packages importable
- **Estimated workload**: Small

### Step 3.4: Write Data Loader with Auto-Download

- **Scope**: `evaluation/data_loader.py` or `scripts/data_loader.py`
- **Input**: Dataset info from `scope/datasets.md`
- **Actions**:
  1. Implement auto-download from HuggingFace:
     - `aime24` from `HuggingFaceH4/aime_2024`
     - `aime25` from `MathArena/aime_2025`
     - `math500` from `HuggingFaceH4/MATH-500`
  2. Handle field name mapping (`problem` -> `question` for AIME datasets)
  3. Cache downloaded data to `./data/` directory
  4. Preserve existing 3-layer field name fallback mechanism
- **Output**: Data loader that auto-downloads on first run
- **Verification**: Delete local cache, run loader, confirm data downloads and loads correctly
- **Estimated workload**: Small-Medium

### Step 3.5: Write README and Docs

- **Scope**: `README.md`, `LICENSE`
- **Input**: README outline from `components/readme_outline.md`; all experiment settings from `scope/experiment_settings.md`
- **Actions**:
  1. Write `README.md` following the outline (15 sections)
     - Use placeholder markers for: demo video, method figure, arXiv link, exact GitHub URL, BibTeX
     - Fill in: installation, quick start, model table, result tables (from paper), reproduction guide
  2. Write `LICENSE` file (Apache 2.0 full text)
  3. Write `.gitignore` (Python-standard + project-specific: `data/`, `calibration/*.pt` if user-generated, `*.pyc`, etc.)
- **Output**: `README.md`, `LICENSE`, `.gitignore`
- **Verification**: Manual review; check all placeholders are clearly marked with `<!-- TODO: ... -->`
- **Estimated workload**: Medium

> Steps 3.1 and 3.3 and 3.5 can run in **parallel** (no dependencies between them).
> Step 3.2 depends on 3.1 conceptually but uses internal tools, so it can run in parallel with 3.1.
> Step 3.4 can run in parallel with others.

### CHECKPOINT 4: Content Verify

See `checkpoint_protocol.md` -- Checkpoint C4.

---

## Phase 4: Testing and Verification

**Goal**: Ensure the release code is correct and clean.

### Step 4.1: Level 1 Unit Tests (No GPU, ~1 second)

- **Scope**: Write and run `tests/test_triattention.py`
- **Input**: Working `dc1-release/` with `pip install -e .`
- **Actions**:
  1. Write pure scoring function equivalence test:
     - Synthetic tensor construction for K_unrot / K_rot
     - Original `score_keys_for_round()` vs release `compute_scores_pytorch()` (or equivalent)
     - Tolerance: `atol=1e-5`
     - Note alignment: original scores K_unrot, release scores K_rot
  2. Run test:
     ```bash
     cd ../dc1-release && python -m pytest tests/test_triattention.py -v
     ```
- **Output**: Passing Level 1 tests
- **Verification**: pytest exit code 0
- **Estimated workload**: Small

### Step 4.2: Level 2 Unit Tests (Minimal GPU, ~5 seconds)

- **Scope**: Write and run `tests/test_pruner_equivalence.py`
- **Input**: Working `dc1-release/` with calibration stats
- **Actions**:
  1. Write pruner equivalence test:
     - Load stats `.pt` + model `config.json` (only RoPE params, no model weights)
     - Compare original `SparseRoundPruner` vs release `TriAttentionCompressor` (or equivalent)
     - Compare per-head scores + keep/evict indices
     - Ensure `normalize_scores` and per-head sampling ranges align
  2. Write KV cache peak alignment test:
     - Verify `budget + divide_length` peak is consistent
  3. Run tests:
     ```bash
     cd ../dc1-release && python -m pytest tests/ -v
     ```
- **Output**: Passing Level 1+2 tests
- **Verification**: pytest exit code 0; <100MB GPU memory used
- **Estimated workload**: Small-Medium

### Step 4.3: Compilation and Import Verification

- **Scope**: Full codebase syntax check
- **Input**: Complete `dc1-release/`
- **Actions**:
  1. `python -m compileall dc1-release/ -q` -- must exit 0
  2. Verify all package imports work:
     ```bash
     cd ../dc1-release
     python -c "from triattention.triattention import TriAttention"
     python -c "from kv_compress.r1_kv import *"
     python -c "from integration.modeling import *"
     python -c "from evaluation.evaluate import *"
     ```
  3. Verify CLI loads: `python scripts/cli.py --help` (should print usage without error)
- **Output**: All code compiles and imports succeed
- **Verification**: All commands above exit 0
- **Estimated workload**: Small

### Step 4.4: Final Sensitive Content Scan

- **Scope**: Read-only scan of entire `dc1-release/`
- **Input**: Complete `dc1-release/`
- **Actions**:
  1. Run exhaustive keyword scan (expanded list):
     ```bash
     for kw in "weian" "/data/rbg" "PD-L1" "mask_process" "binder" "csail" \
       "gpu_occupier" "/home/linxi" "/tmp/kewan" "speckv" "dc1" "dc/R-KV" \
       "rkv_sharded" "rkv_speckv" "CHANGELOG_weian"; do
       grep -rn "$kw" ../dc1-release/ && echo "=== FOUND: $kw ==="
     done
     ```
  2. Scan `.pt` files for embedded strings:
     ```python
     import torch, json
     for f in glob("calibration/*.pt"):
         data = torch.load(f)
         text = json.dumps(str(data))
         for kw in ["aime", "weian", "/data/rbg", "csail"]:
             assert kw not in text.lower(), f"{kw} found in {f}"
     ```
  3. Scan for any remaining absolute paths: `grep -rn "^/" ../dc1-release/ --include="*.py" | grep -v "^#\|#!/"`
- **Output**: Clean scan report (zero findings or documented exceptions)
- **Verification**: All scans return 0 findings
- **Estimated workload**: Small

> Steps 4.1 and 4.2 are **sequential** (4.2 extends 4.1).
> Step 4.3 can run in **parallel** with 4.1/4.2.
> Step 4.4 can run in **parallel** with 4.1/4.2/4.3.

### CHECKPOINT 5: Pre-Release Gate

See `checkpoint_protocol.md` -- Checkpoint C5. **This is the go/no-go decision point.**

---

## Phase 5: Release

**Goal**: Produce the final clean-room public repository.

### Step 5.1: End-to-End Smoke Test

- **Scope**: Fresh-environment test of the complete user experience
- **Input**: Complete, verified `dc1-release/`
- **Actions**:
  1. In a fresh terminal with the `triattention` conda env:
     ```bash
     conda activate triattention
     cd /tmp/triattention-test
     cp -r /path/to/dc1-release/* .
     pip install -e .
     ```
  2. Verify data auto-download: `python -c "from evaluation.data_loader import load_data; load_data('math500')"`
  3. Verify CLI help: `python scripts/cli.py --help`
  4. Verify calibration script syntax: `python scripts/calibrate.py --help`
  5. Run Level 1+2 tests: `python -m pytest tests/ -v`
- **Output**: Smoke test pass
- **Verification**: All commands succeed
- **Estimated workload**: Small

### Step 5.2: Clean-Room Repository Creation

- **Scope**: Create the final public repo (no git history)
- **Input**: Verified `dc1-release/`
- **Actions**:
  1. Create clean directory:
     ```bash
     mkdir ~/triattention-public
     # Copy only tracked/intended files (not .git, not __pycache__)
     rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
       ../dc1-release/ ~/triattention-public/
     ```
  2. Initialize fresh git:
     ```bash
     cd ~/triattention-public
     git init
     git add .
     git commit -m "Initial release: TriAttention v0.1.0"
     ```
  3. Final verification in the clean repo:
     ```bash
     pip install -e .
     python -m pytest tests/ -v
     python -m compileall . -q
     ```
- **Output**: `~/triattention-public/` with single clean commit
- **Verification**: Tests pass in the clean repo
- **Estimated workload**: Small

### Step 5.3: Internal Team Review

- **Scope**: Share with team members for review (before public push)
- **Input**: `~/triattention-public/`
- **Actions**:
  1. Push to a private GitHub repo for team review
  2. Team members check:
     - Can clone and install
     - Can run basic examples
     - No sensitive content visible
     - README is clear
  3. Collect feedback, iterate if needed
- **Output**: Team sign-off
- **Verification**: At least 1 team member successfully clones, installs, and runs tests
- **Estimated workload**: Medium (calendar time for review)

### Step 5.4: GPU Head-to-Head Comparison (Level 3)

- **Scope**: Full model inference comparison (requires ~16GB GPU, ~60 seconds per model)
- **Input**: Both `dc1/` (original) and `triattention-public/` (release)
- **Actions**:
  1. For each of 3 models (DS-Qwen-7B, DS-Llama-8B, Qwen3-8B):
     - Run identical input through original code path
     - Run identical input through release code path
     - Compare keep/evict decisions at compression trigger points
     - Must be numerically identical
  2. Run at least 1 full AIME problem end-to-end on each model
- **Output**: Head-to-head comparison report (pass/fail)
- **Verification**: Numerical identity of compression decisions
- **Estimated workload**: Medium (GPU time)

### Step 5.5: Public Release

- **Scope**: Push to public GitHub
- **Input**: Team-reviewed `triattention-public/`
- **Actions**:
  1. Create public GitHub repo `TriAttention`
  2. Push clean-room repo
  3. Verify: clone from GitHub, install, run tests
  4. Update README placeholders with actual GitHub URL
- **Output**: Public repository live
- **Verification**: Fresh clone + install + test from another machine
- **Estimated workload**: Small

### Step 5.6: Cleanup

- **Scope**: Remove temporary workspaces
- **Actions**:
  1. `git worktree remove ../dc1-release`
  2. Optionally delete `release/public` branch if no longer needed
  3. Archive release docs
- **Output**: Clean workspace
- **Estimated workload**: Small

---

## Dependency Graph Summary

```
Phase 1: Foundation
  1.1 (worktree) ──┐
  1.2 (conda)   ───┤
                    ├── C1 (checkpoint)
                    v
Phase 2: Code Cleanup
  2.1 (delete) ────┐
  2.2 (DFS)    ────┤
                    ├── 2.3 (restructure)
                    │     │
                    │     v
                    │   2.4 (imports)
                    │     │
                    │     v
                    │   2.5 (naming)
                    │     │
                    │     v
                    │   2.6 (flags)
                    │     │
                    │     v
                    │   2.7 (paths + sensitive)
                    │     │
                    ├─────┤
                    v     v
                   C2    C3 (checkpoints)
                          │
                          v
Phase 3: Content Creation
  3.1 (calibrate script) ─┐
  3.2 (gen stats)      ───┤
  3.3 (setup.py)       ───┤
  3.4 (data loader)    ───┤
  3.5 (README/docs)    ───┤
                           ├── C4 (checkpoint)
                           v
Phase 4: Testing
  4.1 (L1 tests) ──> 4.2 (L2 tests) ─┐
  4.3 (compile check)              ────┤
  4.4 (sensitive scan)             ────┤
                                       ├── C5 (pre-release gate)
                                       v
Phase 5: Release
  5.1 (e2e smoke) ──> 5.2 (clean-room) ──> 5.3 (team review)
                                              │
                                              v
                                         5.4 (GPU h2h)
                                              │
                                              v
                                         5.5 (public release)
                                              │
                                              v
                                         5.6 (cleanup)
```

## Estimated Total Effort

| Phase | Steps | Parallelizable | Est. Agent Sessions |
|-------|-------|----------------|---------------------|
| Phase 1: Foundation | 2 | Yes (both parallel) | 1 session |
| Phase 2: Code Cleanup | 7 | Partially (2.1||2.2, then sequential) | 4-5 sessions |
| Phase 3: Content Creation | 5 | Mostly parallel | 2-3 sessions |
| Phase 4: Testing | 4 | Mostly parallel | 1-2 sessions |
| Phase 5: Release | 6 | Sequential | 2-3 sessions |
| **Total** | **24 steps + 5 checkpoints** | | **~12-15 agent sessions** |
