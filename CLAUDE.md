# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DeepConf (Deep Think with Confidence)** is an efficient parallel thinking framework built on vLLM for LLM reasoning tasks (math, science, coding). It provides confidence-based early stopping and multiple voting strategies for enhanced reasoning.

## Build & Development Commands

```bash
# Install dependencies
pip install deepconf
uv pip install -r requirements.txt

# Syntax check without running inference
python -m compileall deepconf examples

# Run online mode example
python examples/example_online.py --qid 0 --rid 0 --dataset brumo_2025.jsonl --total_budget 256 --output_dir online-dpsk

# Run baseline (no early-exit) for comparison
python examples/example_online_baseline.py --qid 0 --rid 0 --dataset brumo_2025.jsonl --total_budget 256 --output_dir baseline-dpsk

# Run offline mode
python examples/example_offline.py --help

# Analyze results
python examples/example_analyze_online.py --output_dir ./online-dpsk/ --max_qid 29 --rids 1
```

## Conda Environments

- **`dc`**: Default environment for DeepConf development
- **`lazy_evict`**: LazyEviction subproject (`conda activate lazy_evict` before running LazyEviction scripts)
- **`rkv`**: R-KV compression experiments for Qwen2.5 models (Python 3.10, torch 2.3.1+cu121, flash-attn 2.5.8, transformers 4.48.1)
- **`rkv1`**: R-KV compression experiments for Qwen3 models (cloned from `rkv`, transformers 4.57.3). Created 2025-12-28 for `aime_sampled8_qwen3` experiments. See `R-KV/weian_script/aime_sampled8_qwen3/MIGRATION_PLAN.md` for details.

## Architecture

### Core Library (`deepconf/`)
- `wrapper.py`: `DeepThinkLLM` class - main interface wrapping vLLM with `generate()` and `deepthink()` methods
- `outputs.py`: `DeepThinkOutput` dataclass for structured results (answers, traces, confidence, voting results)
- `utils.py`: Voting algorithms (`weighted_majority_vote`, `compute_all_voting_results`), result processing
- `processors.py`: `WrappedPerReqLogitsProcessor` for custom logits processing

### Modes of Operation
- **Online Mode**: Confidence-based early stopping with warmup traces and dynamic thresholds
- **Offline Mode**: Batch generation with multiple voting strategies applied post-hoc

### Key Directories
- `examples/`: Runnable scripts demonstrating online/offline workflows
- `scripts/`: YAML config loader (`config_loader.py`), configs in `scripts/configs/`, GPU dispatch scripts
- `weian_development/`: Helper scripts, HF offline runners (`hf_offline_runner/`, `hf_offline_runner_sparse/`), analysis tools
- `LazyEviction/`: Separate KV cache eviction subproject for long reasoning (uses `lazy_evict` env)
- `kvpress/`: KV cache compression library (external)
- `paper_visualizations/`: Paper visualization scripts and outputs (attention maps, frequency diagnostics, Q/K scatter plots). See `paper_visualizations/README.md` for detailed documentation.

### Configuration
YAML configs in `scripts/configs/` define model paths, sampling params (`temperature`, `top_p`, `top_k`), budgets, and output dirs. Key config: `deepseek_r1_qwen3_8b_64trace.yaml`.

### Datasets
Symlinked in repo root: `aime25.jsonl`, `brumo_2025.jsonl`. Format: JSONL with `{"question": "...", "answer": "..."}`.

## Code Style

- Python 3.9+, 4-space indentation
- snake_case for variables/functions, CapWords for classes
- Minimal docstrings; inline comments only for non-obvious logic
- Conventional Commits: `feat:`, `fix:`, `docs:`

## Process Naming Convention

Long-running jobs should use `PD-L1_binder` process name prefix (via wrapper scripts or `setproctitle`) for cluster compatibility.

## Code Isolation Policy

Do not modify existing algorithm core logic without explicit authorization. New algorithms/variants should be developed in isolated scripts/subclasses, reusing existing API interfaces.

## Release 工作规范（必读）

**如果你是新启动的 Controller Agent，请先读 `release_doc/CONTROLLER_START_HERE.md`。**

任何与 release 相关的任务，开始前必须先读以下文档：

1. `release_doc/CURRENT_STATUS.md` — 当前进展、断点恢复信息
2. `release_doc/plan/execution_plan.md` — **完整执行计划**（阶段、步骤、依赖、检查点）
3. `release_doc/plan/dev_standards.md` — **开发规范**（命名、commit、角色职责、中断恢复、纠偏机制）
4. `release_doc/plan/checkpoint_protocol.md` — **检查点协议**（检查内容、通过标准、失败处理）
5. `release_doc/plan/execution_log.md` — **执行日志**（已完成工作、已知问题）
6. `release_doc/plan/unconfirmed_decisions.md` — **未确认决策日志**（agent 自主判断的记录，供用户事后审查）
7. `release_doc/guidelines/agent_workflow.md` — agent 工作流程
8. `release_doc/guidelines/confirmation_protocol.md` — 与用户确认决策的标准流程

**关键规则**：
- 计划是蓝图，不是教条——必须读相关代码理解实际情况，根据实际情况执行
- Controller agent 有权启动调查 agent、发起计划外检查、执行重规划
- 所有 agent 必须维护 `plan/execution_log.md`，确保中断后下一个 agent 能接手
- 每 2-3 轮对话必须更新 `CURRENT_STATUS.md` 并 commit
- 详情见上述文档
