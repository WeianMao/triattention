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
- **`rkv`**: R-KV compression experiments (Python 3.10, torch 2.3.1+cu121, flash-attn 2.5.8)

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
