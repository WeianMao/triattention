# SpeckV Experiments (R-KV)

This folder provides an out-of-the-box experiment orchestration layer for SpeckV, R-KV, and FullKV runs.
All outputs, logs, and stats are written under `R-KV/speckv_experiments/`.

## Test Flow

Run from `R-KV/speckv_experiments/`:

```bash
bash scripts/download_models.sh     
bash scripts/run_fullkv.sh         
bash scripts/build_all_stats.sh    
bash scripts/run_rkv.sh          
bash scripts/run_speckv.sh       
bash scripts/run_all_sweep.sh    
```

Notes:
- `download_models.sh` downloads all required models into `models/`.
- `run_fullkv.sh` runs the FullKV evaluation.
- `build_all_stats.sh` calibrates stats based on FullKV outputs (require `run_fullkv.sh`).
- `run_rkv.sh` runs R-KV evaluation.
- `run_speckv.sh` runs SpeckV evaluation (require `build_att_stats.sh`).
- `run_all_sweep.sh` runs all budgets for R-KV + SpeckV (require `build_att_stats.sh`).
- For a single experiment, use `run_one.sh` with parameters.
- Model-specific wrappers live under `scripts/qwen3/` and `scripts/distill_qwen7b/`.

Shared runner defaults live at `configs/shared/runner_defaults.yaml`.

Use `DRY_RUN=1` with run scripts to print commands without executing. In dry-run mode, model paths are reported but not validated.

## Datasets

Expected dataset symlinks in R-KV/Huggingface/data:

- `aime24.jsonl`
- `aime25.jsonl`
- `math500.jsonl`

If `math500.jsonl` is missing, the runner falls back to `math.jsonl` when available.

## Output Layout

- Logs: `logs/<dataset>/<model>/<mode>/<budget>/run.log`
- Outputs: `outputs/<dataset>/<model>/<mode>/<budget>/`
- Stats: `stats/<dataset>/<model>/stats_budget_<budget>.pt`
- Corresponding Config: `configs/generated/<dataset>/<model>/<mode>_budget_<budget>.yaml`

`mode` is one of `fullkv`, `rkv`, `speckv`. FullKV uses `full` as the budget tag.

## Budget Configuration

- Default budget: `configs/shared/defaults.yaml`
- Sweep budgets: `configs/shared/budgets.yaml`

All scripts read budgets from these files.
