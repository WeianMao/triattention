# DeepSeek-R1-Distill-Llama-8B TriAttention Automation

This folder mirrors the standard `experiments/scripts` helpers, but is pre-filtered for the `DeepSeek-R1-Distill-Llama-8B` model so you can recycle the same workflows while staying focused on the new checkpoints. The scripts all respect two env vars:

- `DRY_RUN=1` prints the commands/batch plan without dispatching jobs.
- `JOB_PARALLEL=N` controls how many dataset/method slots run concurrently (default `1`). Dry runs echo the batch layout so you can verify GPU utilization before kicking off work.

## Recommended Run Order

1. **Model sync** – `bash scripts/distill_llama8b/download_models.sh`  
   Downloads only `DeepSeek-R1-Distill-Llama-8B` into `experiments/models/`.
2. **FullKV traces** – `JOB_PARALLEL=2 bash scripts/distill_llama8b/run_fullkv.sh`  
   Generates baseline traces for `{aime24,aime25,math500}`; required for stats building.
3. **TriAttention stats** – `bash scripts/distill_llama8b/build_all_stats.sh`  
   Invokes `scripts/cli.py build-stats` with `--model DeepSeek-R1-Distill-Llama-8B` and all datasets. The CLI now surfaces batches in dry runs and only launches calibrations when stats files are missing.
4. **Compression sweeps** – run whichever method scripts you need:
   - `budget512/run_rkv.sh`, `budget1024/run_rkv.sh`, ..., `budget4096/run_rkv.sh` for fixed-budget R-KV runs (each wrapper forwards `--budget` to `run_rkv.sh`).
   - Matching `budgetXX/run_triattention_per_head.sh` wrappers to sweep TriAttention per-head pruning at the same budgets.
   - `run_triattention.sh` once stats exist to exercise the TriAttention pruning flow.
   - `run_rkv.sh` and `run_triattention_per_head.sh` still exist at the folder root and now accept `--budget N` for ad-hoc overrides.
   - `run_one.sh` as a convenience wrapper when testing a single dataset/method/budget (defaults: `dataset=aime24`, `model=DeepSeek-R1-Distill-Llama-8B`, `method=rkv`).

Each of the `run_*.sh` scripts fans out over the dataset list, spawns up to `JOB_PARALLEL` background jobs, and enforces failure propagation so a single bad shard doesn’t silently pass.

## Tips

- Shared config is in `triattention/configs/shared/runner_defaults.yaml`.
- Use `JOB_PARALLEL=N` on any script to speed up throughput.
- Logs for each run land in `experiments/logs/<dataset>/<model>/<mode>/<tag>/`. Check them if a batch failure is reported.
