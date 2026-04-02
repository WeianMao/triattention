# DFS State Query

This folder collects the DFS state query dataset, evaluation scripts, and related docs.

## Layout
- datasets/dfs_state_query_small.json: primary dataset to keep and extend
- datasets/legacy/: older full/test sets kept for reference
- scripts/: evaluation, generation, verification, and quick test scripts
- docs/: dataset and evaluation notes
- analysis/: step distribution analyses

## Common commands (run from AQA-Bench root)
- Generate/update the small dataset:
  - python dfs_state_query/scripts/generate_dfs_state_dataset.py --output dfs_state_query/datasets/dfs_state_query_small.json
- Verify a dataset:
  - python dfs_state_query/scripts/verify_dfs_dataset.py dfs_state_query/datasets/dfs_state_query_small.json --verify-all
  - python dfs_state_query/scripts/verify_step_uniformity.py --dataset dfs_state_query/datasets/dfs_state_query_small.json
- Evaluate:
  - python dfs_state_query/scripts/eval_dfs_state_query.py --dataset dfs_state_query/datasets/dfs_state_query_small.json --model-path <model> --output dfs_state_query/analysis/dfs_eval.json
