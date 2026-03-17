# SpeckV KV Compression Test Plan

Goal: isolate why the SpeckV refactor under `R-KV/` underperforms compared to the original simulation (`weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py`) and the LazyEviction run. The plan is split into independent workstreams so multiple people can execute in parallel.

## A. Offline stats and sampling parity
- Objective: confirm the stats file consumed by R-KV matches the original head sampling, RoPE metadata, and frequency means used by the simulation/LazyEviction runs.
- Scope/targets: `R-KV/weian_development/rkv_sparse_round_calibrate.py`, `R-KV/weian_development/speckv/stats_utils.py`, `R-KV/weian_development/speckv/round_pruning_utils.py`, stats artifact (e.g., `R-KV/outputs/.../stats/*.pt`), sampled head JSON.
- Actions:
  1) Regenerate stats from the same trace used in the successful LazyEviction run (same model path, dataset slice, sampled heads) using the R-KV calibrate script; regenerate once more using the original `attention_pruning_case_study_hybrid_rounds_xtrace.py` to create a reference.
  2) Diff metadata: rope_style/type, head_dim, sampled_heads ordering, offset grid length, dtype. Fail if any mismatch.
  3) Compute per-head cosine similarity between q_mean_complex/q_abs_mean from both stats files; flag any head with similarity < 0.999.
  4) Verify head sampling reproducibility: rerun with the same seed and check the JSON is identical (size, ordering).
- Signals to capture: summary table of mismatched heads/fields; confirmation that stats capture uses the Llama RoPE branch (not Qwen) when running the Llama-8B model.

## B. Scoring parity on synthetic caches (no HF generate)
- Objective: ensure the round-based scoring and selection logic in `SparseRoundPruner` matches the reference `simulate_round_pruning` behavior.
- Scope/targets: `SparseRoundPruner._compute_head_scores`, `_select_keep_indices`, `score_keys_for_round`, kv-group mapping for multi-query heads.
- Actions:
  1) Build a CPU-only harness that seeds identical q/k tensors for a tiny cache (e.g., seq_len=64, head_dim from stats, kv_budget=16, round_window=8) and runs both paths: (a) reference simulate_round_pruning using the captured stats; (b) `SparseRoundPruner` slicing the same tensors with `cache_positions` preset to absolute positions.
  2) Cover cases: round_window boundary (exact multiple, off-by-one), head_limit set/unset, kv_grouped mapping (simulate num_key_value_heads < num_attention_heads), and noisy tie-breaking with seed set.
  3) Compare keep indices after each round and per-head score ranks; fail on any divergence.
- Signals to capture: per-round selected indices, per-head top-k indices, cache_positions evolution; a minimal repro script checked into `debug_docs` outputs (no code changes needed to library).

## C. RoPE alignment and inversion correctness
- Objective: confirm RoPE parameters and inversion are identical between the pruner and the live model for Llama/Qwen paths.
- Scope/targets: `verify_rotary_alignment`, `compute_frequency_scaling`, `invert_rope`, `determine_rope_style`; live model `self_attn.rotary_emb`.
- Actions:
  1) Instrument a short script that loads the Llama-8B model and constructs the pruner rotary; dump `inv_freq`, `rope_type/style`, `frequency_scale` norms for both.
  2) For a random tensor batch, apply model rotary then pruner `invert_rope`, measure reconstruction error (should be ~1e-5 relative). Run both half-pairing and interleaved branches explicitly.
  3) Validate cache_position vs position_ids: confirm pruner absolute positions match the RoPE positions used in forward (see Workstream D for logging hookup).
- Signals to capture: max diff of inv_freq arrays, reconstruction error stats, explicit confirmation of rope_style branch hit.

## D. HF generate integration: positions, masks, and cache reset
- Objective: verify the patched forward in `rkv_speckv_generate.py` drives correct absolute positions, cache_positions, attention_mask handling, and round resets.
- Scope/targets: patched `model.forward`, `SparseRoundPruner` state machine (`absolute_position`, `tokens_in_round`, `cache_positions`, `prefix_length`), cache_position overrides, attention_mask overrides.
- Actions:
  1) Run a 1-question, short prompt (e.g., 32 prefill + 64 decode) with `method=speckv` and enable debug logging that records per-step: incoming cache length, absolute_position before/after, cache_position tensor passed to forward, position_ids passed, tokens pruned each step, tokens_in_round, round transitions.
  2) Repeat with LazyEviction manual loop using the same prompt and seeds; log the equivalent fields.
  3) Check for divergences: cache_position non-contiguous, absolute_position desync after pruning, stale attention_mask being used, or prefix_length being zeroed accidentally after reset.
- Signals to capture: per-step table (step idx, cache_len_before/after, abs_pos, round_id, pruned_count, cache_position min/max); screenshots/logs preserved under `debug_docs`.

## E. Budget and prefix retention checks
- Objective: ensure prefill tokens stay pinned and the dynamic budget is actually enforced at run time.
- Scope/targets: `_dynamic_cache_size`, `enforce_max_limit`, `ensure_capacity`, `_prune_to_size` dynamic_only path, prefix_length handling.
- Actions:
  1) Construct two scenarios: (a) long prompt > kv_budget (should not prune prefix); (b) prompt < kv_budget but multiple rounds of decoding push over budget. Log cache lengths and pruned indices after each enforce/ensure call.
  2) Verify keep_capacity logic: expect max_keys - round_window dynamic capacity and monotonic prefix_length.
  3) Cross-check with on-disk traces: after a short real decode run, load saved cache tensors (or wrap forward) to verify that prefix slices remain intact and only dynamic tail is truncated.
- Signals to capture: cache length vs expected formula, evidence that prefix indices are untouched, any instance where _dynamic_cache_size exceeds budget without pruning.

## F. End-to-end parity probe vs LazyEviction
- Objective: surface whether SpeckV logic is simply not engaging or diverging under full generation compared to the known-good LazyEviction path.
- Scope/targets: `R-KV/weian_development/rkv_sharded_eval.py` invocation with `method=speckv`, LazyEviction `run_sparse_prefill_keep_sharded_eval.sh` with equivalent kv_budget/window/offset/seed/model.
- Actions:
  1) Run a minimal shard (e.g., 2 AIME questions, num_samples=2, kv_budget=2048, round_window=128) on both stacks with verbose SpeckV logging turned on (logs from Workstreams D/E).
  2) Collect per-step prune counts and final cache sizes; verify SpeckV prunes similar volume to LazyEviction. If SpeckV prunes near-zero or explodes cache length, flag immediately.
  3) Compare generated answers qualitatively and note any early stop / EOS timing differences that could stem from cache misalignment.
- Signals to capture: side-by-side log snippets showing pruning activity (or lack thereof), final cache length distributions, generation length differences.

## G. Config and sampling consistency audit
- Objective: rule out configuration drift that could mask the algorithm (e.g., different temps, kv budgets, sampling params, stats paths).
- Scope/targets: `R-KV/weian_script/configs/sample*_speckv*.yaml`, `LazyEviction/weian_script/configs/`, runner args in dispatch scripts.
- Actions:
  1) Build a checklist comparing R-KV vs LazyEviction: kv_budget, round_window, offset_max_length, score_aggregation, head_limit, temperature/top_p/top_k, seed, use_chat_template, fp32_topk.
  2) Verify the stats file path and model_path resolve to the same assets; confirm sampled_heads file matches Workstream A.
  3) Ensure cache reset behavior (`reset_cache_each_batch`) is aligned between runs.
- Signals to capture: diff table of config fields; any mismatched default noted explicitly.

Deliverables: for each workstream, produce a short log (commands used, key observations) under `R-KV/debug_docs/` so findings can be cross-correlated without rerunning everything.
