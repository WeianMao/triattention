# Prefill Chunk Demo Runbook

Updated: 2026-02-27

## Goal

Show long-prefill behavior contrast:
1. FullKV baseline (more likely to hit OOM under long prefill pressure).
2. TriAttention + pre-chunk (prefill auto-chunk enabled, compression enabled).

## Configs

1. FullKV baseline:
   - `TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24_fullkv_qwen3_demo.yaml`
2. TriAttention pre-chunk:
   - `TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24_hf_perhead_anchor_qwen3_prefill_chunk.yaml`

## Commands

```bash
conda run -n trivllm python TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py \
  --config TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24_fullkv_qwen3_demo.yaml
```

```bash
conda run -n trivllm python TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py \
  --config TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24_hf_perhead_anchor_qwen3_prefill_chunk.yaml
```

## Notes

1. Current pre-chunk integration maps chunk trigger and chunk size to one vLLM knob, so use equal values.
2. Default demo setting is `prefill_chunk_threshold = prefill_chunk_size = 2048`.
3. Compression event payload now includes `scheduled_tokens`, `estimated_cache_len`, and `prefill_len` for observability.

## Validation Snapshot (2026-02-27)

1. Qwen3 smoke:
   - output: `evaluation/outputs/qwen3_smoke_run/shard00/run000.jsonl`
   - status: success, no OOM.
2. Long-prefill smoke (pre-chunk enabled):
   - input file: `evaluation/outputs/qwen3_longprefill_dataset.jsonl`
   - output: `evaluation/outputs/qwen3_longprefill_run2/shard00/run000.jsonl`
   - observed: `prefill_tokens=3058`, `total_tokens=4096`, run completed without OOM.
3. Ultra-long pair check (same sample, 12058-token prefill):
   - baseline output: `evaluation/outputs/qwen3_verylong_fullkv_run/shard00/run000.jsonl`
   - triattention output: `evaluation/outputs/qwen3_verylong_tri_run/shard00/run000.jsonl`
   - observed: both succeeded on this machine (no baseline OOM at 12k prefill),
     so OOM-contrast demo needs stronger memory pressure setup.
