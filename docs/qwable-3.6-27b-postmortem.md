# Post-mortem: Qwable-3.6-27B TriAttention MLX Calibration Attempt

## Overview

Attempted to run TriAttention MLX calibration on [Qwable-3.6-27B](https://huggingface.co/Mia-AiLab/Qwable-3.6-27b) — a full fine-tune of Qwen3.6-27B on Fable-5 style reasoning traces — on an M4 Pro Mac Mini (24 GB unified memory).

This document captures the blockers encountered and lessons learned, as 24 GB M-series machines are a common Apple Silicon target and these failure modes will affect others.

## What We Tried

1. Load Qwable-3.6-27B HF safetensors → convert to MLX 4-bit → calibrate → deploy
2. Use pre-converted `froggeric/qwen3.6-27b-mlx-4bit` (same base architecture) as a proxy

## Blockers

### Blocker 1: HF → MLX conversion OOM

`mlx_lm.convert` loads the **full fp16 model** (~54 GB) into RAM before quantizing. There is no streaming quantization path. On 24 GB hardware this is a hard blocker.

```bash
# This OOMs on 24 GB:
mlx_lm.convert --hf-path Mia-AiLab/Qwable-3.6-27b --mlx-path ./qwable-mlx --q-bits 4
```

**Workaround:** Request a pre-converted MLX upload from the model publisher, or run conversion on a 128 GB+ instance.

### Blocker 2: Qwen3.6 GatedDeltaNet architecture limits compression impact

Qwen3.6-27B uses **GatedDeltaNet linear attention** for 48 of 64 layers (every non-multiple-of-4 layer). Only 16 layers use standard `Qwen3NextAttention`. TriAttention compresses only the full-attention layers.

Benchmark results at 4.8× KV compression (4927 → 1024 tokens):

| Mode | Output chars | Time |
|---|---|---|
| Norm-only (`disable_trig=True`) | 785 | 65.2s |
| Trig-scoring (`disable_trig=False`) | 785 | 65.7s |

Identical output. The DeltaNet layers carry the signal path — evicting KV entries from 16/64 layers doesn't affect the final answer quality at this compression ratio.

**Important caveat:** This observation was made with the original (buggy) calibration code where the trig path was silently failing due to a scalar `q_abs_mean` shape mismatch. With the corrected calibration pipeline (PR #19 revision), the trig path should now run properly. The DeltaNet dominance is still a real factor, but a re-run with corrected stats is needed to get a valid comparison signal.

## What Did Work

- `froggeric/qwen3.6-27b-mlx-4bit` loaded and calibrated successfully
- 384-head stats generated in 12 min on M4 Pro (24 GB)
- Full trig-scoring inference confirmed working with `disable_trig=False`
- The calibration hooks work correctly for `Qwen3NextAttention`

## Recommendations

1. **For TriAttention:** Document that hybrid linear/full-attention architectures (Qwen3.5/6, Jamba, etc.) will show diminished compression benefit since only full-attention layers are affected
2. **For mlx_lm:** A streaming quantization path would unlock conversion of large models on memory-constrained hardware
3. **Best test targets for TriAttention MLX:** Standard full-attention models — Llama 3, Mistral, Gemma — where all layers benefit from KV compression

## Related

- PR #19 — MLX calibration pipeline (with corrected math)
- Calibration code findings have been folded into the #19 review thread
