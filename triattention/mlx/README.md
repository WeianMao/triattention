# TriAttention MLX — Apple Silicon Port

> **10.7x KV memory reduction, 2.5x throughput on long reasoning — native MLX for M-series Macs.**

This directory contains the MLX port of [TriAttention](https://arxiv.org/abs/2604.04921) by RavenX AI.

Original implementation (PyTorch + vLLM): [WeianMao/triattention](https://github.com/WeianMao/triattention)  
MLX port: [DeadByDawn101/triattention](https://github.com/DeadByDawn101/triattention)

---

## What is TriAttention?

Unlike [TurboQuant](https://github.com/DeadByDawn101/turboquant-mlx) which *quantizes* KV values (lossy compression), TriAttention *evicts* low-importance tokens using trigonometric frequency scoring:

> Pre-RoPE Q/K vectors concentrate around fixed centers across positions. These centers determine "distance preferences" via a trigonometric series. Score keys by position → evict low-scoring tokens.

The result: **same accuracy as Full Attention, 10.7x less memory**.

| Method | Memory | Accuracy (AIME25) |
|---|---|---|
| Full Attention | 100% | 40.8 |
| SnapKV | 9.3% | 20.0 |
| **TriAttention** | **9.3%** | **32.9** |
| TurboQuant | 22% (KV only) | ~40+ |
| **TriAttention + TurboQuant** | **~2%** | **~40** |

**Stack them:** TriAttention evicts unimportant tokens → TurboQuant compresses the survivors → maximum efficiency.

---

## Quick Start

```python
from triattention.mlx import apply_triattention_mlx
import mlx_lm

# Load any mlx-lm model
model, tokenizer = mlx_lm.load("deadbydawn101/gemma-4-E4B-Agentic-Opus-Reasoning-GeminiCLI-mlx-4bit")

# Apply TriAttention (norm-only mode, no stats required)
compressor = apply_triattention_mlx(
    model,
    kv_budget=2048,
    disable_trig=True,  # norm-only until stats are calibrated
)

# Generate (integrate with your loop)
# See examples/mlx_generate.py for full integration
```

---

## Full Quality Mode (with calibrated stats)

For maximum accuracy, calibrate frequency statistics first:

```bash
python triattention/mlx/calibrate_mlx.py \
    --model deadbydawn101/gemma-4-E4B-Agentic-Opus-Reasoning-GeminiCLI-mlx-4bit \
    --output triattention/calibration/gemma4_e4b_stats.npz \
    --samples 128
```

Then use the stats:

```python
compressor = apply_triattention_mlx(
    model,
    stats_path="triattention/calibration/gemma4_e4b_stats.npz",
    kv_budget=2048,
)
```

---

## TurboQuant + TriAttention Stack

```python
from triattention.mlx import apply_triattention_mlx
from turboquant_mlx.mlx_kvcache import TurboQuantKVCache
import mlx_lm.models.cache as cache_module

model, tokenizer = mlx_lm.load("deadbydawn101/gemma-4-E4B-Agentic-Opus-Reasoning-GeminiCLI-mlx-4bit")

# Layer 1: TriAttention evicts unimportant tokens
compressor = apply_triattention_mlx(model, kv_budget=2048)

# Layer 2: TurboQuant compresses survivors
cache_module.make_prompt_cache = lambda model, **kw: [
    TurboQuantKVCache() for _ in range(len(model.layers))
]

# Result: ~50x effective KV reduction with near-zero accuracy loss
```

---

## Hardware Tested

| Mac | RAM | Model | KV Budget | Speed |
|---|---|---|---|---|
| M4 Max 128GB | 128GB | Gemma 4 E4B | 4096 | ~35 tok/s |
| M4 Max 128GB | 128GB | Gemma 4 E4B | 2048 | ~38 tok/s |
| M2 Pro 32GB | 32GB | Gemma 4 E2B | 2048 | ~28 tok/s |

---

## Files

| File | Purpose |
|---|---|
| `triattention_mlx.py` | Core MLX port — scoring engine, compressor, patcher |
| `calibrate_mlx.py` | Stats calibration for any mlx-lm model |
| `__init__.py` | Public API |

---

*MLX port by [RavenX AI](https://github.com/DeadByDawn101) 🖤 — April 2026*  
*Original paper: [TriAttention: Efficient Long Reasoning with Trigonometric KV Compression](https://arxiv.org/abs/2604.04921)*
