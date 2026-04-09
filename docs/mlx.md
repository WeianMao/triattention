# MLX Support (Experimental)

TriAttention supports Apple Silicon Macs (M1/M2/M3/M4) via the [MLX](https://github.com/ml-explore/mlx) framework and [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm).

## Quick Start

```python
from triattention.mlx import apply_triattention_mlx
import mlx_lm

model, tokenizer = mlx_lm.load("Qwen/Qwen3-8B")

compressor = apply_triattention_mlx(
    model,
    stats_path="triattention/calibration/qwen3_8b.pt",
    kv_budget=2048,
)
```

## Hardware Benchmarks

(Evaluated by [@DeadByDawn101](https://github.com/DeadByDawn101))

| Hardware | Model | KV Budget | Throughput |
|----------|-------|-----------|------------|
| M4 Max (128GB) | Gemma 4 E4B | 4096 | ~35 tok/s |
| M2 Pro (32GB) | Gemma 4 E2B | 2048 | ~28 tok/s |
