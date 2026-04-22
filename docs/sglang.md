# SGLang Integration

TriAttention supports [SGLang](https://github.com/sgl-project/sglang) as an inference backend, in parallel with the vLLM path. KV cache compression is installed via scheduler-side monkey-patches, so no upstream SGLang changes are required.

## Prerequisites

```bash
pip install -e .            # from the triattention repo root
pip install sglang[all]     # or follow SGLang's official install guide
```

## Quick Start

Launch an SGLang server with TriAttention enabled:

```bash
# Required: path to precomputed frequency statistics
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=triattention/calibration/qwen3_8b.pt

# KV budget (default: 2048)
export TRIATTN_RUNTIME_KV_BUDGET=2048

python -m triattention.sglang \
    --model-path <model_path> \
    --dtype bfloat16 \
    --context-length 32768 \
    --trust-remote-code
```

The `python -m triattention.sglang` entry point wraps SGLang's server launcher. It:

1. Sets `ENABLE_TRIATTENTION=1`.
2. Installs scheduler / worker / input-patch hooks in the main process and in every TP child.
3. Appends `--disable-radix-cache` if you did not pass it explicitly (radix cache is incompatible with KV compression).
4. Forwards all remaining CLI arguments to SGLang.

## Key Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `TRIATTN_RUNTIME_SPARSE_STATS_PATH` | Path to `.pt` file with precomputed Q/K frequency statistics | *(required)* |
| `TRIATTN_RUNTIME_KV_BUDGET` | Max KV tokens kept per sequence after compression | `2048` |
| `TRIATTN_RUNTIME_DIVIDE_LENGTH` | Trigger: compress when `kv_len >= kv_budget + divide_length` | `512` |
| `ENABLE_TRIATTENTION` | Master switch (launcher sets this automatically) | `1` |

See [Calibration Guide](calibration.md) for how to generate the stats file.

## Programmatic Integration

If you prefer to call SGLang's launcher yourself, install the hooks before starting the server:

```python
import os
os.environ["ENABLE_TRIATTENTION"] = "1"
os.environ["TRIATTN_RUNTIME_SPARSE_STATS_PATH"] = "triattention/calibration/qwen3_8b.pt"

from triattention.sglang import install_sglang_integration, install_tp_hooks

install_sglang_integration()   # main process
install_tp_hooks()             # TP child processes

# ... start SGLang server as usual, with --disable-radix-cache ...
```

## Notes

- **Radix cache must be disabled.** KV compression rewrites the KV pool in-place; radix cache's content-addressed reuse would hit stale entries.
- **Prefix caching is not supported** for the same reason.
- **Tensor parallelism** is supported; stats are automatically sharded per TP rank.
