# TriAttention A/B Demo — Setup Guide

Dual-backend demo comparing **baseline vLLM** vs **TriAttention vLLM** side-by-side
in a browser. Both backends serve the same model; the TriAttention backend applies
SpeckV KV-cache compression via a vLLM plugin.

## Prerequisites

| Item | Description |
|------|-------------|
| GPU | 2 free GPUs, each with ≥ 24 GB VRAM (e.g. RTX 4090) |
| Model | GPTQ-INT4 quantized model (default: `Qwen3-32B-INT4`) |
| Stats | SpeckV stats file (`.pt`) for the model |
| Runtime | [pixi](https://pixi.sh) installed (`~/.pixi/bin/pixi`) |
| vLLM | `vllm` pip-installed in the pixi env (≥ 0.15.0) |
| Plugin | TriAttention vLLM plugin installed (`pip install -e TriAttention_vLLM/`) |

## Quick Start

```bash
# 1. Pick two free GPUs (check with nvidia-smi)
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# 2. Start (uses GPU 3 & 4 by default)
cd /path/to/speckv
bash demo/vllm/start_remote_demo.sh

# 3. Open browser (from your local machine via SSH tunnel)
ssh -L 8010:127.0.0.1:8010 <remote-host>
# Then visit http://127.0.0.1:8010
```

## Stop

```bash
bash demo/vllm/stop_remote_demo.sh
```

## Configuration

All settings are controlled via environment variables with sensible defaults.
Override by prefixing the start command:

```bash
BASELINE_CUDA=0 TRIATTENTION_CUDA=1 GPU_MEMORY_UTILIZATION=0.95 \
  bash demo/vllm/start_remote_demo.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `/home/wayne/linxi/Qwen3-32B-INT4` | Path to the model directory |
| `STATS_PATH` | `demo/openclaw-demo/stats/qwen3_32b_int4_speckv_stats.pt` | SpeckV stats file |
| `BASELINE_CUDA` | `3` | GPU index for baseline backend |
| `TRIATTENTION_CUDA` | `4` | GPU index for TriAttention backend |
| `BASELINE_PORT` | `8001` | Port for baseline vLLM server |
| `TRIATTENTION_PORT` | `8002` | Port for TriAttention vLLM server |
| `DEMO_HOST` | `127.0.0.1` | Host for the FastAPI gateway |
| `DEMO_PORT` | `8010` | Port for the FastAPI gateway |
| `GPU_MEMORY_UTILIZATION` | `0.75` | vLLM GPU memory fraction (increase to 0.95 if OOM on KV cache) |
| `MAX_MODEL_LEN` | `16384` | Maximum sequence length |
| `MAX_NUM_SEQS` | `32` | Maximum concurrent sequences |
| `KV_BUDGET` | `2048` | KV cache token budget for TriAttention compression |
| `DTYPE` | `bfloat16` | Model dtype |
| `ENFORCE_EAGER` | `true` | Disable CUDA graphs (saves memory on 4090) |

## Architecture

```
Browser  ──SSH tunnel──>  FastAPI Gateway (:8010)
                              ├── baseline vLLM    (:8001, GPU X)
                              └── triattention vLLM (:8002, GPU Y)
```

The gateway proxies chat completion requests to both backends simultaneously.
A server-sent events (SSE) live feed pushes per-token updates to the browser,
which renders both outputs side-by-side with real-time metrics (TTFT, TPS, etc.).

## Troubleshooting

### OOM: "No available memory for the cache blocks"

The model uses ~18 GB on a 24 GB GPU. With the default `GPU_MEMORY_UTILIZATION=0.75`
(18.4 GB cap), there is almost no room for KV cache.

Fix: increase utilization and/or reduce max sequence length:

```bash
GPU_MEMORY_UTILIZATION=0.95 MAX_MODEL_LEN=14336 bash demo/vllm/start_remote_demo.sh
```

### Backends started but no model ID returned

Both vLLM servers need ~10–20 seconds to load the model. Wait for
`Application startup complete` in the logs before sending requests:

```bash
tail -f demo/vllm/logs/baseline.log
tail -f demo/vllm/logs/triattention.log
```

### Port already in use

Stop the existing demo first:

```bash
bash demo/vllm/stop_remote_demo.sh
```

## Logs

All logs are written to `demo/vllm/logs/`:

- `baseline.log` — baseline vLLM server
- `triattention.log` — TriAttention vLLM server
- `demo.log` — FastAPI gateway
