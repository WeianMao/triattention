# Demo Startup Guide (Current Branch)

This guide covers how to start the demo gateway in this branch with a vLLM backend, including both `pixi run` and non-pixi workflows.

The demo service is:

- Gateway: `demo/vllm/server.py` (OpenAI-compatible API + web UI)
- Backend: `vllm serve` (with optional TriAttention)

## 1. Common Variables

Set these once before running commands:

```bash
export REPO_ROOT=/home/wayne/linxi/speckv
export MODEL_PATH=/home/wayne/linxi/Qwen3-32B-INT4
export STATS_PATH=/home/wayne/linxi/speckv/R-KV/outputs/qwen3-8b_aime24_sample8_chat/stats/qwen3_32b_int4_speckv_stats.pt
export CUDA_VISIBLE_DEVICES=5
export BACKEND_HOST=127.0.0.1
export BACKEND_PORT=8002
export DEMO_HOST=127.0.0.1
export DEMO_PORT=8010
```

Prepare runtime dirs:

```bash
mkdir -p "$REPO_ROOT/demo/vllm/logs" "$REPO_ROOT/demo/vllm/run"
```

## 2. Start With pixi (Recommended)

Start vLLM backend (TriAttention enabled):

```bash
nohup bash -lc "
  cd \"$REPO_ROOT\"
  exec -a PD-L1_binder env \
    CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" \
    VLLM_PLUGINS=triattention \
    TRIATTENTION_STATS_PATH=\"$STATS_PATH\" \
    TRIATTENTION_KV_BUDGET=2048 \
    TRIATTENTION_DIVIDE_LENGTH=128 \
    TRIATTENTION_WINDOW_SIZE=128 \
    TRIATTENTION_PRUNING_MODE=per_head \
    TRIATTENTION_QUIET=0 \
    TRIATTENTION_LOG_TRIGGER=1 \
    TRIATTENTION_LOG_DECISIONS=0 \
    pixi run vllm serve \"$MODEL_PATH\" \
      --host \"$BACKEND_HOST\" \
      --port \"$BACKEND_PORT\" \
      --dtype float16 \
      --quantization gptq_marlin \
      --max-model-len 11000 \
      --gpu-memory-utilization 0.90 \
      --tensor-parallel-size 1 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes \
      --trust-remote-code \
      --enforce-eager \
      --max-num-seqs 1 \
      --attention-backend CUSTOM
" > "$REPO_ROOT/demo/vllm/logs/triattention_backend.log" 2>&1 &
echo $! > "$REPO_ROOT/demo/vllm/run/triattention.pid"
```

Start demo gateway:

```bash
nohup bash -lc "
  cd \"$REPO_ROOT\"
  exec -a PD-L1_binder env \
    VLLM_BACKEND_URL=\"http://$BACKEND_HOST:$BACKEND_PORT\" \
    DEMO_HOST=\"$DEMO_HOST\" \
    DEMO_PORT=\"$DEMO_PORT\" \
    pixi run python -m uvicorn demo.vllm.server:app --host \"$DEMO_HOST\" --port \"$DEMO_PORT\"
" > "$REPO_ROOT/demo/vllm/logs/demo.log" 2>&1 &
echo $! > "$REPO_ROOT/demo/vllm/run/demo.pid"
```

## 3. Start Without pixi

Use this when you want to run from conda/venv directly.

1) Activate your runtime environment and ensure dependencies are installed:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_clean
python -m pip install -e "$REPO_ROOT/TriAttention_vLLM"
```

2) Start backend (same TriAttention settings, no `pixi run`):

```bash
nohup bash -lc "
  cd \"$REPO_ROOT\"
  exec -a PD-L1_binder env \
    CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" \
    VLLM_PLUGINS=triattention \
    TRIATTENTION_STATS_PATH=\"$STATS_PATH\" \
    TRIATTENTION_KV_BUDGET=2048 \
    TRIATTENTION_DIVIDE_LENGTH=128 \
    TRIATTENTION_WINDOW_SIZE=128 \
    TRIATTENTION_PRUNING_MODE=per_head \
    TRIATTENTION_QUIET=0 \
    TRIATTENTION_LOG_TRIGGER=1 \
    TRIATTENTION_LOG_DECISIONS=0 \
    vllm serve \"$MODEL_PATH\" \
      --host \"$BACKEND_HOST\" \
      --port \"$BACKEND_PORT\" \
      --dtype float16 \
      --quantization gptq_marlin \
      --max-model-len 11000 \
      --gpu-memory-utilization 0.90 \
      --tensor-parallel-size 1 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes \
      --trust-remote-code \
      --enforce-eager \
      --max-num-seqs 1 \
      --attention-backend CUSTOM
" > "$REPO_ROOT/demo/vllm/logs/triattention_backend.log" 2>&1 &
echo $! > "$REPO_ROOT/demo/vllm/run/triattention.pid"
```

3) Start gateway (no `pixi run`):

```bash
nohup bash -lc "
  cd \"$REPO_ROOT\"
  exec -a PD-L1_binder env \
    VLLM_BACKEND_URL=\"http://$BACKEND_HOST:$BACKEND_PORT\" \
    DEMO_HOST=\"$DEMO_HOST\" \
    DEMO_PORT=\"$DEMO_PORT\" \
    python -m uvicorn demo.vllm.server:app --host \"$DEMO_HOST\" --port \"$DEMO_PORT\"
" > "$REPO_ROOT/demo/vllm/logs/demo.log" 2>&1 &
echo $! > "$REPO_ROOT/demo/vllm/run/demo.pid"
```

## 4. Health Checks

```bash
curl -sS "http://$DEMO_HOST:$DEMO_PORT/healthz"
curl -sS "http://$DEMO_HOST:$DEMO_PORT/v1/models"
```

Open UI:

```text
http://127.0.0.1:8010
```

## 5. Verify TriAttention Is Actually Used

Check backend log:

```bash
grep -n "V1 Backend registered successfully as CUSTOM" "$REPO_ROOT/demo/vllm/logs/triattention_backend.log"
grep -n "Using AttentionBackendEnum.CUSTOM backend" "$REPO_ROOT/demo/vllm/logs/triattention_backend.log"
grep -n "V1 Impl initialized" "$REPO_ROOT/demo/vllm/logs/triattention_backend.log"
grep -n "Loaded config from environment: stats_path=" "$REPO_ROOT/demo/vllm/logs/triattention_backend.log"
```

Note: short prompts may not hit compression threshold, so `[TRIGGER]` lines may be absent even when backend integration is active.

## 6. Stop Demo

```bash
lsof -tiTCP:$DEMO_PORT -sTCP:LISTEN | xargs -r kill
lsof -tiTCP:$BACKEND_PORT -sTCP:LISTEN | xargs -r kill
sleep 1
lsof -tiTCP:$DEMO_PORT -sTCP:LISTEN | xargs -r kill -9
lsof -tiTCP:$BACKEND_PORT -sTCP:LISTEN | xargs -r kill -9
```

Or stop by pid files:

```bash
kill "$(cat "$REPO_ROOT/demo/vllm/run/demo.pid")" 2>/dev/null || true
kill "$(cat "$REPO_ROOT/demo/vllm/run/triattention.pid")" 2>/dev/null || true
```
