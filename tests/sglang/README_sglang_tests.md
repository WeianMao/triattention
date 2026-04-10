# sglang Integration Tests for TriAttention

End-to-end tests for the TriAttention sglang backend.

## Prerequisites

1. **Conda environment**: Create the `trisglang` environment:
   ```bash
   bash scripts/setup_trisglang_env.sh
   conda activate trisglang
   ```

2. **GPU**: At least one NVIDIA GPU with enough VRAM for the target model
   (e.g., ~24 GB for Qwen3-32B-INT4 with INT4 quantization on TP=2).

3. **Model**: A local or HuggingFace model path (e.g. `Qwen/Qwen3-32B-INT4`).

4. **Stats file** (optional but recommended): The sparse attention statistics
   `.pt` file for the model. Without it, scoring falls back to uniform weights.

## Test Overview

| Test | File | GPU Required | Description |
|------|------|:---:|-------------|
| Smoke test | `test_sglang_smoke.py` | Yes | Start server, send one request, validate response |
| Multiturn replay | `test_sglang_multiturn_replay.py` | Yes | Replay multi-turn tool-calling conversation |

## Running the Smoke Test

```bash
conda activate trisglang

# Full run (starts and stops the server automatically):
python tests/sglang/test_sglang_smoke.py \
    --model Qwen/Qwen3-32B-INT4 \
    --port 8899 \
    --tp 2 \
    --kv-budget 512 \
    --divide-length 128 \
    --stats-path /path/to/qwen3_32b_int4_stats.pt

# If you already have a server running:
python tests/sglang/test_sglang_smoke.py \
    --model Qwen/Qwen3-32B-INT4 \
    --port 8899 \
    --skip-server
```

### Expected Output

```
[OK] Server is ready
[smoke] Sending request to http://localhost:8899/v1/chat/completions
[OK] Response length: 1234 chars
[OK] Preview: ...
[OK] finish_reason: stop
[OK] Smoke test PASSED
```

## Running the Multiturn Replay Test

### Step 1: Get the Fixtures

The multi-turn fixtures live on the `test/multiturn-replay` branch. Extract
them into the working tree:

```bash
cd dc1-release/

# Option A: checkout just the fixtures directory
git checkout origin/test/multiturn-replay -- tests/multiturn_replay/fixtures/

# Option B: point --fixture-dir to a temporary extraction
git show origin/test/multiturn-replay:tests/multiturn_replay/fixtures/turn_1.json > /tmp/fixtures/turn_1.json
git show origin/test/multiturn-replay:tests/multiturn_replay/fixtures/turn_2.json > /tmp/fixtures/turn_2.json
git show origin/test/multiturn-replay:tests/multiturn_replay/fixtures/turn_3.json > /tmp/fixtures/turn_3.json
```

### Step 2: Run the Test

```bash
conda activate trisglang

# Full run (starts server, replays all turns, stops server):
python tests/sglang/test_sglang_multiturn_replay.py \
    --model Qwen/Qwen3-32B-INT4 \
    --port 8899 \
    --tp 2 \
    --kv-budget 512 \
    --divide-length 128 \
    --stats-path /path/to/qwen3_32b_int4_stats.pt

# With custom fixture directory:
python tests/sglang/test_sglang_multiturn_replay.py \
    --model Qwen/Qwen3-32B-INT4 \
    --port 8899 \
    --fixture-dir /tmp/fixtures \
    --skip-server

# Against an already-running server:
python tests/sglang/test_sglang_multiturn_replay.py \
    --model Qwen/Qwen3-32B-INT4 \
    --port 8899 \
    --skip-server
```

### Expected Output

```
[replay] Loaded 3 fixtures from tests/multiturn_replay/fixtures
=== Turn 1/3: turn_1.json (2 messages) ===
  status: OK (3.2s)
  finish_reason: tool_calls
  tool_calls: 1
    - read_document({"path": "..."})
=== Turn 2/3: turn_2.json (4 messages) ===
  status: OK (4.1s)
  ...
=== Turn 3/3: turn_3.json (6 messages) ===
  status: OK (5.3s)
  ...
=== Results: 3/3 turns passed ===
[OK] Multiturn replay test PASSED
```

## Running the Original vLLM Multiturn Test

The original vLLM test is on branch `test/multiturn-replay` and is not
modified by the sglang integration. To run it against a vLLM server:

```bash
git checkout origin/test/multiturn-replay -- tests/multiturn_replay/

python tests/multiturn_replay/test_multiturn_replay.py \
    --backend-url http://localhost:8000/v1/chat/completions \
    --model <model_name>
```

## CLI Arguments Reference

All test scripts accept these common arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | Model name or path |
| `--port` | 8899 | sglang server port |
| `--tp` | 1 | Tensor parallelism degree |
| `--kv-budget` | 512 | KV cache budget for compression |
| `--divide-length` | 128 | Token interval between compression triggers |
| `--stats-path` | None | Path to sparse attention stats `.pt` file |
| `--timeout` | 300 | Server startup timeout (seconds) |
| `--skip-server` | False | Skip server startup, use existing server |

## Troubleshooting

- **Server fails to start**: Check that `CUDA_HOME` is set and sglang is
  properly installed. Try `python -m sglang.launch_server --model <model>
  --port 8899` without TriAttention first.

- **Fixtures not found**: Make sure to extract fixtures from the
  `test/multiturn-replay` branch (see Step 1 above).

- **OOM errors**: Reduce `--kv-budget` or increase `--tp`.

- **No compression observed**: The prompt may be too short. Ensure
  `--divide-length` is smaller than the expected generation length.
