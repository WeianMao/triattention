# Demo Handoff: Startup and Payload Replay

This note documents the current demo startup flow and the captured-payload replay flow.

## Scope

- Start the full demo stack:
  - baseline vLLM backend
  - triattention vLLM backend
  - demo gateway on port `8010`
- Replay a captured OpenClaw `/v1/chat/completions` payload against the demo.
- Inspect logs and live streaming output when debugging tool-calling or streaming issues.

## Key Files

- Startup script: `demo/vllm/start_remote_demo.sh`
- Stop script: `demo/vllm/stop_remote_demo.sh`
- Gateway: `demo/vllm/server.py`
- Replay script: `weian_development/demo_debug/replay_openclaw_real_payload.py`
- Captured payload fixture: `weian_development/demo_debug/fixtures/openclaw_real_payload_20260313_2252.json`
- Runtime logs:
  - `demo/vllm/logs/baseline.log`
  - `demo/vllm/logs/triattention.log`
  - `demo/vllm/logs/demo.log`

## Current Demo Defaults

The current startup script defaults are:

- gateway port: `8010`
- baseline backend: `8001`
- triattention backend: `8002`
- `MAX_MODEL_LEN=16384` in the script by default
- `KV_BUDGET=14336`
- `GPU_MEMORY_UTILIZATION=0.75` in the script by default

Important operational notes:

- In the currently working deployment, model length may be overridden to `32768` at launch time to bypass client-side length checks.
- The actual physical KV capacity is still much smaller than the advertised model length, so long requests can still fail or degrade.
- The gateway now emits extra live events for debugging:
  - `raw_chat_delta`
  - `raw_completion_delta`

## Start the Demo

Run from the repo root:

```bash
bash demo/vllm/start_remote_demo.sh
```

If you need the currently used 32K advertised model length:

```bash
MAX_MODEL_LEN=32768 GPU_MEMORY_UTILIZATION=0.95 bash demo/vllm/start_remote_demo.sh
```

If you need to override ports or budget:

```bash
DEMO_PORT=8010 BASELINE_PORT=8001 TRIATTENTION_PORT=8002 KV_BUDGET=14336 bash demo/vllm/start_remote_demo.sh
```

Notes:

- The script launches all three processes and writes pid files under `demo/vllm/run/`.
- Logs go to `demo/vllm/logs/`.
- Long-running processes are started with visible process name `PD-L1_binder`.

## SSH Port Forwarding

If the demo is running on `zju_230`, create local forwards before opening the UI,
calling `/healthz`, or running the replay script from your laptop:

```bash
mkdir -p /tmp/codex-ssh
ssh -fN -M -S /tmp/codex-ssh/speckv-zju_230-demo.sock \
  -o ExitOnForwardFailure=yes \
  -L 8001:127.0.0.1:8001 \
  -L 8002:127.0.0.1:8002 \
  -L 8010:127.0.0.1:8010 \
  zju_230
```

This exposes the remote services locally at:

- `http://127.0.0.1:8010` for the demo gateway
- `http://127.0.0.1:8001` for the baseline backend
- `http://127.0.0.1:8002` for the triattention backend

To confirm the tunnel is alive:

```bash
ssh -S /tmp/codex-ssh/speckv-zju_230-demo.sock -O check zju_230
curl -s http://127.0.0.1:8010/healthz
```

To close the tunnel cleanly:

```bash
ssh -S /tmp/codex-ssh/speckv-zju_230-demo.sock -O exit zju_230
```

## Stop the Demo

Run from the repo root:

```bash
bash demo/vllm/stop_remote_demo.sh
```

This stops:

- gateway
- triattention backend
- baseline backend

## Health Check

After startup:

```bash
curl -s http://127.0.0.1:8010/healthz
```

Healthy output should report:

- `gateway.ok=true`
- `baseline.ok=true`
- `triattention.ok=true`

## Replay the Captured OpenClaw Payload

The fixture is:

```text
weian_development/demo_debug/fixtures/openclaw_real_payload_20260313_2252.json
```

The replay script is stdlib-only and can run locally or on the demo host.

Run from the repo root:

```bash
python3 weian_development/demo_debug/replay_openclaw_real_payload.py \
  --base-url http://127.0.0.1:8010 \
  --out weian_development/demo_debug/artifacts/replay_openclaw_real_payload.json
```

This script:

- posts the captured payload to `/v1/chat/completions`
- captures the raw SSE response
- writes a compact JSON report

The output JSON contains:

- request metadata
- `status_code`
- `done_seen`
- `content_preview`
- `tool_deltas_preview`
- `finish_reasons`
- `raw_lines_tail`

## Debugging Streaming Problems

### 1. Check the live stream

Open:

```text
http://127.0.0.1:8010/api/live/stream
```

Useful event types:

- `request_started`
- `text_delta`
- `tool_call_delta`
- `tool_call_finish`
- `raw_chat_delta`
- `raw_completion_delta`
- `request_finish`
- `request_error`

`raw_chat_delta` is especially useful for this issue because it exposes the upstream parsed chat delta object even when the normal visible text stream looks stuck.

### 2. Inspect gateway and backend logs

Check:

```bash
tail -n 200 demo/vllm/logs/demo.log
tail -n 200 demo/vllm/logs/baseline.log
tail -n 200 demo/vllm/logs/triattention.log
```

For tool-calling and streaming issues, look for:

- parser exceptions
- `tool_choice` validation failures
- long-running requests with no visible streaming output
- triattention compression/reclaim logs

### 3. Known issue pattern

One recurring pattern is that the model emits text that looks like tool-calling instructions or examples, such as XML-tag tool-call template text. That can confuse the visible streaming behavior.

The current gateway patch does not expose tokenizer-level raw bytes for `/v1/chat/completions`, but it does expose raw parsed chat deltas via `raw_chat_delta`, which is usually enough to determine whether the backend is still streaming text or has switched into tool-call-shaped deltas.

## Recommended Debug Loop

1. Start the demo.
2. Confirm `/healthz` is green.
3. Start watching `/api/live/stream`.
4. Run the replay script.
5. Compare:
   - replay JSON artifact
   - live stream events
   - `demo.log`
   - `baseline.log`
   - `triattention.log`

## Current Testcase Command

Use this exact command when reproducing the captured OpenClaw case:

```bash
python3 weian_development/demo_debug/replay_openclaw_real_payload.py \
  --base-url http://127.0.0.1:8010 \
  --payload-file weian_development/demo_debug/fixtures/openclaw_real_payload_20260313_2252.json \
  --out weian_development/demo_debug/artifacts/replay_openclaw_real_payload_latest.json
```
