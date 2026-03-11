# OpenClaw Demo Testing Guide

TriAttention A/B comparison test. The OpenClaw agent reads all documents in `sample-docs/` (~30KB) and generates a weekly report, producing a long enough context to trigger KV compression.

## 1. Start Backend Services

### 1.1 Start vLLM (Remote GPU Machine)

> Always export `MAX_MODEL_LEN=32768` before launching either backend (baseline or TriAttention). This value intentionally overstates the true 16K runtime capacity so OpenClaw's context-length validation accepts the provider. Also set `VLLM_RELAXED_KV_CHECK=1` to bypass vLLM's startup KV-capacity guard. Keep `max_tokens` low (Section 4.1) to avoid hitting the backend's real limit.

SSH into the remote GPU machine and start using `swap_backend.sh`.

**TriAttention mode**:

```bash
cd TriAttention_vLLM

# Default: budget=2048, protect_prefill=false, port=8002, GPU=1
VLLM_RELAXED_KV_CHECK=1 MAX_MODEL_LEN=32768 KV_BUDGET=2048 bash linxi_dev/swap_backend.sh triattention
```

**Baseline mode**:

```bash
bash linxi_dev/swap_backend.sh baseline
```

**Check status**:

```bash
bash linxi_dev/swap_backend.sh status
```

Default parameters can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_PORT` | `8002` | vLLM listening port |
| `KV_BUDGET` | `2048` | TriAttention KV budget |
| `CUDA_VISIBLE_DEVICES` | `1` | GPU index |
| `GPU_MEMORY_UTILIZATION` | `0.95` | GPU memory utilization |
| `MAX_MODEL_LEN` | `32768` | Maximum sequence length (set higher than the actual 16K runtime limit to bypass OpenClaw's context guard) |
| `TRIATTN_RUNTIME_PROTECT_PREFILL` | `false` | Whether to protect prefill tokens from compression |

### 1.2 Start Demo Gateway (Remote)

```bash
export PATH="$HOME/.pixi/bin:$PATH"

VLLM_BACKEND_URL=http://127.0.0.1:8002 \
DEMO_HOST=0.0.0.0 \
DEMO_PORT=8010 \
nohup pixi run python -m uvicorn demo.vllm.server:app \
  --host 0.0.0.0 --port 8010 \
  > /tmp/demo_gateway.log 2>&1 &

# Verify
curl -s http://127.0.0.1:8010/healthz | python3 -m json.tool
```

### 1.3 Set Up SSH Tunnel (Local)

```bash
# Keep the terminal open
ssh -L 8010:127.0.0.1:8010 <remote-host>
```

Verify:

```bash
curl http://127.0.0.1:8010/healthz
curl http://127.0.0.1:8010/v1/models
```

## 2. Configure OpenClaw

### 2.1 Edit `~/.openclaw/openclaw.json`

Add a provider pointing to the local tunnel under `models.providers`:

```jsonc
{
  "models": {
    "mode": "merge",
    "providers": {
      // ... existing providers ...
      "triattention-demo": {
        "baseUrl": "http://127.0.0.1:8010/v1",
        "apiKey": "local",
        "auth": "api-key",
        "api": "openai-completions",
        "models": [
          {
            "id": "<model-path-to-Qwen3-32B-INT4>",
            "name": "Qwen3-32B-INT4 (TriAttention Demo)",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 32768,
            "maxTokens": 4096
          }
        ]
      }
    }
  }
}
```

> **Note**: The `id` must exactly match the model path loaded on the remote vLLM server, otherwise you'll get a 404. Check the actual model ID with `curl http://127.0.0.1:8010/v1/models`.

### 2.2 Create a Test Agent

Add the following to `agents.list`:

```jsonc
{
  "agents": {
    "list": [
      // ... existing agents ...
      {
        "id": "docmind",
        "name": "docmind",
        "workspace": "<repo_root>/demo/openclaw-demo/sample-docs",
        "model": "triattention-demo/<model-path-to-Qwen3-32B-INT4>"
      }
    ]
  }
}
```

Replace `<repo_root>` with the absolute path to the local speckv repository.

### 2.3 Verify Configuration

```bash
# Confirm the agent is visible
openclaw agents list

# Quick connectivity check
openclaw agent --agent docmind -m "hello"
```

## 3. Clear Session

You must clear the history context before each test, otherwise residual conversations will interfere with results.

```bash
# Delete all session data for the docmind agent
rm -rf ~/.openclaw/agents/docmind/sessions/*

# Verify cleanup
ls ~/.openclaw/agents/docmind/sessions/
# Should be empty
```

Also confirm the backend KV cache is idle:

```bash
curl -s http://127.0.0.1:8010/api/kv-cache | python3 -c "
import sys,json; d=json.load(sys.stdin)
print('KV Usage: %.1f%%' % d.get('usage_percent', 0))"
# Should be close to 0%
```

## 4. Send Test Prompt

### 4.1 Standard Test Command

```bash
openclaw agent --agent docmind -m "Please read all files in the current directory, then write a weekly report (W9) for the DocMind project based on these documents.

Requirements:
1. Read all 6 files one by one (meeting_notes_0115.md, technical_design_RAG_pipeline.md, review_comments_0207.md, selection_report_vector_database.md, bug_report_parsing_garbled_text.md, weekly_report_W8.md)
2. Synthesize all document information and write the report with the following structure:
   - I. This Week's Progress (grouped by Backend/Algorithm/Frontend/QA)
   - II. Risk Tracking (update risk table, note status changes)
   - III. Key Decision Log
   - IV. Next Week's Plan
   - V. Coordination Items
3. Compare with last week's report (W8), noting which items are completed and which are delayed
4. Report length: 2000-3000 words"
```

> **Important**: set `--max-tokens 4000` (or similar) on the OpenClaw command. We publish `MAX_MODEL_LEN=32768` for both backends purely to satisfy OpenClaw's 32K context check, but the GPUs only sustain ~16K context in practice—any larger `max_tokens` will trigger vLLM validation failures.

### 4.2 Expected Behavior

1. The agent calls the `read` tool to read all 6 files one by one (~30KB total text enters context)
2. After all files are read, the agent starts generating the weekly report
3. KV cache grows progressively during generation
4. In TriAttention mode with a small budget (e.g., 512-2048), compression is triggered during generation

### 4.3 Monitoring Metrics

Monitor in another terminal during the test, or view real-time streaming output at http://127.0.0.1:8010:

```bash
# Real-time KV cache usage monitoring (refresh every 2 seconds)
watch -n 2 'curl -s http://127.0.0.1:8010/api/kv-cache | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(\"KV Usage: %.1f%%  Used: %s  Capacity: %s\" % (
    d.get(\"usage_percent\",0),
    int(d.get(\"used_tokens_estimate\",0)),
    int(d.get(\"capacity_tokens_estimate\",0)),
))"'
```

```bash
# View compression logs (remote, TriAttention mode)
ssh <remote-host> 'tail -f /tmp/vllm_triattention.log | grep -E "compression (applied|skipped)|reclaimed_blocks"'
```

**TriAttention expected behavior**:

- KV usage **drops** after compression is triggered (instead of monotonically increasing)
- Logs show `compression applied before=XXXX after=YYYY`
- Stable generation throughput (~25-30 tok/s)

**Baseline expected behavior**:

- KV usage monotonically increases
- In long-context scenarios, preemption may occur (requests evicted when KV is full, throughput drops to 0)

## 5. A/B Comparison Test Procedure

### 5.1 TriAttention Test

```bash
# 1. Remote: Start TriAttention backend
VLLM_RELAXED_KV_CHECK=1 MAX_MODEL_LEN=32768 KV_BUDGET=2048 bash linxi_dev/swap_backend.sh triattention

# 2. Local: Clear session
rm -rf ~/.openclaw/agents/docmind/sessions/*

# 3. Local: Send standard prompt (Section 4.1)
openclaw agent --agent docmind -m "..."

# 4. Record: output quality, KV curve, compression count, total time
```

### 5.2 Baseline Test

```bash
# 1. Remote: Switch to baseline backend
bash linxi_dev/swap_backend.sh baseline

# 2. Local: Clear session
rm -rf ~/.openclaw/agents/docmind/sessions/*

# 3. Local: Send the same prompt
openclaw agent --agent docmind -m "..."

# 4. Record: output quality, KV curve, whether preemption occurred, total time
```

### 5.3 Comparison Dimensions

| Metric | TriAttention | Baseline |
|--------|-------------|----------|
| Output completed | | |
| Output quality (manual score 1-5) | | |
| Peak KV usage | | |
| Preemption triggered | | |
| Total time | | |
| Generation throughput (tok/s) | | |

## 6. Troubleshooting

### Output contains garbled/repeated characters

**This is the most significant known issue that needs to be resolved: garbled/repeated characters appear after compression is triggered**

### OpenClaw cannot connect

```bash
# Confirm tunnel exists
lsof -i :8010

# Confirm gateway is healthy
curl http://127.0.0.1:8010/healthz

# Confirm backend is healthy
curl http://127.0.0.1:8010/v1/models
```

### Agent does not call tools to read files

Confirm the `workspace` path points to the `sample-docs/` directory. If it still doesn't call tools, explicitly list the full file paths in the prompt:

```bash
openclaw agent --agent docmind -m "Please read the following files one by one and write a weekly report:
- meeting_notes_0115.md
- technical_design_RAG_pipeline.md
- review_comments_0207.md
- selection_report_vector_database.md
- bug_report_parsing_garbled_text.md
- weekly_report_W8.md"
```

### Compression not triggered

- When `protect_prefill=true`, the prefill length must be < budget for compression to occur
- 30KB of documents produce ~15K tokens during prefill; requires `protect_prefill=false` or `budget > 15K`
- `swap_backend.sh` already sets `protect_prefill=false` by default

### GPU out of memory on startup

```bash
# Check GPU usage
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# Find and clean up stale vLLM processes
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
kill <stale_pid>
```

## 7. Stats File Description

`stats/qwen3_32b_int4_speckv_stats.pt` is the TriAttention attention statistics file for Qwen3-32B-INT4 (6.8MB), generated by R-KV offline evaluation, used to guide runtime KV compression token retention strategy.

When starting vLLM on the remote machine, specify this file path via the `STATS_PATH` environment variable:

```bash
# swap_backend.sh uses the following default path (confirm it exists on the remote machine)
STATS_PATH=/path/to/stats/qwen3_32b_int4_speckv_stats.pt

# If the stats file is not at the default location, specify manually
STATS_PATH=/path/to/stats.pt KV_BUDGET=2048 bash linxi_dev/swap_backend.sh triattention
```

This file is also included in the repository under `demo/openclaw-demo/stats/` and can be copied to the remote deployment as needed.

## 8. Sample Docs Description

`sample-docs/` contains 6 fictional DocMind project documents, totaling ~30KB:

| File | Content | Size |
|------|---------|------|
| meeting_notes_0115.md | Project kickoff meeting, defining M1-M4 milestones | 6.4K |
| technical_design_RAG_pipeline.md | RAG pipeline technical design v2.0 | 7.1K |
| review_comments_0207.md | CTO technical design review feedback | 4.2K |
| selection_report_vector_database.md | Qdrant/Milvus/Weaviate selection | 3.9K |
| bug_report_parsing_garbled_text.md | PDF parsing Chinese garbled text P1 bug | 4.4K |
| weekly_report_W8.md | Last week's report, used as comparison baseline | 4.9K |

> **Note:** Keep the demo provider `contextWindow` aligned with the model's intrinsic limit (do not reduce to 16K) to avoid OpenClaw's context-size validation from rejecting the agent configuration.

These documents form a coherent project narrative; the agent needs to synthesize them to produce a quality weekly report.
