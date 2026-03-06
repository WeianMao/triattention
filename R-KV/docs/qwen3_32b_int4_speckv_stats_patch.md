# Qwen3-32B-INT4 SpeckV Stats Runtime Patch

## Symptom
- Direct stats generation for Qwen3-32B-INT4 fails at model init/runtime (commonly `NotImplementedError` in GPTQ fused qlinear, or post-init CUDA-path errors).

## Root Cause
- `from_pretrained` may choose a CPU-first quant backend, then execution moves to CUDA.
- In this GPTQ INT4 setup, the selected fused path can be incompatible after that transition.
- Grad-enabled init can also trigger in-place/post-init conflicts in the quantized load flow.

## Non-invasive Fix
- Keep calibration/core algorithm code unchanged.
- Apply a runtime wrapper patch that:
  - runs `torch.set_grad_enabled(False)` before model construction;
  - monkey-patches `AutoModelForCausalLM.from_pretrained` to default `device_map="cuda"`.

## Long-Trace Memory Patch (`logits_to_keep`)
- The patched launcher also monkey-patches Qwen3 CausalLM `forward` to default `logits_to_keep=1` only when the caller does not pass it.
- This avoids materializing full-sequence logits during long trace calibration steps, which lowers activation/logit memory pressure and helps prevent OOM.

## Long-Trace Memory Patch (`logits_to_keep`)
- The patched launcher also monkey-patches Qwen3 CausalLM `forward` to default `logits_to_keep=1` only when the caller does not pass it.
- This avoids materializing full-sequence logits during long trace calibration steps, which lowers activation/logit memory pressure and helps prevent OOM.

## Patched Script Path
- `R-KV/linxi_experiments/run_qwen3-32b-int4_speckv_stats_patched.sh`

## Usage (NUM_TRACES=30)
```bash
NUM_TRACES=30 bash R-KV/linxi_experiments/run_qwen3-32b-int4_speckv_stats_patched.sh
```

## Key Environment Overrides
- `NUM_TRACES` (default `3`)
- `DTYPE` (default `bfloat16`)
- `ATTN_IMPLEMENTATION` (default `flash_attention_2`)
- `KV_BUDGET` (default `2048`)
- `FREE_MEM_THRESHOLD_MIB` (default `5000`)
- `POLL_SECONDS` (default `30`)
- `TRACE_ROOT`, `MODEL_PATH`, `OUTPUT_PATH`, `PIXI_BIN`

## Output Path
- Default: `/home/wayne/linxi/speckv/R-KV/outputs/qwen3-8b_aime24_sample8_chat/stats/qwen3_32b_int4_speckv_stats.pt`
