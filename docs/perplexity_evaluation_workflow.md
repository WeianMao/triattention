# Reasoning Perplexity Evaluation Workflow

This note captures the current procedure for recomputing per-token log probabilities on DeepConf reasoning traces.

## Scripts

- `weian_development/extract_reasoning_texts.py` – streams DeepConf `.msgpack.*` outputs and saves per-trace reasoning text and metadata into JSON.
- `weian_development/compute_reasoning_perplexity.py` – loads the chat-formatted prompt, replays the full prompt + trace with the base model, and records token ids + log probs (prompt tokens are retained but excluded from metrics).
- `weian_development/run_perplexity_distributed.py` – dynamic worker pool that assigns JSON files to GPUs one at a time.

## Runtime expectations

- Each worker launches one full `AutoModelForCausalLM` instance on a single GPU (no tensor parallel).
- Per-trace processing is sequential (`batch_size = 1`). Chunked decoding (`--chunk-size`, default 256) controls the forward window / cached context growth.
- Prompt construction follows the DeepSeek chat template: system prefix + question from the JSON. Non-generated tokens are skipped when computing perplexity.

## Typical commands

1. Quick smoke test on a tiny subset:
   ```bash
   python weian_development/run_perplexity_distributed.py \
     outputs/deepseek_r1_qwen3_8b/offline_reasoning_json \
     outputs/deepseek_r1_qwen3_8b/perplexity_smoke \
     --gpus 0,1,2,3,4,5,6,7 \
     --limit-files 4 --limit-traces 1 \
     --chunk-size 256 --model-type deepseek --verbose
   ```

2. Full evaluation (no limits):
   ```bash
   python weian_development/run_perplexity_distributed.py \
     outputs/deepseek_r1_qwen3_8b/offline_reasoning_json \
     outputs/deepseek_r1_qwen3_8b/perplexity_full \
     --gpus 0,1,2,3,4,5,6,7 \
     --chunk-size 256 --model-type deepseek --verbose
   ```

3. Cleaning previous outputs:
   ```bash
   rm -rf outputs/deepseek_r1_qwen3_8b/perplexity_full
   ```

## Output structure

```
perplexity_full/
  metadata/
    deepthink_offline_qidXX...json   # summary per question (prompt length, avg log prob, per-trace stats)
  tensors/
    deepthink_offline_qidXX...pt     # torch.save payload with prompt token ids and per-trace tensors
```

Each `.pt` payload contains:

```python
{
  "prompt_token_ids": torch.Tensor([...], dtype=torch.int32),
  "trace_data": [
    {
      "trace_index": int,
      "generated_token_ids": torch.Tensor(dtype=int32),
      "log_probs": torch.Tensor(dtype=float32),
    },
    ...
  ],
}
```

`metadata/*.json` captures the aggregated metrics (total tokens, weighted average log prob, overall perplexity, per-trace summaries).

## Operational notes

- Monitor GPUs with `watch -n 5 nvidia-smi`; each worker maps 1:1 to one device.
- `PD-L1_binder*` processes correspond to active workers; use `pgrep -fl PD-L1_binder` for quick cleanup.
- If a worker dies (e.g., CUDA OOM), the controller terminates remaining children and exits with a non-zero code—always check the log before trusting outputs.

## Future considerations

- If memory headroom becomes an issue, reduce `--chunk-size` (default 256) or lower the number of simultaneous GPUs.
- For alternate models, change `--model-type` and ensure the prompt builder matches the generation template.
