# Streaming Perplexity Workflow

This note documents how to reproduce the streaming-vs-full perplexity comparison that now lives under `weian_development/streaming_perplexity/`.

## 1. Generate streaming perplexity tensors

```
bash scripts/run_streaming_perplexity.sh
```

The helper script is self-contained and launches

```
python weian_development/streaming_perplexity/run_streaming_perplexity_distributed.py \
  outputs/deepseek_r1_qwen3_8b/offline_reasoning_json \
  outputs/deepseek_r1_qwen3_8b/perplexity_stream \
  --gpus 0 --limit-files 1 --limit-traces 1 \
  --chunk-size 128 --stream-window 2048 --verbose
```

You can adjust env vars (`PYTHON_BIN`, `PYTHONPATH`) or edit the script when pointing at other datasets.

## 2. Compare against full-attention baseline

Once `outputs/deepseek_r1_qwen3_8b/perplexity_stream` exists (matching the already-collected `perplexity_full` tensors), run:

```
python weian_development/streaming_perplexity/analyze_stream_vs_full.py
```

This script

1. Loads the tokenizer from `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B`.
2. For each JSON metadata entry under `perplexity_full/metadata`, aligns the same trace from the streaming tensors.
3. Splits the generated text into sentence spans, computes average log-prob/perplexity per sentence, and records KL-style deltas between the two modes.
4. Writes the structured results to `outputs/deepseek_r1_qwen3_8b/perplexity_stream_sentence_stats.json`.

`TRACE_OFFSET` inside the script controls which trace from each JSON is analyzed (default = first trace). Adjust as needed for multi-trace experiments.

## 3. Outputs

- `outputs/deepseek_r1_qwen3_8b/perplexity_stream_sentence_stats.json` now contains an `aggregate` block (overall KL stats) and per-question sentence metrics.
- Each sentence entry has streaming vs full log-prob, perplexity, and token counts, which can be used for downstream visualizations or regression tracking.

Remember to refresh `perplexity_stream` whenever the streaming evaluator changes; the analysis script assumes both directories contain tensors for the same set of questions/trace indices.
