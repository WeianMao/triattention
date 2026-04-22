# LongLive Video Generation

TriAttention supports KV cache compression for [LongLive](https://github.com/NVlabs/LongLive), a real-time causal long video generation model built on Wan2.1-T2V-1.3B. The integration uses a monkey-patch approach to inject trigonometric KV scoring into LongLive's causal inference pipeline. Compression runs *on top of* LongLive's local-attention window: the compressor selects which tokens inside the window to keep, cutting KV memory roughly in half with no changes to the upstream model code.

## Setup

### Clone with Submodule

The LongLive source is included as a git submodule under `longlive/`:

```bash
git clone --recursive https://github.com/WeianMao/triattention.git
cd triattention
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### Install Dependencies

```bash
# Install TriAttention
pip install -e .

# Install LongLive dependencies
pip install -r longlive/longlive/requirements.txt

# Flash Attention (recommended)
pip install flash-attn --no-build-isolation
```

### Download Model Weights

LongLive requires the Wan2.1-T2V-1.3B base model and a LoRA checkpoint. Follow the instructions in the [LongLive README](https://github.com/NVlabs/LongLive) to download:

- `longlive_models/models/longlive_base.pt` -- base generator checkpoint
- `longlive_models/models/lora.pt` -- LoRA adapter weights

Place them under the `longlive_models/` directory inside the LongLive submodule.

A pre-computed TriAttention calibration file is bundled at `longlive/assets/calibration_stats_81f.pt`. It is domain-agnostic (depends only on model architecture, not on prompts), so no separate calibration step is required before running inference.

## Usage

Two settings are supported out of the box:

### Setting 1 -- Multi-prompt interactive demo

Run LongLive's interactive pipeline with multiple prompts that switch at configurable frame indices, under 50% KV compression:

```bash
python -m longlive.run_interactive \
    --config_path longlive/configs/triattention_interactive.yaml
```

The default config generates 240 latent frames with five prompt switches at frames 40, 80, 120, 160, 200. Output videos are written to `videos/triattention_interactive/`.

### Setting 2 -- Single-prompt generation

Run LongLive's standard causal pipeline with a single prompt per sample under 50% KV compression:

```bash
python -m longlive.run \
    --config_path longlive/configs/triattention_120f.yaml
```

The default config generates 120 latent frames. Output videos are written to `videos/triattention_120f/`.

### Distributed Inference

Both entry points support multi-GPU inference via `torchrun`:

```bash
torchrun --nproc_per_node=8 -m longlive.run_interactive \
    --config_path longlive/configs/triattention_interactive.yaml
```

KV compression works transparently in distributed mode.

## Configuration Reference

All TriAttention parameters are set under `model_kwargs` in the YAML config file.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `local_attn_size` | LongLive's local-attention window, in frames. **Must be > 0** when compression is enabled; compression runs inside this window | `12` |
| `sink_size` | Number of sink frames pinned to absolute positions 0..sink_tokens-1 (never evicted) | `3` |
| `kv_compression_mode` | Operating mode: `off` (disabled) or `compress` (prune KV cache) | `off` |
| `kv_stats_path` | Path to the pre-computed calibration `.pt` file (bundled at `longlive/assets/calibration_stats_81f.pt`) | -- |
| `kv_budget_tokens` | Maximum number of KV tokens retained after pruning. Must be **strictly less than** `local_attn_size * frame_seq_length` (else compression is a no-op) | -- |
| `kv_compress_every_n_frames` | Trigger compression every N decoded frames | `10` |
| `kv_keep_last_frames` | Number of most-recent frames never evicted | `num_frame_per_block` |
| `kv_pruning_mode` | Pruning granularity: `perhead` (shared across layers) or `layer_perhead` (independent per layer and head) | `perhead` |
| `kv_score_aggregation` | Score aggregation across offset distances: `mean` or `max` | `mean` |
| `kv_perhead_layer_aggregation` | Layer aggregation strategy for `perhead` mode: `mean_of_layer_max` or `max` | `mean_of_layer_max` |
| `kv_offset_max_frames` | Maximum frame offset for geometric probing | `128` |
| `kv_normalize_scores` | Normalize scores to zero-mean unit-variance before ranking | `true` |
| `kv_tie_break_noise` | Add small random noise to break ties in score ranking | `true` |
| `kv_tie_break_noise_scale` | Scale of tie-breaking noise | `1e-6` |
| `kv_random_seed` | Random seed for reproducibility | `0` |
| `kv_protect_sink` | When `true`, sink slots are never evicted by the compressor | `true` |

Top-level config parameters used by the runners:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_frame_per_block` | Frames decoded per block (temporal chunk size) | `3` |
| `num_output_frames` | Total number of latent frames to generate | -- |
| `data_path` | Path to prompt file (`.txt` for Setting 2, `.jsonl` for Setting 1) | -- |
| `output_folder` | Directory for output videos | -- |
| `switch_frame_indices` | Setting 1 only: frame indices where the next prompt becomes active (comma-separated) | -- |

## Important Notes

- **`local_attn_size` must be > 0**: Compression runs on top of LongLive's local attention. The compressor selects tokens from the local-attention window, so `local_attn_size` must be a positive integer and the budget must be strictly smaller than `local_attn_size * frame_seq_length`.
- **Budget sizing**: For the bundled configs we use `local_attn_size=12`, `frame_seq_length=1560`, so the local-attention window is `12 * 1560 = 18720` tokens. A budget of `9360` keeps ~50% of the window.
- **Pre-computed calibration**: The shipped file `longlive/assets/calibration_stats_81f.pt` is model-specific but domain-agnostic. Re-calibration is only needed if you change the model architecture.
- **Multi-prompt support**: The interactive pipeline handles prompt switching at user-defined frame indices. After each switch, the compressor re-syncs its position state defensively and re-runs compression so the new prompt's context is retained.
- **Zero overhead when disabled**: When `kv_compression_mode` is `off` (or unset), no compression code runs and the inference path is identical to the original LongLive pipeline.

## Citation

```bibtex
@article{mao2026triattention,
    title={TriAttention: Efficient Long Reasoning with Trigonometric KV Compression},
    author={Weian Mao and Xi Lin and Wei Huang and Yuxin Xie and Tianfu Fu and Bohan Zhuang and Song Han and Yukang Chen},
    year={2026},
    eprint={2604.04921},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
