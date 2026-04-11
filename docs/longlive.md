# LongLive Video Generation

TriAttention supports KV cache compression for [LongLive](https://github.com/NVlabs/LongLive), a real-time causal long video generation model built on Wan2.1-T2V-1.3B. The integration uses a monkey-patch approach to inject trigonometric KV scoring into LongLive's causal inference pipeline, enabling longer video generation on the same GPU without modifying the upstream model code.

## Setup

### Clone with Submodule

The LongLive source is included as a git submodule under `triattention/longlive/`:

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
pip install -r triattention/longlive/requirements.txt

# Flash Attention (recommended)
pip install flash-attn --no-build-isolation
```

### Download Model Weights

LongLive requires the Wan2.1-T2V-1.3B base model, a LoRA checkpoint, and (optionally) pre-computed KV calibration statistics. Follow the instructions in the [LongLive README](https://github.com/NVlabs/LongLive) to download:

- `longlive_models/models/longlive_base.pt` -- base generator checkpoint
- `longlive_models/models/lora.pt` -- LoRA adapter weights

Place them under the `longlive_models/` directory inside the LongLive submodule.

## Usage

The workflow has two steps: **calibrate** (collect Q statistics) and **compress** (run inference with a reduced KV cache). A pre-computed calibration file is included at `triattention/longlive/assets/normal_q_stats_120f_peak49.pt`, so you can skip directly to Step 2.

### Step 1: Calibrate (optional — pre-computed file provided)

A pre-computed calibration file is already included at `triattention/longlive/assets/normal_q_stats_120f_peak49.pt`. **You can skip this step entirely** and go directly to Step 2. Re-calibration is only needed if you change the model architecture or want to calibrate on a different frame count.

To run calibration yourself, collect pre-RoPE Q statistics from a reference run. This produces a `.pt` file containing per-head frequency statistics used for scoring.

```bash
python -m triattention.longlive.run \
    --config_path triattention/longlive/configs/longlive_inference_triattention_120f.yaml \
    --model_kwargs.kv_compression_mode calibrate \
    --model_kwargs.kv_stats_path longlive_models/kv_stats/normal_q_stats_120f_peak49.pt
```

Calibration runs a normal forward pass and records Q-state statistics from every attention layer. The resulting file is model-specific but domain-agnostic -- you do not need to re-calibrate when changing prompts.

### Step 2: Compress

Run inference with the compressed KV cache:

```bash
python -m triattention.longlive.run \
    --config_path triattention/longlive/configs/longlive_inference_triattention_120f.yaml
```

The example config retains 46 out of 120 latent frames (~38% budget) using layer-per-head pruning.

### Example: 120-frame Generation

```bash
# Generate a 120-frame video with KV compression (budget = 46 frames)
python -m triattention.longlive.run \
    --config_path triattention/longlive/configs/longlive_inference_triattention_120f.yaml

# Output is saved to videos/triattention_120f/
```

## Configuration Reference

All parameters are set under `model_kwargs` in the YAML config file.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `kv_compression_mode` | Operating mode: `off` (disabled), `calibrate` (collect Q stats), or `compress` (prune KV cache) | `off` |
| `kv_stats_path` | Path to save (calibrate) or load (compress) the Q statistics `.pt` file | -- |
| `kv_budget_tokens` | Maximum number of KV tokens retained after pruning. Controls peak GPU memory | -- |
| `kv_compress_every_n_frames` | Trigger compression every N decoded frames | `10` |
| `kv_keep_last_frames` | Number of most-recent frames never evicted | `num_frame_per_block` |
| `kv_pruning_mode` | Pruning granularity: `perhead` (shared across layers) or `layer_perhead` (independent per layer and head) | `perhead` |
| `kv_score_aggregation` | Score aggregation across offset distances: `mean` or `max` | `mean` |
| `kv_perhead_layer_aggregation` | Layer aggregation strategy for `layer_perhead` mode: `mean_of_layer_max` | `mean_of_layer_max` |
| `kv_offset_max_frames` | Maximum frame offset for geometric probing | `128` |
| `kv_normalize_scores` | Normalize scores to zero-mean unit-variance before ranking | `true` |
| `kv_tie_break_noise` | Add small random noise to break ties in score ranking | `true` |
| `kv_tie_break_noise_scale` | Scale of tie-breaking noise | `1e-6` |
| `kv_random_seed` | Random seed for reproducibility | `0` |
| `local_attn_size` | **Must be `-1`** when using KV compression (see below) | `-1` |
| `sink_size` | Number of sink tokens. Set to `0` when compression is active | `0` |

Top-level config parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_frame_per_block` | Frames decoded per block (temporal chunk size) | `3` |
| `num_output_frames` | Total number of latent frames to generate | `120` |
| `data_path` | Path to a text file with one prompt per line | -- |
| `output_folder` | Directory for output videos | -- |

## Important Notes

- **`local_attn_size` must be `-1`**: KV compression requires global attention. The compressor manages its own token eviction and is incompatible with LongLive's native sliding-window attention. Setting a positive `local_attn_size` with compression enabled will raise a `ValueError`.
- **Pre-computed calibration**: The provided calibration file (`triattention/longlive/assets/normal_q_stats_120f_peak49.pt`) was collected on 120 latent frames. Re-calibration is only needed if you change the model architecture or the target frame count significantly.
- **Memory savings**: The `kv_budget_tokens` parameter directly controls peak GPU memory. A budget of ~38% of total KV tokens typically preserves video quality while saving substantial memory.
- **Zero overhead when disabled**: When `kv_compression_mode` is `off` (or unset), no compression code runs and the inference path is identical to the original LongLive pipeline.
- **Distributed inference**: The pipeline supports multi-GPU inference via `torchrun`. KV compression works transparently in distributed mode.

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
