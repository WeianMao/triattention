"""Generate combined 2x3 grid figure showing Panel B and Panel C for three models.

Layout:
- Row 1 (A): Histograms of Pearson correlation for all heads
- Row 2 (B): Per-layer percentage of heads with r > threshold
- Columns: Qwen3-8B, Qwen2-7B, Llama-8B

Usage:
    conda activate rkv
    python paper_visualizations/scripts/freq_magnitude_v2/generate_combined_multimodel_grid.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force Times New Roman font for all text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts match Times New Roman for math
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np
import torch
from scipy import stats
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str  # Display name
    trace_dir: Path  # Directory containing qk.pt and metadata.json
    model_path: Path  # Model directory for RoPE parameters


# Predefined model configurations
MODELS = [
    ModelConfig(
        name="DS-Qwen3-8B",
        trace_dir=ROOT / "outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34",
        model_path=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
    ),
    ModelConfig(
        name="DS-Qwen-7B",
        trace_dir=ROOT / "paper_visualizations/outputs/qk_traces/deepseek_r1_qwen2_7b/trace_aime24_q76",
        model_path=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B"),
    ),
    ModelConfig(
        name="DS-Llama-8B",
        trace_dir=ROOT / "paper_visualizations/outputs/qk_traces/deepseek_r1_llama_8b/trace_aime24_q70",
        model_path=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B"),
    ),
]


def get_model_family(config) -> str:
    """Detect model family from config."""
    model_type = getattr(config, "model_type", "").lower()
    architectures = getattr(config, "architectures", [])

    if "qwen3" in model_type or any("Qwen3" in a for a in architectures):
        return "qwen3"
    elif "qwen2" in model_type or any("Qwen2" in a for a in architectures):
        return "qwen2"
    elif "llama" in model_type or any("Llama" in a for a in architectures):
        return "llama"
    else:
        raise ValueError(f"Unsupported model type: {model_type}, architectures: {architectures}")


def get_rotary_embedding(config, model_family: str, device):
    """Get the appropriate RotaryEmbedding class for the model family."""
    if model_family == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        return Qwen3RotaryEmbedding(config=config, device=device)
    elif model_family == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        return Qwen2RotaryEmbedding(config=config, device=device)
    elif model_family == "llama":
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        return LlamaRotaryEmbedding(config=config, device=device)
    else:
        raise ValueError(f"Unknown model family: {model_family}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def invert_rope(rotated, cos, sin, scale):
    z = rotated / scale
    return z * cos - rotate_half(z) * sin


def to_complex_pairs(tensor):
    num_freq = tensor.shape[-1] // 2
    return torch.complex(tensor[:, :num_freq].float(), tensor[:, num_freq:].float())


def compute_per_query_pearson(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
    inv_freq: torch.Tensor,
    max_dist: int,
    device: torch.device,
) -> float:
    """Compute per-query Pearson correlation with log-spaced distance sampling."""
    token_count = q_block.shape[0]
    head_dim = q_block.shape[-1]
    num_freq = head_dim // 2

    # Invert RoPE
    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

    # Complex representation
    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)

    # Mean vectors
    q_mean = q_complex.mean(dim=0)
    k_mean = k_complex.mean(dim=0)

    phi_f = torch.angle(q_mean) - torch.angle(k_mean)
    amplitude = torch.abs(q_mean) * torch.abs(k_mean)
    omega = inv_freq[:num_freq].to(device=device, dtype=torch.float32)

    # Precompute reconstruction values for all distances
    all_distances = torch.arange(1, token_count, device=device, dtype=torch.float32)
    recon_all = (torch.cos(all_distances.unsqueeze(1) * omega.unsqueeze(0) + phi_f.unsqueeze(0)) * amplitude.unsqueeze(0)).sum(dim=1)
    recon_dict = {int(d): recon_all[i].item() for i, d in enumerate(all_distances.long().tolist())}

    # Per-Query correlation with log-spaced sampling
    q_float = q_block.float()
    k_float = k_block.float()
    min_history = 50
    num_log_samples = 50
    num_query_samples = 500

    # Sample query positions
    query_positions = torch.unique(torch.logspace(
        math.log10(min_history), math.log10(token_count - 1), num_query_samples, device=device
    ).long())
    query_positions = query_positions[(query_positions >= min_history) & (query_positions < token_count)]

    per_query_pearsons = []

    for query_pos in query_positions.tolist():
        log_distances = torch.unique(torch.logspace(0, math.log10(query_pos), num_log_samples, device=device).long())
        log_distances = log_distances[(log_distances >= 1) & (log_distances <= query_pos)]

        if len(log_distances) < 3:
            continue

        key_positions = query_pos - log_distances
        q_vec = q_float[query_pos]
        k_vecs = k_float[key_positions]
        actual_scores = (q_vec.unsqueeze(0) * k_vecs).sum(dim=1)

        predicted_scores = torch.tensor([recon_dict[int(d)] for d in log_distances.tolist()], device=device)

        r, _ = stats.pearsonr(actual_scores.cpu().numpy(), predicted_scores.cpu().numpy())
        if not np.isnan(r):
            per_query_pearsons.append(r)

    return np.mean(per_query_pearsons) if per_query_pearsons else 0.0


def load_model_data(model_cfg: ModelConfig, device: torch.device, threshold: float = 0.55, max_distance: int = 5000):
    """Load data and compute correlation for a single model."""
    print(f"\n{'='*60}")
    print(f"Processing: {model_cfg.name}")
    print(f"{'='*60}")

    # Load trace metadata
    qk_path = model_cfg.trace_dir / "qk.pt"
    meta_path = model_cfg.trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {model_cfg.trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])

    # Load model config and detect family
    print(f"Loading config from {model_cfg.model_path}...")
    config = AutoConfig.from_pretrained(model_cfg.model_path, trust_remote_code=True)
    model_family = get_model_family(config)
    print(f"Detected model family: {model_family}")

    # Handle rope_scaling
    rope_scaling = config.rope_scaling
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
            rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
        rope_scaling.pop("attn_factor", None)
        if not rope_scaling:
            rope_scaling = None
    config.rope_scaling = rope_scaling

    # Get rotary embedding
    rotary = get_rotary_embedding(config, model_family, device)
    inv_freq = rotary.inv_freq.to(torch.float64)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # Load Q/K tensors
    print("Loading Q/K tensors...")
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
    k_tensor = data["k"]

    num_layers, num_heads, _, head_dim = q_tensor.shape
    total_heads = num_layers * num_heads
    print(f"Model: {num_layers} layers, {num_heads} heads/layer = {total_heads} total heads")
    print(f"Token count: {token_count}")

    # Build RoPE tables
    position_ids = torch.arange(token_count, device=device).unsqueeze(0)
    base = torch.zeros(1, token_count, head_dim, device=device, dtype=torch.float32)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=torch.float32)
    sin_table = sin_table[0].to(dtype=torch.float32)

    # Compute correlation for all heads
    print("Computing correlation for all heads...")
    results = []

    for layer in range(num_layers):
        for head in range(num_heads):
            q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)
            k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)

            pearson = compute_per_query_pearson(
                q_block, k_block, cos_table, sin_table,
                attention_scale, inv_freq, max_distance, device
            )
            results.append({
                'layer': layer,
                'head': head,
                'ind_pearson': float(pearson),
            })

        print(f"  Layer {layer + 1}/{num_layers} done")

    all_pearson = np.array([r['ind_pearson'] for r in results])

    # Per-layer stats
    layer_above_thr_pct = []
    for layer in range(num_layers):
        layer_pearson = [r['ind_pearson'] for r in results if r['layer'] == layer]
        above_thr = sum(1 for p in layer_pearson if p > threshold)
        pct = above_thr / len(layer_pearson) * 100
        layer_above_thr_pct.append(pct)

    print(f"Mean Pearson r = {all_pearson.mean():.4f}, Std = {all_pearson.std():.4f}")

    return {
        'name': model_cfg.name,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'all_pearson': all_pearson,
        'layer_above_thr_pct': layer_above_thr_pct,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate combined 2x3 grid figure for three models"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--threshold", type=float, default=0.55, help="Correlation threshold")
    parser.add_argument("--max-distance", type=int, default=5000, help="Maximum token distance")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "paper_visualizations/outputs/freq_magnitude_v2/fig_freq_reconstruction_multimodel_grid.png",
        help="Output path for the figure",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "paper_visualizations/outputs/freq_magnitude_v2/cache",
        help="Directory for caching computed data",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load from cache if available (skip computation)",
    )
    parser.add_argument(
        "--save-cache",
        action="store_true",
        default=True,
        help="Save computed data to cache (default: True)",
    )
    return parser.parse_args()


def get_cache_path(cache_dir: Path, model_name: str) -> Path:
    """Get cache file path for a model."""
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    return cache_dir / f"{safe_name}_correlation_data.npz"


def save_to_cache(cache_path: Path, data: dict) -> None:
    """Save computed data to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        name=data['name'],
        num_layers=data['num_layers'],
        num_heads=data['num_heads'],
        all_pearson=data['all_pearson'],
        layer_above_thr_pct=np.array(data['layer_above_thr_pct']),
    )
    print(f"  Saved to cache: {cache_path}")


def load_from_cache(cache_path: Path) -> dict:
    """Load computed data from cache."""
    data = np.load(cache_path, allow_pickle=True)
    return {
        'name': str(data['name']),
        'num_layers': int(data['num_layers']),
        'num_heads': int(data['num_heads']),
        'all_pearson': data['all_pearson'],
        'layer_above_thr_pct': data['layer_above_thr_pct'].tolist(),
    }


def main() -> None:
    args = parse_args()
    mask_process_command(f"PD-L1_binder_fig{args.gpu}")

    np.random.seed(42)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    # Load data for all models (with caching support)
    model_data_list = []
    for model_cfg in MODELS:
        cache_path = get_cache_path(args.cache_dir, model_cfg.name)

        if args.use_cache and cache_path.exists():
            print(f"\n{'='*60}")
            print(f"Loading from cache: {model_cfg.name}")
            print(f"{'='*60}")
            data = load_from_cache(cache_path)
            print(f"  Loaded {data['num_layers']} layers, {data['num_heads']} heads")
            print(f"  Mean Pearson r = {data['all_pearson'].mean():.4f}")
        else:
            data = load_model_data(model_cfg, device, args.threshold, args.max_distance)
            if args.save_cache:
                save_to_cache(cache_path, data)

        model_data_list.append(data)

    # ========== Plotting ==========
    print("\n" + "=" * 60)
    print("Generating combined figure...")
    print("=" * 60)

    # Colors
    color_recon = (85 / 250, 104 / 250, 154 / 250)
    color_bar = (187 / 250, 130 / 250, 90 / 250)
    face_color = (231 / 250, 231 / 250, 240 / 250)

    FONT_SIZE = 12
    LABEL_FONT_SIZE = 16
    TITLE_FONT_SIZE = 14
    LABEL_FONT = 'Times New Roman'

    def style_ax(ax):
        ax.set_facecolor(face_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=FONT_SIZE)
        ax.set_axisbelow(True)
        ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)

    # Create figure with 2 rows, 3 columns
    # Adjust to be wider (closer to 4-column ratio): increase width, reduce height
    fig = plt.figure(figsize=(14, 5.5))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.28, wspace=0.25)

    # Row 1: Histograms (Panel B from original)
    for col, data in enumerate(model_data_list):
        ax = fig.add_subplot(gs[0, col])
        style_ax(ax)

        all_pearson = data['all_pearson']
        bin_edges = np.arange(-0.2, 1.025, 0.05)
        ax.hist(all_pearson, bins=bin_edges, color=color_recon, alpha=0.85, edgecolor='white', linewidth=0.8)
        ax.axvline(all_pearson.mean(), color='#E24A33', linestyle='--', linewidth=2.5,
                   label=f'Mean = {all_pearson.mean():.2f}')
        ax.set_xlabel('Attn Reconstruction Pearson $\\bar{r}$', fontsize=FONT_SIZE)
        if col == 0:
            ax.set_ylabel('Count', fontsize=FONT_SIZE)
        ax.legend(frameon=False, fontsize=FONT_SIZE - 1, loc='upper left')
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_xlim(-0.25, 1.05)

        # Column title (model name) - bold
        ax.set_title(data['name'], fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=10)

        # Row label (A) only on first column
        if col == 0:
            ax.text(-0.22, 1.08, '(A)', transform=ax.transAxes, fontsize=LABEL_FONT_SIZE,
                    fontweight='bold', va='bottom', fontname=LABEL_FONT)

    # Row 2: Per-layer percentage (Panel C from original) - made square-like with narrow bars
    for col, data in enumerate(model_data_list):
        ax = fig.add_subplot(gs[1, col])
        style_ax(ax)
        ax.grid(True, axis='y', alpha=0.7, color='white', linewidth=1.5)

        num_layers = data['num_layers']
        layer_above_thr_pct = data['layer_above_thr_pct']
        layers_arr = np.arange(num_layers)

        # Narrower bars with smaller gap
        bar_width = 0.6
        ax.bar(layers_arr, layer_above_thr_pct, color=color_bar, alpha=0.85,
               edgecolor='white', linewidth=0.3, width=bar_width)

        ax.set_xlabel('Layer Index', fontsize=FONT_SIZE)
        if col == 0:
            ax.set_ylabel(f'% Heads with $\\bar{{r}}$ > {args.threshold:.2f}', fontsize=FONT_SIZE)

        # Adjust x-ticks based on number of layers
        if num_layers <= 30:
            ax.set_xticks(layers_arr[::4])
        else:
            ax.set_xticks(layers_arr[::5])

        ax.set_ylim(0, 100)
        ax.set_xlim(-0.5, num_layers - 0.5)

        # Row label (B) only on first column
        if col == 0:
            ax.text(-0.22, 1.08, '(B)', transform=ax.transAxes, fontsize=LABEL_FONT_SIZE,
                    fontweight='bold', va='bottom', fontname=LABEL_FONT)

    plt.tight_layout()

    # Save
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"\nFigure saved to {args.output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for data in model_data_list:
        print(f"\n{data['name']}:")
        print(f"  Layers: {data['num_layers']}, Heads: {data['num_heads']}")
        print(f"  Mean Pearson r = {data['all_pearson'].mean():.4f}")
        print(f"  Std = {data['all_pearson'].std():.4f}")


if __name__ == "__main__":
    main()
