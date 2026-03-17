"""Generate score visualization for Layer 3, Head 5.

This script visualizes:
1. Norm-dependent term (additive) as a bar/strip heatmap
2. Full scoring function as a pseudo attention map
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants ============
FONT_SIZE = 16
color_dependent = (85 / 250, 104 / 250, 154 / 250)    # blue
face_color = (231 / 250, 231 / 250, 240 / 250)        # light gray-purple background

attn_cmap_custom = LinearSegmentedColormap.from_list(
    "attn_custom", [face_color, color_dependent]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate L3H5 score visualization")
    parser.add_argument(
        "trace_dir",
        type=Path,
        nargs="?",
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for RoPE parameters",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Pooling window size for attention maps",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figure")
    parser.add_argument("--layer", type=int, default=3, help="Layer index")
    parser.add_argument("--head", type=int, default=5, help="Head index")
    return parser.parse_args()


# ============ RoPE Utilities ============
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def invert_rope(rotated: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, scale: float) -> torch.Tensor:
    z = rotated / scale
    return z * cos - rotate_half(z) * sin


def to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    num_freq = tensor.shape[-1] // 2
    return torch.complex(tensor[..., :num_freq].float(), tensor[..., num_freq:].float())


def compute_frequency_scaling(rotary, head_dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Compute frequency scaling factors."""
    position_ids = torch.zeros(1, 1, device=device, dtype=torch.long)
    probe = torch.zeros(1, 1, head_dim, device=device, dtype=dtype)
    cos, sin = rotary(probe, position_ids)
    cos0 = cos[0, 0]
    sin0 = sin[0, 0]
    # For half-style: cos0 and sin0 are [head_dim], pairs are (cos0[:d], cos0[d:])
    d = head_dim // 2
    scale = torch.sqrt(cos0[:d].pow(2) + sin0[:d].pow(2))
    return scale.to(device=device, dtype=torch.float32)


def compute_score_components(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    inv_freq: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute scoring components from Q/K.

    Returns:
        q_mean_complex: E[q] in complex form [num_freq]
        q_abs_mean: E[|q|] [num_freq]
        k_complex: k in complex form [seq_len, num_freq]
        k_mean_complex: E[k] in complex form [num_freq]
        omega: angular frequencies [num_freq]
    """
    head_dim = q_orig.shape[-1]
    num_freq = head_dim // 2

    q_complex = to_complex_pairs(q_orig)  # [seq_len, num_freq]
    k_complex = to_complex_pairs(k_orig)  # [seq_len, num_freq]

    # Statistics from Q
    q_mean_complex = q_complex.mean(dim=0)  # E[q]
    q_abs_mean = torch.abs(q_complex).mean(dim=0)  # E[|q|]

    # Statistics from K
    k_mean_complex = k_complex.mean(dim=0)  # E[k]

    omega = inv_freq[:num_freq].to(device=device, dtype=torch.float32)

    return q_mean_complex, q_abs_mean, k_complex, k_mean_complex, omega


def compute_position_dependent_kernel(
    q_mean_complex: torch.Tensor,
    k_mean_complex: torch.Tensor,
    omega: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    max_distance: int,
    device: torch.device,
    gaussian_sigma: float = 750.0,
) -> torch.Tensor:
    """
    Compute position-dependent kernel using E[q] and E[k].

    kernel[d] = Σ_f |E[q]| * |E[k]| * freq_scale_sq * cos(ω_f * d + φ_f)
    where φ_f = angle(E[q]) - angle(E[k])
    """
    phi_f = torch.angle(q_mean_complex) - torch.angle(k_mean_complex)
    amplitude = torch.abs(q_mean_complex) * torch.abs(k_mean_complex) * freq_scale_sq

    # Compute kernel for extended range (for proper smoothing)
    extended_length = int(max_distance * 2)
    distances = torch.arange(extended_length, device=device, dtype=torch.float32)
    phase_matrix = distances.unsqueeze(1) * omega.unsqueeze(0) + phi_f.unsqueeze(0)
    kernel = (torch.cos(phase_matrix) * amplitude.unsqueeze(0)).sum(dim=1)

    # Apply Gaussian smoothing
    if gaussian_sigma > 0:
        kernel_np = kernel.cpu().numpy()
        kernel_np = gaussian_filter1d(kernel_np, sigma=gaussian_sigma)
        kernel = torch.from_numpy(kernel_np).to(device=device, dtype=torch.float32)

    # Truncate to max_distance
    kernel = kernel[:max_distance]

    return kernel


def compute_norm_dependent_scores(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    k_complex: torch.Tensor,
    freq_scale_sq: torch.Tensor,
) -> torch.Tensor:
    """
    Compute norm-dependent (additive) scores for each key position.

    Formula: additive = Σ_f (extra_f * freq_scale_sq_f)
    where extra = (|E[|q|]| - |E[q]|) * |k|

    Returns:
        scores: [seq_len] norm-dependent score for each key position
    """
    q_mean_abs = torch.abs(q_mean_complex)  # |E[q]|
    k_abs = torch.abs(k_complex)  # |k| for each position

    # extra = (E[|q|] - |E[q]|) * |k|
    extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs  # [seq_len, num_freq]

    # additive = Σ_f (extra_f * freq_scale_sq_f)
    additive = (extra * freq_scale_sq.view(1, -1)).sum(dim=1)  # [seq_len]

    return additive


def compute_full_scores(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    k_complex: torch.Tensor,
    omega: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute full scoring function for a query-key position pair.

    Formula:
        base_scores = Σ_f (amp_f * freq_scale_sq_f * cos(Δ * ω_f + φ_f))
        additive = Σ_f (extra_f * freq_scale_sq_f)
        combined = base_scores + additive

    Args:
        query_positions: [num_queries] absolute positions
        key_positions: [num_keys] absolute positions

    Returns:
        scores: [num_queries, num_keys] combined scores
    """
    num_queries = query_positions.shape[0]
    num_keys = key_positions.shape[0]
    num_freq = omega.shape[0]

    q_mean_abs = torch.abs(q_mean_complex)  # |E[q]|
    k_abs = torch.abs(k_complex)  # [num_keys, num_freq]

    # Amplitude: amp = |E[q]| * |k|
    amp = q_mean_abs.unsqueeze(0) * k_abs  # [num_keys, num_freq]

    # Phase: phi = angle(E[q] * conj(k))
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)  # [num_keys, num_freq]
    phi = torch.atan2(relative.imag, relative.real)  # [num_keys, num_freq]

    # Extra for additive term
    extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs  # [num_keys, num_freq]

    # Distance matrix: Δ = query_pos - key_pos
    # [num_queries, num_keys]
    delta = query_positions.unsqueeze(1).float() - key_positions.unsqueeze(0).float()

    # Phase matrix: [num_queries, num_keys, num_freq]
    phase = delta.unsqueeze(2) * omega.view(1, 1, -1) + phi.unsqueeze(0)
    cos_phase = torch.cos(phase)

    # Position-dependent term: [num_queries, num_keys]
    base_scores = (amp.unsqueeze(0) * freq_scale_sq.view(1, 1, -1) * cos_phase).sum(dim=2)

    # Norm-dependent term (additive): [num_keys] -> broadcast to [num_queries, num_keys]
    additive = (extra * freq_scale_sq.view(1, -1)).sum(dim=1)  # [num_keys]

    # Combined score
    combined = base_scores + additive.unsqueeze(0)

    return combined


def compute_score_heatmap(
    position_kernel: torch.Tensor,
    norm_scores: torch.Tensor,
    seq_len: int,
    patch_size: int,
    device: torch.device,
    temperature: float = 5.0,
) -> torch.Tensor:
    """
    Compute pooled score heatmap using kernel method for position-dependent term.

    Position-dependent: look up kernel[q - k]
    Norm-dependent: add norm_scores[k] per key position
    """
    num_q_groups = math.ceil(seq_len / patch_size)
    num_k_groups = math.ceil(seq_len / patch_size)

    pooled_groups = torch.zeros((num_q_groups, num_k_groups), device=device, dtype=torch.float32)

    for q_group in range(num_q_groups):
        q_start = q_group * patch_size
        q_end = min((q_group + 1) * patch_size, seq_len)

        for k_group in range(num_k_groups):
            k_start = k_group * patch_size
            k_end = min((k_group + 1) * patch_size, seq_len)

            # Create distance matrix for this patch
            q_indices = torch.arange(q_start, q_end, device=device)
            k_indices = torch.arange(k_start, k_end, device=device)

            # distances[i, j] = q_indices[i] - k_indices[j]
            distances = q_indices.unsqueeze(1) - k_indices.unsqueeze(0)

            # Causal mask: only keep distance >= 0
            causal_mask = distances >= 0

            # Get position-dependent values from kernel
            valid_distances = distances.clamp(min=0, max=len(position_kernel) - 1)
            position_values = position_kernel[valid_distances]

            # Add norm-dependent values (broadcast per key position)
            norm_values = norm_scores[k_start:k_end].unsqueeze(0).expand_as(position_values)
            combined_values = position_values + norm_values

            # Apply causal mask
            combined_values = torch.where(causal_mask, combined_values, torch.tensor(float('-inf'), device=device))

            # Max pool
            valid_values = combined_values[causal_mask]
            if len(valid_values) > 0:
                pooled_groups[q_group, k_group] = valid_values.max()
            else:
                pooled_groups[q_group, k_group] = float('-inf')

    # Per-row softmax with temperature
    pooled_groups = torch.softmax(pooled_groups / temperature, dim=1)

    # Per-row min-max normalization for visualization
    for row_idx in range(num_q_groups):
        row = pooled_groups[row_idx]
        row_min = row.min()
        row_max = row.max()
        denom = (row_max - row_min).clamp_min(1e-12)
        pooled_groups[row_idx] = (row - row_min) / denom

    return pooled_groups.detach().cpu()


def generate_figures(
    trace_dir: Path,
    model_path: Path,
    device: torch.device,
    patch_size: int,
    dpi: int,
    layer: int,
    head: int,
) -> None:
    """Generate both score visualizations."""
    mask_process_command("PD-L1_binder_l3h5_score")

    # Load data
    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])
    print(f"Token count: {token_count}")

    # Load model config for RoPE
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    config.rope_scaling = rope_scaling

    rotary = Qwen3RotaryEmbedding(config=config, device=device)
    inv_freq = rotary.inv_freq.to(torch.float64)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    head_dim = 128
    freq_scale_sq = compute_frequency_scaling(rotary, head_dim, torch.float32, device).pow(2)

    # Build RoPE tables
    position_ids = torch.arange(token_count, device=device).unsqueeze(0)
    base = torch.zeros(1, token_count, head_dim, device=device, dtype=torch.float32)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=torch.float32)
    sin_table = sin_table[0].to(dtype=torch.float32)

    # Load Q/K tensors
    print("Loading Q/K tensors...")
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
    k_tensor = data["k"]

    print(f"Processing Layer {layer}, Head {head}...")
    q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)
    k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)

    # Invert RoPE
    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

    # Compute score components
    print("Computing score components...")
    q_mean_complex, q_abs_mean, k_complex, k_mean_complex, omega = compute_score_components(
        q_orig, k_orig, inv_freq, freq_scale_sq, device
    )

    # ========== Figure 3: Norm-dependent strip ==========
    print("Computing norm-dependent scores...")
    norm_scores = compute_norm_dependent_scores(q_mean_complex, q_abs_mean, k_complex, freq_scale_sq)

    # Max pooling with same patch_size as attention map
    num_groups = math.ceil(token_count / patch_size)
    pooled_norm_scores = torch.zeros(num_groups, device=device, dtype=torch.float32)
    for g in range(num_groups):
        start = g * patch_size
        end = min((g + 1) * patch_size, token_count)
        pooled_norm_scores[g] = norm_scores[start:end].max()

    # Apply softmax with temperature for discrimination (like pseudo attention map)
    temperature = 5.0
    norm_scores_softmax = torch.softmax(pooled_norm_scores / temperature, dim=0)

    # Then min-max normalize to [0, 1] for visualization
    norm_scores_np = norm_scores_softmax.cpu().numpy()
    norm_min, norm_max = norm_scores_np.min(), norm_scores_np.max()
    norm_scores_normalized = (norm_scores_np - norm_min) / (norm_max - norm_min + 1e-12)

    # Create strip visualization
    fig3, ax3 = plt.subplots(figsize=(12, 1.5), dpi=dpi)

    # Reshape to 2D for imshow (1 row x seq_len columns)
    strip_data = norm_scores_normalized.reshape(1, -1)
    ax3.imshow(strip_data, cmap=attn_cmap_custom, aspect='auto', origin='upper')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel('Key Position', fontsize=FONT_SIZE * 1.5)
    for spine in ax3.spines.values():
        spine.set_visible(False)

    output_dir = Path("paper_visualizations/outputs/l3h5_attn_map")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig3.savefig(output_dir / f"fig_l{layer}h{head}_norm_score_strip.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: fig_l{layer}h{head}_norm_score_strip.png")

    # ========== Figure 4: Full score pseudo attention map ==========
    print("Computing position-dependent kernel...")
    position_kernel = compute_position_dependent_kernel(
        q_mean_complex, k_mean_complex, omega, freq_scale_sq,
        max_distance=token_count, device=device, gaussian_sigma=750.0
    )

    # Scale norm_scores relative to position_kernel
    # Check the magnitude ratio and adjust
    pos_mag = position_kernel.abs().mean().item()
    norm_mag = norm_scores.abs().mean().item()
    print(f"  Position kernel mean magnitude: {pos_mag:.4f}")
    print(f"  Norm scores mean magnitude: {norm_mag:.4f}")

    # Generate multiple versions with different norm weights
    norm_weights = [0.25, 0.3, 0.35, 0.4, 0.5]

    for norm_weight in norm_weights:
        norm_scale = (pos_mag / (norm_mag + 1e-12)) * norm_weight
        norm_scores_scaled = norm_scores * norm_scale
        print(f"  Generating with norm_weight={norm_weight}, scale={norm_scale:.4f}")

        with torch.no_grad():
            score_heatmap = compute_score_heatmap(
                position_kernel, norm_scores_scaled,
                token_count, patch_size, device, temperature=5.0
            )

        fig4, ax4 = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax4.imshow(score_heatmap.numpy(), cmap=attn_cmap_custom, aspect="equal", origin="upper")
        ax4.set_xticks([])
        ax4.set_yticks([])
        for spine in ax4.spines.values():
            spine.set_visible(False)
        ax4.set_title(f"L{layer}H{head} Score Map (norm={norm_weight})", fontsize=FONT_SIZE, fontweight='bold')

        fig4.savefig(output_dir / f"fig_l{layer}h{head}_score_map_norm{norm_weight}.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig4)
        print(f"Saved: fig_l{layer}h{head}_score_map_norm{norm_weight}.png")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    generate_figures(
        trace_dir=args.trace_dir,
        model_path=args.model_path,
        device=device,
        patch_size=args.patch_size,
        dpi=args.dpi,
        layer=args.layer,
        head=args.head,
    )


if __name__ == "__main__":
    main()
