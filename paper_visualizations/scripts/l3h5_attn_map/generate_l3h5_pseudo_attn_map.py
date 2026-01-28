"""Generate pseudo attention map using trigonometric reconstruction for Layer 3, Head 5.

The pseudo attention map uses the reconstructed kernel:
    reconstructed(Δ) = Σ_f |Q|_f |K|_f cos(ω_f Δ + φ_f)
where Δ = query_pos - key_pos.
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
color_dependent = (85 / 250, 104 / 250, 154 / 250)    # blue for dependent
face_color = (231 / 250, 231 / 250, 240 / 250)        # light gray-purple background

# Custom colormap: face_color (low attention) -> color_dependent (high attention)
attn_cmap_custom = LinearSegmentedColormap.from_list(
    "attn_custom", [face_color, color_dependent]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate L3H5 pseudo attention map")
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
        help="Pooling window size for attention maps (larger = coarser)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figure")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for figure",
    )
    parser.add_argument("--layer", type=int, default=3, help="Layer index")
    parser.add_argument("--head", type=int, default=5, help="Head index")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("--gaussian-var", type=float, default=0.0, help="Gaussian smoothing variance (0 to disable)")
    return parser.parse_args()


# ============ RoPE Utilities ============
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def invert_rope(rotated: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, scale: float) -> torch.Tensor:
    """Invert YaRN-scaled RoPE."""
    z = rotated / scale
    return z * cos - rotate_half(z) * sin


def to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to complex representation."""
    num_freq = tensor.shape[-1] // 2
    return torch.complex(tensor[..., :num_freq].float(), tensor[..., num_freq:].float())


def compute_reconstruction_kernel(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    inv_freq: torch.Tensor,
    max_distance: int,
    device: torch.device,
    length_multiplier: float = 2.0,
) -> torch.Tensor:
    """Compute reconstruction kernel for distances 0 to max_distance * length_multiplier.

    Uses the formula: kernel[d] = Σ_f |E[q]_f| * |E[k]_f| * cos(ω_f * d + φ_f)
    where φ_f = angle(E[q]_f) - angle(E[k]_f)

    The kernel is computed for a longer range (2x by default) to allow proper
    Gaussian smoothing without boundary effects.
    """
    head_dim = q_orig.shape[-1]
    num_freq = head_dim // 2

    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)

    # Mean over sequence dimension
    q_mean = q_complex.mean(dim=0)
    k_mean = k_complex.mean(dim=0)

    # Phase difference and amplitude product
    phi_f = torch.angle(q_mean) - torch.angle(k_mean)
    amplitude = torch.abs(q_mean) * torch.abs(k_mean)
    omega = inv_freq[:num_freq].to(device=device, dtype=torch.float32)

    # Compute kernel for extended range (2x length for proper smoothing)
    extended_length = int(max_distance * length_multiplier)
    distances = torch.arange(extended_length, device=device, dtype=torch.float32)
    phase_matrix = distances.unsqueeze(1) * omega.unsqueeze(0) + phi_f.unsqueeze(0)
    kernel = (torch.cos(phase_matrix) * amplitude.unsqueeze(0)).sum(dim=1)

    return kernel


def compute_pseudo_attention_heatmap(
    kernel: torch.Tensor,
    seq_len: int,
    patch_size: int,
    device: torch.device,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute pooled pseudo attention heatmap using reconstruction kernel.

    For each (i, j) where i >= j, the value is kernel[i - j].
    Per-row min-max normalization is applied before pooling to match attention behavior.
    """
    num_q_groups = math.ceil(seq_len / patch_size)
    num_k_groups = math.ceil(seq_len / patch_size)

    pooled_groups = torch.zeros((num_q_groups, num_k_groups), device=device, dtype=torch.float32)

    # Process each query group
    for q_group in range(num_q_groups):
        q_start = q_group * patch_size
        q_end = min((q_group + 1) * patch_size, seq_len)

        # For each query in this group, we need to normalize across all its visible keys
        # Then max pool within the group

        for k_group in range(num_k_groups):
            k_start = k_group * patch_size
            k_end = min((k_group + 1) * patch_size, seq_len)

            # Create distance matrix for this patch
            q_indices = torch.arange(q_start, q_end, device=device)
            k_indices = torch.arange(k_start, k_end, device=device)

            # distances[i, j] = q_indices[i] - k_indices[j]
            distances = q_indices.unsqueeze(1) - k_indices.unsqueeze(0)

            # Causal mask: only keep i >= j (distance >= 0)
            causal_mask = distances >= 0

            # Get kernel values for valid distances
            valid_distances = distances.clamp(min=0, max=len(kernel) - 1)
            patch_values = kernel[valid_distances]

            # Apply causal mask (set invalid to -inf for now)
            patch_values = torch.where(causal_mask, patch_values, torch.tensor(float('-inf'), device=device))

            # Max pool: take max over the patch (excluding -inf)
            valid_values = patch_values[causal_mask]
            if len(valid_values) > 0:
                pooled_groups[q_group, k_group] = valid_values.max()
            else:
                pooled_groups[q_group, k_group] = float('-inf')

    # Per-row softmax normalization with temperature
    # -inf values will become 0 after softmax (as exp(-inf) = 0)
    pooled_groups = torch.softmax(pooled_groups / temperature, dim=1)

    # Normalize to [0, 1] per row for visualization (softmax values can be very small)
    for row_idx in range(num_q_groups):
        row = pooled_groups[row_idx]
        row_min = row.min()
        row_max = row.max()
        denom = (row_max - row_min).clamp_min(1e-12)
        pooled_groups[row_idx] = (row - row_min) / denom

    return pooled_groups.detach().cpu()


def generate_figure(
    trace_dir: Path,
    model_path: Path,
    device: torch.device,
    patch_size: int,
    dpi: int,
    output_path: Path,
    layer: int,
    head: int,
    temperature: float = 1.0,
    gaussian_var: float = 0.0,
) -> None:
    """Generate the pseudo attention map figure."""
    mask_process_command("PD-L1_binder_l3h5_pseudo")

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

    # Invert RoPE to get original Q/K
    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

    # Compute reconstruction kernel
    print("Computing reconstruction kernel...")
    kernel = compute_reconstruction_kernel(q_orig, k_orig, inv_freq, token_count, device, length_multiplier=1.0)

    # Apply Gaussian smoothing if requested
    if gaussian_var > 0:
        sigma = math.sqrt(gaussian_var)
        print(f"Applying Gaussian smoothing (variance={gaussian_var}, sigma={sigma:.2f})...")
        kernel_np = kernel.cpu().numpy()
        kernel_np = gaussian_filter1d(kernel_np, sigma=sigma, mode='nearest')
        kernel = torch.from_numpy(kernel_np).to(device=device, dtype=torch.float32)

    # Compute pseudo attention heatmap
    print(f"Computing pseudo attention heatmap (temperature={temperature})...")
    with torch.no_grad():
        heatmap = compute_pseudo_attention_heatmap(kernel, token_count, patch_size, device, temperature)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    ax.imshow(heatmap.numpy(), cmap=attn_cmap_custom, aspect="equal", origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"L{layer}H{head} Pseudo Attention (T={temperature})", fontsize=FONT_SIZE, fontweight='bold')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


def main() -> None:
    args = parse_args()

    if args.output_path is None:
        output_dir = Path("paper_visualizations/outputs/l3h5_attn_map")
        output_path = output_dir / f"fig_l{args.layer}h{args.head}_pseudo_attn_map.png"
    else:
        output_path = args.output_path

    device = torch.device(args.device)

    generate_figure(
        trace_dir=args.trace_dir,
        model_path=args.model_path,
        device=device,
        patch_size=args.patch_size,
        dpi=args.dpi,
        output_path=output_path,
        layer=args.layer,
        head=args.head,
        temperature=args.temperature,
        gaussian_var=args.gaussian_var,
    )


if __name__ == "__main__":
    main()
