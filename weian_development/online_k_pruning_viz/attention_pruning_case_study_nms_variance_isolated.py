"""Isolated NMS Variance ablation study script.

验证 NMS Variance 算法的独立效果 (Verify NMS Variance algorithm's isolated effect)

This script is an isolated copy focused on **pure variance-aware NMS** ablation:
- 移除 keep_capacity / score-based 裁剪，只保留 NMS coverage_score 判定
- 每轮先加入本轮新 K，再在轮末执行一次 NMS（含增量 NMS）
- 与 docs/variance_aware_nms.md 的算法保持一致（epsilon=0, w_low/w_high 百分位）

Purpose:
  - Compare retention and drop-k metrics when nms_enabled=True vs nms_enabled=False under NMS-only flow
  - Use same input data (qid0003_trace34, hybrid_sample_heads_lowret_top10.json) for fair comparison
  - Ensure前置打分/TopK 不再干扰，凸显 NMS 本身的压缩效果

Implementation:
  - Uses variance-aware NMS with Q-magnitude percentile weights (w_low, w_high)
  - Conservative weight selection: positive score → w_low, negative score → w_high
  - No normalization (epsilon=0 design) - judgment based on sign only
  - See docs/variance_aware_nms.md for algorithm details

Comparison runs:
  1. Baseline: python script.py ... (nms_enabled=False by default)
  2. NMS enabled: python script.py ... --nms-enabled --low-percentile 50 --high-percentile 50

Expected metrics:
  - overall_retention: average per-head retention rate
  - nms_drop_rate: total_drops / total_rounds (NMS preprocessing drops)
  - per_head retention: individual head performance
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command
from weian_development.hf_offline_runner_sparse.round_pruning_utils import (
    invert_rope,
    to_complex_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize hybrid frequency scoring on sampled heads using cross-trace statistics.",
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Directory containing qid*_trace*/qk.pt (e.g., outputs/.../qk_bf16_traces).",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Trace folder name to process (e.g., qid0003_trace34).",
    )
    parser.add_argument(
        "--head-sample-file",
        type=Path,
        default=Path("weian_development/online_k_pruning_viz/hybrid_sample_heads_lowret_top10.json"),
        help="Path to JSON file storing sampled (layer, head) indices (generated if missing).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Number of heads to sample when creating a new sample file.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when sampling heads for a missing sample file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base directory for outputs (defaults to <input_root>/../attention_pruning_case_studies_hybrid_rounds_xtrace).",
    )
    parser.add_argument(
        "--stats-trace",
        type=Path,
        required=True,
        help="Trace directory (containing qk.pt) used to compute cross-trace frequency statistics.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=4096,
        help="Target pixel count (query/key) when inferring pooling window.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=32,
        help="Pooling window along query/key axes (overrides --target-size inference).",
    )
    parser.add_argument(
        "--q-tile",
        type=int,
        default=512,
        help="Number of queries to process per tile when computing attention logits.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for attention computation (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for recovering RoPE parameters.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Computation dtype for Q/K tensors (float32 suggested).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI when saving images.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(20.48, 10.24),
        help="Figure size (inches) for the side-by-side attention heatmaps.",
    )
    parser.add_argument(
        "--round-window",
        type=int,
        default=64,
        help="Number of decoded tokens per cache maintenance round.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log detailed progress information.",
    )
    # Variance-aware NMS arguments
    parser.add_argument(
        "--nms-enabled",
        action="store_true",
        default=False,
        help="Enable variance-aware Q-magnitude weighted NMS (applied at each round end).",
    )
    parser.add_argument(
        "--low-percentile",
        type=float,
        default=20.0,
        help="Low percentile for Q-magnitude weights (default: 20.0).",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=80.0,
        help="High percentile for Q-magnitude weights (default: 80.0).",
    )
    return parser.parse_args()


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_or_create_sample(
    sample_file: Path,
    sample_count: int,
    seed: int,
    layer_count: int,
    head_count: int,
) -> List[Tuple[int, int]]:
    if sample_file.exists():
        with sample_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        sample = [(int(item[0]), int(item[1])) for item in data]
        return sample

    total_heads = layer_count * head_count
    if sample_count > total_heads:
        raise ValueError(
            f"Sample count {sample_count} exceeds total available heads {total_heads}"
        )

    indices = [(layer, head) for layer in range(layer_count) for head in range(head_count)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(len(indices), generator=generator)
    selected = [indices[idx] for idx in perm[:sample_count].tolist()]

    sample_file.parent.mkdir(parents=True, exist_ok=True)
    with sample_file.open("w", encoding="utf-8") as f:
        json.dump([[layer, head] for layer, head in selected], f, indent=2)

    return selected


def resolve_patch_size(seq_len: int, target_size: int, patch_arg: int | None) -> int:
    if patch_arg and patch_arg > 0:
        return patch_arg
    if seq_len <= target_size:
        return 1
    return math.ceil(seq_len / target_size)


def build_rotary(cache_device: torch.device, model_path: Path, dtype: torch.dtype) -> Qwen3RotaryEmbedding:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    config.rope_scaling = rope_scaling
    rotary = Qwen3RotaryEmbedding(config=config, device=cache_device)
    rotary.to(dtype=dtype)
    return rotary


def compute_rotary_tables(
    rotary: Qwen3RotaryEmbedding,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    base = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0]
    sin_table = sin_table[0]
    inv_freq = rotary.inv_freq.to(device=device, dtype=torch.float64)
    return cos_table, sin_table, inv_freq


def invert_qk(
    q_head: torch.Tensor,
    k_head: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_unrot = invert_rope(q_head, cos_table, sin_table, attention_scale)
    k_unrot = invert_rope(k_head, cos_table, sin_table, attention_scale)
    return q_unrot, k_unrot


def compute_frequency_statistics_from_means(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    k_unrot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_complex = to_complex_pairs(k_unrot)
    q_mean_abs = torch.abs(q_mean_complex)
    k_abs = torch.abs(k_complex)
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    phi = torch.atan2(relative.imag, relative.real)
    amp = q_mean_abs.unsqueeze(0) * k_abs
    extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
    return amp, phi, extra


def build_geometric_offsets(max_length: int, device: torch.device) -> torch.Tensor:
    if max_length < 1:
        raise ValueError("offset_max_length must be >= 1")
    offsets: List[float] = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def compute_frequency_energy_weights(
    q_complex: torch.Tensor,
    k_complex: torch.Tensor,
    method: str,
    device: torch.device,
) -> torch.Tensor:
    """Compute frequency band energy weights for spectrum-aware NMS.

    Args:
        q_complex: [seq_len, freq_count] complex tensor (unrotated for amplitude, rotated for causal)
        k_complex: [seq_len, freq_count] complex tensor (unrotated for amplitude, rotated for causal)
        method: 'amplitude' or 'causal'
        device: target device

    Returns:
        freq_weights: [freq_count] tensor normalized to sum=1
    """
    q_complex = q_complex.to(device=device)
    k_complex = k_complex.to(device=device)

    if method == "amplitude":
        # Method A: E_f = E[|q_f| * |k_f|]
        q_abs = torch.abs(q_complex)  # [seq_len, freq_count]
        k_abs = torch.abs(k_complex)  # [seq_len, freq_count]
        energy_per_freq = (q_abs * k_abs).mean(dim=0)  # [freq_count]
    elif method == "meanvec":
        # Method C: E_f = |E[q_f]| * |E[k_f]|
        # First compute mean complex vector, then take magnitude product
        # Reference: freq_magnitude_single_plot_meanvec_scatter.py mean_vector_product()
        q_mean = q_complex.mean(dim=0)  # [freq_count] complex
        k_mean = k_complex.mean(dim=0)  # [freq_count] complex
        q_mean_abs = torch.abs(q_mean)  # [freq_count]
        k_mean_abs = torch.abs(k_mean)  # [freq_count]
        energy_per_freq = q_mean_abs * k_mean_abs  # [freq_count]
    elif method == "causal":
        # Method B: Causal attention weighted (using rotated Q/K)
        # Memory-efficient chunked computation to avoid O(n^2 * F) tensor
        seq_len = q_complex.shape[0]
        freq_count = q_complex.shape[1]
        k_conj = k_complex.conj()

        # Accumulate energy per frequency using chunked causal computation
        energy_per_freq = torch.zeros(freq_count, device=device, dtype=torch.float32)
        chunk_size = min(256, seq_len)  # Process in chunks to save memory

        for i_start in range(0, seq_len, chunk_size):
            i_end = min(i_start + chunk_size, seq_len)
            q_chunk = q_complex[i_start:i_end]  # [chunk, F]
            # For causal: only sum over j <= i
            # attn_contrib[i, j, f] = Real(q_i^f * conj(k_j^f)) for j <= i
            for i_local, i_global in enumerate(range(i_start, i_end)):
                # Only consider k[0:i_global+1] for query at i_global
                q_i = q_chunk[i_local]  # [F]
                k_causal = k_conj[:i_global + 1]  # [i_global+1, F]
                # Dot product: sum over positions for each frequency
                attn_contrib = (q_i.unsqueeze(0) * k_causal).real.sum(dim=0)  # [F]
                energy_per_freq += attn_contrib

        # Normalize by number of valid pairs: n*(n+1)/2
        num_valid_pairs = seq_len * (seq_len + 1) / 2
        energy_per_freq = energy_per_freq / num_valid_pairs  # [freq_count]
    else:
        raise ValueError(f"Unknown energy method: {method}")

    # Normalize to sum=1
    # Per document 4.2: keep signed values, do NOT take abs()
    energy_per_freq = energy_per_freq.to(dtype=torch.float32)
    total = energy_per_freq.sum()
    if torch.abs(total) > 1e-8:
        freq_weights = energy_per_freq / total
    else:
        # Fallback to uniform weights if sum is near zero (energies cancel out)
        freq_weights = torch.ones_like(energy_per_freq) / energy_per_freq.numel()

    return freq_weights


def fast_parallel_nms(
    k_complex: torch.Tensor,
    freq_weights: torch.Tensor,
) -> torch.Tensor:
    """Fast Parallel NMS based on projection coverage.

    coverage_score(A, B) = sum_f w_f * (Real(A_f * conj(B_f)) / |B_f| - |B_f|)
    A suppresses B when coverage_score(A, B) > 0 (epsilon fixed at 0)

    Args:
        k_complex: [N, F] tensor of RoPE-rotated K converted to complex pairs
        freq_weights: [F] tensor of frequency band weights (sum=1)

    Returns:
        keep_mask: [N] bool tensor, True = keep
    """
    N, F = k_complex.shape
    device = k_complex.device

    # 1. Compute per-K per-frequency magnitude
    k_abs = torch.abs(k_complex)  # [N, F]
    k_abs_safe = k_abs.clamp(min=1e-5)  # Avoid division by zero

    # 2. Compute A's projection onto B direction (per-frequency)
    #    proj[a, b, f] = Real(k_a^f * conj(k_b^f)) / |k_b^f|
    #                  = |k_a^f| * cos(theta_ab^f)
    real_dot = torch.einsum("af,bf->abf", k_complex, k_complex.conj()).real  # [N, N, F]
    proj_on_b = real_dot / k_abs_safe.unsqueeze(0)  # [N, N, F]: proj[a,b,f]

    # 3. Per-frequency coverage score: proj - |B|
    #    score[a, b, f] = proj[a, b, f] - |k_b^f|
    per_freq_score = proj_on_b - k_abs.unsqueeze(0)  # [N, N, F]

    # 4. Weighted sum over frequencies
    #    total_score[a, b] = sum_f w_f * score[a, b, f]
    coverage_score = (per_freq_score * freq_weights.view(1, 1, -1)).sum(dim=2)  # [N, N]

    # 5. A suppresses B when coverage_score[a, b] > 0 (epsilon=0 fixed)
    suppresses = coverage_score > 0  # [N, N]

    # 6. Exclude self-suppression (diagonal)
    suppresses.fill_diagonal_(False)

    # 7. B is suppressed if any A suppresses it
    is_suppressed = suppresses.any(dim=0)  # [N]
    keep_mask = ~is_suppressed

    return keep_mask


def compute_q_magnitude_percentile_weights(
    q_complex: torch.Tensor,
    low_percentile: float = 20.0,
    high_percentile: float = 80.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Q-magnitude percentile weights for variance-aware NMS.

    Computes the low and high percentile of |Q| per frequency from stats_trace.
    No normalization needed since epsilon=0 makes judgment based on sign only.

    IMPORTANT: Conservative weight selection requires low_percentile <= high_percentile.
    See docs/variance_aware_nms.md Section 6.5 for details:
    - positive score → use w_low (underestimate contribution)
    - negative score → use w_high (amplify resistance)
    This only works correctly when w_low <= w_high.

    Args:
        q_complex: [num_samples, freq_count] Q vectors in complex form (unrotated)
        low_percentile: percentile for positive-score frequencies (default 20)
        high_percentile: percentile for negative-score frequencies (default 80)

    Returns:
        w_low: [freq_count] low percentile Q-magnitude weights
        w_high: [freq_count] high percentile Q-magnitude weights

    Raises:
        ValueError: if low_percentile > high_percentile (violates conservative principle)
    """
    if low_percentile > high_percentile:
        raise ValueError(
            f"low_percentile ({low_percentile}) must be <= high_percentile ({high_percentile}). "
            f"Violating this breaks the conservative weight selection principle. "
            f"See docs/variance_aware_nms.md Section 6.5."
        )
    q_magnitudes = torch.abs(q_complex)  # [num_samples, freq_count]
    w_low = torch.quantile(q_magnitudes, low_percentile / 100.0, dim=0)
    w_high = torch.quantile(q_magnitudes, high_percentile / 100.0, dim=0)
    return w_low, w_high


def variance_aware_fast_nms(
    k_complex: torch.Tensor,
    w_low: torch.Tensor,
    w_high: torch.Tensor,
) -> torch.Tensor:
    """Variance-aware Fast Parallel NMS with conservative weight selection.

    Uses Q-magnitude percentile weights for conservative weight selection:
    - Positive per_freq_score: use w_low (assume Q magnitude is low)
    - Negative per_freq_score: use w_high (assume Q magnitude is high)

    A suppresses B when conservative_coverage_score(A, B) > 0

    Args:
        k_complex: [N, F] tensor of RoPE-rotated K converted to complex pairs
        w_low: [F] tensor of low percentile Q-magnitude weights
        w_high: [F] tensor of high percentile Q-magnitude weights

    Returns:
        keep_mask: [N] bool tensor, True = keep
    """
    N, F = k_complex.shape

    # 1. Compute per-K per-frequency magnitude
    k_abs = torch.abs(k_complex)  # [N, F]
    k_abs_safe = k_abs.clamp(min=1e-5)

    # 2. Compute A's projection onto B direction (per-frequency)
    real_dot = torch.einsum("af,bf->abf", k_complex, k_complex.conj()).real  # [N, N, F]
    proj_on_b = real_dot / k_abs_safe.unsqueeze(0)  # [N, N, F]

    # 3. Per-frequency coverage score: proj - |B|
    per_freq_score = proj_on_b - k_abs.unsqueeze(0)  # [N, N, F]

    # 4. Conservative weight selection based on score sign
    #    positive score (helps suppression) → use w_low (underestimate contribution)
    #    negative score (opposes suppression) → use w_high (amplify resistance)
    weights = torch.where(
        per_freq_score > 0,
        w_low.view(1, 1, -1),
        w_high.view(1, 1, -1),
    )  # [N, N, F]

    # 5. Weighted sum for conservative coverage score
    conservative_score = (per_freq_score * weights).sum(dim=2)  # [N, N]

    # 6. A suppresses B when conservative_score > 0 (epsilon=0)
    suppresses = conservative_score > 0  # [N, N]
    suppresses.fill_diagonal_(False)

    # 7. B is suppressed if any A suppresses it
    is_suppressed = suppresses.any(dim=0)
    keep_mask = ~is_suppressed

    return keep_mask


def incremental_variance_aware_nms(
    historical_k_complex: torch.Tensor,
    new_k_complex: torch.Tensor,
    w_low: torch.Tensor,
    w_high: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Incremental NMS: only compute interactions involving new keys.

    Instead of O((H+N)² × F) for full NMS, this computes:
    - Historical K suppresses New K: H × N × F
    - New K suppresses Historical K: N × H × F
    - New K suppresses New K: N × N × F

    Total: O((H×N + N²) × F) - typically 33x faster.

    Args:
        historical_k_complex: [H, F] tensor of existing cache K (already NMS-processed)
        new_k_complex: [N, F] tensor of new K from this round
        w_low: [F] tensor of low percentile Q-magnitude weights
        w_high: [F] tensor of high percentile Q-magnitude weights

    Returns:
        historical_keep_mask: [H] bool tensor, True = keep historical key
        new_keep_mask: [N] bool tensor, True = keep new key
    """
    H, F = historical_k_complex.shape
    N = new_k_complex.shape[0]

    # Magnitudes
    hist_abs = torch.abs(historical_k_complex).clamp(min=1e-5)  # [H, F]
    new_abs = torch.abs(new_k_complex).clamp(min=1e-5)  # [N, F]

    # --- Part 1: Historical K suppresses New K (H×N) ---
    # For each (historical A, new B), check if A suppresses B
    real_dot_hist_new = torch.einsum("hf,nf->hnf", historical_k_complex, new_k_complex.conj()).real
    proj_on_new = real_dot_hist_new / new_abs.unsqueeze(0)  # [H, N, F]
    per_freq_score_hn = proj_on_new - new_abs.unsqueeze(0)  # [H, N, F]
    weights_hn = torch.where(per_freq_score_hn > 0, w_low.view(1, 1, -1), w_high.view(1, 1, -1))
    conservative_score_hn = (per_freq_score_hn * weights_hn).sum(dim=2)  # [H, N]
    hist_suppresses_new = conservative_score_hn > 0  # [H, N]
    new_suppressed_by_hist = hist_suppresses_new.any(dim=0)  # [N]

    # --- Part 2: New K suppresses Historical K (N×H) ---
    real_dot_new_hist = torch.einsum("nf,hf->nhf", new_k_complex, historical_k_complex.conj()).real
    proj_on_hist = real_dot_new_hist / hist_abs.unsqueeze(0)  # [N, H, F]
    per_freq_score_nh = proj_on_hist - hist_abs.unsqueeze(0)  # [N, H, F]
    weights_nh = torch.where(per_freq_score_nh > 0, w_low.view(1, 1, -1), w_high.view(1, 1, -1))
    conservative_score_nh = (per_freq_score_nh * weights_nh).sum(dim=2)  # [N, H]
    new_suppresses_hist = conservative_score_nh > 0  # [N, H]
    hist_suppressed_by_new = new_suppresses_hist.any(dim=0)  # [H]

    # --- Part 3: New K suppresses New K (N×N) ---
    real_dot_new_new = torch.einsum("af,bf->abf", new_k_complex, new_k_complex.conj()).real
    proj_new_new = real_dot_new_new / new_abs.unsqueeze(0)  # [N, N, F]
    per_freq_score_nn = proj_new_new - new_abs.unsqueeze(0)  # [N, N, F]
    weights_nn = torch.where(per_freq_score_nn > 0, w_low.view(1, 1, -1), w_high.view(1, 1, -1))
    conservative_score_nn = (per_freq_score_nn * weights_nn).sum(dim=2)  # [N, N]
    new_suppresses_new = conservative_score_nn > 0  # [N, N]
    new_suppresses_new.fill_diagonal_(False)  # Don't self-suppress
    new_suppressed_by_new = new_suppresses_new.any(dim=0)  # [N]

    # --- Combine results ---
    historical_keep_mask = ~hist_suppressed_by_new  # [H]
    new_keep_mask = ~(new_suppressed_by_hist | new_suppressed_by_new)  # [N]

    return historical_keep_mask, new_keep_mask


def score_keys_for_round(
    key_indices: torch.Tensor,
    round_start: int,
    amp: torch.Tensor,
    phi: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
    offsets: torch.Tensor,
    aggregation: str,
) -> torch.Tensor:
    if key_indices.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    base_delta = (
        round_start
        - key_indices.to(device=amp.device, dtype=torch.float32)
    )
    delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)

    amp_sel = amp.index_select(0, key_indices)
    phi_sel = phi.index_select(0, key_indices)
    extra_sel = extra.index_select(0, key_indices)

    phase = delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi_sel.unsqueeze(1)
    cos_phase = torch.cos(phase)
    base_scores = (amp_sel.unsqueeze(1) * cos_phase).sum(dim=2)
    additive = extra_sel.sum(dim=1, keepdim=True)
    combined = base_scores + additive

    if aggregation == "mean":
        return combined.mean(dim=1)
    return combined.max(dim=1).values


def simulate_round_pruning(
    seq_len: int,
    round_window: int,
    device: torch.device,
    nms_enabled: bool = False,
    w_low: torch.Tensor | None = None,
    w_high: torch.Tensor | None = None,
    k_rotated_complex: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, List[int]]:
    """仅使用 variance-aware NMS 的 round 级裁剪模拟。"""
    if round_window < 1:
        raise ValueError("round_window must be >= 1")

    prune_mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)
    current_cache: List[int] = []
    nms_drop_counts: List[int] = []
    nms_processed_indices: set = set()

    for q_idx in range(seq_len):
        current_cache.append(q_idx)

        is_round_end = ((q_idx + 1) % round_window == 0) or (q_idx == seq_len - 1)
        if (
            is_round_end
            and nms_enabled
            and len(current_cache) > 1
            and w_low is not None
            and w_high is not None
            and k_rotated_complex is not None
        ):
            historical_indices = [idx for idx in current_cache if idx in nms_processed_indices]
            new_indices = [idx for idx in current_cache if idx not in nms_processed_indices]

            drop_count = 0
            if len(historical_indices) == 0:
                if len(new_indices) > 1:
                    new_tensor = torch.tensor(new_indices, device=device, dtype=torch.long)
                    new_k_complex = k_rotated_complex[new_tensor]
                    keep_mask = variance_aware_fast_nms(new_k_complex, w_low, w_high)
                    drop_count = (~keep_mask).sum().item()
                    surviving_new = new_tensor[keep_mask].tolist()
                else:
                    surviving_new = new_indices
                current_cache = surviving_new
            elif len(new_indices) == 0:
                pass
            else:
                hist_tensor = torch.tensor(historical_indices, device=device, dtype=torch.long)
                new_tensor = torch.tensor(new_indices, device=device, dtype=torch.long)
                hist_k_complex = k_rotated_complex[hist_tensor]
                new_k_complex = k_rotated_complex[new_tensor]

                hist_keep, new_keep = incremental_variance_aware_nms(
                    hist_k_complex, new_k_complex, w_low, w_high
                )

                hist_drops = (~hist_keep).sum().item()
                new_drops = (~new_keep).sum().item()
                drop_count = hist_drops + new_drops

                surviving_hist = hist_tensor[hist_keep].tolist()
                surviving_new = new_tensor[new_keep].tolist()
                current_cache = surviving_hist + surviving_new
                current_cache.sort()

            nms_drop_counts.append(int(drop_count))
            nms_processed_indices = set(current_cache)
        elif is_round_end and nms_enabled:
            nms_drop_counts.append(0)

        allowed_keys = [idx for idx in current_cache if idx <= q_idx]
        if allowed_keys:
            idx_tensor = torch.tensor(
                allowed_keys,
                device=device,
                dtype=torch.long,
            )
            prune_mask[q_idx, idx_tensor] = True

        prune_mask[q_idx, q_idx] = True

    return prune_mask, nms_drop_counts


def compute_pooled_attention(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    patch_size: int,
    q_tile: int,
    device: torch.device,
    prune_mask: torch.Tensor | None = None,
    return_argmax: bool = False,
    return_query_argmax: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    head_count, seq_q, head_dim = q_block.shape
    _, seq_k, _ = k_block.shape
    scale = head_dim ** -0.5

    num_q_groups = math.ceil(seq_len / patch_size)
    num_k_groups = math.ceil(seq_len / patch_size)

    q_pad = num_q_groups * patch_size - seq_len
    k_pad = num_k_groups * patch_size - seq_len
    if q_pad > 0:
        pad = torch.zeros(head_count, q_pad, head_dim, device=device, dtype=q_block.dtype)
        q_block = torch.cat([q_block, pad], dim=1)
    if k_pad > 0:
        pad = torch.zeros(head_count, k_pad, head_dim, device=device, dtype=k_block.dtype)
        k_block = torch.cat([k_block, pad], dim=1)

    seq_q_real = seq_len
    seq_k_padded = k_block.shape[1]

    key_positions = torch.arange(seq_k_padded, device=device)
    key_valid = key_positions < seq_len

    pooled_groups = torch.zeros(
        (head_count, num_q_groups, num_k_groups),
        device=device,
        dtype=torch.float32,
    )

    argmax_groups = None
    if return_argmax:
        argmax_groups = torch.zeros_like(pooled_groups)

    query_argmax = None
    if return_query_argmax:
        query_argmax = torch.full(
            (head_count, seq_len),
            -1,
            device=device,
            dtype=torch.long,
        )

    if prune_mask is not None and prune_mask.shape[1] < seq_k_padded:
        pad = torch.zeros(
            prune_mask.shape[0],
            seq_k_padded - prune_mask.shape[1],
            device=device,
            dtype=prune_mask.dtype,
        )
        prune_mask = torch.cat([prune_mask, pad], dim=1)

    k_t = k_block.transpose(1, 2).contiguous()

    for q_start in range(0, seq_q_real, q_tile):
        q_end = min(q_start + q_tile, seq_q_real)
        indices = torch.arange(q_start, q_end, device=device)
        q_slice = q_block[:, q_start:q_end, :]

        scores = torch.matmul(q_slice, k_t) * scale
        scores = scores.to(torch.float32)

        future_mask = key_positions.unsqueeze(0) > indices.unsqueeze(1)
        valid_mask = (~future_mask).unsqueeze(0) & key_valid.view(1, 1, -1)

        scores = scores.masked_fill(~valid_mask, float("-inf"))

        if prune_mask is not None:
            prune_slice = prune_mask[q_start:q_end]
            prune_slice = prune_slice.unsqueeze(0)
            scores = scores.masked_fill(~prune_slice, float("-inf"))

        scores_flat = scores.view(head_count, -1, num_k_groups * patch_size)
        row_max = scores_flat.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

        stable = torch.exp(scores_flat - row_max)
        row_sum = stable.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = (stable / row_sum).view(head_count, -1, num_k_groups, patch_size)

        key_mask = key_valid.view(1, 1, num_k_groups, patch_size)
        weights = weights * key_mask

        pooled_k = weights.max(dim=-1).values

        if return_argmax:
            weights_flat = weights.view(head_count, -1, num_k_groups * patch_size)
            top_indices = weights_flat.argmax(dim=-1, keepdim=True)
            top_mask = torch.zeros_like(weights_flat)
            top_mask.scatter_(dim=-1, index=top_indices, value=1.0)
            top_mask = top_mask.view(head_count, -1, num_k_groups, patch_size)
            top_mask = top_mask * key_mask
            argmax_tile = top_mask.max(dim=-1).values

        if return_query_argmax:
            finite = torch.isfinite(scores)
            has_valid = finite.any(dim=-1)
            local_argmax = scores.argmax(dim=-1)
            local_argmax = torch.where(
                has_valid,
                local_argmax,
                torch.zeros_like(local_argmax),
            )
            tile_len = q_end - q_start
            query_argmax[:, q_start:q_end] = local_argmax[:, :tile_len]

        query_groups = indices // patch_size
        base_group = int(query_groups.min().item())
        local_groups = (query_groups - base_group).to(torch.int64)
        groups_in_tile = int(local_groups.max().item()) + 1

        expanded_index = local_groups.view(1, -1, 1).expand(head_count, -1, num_k_groups)
        tile_max = torch.zeros(
            (head_count, groups_in_tile, num_k_groups),
            device=device,
            dtype=torch.float32,
        )
        tile_max.scatter_reduce_(
            dim=1,
            index=expanded_index,
            src=pooled_k,
            reduce="amax",
        )

        end_group = base_group + groups_in_tile
        pooled_groups[:, base_group:end_group] = torch.maximum(
            pooled_groups[:, base_group:end_group], tile_max
        )

        if return_argmax:
            argmax_tile_groups = torch.zeros(
                (head_count, groups_in_tile, num_k_groups),
                device=device,
                dtype=torch.float32,
            )
            argmax_tile_groups.scatter_reduce_(
                dim=1,
                index=expanded_index,
                src=argmax_tile,
                reduce="amax",
            )
            argmax_groups[:, base_group:end_group] = torch.maximum(
                argmax_groups[:, base_group:end_group], argmax_tile_groups
            )

    valid_query_groups = math.ceil(seq_len / patch_size)
    valid_key_groups = math.ceil(seq_len / patch_size)
    if valid_query_groups < num_q_groups:
        pooled_groups[:, valid_query_groups:, :] = 0.0
    if valid_key_groups < num_k_groups:
        pooled_groups[:, :, valid_key_groups:] = 0.0

    if return_argmax and argmax_groups is not None:
        if valid_query_groups < num_q_groups:
            argmax_groups[:, valid_query_groups:, :] = 0.0
        if valid_key_groups < num_k_groups:
            argmax_groups[:, :, valid_key_groups:] = 0.0

    row_min = pooled_groups.amin(dim=2, keepdim=True)
    row_max = pooled_groups.amax(dim=2, keepdim=True)
    denom = (row_max - row_min).clamp_min(1e-12)
    norm = torch.clamp((pooled_groups - row_min) / denom, 0.0, 1.0)
    return norm, argmax_groups, query_argmax


def save_comparison_figure(
    baseline: torch.Tensor,
    pruned: torch.Tensor,
    out_path: Path,
    cmap: str,
    figsize: Tuple[float, float],
    dpi: int,
    title: str,
) -> None:
    baseline_np = baseline.squeeze(0).cpu().numpy()
    pruned_np = pruned.squeeze(0).cpu().numpy()

    vmin = min(baseline_np.min(), pruned_np.min())
    vmax = max(baseline_np.max(), pruned_np.max())

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=dpi,
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )
    ax_base, ax_prune = axes

    im0 = ax_base.imshow(baseline_np, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    ax_base.set_title("Baseline attention")
    ax_base.set_xlabel("Key group index")
    ax_base.set_ylabel("Query group index")

    im1 = ax_prune.imshow(pruned_np, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    ax_prune.set_title("Pruned attention")
    ax_prune.set_xlabel("Key group index")

    cbar = fig.colorbar(im1, ax=axes.tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("Normalized pooled weight")

    fig.suptitle(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_layer_retention_plot(
    per_layer: Dict[int, float],
    out_path: Path,
    dpi: int,
) -> None:
    if not per_layer:
        return

    layers = sorted(per_layer.keys())
    values = [per_layer[layer] for layer in layers]

    fig, ax = plt.subplots(figsize=(12.0, 6.0), dpi=dpi)
    ax.bar(layers, values, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean retention")
    ax.set_title("Per-layer baseline argmax retention")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_pruning_case")

    trace_dir = args.input_root / args.trace
    if not trace_dir.exists():
        raise SystemExit(f"Trace directory not found: {trace_dir}")

    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise SystemExit(f"Missing qk assets in {trace_dir}")

    output_root = args.output_root
    if output_root is None:
        output_root = (
            args.input_root.parent / "attention_pruning_case_studies_hybrid_rounds_xtrace"
        )
    config_dir = f"nms_only_w{args.round_window}"
    if args.nms_enabled:
        config_dir += f"_p{int(args.low_percentile)}_{int(args.high_percentile)}"
    output_root = output_root / trace_dir.name / config_dir
    output_root.mkdir(parents=True, exist_ok=True)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    seq_len = int(meta["sequence_length"])

    patch_size = resolve_patch_size(seq_len, args.target_size, args.patch_size)

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    data = torch.load(qk_path, map_location="cpu")
    q_tensor: torch.Tensor = data["q"]
    k_tensor: torch.Tensor = data["k"]

    layer_count = q_tensor.shape[0]
    head_per_layer = q_tensor.shape[1]

    stats_trace_dir = args.stats_trace
    if not stats_trace_dir.is_absolute():
        stats_trace_dir = (Path.cwd() / stats_trace_dir).resolve()
    if not stats_trace_dir.exists():
        raise SystemExit(f"Stats trace directory not found: {stats_trace_dir}")

    stats_qk_path = stats_trace_dir / "qk.pt"
    stats_meta_path = stats_trace_dir / "metadata.json"
    if not stats_qk_path.exists() or not stats_meta_path.exists():
        raise SystemExit(f"Missing qk assets in stats trace {stats_trace_dir}")

    with stats_meta_path.open("r", encoding="utf-8") as f:
        stats_meta = json.load(f)
    stats_seq_len = int(stats_meta["sequence_length"])

    stats_data = torch.load(stats_qk_path, map_location="cpu")
    stats_q_tensor: torch.Tensor = stats_data["q"]
    stats_k_tensor: torch.Tensor = stats_data["k"]
    if stats_q_tensor.shape[0] != layer_count or stats_q_tensor.shape[1] != head_per_layer:
        raise SystemExit("Stats trace dimensions do not match primary trace dimensions")

    sampled_heads = load_or_create_sample(
        args.head_sample_file,
        args.sample_count,
        args.sample_seed,
        layer_count,
        head_per_layer,
    )

    sampled_heads = sorted(sampled_heads)
    if args.verbose:
        print(
            f"Using {len(sampled_heads)} sampled heads from {args.head_sample_file}"  # noqa: E501
        )

    retention_records: List[Tuple[int, int, float]] = []
    per_layer_rates: Dict[int, List[float]] = defaultdict(list)
    all_nms_drop_counts: List[List[int]] = []  # Track NMS drops per head

    rotary = build_rotary(device, args.model_path, dtype)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    head_dim = q_tensor.shape[-1]
    stats_cos_table, stats_sin_table, _ = compute_rotary_tables(
        rotary, stats_seq_len, head_dim, dtype, device
    )

    layer_to_heads_map: Dict[int, List[int]] = defaultdict(list)
    for layer, head in sampled_heads:
        layer_to_heads_map[layer].append(head)

    q_magnitude_weights_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    for layer, heads in layer_to_heads_map.items():
        stats_q_layer = stats_q_tensor[layer].to(device=device, dtype=dtype)
        stats_k_layer = stats_k_tensor[layer].to(device=device, dtype=dtype)
        for head in heads:
            stats_q_head = stats_q_layer[head, :stats_seq_len, :].contiguous()
            stats_k_head = stats_k_layer[head, :stats_seq_len, :].contiguous()
            stats_unrot = invert_rope(
                stats_q_head,
                stats_cos_table,
                stats_sin_table,
                attention_scale,
            )
            stats_complex = to_complex_pairs(stats_unrot)

            # Pre-compute Q-magnitude percentile weights from stats_trace if NMS enabled
            # Uses unrotated Q in complex form (stats_complex already computed above)
            if args.nms_enabled:
                w_low, w_high = compute_q_magnitude_percentile_weights(
                    stats_complex,
                    args.low_percentile,
                    args.high_percentile,
                )
                q_magnitude_weights_cache[(layer, head)] = (
                    w_low.detach().cpu(),
                    w_high.detach().cpu(),
                )

        del stats_q_layer, stats_k_layer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    current_layer = None
    q_layer = None
    k_layer = None

    for layer, head in sampled_heads:
        if layer >= layer_count or head >= head_per_layer:
            raise IndexError(
                f"Sampled head ({layer}, {head}) exceeds tensor dimensions"
            )

        if layer != current_layer:
            q_layer = q_tensor[layer].to(device=device, dtype=dtype)
            k_layer = k_tensor[layer].to(device=device, dtype=dtype)
            current_layer = layer
            if args.verbose:
                print(f"Loaded layer {layer} to device")

        if args.verbose:
            print(f"Processing layer {layer}, head {head}")

        q_head = q_layer[head, :seq_len, :].contiguous()
        k_head = k_layer[head, :seq_len, :].contiguous()

        # 准备 NMS 所需的 K（RoPE 后的复数）和权重
        w_low = None
        w_high = None
        k_rotated_complex = None
        if args.nms_enabled:
            k_rotated_complex = to_complex_pairs(k_head)
            w_low, w_high = q_magnitude_weights_cache[(layer, head)]
            w_low = w_low.to(device=device)
            w_high = w_high.to(device=device)

        prune_mask, nms_drop_counts = simulate_round_pruning(
            seq_len,
            args.round_window,
            device,
            nms_enabled=args.nms_enabled,
            w_low=w_low,
            w_high=w_high,
            k_rotated_complex=k_rotated_complex,
        )
        if args.nms_enabled:
            all_nms_drop_counts.append(nms_drop_counts)

        q_block = q_head.unsqueeze(0)
        k_block = k_head.unsqueeze(0)

        baseline_heatmap, baseline_argmax, baseline_query_argmax = compute_pooled_attention(
            q_block,
            k_block,
            seq_len,
            patch_size,
            args.q_tile,
            device,
            prune_mask=None,
            return_argmax=True,
            return_query_argmax=True,
        )
        baseline_heatmap = baseline_heatmap.detach().cpu()
        baseline_argmax = baseline_argmax.detach().cpu() if baseline_argmax is not None else None
        baseline_query_argmax = (
            baseline_query_argmax.detach().cpu()
            if baseline_query_argmax is not None
            else None
        )

        pruned_heatmap, pruned_argmax, _ = compute_pooled_attention(
            q_block,
            k_block,
            seq_len,
            patch_size,
            args.q_tile,
            device,
            prune_mask=prune_mask,
            return_argmax=True,
        )
        pruned_heatmap = pruned_heatmap.detach().cpu()
        pruned_argmax = pruned_argmax.detach().cpu() if pruned_argmax is not None else None

        hit_rate = None
        if baseline_query_argmax is not None:
            indices = baseline_query_argmax[0, :seq_len].clamp(0, seq_len - 1)
            prune_cpu = prune_mask[:seq_len, :seq_len].to(torch.bool).cpu()
            hits = prune_cpu[torch.arange(seq_len), indices]
            hit_rate = hits.to(torch.float32).mean().item()
            retention_records.append((layer, head, hit_rate))
            per_layer_rates[layer].append(hit_rate)

        nms_desc = (
            f"nms p{args.low_percentile:.0f}/p{args.high_percentile:.0f}"
            if args.nms_enabled
            else "nms off"
        )
        title = (
            f"Layer {layer:02d} Head {head:02d} "
            f"(W {args.round_window}, {nms_desc})"
        )
        out_path = (
            output_root
            / (
                f"layer_{layer:02d}_head_{head:02d}_"
                f"pruning_comparison_nms_only.png"
            )
        )
        save_comparison_figure(
            baseline_heatmap,
            pruned_heatmap,
            out_path,
            cmap="inferno",
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            title=title,
        )

        if baseline_argmax is not None and pruned_argmax is not None:
            argmax_path = (
                output_root
                / (
                    f"layer_{layer:02d}_head_{head:02d}_"
                    f"pruning_argmax_nms_only.png"
                )
            )
            save_comparison_figure(
                baseline_argmax,
                pruned_argmax,
                argmax_path,
                cmap="binary",
                figsize=tuple(args.figsize),
                dpi=args.dpi,
                title=f"Argmax coverage {title}",
            )

        if args.verbose:
            rel = out_path.relative_to(output_root)
            msg = f"Saved comparison figure {rel}"
            if baseline_argmax is not None:
                rel_arg = argmax_path.relative_to(output_root)
                msg += f" and {rel_arg}"
            print(msg)
        if hit_rate is not None:
            print(
                f"Layer {layer:02d} Head {head:02d} baseline argmax retention rate: {hit_rate:.4f}"
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not retention_records:
        print("No retention records computed", file=sys.stderr)
        return

    overall_rate = sum(rate for _, _, rate in retention_records) / len(retention_records)
    per_layer_avg = {layer: sum(vals) / len(vals) for layer, vals in per_layer_rates.items()}

    # Compute NMS metrics if enabled
    nms_metrics = {}
    if args.nms_enabled and all_nms_drop_counts:
        # Flatten all drop counts
        all_drops = [d for drops in all_nms_drop_counts for d in drops]
        total_drops = sum(all_drops)
        total_rounds = len(all_drops)
        nms_metrics = {
            "nms_enabled": True,
            "low_percentile": args.low_percentile,
            "high_percentile": args.high_percentile,
            "nms_drop_rate": total_drops / max(total_rounds, 1),
            "nms_total_drops": total_drops,
            "nms_total_rounds": total_rounds,
            "nms_drop_count_per_head": [sum(drops) for drops in all_nms_drop_counts],
        }
    else:
        nms_metrics = {
            "nms_enabled": False,
            "low_percentile": None,
            "high_percentile": None,
            "nms_drop_rate": 0.0,
        }

    metrics = {
        "overall_retention": overall_rate,
        "head_count": len(retention_records),
        "round_window": args.round_window,
        "per_head": [
            {"layer": layer, "head": head, "retention": rate}
            for layer, head, rate in retention_records
        ],
        "per_layer": per_layer_avg,
        "sample_file": str(args.head_sample_file),
        "trace": str(trace_dir),
        "stats_trace": str(stats_trace_dir),
        **nms_metrics,
    }

    metrics_path = output_root / "retention_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Overall retention across {len(retention_records)} heads: {overall_rate:.4f}")

    plot_path = output_root / "layer_retention.png"
    save_layer_retention_plot(per_layer_avg, plot_path, dpi=args.dpi)

    if args.verbose:
        print(f"Saved retention metrics to {metrics_path}")
        print(f"Saved layer retention plot to {plot_path}")


if __name__ == "__main__":
    main()
