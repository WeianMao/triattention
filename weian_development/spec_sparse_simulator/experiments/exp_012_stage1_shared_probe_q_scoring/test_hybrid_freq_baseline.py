"""
Test Hybrid Frequency baseline with exp_012 evaluation setting.

This script evaluates the original Hybrid Frequency method using TopK Hit Rate.
Two modes:
1. Original: Multiple offsets (1, 2, 4, 8, 16...) with aggregation
2. Single offset: Only use offset=64 (round_window/2)

Usage:
    python test_hybrid_freq_baseline.py --mode original
    python test_hybrid_freq_baseline.py --mode single_offset
"""
import sys
from pathlib import Path
import logging
import json
import math

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

# Add parent paths for imports
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.hf_offline_runner_sparse.round_pruning_utils import (
    invert_rope,
    to_complex_pairs,
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def build_rotary(device, model_path, dtype):
    """Build RoPE embedding from model config."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    config.rope_scaling = rope_scaling
    rotary = Qwen3RotaryEmbedding(config=config, device=device)
    rotary.to(dtype=dtype)
    return rotary


def compute_rotary_tables(rotary, seq_len, head_dim, dtype, device):
    """Compute cos/sin tables and inv_freq for RoPE."""
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    base = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0]
    sin_table = sin_table[0]
    inv_freq = rotary.inv_freq.to(device=device, dtype=torch.float64)
    return cos_table, sin_table, inv_freq


def build_geometric_offsets(max_length, device):
    """Build geometric offset grid: 1, 2, 4, 8, 16, ..."""
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    offsets = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def build_single_offset(offset_value, device):
    """Build single offset tensor."""
    return torch.tensor([float(offset_value)], device=device, dtype=torch.float32)


def compute_frequency_statistics_from_means(q_mean_complex, q_abs_mean, k_unrot):
    """Compute amp, phi, extra for hybrid frequency scoring."""
    k_complex = to_complex_pairs(k_unrot)
    q_mean_abs = torch.abs(q_mean_complex)
    k_abs = torch.abs(k_complex)
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    phi = torch.atan2(relative.imag, relative.real)
    amp = q_mean_abs.unsqueeze(0) * k_abs
    extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
    return amp, phi, extra


def score_keys_hybrid_freq(
    key_indices,
    reference_pos,
    amp,
    phi,
    omega,
    extra,
    offsets,
    aggregation="mean",
):
    """
    Score keys using hybrid frequency method.

    Args:
        key_indices: Tensor of key positions to score
        reference_pos: Reference position for scoring
        amp, phi, extra: Frequency statistics
        omega: Angular frequencies
        offsets: Offset values for future positions
        aggregation: "mean" or "max" for aggregating across offsets

    Returns:
        scores: Tensor of shape (num_keys,)
    """
    if key_indices.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    # Compute delta (distance from key to reference)
    base_delta = reference_pos - key_indices.to(device=amp.device, dtype=torch.float32)

    # Add offsets to get future positions
    delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)  # (num_keys, num_offsets)

    # Select amp, phi, extra for these keys
    amp_sel = amp.index_select(0, key_indices)  # (num_keys, num_freqs)
    phi_sel = phi.index_select(0, key_indices)  # (num_keys, num_freqs)
    extra_sel = extra.index_select(0, key_indices)  # (num_keys, num_freqs)

    # Compute phase: delta * omega + phi
    # delta_grid: (num_keys, num_offsets)
    # omega: (num_freqs,)
    # phi_sel: (num_keys, num_freqs)
    phase = delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi_sel.unsqueeze(1)
    cos_phase = torch.cos(phase)  # (num_keys, num_offsets, num_freqs)

    # Base scores: sum over frequencies of amp * cos(phase)
    base_scores = (amp_sel.unsqueeze(1) * cos_phase).sum(dim=2)  # (num_keys, num_offsets)

    # Add extra term (sum over frequencies)
    additive = extra_sel.sum(dim=1, keepdim=True)  # (num_keys, 1)
    combined = base_scores + additive  # (num_keys, num_offsets)

    # Aggregate across offsets
    if aggregation == "mean":
        return combined.mean(dim=1)
    return combined.max(dim=1).values


def compute_topk_hit_rate_hybrid_freq(
    Q, K,
    q_mean_complex, q_abs_mean,
    omega, offsets,
    K_values, round_window, exclude_tail,
    cos_table, sin_table, attention_scale,
    device, logger,
    aggregation="mean",
):
    """
    Compute TopK hit rate using hybrid frequency scoring.

    Similar to evaluate.py's compute_topk_hit_rate but using hybrid freq scoring.
    """
    seq_len = Q.shape[0]

    # Invert RoPE on K to get k_unrot
    K_device = K.to(device)
    k_unrot = invert_rope(K_device, cos_table, sin_table, attention_scale)

    # Compute amp, phi, extra for all keys
    amp, phi, extra = compute_frequency_statistics_from_means(q_mean_complex, q_abs_mean, k_unrot)
    amp = amp.to(torch.float32)
    phi = phi.to(torch.float32)
    extra = extra.to(torch.float32)

    # Initialize results
    results = {k: {'total_queries': 0, 'total_hits': 0, 'recent_hits': 0, 'bin_hits': 0}
               for k in K_values}

    num_rounds = (seq_len + round_window - 1) // round_window

    for round_idx in range(1, num_rounds):  # Skip first round
        round_start = round_idx * round_window
        round_end = min(round_start + round_window, seq_len)

        # Reference position (middle of round)
        ref_pos = round_start + round_window // 2

        # Historical keys (before this round)
        historical_end = round_start
        if historical_end <= 0:
            continue

        # Keys in current round for "recent" hits
        round_keys_start = round_start
        round_keys_end = round_end

        # Score all historical keys
        key_indices = torch.arange(historical_end, device=device, dtype=torch.long)
        scores = score_keys_hybrid_freq(
            key_indices, ref_pos,
            amp, phi, omega, extra, offsets,
            aggregation=aggregation,
        )

        # Process each query in this round
        valid_end = min(round_end, seq_len - exclude_tail)
        for q_idx in range(round_start, valid_end):
            # Compute ground truth attention for this query
            Q_q = Q[q_idx:q_idx+1].to(device)
            K_all = K[:q_idx+1].to(device)

            attn_scores = torch.matmul(Q_q, K_all.t()).squeeze(0)
            gt_top1_idx = attn_scores.argmax().item()

            for k_val in K_values:
                results[k_val]['total_queries'] += 1

                # Check if top1 is in recent keys (current round)
                if gt_top1_idx >= round_keys_start:
                    results[k_val]['total_hits'] += 1
                    results[k_val]['recent_hits'] += 1
                    continue

                # Check if top1 is in top-K historical keys
                if historical_end > 0:
                    topk_count = min(k_val, historical_end)
                    _, topk_indices = torch.topk(scores[:historical_end], topk_count)
                    topk_positions = key_indices[:historical_end][topk_indices]

                    if gt_top1_idx in topk_positions:
                        results[k_val]['total_hits'] += 1
                        results[k_val]['bin_hits'] += 1

    # Compute percentages
    final_results = {}
    for k_val in K_values:
        total = results[k_val]['total_queries']
        if total > 0:
            hit_rate = 100.0 * results[k_val]['total_hits'] / total
            recent_rate = 100.0 * results[k_val]['recent_hits'] / total
            bin_rate = 100.0 * results[k_val]['bin_hits'] / total
        else:
            hit_rate = recent_rate = bin_rate = 0.0

        final_results[k_val] = {
            'hit_rate': hit_rate,
            'recent_hit_rate': recent_rate,
            'bin_hit_rate': bin_rate,
            'total_queries': total,
            'total_hits': results[k_val]['total_hits'],
            'recent_hits': results[k_val]['recent_hits'],
            'bin_hits': results[k_val]['bin_hits'],
        }

        logger.info(f"K={k_val}: Hit Rate = {hit_rate:.2f}% "
                   f"(Recent: {recent_rate:.2f}%, Bin: {bin_rate:.2f}%)")

    return final_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['original', 'single_offset'], default='original',
                       help='original: multi-offset (1,2,4,8...), single_offset: only offset=64')
    parser.add_argument('--aggregation', choices=['mean', 'max'], default='mean',
                       help='How to aggregate scores across offsets (for original mode)')
    parser.add_argument('--offset-max-length', type=int, default=65536,
                       help='Max offset for geometric grid (for original mode)')
    args = parser.parse_args()

    logger = setup_logging()

    # Load config
    exp_dir = Path(__file__).parent
    with open(exp_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Mode: {args.mode}")
    if args.mode == 'original':
        logger.info(f"Aggregation: {args.aggregation}, Max offset: {args.offset_max_length}")

    # Load model path from config or use default
    model_path = Path(config.get('model', {}).get('model_path',
        "/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"))

    # Load test data
    logger.info("Loading test data...")
    test_trace_path = exp_dir / config['data']['test_trace_path']
    qk_data = torch.load(test_trace_path, map_location='cpu')

    # Load head sample
    head_sample_path = exp_dir / config['data']['head_sample_file']
    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)
    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}]")

    Q = qk_data['q'][layer, head]
    K = qk_data['k'][layer, head]
    seq_len, head_dim = Q.shape
    logger.info(f"Trace shape: Q={Q.shape}, K={K.shape}")

    # Load calibration data (for computing Q statistics)
    logger.info("Loading calibration trace...")
    calib_trace_path = exp_dir / config['data']['trace_path']
    calib_data = torch.load(calib_trace_path, map_location='cpu')
    calib_Q = calib_data['q'][layer, head]
    calib_seq_len = calib_Q.shape[0]

    # Build RoPE
    dtype = torch.float32
    rotary = build_rotary(device, model_path, dtype)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # Compute rotary tables
    cos_table, sin_table, inv_freq = compute_rotary_tables(
        rotary, seq_len, head_dim, dtype, device
    )
    calib_cos_table, calib_sin_table, _ = compute_rotary_tables(
        rotary, calib_seq_len, head_dim, dtype, device
    )

    freq_count = head_dim // 2
    omega = inv_freq[:freq_count].to(device=device, dtype=torch.float32)

    # Compute Q statistics from calibration trace
    logger.info("Computing Q statistics from calibration trace...")
    calib_Q_device = calib_Q.to(device=device, dtype=dtype)
    calib_unrot = invert_rope(calib_Q_device, calib_cos_table, calib_sin_table, attention_scale)
    calib_complex = to_complex_pairs(calib_unrot)
    q_mean_complex = calib_complex.mean(dim=0)
    q_abs_mean = torch.abs(calib_complex).mean(dim=0)

    # Build offsets based on mode
    round_window = config['training']['round_window']
    if args.mode == 'original':
        offsets = build_geometric_offsets(args.offset_max_length, device)
        logger.info(f"Using geometric offsets: {offsets.tolist()}")
    else:  # single_offset
        offset_value = round_window // 2  # 64 for round_window=128
        offsets = build_single_offset(offset_value, device)
        logger.info(f"Using single offset: {offset_value}")

    # Evaluate
    K_values = config['evaluation']['topk_K']
    exclude_tail = config['training']['exclude_tail']

    logger.info("=" * 60)
    logger.info(f"HYBRID FREQUENCY BASELINE - {args.mode.upper()}")
    logger.info("=" * 60)

    hit_rates = compute_topk_hit_rate_hybrid_freq(
        Q, K,
        q_mean_complex, q_abs_mean,
        omega, offsets,
        K_values, round_window, exclude_tail,
        cos_table, sin_table, attention_scale,
        device, logger,
        aggregation=args.aggregation,
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"=== Hybrid Frequency Baseline ({args.mode}) Results ===")
    logger.info("=" * 60)
    for k_val, metrics in hit_rates.items():
        logger.info(f"K={k_val}: {metrics['hit_rate']:.2f}% hit rate "
                   f"(Recent: {metrics['recent_hit_rate']:.2f}%, Bin: {metrics['bin_hit_rate']:.2f}%)")


if __name__ == '__main__':
    main()
