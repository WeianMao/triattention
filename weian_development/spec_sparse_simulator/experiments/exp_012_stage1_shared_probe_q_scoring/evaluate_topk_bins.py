"""
Evaluate with TopK bins instead of Top1 bin.

Compare hit rates when using multiple bins per query.
"""

import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model import create_model


def setup_logging():
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


def load_config():
    exp_dir = Path(__file__).parent
    with open(exp_dir / 'config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_trace_data(config, logger):
    exp_dir = Path(__file__).parent
    trace_path = exp_dir / config['data']['test_trace_path']

    logger.info(f"Loading trace data from: {trace_path}")
    qk_data = torch.load(trace_path, map_location='cpu')

    head_sample_path = exp_dir / config['data']['head_sample_file']
    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)

    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}]")

    Q = qk_data['q'][layer, head]
    K = qk_data['k'][layer, head]

    seq_len, head_dim = Q.shape
    scale = head_dim ** -0.5
    attention_logits = Q @ K.T * scale

    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attention_logits.masked_fill_(causal_mask, float('-inf'))
    attention = F.softmax(attention_logits, dim=-1)

    return {'Q': Q, 'K': K, 'attention': attention, 'seq_len': seq_len}


def load_model(config, device, logger):
    exp_dir = Path(__file__).parent
    checkpoint_path = exp_dir / 'output/checkpoints/best_model.pt'

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    return model


def evaluate_topk_bins(model, trace_data, K_val, num_bins_to_use, round_window, exclude_tail, device):
    """
    Evaluate hit rate using top-N bins instead of just top-1.

    For each query:
    1. Get top-N bins from query network
    2. For each bin, get top-K keys
    3. Union all keys from N bins
    4. Check if argmax key is in the union
    """
    model.eval()

    Q = trace_data['Q'].to(device)
    K = trace_data['K'].to(device)
    attention = trace_data['attention'].to(device)
    seq_len = trace_data['seq_len']

    valid_end = min(seq_len - exclude_tail, seq_len)

    total = 0
    hits = 0
    recent_hits = 0
    bin_hits = 0

    with torch.no_grad():
        for round_start in range(0, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)

            if round_start == 0:
                continue

            historical_keys = K[:round_start]
            num_historical = round_start

            reference_angles = model.compute_reference_angles(round_start, round_window)
            key_probs = model.forward_keys(historical_keys, reference_angles)

            for q_idx in range(round_start, min(round_end, valid_end)):
                attn_weights = attention[q_idx, :q_idx + 1]
                argmax_key = attn_weights.argmax().item()
                argmax_in_recent = argmax_key >= round_start

                query = Q[q_idx:q_idx + 1]
                empty_bin_mask = key_probs.sum(dim=0) == 0
                query_bin_probs = model.forward_queries(query, reference_angles, empty_bin_mask)

                # Get top-N bins
                _, top_bins = torch.topk(query_bin_probs.squeeze(0), min(num_bins_to_use, model.num_bins))

                total += 1

                if argmax_in_recent:
                    hits += 1
                    recent_hits += 1
                else:
                    # Collect keys from all top-N bins
                    all_topk_keys = set()
                    actual_k = min(K_val, num_historical)

                    for bin_idx in top_bins:
                        bin_scores = key_probs[:, bin_idx]
                        _, topk_indices = torch.topk(bin_scores, actual_k)
                        all_topk_keys.update(topk_indices.cpu().tolist())

                    if argmax_key in all_topk_keys:
                        hits += 1
                        bin_hits += 1

    hit_rate = hits / total * 100 if total > 0 else 0
    recent_rate = recent_hits / total * 100 if total > 0 else 0
    bin_rate = bin_hits / total * 100 if total > 0 else 0

    return {
        'hit_rate': hit_rate,
        'recent_hit_rate': recent_rate,
        'bin_hit_rate': bin_rate,
        'total': total,
        'hits': hits,
        'total_keys_per_query': num_bins_to_use * K_val
    }


def main():
    logger = setup_logging()
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    trace_data = load_trace_data(config, logger)
    model = load_model(config, device, logger)

    round_window = config['training']['round_window']
    exclude_tail = config['training']['exclude_tail']

    # Test different number of bins
    K_val = 50
    bins_to_test = [1, 2, 4, 8, 16, 32]

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating TopK Bins (K={K_val} keys per bin)")
    logger.info(f"{'='*60}")

    results = {}
    for num_bins in bins_to_test:
        result = evaluate_topk_bins(
            model, trace_data, K_val, num_bins, round_window, exclude_tail, device
        )
        results[num_bins] = result
        logger.info(
            f"Top-{num_bins:2d} bins: Hit Rate = {result['hit_rate']:.2f}% "
            f"(Bin: {result['bin_hit_rate']:.2f}%) "
            f"[{result['total_keys_per_query']} keys/query]"
        )

    # Also test with fixed total keys budget
    logger.info(f"\n{'='*60}")
    logger.info(f"Fixed budget: 800 keys total (K*bins = 800)")
    logger.info(f"{'='*60}")

    fixed_budget_configs = [
        (800, 1),   # 800 keys from 1 bin
        (400, 2),   # 400 keys from 2 bins
        (200, 4),   # 200 keys from 4 bins
        (100, 8),   # 100 keys from 8 bins
        (50, 16),   # 50 keys from 16 bins
    ]

    for k_per_bin, num_bins in fixed_budget_configs:
        result = evaluate_topk_bins(
            model, trace_data, k_per_bin, num_bins, round_window, exclude_tail, device
        )
        logger.info(
            f"K={k_per_bin:3d} x {num_bins:2d} bins = {k_per_bin*num_bins} keys: "
            f"Hit Rate = {result['hit_rate']:.2f}% (Bin: {result['bin_hit_rate']:.2f}%)"
        )


if __name__ == '__main__':
    main()
