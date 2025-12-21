"""
Module 2 Evaluation Script

Evaluation Metrics:
- TopK Hit Rate: argmax key in selected bin's TopK keys
- Keys per Query: TopK + num_recent_keys
- Handle recent keys as auto-hit
- Handle empty bins (mask with -inf)

Baseline Modes:
- --baseline hybrid_freq: Hybrid frequency-based scoring from attention_pruning_case_study_hybrid_rounds_xtrace.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoConfig

from model import Module2Network, create_model

# Import utilities from hf_offline_runner_sparse for hybrid frequency baseline
import sys as _sys
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from weian_development.hf_offline_runner_sparse.round_pruning_utils import (
    invert_rope,
    to_complex_pairs,
    build_rotary,
    compute_rotary_tables,
    build_geometric_offsets,
    compute_frequency_statistics_from_means,
)


###############################################################################
# Hybrid Frequency Baseline Algorithm
# From: attention_pruning_case_study_hybrid_rounds_xtrace.py
###############################################################################

def score_keys_for_round_baseline(
    key_indices: torch.Tensor,
    round_start: int,
    amp: torch.Tensor,
    phi: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
    offsets: torch.Tensor,
    aggregation: str,
) -> torch.Tensor:
    """
    Score keys for a round using hybrid frequency method.

    This is the core scoring function from attention_pruning_case_study_hybrid_rounds_xtrace.py.
    """
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


class HybridFrequencyBaseline:
    """
    Hybrid frequency-based key scoring baseline.

    This implements the algorithm from attention_pruning_case_study_hybrid_rounds_xtrace.py
    for fair comparison with the Module 2 learned approach.
    """

    def __init__(
        self,
        calibration_Q: torch.Tensor,
        calibration_K: torch.Tensor,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        max_keys: int = 2048,
        round_window: int = 128,
        offset_max_length: int = 65536,
        score_aggregation: str = "mean",
        seed: int = 0,
    ):
        """
        Initialize hybrid frequency baseline with calibration data.

        Args:
            calibration_Q: Q tensor from calibration/training trace (seq_len, head_dim)
            calibration_K: K tensor from calibration/training trace (seq_len, head_dim)
            model_path: Path to model for RoPE parameters
            device: torch device
            dtype: computation dtype
            max_keys: maximum keys to retain after pruning
            round_window: size of each maintenance round
            offset_max_length: max offset for geometric grid
            score_aggregation: "mean" or "max"
            seed: random seed for tie-breaking
        """
        self.device = device
        self.dtype = dtype
        self.max_keys = max_keys
        self.round_window = round_window
        self.score_aggregation = score_aggregation
        self.seed = seed

        # Build RoPE
        self.rotary = build_rotary(device, model_path, dtype)
        self.attention_scale = float(getattr(self.rotary, "attention_scaling", 1.0))

        # Get dimensions
        calib_seq_len, head_dim = calibration_Q.shape
        self.head_dim = head_dim
        freq_count = head_dim // 2

        # Compute rotary tables for calibration sequence
        cos_table, sin_table, inv_freq, _ = compute_rotary_tables(
            self.rotary, calib_seq_len, head_dim, dtype, device
        )

        # Invert RoPE on calibration Q to get unrotated Q
        calib_Q_device = calibration_Q.to(device=device, dtype=dtype)
        calib_unrot = invert_rope(calib_Q_device, cos_table, sin_table, self.attention_scale)

        # Compute calibration statistics (mean Q in complex form)
        calib_complex = to_complex_pairs(calib_unrot)
        self.q_mean_complex = calib_complex.mean(dim=0)  # (freq_count,)
        self.q_abs_mean = torch.abs(calib_complex).mean(dim=0)  # (freq_count,)

        # Store inv_freq for scoring
        self.omega = inv_freq[:freq_count].to(device=device, dtype=torch.float32)

        # Build geometric offsets
        self.offsets = build_geometric_offsets(offset_max_length, device)

    def compute_key_scores(
        self,
        K: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute frequency statistics for all keys in the test sequence.

        Args:
            K: Key tensor (test_seq_len, head_dim)
            seq_len: sequence length

        Returns:
            amp, phi, extra tensors for scoring
        """
        # Compute rotary tables for test sequence
        cos_table, sin_table, inv_freq, _ = compute_rotary_tables(
            self.rotary, seq_len, self.head_dim, self.dtype, self.device
        )

        # Invert RoPE on K
        K_device = K.to(device=self.device, dtype=self.dtype)
        k_unrot = invert_rope(K_device, cos_table, sin_table, self.attention_scale)

        # Compute frequency statistics
        amp, phi, extra = compute_frequency_statistics_from_means(
            self.q_mean_complex,
            self.q_abs_mean,
            k_unrot,
        )

        return (
            amp.to(dtype=torch.float32),
            phi.to(dtype=torch.float32),
            extra.to(dtype=torch.float32),
        )

    def get_topk_keys_for_query(
        self,
        q_idx: int,
        round_start: int,
        amp: torch.Tensor,
        phi: torch.Tensor,
        extra: torch.Tensor,
        k_val: int,
    ) -> torch.Tensor:
        """
        Get top-k historical keys for a query using hybrid frequency scoring.

        Args:
            q_idx: query index (for determining which keys are in current round)
            round_start: start of current round
            amp, phi, extra: frequency statistics
            k_val: number of top keys to return

        Returns:
            indices of top-k historical keys
        """
        # Historical keys are indices [0, round_start)
        num_historical = round_start
        if num_historical == 0:
            return torch.tensor([], device=self.device, dtype=torch.long)

        key_indices = torch.arange(num_historical, device=self.device, dtype=torch.long)

        # Score all historical keys
        scores = score_keys_for_round_baseline(
            key_indices,
            round_start,
            amp,
            phi,
            self.omega,
            extra,
            self.offsets,
            self.score_aggregation,
        )

        # Add small noise for tie-breaking
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + q_idx)
        noise = torch.rand(scores.shape, device=self.device, dtype=scores.dtype, generator=generator) * 1e-6
        scores = scores + noise

        # Get top-k
        actual_k = min(k_val, num_historical)
        _, topk_indices = torch.topk(scores, actual_k)

        return key_indices[topk_indices]


###############################################################################
# Original evaluation code
###############################################################################

def setup_logging(config):
    """Setup logging configuration."""
    exp_dir = Path(__file__).parent
    log_dir = exp_dir / config['output']['logs_dir']
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'evaluate.log'

    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )

    return logging.getLogger(__name__)


def load_trace_data(config, logger, trace_type='test'):
    """
    Load trace data from qk.pt file.

    Args:
        config: Configuration dict
        logger: Logger instance
        trace_type: 'test' for test trace, 'train' for training trace

    Returns:
        dict with Q, K tensors and metadata
    """
    exp_dir = Path(__file__).parent

    if trace_type == 'train':
        trace_path = exp_dir / config['data']['trace_path']
        logger.info("Loading TRAINING trace for calibration")
    else:
        # Use test_trace_path for evaluation if available, otherwise fall back to trace_path
        if 'test_trace_path' in config['data']:
            trace_path = exp_dir / config['data']['test_trace_path']
            logger.info("Using TEST trace for evaluation (cross-trace validation mode)")
        else:
            trace_path = exp_dir / config['data']['trace_path']
            logger.info("Using TRAINING trace for evaluation (overfit validation mode)")

    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    logger.info(f"Loading trace data from: {trace_path}")
    qk_data = torch.load(trace_path, map_location='cpu')

    # Load head sample file
    head_sample_path = exp_dir / config['data']['head_sample_file']
    if not head_sample_path.exists():
        raise FileNotFoundError(f"Head sample file not found: {head_sample_path}")

    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)

    # Select first head for POC
    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}] for POC evaluation")

    # Extract Q, K for selected head
    Q = qk_data['q'][layer, head]  # (seq_len, head_dim)
    K = qk_data['k'][layer, head]  # (seq_len, head_dim)

    seq_len, head_dim = Q.shape
    logger.info(f"Trace shape: Q={Q.shape}, K={K.shape}")

    # Compute attention matrix with causal mask
    scale = head_dim ** -0.5
    attention_logits = Q @ K.T * scale  # (seq_len, seq_len)

    # Apply causal mask: keys must be <= query position
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attention_logits.masked_fill_(causal_mask, float('-inf'))

    attention = F.softmax(attention_logits, dim=-1)  # (seq_len, seq_len)

    return {
        'Q': Q,
        'K': K,
        'attention': attention,
        'seq_len': seq_len,
        'head_dim': head_dim,
        'layer': layer,
        'head': head
    }


def load_checkpoint(checkpoint_path, config, device, logger, use_l2_norm=False):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dict
        device: torch.device
        logger: Logger instance
        use_l2_norm: Whether to use L2 normalization (default: False after RoPE layout fix)

    Returns:
        Loaded model
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(config, use_l2_norm=use_l2_norm)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.6f}")

    return model


def compute_topk_hit_rate(
    model,
    trace_data,
    K_values,
    round_window,
    exclude_tail,
    device,
    logger,
    top_bins: int = 1
):
    """
    Compute TopK Hit Rate for different K values.

    TopK Hit Rate: Percentage of queries whose argmax key is either:
    1. In the selected bin(s)'s TopK keys (historical)
    2. In recent keys (auto-hit, always included via full attention)

    Args:
        model: Module2Network instance (in eval mode)
        trace_data: Dict with Q, K, attention, seq_len
        K_values: List of K values to evaluate (e.g., [50, 500, 1000])
        round_window: Size of each round
        exclude_tail: Number of tail queries to exclude
        device: torch.device
        logger: Logger instance
        top_bins: Number of top bins to use for key selection (default: 1)

    Returns:
        Dict with hit rates for each K value and detailed statistics
    """
    model.eval()

    Q = trace_data['Q'].to(device)
    K = trace_data['K'].to(device)
    attention = trace_data['attention'].to(device)
    seq_len = trace_data['seq_len']

    # Initialize counters for each K value
    results = {k: {'hits': 0, 'total': 0, 'recent_hits': 0, 'bin_hits': 0} for k in K_values}

    # Determine valid query range (exclude tail)
    valid_end = min(seq_len - exclude_tail, seq_len)

    with torch.no_grad():
        # Iterate over rounds
        for round_start in range(0, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)

            # Skip first round (no historical keys)
            if round_start == 0:
                continue

            # Historical keys (< round_start)
            historical_keys = K[:round_start]  # (round_start, head_dim)
            num_historical = round_start

            # Compute reference angles for this round
            reference_angles = model.compute_reference_angles(round_start, round_window)

            # Forward pass: Key network on historical keys
            key_probs = model.forward_keys(historical_keys, reference_angles)  # (num_historical, num_bins)

            # Detect empty bins (for masking)
            empty_bin_mask = (key_probs.sum(dim=0) == 0).detach()  # (num_bins,)

            # Iterate over queries in this round
            for q_idx in range(round_start, min(round_end, valid_end)):
                # Get attention weights for all keys <= q_idx (causal)
                attn_weights = attention[q_idx, :q_idx + 1]

                # Find argmax key
                argmax_key = attn_weights.argmax().item()

                # Check if argmax is in recent keys (>= round_start)
                argmax_in_recent = argmax_key >= round_start

                # Get query vector
                query = Q[q_idx:q_idx + 1]  # (1, head_dim)

                # Forward pass: Query network
                query_bin_probs = model.forward_queries(query, reference_angles, empty_bin_mask)  # (1, num_bins)

                # Get top-N bins
                actual_top_bins = min(top_bins, model.num_bins)
                _, top_bin_indices = torch.topk(query_bin_probs[0], actual_top_bins)

                # For each K value, check if argmax key is hit
                for k_val in K_values:
                    results[k_val]['total'] += 1

                    if argmax_in_recent:
                        # Auto-hit: argmax is in recent keys (always included via full attention)
                        results[k_val]['hits'] += 1
                        results[k_val]['recent_hits'] += 1
                    else:
                        # Collect top-K keys from each selected bin
                        all_selected_keys = set()
                        for bin_idx in top_bin_indices:
                            bin_scores = key_probs[:, bin_idx]
                            actual_k = min(k_val, num_historical)
                            _, topk_indices = torch.topk(bin_scores, actual_k)
                            all_selected_keys.update(topk_indices.tolist())

                        if argmax_key in all_selected_keys:
                            results[k_val]['hits'] += 1
                            results[k_val]['bin_hits'] += 1

    # Compute hit rates
    hit_rates = {}
    for k_val in K_values:
        total = results[k_val]['total']
        if total > 0:
            hit_rate = results[k_val]['hits'] / total * 100
            recent_rate = results[k_val]['recent_hits'] / total * 100
            bin_rate = results[k_val]['bin_hits'] / total * 100
        else:
            hit_rate = 0.0
            recent_rate = 0.0
            bin_rate = 0.0

        # Calculate effective keys per query
        keys_per_query = top_bins * k_val

        hit_rates[k_val] = {
            'hit_rate': hit_rate,
            'recent_hit_rate': recent_rate,
            'bin_hit_rate': bin_rate,
            'total_queries': total,
            'total_hits': results[k_val]['hits'],
            'recent_hits': results[k_val]['recent_hits'],
            'bin_hits': results[k_val]['bin_hits'],
            'top_bins': top_bins,
            'keys_per_query': keys_per_query
        }

        if top_bins == 1:
            logger.info(
                f"K={k_val}: Hit Rate = {hit_rate:.2f}% "
                f"(Recent: {recent_rate:.2f}%, Bin: {bin_rate:.2f}%, "
                f"Total: {results[k_val]['hits']}/{total})"
            )
        else:
            logger.info(
                f"K={k_val} (top-{top_bins} bins, {keys_per_query} keys/query): Hit Rate = {hit_rate:.2f}% "
                f"(Recent: {recent_rate:.2f}%, Bin: {bin_rate:.2f}%, "
                f"Total: {results[k_val]['hits']}/{total})"
            )

    return hit_rates


def compute_topk_hit_rate_baseline(
    baseline: HybridFrequencyBaseline,
    trace_data: Dict,
    K_values: List[int],
    round_window: int,
    exclude_tail: int,
    device: torch.device,
    logger,
):
    """
    Compute TopK Hit Rate using hybrid frequency baseline.

    This mirrors the evaluation logic in compute_topk_hit_rate but uses
    the hybrid frequency scoring instead of the learned model.

    Args:
        baseline: HybridFrequencyBaseline instance
        trace_data: Dict with Q, K, attention, seq_len
        K_values: List of K values to evaluate
        round_window: Size of each round
        exclude_tail: Number of tail queries to exclude
        device: torch.device
        logger: Logger instance

    Returns:
        Dict with hit rates for each K value
    """
    K = trace_data['K']
    attention = trace_data['attention'].to(device)
    seq_len = trace_data['seq_len']

    # Compute frequency statistics for all keys once
    amp, phi, extra = baseline.compute_key_scores(K, seq_len)

    # Initialize counters for each K value
    results = {k: {'hits': 0, 'total': 0, 'recent_hits': 0, 'bin_hits': 0} for k in K_values}

    # Determine valid query range (exclude tail)
    valid_end = min(seq_len - exclude_tail, seq_len)

    with torch.no_grad():
        # Iterate over rounds
        for round_start in range(0, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)

            # Skip first round (no historical keys)
            if round_start == 0:
                continue

            num_historical = round_start

            # Iterate over queries in this round
            for q_idx in range(round_start, min(round_end, valid_end)):
                # Get attention weights for all keys <= q_idx (causal)
                attn_weights = attention[q_idx, :q_idx + 1]

                # Find argmax key
                argmax_key = attn_weights.argmax().item()

                # Check if argmax is in recent keys (>= round_start)
                argmax_in_recent = argmax_key >= round_start

                # For each K value, check if argmax key is hit
                for k_val in K_values:
                    results[k_val]['total'] += 1

                    if argmax_in_recent:
                        # Auto-hit: argmax is in recent keys
                        results[k_val]['hits'] += 1
                        results[k_val]['recent_hits'] += 1
                    else:
                        # Get top-k keys using hybrid frequency scoring
                        topk_indices = baseline.get_topk_keys_for_query(
                            q_idx, round_start, amp, phi, extra, k_val
                        )

                        if argmax_key in topk_indices:
                            results[k_val]['hits'] += 1
                            results[k_val]['bin_hits'] += 1

    # Compute hit rates
    hit_rates = {}
    for k_val in K_values:
        total = results[k_val]['total']
        if total > 0:
            hit_rate = results[k_val]['hits'] / total * 100
            recent_rate = results[k_val]['recent_hits'] / total * 100
            bin_rate = results[k_val]['bin_hits'] / total * 100
        else:
            hit_rate = 0.0
            recent_rate = 0.0
            bin_rate = 0.0

        hit_rates[k_val] = {
            'hit_rate': hit_rate,
            'recent_hit_rate': recent_rate,
            'bin_hit_rate': bin_rate,
            'total_queries': total,
            'total_hits': results[k_val]['hits'],
            'recent_hits': results[k_val]['recent_hits'],
            'bin_hits': results[k_val]['bin_hits']
        }

        logger.info(
            f"K={k_val}: Hit Rate = {hit_rate:.2f}% "
            f"(Recent: {recent_rate:.2f}%, Bin: {bin_rate:.2f}%, "
            f"Total: {results[k_val]['hits']}/{total})"
        )

    return hit_rates


def evaluate_baseline(config, logger, model_path: Path = None):
    """
    Evaluate using hybrid frequency baseline.

    Args:
        config: Configuration dict
        logger: Logger instance
        model_path: Path to model for RoPE parameters

    Returns:
        Evaluation results dict
    """
    logger.info("Starting Hybrid Frequency Baseline evaluation...")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load calibration data (training trace)
    logger.info("Loading calibration data from training trace...")
    calib_data = load_trace_data(config, logger, trace_type='train')

    # Load test data
    logger.info("Loading test data...")
    test_data = load_trace_data(config, logger, trace_type='test')

    # Get evaluation parameters
    K_values = config['evaluation']['topk_K']
    round_window = config['evaluation'].get('round_window', config['training']['round_window'])
    exclude_tail = config['training']['exclude_tail']

    # Default model path
    if model_path is None:
        model_path = Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B")

    logger.info(f"Using model path: {model_path}")
    logger.info(f"Calibration trace seq_len: {calib_data['seq_len']}")
    logger.info(f"Test trace seq_len: {test_data['seq_len']}")

    # Initialize baseline with calibration data
    baseline = HybridFrequencyBaseline(
        calibration_Q=calib_data['Q'],
        calibration_K=calib_data['K'],
        model_path=model_path,
        device=device,
        dtype=torch.float32,
        max_keys=2048,  # Default from original script
        round_window=round_window,
        offset_max_length=65536,
        score_aggregation="mean",  # Default from original script
        seed=0,
    )

    logger.info(f"Evaluating TopK Hit Rate for K={K_values}")

    # Compute hit rates
    hit_rates = compute_topk_hit_rate_baseline(
        baseline, test_data, K_values, round_window, exclude_tail, device, logger
    )

    # Save results
    exp_dir = Path(__file__).parent
    results_dir = exp_dir / config['output']['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / 'evaluation_results_baseline_hybrid_freq.json'
    with open(results_file, 'w') as f:
        json.dump(hit_rates, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    return hit_rates


def evaluate(config, checkpoint_path, logger, top_bins_list: List[int] = None):
    """
    Main evaluation function.

    Args:
        config: Configuration dict
        checkpoint_path: Path to model checkpoint
        logger: Logger instance
        top_bins_list: List of top_bins values to evaluate (default: [1, 8])

    Returns:
        Evaluation results dict with results for each top_bins setting
    """
    logger.info("Starting Module 2 evaluation...")

    # Default to evaluating both top-1 and top-8 bins
    if top_bins_list is None:
        top_bins_list = [1, 8]

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load trace data
    trace_data = load_trace_data(config, logger)

    # Load model
    model = load_checkpoint(checkpoint_path, config, device, logger)

    # Get evaluation parameters
    K_values = config['evaluation']['topk_K']
    round_window = config['evaluation'].get('round_window', config['training']['round_window'])
    exclude_tail = config['training']['exclude_tail']

    # Evaluate for each top_bins setting
    all_results = {}
    for top_bins in top_bins_list:
        logger.info(f"\n--- Evaluating with top-{top_bins} bins ---")
        logger.info(f"Evaluating TopK Hit Rate for K={K_values}")

        hit_rates = compute_topk_hit_rate(
            model, trace_data, K_values, round_window, exclude_tail, device, logger,
            top_bins=top_bins
        )

        all_results[f'top_{top_bins}_bins'] = hit_rates

    # Save results
    exp_dir = Path(__file__).parent
    results_dir = exp_dir / config['output']['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    return all_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Module 2 Evaluation')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='output/checkpoints/best_model.pt',
        help='Path to checkpoint (relative to experiment dir)'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        choices=['hybrid_freq'],
        default=None,
        help='Baseline algorithm to evaluate (hybrid_freq: Hybrid Frequency from attention_pruning_case_study)'
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help='Model directory for RoPE parameters (only used with --baseline)'
    )
    parser.add_argument(
        '--top-bins',
        type=str,
        default='1,8',
        help='Comma-separated list of top_bins values to evaluate (default: 1,8)'
    )
    args = parser.parse_args()

    # Parse top_bins list
    top_bins_list = [int(x.strip()) for x in args.top_bins.split(',')]

    # Load configuration
    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger = setup_logging(config)

    try:
        if args.baseline == 'hybrid_freq':
            # Run baseline evaluation
            logger.info("Running Hybrid Frequency baseline evaluation...")
            results = evaluate_baseline(config, logger, model_path=args.model_path)
            logger.info("Baseline evaluation completed successfully")

            # Print summary
            logger.info("\n=== Baseline (Hybrid Frequency) Evaluation Summary ===")
            for k_val, metrics in results.items():
                logger.info(f"K={k_val}: {metrics['hit_rate']:.2f}% hit rate")
        else:
            # Run standard model evaluation
            checkpoint_path = exp_dir / args.checkpoint

            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                logger.info("Available checkpoints:")
                checkpoints_dir = exp_dir / config['output']['checkpoints_dir']
                if checkpoints_dir.exists():
                    for ckpt in checkpoints_dir.glob('*.pt'):
                        logger.info(f"  - {ckpt.name}")
                return

            # Run evaluation
            results = evaluate(config, checkpoint_path, logger, top_bins_list=top_bins_list)
            logger.info("Evaluation completed successfully")

            # Print summary
            logger.info("\n=== Evaluation Summary ===")
            for top_bins_key, top_bins_results in results.items():
                logger.info(f"\n{top_bins_key}:")
                for k_val, metrics in top_bins_results.items():
                    keys_per_query = metrics.get('keys_per_query', k_val)
                    logger.info(f"  K={k_val}: {metrics['hit_rate']:.2f}% hit rate ({keys_per_query} keys/query)")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
