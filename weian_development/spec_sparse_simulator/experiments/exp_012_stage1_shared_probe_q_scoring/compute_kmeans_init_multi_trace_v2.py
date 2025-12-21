"""
Multi-Trace K-means Initialization for Probe Vectors (V2).

Based on compute_kmeans_init.py with correct RoPE layout and multi-trace support.

Key differences from compute_kmeans_init_multi_trace.py:
1. Uses correct RoPE front/back vector layout (not interleaved)
2. Uses model's YaRN-scaled inv_freq
3. Properly handles use_l2_norm and invert_to_origin flags
"""

import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from sklearn.cluster import KMeans

# Import correct RoPE functions from single-trace version
from compute_kmeans_init import (
    apply_rope_rotation,
    l2_normalize,
    load_model_inv_freq,
    compute_magnitude_features,
    compute_magnitude_init,
)


def get_training_trace_paths(test_trace_path, logger):
    """
    Get all training trace paths (excluding test trace).

    Args:
        test_trace_path: Path to test trace (to exclude)
        logger: Logger instance

    Returns:
        List of Path objects for training traces
    """
    traces_dir = Path("/data/rbg/users/weian/project/rl/dc/outputs/deepseek_r1_qwen3_8b/qk_bf16_traces")

    # Resolve test trace path
    test_trace_resolved = Path(test_trace_path).resolve()

    # Find all trace directories
    trace_paths = []
    for trace_dir in sorted(traces_dir.glob("qid*")):
        qk_file = trace_dir / "qk.pt"
        if qk_file.exists():
            # Exclude test trace
            if qk_file.resolve() != test_trace_resolved:
                trace_paths.append(qk_file)

    logger.info(f"Found {len(trace_paths)} training traces (excluding test trace)")
    for p in trace_paths:
        logger.info(f"  - {p.parent.name}")

    return trace_paths


def compute_kmeans_init_multi_trace(config, logger, n_clusters=128, random_state=42,
                                     return_extras=False, use_l2_norm=False,
                                     invert_to_origin=False, inv_freq=None):
    """
    Compute K-means initialization from ALL training traces.

    Algorithm:
    1. Load all training traces (excluding test trace)
    2. For each trace, for each round:
       - Compute reference position: ref_pos = round_start + round_window / 2
       - For each Q, compute Q_relative = RoPE(Q_post, -ref_pos)
    3. Optionally L2 normalize
    4. Run K-means on all collected vectors

    Args:
        config: Configuration dict
        logger: Logger instance
        n_clusters: Number of clusters (default: 128)
        random_state: Random seed for K-means
        return_extras: If True, also return cluster labels and unnormalized Q_relatives
        use_l2_norm: If True, L2 normalize vectors before K-means (default: False)
        invert_to_origin: If True, invert each Q to position 0 (default: False)
        inv_freq: Optional inverse frequency tensor from model's RoPE

    Returns:
        If return_extras=False:
            cluster_centers: Tensor of shape (n_clusters, head_dim)
        If return_extras=True:
            tuple of (cluster_centers, cluster_labels, Q_relatives_unnorm)
    """
    exp_dir = Path(__file__).parent

    # Load inv_freq from model if not provided
    if inv_freq is None:
        model_path = config.get('model', {}).get('model_path',
            "/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B")
        inv_freq = load_model_inv_freq(model_path, logger)

    # Load head sample file
    head_sample_path = exp_dir / config['data']['head_sample_file']
    if not head_sample_path.exists():
        raise FileNotFoundError(f"Head sample file not found: {head_sample_path}")

    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)

    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}] for K-means initialization")

    # Get test trace path and find training traces
    test_trace_path = exp_dir / config['data']['test_trace_path']
    trace_paths = get_training_trace_paths(test_trace_path, logger)

    if len(trace_paths) == 0:
        raise ValueError("No training traces found!")

    round_window = config['training']['round_window']
    exclude_tail = config['training']['exclude_tail']

    # Collect Q_relative vectors from all traces
    all_Q_relatives_unnorm = []
    total_vectors = 0

    for trace_idx, trace_path in enumerate(trace_paths):
        logger.info(f"[{trace_idx+1}/{len(trace_paths)}] Processing {trace_path.parent.name}...")

        # Load trace and extract only the required head (memory efficient)
        qk_data = torch.load(trace_path, map_location='cpu')
        Q = qk_data['q'][layer, head].clone()  # (seq_len, head_dim)
        del qk_data  # Free memory

        seq_len, head_dim = Q.shape

        # Collect Q_relative for this trace
        trace_Q_relatives = []

        for round_start in range(0, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)

            # Skip first round (no historical keys)
            if round_start == 0:
                continue

            # Compute reference position (middle of round)
            ref_pos = round_start + round_window // 2

            # Get queries in this round (exclude tail)
            valid_end = min(round_end, seq_len - exclude_tail)
            if valid_end <= round_start:
                continue

            # Extract Q vectors for this round
            round_Q = Q[round_start:valid_end]  # (num_queries, head_dim)

            if invert_to_origin:
                # Invert each Q to position 0 (like Hybrid Frequency)
                positions = torch.arange(round_start, valid_end, dtype=round_Q.dtype)
                Q_relative = apply_rope_rotation(round_Q, -positions, inv_freq=inv_freq)
            else:
                # Rotate all Q to reference position
                Q_relative = apply_rope_rotation(round_Q, -ref_pos, inv_freq=inv_freq)

            trace_Q_relatives.append(Q_relative)

        # Concatenate this trace's vectors
        if trace_Q_relatives:
            trace_vectors = torch.cat(trace_Q_relatives, dim=0)
            all_Q_relatives_unnorm.append(trace_vectors)
            total_vectors += trace_vectors.shape[0]
            logger.info(f"  Collected {trace_vectors.shape[0]} vectors (total: {total_vectors})")

        # Free memory
        del Q, trace_Q_relatives

    # Stack all vectors from all traces
    Q_relatives_unnorm = torch.cat(all_Q_relatives_unnorm, dim=0)  # (total_vectors, head_dim)
    logger.info(f"Total Q_relative vectors from all traces: {Q_relatives_unnorm.shape[0]}")

    # Optionally L2 normalize for K-means clustering
    if use_l2_norm:
        Q_relatives = l2_normalize(Q_relatives_unnorm)
        logger.info("Using L2 normalized vectors for K-means")
    else:
        Q_relatives = Q_relatives_unnorm.clone()
        logger.info("Using unnormalized vectors for K-means (no L2 norm)")

    # Convert to float32 for sklearn
    Q_relatives = Q_relatives.float()
    Q_relatives_unnorm = Q_relatives_unnorm.float()

    # Run K-means clustering
    logger.info(f"Running K-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(Q_relatives.numpy())

    cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float()
    logger.info(f"K-means clustering completed. Cluster centers shape: {cluster_centers.shape}")
    logger.info(f"K-means inertia: {kmeans.inertia_:.4f}")

    if return_extras:
        cluster_labels = kmeans.labels_
        return cluster_centers, cluster_labels, Q_relatives_unnorm

    return cluster_centers


def main():
    """Main entry point for standalone execution."""
    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    # Use fixed settings (based on RoPE layout fix experiments)
    use_l2_norm = False
    invert_to_origin = False
    logger.info(f"Settings: use_l2_norm={use_l2_norm}, invert_to_origin={invert_to_origin}")

    cluster_centers = compute_kmeans_init_multi_trace(
        config, logger,
        use_l2_norm=use_l2_norm,
        invert_to_origin=invert_to_origin
    )

    # Save cluster centers
    output_path = exp_dir / 'output' / 'kmeans_init_multi_trace.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cluster_centers, output_path)
    logger.info(f"Saved cluster centers to: {output_path}")

    return cluster_centers


if __name__ == '__main__':
    main()
