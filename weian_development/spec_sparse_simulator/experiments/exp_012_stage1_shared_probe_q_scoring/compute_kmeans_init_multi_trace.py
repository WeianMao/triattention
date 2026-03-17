"""
K-means Initialization for Probe Vectors (Multi-Trace).

Computes K-means cluster centers from Q vectors across ALL training traces
to initialize shared probe vectors with better coverage of the data distribution.

Core algorithm:
1. Load all training traces (same as train_multi_trace.py)
2. For each trace, extract Q vectors and compute Q_relative = RoPE(Q_post, -ref_pos)
3. L2 normalize all Q_relative to unit norm (Stage 3 alignment)
4. Combine Q_relative from all traces
5. Run K-means clustering on combined data
6. Return cluster centers as probe initialization

Based on compute_kmeans_init.py with multi-trace support.
"""

import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from sklearn.cluster import KMeans


def apply_rope_rotation(vectors, position, base=10000):
    """
    Apply RoPE rotation to vectors.

    Args:
        vectors: Tensor of shape (num_vectors, head_dim) or (head_dim,)
        position: Target position for rotation (scalar or tensor)
        base: RoPE base (default: 10000)

    Returns:
        rotated_vectors: Same shape as input
    """
    is_single = vectors.dim() == 1
    if is_single:
        vectors = vectors.unsqueeze(0)

    num_vectors, head_dim = vectors.shape
    num_freqs = head_dim // 2

    # Compute angular frequencies: omega_j = 1 / (base^(2j/d))
    dim_indices = torch.arange(num_freqs, device=vectors.device, dtype=vectors.dtype)
    omega = 1.0 / (base ** (2 * dim_indices / head_dim))

    # Compute rotation angles: theta_j = pos * omega_j
    if isinstance(position, (int, float)):
        theta = position * omega  # shape: (num_freqs,)
    else:
        theta = position.unsqueeze(-1) * omega  # shape: (num_vectors, num_freqs)

    # Compute cos and sin
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Reshape vectors to complex pairs: (num_vectors, num_freqs, 2)
    vectors_complex = vectors.view(num_vectors, num_freqs, 2)

    # Apply 2D rotation
    if theta.dim() == 1:
        # Same rotation for all vectors
        rotated = torch.stack([
            vectors_complex[..., 0] * cos_theta - vectors_complex[..., 1] * sin_theta,
            vectors_complex[..., 0] * sin_theta + vectors_complex[..., 1] * cos_theta
        ], dim=-1)
    else:
        # Per-vector rotation
        rotated = torch.stack([
            vectors_complex[..., 0] * cos_theta - vectors_complex[..., 1] * sin_theta,
            vectors_complex[..., 0] * sin_theta + vectors_complex[..., 1] * cos_theta
        ], dim=-1)

    # Restore original shape
    rotated = rotated.view(num_vectors, head_dim)

    if is_single:
        rotated = rotated.squeeze(0)

    return rotated


def l2_normalize(x, eps=1e-8):
    """L2 normalize vectors to unit norm."""
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    return x / (norm + eps)


def get_training_trace_paths(config, logger):
    """
    Get all training trace paths (excluding test trace).

    Returns:
        List of Path objects for training traces
    """
    traces_dir = Path("/data/rbg/users/weian/project/rl/dc/outputs/deepseek_r1_qwen3_8b/qk_bf16_traces")

    # Test trace to exclude
    exp_dir = Path(__file__).parent
    test_trace_path = exp_dir / config['data']['test_trace_path']
    test_trace_resolved = test_trace_path.resolve()

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


def compute_kmeans_init_multi_trace(config, logger, n_clusters=128, random_state=42):
    """
    Compute K-means initialization using Q vectors from all training traces.

    Algorithm:
    1. Load all training traces
    2. For each trace:
       a. For each round, compute reference position: ref_pos = round_start + round_window / 2
       b. For each Q in the round, compute Q_relative = RoPE(Q_post, -ref_pos)
       c. L2 normalize Q_relative to unit norm (Stage 3 alignment)
    3. Combine all Q_relative vectors from all traces
    4. Run K-means to get cluster centers

    Args:
        config: Configuration dict
        logger: Logger instance
        n_clusters: Number of clusters (default: 128)
        random_state: Random seed for K-means

    Returns:
        cluster_centers: Tensor of shape (n_clusters, head_dim)
    """
    exp_dir = Path(__file__).parent

    # Load head sample file
    head_sample_path = exp_dir / config['data']['head_sample_file']
    if not head_sample_path.exists():
        raise FileNotFoundError(f"Head sample file not found: {head_sample_path}")

    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)

    # Select first head
    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}] for K-means initialization")

    # Get training trace paths
    trace_paths = get_training_trace_paths(config, logger)

    round_window = config['training']['round_window']
    exclude_tail = config['training']['exclude_tail']

    # Collect Q_relative vectors from all traces
    all_Q_relatives = []
    total_vectors = 0

    for i, trace_path in enumerate(trace_paths):
        logger.info(f"[{i+1}/{len(trace_paths)}] Processing {trace_path.parent.name}...")

        # Load trace and extract only required head
        qk_data = torch.load(trace_path, map_location='cpu')
        Q = qk_data['q'][layer, head]  # (seq_len, head_dim)
        del qk_data  # Free memory

        seq_len, head_dim = Q.shape

        # Collect Q_relative vectors for this trace
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

            # Compute Q_relative = RoPE(Q_post, -ref_pos)
            Q_relative = apply_rope_rotation(round_Q, -ref_pos)

            # L2 normalize to unit norm (Stage 3 alignment)
            Q_relative = l2_normalize(Q_relative)

            trace_Q_relatives.append(Q_relative)

        if trace_Q_relatives:
            trace_Q_concat = torch.cat(trace_Q_relatives, dim=0)
            all_Q_relatives.append(trace_Q_concat)
            total_vectors += trace_Q_concat.shape[0]
            logger.info(f"       {trace_Q_concat.shape[0]} vectors (total: {total_vectors})")

        del Q  # Free memory

    # Stack all Q_relative vectors
    Q_relatives = torch.cat(all_Q_relatives, dim=0)  # (total_vectors, head_dim)
    logger.info(f"Total Q_relative vectors from all traces: {Q_relatives.shape[0]}")

    # Convert to float32 for sklearn (may be bfloat16)
    Q_relatives = Q_relatives.float()

    # Run K-means clustering
    logger.info(f"Running K-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(Q_relatives.numpy())

    cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float()  # (n_clusters, head_dim)
    logger.info(f"K-means clustering completed. Cluster centers shape: {cluster_centers.shape}")

    # Compute inertia (sum of squared distances)
    logger.info(f"K-means inertia: {kmeans.inertia_:.4f}")

    return cluster_centers


def main():
    """Main entry point for standalone execution."""
    # Load configuration
    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    # Compute K-means initialization
    n_clusters = config['model']['num_bins']
    cluster_centers = compute_kmeans_init_multi_trace(config, logger, n_clusters=n_clusters)

    # Save cluster centers
    output_path = exp_dir / 'output' / 'kmeans_init_multi_trace.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cluster_centers, output_path)
    logger.info(f"Saved cluster centers to: {output_path}")

    return cluster_centers


if __name__ == '__main__':
    main()
