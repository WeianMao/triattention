"""
K-means Initialization for Probe Vectors.

Computes K-means cluster centers from Q vectors in relative space
to initialize shared probe vectors.

Core algorithm:
1. Load training data (Q, round positions)
2. For each Q, compute Q_relative = RoPE(Q_post, -ref_pos)
3. L2 normalize Q_relative to unit norm (Stage 3 alignment)
4. Run K-means clustering on all normalized Q_relative vectors
5. Return cluster centers as probe initialization

Reference: exp_012_asymmetric_probe_network.md
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


def compute_kmeans_init(config, logger, n_clusters=128, random_state=42):
    """
    Compute K-means initialization for probe vectors.

    Algorithm:
    1. Load training data
    2. For each round, compute reference position: ref_pos = round_start + round_window / 2
    3. For each Q in the round, compute Q_relative = RoPE(Q_post, -ref_pos)
    4. L2 normalize Q_relative to unit norm (Stage 3 alignment)
    5. Collect all normalized Q_relative vectors
    6. Run K-means to get cluster centers (already unit norm)

    Args:
        config: Configuration dict
        logger: Logger instance
        n_clusters: Number of clusters (default: 128)
        random_state: Random seed for K-means

    Returns:
        cluster_centers: Tensor of shape (n_clusters, head_dim)
    """
    exp_dir = Path(__file__).parent
    trace_path = exp_dir / config['data']['trace_path']

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

    # Select first head
    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}] for K-means initialization")

    # Extract Q for selected head
    Q = qk_data['q'][layer, head]  # (seq_len, head_dim)
    seq_len, head_dim = Q.shape
    logger.info(f"Q shape: {Q.shape}")

    round_window = config['training']['round_window']
    exclude_tail = config['training']['exclude_tail']

    # Collect Q_relative vectors
    Q_relatives = []

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

        Q_relatives.append(Q_relative)

    # Stack all Q_relative vectors
    Q_relatives = torch.cat(Q_relatives, dim=0)  # (total_queries, head_dim)
    logger.info(f"Total Q_relative vectors: {Q_relatives.shape[0]}")

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
    cluster_centers = compute_kmeans_init(config, logger)

    # Save cluster centers
    output_path = exp_dir / 'output' / 'kmeans_init.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cluster_centers, output_path)
    logger.info(f"Saved cluster centers to: {output_path}")

    return cluster_centers


if __name__ == '__main__':
    main()
