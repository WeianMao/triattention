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

Optional magnitude weight initialization (--use-magnitude-init):
For each cluster b, compute:
- mean_of_magnitude[b, f] = mean(sqrt(x_{2f}^2 + x_{2f+1}^2))
- magnitude_of_mean[b, f] = sqrt(mean_x_{2f}^2 + mean_x_{2f+1}^2)
- init[b, f] = mean_of_magnitude[b, f] - magnitude_of_mean[b, f]

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


def compute_magnitude_features(x, num_freqs, eps=1e-8):
    """
    Compute per-frequency magnitude features.

    m_f = sqrt(x_{2f}^2 + x_{2f+1}^2 + eps)

    Args:
        x: Tensor of shape (..., head_dim) where head_dim = num_freqs * 2
        num_freqs: Number of frequency components (F = head_dim // 2)
        eps: Small value for numerical stability

    Returns:
        magnitude: Tensor of shape (..., num_freqs)
    """
    original_shape = x.shape[:-1]
    head_dim = x.shape[-1]
    x_freq = x.view(*original_shape, num_freqs, 2)
    magnitude = torch.sqrt(torch.sum(x_freq ** 2, dim=-1) + eps)
    return magnitude


def compute_magnitude_init(Q_relatives_unnorm, cluster_labels, n_clusters, num_freqs, eps=1e-8):
    """
    Compute magnitude weight initialization for each cluster.

    For each cluster b:
    - mean_of_magnitude[b, f] = mean(sqrt(x_{2f}^2 + x_{2f+1}^2))  over cluster points
    - magnitude_of_mean[b, f] = sqrt(mean_x_{2f}^2 + mean_x_{2f+1}^2)  of cluster mean
    - init[b, f] = mean_of_magnitude[b, f] - magnitude_of_mean[b, f]

    This follows the pattern from hybrid frequency baseline where:
    extra = (q_abs_mean - q_mean_abs) * k_abs

    Args:
        Q_relatives_unnorm: Tensor of shape (n_points, head_dim) - NOT L2 normalized
        cluster_labels: Array of shape (n_points,) with cluster assignments
        n_clusters: Number of clusters
        num_freqs: Number of frequency components (head_dim // 2)
        eps: Small value for numerical stability

    Returns:
        magnitude_init: Tensor of shape (n_clusters, num_freqs)
    """
    head_dim = Q_relatives_unnorm.shape[1]
    magnitude_init = torch.zeros(n_clusters, num_freqs, dtype=Q_relatives_unnorm.dtype)

    # Convert cluster_labels to tensor if needed
    if not isinstance(cluster_labels, torch.Tensor):
        cluster_labels = torch.from_numpy(cluster_labels)

    for b in range(n_clusters):
        # Find points belonging to cluster b
        mask = cluster_labels == b
        cluster_vectors = Q_relatives_unnorm[mask]  # (n_points_in_cluster, head_dim)

        if cluster_vectors.shape[0] == 0:
            continue

        # Reshape to frequency pairs: (n_points, num_freqs, 2)
        vectors_freq = cluster_vectors.view(-1, num_freqs, 2)

        # Compute magnitude for each point: (n_points, num_freqs)
        magnitudes = torch.sqrt(vectors_freq[..., 0]**2 + vectors_freq[..., 1]**2 + eps)

        # mean_of_magnitude: average magnitude across points (num_freqs,)
        mean_of_mag = magnitudes.mean(dim=0)

        # Compute mean vector of the cluster: (head_dim,)
        mean_vector = cluster_vectors.mean(dim=0)

        # Reshape mean vector to frequency pairs: (num_freqs, 2)
        mean_vector_freq = mean_vector.view(num_freqs, 2)

        # magnitude_of_mean: magnitude of the mean vector (num_freqs,)
        mag_of_mean = torch.sqrt(mean_vector_freq[..., 0]**2 + mean_vector_freq[..., 1]**2 + eps)

        # init = mean_of_magnitude - magnitude_of_mean
        magnitude_init[b] = mean_of_mag - mag_of_mean

    return magnitude_init


def compute_kmeans_init(config, logger, n_clusters=128, random_state=42, return_extras=False,
                        use_l2_norm=True):
    """
    Compute K-means initialization for probe vectors.

    Algorithm:
    1. Load training data
    2. For each round, compute reference position: ref_pos = round_start + round_window / 2
    3. For each Q in the round, compute Q_relative = RoPE(Q_post, -ref_pos)
    4. Optionally L2 normalize Q_relative to unit norm (Stage 3 alignment)
    5. Collect all Q_relative vectors
    6. Run K-means to get cluster centers

    Args:
        config: Configuration dict
        logger: Logger instance
        n_clusters: Number of clusters (default: 128)
        random_state: Random seed for K-means
        return_extras: If True, also return cluster labels and unnormalized Q_relatives
                       for magnitude initialization
        use_l2_norm: If True (default), L2 normalize vectors before K-means

    Returns:
        If return_extras=False:
            cluster_centers: Tensor of shape (n_clusters, head_dim)
        If return_extras=True:
            tuple of (cluster_centers, cluster_labels, Q_relatives_unnorm)
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

    # Collect Q_relative vectors (both normalized and unnormalized)
    Q_relatives_unnorm_list = []

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

        # Store unnormalized version for magnitude init
        Q_relatives_unnorm_list.append(Q_relative.clone())

    # Stack all unnormalized Q_relative vectors
    Q_relatives_unnorm = torch.cat(Q_relatives_unnorm_list, dim=0)  # (total_queries, head_dim)

    # Optionally L2 normalize for K-means clustering
    if use_l2_norm:
        Q_relatives = l2_normalize(Q_relatives_unnorm)
        logger.info("Using L2 normalized vectors for K-means")
    else:
        Q_relatives = Q_relatives_unnorm.clone()
        logger.info("Using unnormalized vectors for K-means (no L2 norm)")

    logger.info(f"Total Q_relative vectors: {Q_relatives.shape[0]}")

    # Convert to float32 for sklearn (may be bfloat16)
    Q_relatives = Q_relatives.float()
    Q_relatives_unnorm = Q_relatives_unnorm.float()

    # Run K-means clustering
    logger.info(f"Running K-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(Q_relatives.numpy())

    cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float()  # (n_clusters, head_dim)
    logger.info(f"K-means clustering completed. Cluster centers shape: {cluster_centers.shape}")

    # Compute inertia (sum of squared distances)
    logger.info(f"K-means inertia: {kmeans.inertia_:.4f}")

    if return_extras:
        # Return extras for magnitude initialization
        cluster_labels = kmeans.labels_  # numpy array of shape (n_points,)
        return cluster_centers, cluster_labels, Q_relatives_unnorm

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
