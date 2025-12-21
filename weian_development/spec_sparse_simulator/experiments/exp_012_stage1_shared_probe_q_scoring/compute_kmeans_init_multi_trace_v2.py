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
import time
from pathlib import Path

import torch
import yaml
from fast_pytorch_kmeans import KMeans

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
                                     invert_to_origin=False, inv_freq=None,
                                     preloaded_traces=None):
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
        preloaded_traces: Optional list of preloaded trace dicts with 'Q', 'name', 'seq_len'
                         If provided, uses these instead of loading from disk

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

    round_window = config['training']['round_window']
    exclude_tail = config['training']['exclude_tail']

    # Collect Q_relative vectors from all traces
    all_Q_relatives_unnorm = []
    total_vectors = 0

    # Move inv_freq to GPU once
    inv_freq_gpu = inv_freq.cuda() if inv_freq is not None else None

    # Determine data source
    if preloaded_traces is not None:
        # Use preloaded traces (already on GPU)
        logger.info(f"Using {len(preloaded_traces)} preloaded traces for K-means init")
        traces_to_process = preloaded_traces
        use_preloaded = True
    else:
        # Load from disk (legacy path)
        head_sample_path = exp_dir / config['data']['head_sample_file']
        if not head_sample_path.exists():
            raise FileNotFoundError(f"Head sample file not found: {head_sample_path}")

        with open(head_sample_path, 'r') as f:
            head_samples = json.load(f)

        layer, head = head_samples[0]
        logger.info(f"Using head [layer={layer}, head={head}] for K-means initialization")

        test_trace_path = exp_dir / config['data']['test_trace_path']
        trace_paths = get_training_trace_paths(test_trace_path, logger)

        if len(trace_paths) == 0:
            raise ValueError("No training traces found!")

        traces_to_process = trace_paths
        use_preloaded = False

    # Step 1: Collect Q_relative vectors from all traces
    step1_start = time.time()
    for trace_idx, trace_item in enumerate(traces_to_process):
        if use_preloaded:
            # Use preloaded trace data
            trace_name = trace_item['name']
            Q = trace_item['Q']  # Already on GPU
            seq_len = trace_item['seq_len']
            logger.info(f"[{trace_idx+1}/{len(traces_to_process)}] Processing {trace_name}...")
        else:
            # Load from disk
            trace_path = trace_item
            logger.info(f"[{trace_idx+1}/{len(traces_to_process)}] Processing {trace_path.parent.name}...")
            qk_data = torch.load(trace_path, map_location='cuda')
            Q = qk_data['q'][layer, head]  # (seq_len, head_dim), already on GPU
            del qk_data
            seq_len = Q.shape[0]

        head_dim = Q.shape[1]

        # Vectorized: compute all valid indices at once
        # Skip first round (round_start=0) and exclude tail
        valid_start = round_window  # First valid round starts at round_window
        valid_end_seq = seq_len - exclude_tail

        if valid_start >= valid_end_seq:
            logger.info(f"  Skipped (too short)")
            del Q
            continue

        # Extract all valid Q vectors at once
        Q_valid = Q[valid_start:valid_end_seq]  # (num_valid, head_dim)
        num_valid = Q_valid.shape[0]

        if invert_to_origin:
            # Invert each Q to position 0
            positions = torch.arange(valid_start, valid_end_seq, device='cuda', dtype=Q_valid.dtype)
            Q_relative = apply_rope_rotation(Q_valid, -positions, inv_freq=inv_freq_gpu)
        else:
            # Compute reference positions for each query (vectorized)
            # Each query at position p belongs to round (p // round_window)
            # Reference position = round_start + round_window // 2
            query_positions = torch.arange(valid_start, valid_end_seq, device='cuda')
            round_starts = (query_positions // round_window) * round_window
            ref_positions = round_starts + round_window // 2
            # Rotation amount: -(ref_pos - 0) = -ref_pos
            Q_relative = apply_rope_rotation(Q_valid, -ref_positions, inv_freq=inv_freq_gpu)

        # Move result to CPU to save GPU memory (non_blocking for async transfer)
        all_Q_relatives_unnorm.append(Q_relative.cpu(memory_format=torch.contiguous_format))
        total_vectors += num_valid
        logger.info(f"  Collected {num_valid} vectors (total: {total_vectors})")

        # Free GPU memory (only delete Q if loaded from disk, not if preloaded)
        del Q_valid, Q_relative
        if not use_preloaded:
            del Q

    # Stack all vectors from all traces
    Q_relatives_unnorm = torch.cat(all_Q_relatives_unnorm, dim=0)  # (total_vectors, head_dim)
    step1_time = time.time() - step1_start
    logger.info(f"Step 1: Collected {Q_relatives_unnorm.shape[0]} Q_relative vectors in {step1_time:.1f}s")

    # Step 2: Optionally L2 normalize for K-means clustering
    step2_start = time.time()
    if use_l2_norm:
        Q_relatives = l2_normalize(Q_relatives_unnorm)
        logger.info("Using L2 normalized vectors for K-means")
    else:
        Q_relatives = Q_relatives_unnorm.clone()
        logger.info("Using unnormalized vectors for K-means (no L2 norm)")

    # Convert to float32 for K-means
    Q_relatives = Q_relatives.float()
    Q_relatives_unnorm = Q_relatives_unnorm.float()

    # Step 3: Run K-means clustering on GPU
    step3_start = time.time()
    logger.info(f"Step 3: Running K-means with {n_clusters} clusters on GPU...")
    kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', max_iter=100)
    # Move to GPU for fast clustering
    Q_relatives_gpu = Q_relatives.cuda()
    kmeans.fit(Q_relatives_gpu)

    cluster_centers = kmeans.centroids.cpu().float()
    step3_time = time.time() - step3_start
    logger.info(f"Step 3: K-means completed in {step3_time:.1f}s. Cluster centers shape: {cluster_centers.shape}")

    if return_extras:
        # Get cluster labels by predicting on the data
        cluster_labels = kmeans.predict(Q_relatives_gpu).cpu().numpy()
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
