"""Magnitude distribution visualization for Query and Key across all frequency bands.

展示测试集中每个 Query 和 Key 在所有 64 个频段上的模长分布直方图，
用红色标记 248 个错误点对应的模长位置。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from viz_utils import (
    load_miss_cases,
    load_qk_data,
    load_query_data,
    to_complex_pairs,
    setup_matplotlib
)


def compute_mean_magnitude(tensor: torch.Tensor) -> np.ndarray:
    """Compute mean magnitude across all 64 frequency bands for each position.

    Args:
        tensor: [seq_len, 128] tensor (post-RoPE rotated)

    Returns:
        np.ndarray of shape [seq_len] with mean magnitude per position
    """
    # Convert to complex: [seq_len, 128] -> [seq_len, 64]
    complex_tensor = to_complex_pairs(tensor)

    # Compute magnitude for each frequency: [seq_len, 64]
    magnitudes = torch.abs(complex_tensor)

    # Mean across all frequencies: [seq_len]
    mean_mag = magnitudes.mean(dim=1)

    return mean_mag.numpy()


def get_failure_indices(miss_cases: dict) -> tuple[set[int], set[int]]:
    """Extract failure indices for both Key and Query.

    Returns:
        (failure_key_indices, failure_query_indices)
    """
    key_indices = {case['argmax_key'] for case in miss_cases['miss_cases']}
    query_indices = {case['query_idx'] for case in miss_cases['miss_cases']}
    return key_indices, query_indices


def plot_magnitude_distribution(
    k_head: torch.Tensor,
    q_head: torch.Tensor,
    failure_key_indices: set[int],
    failure_query_indices: set[int],
    output_path: Path,
    layer: int,
    head: int
) -> None:
    """Create 1×2 subplot showing magnitude distribution for Key and Query.

    Left: Key magnitude histogram (all in blue, errors in red overlay)
    Right: Query magnitude histogram (all in blue, errors in red overlay)
    """
    # Compute magnitudes
    k_magnitudes = compute_mean_magnitude(k_head)
    q_magnitudes = compute_mean_magnitude(q_head)

    # Extract error magnitudes
    k_error_indices = [i for i in failure_key_indices if i < len(k_magnitudes)]
    q_error_indices = [i for i in failure_query_indices if i < len(q_magnitudes)]

    k_error_mags = k_magnitudes[k_error_indices] if k_error_indices else np.array([])
    q_error_mags = q_magnitudes[q_error_indices] if q_error_indices else np.array([])

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)

    # Determine common bin edges for each subplot
    k_bins = np.linspace(k_magnitudes.min(), k_magnitudes.max(), 101)
    q_bins = np.linspace(q_magnitudes.min(), q_magnitudes.max(), 101)

    # Left subplot: Key magnitude distribution
    ax = axes[0]
    ax.hist(k_magnitudes, bins=k_bins, color='blue', alpha=0.6,
            label=f'All Keys (N={len(k_magnitudes)})', edgecolor='none')
    if len(k_error_mags) > 0:
        ax.hist(k_error_mags, bins=k_bins, color='red', alpha=0.8,
                label=f'Error Keys (N={len(k_error_mags)})', edgecolor='none')
    ax.set_xlabel('Mean Magnitude (across 64 freq bands)')
    ax.set_ylabel('Count')
    ax.set_title(f'Key Magnitude Distribution - L{layer} H{head}')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')

    # Right subplot: Query magnitude distribution
    ax = axes[1]
    ax.hist(q_magnitudes, bins=q_bins, color='blue', alpha=0.6,
            label=f'All Queries (N={len(q_magnitudes)})', edgecolor='none')
    if len(q_error_mags) > 0:
        ax.hist(q_error_mags, bins=q_bins, color='red', alpha=0.8,
                label=f'Error Queries (N={len(q_error_mags)})', edgecolor='none')
    ax.set_xlabel('Mean Magnitude (across 64 freq bands)')
    ax.set_ylabel('Count')
    ax.set_title(f'Query Magnitude Distribution - L{layer} H{head}')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')

    fig.suptitle(
        f'Magnitude Distribution (Post-RoPE) - Layer {layer} Head {head}\n'
        f'248 error cases marked in red',
        fontsize=14
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize magnitude distribution for Query and Key'
    )
    parser.add_argument(
        '--layer-head',
        required=True,
        help='Format: layer-head, e.g., 15-20'
    )
    args = parser.parse_args()

    layer, head = map(int, args.layer_head.split('-'))

    # Setup paths
    base_dir = Path(__file__).parent.parent
    qk_path = base_dir / 'data' / 'qk_test.pt'
    analysis_path = base_dir / 'output' / 'analysis' / 'miss_case_analysis.json'
    output_dir = base_dir / 'output' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'magnitude_distribution_L{layer}_H{head}.png'

    # Setup matplotlib
    setup_matplotlib()

    # Load data
    miss_cases = load_miss_cases(analysis_path)
    failure_key_indices, failure_query_indices = get_failure_indices(miss_cases)
    k_head = load_qk_data(qk_path, layer, head)
    q_head = load_query_data(qk_path, layer, head)

    # Generate visualization
    plot_magnitude_distribution(
        k_head, q_head,
        failure_key_indices, failure_query_indices,
        output_path, layer, head
    )


if __name__ == '__main__':
    main()
