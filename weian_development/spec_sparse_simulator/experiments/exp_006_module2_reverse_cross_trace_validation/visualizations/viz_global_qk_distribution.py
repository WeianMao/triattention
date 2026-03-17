"""Global QK distribution visualization showing top-6 frequencies in post-RoPE space.

This script creates a 2x3 subplot grid visualizing the distribution of keys in
post-RoPE complex space for the top-6 frequencies by mean magnitude.
Failure cases (248 argmax_key indices from miss_case_analysis.json) are highlighted in red.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from viz_utils import (
    load_miss_cases,
    load_qk_data,
    to_complex_pairs,
    get_top_k_frequencies,
    setup_matplotlib
)


def get_failure_key_indices(miss_cases: dict) -> set[int]:
    """Extract argmax_key indices from miss cases.

    Args:
        miss_cases: dict containing 'miss_cases' list

    Returns:
        set of key indices that were ground truth for failed queries
    """
    return {case['argmax_key'] for case in miss_cases['miss_cases']}


def plot_global_distribution(
    k_head: torch.Tensor,
    failure_key_indices: set[int],
    output_path: Path,
    layer: int,
    head: int
) -> None:
    """Create 2x3 subplot grid showing top-6 frequencies in post-RoPE complex space.

    Args:
        k_head: [seq_len, 128] tensor of keys (already post-RoPE rotated)
        failure_key_indices: set of key indices to highlight in red
        output_path: path to save output PNG
        layer: layer index for title
        head: head index for title
    """
    # Convert to complex representation: [seq, 128] -> [seq, 64]
    k_complex = to_complex_pairs(k_head)

    # Find top-6 frequencies by mean magnitude
    top_freq_indices = get_top_k_frequencies(k_complex, k=6)
    k_magnitude = torch.abs(k_complex.mean(dim=0))

    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=100)
    axes = axes.flatten()

    # Create failure mask
    seq_len = k_complex.shape[0]
    is_failure = torch.zeros(seq_len, dtype=torch.bool)
    for idx in failure_key_indices:
        if idx < seq_len:
            is_failure[idx] = True

    # Plot each top-6 frequency
    for ax_idx, freq_idx in enumerate(top_freq_indices):
        ax = axes[ax_idx]
        k_freq = k_complex[:, freq_idx]  # [seq] complex

        # Plot all keys in blue
        ax.scatter(
            k_freq[~is_failure].real.numpy(),
            k_freq[~is_failure].imag.numpy(),
            c='blue',
            s=10,
            alpha=0.5,
            label='All keys',
            edgecolors='none'
        )

        # Plot failure keys in red (on top)
        if is_failure.any():
            ax.scatter(
                k_freq[is_failure].real.numpy(),
                k_freq[is_failure].imag.numpy(),
                c='red',
                s=30,
                alpha=0.8,
                label=f'Failures ({is_failure.sum().item()})',
                edgecolors='none'
            )

        # Formatting
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Re(k)')
        ax.set_ylabel('Im(k)')
        ax.set_title(f'Freq {freq_idx} (mag={k_magnitude[freq_idx]:.2f})')
        ax.grid(alpha=0.2, linestyle='--')
        if ax_idx == 0:
            ax.legend(loc='upper right')

    fig.suptitle(
        f'Global QK Distribution (Post-RoPE) - Layer {layer} Head {head}\n'
        f'248 failure keys marked in red',
        fontsize=14
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Visualize global QK distribution for top-6 frequencies'
    )
    parser.add_argument(
        '--layer-head',
        required=True,
        help='Format: layer-head, e.g., 15-20'
    )
    args = parser.parse_args()

    # Parse layer-head argument
    layer, head = map(int, args.layer_head.split('-'))

    # Setup paths
    base_dir = Path(__file__).parent.parent
    qk_path = base_dir / 'data' / 'qk_test.pt'
    analysis_path = base_dir / 'output' / 'analysis' / 'miss_case_analysis.json'
    output_dir = base_dir / 'output' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'global_qk_distribution_L{layer}_H{head}.png'

    # Setup matplotlib
    setup_matplotlib()

    # Load data
    miss_cases = load_miss_cases(analysis_path)
    failure_key_indices = get_failure_key_indices(miss_cases)
    k_head = load_qk_data(qk_path, layer, head)

    # Generate visualization
    plot_global_distribution(k_head, failure_key_indices, output_path, layer, head)


if __name__ == '__main__':
    main()
