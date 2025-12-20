"""Per-round failure analysis visualization (3 rounds × 6 frequencies, post-RoPE).

This script creates a 3×6 subplot grid showing:
- 3 randomly selected rounds with failures (rows)
- 6 top frequencies per round (columns)
- Each subplot: Re(k) vs Im(k) scatter plot in POST-ROPE complex plane
  - Blue small dots (s=10, alpha=0.5): all keys in historical context
  - Red large dots (s=30, alpha=0.8): failure cases' argmax_key

Usage:
    python visualizations/viz_per_round_analysis.py --layer-head 15-20 --seed 42

Output:
    output/visualizations/per_round_analysis_L{layer}_H{head}_seed{seed}.png
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# Add current directory to path for viz_utils import
sys.path.insert(0, str(Path(__file__).parent))
from viz_utils import load_miss_cases, load_qk_data, to_complex_pairs, get_top_k_frequencies


def select_failed_rounds(
    miss_cases: dict,
    round_window: int = 128,
    seed: int = 42,
    n_rounds: int = 3
) -> tuple[list[int], dict[int, list[dict]]]:
    """Randomly select n_rounds from rounds with at least 1 failure.

    Args:
        miss_cases: dict from miss_case_analysis.json with 'miss_cases' list
        round_window: size of each round in tokens (default: 128)
        seed: random seed for reproducibility
        n_rounds: number of rounds to select (default: 3)

    Returns:
        tuple of (selected_rounds, round_failures) where:
            selected_rounds: sorted list of round indices
            round_failures: dict mapping round_idx -> list of failure cases
    """
    # Group failures by round
    round_failures = defaultdict(list)
    for case in miss_cases['miss_cases']:
        round_idx = case['query_idx'] // round_window
        round_failures[round_idx].append(case)

    # Randomly select n_rounds from rounds with failures
    failed_rounds = list(round_failures.keys())
    random.seed(seed)
    selected = random.sample(failed_rounds, min(n_rounds, len(failed_rounds)))

    return sorted(selected), round_failures


def plot_round_frequencies(
    axes_row,
    k_head: torch.Tensor,
    round_idx: int,
    round_failures: list[dict],
    round_window: int = 128
) -> None:
    """Plot 6 frequency subplots for a single round.

    Creates 6 subplots showing Re(k) vs Im(k) for top-6 frequencies:
    - Blue small dots: all historical keys
    - Red large dots: failure cases' argmax_key

    Args:
        axes_row: array of 6 matplotlib axes for this row
        k_head: [seq_len, head_dim=128] key tensor (post-RoPE)
        round_idx: round index
        round_failures: list of failure cases in this round
        round_window: size of each round in tokens (default: 128)
    """
    # Get round boundaries
    round_start = round_idx * round_window

    # Historical keys: all keys before this round (what the model can attend to)
    k_historical = k_head[:round_start]  # [round_start, head_dim]
    if len(k_historical) == 0:
        return  # Skip first round with no history

    # Convert to complex
    k_complex = to_complex_pairs(k_historical)  # [round_start, 64]

    # Find top-6 frequencies
    top_freq_indices = get_top_k_frequencies(k_complex, k=6)

    # Get failure key indices for this round
    failure_key_indices = {case['argmax_key'] for case in round_failures}
    is_failure = torch.zeros(len(k_historical), dtype=torch.bool)
    for idx in failure_key_indices:
        if idx < len(k_historical):
            is_failure[idx] = True

    # Plot each of top-6 frequencies
    for col, freq_idx in enumerate(top_freq_indices):
        ax = axes_row[col]
        k_freq = k_complex[:, freq_idx]

        # Plot all keys in blue
        ax.scatter(
            k_freq[~is_failure].real.numpy(),
            k_freq[~is_failure].imag.numpy(),
            c='blue',
            s=10,
            alpha=0.5,
            edgecolors='none'
        )

        # Plot failure keys in red
        if is_failure.any():
            ax.scatter(
                k_freq[is_failure].real.numpy(),
                k_freq[is_failure].imag.numpy(),
                c='red',
                s=30,
                alpha=0.8,
                edgecolors='none'
            )

        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Freq {freq_idx}')
        ax.grid(alpha=0.2, linestyle='--')


def main():
    """Main entry point for per-round failure analysis visualization."""
    parser = argparse.ArgumentParser(
        description='Per-round failure analysis visualization (3 rounds × 6 frequencies, post-RoPE)'
    )
    parser.add_argument(
        '--layer-head',
        required=True,
        help='Layer-head pair in format: layer-head (e.g., 15-20)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for round selection (default: 42)'
    )
    args = parser.parse_args()

    # Parse layer-head
    layer, head = map(int, args.layer_head.split('-'))

    # Setup paths
    base_dir = Path(__file__).parent.parent
    qk_path = base_dir / 'data' / 'qk_test.pt'
    analysis_path = base_dir / 'output' / 'analysis' / 'miss_case_analysis.json'
    output_dir = base_dir / 'output' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'per_round_analysis_L{layer}_H{head}_seed{args.seed}.png'

    # Load data
    miss_cases = load_miss_cases(analysis_path)
    k_head = load_qk_data(qk_path, layer, head)

    # Select 3 random failed rounds
    selected_rounds, round_failures = select_failed_rounds(miss_cases, seed=args.seed)

    # Create 3×6 subplot grid
    fig, axes = plt.subplots(3, 6, figsize=(24, 12), dpi=100)

    for row, round_idx in enumerate(selected_rounds):
        failures = round_failures[round_idx]
        plot_round_frequencies(axes[row], k_head, round_idx, failures, round_window=128)
        # Add row label
        axes[row, 0].set_ylabel(
            f'Round {round_idx}\n({len(failures)} failures)',
            fontsize=10
        )

    fig.suptitle(
        f'Per-Round Failure Analysis (Post-RoPE) - Layer {layer} Head {head}\n'
        f'3 random failed rounds × 6 top frequencies',
        fontsize=14
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
