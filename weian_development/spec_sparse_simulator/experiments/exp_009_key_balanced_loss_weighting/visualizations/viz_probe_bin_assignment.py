"""Probe/Bin Assignment Visualization for Module 2

Visualizes how keys and queries are assigned to bins in POST-ROPE space.
Creates 2x6 subplot layout:
- Row 1: Key bin assignments for top-6 frequencies
- Row 2: Query bin assignments for top-6 frequencies

Each subplot shows Re vs Im colored by bin assignment:
- Top-10 bins by usage count: distinct colors (tab10 colormap)
- For each top-10 bin: show only top-10 keys/queries by softmax probability
- Keys/queries NOT in any top-10 bin: GRAY color (s=8, alpha=0.3)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

# Add experiment directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import Module2Network, create_model

# Add visualizations directory to path
sys.path.insert(0, str(Path(__file__).parent))
from viz_utils import load_qk_data, to_complex_pairs, get_top_k_frequencies


def load_model(config_path: Path, checkpoint_path: Path):
    """Load trained model from checkpoint.

    Args:
        config_path: Path to config.yaml
        checkpoint_path: Path to best_model.pt

    Returns:
        Tuple of (model, config)
    """
    config = yaml.safe_load(open(config_path))
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def compute_bin_assignments(model, k_historical, q_round, round_start, round_window):
    """Compute bin assignments for historical keys and round queries.

    Args:
        model: Module2Network instance
        k_historical: Historical keys [num_historical, head_dim]
        q_round: Queries for current round [round_window, head_dim]
        round_start: Starting position of current round
        round_window: Round window size

    Returns:
        Tuple of (key_bins, key_probs, query_bins, query_probs)
        - key_bins: [num_historical] argmax bin for each key
        - key_probs: [num_historical, num_bins] softmax probabilities
        - query_bins: [round_window] argmax bin for each query
        - query_probs: [round_window, num_bins] softmax probabilities
    """
    with torch.no_grad():
        # Compute reference angles for this round
        ref_angles = model.compute_reference_angles(round_start, round_window)

        # Key bin probabilities [num_historical, 128]
        # forward_keys returns key_probs after softmax(dim=0)
        # But we need per-key bin probabilities, so call forward directly
        key_logits = model.key_network(k_historical, ref_angles)
        key_probs = F.softmax(key_logits, dim=-1)  # Softmax over bins for each key
        key_bins = key_probs.argmax(dim=-1)  # [num_historical]

        # Query bin probabilities [round_window, 128]
        # Compute empty_bin_mask: bins with no keys assigned
        bin_counts = torch.bincount(key_bins, minlength=128)
        empty_bin_mask = (bin_counts == 0)  # [128]

        # forward_queries returns bin_probs after softmax(dim=-1)
        # But we call forward directly to get logits before masking
        bin_logits = model.query_network(q_round, ref_angles)
        # Mask empty bins
        bin_logits = bin_logits.masked_fill(empty_bin_mask.unsqueeze(0), float('-inf'))
        query_probs = F.softmax(bin_logits, dim=-1)
        query_bins = query_probs.argmax(dim=-1)  # [round_window]

    return key_bins, key_probs, query_bins, query_probs


def get_top_bins_and_top_items(bins, probs, top_k_bins=10, top_k_items=10):
    """Get top-k bins by count and top-k items per bin by probability.

    Args:
        bins: [N] tensor of bin assignments (argmax results)
        probs: [N, num_bins] tensor of softmax probabilities
        top_k_bins: Number of top bins to select (default: 10)
        top_k_items: Number of top items per bin to select (default: 10)

    Returns:
        Tuple of (top_bins, bin_to_top_items)
        - top_bins: List of top bin IDs
        - bin_to_top_items: Dict mapping bin_id to set of top item indices
    """
    # Find top-k bins by count
    bin_counts = torch.bincount(bins, minlength=128)
    _, top_bin_indices = torch.topk(bin_counts, k=top_k_bins)
    top_bins = top_bin_indices.tolist()

    # For each top bin, find top-k items by probability
    bin_to_top_items = {}
    for bin_id in top_bins:
        # Get items assigned to this bin
        in_bin = (bins == bin_id)
        if in_bin.sum() == 0:
            continue
        # Get their probabilities for this bin
        bin_probs = probs[:, bin_id]
        # Mask out items not in this bin
        masked_probs = torch.where(in_bin, bin_probs, torch.tensor(-1.0))
        # Get top-k
        k = min(top_k_items, in_bin.sum().item())
        _, top_item_indices = torch.topk(masked_probs, k=k)
        bin_to_top_items[bin_id] = set(top_item_indices.tolist())

    return top_bins, bin_to_top_items


def plot_bin_frequencies(axes_row, data_complex, bins, top_bins, bin_to_top_items, row_label):
    """Plot bin assignments for top-6 frequencies.

    Args:
        axes_row: Row of 6 matplotlib axes
        data_complex: [N, 64] complex tensor
        bins: [N] tensor of bin assignments
        top_bins: List of top bin IDs
        bin_to_top_items: Dict mapping bin_id to set of top item indices
        row_label: Label for y-axis of first subplot
    """
    # Find top-6 frequencies
    top_freq_indices = get_top_k_frequencies(data_complex, k=6)
    cmap = plt.cm.tab10

    for col, freq_idx in enumerate(top_freq_indices):
        ax = axes_row[col]
        data_freq = data_complex[:, freq_idx]

        # First plot all points in gray (background)
        ax.scatter(data_freq.real.numpy(), data_freq.imag.numpy(),
                   c='gray', s=8, alpha=0.3, edgecolors='none')

        # Then overlay top-10 items from each top-10 bin in distinct colors
        for bin_rank, bin_id in enumerate(top_bins[:10]):
            if bin_id not in bin_to_top_items:
                continue
            top_items = list(bin_to_top_items[bin_id])
            if len(top_items) == 0:
                continue
            mask = torch.zeros(len(data_freq), dtype=torch.bool)
            for idx in top_items:
                if idx < len(mask):
                    mask[idx] = True
            if mask.any():
                ax.scatter(data_freq[mask].real.numpy(), data_freq[mask].imag.numpy(),
                           c=[cmap(bin_rank)], s=15, alpha=0.8, edgecolors='none',
                           label=f'Bin {bin_id}' if col == 0 else None)

        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Freq {freq_idx}', fontsize=10)
        ax.grid(alpha=0.2, linestyle='--')

    axes_row[0].set_ylabel(row_label, fontsize=10)


def main():
    """Main entry point for probe/bin assignment visualization."""
    parser = argparse.ArgumentParser(description='Visualize probe/bin assignments in post-RoPE space')
    parser.add_argument('--layer-head', required=True, help='Layer-head in format L-H (e.g., 15-20)')
    parser.add_argument('--round-idx', type=int, default=40,
                        help='Round index for middle round (default: 40)')
    args = parser.parse_args()

    # Parse layer and head
    layer, head = map(int, args.layer_head.split('-'))
    round_idx = args.round_idx
    round_window = 128

    # Setup paths
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / 'config.yaml'
    checkpoint_path = base_dir / 'output' / 'checkpoints' / 'best_model.pt'
    qk_path = base_dir / 'data' / 'qk_test.pt'
    output_dir = base_dir / 'output' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'probe_bin_assignment_L{layer}_H{head}_round{round_idx}.png'

    # Load model and data
    print(f"Loading model from {checkpoint_path}")
    model, config = load_model(config_path, checkpoint_path)

    print(f"Loading QK data from {qk_path}")
    k_head = load_qk_data(qk_path, layer, head)  # [seq, 128]
    print(f"K shape: {k_head.shape}")

    # Get round boundaries
    round_start = round_idx * round_window
    k_historical = k_head[:round_start]  # Historical keys for this round
    q_round = k_head[round_start:round_start + round_window]  # Queries in this round (using K as proxy)

    print(f"Round {round_idx}: start={round_start}, k_historical={k_historical.shape}, q_round={q_round.shape}")

    # Compute bin assignments
    print("Computing bin assignments...")
    key_bins, key_probs, query_bins, query_probs = compute_bin_assignments(
        model, k_historical, q_round, round_start, round_window)

    print(f"Key bins: {key_bins.shape}, Query bins: {query_bins.shape}")

    # Get top bins and top items
    print("Finding top bins and top items...")
    key_top_bins, key_bin_to_top = get_top_bins_and_top_items(key_bins, key_probs)
    query_top_bins, query_bin_to_top = get_top_bins_and_top_items(query_bins, query_probs)

    print(f"Key top bins: {key_top_bins}")
    print(f"Query top bins: {query_top_bins}")

    # Convert to complex
    k_complex = to_complex_pairs(k_historical)  # [num_historical, 64]
    q_complex = to_complex_pairs(q_round)  # [round_window, 64]

    # Create 2x6 subplot grid
    print("Creating visualization...")
    fig, axes = plt.subplots(2, 6, figsize=(24, 8), dpi=100)

    # Row 1: Key bin assignments
    plot_bin_frequencies(axes[0], k_complex, key_bins, key_top_bins, key_bin_to_top,
                         f'Keys\n(round {round_idx})')

    # Row 2: Query bin assignments
    plot_bin_frequencies(axes[1], q_complex, query_bins, query_top_bins, query_bin_to_top,
                         f'Queries\n(round {round_idx})')

    fig.suptitle(f'Probe/Bin Assignment (Post-RoPE) - Layer {layer} Head {head}\n'
                 f'Round {round_idx}: Top-10 bins, each showing top-10 keys/queries', fontsize=14)

    # Add legend to first subplot
    axes[0, 0].legend(loc='upper right', fontsize=6, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
