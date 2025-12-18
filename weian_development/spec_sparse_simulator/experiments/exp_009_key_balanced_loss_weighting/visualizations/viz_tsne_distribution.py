"""t-SNE visualization for train/test distribution comparison.

Visualizes training and test set distributions using t-SNE on full-spectrum
post-RoPE features (128-dimensional). All 248 error cases are forcibly included
in the visualization to analyze their distribution patterns.

Sampling strategy:
- Force include all 248 error points
- Sample ~2000 train points
- Sample ~2750 test non-error points
- Total ~5000 points

Color scheme:
- Blue: Training set points
- Green: Test set points (non-error)
- Red: Error points (highlighted with black edges)
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
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent))

from viz_utils import (
    load_miss_cases,
    load_qk_data,
    load_query_data,
    to_complex_pairs,
    setup_matplotlib
)


def prepare_features(tensor: torch.Tensor) -> np.ndarray:
    """Convert post-RoPE tensor to real features for t-SNE.

    Stacks real and imaginary parts of complex representation to create
    128-dimensional feature vectors from 64 complex frequencies.

    Args:
        tensor: [seq_len, 128] tensor

    Returns:
        np.ndarray of shape [seq_len, 128] (real features)
    """
    # Convert to complex: [seq_len, 128] -> [seq_len, 64]
    complex_tensor = to_complex_pairs(tensor)

    # Stack real and imaginary parts: [seq_len, 128]
    features = np.concatenate([
        complex_tensor.real.numpy(),
        complex_tensor.imag.numpy()
    ], axis=1)

    return features


def sample_with_forced_indices(
    train_features: np.ndarray,
    test_features: np.ndarray,
    error_indices: set[int],  # Indices into test_features
    n_train_samples: int = 2000,
    n_test_samples: int = 2750
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample data ensuring all error indices are included.

    Implements forced sampling strategy:
    1. Include ALL error indices from test set
    2. Sample random train points
    3. Sample random non-error test points

    Args:
        train_features: [n_train, 128] training features
        test_features: [n_test, 128] test features
        error_indices: Set of test indices that are error cases
        n_train_samples: Target number of train samples
        n_test_samples: Target number of test samples (excluding errors)

    Returns:
        Tuple of:
        - sampled_features: [total_samples, 128] combined features
        - labels: [total_samples] 0=train, 1=test
        - is_error: [total_samples] boolean mask for error points
        - original_indices: [total_samples] original indices in source arrays
    """
    # Sample from train (no errors in train)
    train_indices = np.random.choice(
        len(train_features),
        min(n_train_samples, len(train_features)),
        replace=False
    )

    # For test: force include all error indices
    error_idx_list = [i for i in error_indices if i < len(test_features)]
    non_error_test_indices = [i for i in range(len(test_features)) if i not in error_indices]

    # Sample non-error test points
    n_non_error_samples = n_test_samples - len(error_idx_list)
    if n_non_error_samples > 0:
        sampled_non_error = np.random.choice(
            non_error_test_indices,
            min(n_non_error_samples, len(non_error_test_indices)),
            replace=False
        )
        test_indices = np.concatenate([error_idx_list, sampled_non_error])
    else:
        test_indices = np.array(error_idx_list)

    # Combine features
    sampled_train = train_features[train_indices]
    sampled_test = test_features[test_indices]

    features = np.vstack([sampled_train, sampled_test])
    labels = np.array([0] * len(train_indices) + [1] * len(test_indices))

    # Mark error points (only in test portion)
    is_error = np.zeros(len(features), dtype=bool)
    error_mask_in_test = np.isin(test_indices, list(error_indices))
    is_error[len(train_indices):] = error_mask_in_test

    return features, labels, is_error, np.concatenate([train_indices, test_indices])


def plot_tsne_distribution(
    k_train: torch.Tensor,
    k_test: torch.Tensor,
    q_train: torch.Tensor,
    q_test: torch.Tensor,
    failure_key_indices: set[int],
    failure_query_indices: set[int],
    output_dir: Path,
    layer: int,
    head: int
) -> None:
    """Create t-SNE visualizations for Key and Query, with and without error markers.

    Generates TWO PNG files sharing the same t-SNE embedding:
    1. tsne_distribution_L{layer}_H{head}.png - without error markers
    2. tsne_distribution_L{layer}_H{head}_with_errors.png - with error markers

    Args:
        k_train: Training Key tensor [train_seq_len, 128]
        k_test: Test Key tensor [test_seq_len, 128]
        q_train: Training Query tensor [train_seq_len, 128]
        q_test: Test Query tensor [test_seq_len, 128]
        failure_key_indices: Set of Key error indices in test set
        failure_query_indices: Set of Query error indices in test set
        output_dir: Directory to save the figures
        layer: Layer index for title
        head: Head index for title
    """

    # Prepare features
    k_train_feat = prepare_features(k_train)
    k_test_feat = prepare_features(k_test)
    q_train_feat = prepare_features(q_train)
    q_test_feat = prepare_features(q_test)

    # Sample with forced error inclusion
    np.random.seed(42)
    k_features, k_labels, k_is_error, _ = sample_with_forced_indices(
        k_train_feat, k_test_feat, failure_key_indices
    )
    q_features, q_labels, q_is_error, _ = sample_with_forced_indices(
        q_train_feat, q_test_feat, failure_query_indices
    )

    # Run t-SNE (shared embedding for both figures)
    print(f"Running t-SNE on Key features ({len(k_features)} points)...")
    k_tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
    k_embedded = k_tsne.fit_transform(k_features)

    print(f"Running t-SNE on Query features ({len(q_features)} points)...")
    q_tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
    q_embedded = q_tsne.fit_transform(q_features)

    # Helper function to plot
    def plot_tsne(ax, embedded, labels, is_error, title, show_errors=True):
        """Plot t-SNE embedding with train/test/error point coloring."""
        # Plot train points (blue)
        train_mask = (labels == 0)
        ax.scatter(
            embedded[train_mask, 0], embedded[train_mask, 1],
            c='blue', s=10, alpha=0.5, label=f'Train ({train_mask.sum()})',
            edgecolors='none'
        )

        if show_errors:
            # Plot test points (green, non-error only)
            test_non_error = (labels == 1) & (~is_error)
            ax.scatter(
                embedded[test_non_error, 0], embedded[test_non_error, 1],
                c='green', s=10, alpha=0.5, label=f'Test ({test_non_error.sum()})',
                edgecolors='none'
            )

            # Plot error points (red, on top)
            if is_error.any():
                ax.scatter(
                    embedded[is_error, 0], embedded[is_error, 1],
                    c='red', s=50, alpha=0.9, label=f'Error ({is_error.sum()})',
                    edgecolors='black', linewidths=0.5
                )
        else:
            # Plot ALL test points (green, including errors)
            test_mask = (labels == 1)
            ax.scatter(
                embedded[test_mask, 0], embedded[test_mask, 1],
                c='green', s=10, alpha=0.5, label=f'Test ({test_mask.sum()})',
                edgecolors='none'
            )

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')

    # === Figure 1: WITHOUT error markers ===
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

    plot_tsne(axes1[0], k_embedded, k_labels, k_is_error,
              f'Key t-SNE - L{layer} H{head}', show_errors=False)
    plot_tsne(axes1[1], q_embedded, q_labels, q_is_error,
              f'Query t-SNE - L{layer} H{head}', show_errors=False)

    fig1.suptitle(
        f't-SNE Distribution (Post-RoPE, 128-dim features) - Layer {layer} Head {head}\n'
        f'Train vs Test comparison (no error markers)',
        fontsize=14
    )
    plt.tight_layout()
    output_path1 = output_dir / f'tsne_distribution_L{layer}_H{head}.png'
    fig1.savefig(output_path1, dpi=100, bbox_inches='tight')
    plt.close(fig1)
    print(f'Saved: {output_path1}')

    # === Figure 2: WITH error markers ===
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

    plot_tsne(axes2[0], k_embedded, k_labels, k_is_error,
              f'Key t-SNE - L{layer} H{head}', show_errors=True)
    plot_tsne(axes2[1], q_embedded, q_labels, q_is_error,
              f'Query t-SNE - L{layer} H{head}', show_errors=True)

    fig2.suptitle(
        f't-SNE Distribution (Post-RoPE, 128-dim features) - Layer {layer} Head {head}\n'
        f'All {k_is_error.sum()} Key / {q_is_error.sum()} Query error cases marked in red',
        fontsize=14
    )
    plt.tight_layout()
    output_path2 = output_dir / f'tsne_distribution_L{layer}_H{head}_with_errors.png'
    fig2.savefig(output_path2, dpi=100, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {output_path2}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize t-SNE distribution for train/test comparison'
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
    train_path = base_dir / 'data' / 'qk.pt'
    test_path = base_dir / 'data' / 'qk_test.pt'
    analysis_path = base_dir / 'output' / 'analysis' / 'miss_case_analysis.json'
    output_dir = base_dir / 'output' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib()

    # Load data
    print(f'Loading failure case analysis from {analysis_path}...')
    miss_cases = load_miss_cases(analysis_path)
    failure_key_indices = {case['argmax_key'] for case in miss_cases['miss_cases']}
    failure_query_indices = {case['query_idx'] for case in miss_cases['miss_cases']}
    print(f'Found {len(failure_key_indices)} unique Key error indices')
    print(f'Found {len(failure_query_indices)} unique Query error indices')

    print(f'Loading training data from {train_path}...')
    k_train = load_qk_data(train_path, layer, head)
    q_train = load_query_data(train_path, layer, head)
    print(f'Train: K shape={k_train.shape}, Q shape={q_train.shape}')

    print(f'Loading test data from {test_path}...')
    k_test = load_qk_data(test_path, layer, head)
    q_test = load_query_data(test_path, layer, head)
    print(f'Test: K shape={k_test.shape}, Q shape={q_test.shape}')

    # Generate visualization (two figures: with and without error markers)
    plot_tsne_distribution(
        k_train, k_test, q_train, q_test,
        failure_key_indices, failure_query_indices,
        output_dir, layer, head
    )


if __name__ == '__main__':
    main()
