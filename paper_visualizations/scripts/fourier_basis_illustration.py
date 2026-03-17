"""Generate Fourier basis illustration figure.

This script generates a figure showing Fourier basis functions (cosine waves)
to illustrate the concept of Fourier synthesis, with a target function
(Gaussian-smoothed reconstruction curve) to be synthesized.

Output: fig_fourier_basis.png (aspect ratio 2:1, width:height)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Fourier basis illustration")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for the figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(42)

    # Colors
    color_target = (85 / 250, 104 / 250, 154 / 250)      # Blue-purple for target
    basis_colors = [
        (187 / 250, 130 / 250, 90 / 250),    # Orange-tan
        (210 / 250, 160 / 250, 120 / 250),   # Light orange
        (120 / 250, 140 / 250, 190 / 250),   # Light blue
        (170 / 250, 120 / 250, 140 / 250),   # Mauve
    ]
    axis_color = (0.5, 0.5, 0.5)  # Medium gray for axes

    curve_linewidth = 2.5
    basis_linewidth = 2.0
    axis_linewidth = curve_linewidth * 1.5  # 1.5x curve width

    # Figure with 2:1 aspect ratio (width:height)
    fig_width = 8
    fig_height = 4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # White/transparent background
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    # Hide default spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Generate x coordinates
    x = np.linspace(0, 4 * np.pi, 500)

    # ============ Target function: Gaussian-smoothed reconstruction ============
    # Simulate the reconstruction curve: Σ A_f cos(ω_f x + φ_f)
    # Using frequencies and random phases/amplitudes similar to actual RoPE
    target_frequencies = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    target_amplitudes = [1.0, 0.8, 0.5, 0.4, 0.2, 0.15, 0.1]  # Decaying amplitudes
    target_phases = np.random.uniform(0, 2 * np.pi, len(target_frequencies))

    # Build reconstruction curve (full range, computed at actual x coordinates)
    reconstruction = np.zeros_like(x)
    for freq, amp, phi in zip(target_frequencies, target_amplitudes, target_phases):
        reconstruction += amp * np.cos(freq * x + phi)

    # Apply Gaussian smoothing
    sigma = 15  # Smoothing parameter
    target_curve_full = gaussian_filter1d(reconstruction, sigma=sigma, mode='nearest')

    # Normalize
    target_curve_full = target_curve_full / np.abs(target_curve_full).max() * 0.85

    # Take first half and flip it
    half_len = len(x) // 2
    x_target = x[:half_len]
    target_curve = target_curve_full[:half_len][::-1]  # Flip the first half

    # ============ Basis functions with random weights ============
    basis_frequencies = [1.0, 2.0]  # Two frequencies
    basis_scale = 0.3  # Smaller y-scale for basis functions
    basis_alpha = 0.5  # Higher transparency (lower alpha)

    # Random weights for visualization (just to show they contribute)
    weights_cos = [0.6, 0.4]
    weights_sin = [0.5, 0.3]

    # Draw basis functions (behind target)
    for i, freq in enumerate(basis_frequencies):
        # Cosine basis
        y_cos = weights_cos[i] * basis_scale * np.cos(freq * x)
        ax.plot(x, y_cos, color=basis_colors[i * 2], linewidth=basis_linewidth,
                alpha=basis_alpha, zorder=2)
        # Sine basis
        y_sin = weights_sin[i] * basis_scale * np.sin(freq * x)
        ax.plot(x, y_sin, color=basis_colors[i * 2 + 1], linewidth=basis_linewidth,
                alpha=basis_alpha, zorder=2)

    # Draw target function (on top, first half of x range, flipped)
    ax.plot(x_target, target_curve, color=color_target, linewidth=curve_linewidth,
            alpha=0.95, zorder=5)

    ax.set_xlim(0, 4 * np.pi)
    ax.set_ylim(-1.1, 1.1)

    # Draw coordinate axes (origin at left-center) using plot for consistent width
    # X-axis
    ax.plot([0, 4 * np.pi], [0, 0], color=axis_color, linewidth=axis_linewidth, solid_capstyle='butt', zorder=1)
    # Y-axis
    ax.plot([0, 0], [-1.1, 1.1], color=axis_color, linewidth=axis_linewidth, solid_capstyle='butt', zorder=1)

    # Hide all axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # Save
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = Path("paper_visualizations/outputs/fig_fourier_basis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)

    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
