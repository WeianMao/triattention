"""Plot step accuracy curve for sample8 data (RKV vs SpecKV)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force Times New Roman font for all text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts match Times New Roman for math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants (matching visualize_reconstruction_comparison.py) ============
FONT_SIZE = 14
MARKER_SIZE = 7
LINE_WIDTH = 2.5

# Background color (light gray-purple)
face_color = (231 / 250, 231 / 250, 240 / 250)

# Colors for different implementations (matching paper style)
COLORS = {
    "speckv": (187 / 250, 130 / 250, 90 / 250),    # brown (color_gt style)
    "rkv": (120 / 250, 160 / 250, 120 / 250),      # green
}

LABELS = {
    "speckv": "TriAttention",
    "rkv": "R-KV",
}

# Plot order (top to bottom in legend)
PLOT_ORDER = ["speckv", "rkv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot step accuracy for sample8 data")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("paper_visualizations/Materials/step_accuracy_is_correct_sample8.csv"),
        help="Path to CSV file with metrics",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_visualizations/outputs/dfs_accuracy"),
        help="Output directory for figures",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(6.0, 4.5),
        help="Figure size in inches",
    )
    return parser.parse_args()


def style_ax(ax):
    """Apply consistent styling to axes."""
    ax.set_facecolor(face_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=FONT_SIZE)


def plot_figure(df, output_path, args, use_log_y: bool) -> None:
    """Plot accuracy figure with optional log y-axis."""
    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)

    # Apply styling
    style_ax(ax)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)

    # Plot each implementation
    for impl in PLOT_ORDER:
        col_name = f"{impl}_accuracy"
        ax.plot(
            df["step"],
            df[col_name],
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            color=COLORS[impl],
            label=LABELS[impl],
            alpha=0.9,
        )

    # Axis labels and limits
    ax.set_xlabel("Step", fontsize=FONT_SIZE)
    ax.set_ylabel("Accuracy", fontsize=FONT_SIZE)

    if use_log_y:
        # Use "inverted log" - plot error rate (1-acc) on log scale, label as accuracy
        ax.set_yscale("function", functions=(
            lambda y: -np.log10(1.001 - y),  # forward: accuracy -> display
            lambda y: 1.001 - 10**(-y)       # inverse: display -> accuracy
        ))
        y_ticks = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{y:.1f}" for y in y_ticks])
        ax.set_ylim(0.05, 1.0)
    else:
        ax.set_ylim(0.2, 1.05)

    # Legend with transparent white background
    ax.legend(
        fontsize=FONT_SIZE,
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        edgecolor="none",
    )

    # Set x-axis ticks to match data points
    steps = df["step"].tolist()
    ax.set_xticks(steps)
    ax.set_xticklabels([str(s) for s in steps])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_dfs_acc_sample8")

    # Load data
    df = pd.read_csv(args.data_path)

    # Generate figures
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Linear y-axis
    plot_figure(df, args.output_dir / "fig_step_accuracy_sample8_linear.png", args, use_log_y=False)

    # Log y-axis
    plot_figure(df, args.output_dir / "fig_step_accuracy_sample8_log.png", args, use_log_y=True)


if __name__ == "__main__":
    main()
