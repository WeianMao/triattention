"""Plot DFS accuracy vs recursion depth for different KV cache implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    "fullkv": (85 / 250, 104 / 250, 154 / 250),    # blue (color_recon style)
    "speckv": (187 / 250, 130 / 250, 90 / 250),    # brown (color_gt style)
    "rkv": (120 / 250, 160 / 250, 120 / 250),      # green
}

LABELS = {
    "fullkv": "FullKV",
    "speckv": "TriAttention",
    "rkv": "R-KV",
}

# Plot order (top to bottom in legend)
PLOT_ORDER = ["fullkv", "speckv", "rkv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DFS accuracy vs recursion depth")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("paper_visualizations/Materials/dfs_lite/dfs_metrics_by_step.csv"),
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
        impl_data = df[df["implementation"] == impl].sort_values("num_step")
        ax.plot(
            impl_data["num_step"],
            impl_data["is_correct"],
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            color=COLORS[impl],
            label=LABELS[impl],
            alpha=0.9,
        )

    # Axis labels and limits
    ax.set_xscale("log")
    ax.set_xlabel("Recursion Depth", fontsize=FONT_SIZE)
    ax.set_ylabel("Accuracy", fontsize=FONT_SIZE)

    if use_log_y:
        # Use "inverted log" - plot error rate (1-acc) on log scale, label as accuracy
        # This stretches differences near 1.0 and compresses near 0.1
        ax.set_yscale("function", functions=(
            lambda y: -np.log10(1.001 - y),  # forward: accuracy -> display
            lambda y: 1.001 - 10**(-y)       # inverse: display -> accuracy
        ))
        # Set y-axis ticks at accuracy values
        y_ticks = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{y:.1f}" for y in y_ticks])
        ax.set_ylim(0.05, 1.0)
    else:
        ax.set_ylim(0, 1.05)

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
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def plot_relative_figure(df, output_path, args, show_abs_labels: bool = False) -> None:
    """Plot relative performance (% of FullKV) figure."""
    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)

    # Apply styling
    style_ax(ax)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)

    # Get FullKV data as baseline
    fullkv_data = df[df["implementation"] == "fullkv"].set_index("num_step")["is_correct"]

    # Plot each implementation (except FullKV which is always 100%)
    for impl in PLOT_ORDER:
        impl_data = df[df["implementation"] == impl].sort_values("num_step")

        if impl == "fullkv":
            # FullKV is always 100%
            relative_acc = [100.0] * len(impl_data)
        else:
            # Compute relative to FullKV
            relative_acc = []
            for _, row in impl_data.iterrows():
                step = row["num_step"]
                fullkv_acc = fullkv_data.loc[step]
                rel = (row["is_correct"] / fullkv_acc) * 100 if fullkv_acc > 0 else 0
                relative_acc.append(rel)

        ax.plot(
            impl_data["num_step"],
            relative_acc,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            color=COLORS[impl],
            label=LABELS[impl],
            alpha=0.9,
        )

        # Add absolute accuracy labels at selected points
        if show_abs_labels:
            # Label at depths 8, 16, 32
            label_depths = [8, 16, 32]
            for i, (_, row) in enumerate(impl_data.iterrows()):
                step = row["num_step"]
                if step in label_depths:
                    abs_acc = row["is_correct"]
                    rel_acc = relative_acc[i]
                    # Position labels to the right of points, avoid curve overlap
                    x_offset = 8
                    y_offset = 0
                    ax.annotate(
                        f"{abs_acc:.0%}",
                        (step, rel_acc),
                        textcoords="offset points",
                        xytext=(x_offset, y_offset),
                        ha="left",
                        va="center",
                        fontsize=FONT_SIZE,
                        color=COLORS[impl],
                    )

    # Axis labels and limits
    ax.set_xscale("log")
    ax.set_xlabel("Recursion Depth", fontsize=FONT_SIZE)
    ax.set_ylabel("Relative Performance (% of FullKV)", fontsize=FONT_SIZE)
    ax.set_ylim(0, 105)

    # Legend with transparent white background
    ax.legend(
        fontsize=FONT_SIZE,
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        edgecolor="none",
    )

    # Consistent tick label size
    ax.tick_params(axis='both', labelsize=FONT_SIZE)

    # Set x-axis ticks to match data points
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_dfs_acc")

    # Load data
    df = pd.read_csv(args.data_path)

    # Generate all versions
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Linear y-axis
    plot_figure(df, args.output_dir / "fig_dfs_accuracy_linear.png", args, use_log_y=False)

    # Log y-axis
    plot_figure(df, args.output_dir / "fig_dfs_accuracy_log.png", args, use_log_y=True)

    # Relative performance (% of FullKV)
    plot_relative_figure(df, args.output_dir / "fig_dfs_accuracy_relative.png", args, show_abs_labels=False)

    # Relative performance with absolute accuracy labels
    plot_relative_figure(df, args.output_dir / "fig_dfs_accuracy_relative_labeled.png", args, show_abs_labels=True)


if __name__ == "__main__":
    main()
