"""Generate KV Cache budget vs accuracy curves for different compression methods.

This script generates accuracy curves comparing different KV Cache compression methods
(FullKV, R-KV, SpecKV) across different budget levels for Qwen3-8B model.

Output: Combined figure with 3 panels (A, B, C) for MATH, AIME24, AIME25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KV budget vs accuracy curves"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("paper_visualizations/Materials/SpecKV Experiment Data - Performance.csv"),
        help="Path to the performance CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_visualizations/outputs/kv_budget_accuracy"),
        help="Output directory for figures",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument("--combined", action="store_true", default=True,
                       help="Generate combined figure with all benchmarks")
    return parser.parse_args()


def load_and_filter_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV and filter to Qwen3-8B only."""
    df = pd.read_csv(csv_path)

    # Filter to Qwen3-8B only
    df = df[df['Model'] == 'Qwen3-8B'].copy()

    # Budget columns
    budget_cols = ['512', '1024', '2048', '3072', '4096']

    # Clean numeric values (handle cases like "27.5/23.3")
    for col in budget_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: float(str(x).split('/')[0]) if pd.notna(x) and str(x).strip() else np.nan)

    return df


def get_method_data(df: pd.DataFrame, benchmark: str, method: str) -> tuple[list, list]:
    """Extract budget and accuracy data for a specific method."""
    row = df[(df['Benchmark'] == benchmark) & (df['Method'] == method)]
    if row.empty:
        return [], []

    budget_cols = ['512', '1024', '2048', '3072', '4096']
    budgets = []
    accuracies = []

    for col in budget_cols:
        if col in row.columns:
            val = row[col].values[0]
            if pd.notna(val):
                budgets.append(int(col))
                accuracies.append(float(val))

    return budgets, accuracies


def has_sufficient_data(budgets: list, min_points: int = 3) -> bool:
    """Check if method has enough data points to plot."""
    return len(budgets) >= min_points


def style_ax(ax, face_color):
    """Apply consistent styling to axes (following reference style)."""
    ax.set_facecolor(face_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=14)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)


def plot_benchmark(df: pd.DataFrame, benchmark: str, output_path: Path, dpi: int) -> None:
    """Generate plot for a single benchmark."""

    # Style constants (from reference)
    face_color = (231 / 250, 231 / 250, 240 / 250)
    FONT_SIZE = 14
    TITLE_FONT_SIZE = 16

    # Color scheme for methods
    colors = {
        'FullKV': '#E24A33',      # Red (baseline)
        'R-KV': (85/250, 104/250, 154/250),    # Blue
        'SpecKV': (46/250, 139/250, 87/250),   # Green
        'SnapKV': (187/250, 130/250, 90/250),  # Orange/brown
    }

    # Line styles
    linestyles = {
        'FullKV': '--',   # Dashed for baseline
        'R-KV': '-',
        'SpecKV': '-',
        'SnapKV': '-',
    }

    # Markers
    markers = {
        'FullKV': '',      # No marker for baseline
        'R-KV': 'o',
        'SpecKV': 's',
        'SnapKV': '^',
    }

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    style_ax(ax, face_color)

    # Methods to plot (in order)
    methods = ['FullKV', 'R-KV', 'SpecKV', 'SnapKV']

    plotted_methods = []

    for method in methods:
        budgets, accuracies = get_method_data(df, benchmark, method)

        # Skip if insufficient data
        if method != 'FullKV' and not has_sufficient_data(budgets):
            continue

        if not budgets:
            continue

        if method == 'FullKV':
            # FullKV is a horizontal baseline line
            fullkv_value = accuracies[0] if accuracies else None
            if fullkv_value is not None:
                ax.axhline(y=fullkv_value, color=colors[method], linestyle='--',
                          linewidth=2.5, label=f'FullKV ({fullkv_value:.1f}%)', alpha=0.9)
                plotted_methods.append(method)
        else:
            # Other methods as line plots
            ax.plot(budgets, accuracies,
                   color=colors[method],
                   linestyle=linestyles[method],
                   marker=markers[method],
                   markersize=8,
                   linewidth=2.5,
                   label=method,
                   alpha=0.9)
            plotted_methods.append(method)

    # Axis labels and title
    ax.set_xlabel('KV Cache Budget', fontsize=FONT_SIZE)
    ax.set_ylabel('Accuracy (%)', fontsize=FONT_SIZE)
    ax.set_title(f'Qwen3-8B on {benchmark}', fontsize=TITLE_FONT_SIZE, fontweight='bold')

    # X-axis ticks
    all_budgets = [512, 1024, 2048, 3072, 4096]
    ax.set_xticks(all_budgets)
    ax.set_xticklabels([str(b) for b in all_budgets])
    ax.set_xlim(400, 4200)

    # Legend
    ax.legend(frameon=False, fontsize=FONT_SIZE, loc='lower right')

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"  Methods plotted: {plotted_methods}")


def plot_combined(df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """Generate combined figure with all three benchmarks as panels A, B, C."""

    # Style constants (enlarged for combined figure)
    face_color = (231 / 250, 231 / 250, 240 / 250)
    FONT_SIZE = 16
    TICK_SIZE = 14
    LABEL_FONT_SIZE = 22
    LABEL_FONT = 'DejaVu Sans'

    # Color scheme for methods
    colors = {
        'FullKV': '#E24A33',      # Red (baseline)
        'R-KV': (85/250, 104/250, 154/250),    # Blue
        'SpecKV': (46/250, 139/250, 87/250),   # Green
        'SnapKV': (187/250, 130/250, 90/250),  # Orange/brown
    }

    # Markers
    markers = {
        'R-KV': 'o',
        'SpecKV': 's',
        'SnapKV': '^',
    }

    def style_ax_combined(ax):
        ax.set_facecolor(face_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=TICK_SIZE)
        ax.set_axisbelow(True)
        ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)

    # Create figure with 1 row, 3 columns (each subplot is square)
    fig = plt.figure(figsize=(12, 4.5), dpi=dpi, constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, wspace=0.05)

    benchmarks = ['MATH', 'AIME24', 'AIME25']
    panel_labels = ['(A)', '(B)', '(C)']

    for idx, (benchmark, label) in enumerate(zip(benchmarks, panel_labels)):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_box_aspect(1)  # Force square subplot
        style_ax_combined(ax)

        methods = ['FullKV', 'R-KV', 'SpecKV', 'SnapKV']

        for method in methods:
            budgets, accuracies = get_method_data(df, benchmark, method)

            # Skip if insufficient data
            if method != 'FullKV' and not has_sufficient_data(budgets):
                continue

            if not budgets:
                continue

            if method == 'FullKV':
                fullkv_value = accuracies[0] if accuracies else None
                if fullkv_value is not None:
                    ax.axhline(y=fullkv_value, color=colors[method], linestyle='--',
                              linewidth=2.5, label='FullKV', alpha=0.9)
            else:
                ax.plot(budgets, accuracies,
                       color=colors[method],
                       linestyle='-',
                       marker=markers.get(method, 'o'),
                       markersize=9,
                       linewidth=2.5,
                       label=method,
                       alpha=0.9)

        # Axis labels
        ax.set_xlabel('KV Cache Budget', fontsize=FONT_SIZE)
        if idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=FONT_SIZE)

        # Title as benchmark name
        ax.set_title(benchmark, fontsize=FONT_SIZE + 2, fontweight='bold', pad=10)

        # X-axis ticks
        all_budgets = [512, 1024, 2048, 3072, 4096]
        ax.set_xticks(all_budgets)
        ax.set_xticklabels(['512', '1k', '2k', '3k', '4k'])
        ax.set_xlim(400, 4200)

        # Panel label (A), (B), (C)
        ax.text(-0.08, 1.08, label, transform=ax.transAxes,
               fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom',
               fontname=LABEL_FONT)

        # Legend (only on first panel to avoid redundancy, or on each)
        ax.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='lower right')

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved combined figure: {output_path}")


def main() -> None:
    args = parse_args()

    # Load data
    df = load_and_filter_data(args.input_csv)

    if df.empty:
        raise ValueError("No Qwen3-8B data found in CSV")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.combined:
        # Generate combined figure with all benchmarks
        output_path = args.output_dir / "fig_kv_budget_accuracy_combined.png"
        plot_combined(df, output_path, args.dpi)
    else:
        # Generate one plot per benchmark
        benchmarks = ['MATH', 'AIME24', 'AIME25']
        for benchmark in benchmarks:
            output_path = args.output_dir / f"kv_budget_accuracy_{benchmark.lower()}.png"
            plot_benchmark(df, benchmark, output_path, args.dpi)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
