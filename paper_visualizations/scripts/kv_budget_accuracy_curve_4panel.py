"""Generate KV Cache budget vs accuracy curves with 4 panels including step accuracy.

This script generates accuracy curves comparing different KV Cache compression methods
(FullKV, R-KV, TriAttention) across different budget levels for Qwen3-8B model,
plus a step accuracy panel.

Output: Combined figure with 4 panels (A, B, C, D) for MATH, AIME24, AIME25, Step Accuracy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force Times New Roman font for all text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KV budget vs accuracy curves (4 panels)"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("paper_visualizations/Materials/SpecKV Experiment Data - Performance.csv"),
        help="Path to the performance CSV file",
    )
    parser.add_argument(
        "--step-accuracy-csv",
        type=Path,
        default=Path("paper_visualizations/Materials/step_accuracy_is_correct_sample8.csv"),
        help="Path to the step accuracy CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_visualizations/outputs/kv_budget_accuracy"),
        help="Output directory for figures",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
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
    # Map display name to CSV method name
    csv_method = 'SpecKV' if method == 'TriAttention' else method
    row = df[(df['Benchmark'] == benchmark) & (df['Method'] == csv_method)]
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


def main() -> None:
    args = parse_args()

    # Load data
    df = load_and_filter_data(args.input_csv)
    df_step = pd.read_csv(args.step_accuracy_csv)

    if df.empty:
        raise ValueError("No Qwen3-8B data found in CSV")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Style constants (enlarged for combined figure)
    face_color = (231 / 250, 231 / 250, 240 / 250)
    FONT_SIZE = 16
    TICK_SIZE = 14
    LABEL_FONT_SIZE = 22
    LABEL_FONT = 'Times New Roman'

    # Color scheme for methods
    colors = {
        'Full Attention': '#E24A33',      # Red (baseline)
        'R-KV': (85/250, 104/250, 154/250),    # Blue
        'TriAttention': (46/250, 139/250, 87/250),   # Green
        'SnapKV': (187/250, 130/250, 90/250),  # Orange/brown
    }

    # Markers
    markers = {
        'R-KV': 'o',
        'TriAttention': 's',
        'SnapKV': '^',
    }

    def style_ax_combined(ax):
        ax.set_facecolor(face_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=TICK_SIZE)
        ax.set_axisbelow(True)
        ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)

    # Create figure with 1 row, 4 columns (each subplot is square)
    fig = plt.figure(figsize=(16, 4.5), dpi=args.dpi, constrained_layout=True)
    gs = GridSpec(1, 4, figure=fig, wspace=0.05)

    benchmarks = ['MATH', 'AIME24', 'AIME25']
    display_titles = ['MATH500', 'AIME24', 'AIME25']
    panel_labels = ['(A)', '(B)', '(C)']

    # Panels A, B, C: KV Budget vs Accuracy
    for idx, (benchmark, title, label) in enumerate(zip(benchmarks, display_titles, panel_labels)):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_box_aspect(1)
        style_ax_combined(ax)

        methods = ['Full Attention', 'R-KV', 'TriAttention', 'SnapKV']
        method_csv_map = {'Full Attention': 'FullKV', 'R-KV': 'R-KV', 'TriAttention': 'TriAttention', 'SnapKV': 'SnapKV'}

        for method in methods:
            csv_method = method_csv_map[method]
            budgets, accuracies = get_method_data(df, benchmark, csv_method)

            # Skip if insufficient data
            if csv_method != 'FullKV' and not has_sufficient_data(budgets):
                continue

            if not budgets:
                continue

            if csv_method == 'FullKV':
                fullkv_value = accuracies[0] if accuracies else None
                if fullkv_value is not None:
                    ax.axhline(y=fullkv_value, color=colors[method], linestyle='--',
                              linewidth=2.5, label='Full Attention', alpha=0.9)
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

        # Title
        ax.set_title(title, fontsize=FONT_SIZE + 2, fontweight='bold', pad=10)

        # X-axis ticks
        all_budgets = [512, 1024, 2048, 3072, 4096]
        ax.set_xticks(all_budgets)
        ax.set_xticklabels(['512', '1k', '2k', '3k', '4k'])
        ax.set_xlim(400, 4200)

        # Panel label
        ax.text(-0.12, 1.08, label, transform=ax.transAxes,
               fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom',
               fontname=LABEL_FONT)

        # Legend
        ax.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='lower right')

    # Panel D: Step Accuracy
    ax_d = fig.add_subplot(gs[0, 3])
    ax_d.set_box_aspect(1)
    style_ax_combined(ax_d)

    # Plot R-KV and TriAttention step accuracy
    ax_d.plot(df_step['step'], df_step['rkv_accuracy'] * 100,
             color=colors['R-KV'], linestyle='-', marker=markers['R-KV'],
             markersize=9, linewidth=2.5, label='R-KV', alpha=0.9)
    ax_d.plot(df_step['step'], df_step['speckv_accuracy'] * 100,
             color=colors['TriAttention'], linestyle='-', marker=markers['TriAttention'],
             markersize=9, linewidth=2.5, label='TriAttention', alpha=0.9)

    ax_d.set_xlabel('Step', fontsize=FONT_SIZE)
    ax_d.set_title('Memory Retention Benchmark', fontsize=FONT_SIZE + 2, fontweight='bold', pad=10)

    # X-axis ticks
    steps = df_step['step'].tolist()
    ax_d.set_xticks(steps)
    ax_d.set_xticklabels([str(s) for s in steps])

    # Y-axis
    ax_d.set_ylim(20, 105)

    # Panel label
    ax_d.text(-0.12, 1.08, '(D)', transform=ax_d.transAxes,
             fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom',
             fontname=LABEL_FONT)

    # Legend
    ax_d.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='lower left')

    output_path = args.output_dir / "fig_kv_budget_accuracy_combined_4panel.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved combined 4-panel figure: {output_path}")


if __name__ == "__main__":
    main()
