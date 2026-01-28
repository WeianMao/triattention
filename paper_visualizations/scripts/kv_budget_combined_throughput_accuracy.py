"""Generate combined figure: KV Budget vs Throughput + KV Budget vs Accuracy (AIME25).

This script creates a 2-panel figure combining:
- Panel (A): Throughput comparison at 16K generation length
- Panel (B): Accuracy comparison on AIME25 benchmark (Qwen3-8B)

Output: Combined figure with 2 panels
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate combined throughput + accuracy figure"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_visualizations/outputs/kv_budget_combined"),
        help="Output directory for figures",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Style constants (matching reference combined figure)
    face_color = (231 / 250, 231 / 250, 240 / 250)
    FONT_SIZE = 16
    TICK_SIZE = 14
    LABEL_FONT_SIZE = 22
    LABEL_FONT = 'Times New Roman'

    # Color scheme for methods (matching reference)
    colors = {
        'FullKV': '#E24A33',                          # Red (baseline)
        'R-KV': (85/250, 104/250, 154/250),           # Blue
        'TriAttention': (46/250, 139/250, 87/250),    # Green
    }

    # Markers
    markers = {
        'R-KV': 'o',
        'TriAttention': 's',
    }

    def style_ax(ax):
        ax.set_facecolor(face_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=TICK_SIZE)
        ax.set_axisbelow(True)
        ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)

    # ==========================================================================
    # Data
    # ==========================================================================

    # --- Throughput data (16K generation length, Fixed budget) ---
    throughput_fullkv = 222.76  # batch size 26
    throughput_rkv_budgets = [512, 1024, 2048, 3072, 4096]
    throughput_rkv = [1927.55, 1345.53, 760.44, 543.59, 279.01]
    throughput_triatt_budgets = [512, 1024, 2048, 3072]  # 4096 has no data
    throughput_triatt = [2009.31, 1405.18, 787.09, 563.49]

    # --- Accuracy data (AIME25, Qwen3-8B) ---
    accuracy_fullkv = 40.8
    accuracy_rkv_budgets = [512, 1024, 2048, 3072, 4096]
    accuracy_rkv = [6.0, 8.8, 17.5, 21.7, 31.7]
    accuracy_triatt_budgets = [512, 1024, 2048, 3072, 4096]
    accuracy_triatt = [8.3, 15.4, 32.9, 40.8, 43.3]

    # ==========================================================================
    # Create figure with 2 panels
    # ==========================================================================
    # Reference: 12 width for 3 panels → 4 per panel
    # Here: 8 width for 2 panels → 4 per panel (matching reference proportion)
    fig = plt.figure(figsize=(8, 4.5), dpi=args.dpi, constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, wspace=0.02)

    all_budgets = [512, 1024, 2048, 3072, 4096]

    # --- Panel (A): Throughput ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_box_aspect(1)
    style_ax(ax1)

    # FullKV horizontal line
    ax1.axhline(y=throughput_fullkv, color=colors['FullKV'], linestyle='--',
                linewidth=2.5, label='FullKV', alpha=0.9)

    # R-KV
    ax1.plot(throughput_rkv_budgets, throughput_rkv,
             color=colors['R-KV'], linestyle='-', marker=markers['R-KV'],
             markersize=9, linewidth=2.5, label='R-KV', alpha=0.9)

    # TriAttention
    ax1.plot(throughput_triatt_budgets, throughput_triatt,
             color=colors['TriAttention'], linestyle='-', marker=markers['TriAttention'],
             markersize=9, linewidth=2.5, label='TriAttention', alpha=0.9)

    ax1.set_xlabel('KV Cache Budget', fontsize=FONT_SIZE)
    ax1.set_ylabel('Throughput (tok/s)', fontsize=FONT_SIZE)
    ax1.set_title('Throughput', fontsize=FONT_SIZE + 2, fontweight='bold', pad=10)
    ax1.set_xticks(all_budgets)
    ax1.set_xticklabels(['512', '1k', '2k', '3k', '4k'])
    ax1.set_xlim(400, 4200)
    ax1.set_ylim(0, 2300)
    ax1.text(-0.08, 1.08, '(A)', transform=ax1.transAxes,
             fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom',
             fontname=LABEL_FONT)
    ax1.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='upper right')

    # --- Panel (B): Accuracy (AIME25) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_box_aspect(1)
    style_ax(ax2)

    # FullKV horizontal line
    ax2.axhline(y=accuracy_fullkv, color=colors['FullKV'], linestyle='--',
                linewidth=2.5, label='FullKV', alpha=0.9)

    # R-KV
    ax2.plot(accuracy_rkv_budgets, accuracy_rkv,
             color=colors['R-KV'], linestyle='-', marker=markers['R-KV'],
             markersize=9, linewidth=2.5, label='R-KV', alpha=0.9)

    # TriAttention
    ax2.plot(accuracy_triatt_budgets, accuracy_triatt,
             color=colors['TriAttention'], linestyle='-', marker=markers['TriAttention'],
             markersize=9, linewidth=2.5, label='TriAttention', alpha=0.9)

    ax2.set_xlabel('KV Cache Budget', fontsize=FONT_SIZE)
    ax2.set_ylabel('Accuracy (%)', fontsize=FONT_SIZE)
    ax2.set_title('AIME25', fontsize=FONT_SIZE + 2, fontweight='bold', pad=10)
    ax2.set_xticks(all_budgets)
    ax2.set_xticklabels(['512', '1k', '2k', '3k', '4k'])
    ax2.set_xlim(400, 4200)
    ax2.text(-0.08, 1.08, '(B)', transform=ax2.transAxes,
             fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom',
             fontname=LABEL_FONT)
    ax2.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='lower right')

    output_path = args.output_dir / "fig_kv_budget_throughput_accuracy_combined.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
