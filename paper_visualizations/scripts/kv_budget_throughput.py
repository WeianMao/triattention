"""Generate KV Cache budget vs throughput curves for different compression methods.

This script visualizes throughput performance comparing FullKV, R-KV, and TriAttention
at 16K generation length.

Output: Single panel figure showing throughput vs KV budget
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate KV budget vs throughput curves"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_visualizations/outputs/kv_budget_throughput"),
        help="Output directory for figures",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Style constants (matching reference)
    face_color = (231 / 250, 231 / 250, 240 / 250)
    FONT_SIZE = 16
    TICK_SIZE = 14
    TITLE_FONT_SIZE = 18

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

    # ==========================================================================
    # Data for 16K generation length (Fixed budget only)
    # ==========================================================================

    # FullKV: batch size 26, throughput = 222.76 tok/s
    fullkv_throughput = 222.76

    # R-KV data (Fixed budgets only)
    rkv_budgets = [512, 1024, 2048, 3072, 4096]
    rkv_throughputs = [1927.55, 1345.53, 760.44, 543.59, 279.01]

    # TriAttention (SpecKV) data (Fixed budgets only, 4096 has no data)
    triatt_budgets = [512, 1024, 2048, 3072]
    triatt_throughputs = [2009.31, 1405.18, 787.09, 563.49]

    # ==========================================================================
    # Create figure
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(7, 6), dpi=args.dpi)

    # Style axes (matching reference)
    ax.set_facecolor(face_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=TICK_SIZE)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)

    # Plot FullKV as horizontal dashed line
    ax.axhline(y=fullkv_throughput, color=colors['FullKV'], linestyle='--',
               linewidth=2.5, label='FullKV', alpha=0.9)

    # Plot R-KV
    ax.plot(rkv_budgets, rkv_throughputs,
            color=colors['R-KV'],
            linestyle='-',
            marker=markers['R-KV'],
            markersize=9,
            linewidth=2.5,
            label='R-KV',
            alpha=0.9)

    # Plot TriAttention
    ax.plot(triatt_budgets, triatt_throughputs,
            color=colors['TriAttention'],
            linestyle='-',
            marker=markers['TriAttention'],
            markersize=9,
            linewidth=2.5,
            label='TriAttention',
            alpha=0.9)

    # Axis labels
    ax.set_xlabel('KV Cache Budget', fontsize=FONT_SIZE)
    ax.set_ylabel('Throughput (tok/s)', fontsize=FONT_SIZE)
    ax.set_title('Qwen3-8B (16K Generation)', fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=10)

    # X-axis ticks
    all_budgets = [512, 1024, 2048, 3072, 4096]
    ax.set_xticks(all_budgets)
    ax.set_xticklabels(['512', '1k', '2k', '3k', '4k'])
    ax.set_xlim(400, 4200)

    # Y-axis range
    ax.set_ylim(0, 2300)

    # Legend
    ax.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='upper right')

    plt.tight_layout()
    output_path = args.output_dir / "fig_kv_budget_throughput.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
