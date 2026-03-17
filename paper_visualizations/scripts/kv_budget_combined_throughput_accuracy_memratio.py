"""Generate combined figure: KV Budget vs Throughput + KV Memory Ratio vs Accuracy (AIME25).

This script creates a 2-panel figure combining:
- Panel (A): Throughput comparison at 16K generation length
- Panel (B): Accuracy comparison on AIME25 benchmark (Qwen3-8B) with KV Memory (%) x-axis

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
        description="Generate combined throughput + accuracy figure (with KV Memory ratio)"
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

    # ==========================================================================
    # KV Memory Ratio Configuration
    # ==========================================================================
    # Model: Qwen3-8B
    #   - num_hidden_layers: 36
    #   - num_key_value_heads: 8 (GQA)
    #   - head_dim: 128
    #
    # KV Cache Size formula (bf16):
    #   size_bytes = 2 (K+V) × num_layers × num_kv_heads × seq_len × head_dim × 2 (bf16)
    #              = 147456 × seq_len bytes
    #
    # >>> TO UPDATE SEQUENCE LENGTH: Modify the value below <<<
    # This is the full sequence length for Full Attention (100% KV Memory)
    # Default: 32768 (32K). Replace with actual value if different.
    FULL_SEQUENCE_LENGTH = 32768  # TODO: Replace with actual sequence length if needed

    def budget_to_memory_ratio(budget: int) -> float:
        """Convert KV budget to memory ratio (%) relative to full sequence length."""
        return (budget / FULL_SEQUENCE_LENGTH) * 100.0

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
        'FullKV': '*',  # Star marker for Full Attention point
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

    # -------------------------------------------------------------------------
    # >>> TO UPDATE THROUGHPUT DATA FOR 32K: Replace the values below <<<
    # Currently using 16K generation length throughput data.
    # To switch to 32K, update the following variables with 32K throughput values.
    # -------------------------------------------------------------------------
    # --- Throughput data (16K generation length, Fixed budget) ---
    throughput_fullkv = 222.76  # batch size 26
    throughput_rkv_budgets = [512, 1024, 2048, 3072, 4096]
    throughput_rkv = [1927.55, 1345.53, 760.44, 543.59, 279.01]
    throughput_triatt_budgets = [512, 1024, 2048, 3072, 4096]
    throughput_triatt = [2009.31, 1405.18, 787.09, 563.49, 413.89]

    # --- Accuracy data (AIME25, Qwen3-8B) ---
    accuracy_fullkv = 40.8
    accuracy_rkv_budgets = [512, 1024, 2048, 3072, 4096]
    accuracy_rkv = [6.0, 8.8, 17.5, 21.7, 31.7]
    accuracy_triatt_budgets = [512, 1024, 2048, 3072, 4096]
    accuracy_triatt = [8.3, 15.4, 32.9, 40.8, 43.3]

    # --- Panel A data: Throughput (x) vs Accuracy (y) ---
    # Match throughput and accuracy by budget
    # R-KV: all 5 budgets have both throughput and accuracy
    panelA_rkv_throughput = throughput_rkv  # [512, 1024, 2048, 3072, 4096]
    panelA_rkv_accuracy = accuracy_rkv
    # TriAttention: all 5 budgets have both throughput and accuracy
    panelA_triatt_throughput = throughput_triatt  # [512, 1024, 2048, 3072, 4096]
    panelA_triatt_accuracy = accuracy_triatt

    # Convert budgets to memory ratios for Panel B
    accuracy_rkv_mem_ratios = [budget_to_memory_ratio(b) for b in accuracy_rkv_budgets]
    accuracy_triatt_mem_ratios = [budget_to_memory_ratio(b) for b in accuracy_triatt_budgets]
    fullkv_mem_ratio = 100.0  # Full Attention = 100%

    # ==========================================================================
    # Create figure with 2 panels
    # ==========================================================================
    fig = plt.figure(figsize=(8, 4.5), dpi=args.dpi, constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, wspace=0.02)

    # --- Panel (A): Throughput (x) vs Accuracy (y) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_box_aspect(1)
    style_ax(ax1)

    # Full Attention as a single point (plotted first for legend order)
    ax1.scatter([throughput_fullkv], [accuracy_fullkv],
                color=colors['FullKV'], marker=markers['FullKV'],
                s=350, label='Full Attention', alpha=0.9, zorder=5,
                edgecolors='white', linewidths=1.5)

    # R-KV
    ax1.plot(panelA_rkv_throughput, panelA_rkv_accuracy,
             color=colors['R-KV'], linestyle='-', marker=markers['R-KV'],
             markersize=9, linewidth=2.5, label='R-KV', alpha=0.9)

    # TriAttention
    ax1.plot(panelA_triatt_throughput, panelA_triatt_accuracy,
             color=colors['TriAttention'], linestyle='-', marker=markers['TriAttention'],
             markersize=9, linewidth=2.5, label='TriAttention', alpha=0.9)

    ax1.set_xlabel('Throughput (tokens/s)', fontsize=FONT_SIZE)
    ax1.set_ylabel('Accuracy (%)', fontsize=FONT_SIZE)
    ax1.set_title('Throughput $vs.$ Accuracy', fontsize=FONT_SIZE + 2, fontweight='bold', pad=10)

    # X-axis: log scale
    ax1.set_xscale('log')
    ax1.set_xticks([200, 300, 500, 1000, 2000])
    ax1.set_xticklabels(['200', '300', '500', '1000', '2000'])
    ax1.set_xlim(150, 2500)
    # Y-axis: match Panel B's range and ticks
    ax1.set_ylim(0, 48)
    ax1.set_yticks([10, 20, 30, 40])
    # Remove minor ticks (matching Panel B style)
    ax1.minorticks_off()
    ax1.tick_params(axis='x', which='both', length=0)
    ax1.tick_params(axis='y', which='both', length=0)

    ax1.text(-0.08, 1.08, '(A)', transform=ax1.transAxes,
             fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom',
             fontname=LABEL_FONT)
    ax1.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='lower left')

    # --- Annotation: TriAttention at budget=3072 matches Full Attention accuracy ---
    # TriAttention at budget=3072: throughput=563.49, accuracy=40.8
    # Full Attention: throughput=222.76, accuracy=40.8
    # Speedup: 563.49 / 222.76 = 2.5x
    triatt_match_throughput = 563.49
    speedup = triatt_match_throughput / throughput_fullkv  # 2.5x
    # Shorten arrow to avoid overlapping with markers
    arrow_left = throughput_fullkv * 1.08  # move right from Full Attention
    arrow_right = triatt_match_throughput * 0.92  # move left from TriAttention
    ax1.annotate('', xy=(arrow_right, accuracy_fullkv),
                 xytext=(arrow_left, accuracy_fullkv),
                 arrowprops=dict(arrowstyle='<->', color='#E24A33', lw=2.5,
                                 mutation_scale=18))
    # Add label below the arrow
    mid_x = (arrow_left * arrow_right) ** 0.5  # geometric mean for log scale
    ax1.text(mid_x, accuracy_fullkv - 2.5, f'{speedup:.1f}X Faster',
             ha='center', va='top', fontsize=FONT_SIZE + 2, fontweight='bold',
             color='#E24A33')

    # --- Panel (B): Accuracy (AIME25) with KV Memory (%) x-axis ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_box_aspect(1)
    style_ax(ax2)

    # Full Attention as a single point (plotted first for legend order)
    ax2.scatter([fullkv_mem_ratio], [accuracy_fullkv],
                color=colors['FullKV'], marker=markers['FullKV'],
                s=350, label='Full Attention', alpha=0.9, zorder=5,
                edgecolors='white', linewidths=1.5)

    # R-KV
    ax2.plot(accuracy_rkv_mem_ratios, accuracy_rkv,
             color=colors['R-KV'], linestyle='-', marker=markers['R-KV'],
             markersize=9, linewidth=2.5, label='R-KV', alpha=0.9)

    # TriAttention
    ax2.plot(accuracy_triatt_mem_ratios, accuracy_triatt,
             color=colors['TriAttention'], linestyle='-', marker=markers['TriAttention'],
             markersize=9, linewidth=2.5, label='TriAttention', alpha=0.9)

    ax2.set_xlabel('KV Cache Memory (%)', fontsize=FONT_SIZE)
    ax2.set_ylabel('Accuracy (%)', fontsize=FONT_SIZE)
    ax2.set_title('KV Memory $vs.$ Accuracy', fontsize=FONT_SIZE + 2, fontweight='bold', pad=10)
    # Y-axis: same as Panel A
    ax2.set_ylim(0, 48)
    ax2.set_yticks([10, 20, 30, 40])

    # X-axis: log scale to better show the distribution
    ax2.set_xscale('log')
    # Use clean tick values that align well
    ax2.set_xticks([1, 3, 10, 30, 100])
    ax2.set_xticklabels(['1', '3', '10', '30', '100'])
    ax2.set_xlim(1, 120)
    # Remove minor ticks and ensure no tick marks (matching Panel A style)
    ax2.minorticks_off()
    ax2.tick_params(axis='x', which='both', length=0)
    ax2.tick_params(axis='y', which='both', length=0)

    ax2.text(-0.08, 1.08, '(B)', transform=ax2.transAxes,
             fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom',
             fontname=LABEL_FONT)
    ax2.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='lower right')

    # --- Annotation: TriAttention at budget=3072 matches Full Attention accuracy ---
    # TriAttention at budget=3072: mem_ratio=9.38%, accuracy=40.8
    # Full Attention: mem_ratio=100%, accuracy=40.8
    # Memory saving: 100 / 9.38 = 10.7x
    triatt_match_mem_ratio = budget_to_memory_ratio(3072)  # 9.38%
    mem_saving = fullkv_mem_ratio / triatt_match_mem_ratio  # 10.7x
    # Shorten arrow to avoid overlapping with markers
    arrow_left = triatt_match_mem_ratio * 1.15  # move right from TriAttention
    arrow_right = fullkv_mem_ratio * 0.92  # move left from Full Attention
    ax2.annotate('', xy=(arrow_right, accuracy_fullkv),
                 xytext=(arrow_left, accuracy_fullkv),
                 arrowprops=dict(arrowstyle='<->', color='#E24A33', lw=2.5,
                                 mutation_scale=18))
    # Add label below the arrow
    mid_x = (arrow_left * arrow_right) ** 0.5  # geometric mean for log scale
    ax2.text(mid_x, accuracy_fullkv - 2.5, f'{mem_saving:.1f}X Smaller',
             ha='center', va='top', fontsize=FONT_SIZE + 2, fontweight='bold',
             color='#E24A33')

    output_path = args.output_dir / "fig_kv_budget_throughput_accuracy_memratio.png"
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"\nKV Memory Ratio mapping (FULL_SEQUENCE_LENGTH={FULL_SEQUENCE_LENGTH}):")
    for budget, ratio in zip(accuracy_rkv_budgets, accuracy_rkv_mem_ratios):
        print(f"  Budget {budget:4d} -> {ratio:6.2f}%")
    print(f"  Full Attention -> {fullkv_mem_ratio:6.2f}%")


if __name__ == "__main__":
    main()
