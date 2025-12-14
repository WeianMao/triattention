#!/usr/bin/env python3
"""Visualize Top-3 Frequency Band NMS Experiment Results.

This script creates visualizations showing the results of the Top-3 frequency band NMS experiment,
including per-head retention rates, NMS drop statistics, and summary comparisons.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

mask_process_command("PD-L1_binder_top3_viz")


def load_metrics(metrics_path: Path) -> dict:
    """Load retention_metrics.json.

    Args:
        metrics_path: Path to retention_metrics.json

    Returns:
        dict: Loaded metrics data

    Raises:
        FileNotFoundError: If metrics file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path) as f:
        return json.load(f)


def plot_per_head_retention(metrics: dict, output_dir: Path) -> Path:
    """Create bar chart showing retention rate per (layer, head).

    Args:
        metrics: Dictionary containing retention metrics
        output_dir: Directory to save plot

    Returns:
        Path to saved plot file
    """
    per_head = metrics.get("per_head", [])
    if not per_head:
        print("Warning: No per_head data found in metrics")
        return None

    # Extract data
    head_labels = [f"L{h['layer']}H{h['head']}" for h in per_head]
    retentions = [h['retention'] * 100 for h in per_head]  # Convert to percentage

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

    # Create bar chart
    bars = ax.bar(range(len(head_labels)), retentions, color='steelblue', alpha=0.7)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, retentions)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=8)

    # Add horizontal line for overall retention
    overall_retention = metrics.get("overall_retention", 0) * 100
    ax.axhline(overall_retention, color='red', linestyle='--', linewidth=2,
               label=f'Overall Retention: {overall_retention:.1f}%')

    # Customize plot
    ax.set_xlabel('(Layer, Head)', fontsize=12)
    ax.set_ylabel('Retention Rate (%)', fontsize=12)
    ax.set_title('Top-3 Frequency NMS: Per-Head Retention Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(head_labels)))
    ax.set_xticklabels(head_labels, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 110)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "top3_per_head_retention.png"
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Saved per-head retention plot to {output_path}")
    return output_path


def plot_nms_drop_statistics(metrics: dict, output_dir: Path) -> Path:
    """Create visualization showing NMS drop counts and rates.

    Args:
        metrics: Dictionary containing retention metrics
        output_dir: Directory to save plot

    Returns:
        Path to saved plot file
    """
    # Extract NMS statistics
    nms_drop_rate = metrics.get("nms_drop_rate", 0)
    nms_total_drops = metrics.get("nms_total_drops", 0)
    nms_total_rounds = metrics.get("nms_total_rounds", 0)
    nms_drop_count_per_head = metrics.get("nms_drop_count_per_head", [])
    per_head = metrics.get("per_head", [])

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

    # Subplot 1: Overall NMS statistics
    categories = ['Total Rounds', 'Total Drops', 'Avg Drops/Round']
    values = [
        nms_total_rounds,
        nms_total_drops,
        nms_total_drops / max(nms_total_rounds, 1)
    ]
    colors = ['skyblue', 'salmon', 'lightgreen']

    bars1 = ax1.bar(categories, values, color=colors, alpha=0.7)

    # Add value labels
    for bar, val in zip(bars1, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'NMS Drop Statistics (Drop Rate: {nms_drop_rate:.2f}%)',
                  fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Subplot 2: Per-head drop counts
    if nms_drop_count_per_head and per_head:
        head_labels = [f"L{h['layer']}H{h['head']}" for h in per_head]

        bars2 = ax2.bar(range(len(head_labels)), nms_drop_count_per_head,
                        color='coral', alpha=0.7)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, nms_drop_count_per_head)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:,}',
                    ha='center', va='bottom', fontsize=7, rotation=0)

        # Add mean line
        mean_drops = np.mean(nms_drop_count_per_head)
        ax2.axhline(mean_drops, color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_drops:.0f}')

        ax2.set_xlabel('(Layer, Head)', fontsize=12)
        ax2.set_ylabel('Drop Count', fontsize=12)
        ax2.set_title('NMS Drops per Head', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(head_labels)))
        ax2.set_xticklabels(head_labels, rotation=45, ha='right')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Top-3 Frequency NMS Drop Statistics', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / "top3_nms_drop_statistics.png"
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved NMS drop statistics plot to {output_path}")
    return output_path


def plot_summary_comparison(metrics: dict, output_dir: Path) -> Path:
    """Create summary metrics visualization.

    Args:
        metrics: Dictionary containing retention metrics
        output_dir: Directory to save plot

    Returns:
        Path to saved plot file
    """
    # Extract key metrics
    overall_retention = metrics.get("overall_retention", 0) * 100
    nms_drop_rate = metrics.get("nms_drop_rate", 0)
    head_count = metrics.get("head_count", 0)
    round_window = metrics.get("round_window", 0)

    # Calculate retention vs drop percentages
    retention_pct = overall_retention
    drop_pct = nms_drop_rate

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(14, 6), dpi=100)

    # Subplot 1: Retention vs Drop pie chart
    ax1 = fig.add_subplot(121)

    # Only show pie if we have valid data
    if retention_pct + drop_pct > 0:
        sizes = [retention_pct, drop_pct]
        labels = [f'Retained\n{retention_pct:.1f}%', f'Dropped\n{drop_pct:.1f}%']
        colors = ['#66b3ff', '#ff9999']
        explode = (0.05, 0.05)

        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors,
                                            explode=explode, autopct='',
                                            shadow=True, startangle=90)

        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')

        ax1.set_title('Overall Retention vs Drop Rate', fontsize=13, fontweight='bold')

    # Subplot 2: Key metrics table
    ax2 = fig.add_subplot(122)
    ax2.axis('tight')
    ax2.axis('off')

    # Prepare table data
    table_data = [
        ['Metric', 'Value'],
        ['Overall Retention', f'{retention_pct:.2f}%'],
        ['NMS Drop Rate', f'{drop_pct:.2f}%'],
        ['Total Heads Analyzed', f'{head_count}'],
        ['Round Window Size', f'{round_window}'],
        ['Total Rounds', f"{metrics.get('nms_total_rounds', 0):,}"],
        ['Total Drops', f"{metrics.get('nms_total_drops', 0):,}"],
        ['Avg Drops/Round', f"{metrics.get('nms_total_drops', 0) / max(metrics.get('nms_total_rounds', 1), 1):.1f}"],
    ]

    # Create table
    table = ax2.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(2):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('white')

    ax2.set_title('Experiment Summary', fontsize=13, fontweight='bold', pad=20)

    plt.suptitle('Top-3 Frequency NMS Experiment Summary', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / "top3_summary_comparison.png"
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved summary comparison plot to {output_path}")
    return output_path


def plot_per_layer_retention(metrics: dict, output_dir: Path) -> Path:
    """Create bar chart showing retention rate per layer (averaged across heads).

    Args:
        metrics: Dictionary containing retention metrics
        output_dir: Directory to save plot

    Returns:
        Path to saved plot file
    """
    per_layer = metrics.get("per_layer", {})
    if not per_layer:
        print("Warning: No per_layer data found in metrics")
        return None

    # Extract and sort by layer number
    layers = sorted([int(k) for k in per_layer.keys()])
    retentions = [per_layer[str(layer)] * 100 for layer in layers]  # Convert to percentage

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

    # Create bar chart
    bars = ax.bar(range(len(layers)), retentions, color='mediumseagreen', alpha=0.7)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, retentions)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9)

    # Add horizontal line for overall retention
    overall_retention = metrics.get("overall_retention", 0) * 100
    ax.axhline(overall_retention, color='red', linestyle='--', linewidth=2,
               label=f'Overall: {overall_retention:.1f}%')

    # Customize plot
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Retention Rate (%)', fontsize=12)
    ax.set_title('Top-3 Frequency NMS: Per-Layer Retention Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{layer}' for layer in layers], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 110)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "top3_per_layer_retention.png"
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Saved per-layer retention plot to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize Top-3 Frequency Band NMS Experiment Results"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to retention_metrics.json file from IMPL-005"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weian_development/online_k_pruning_viz/plots"),
        help="Output directory for plots (default: weian_development/online_k_pruning_viz/plots)"
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # Load metrics
    print(f"Loading metrics from {args.results}")
    metrics = load_metrics(args.results)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Generate all plots
    print("\nGenerating plots...")
    plots_generated = []

    # 1. Per-head retention
    plot_path = plot_per_head_retention(metrics, args.output_dir)
    if plot_path:
        plots_generated.append(plot_path)

    # 2. NMS drop statistics
    plot_path = plot_nms_drop_statistics(metrics, args.output_dir)
    if plot_path:
        plots_generated.append(plot_path)

    # 3. Summary comparison
    plot_path = plot_summary_comparison(metrics, args.output_dir)
    if plot_path:
        plots_generated.append(plot_path)

    # 4. Per-layer retention (bonus plot)
    plot_path = plot_per_layer_retention(metrics, args.output_dir)
    if plot_path:
        plots_generated.append(plot_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUCCESS: Generated {len(plots_generated)} plots:")
    for i, plot_path in enumerate(plots_generated, 1):
        print(f"  {i}. {plot_path.name}")
    print(f"{'='*60}")

    # Print key metrics
    print(f"\nKey Metrics:")
    print(f"  Overall Retention: {metrics.get('overall_retention', 0)*100:.2f}%")
    print(f"  NMS Drop Rate: {metrics.get('nms_drop_rate', 0):.2f}%")
    print(f"  Total Drops: {metrics.get('nms_total_drops', 0):,}")
    print(f"  Total Rounds: {metrics.get('nms_total_rounds', 0):,}")
    print(f"  Heads Analyzed: {metrics.get('head_count', 0)}")


if __name__ == "__main__":
    main()
