"""
Visualization Suite for Probe Activation Loss Ablation Study

Generates 4 visualizations:
1. Loss curves over epochs for each lambda value
2. Metric comparison bar charts (Hit Rate per K value)
3. Bin utilization heatmap (optional, requires bin distribution data)
4. Summary report text file

Usage:
    python visualize_ablation.py --results-dir output/ablation
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_ablation_results(results_dir):
    """
    Load ablation results from JSON file.

    Args:
        results_dir: Path to ablation results directory

    Returns:
        dict: Results dict keyed by experiment name with evaluation metrics
    """
    results_path = Path(results_dir) / 'ablation_results.json'

    if not results_path.exists():
        raise FileNotFoundError(f"Ablation results not found: {results_path}")

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results


def plot_loss_curves(results, output_dir, logger=None):
    """
    Plot training loss curves for each lambda configuration.

    Note: This requires loss history data which may not be saved in evaluation.
    If loss_history is not available, this function will skip gracefully.

    Args:
        results: dict from load_ablation_results
        output_dir: Path to save plot
        logger: Optional logger instance
    """
    plt.figure(figsize=(10, 6))

    has_loss_data = False

    # Sort results by lambda value for consistent legend
    sorted_items = sorted(
        results.items(),
        key=lambda x: x[1].get('lambda_activation', 0.0)
    )

    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_items)))

    for (exp_name, data), color in zip(sorted_items, colors):
        if 'error' in data:
            continue

        # Check if loss_history is available
        if 'loss_history' in data:
            loss_history = data['loss_history']
            epochs = range(1, len(loss_history) + 1)
            lambda_val = data.get('lambda_activation', 0.0)
            plt.plot(epochs, loss_history, label=f'λ={lambda_val}', color=color, linewidth=2)
            has_loss_data = True

    if not has_loss_data:
        # If no loss history, create placeholder message
        plt.text(
            0.5, 0.5,
            'Loss history not saved in ablation results.\n'
            'Re-run ablation with loss logging enabled.',
            ha='center', va='center', fontsize=12,
            transform=plt.gca().transAxes
        )
        if logger:
            logger.info("No loss history data available. Skipping loss curves plot.")

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.title('Training Loss Curves (Ablation Study)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    save_path = Path(output_dir) / 'loss_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Loss curves saved to: {save_path}")

    return save_path


def plot_metric_comparison(results, output_dir, logger=None):
    """
    Create grouped bar chart comparing metrics across lambda values.

    Args:
        results: dict from load_ablation_results
        output_dir: Path to save plot
        logger: Optional logger instance
    """
    # Extract data for plotting
    lambda_vals = []
    hit_rates_k50 = []
    hit_rates_k500 = []
    hit_rates_k1000 = []

    # Sort by lambda value
    sorted_items = sorted(
        results.items(),
        key=lambda x: x[1].get('lambda_activation', 0.0)
    )

    for exp_name, data in sorted_items:
        if 'error' in data:
            continue

        eval_data = data.get('evaluation', {})
        if not eval_data:
            continue

        lambda_val = data.get('lambda_activation', 0.0)
        lambda_vals.append(f'λ={lambda_val}')

        # Extract hit rates for each K (evaluation keys are integers or strings)
        k50_data = eval_data.get(50) or eval_data.get('50', {})
        k500_data = eval_data.get(500) or eval_data.get('500', {})
        k1000_data = eval_data.get(1000) or eval_data.get('1000', {})

        hit_rates_k50.append(k50_data.get('hit_rate', 0) * 100 if isinstance(k50_data.get('hit_rate', 0), float) and k50_data.get('hit_rate', 0) <= 1 else k50_data.get('hit_rate', 0))
        hit_rates_k500.append(k500_data.get('hit_rate', 0) * 100 if isinstance(k500_data.get('hit_rate', 0), float) and k500_data.get('hit_rate', 0) <= 1 else k500_data.get('hit_rate', 0))
        hit_rates_k1000.append(k1000_data.get('hit_rate', 0) * 100 if isinstance(k1000_data.get('hit_rate', 0), float) and k1000_data.get('hit_rate', 0) <= 1 else k1000_data.get('hit_rate', 0))

    if not lambda_vals:
        if logger:
            logger.warning("No valid results for metric comparison plot")
        return None

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(lambda_vals))
    width = 0.6
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # K=50
    bars1 = axes[0].bar(x, hit_rates_k50, width, color=colors[0], edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Lambda Value', fontsize=11)
    axes[0].set_ylabel('Hit Rate (%)', fontsize=11)
    axes[0].set_title('TopK Hit Rate (K=50)', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lambda_vals, rotation=45, ha='right')
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, val in zip(bars1, hit_rates_k50):
        axes[0].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         ha='center', va='bottom', fontsize=9)

    # K=500
    bars2 = axes[1].bar(x, hit_rates_k500, width, color=colors[1], edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Lambda Value', fontsize=11)
    axes[1].set_ylabel('Hit Rate (%)', fontsize=11)
    axes[1].set_title('TopK Hit Rate (K=500)', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(lambda_vals, rotation=45, ha='right')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, hit_rates_k500):
        axes[1].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         ha='center', va='bottom', fontsize=9)

    # K=1000
    bars3 = axes[2].bar(x, hit_rates_k1000, width, color=colors[2], edgecolor='black', linewidth=0.5)
    axes[2].set_xlabel('Lambda Value', fontsize=11)
    axes[2].set_ylabel('Hit Rate (%)', fontsize=11)
    axes[2].set_title('TopK Hit Rate (K=1000)', fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(lambda_vals, rotation=45, ha='right')
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, hit_rates_k1000):
        axes[2].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         ha='center', va='bottom', fontsize=9)

    plt.suptitle('Probe Activation Loss Ablation: TopK Hit Rate Comparison', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = Path(output_dir) / 'metric_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Metric comparison saved to: {save_path}")

    return save_path


def plot_bin_utilization_heatmap(results, output_dir, num_bins=128, logger=None):
    """
    Create heatmap showing bin activation patterns per lambda value.

    Note: Requires bin utilization data which may need to be added to evaluation.
    If not available, generates placeholder.

    Args:
        results: dict from load_ablation_results
        output_dir: Path to save plot
        num_bins: Number of bins in the model
        logger: Optional logger instance
    """
    # Try to extract bin utilization data
    lambda_vals = []
    bin_data_list = []

    sorted_items = sorted(
        results.items(),
        key=lambda x: x[1].get('lambda_activation', 0.0)
    )

    for exp_name, data in sorted_items:
        if 'error' in data:
            continue

        lambda_val = data.get('lambda_activation', 0.0)

        # Check for bin utilization data
        if 'bin_utilization' in data:
            lambda_vals.append(f'λ={lambda_val}')
            bin_data_list.append(data['bin_utilization'])
        elif 'bin_distribution' in data:
            lambda_vals.append(f'λ={lambda_val}')
            bin_data_list.append(data['bin_distribution'])

    plt.figure(figsize=(14, 6))

    if not bin_data_list:
        # Create placeholder
        plt.text(
            0.5, 0.5,
            'Bin utilization data not saved in ablation results.\n'
            'Re-run ablation with bin distribution logging enabled.',
            ha='center', va='center', fontsize=12,
            transform=plt.gca().transAxes
        )
        if logger:
            logger.info("No bin utilization data available. Skipping heatmap.")
    else:
        # Create heatmap
        data_matrix = np.array(bin_data_list)
        im = plt.imshow(data_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

        plt.colorbar(im, label='Bin Utilization Rate')
        plt.xlabel('Bin Index', fontsize=12)
        plt.ylabel('Lambda Configuration', fontsize=12)
        plt.title('Bin Utilization Heatmap (Ablation Study)', fontsize=14)

        plt.yticks(range(len(lambda_vals)), lambda_vals)

        # Add tick marks for bins
        if num_bins <= 32:
            plt.xticks(range(num_bins))
        else:
            plt.xticks(np.arange(0, num_bins, 16))

    save_path = Path(output_dir) / 'bin_utilization.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Bin utilization heatmap saved to: {save_path}")

    return save_path


def generate_summary_report(results, output_dir, logger=None):
    """
    Generate text summary report with best configuration and recommendations.

    Args:
        results: dict from load_ablation_results
        output_dir: Path to save report
        logger: Optional logger instance
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PROBE ACTIVATION LOSS ABLATION STUDY - SUMMARY REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Extract metrics for analysis
    metrics_data = []

    for exp_name, data in results.items():
        if 'error' in data:
            metrics_data.append({
                'lambda': data.get('lambda_activation', 0.0),
                'exp_name': exp_name,
                'error': data['error']
            })
            continue

        eval_data = data.get('evaluation', {})
        if not eval_data:
            continue

        # Extract K=1000 hit rate as primary metric
        k1000_data = eval_data.get(1000) or eval_data.get('1000', {})
        k500_data = eval_data.get(500) or eval_data.get('500', {})
        k50_data = eval_data.get(50) or eval_data.get('50', {})

        metrics_data.append({
            'lambda': data.get('lambda_activation', 0.0),
            'exp_name': exp_name,
            'k50_hit_rate': k50_data.get('hit_rate', 0),
            'k500_hit_rate': k500_data.get('hit_rate', 0),
            'k1000_hit_rate': k1000_data.get('hit_rate', 0),
        })

    # Sort by lambda
    metrics_data.sort(key=lambda x: x['lambda'])

    # Print all results
    lines.append("EXPERIMENTAL RESULTS")
    lines.append("-" * 60)
    lines.append(f"{'Lambda':<12} {'K=50 Hit%':<14} {'K=500 Hit%':<14} {'K=1000 Hit%':<14}")
    lines.append("-" * 60)

    baseline_k1000 = None
    best_lambda = None
    best_k1000 = -1

    for m in metrics_data:
        if 'error' in m:
            lines.append(f"{m['lambda']:<12} ERROR: {m['error'][:30]}...")
            continue

        k50 = m['k50_hit_rate']
        k500 = m['k500_hit_rate']
        k1000 = m['k1000_hit_rate']

        # Format percentages (handle both 0-1 and 0-100 ranges)
        k50_pct = k50 * 100 if k50 <= 1 else k50
        k500_pct = k500 * 100 if k500 <= 1 else k500
        k1000_pct = k1000 * 100 if k1000 <= 1 else k1000

        lines.append(f"{m['lambda']:<12} {k50_pct:<14.2f} {k500_pct:<14.2f} {k1000_pct:<14.2f}")

        # Track baseline (lambda=0.0)
        if m['lambda'] == 0.0:
            baseline_k1000 = k1000_pct

        # Track best
        if k1000_pct > best_k1000:
            best_k1000 = k1000_pct
            best_lambda = m['lambda']

    lines.append("")

    # Best configuration
    lines.append("BEST CONFIGURATION")
    lines.append("-" * 60)
    if best_lambda is not None:
        lines.append(f"Best lambda_activation: {best_lambda}")
        lines.append(f"Best K=1000 Hit Rate: {best_k1000:.2f}%")

        if baseline_k1000 is not None and baseline_k1000 > 0:
            improvement = ((best_k1000 - baseline_k1000) / baseline_k1000) * 100
            lines.append(f"Improvement over baseline: {improvement:+.2f}%")

            if improvement > 0:
                lines.append("")
                lines.append("CONCLUSION: Probe Activation Loss IMPROVES hit rate.")
            elif improvement < 0:
                lines.append("")
                lines.append("CONCLUSION: Probe Activation Loss DEGRADES hit rate.")
            else:
                lines.append("")
                lines.append("CONCLUSION: Probe Activation Loss has NO EFFECT.")
    else:
        lines.append("No valid results to determine best configuration.")

    lines.append("")

    # Recommendations
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 60)
    if best_lambda is not None and best_lambda > 0:
        lines.append(f"1. Use lambda_activation={best_lambda} for production")
        lines.append("2. Consider further fine-tuning around this value")
        lines.append("3. Verify results with cross-trace validation")
    elif best_lambda == 0.0:
        lines.append("1. Probe Activation Loss may not be beneficial for this task")
        lines.append("2. Consider alternative anti-collapse strategies")
        lines.append("3. Review bin utilization patterns for collapse diagnosis")
    else:
        lines.append("1. Re-run experiments with more lambda values")
        lines.append("2. Check for training stability issues")

    lines.append("")
    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    # Write report
    report_content = '\n'.join(lines)
    save_path = Path(output_dir) / 'summary_report.txt'

    with open(save_path, 'w') as f:
        f.write(report_content)

    if logger:
        logger.info(f"Summary report saved to: {save_path}")
        logger.info("\n" + report_content)

    return save_path


def visualize_all(results_dir, logger=None):
    """
    Run all visualization functions.

    Args:
        results_dir: Path to ablation results directory
        logger: Optional logger instance
    """
    results = load_ablation_results(results_dir)
    output_dir = Path(results_dir)

    if logger:
        logger.info(f"Loaded {len(results)} ablation experiments")

    # Generate all visualizations
    plot_loss_curves(results, output_dir, logger)
    plot_metric_comparison(results, output_dir, logger)
    plot_bin_utilization_heatmap(results, output_dir, logger=logger)
    generate_summary_report(results, output_dir, logger)

    if logger:
        logger.info("All visualizations completed")


def main():
    """Main entry point."""
    import logging
    import sys

    parser = argparse.ArgumentParser(description='Visualize Ablation Study Results')
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Path to ablation results directory (contains ablation_results.json)'
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    try:
        visualize_all(args.results_dir, logger)
        logger.info("Visualization completed successfully")
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
