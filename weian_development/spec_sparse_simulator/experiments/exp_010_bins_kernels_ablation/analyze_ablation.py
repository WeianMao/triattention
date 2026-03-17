"""
Ablation Study Results Analyzer

Analyzes ablation_summary.json and provides:
- Parameter count vs hit rate trade-off analysis
- Pareto frontier identification
- Optimal configuration recommendation

Usage:
    python analyze_ablation.py
    python analyze_ablation.py --summary-file path/to/ablation_summary.json
"""

import argparse
import json
from pathlib import Path


def load_summary(summary_file):
    """Load ablation summary from JSON file."""
    with open(summary_file, 'r') as f:
        return json.load(f)


def find_pareto_optimal(configurations, metric_key='50'):
    """
    Find Pareto-optimal configurations.

    A configuration is Pareto-optimal if no other configuration has:
    - Fewer parameters AND equal or higher hit rate
    - Equal parameters AND higher hit rate

    Args:
        configurations: List of configuration dicts
        metric_key: Which hit rate metric to use ('50', '500', '1000')

    Returns:
        List of Pareto-optimal configurations
    """
    successful = [c for c in configurations if c['status'] == 'success']

    pareto = []
    for cfg in successful:
        is_dominated = False
        cfg_params = cfg['total_params']
        cfg_hr = cfg.get('hit_rates', {}).get(metric_key, 0)

        for other in successful:
            if cfg == other:
                continue

            other_params = other['total_params']
            other_hr = other.get('hit_rates', {}).get(metric_key, 0)

            # Check if 'other' dominates 'cfg'
            if other_params <= cfg_params and other_hr >= cfg_hr:
                if other_params < cfg_params or other_hr > cfg_hr:
                    is_dominated = True
                    break

        if not is_dominated:
            pareto.append(cfg)

    return sorted(pareto, key=lambda x: x['total_params'])


def recommend_optimal(configurations, metric_key='50', max_hit_rate_loss_pct=2.0):
    """
    Recommend optimal configuration balancing parameters and accuracy.

    Strategy: Find the smallest model that loses at most max_hit_rate_loss_pct
    compared to the baseline (128 bins, 3 kernels).

    Args:
        configurations: List of configuration dicts
        metric_key: Which hit rate metric to use
        max_hit_rate_loss_pct: Maximum acceptable hit rate loss percentage

    Returns:
        Recommended configuration dict
    """
    successful = [c for c in configurations if c['status'] == 'success']

    # Find baseline (128 bins, 3 kernels)
    baseline = None
    for cfg in successful:
        if cfg['num_bins'] == 128 and cfg['num_kernels'] == 3:
            baseline = cfg
            break

    if baseline is None:
        print("Warning: Baseline (128 bins, 3 kernels) not found")
        # Return the one with highest hit rate
        return max(successful, key=lambda x: x.get('hit_rates', {}).get(metric_key, 0))

    baseline_hr = baseline.get('hit_rates', {}).get(metric_key, 0)

    # Find smallest model within acceptable loss
    candidates = []
    for cfg in successful:
        cfg_hr = cfg.get('hit_rates', {}).get(metric_key, 0)
        hr_loss = baseline_hr - cfg_hr

        if hr_loss <= max_hit_rate_loss_pct:
            candidates.append({
                **cfg,
                'hr_loss': hr_loss,
                'efficiency_score': (baseline_hr - hr_loss) / cfg['total_params'] * 1e6  # Higher is better
            })

    if not candidates:
        print(f"Warning: No configuration within {max_hit_rate_loss_pct}% hit rate loss")
        return baseline

    # Return the one with fewest parameters
    return min(candidates, key=lambda x: x['total_params'])


def analyze_ablation(summary_file, output_file=None):
    """
    Perform comprehensive ablation analysis.

    Args:
        summary_file: Path to ablation_summary.json
        output_file: Optional path to save analysis results
    """
    summary = load_summary(summary_file)
    configurations = summary['configurations']
    baseline_params = summary['baseline_params']

    print("=" * 80)
    print("ABLATION STUDY ANALYSIS REPORT")
    print("=" * 80)

    # 1. Overview
    print("\n## Overview")
    print(f"Total configurations tested: {summary['total_configs']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Baseline parameters (128 bins, 3 kernels): {baseline_params:,}")

    # 2. Results Table
    print("\n## Results Summary")
    print(f"\n{'Bins':<6} {'Kernels':<8} {'Params':<12} {'Reduction':<12} {'Hit@50':<10} {'Hit@500':<10} {'Hit@1000':<10}")
    print("-" * 80)

    successful = [c for c in configurations if c['status'] == 'success']
    for cfg in sorted(successful, key=lambda x: (-x['num_bins'], -x['num_kernels'])):
        hr = cfg.get('hit_rates', {})
        reduction = cfg['param_reduction_pct']
        print(
            f"{cfg['num_bins']:<6} {cfg['num_kernels']:<8} "
            f"{cfg['total_params']:<12,} {reduction:>8.1f}%    "
            f"{hr.get('50', 0):<10.2f} {hr.get('500', 0):<10.2f} {hr.get('1000', 0):<10.2f}"
        )

    # 3. Kernel Impact Analysis
    print("\n## Kernel Reduction Impact (3 -> 1)")
    print(f"\n{'Bins':<6} {'Hit@50 (3k)':<12} {'Hit@50 (1k)':<12} {'Delta':<10} {'Params (3k)':<14} {'Params (1k)':<14}")
    print("-" * 80)

    bins_values = sorted(set(c['num_bins'] for c in successful), reverse=True)
    for bins in bins_values:
        cfg_3k = next((c for c in successful if c['num_bins'] == bins and c['num_kernels'] == 3), None)
        cfg_1k = next((c for c in successful if c['num_bins'] == bins and c['num_kernels'] == 1), None)

        if cfg_3k and cfg_1k:
            hr_3k = cfg_3k.get('hit_rates', {}).get('50', 0)
            hr_1k = cfg_1k.get('hit_rates', {}).get('50', 0)
            delta = hr_1k - hr_3k
            print(
                f"{bins:<6} {hr_3k:<12.2f} {hr_1k:<12.2f} {delta:>+8.2f}   "
                f"{cfg_3k['total_params']:<14,} {cfg_1k['total_params']:<14,}"
            )

    # 4. Bins Impact Analysis
    print("\n## Bins Reduction Impact (for 1 kernel)")
    print(f"\n{'Bins':<6} {'Hit@50':<10} {'Delta vs 128':<14} {'Params':<12} {'Reduction':<12}")
    print("-" * 70)

    cfg_baseline_1k = next((c for c in successful if c['num_bins'] == 128 and c['num_kernels'] == 1), None)
    baseline_hr_1k = cfg_baseline_1k.get('hit_rates', {}).get('50', 0) if cfg_baseline_1k else 0

    for bins in bins_values:
        cfg = next((c for c in successful if c['num_bins'] == bins and c['num_kernels'] == 1), None)
        if cfg:
            hr = cfg.get('hit_rates', {}).get('50', 0)
            delta = hr - baseline_hr_1k
            print(
                f"{bins:<6} {hr:<10.2f} {delta:>+12.2f}   "
                f"{cfg['total_params']:<12,} {cfg['param_reduction_pct']:>8.1f}%"
            )

    # 5. Pareto Frontier
    print("\n## Pareto-Optimal Configurations (Hit@50)")
    pareto = find_pareto_optimal(configurations, metric_key='50')

    print(f"\n{'Bins':<6} {'Kernels':<8} {'Params':<12} {'Reduction':<12} {'Hit@50':<10}")
    print("-" * 50)
    for cfg in pareto:
        hr = cfg.get('hit_rates', {}).get('50', 0)
        print(
            f"{cfg['num_bins']:<6} {cfg['num_kernels']:<8} "
            f"{cfg['total_params']:<12,} {cfg['param_reduction_pct']:>8.1f}%    {hr:<10.2f}"
        )

    # 6. Recommendations
    print("\n## Recommendations")

    for max_loss in [1.0, 2.0, 5.0]:
        rec = recommend_optimal(configurations, metric_key='50', max_hit_rate_loss_pct=max_loss)
        hr = rec.get('hit_rates', {}).get('50', 0)
        print(f"\nMax {max_loss}% hit rate loss:")
        print(f"  -> {rec['num_bins']} bins, {rec['num_kernels']} kernels")
        print(f"  -> {rec['total_params']:,} params ({rec['param_reduction_pct']:.1f}% reduction)")
        print(f"  -> Hit@50: {hr:.2f}%")

    # 7. Save analysis
    analysis_result = {
        'summary': summary,
        'pareto_optimal': pareto,
        'recommendations': {
            'max_1pct_loss': recommend_optimal(configurations, metric_key='50', max_hit_rate_loss_pct=1.0),
            'max_2pct_loss': recommend_optimal(configurations, metric_key='50', max_hit_rate_loss_pct=2.0),
            'max_5pct_loss': recommend_optimal(configurations, metric_key='50', max_hit_rate_loss_pct=5.0),
        }
    }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(analysis_result, f, indent=2)
        print(f"\nAnalysis saved to: {output_file}")

    print("\n" + "=" * 80)
    print("END OF ANALYSIS REPORT")
    print("=" * 80)

    return analysis_result


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Results Analyzer')
    parser.add_argument(
        '--summary-file',
        type=str,
        default=None,
        help='Path to ablation_summary.json (default: output/ablation_summary.json)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Path to save analysis results (default: output/ablation_analysis.json)'
    )
    args = parser.parse_args()

    exp_dir = Path(__file__).parent

    summary_file = args.summary_file
    if summary_file is None:
        summary_file = exp_dir / 'output' / 'ablation_summary.json'

    output_file = args.output_file
    if output_file is None:
        output_file = exp_dir / 'output' / 'ablation_analysis.json'

    if not Path(summary_file).exists():
        print(f"Error: Summary file not found: {summary_file}")
        print("Please run ablation experiments first: python run_ablation.py --gpus 0,2,3,4,5")
        return

    analyze_ablation(summary_file, output_file)


if __name__ == '__main__':
    main()
