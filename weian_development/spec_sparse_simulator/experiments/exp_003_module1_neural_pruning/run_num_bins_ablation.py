"""
num_bins Ablation Study for Module 1 Neural Network Key Pruning.

Runs experiments with different num_bins (kernel encoding output dim) values
in parallel across available GPUs. Collects results and generates comparison report.

Fixed parameters:
  - num_kernels: 1
  - mlp_hidden: 32
  - kappa_init: 2.5

Usage:
    python run_num_bins_ablation.py --num-bins-list 64 32 16 8
    python run_num_bins_ablation.py --num-bins-list 64 32 16 8 --gpus 0 1 2 3
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Experiment directory
EXP_DIR = Path(__file__).parent

# Conda environment Python path
DC_PYTHON = '/data/rbg/users/weian/env/miniconda3/envs/dc/bin/python'


def setup_logging():
    """Setup logging configuration."""
    log_dir = EXP_DIR / 'output' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'num_bins_ablation_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ],
        force=True
    )
    return logging.getLogger(__name__)


def get_available_gpus():
    """Get list of available GPU indices."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = [int(idx.strip()) for idx in result.stdout.strip().split('\n') if idx.strip()]
        return gpus
    except Exception:
        return []


def run_single_experiment(num_bins, gpu_id, logger_name):
    """
    Run a single experiment with specified num_bins on specified GPU.

    Fixed parameters:
      - num_kernels: 1
      - mlp_hidden: 32
      - kappa_init: 2.5

    Args:
        num_bins: Number of kernel encoding output bins
        gpu_id: GPU index to use
        logger_name: Logger name for output

    Returns:
        dict: Experiment results or error info
    """
    logger = logging.getLogger(logger_name)

    experiment_name = f'num_bins_ablation_b{num_bins}'

    logger.info(f"[GPU {gpu_id}] Starting experiment: num_bins={num_bins}")

    # Build command with fixed parameters
    cmd = [
        DC_PYTHON,
        str(EXP_DIR / 'run_pruning_experiment.py'),
        '--num-bins', str(num_bins),
        '--num-kernels', '1',
        '--mlp-hidden-dim', '32',
        '--kappa-init', '2.5',
        '--experiment-name', experiment_name
    ]

    # Set environment with specific GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(EXP_DIR),
            timeout=7200  # 120 minute timeout
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            logger.error(f"[GPU {gpu_id}] Experiment failed for num_bins={num_bins}")
            logger.error(f"STDERR: {result.stderr[-1000:]}")  # Last 1000 chars
            return {
                'num_bins': num_bins,
                'status': 'failed',
                'error': result.stderr[-500:],
                'elapsed_time': elapsed
            }

        # Load results from output file
        results_path = EXP_DIR / 'output' / 'pruning_experiments' / f'{experiment_name}.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                exp_results = json.load(f)

            logger.info(f"[GPU {gpu_id}] Experiment completed for num_bins={num_bins}")
            logger.info(f"  Hit rate: {exp_results['metrics']['hit_rate']*100:.2f}%")
            logger.info(f"  Keys/Query: {exp_results['metrics']['keys_per_query']:.2f}")
            logger.info(f"  Params: {exp_results['metrics']['param_count']:,}")
            logger.info(f"  Time: {elapsed:.1f}s")

            return {
                'num_bins': num_bins,
                'status': 'success',
                'results': exp_results,
                'elapsed_time': elapsed
            }
        else:
            logger.error(f"[GPU {gpu_id}] Results file not found: {results_path}")
            return {
                'num_bins': num_bins,
                'status': 'failed',
                'error': 'Results file not found',
                'elapsed_time': elapsed
            }

    except subprocess.TimeoutExpired:
        logger.error(f"[GPU {gpu_id}] Experiment timed out for num_bins={num_bins}")
        return {
            'num_bins': num_bins,
            'status': 'timeout',
            'error': 'Experiment timed out after 120 minutes',
            'elapsed_time': 7200
        }
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Unexpected error for num_bins={num_bins}: {e}")
        return {
            'num_bins': num_bins,
            'status': 'error',
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }


def generate_report(all_results, logger):
    """Generate comparison report from all experiment results."""

    logger.info("\n" + "=" * 80)
    logger.info("NUM_BINS ABLATION STUDY - RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info("Fixed parameters: num_kernels=1, mlp_hidden=32, kappa_init=2.5")
    logger.info("=" * 80)

    # Sort results by num_bins (descending)
    successful_results = [r for r in all_results if r['status'] == 'success']
    successful_results.sort(key=lambda x: x['num_bins'], reverse=True)

    if not successful_results:
        logger.error("No successful experiments!")
        return None

    # Print table header
    logger.info("")
    logger.info(f"{'num_bins':<10} {'Params':<12} {'Hit Rate':<12} {'Keys/Query':<12} {'Threshold':<12} {'Time (s)':<10}")
    logger.info("-" * 68)

    report_data = []

    for r in successful_results:
        metrics = r['results']['metrics']
        row = {
            'num_bins': r['num_bins'],
            'param_count': metrics['param_count'],
            'hit_rate': metrics['hit_rate'],
            'keys_per_query': metrics['keys_per_query'],
            'optimal_threshold': metrics.get('optimal_threshold', 'N/A'),
            'elapsed_time': r['elapsed_time']
        }
        report_data.append(row)

        threshold_str = f"{row['optimal_threshold']:.4f}" if isinstance(row['optimal_threshold'], float) else 'N/A'

        logger.info(
            f"{row['num_bins']:<10} "
            f"{row['param_count']:<12,} "
            f"{row['hit_rate']*100:<11.2f}% "
            f"{row['keys_per_query']:<12.2f} "
            f"{threshold_str:<12} "
            f"{row['elapsed_time']:<10.1f}"
        )

    logger.info("-" * 68)

    # Find best configuration
    best_by_hit_rate = max(successful_results, key=lambda x: x['results']['metrics']['hit_rate'])
    best_by_keys = min(successful_results, key=lambda x: x['results']['metrics']['keys_per_query'])
    best_by_params = min(successful_results, key=lambda x: x['results']['metrics']['param_count'])

    logger.info("")
    logger.info("ANALYSIS:")
    logger.info(f"  Best hit rate: num_bins={best_by_hit_rate['num_bins']} ({best_by_hit_rate['results']['metrics']['hit_rate']*100:.2f}%)")
    logger.info(f"  Fewest keys/query: num_bins={best_by_keys['num_bins']} ({best_by_keys['results']['metrics']['keys_per_query']:.2f})")
    logger.info(f"  Smallest model: num_bins={best_by_params['num_bins']} ({best_by_params['results']['metrics']['param_count']:,} params)")

    # Check if all meet target hit rate
    target_hit_rate = 0.995
    meeting_target = [r for r in successful_results if r['results']['metrics']['hit_rate'] >= target_hit_rate]

    if meeting_target:
        # Among those meeting target, find smallest model
        best_meeting_target = min(meeting_target, key=lambda x: x['results']['metrics']['param_count'])
        logger.info("")
        logger.info(f"RECOMMENDATION (meeting >=99.5% hit rate with smallest model):")
        logger.info(f"  num_bins={best_meeting_target['num_bins']} achieves {best_meeting_target['results']['metrics']['hit_rate']*100:.2f}% hit rate")
        logger.info(f"  with {best_meeting_target['results']['metrics']['param_count']:,} params and {best_meeting_target['results']['metrics']['keys_per_query']:.2f} keys/query")
    else:
        logger.info("")
        logger.info("WARNING: No configuration achieved target hit rate of 99.5%")
        logger.info(f"Best hit rate achieved: num_bins={best_by_hit_rate['num_bins']} ({best_by_hit_rate['results']['metrics']['hit_rate']*100:.2f}%)")

    logger.info("=" * 80)

    # Save report to JSON
    report = {
        'timestamp': datetime.now().isoformat(),
        'fixed_params': {
            'num_kernels': 1,
            'mlp_hidden': 32,
            'kappa_init': 2.5
        },
        'target_hit_rate': target_hit_rate,
        'results': report_data,
        'analysis': {
            'best_hit_rate': {
                'num_bins': best_by_hit_rate['num_bins'],
                'hit_rate': best_by_hit_rate['results']['metrics']['hit_rate']
            },
            'fewest_keys': {
                'num_bins': best_by_keys['num_bins'],
                'keys_per_query': best_by_keys['results']['metrics']['keys_per_query']
            },
            'smallest_model': {
                'num_bins': best_by_params['num_bins'],
                'param_count': best_by_params['results']['metrics']['param_count']
            }
        },
        'failed_experiments': [r for r in all_results if r['status'] != 'success']
    }

    report_path = EXP_DIR / 'output' / 'num_bins_ablation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nReport saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run num_bins Ablation Study (fixed: num_kernels=1, mlp_hidden=32, kappa_init=2.5)"
    )
    parser.add_argument(
        '--num-bins-list',
        type=int,
        nargs='+',
        default=[64, 32, 16, 8],
        help='List of num_bins values to test (default: 64 32 16 8)'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        default=None,
        help='GPU indices to use (default: auto-detect all available)'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run experiments sequentially instead of in parallel'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("NUM_BINS ABLATION STUDY")
    logger.info("=" * 80)
    logger.info(f"num_bins values to test: {args.num_bins_list}")
    logger.info(f"Fixed parameters: num_kernels=1, mlp_hidden=32, kappa_init=2.5")

    # Get available GPUs
    if args.gpus:
        available_gpus = args.gpus
    else:
        available_gpus = get_available_gpus()

    if not available_gpus:
        logger.warning("No GPUs detected, will use CPU (GPU 0 placeholder)")
        available_gpus = [0]

    logger.info(f"Using GPUs: {available_gpus}")
    logger.info(f"Running {'sequentially' if args.sequential else 'in parallel'}")
    logger.info("=" * 80)

    all_results = []

    if args.sequential:
        # Sequential execution
        for i, num_bins in enumerate(args.num_bins_list):
            gpu_id = available_gpus[i % len(available_gpus)]
            result = run_single_experiment(num_bins, gpu_id, __name__)
            all_results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=len(args.num_bins_list)) as executor:
            futures = {}
            for i, num_bins in enumerate(args.num_bins_list):
                gpu_id = available_gpus[i % len(available_gpus)]
                future = executor.submit(
                    run_single_experiment,
                    num_bins,
                    gpu_id,
                    f"{__name__}.worker_{i}"
                )
                futures[future] = num_bins

            for future in as_completed(futures):
                num_bins = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Experiment for num_bins={num_bins} raised exception: {e}")
                    all_results.append({
                        'num_bins': num_bins,
                        'status': 'exception',
                        'error': str(e)
                    })

    # Generate report
    report = generate_report(all_results, logger)

    # Return exit code
    failed_count = sum(1 for r in all_results if r['status'] != 'success')
    if failed_count > 0:
        logger.warning(f"{failed_count} experiment(s) failed")
        return 1

    logger.info("\nAll experiments completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
