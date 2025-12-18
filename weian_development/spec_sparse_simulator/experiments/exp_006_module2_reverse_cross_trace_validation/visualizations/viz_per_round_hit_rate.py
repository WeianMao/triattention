"""Per-round hit rate visualization across LLM decoding rounds.

This script creates a line plot showing hit rate (y-axis) vs round ID (x-axis)
for different TopK values during the LLM decoding process.

The visualization helps understand how hit rate evolves across decoding rounds:
- X-axis: Round ID (from round 1 to last round)
- Y-axis: Hit rate percentage (0-100%)
- Multiple lines for different K values (e.g., K=50, K=500, K=1000)

Usage:
    python visualizations/viz_per_round_hit_rate.py [--k-values 50 500 1000]

Output:
    output/visualizations/per_round_hit_rate.png
    output/visualizations/per_round_hit_rate_data.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml


def load_config(exp_dir: Path) -> dict:
    """Load experiment configuration."""
    config_path = exp_dir / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_trace_data(config: dict, exp_dir: Path) -> dict:
    """Load trace data from qk.pt file."""
    # Use test_trace_path for evaluation
    if 'test_trace_path' in config['data']:
        trace_path = exp_dir / config['data']['test_trace_path']
    else:
        trace_path = exp_dir / config['data']['trace_path']

    qk_data = torch.load(trace_path, map_location='cpu')

    # Load head sample file
    head_sample_path = exp_dir / config['data']['head_sample_file']
    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)

    # Select first head for POC
    layer, head = head_samples[0]

    # Extract Q, K for selected head
    Q = qk_data['q'][layer, head]
    K = qk_data['k'][layer, head]

    seq_len, head_dim = Q.shape

    # Compute attention matrix with causal mask
    scale = head_dim ** -0.5
    attention_logits = Q @ K.T * scale

    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attention_logits.masked_fill_(causal_mask, float('-inf'))
    attention = F.softmax(attention_logits, dim=-1)

    return {
        'Q': Q,
        'K': K,
        'attention': attention,
        'seq_len': seq_len,
        'head_dim': head_dim,
        'layer': layer,
        'head': head
    }


def load_model(config: dict, exp_dir: Path, device: torch.device):
    """Load trained model from checkpoint."""
    sys.path.insert(0, str(exp_dir))
    from model import create_model

    checkpoint_path = exp_dir / 'output' / 'checkpoints' / 'best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def compute_per_round_hit_rate(
    model,
    trace_data: dict,
    K_values: list[int],
    round_window: int,
    exclude_tail: int,
    device: torch.device
) -> dict:
    """Compute hit rate for each round separately.

    Returns:
        dict with structure:
        {
            K_value: {
                'round_ids': [1, 2, 3, ...],
                'hit_rates': [rate1, rate2, rate3, ...],
                'total_queries_per_round': [n1, n2, n3, ...],
                'hits_per_round': [h1, h2, h3, ...]
            }
        }
    """
    model.eval()

    Q = trace_data['Q'].to(device)
    K = trace_data['K'].to(device)
    attention = trace_data['attention'].to(device)
    seq_len = trace_data['seq_len']

    # Determine valid query range (exclude tail)
    valid_end = min(seq_len - exclude_tail, seq_len)

    # Calculate number of rounds
    num_rounds = (seq_len + round_window - 1) // round_window

    # Initialize per-round counters for each K value
    results = {
        k: {
            'round_ids': [],
            'hit_rates': [],
            'total_queries_per_round': [],
            'hits_per_round': [],
            'recent_hits_per_round': [],
            'bin_hits_per_round': []
        }
        for k in K_values
    }

    with torch.no_grad():
        # Iterate over rounds
        for round_idx in range(num_rounds):
            round_start = round_idx * round_window
            round_end = min(round_start + round_window, seq_len)

            # Skip first round (no historical keys)
            if round_start == 0:
                continue

            # Historical keys (< round_start)
            historical_keys = K[:round_start]
            num_historical = round_start

            # Compute reference angles for this round
            reference_angles = model.compute_reference_angles(round_start, round_window)

            # Forward pass: Key network on historical keys
            key_probs = model.forward_keys(historical_keys, reference_angles)

            # Per-round counters
            round_counters = {k: {'hits': 0, 'total': 0, 'recent_hits': 0, 'bin_hits': 0} for k in K_values}

            # Iterate over queries in this round
            for q_idx in range(round_start, min(round_end, valid_end)):
                # Get attention weights for all keys <= q_idx (causal)
                attn_weights = attention[q_idx, :q_idx + 1]

                # Find argmax key
                argmax_key = attn_weights.argmax().item()

                # Check if argmax is in recent keys (>= round_start)
                argmax_in_recent = argmax_key >= round_start

                # Get query vector
                query = Q[q_idx:q_idx + 1]

                # Detect empty bins (for masking)
                empty_bin_mask = key_probs.sum(dim=0) == 0

                # Forward pass: Query network
                query_bin_probs = model.forward_queries(query, reference_angles, empty_bin_mask)

                # Select bin (argmax of query bin probabilities)
                selected_bin = query_bin_probs.argmax(dim=-1).item()

                # Get key scores for selected bin
                bin_scores = key_probs[:, selected_bin]

                # For each K value, check if argmax key is hit
                for k_val in K_values:
                    round_counters[k_val]['total'] += 1

                    if argmax_in_recent:
                        round_counters[k_val]['hits'] += 1
                        round_counters[k_val]['recent_hits'] += 1
                    else:
                        actual_k = min(k_val, num_historical)
                        _, topk_indices = torch.topk(bin_scores, actual_k)

                        if argmax_key in topk_indices:
                            round_counters[k_val]['hits'] += 1
                            round_counters[k_val]['bin_hits'] += 1

            # Record per-round results
            for k_val in K_values:
                total = round_counters[k_val]['total']
                if total > 0:
                    hit_rate = round_counters[k_val]['hits'] / total * 100
                else:
                    hit_rate = 0.0

                results[k_val]['round_ids'].append(round_idx)
                results[k_val]['hit_rates'].append(hit_rate)
                results[k_val]['total_queries_per_round'].append(total)
                results[k_val]['hits_per_round'].append(round_counters[k_val]['hits'])
                results[k_val]['recent_hits_per_round'].append(round_counters[k_val]['recent_hits'])
                results[k_val]['bin_hits_per_round'].append(round_counters[k_val]['bin_hits'])

            # Progress print
            if (round_idx + 1) % 20 == 0:
                print(f"Processed round {round_idx + 1}/{num_rounds}")

    return results


def plot_hit_rate_curve(
    results: dict,
    output_path: Path,
    layer: int,
    head: int
) -> None:
    """Create line plot of hit rate vs round ID."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['o', 's', '^']

    for idx, (k_val, data) in enumerate(sorted(results.items())):
        # Filter out rounds with no queries (due to exclude_tail)
        valid_indices = [i for i, total in enumerate(data['total_queries_per_round']) if total > 0]
        round_ids = [data['round_ids'][i] for i in valid_indices]
        hit_rates = [data['hit_rates'][i] for i in valid_indices]

        ax.plot(
            round_ids,
            hit_rates,
            label=f'K={k_val}',
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            markersize=3,
            linewidth=1.5,
            alpha=0.8
        )

    ax.set_xlabel('Round ID', fontsize=12)
    ax.set_ylabel('Hit Rate (%)', fontsize=12)
    ax.set_title(
        f'Per-Round Hit Rate During LLM Decoding\n'
        f'Layer {layer}, Head {head}',
        fontsize=14
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)

    # Add average line annotation (only for valid rounds)
    for idx, (k_val, data) in enumerate(sorted(results.items())):
        valid_rates = [hr for hr, total in zip(data['hit_rates'], data['total_queries_per_round']) if total > 0]
        avg_rate = sum(valid_rates) / len(valid_rates) if valid_rates else 0
        ax.axhline(
            y=avg_rate,
            color=colors[idx % len(colors)],
            linestyle=':',
            alpha=0.5,
            linewidth=1
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot: {output_path}')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize per-round hit rate across LLM decoding'
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[50, 500, 1000],
        help='TopK values to evaluate (default: 50 500 1000)'
    )
    args = parser.parse_args()

    # Setup paths
    exp_dir = Path(__file__).parent.parent
    output_dir = exp_dir / 'output' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(exp_dir)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    print('Loading trace data...')
    trace_data = load_trace_data(config, exp_dir)
    print(f'Trace shape: Q={trace_data["Q"].shape}, K={trace_data["K"].shape}')
    print(f'Using Layer {trace_data["layer"]}, Head {trace_data["head"]}')

    # Load model
    print('Loading model...')
    model = load_model(config, exp_dir, device)

    # Compute per-round hit rates
    print('Computing per-round hit rates...')
    round_window = config['evaluation'].get('round_window', config['training']['round_window'])
    exclude_tail = config['training']['exclude_tail']

    results = compute_per_round_hit_rate(
        model, trace_data, args.k_values, round_window, exclude_tail, device
    )

    # Save raw data
    data_path = output_dir / 'per_round_hit_rate_data.json'
    with open(data_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved data: {data_path}')

    # Plot
    plot_path = output_dir / 'per_round_hit_rate.png'
    plot_hit_rate_curve(results, plot_path, trace_data['layer'], trace_data['head'])

    # Print summary (only for valid rounds with queries)
    print('\n=== Summary ===')
    for k_val, data in sorted(results.items()):
        valid_rates = [hr for hr, total in zip(data['hit_rates'], data['total_queries_per_round']) if total > 0]
        num_valid_rounds = len(valid_rates)
        avg_rate = sum(valid_rates) / num_valid_rounds if valid_rates else 0
        min_rate = min(valid_rates) if valid_rates else 0
        max_rate = max(valid_rates) if valid_rates else 0
        print(f'K={k_val}: Avg={avg_rate:.2f}%, Min={min_rate:.2f}%, Max={max_rate:.2f}% (across {num_valid_rounds} valid rounds)')


if __name__ == '__main__':
    main()
