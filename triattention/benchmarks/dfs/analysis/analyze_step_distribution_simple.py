#!/usr/bin/env python3
"""
Analyze the step distribution in dfs_state_query_small.json dataset.
Simplified version without matplotlib dependency.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

# Load the dataset
dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "dfs_state_query_small.json"
with open(dataset_path, 'r') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# Extract step information
steps = [item['steps'] for item in data]
total_dfs_steps = [item['metadata']['total_dfs_steps'] for item in data]

# Calculate step ratios (query step / total steps)
step_ratios = [s / t for s, t in zip(steps, total_dfs_steps)]

# Helper functions for statistics
def mean(values):
    return sum(values) / len(values)

def median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n//2-1] + sorted_values[n//2]) / 2
    else:
        return sorted_values[n//2]

def std_dev(values):
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return variance ** 0.5

print(f"\n{'='*70}")
print(f"STEP STATISTICS")
print(f"{'='*70}")
print(f"  Min step: {min(steps)}")
print(f"  Max step: {max(steps)}")
print(f"  Mean step: {mean(steps):.2f}")
print(f"  Median step: {median(steps):.2f}")
print(f"  Std dev: {std_dev(steps):.2f}")

print(f"\n{'='*70}")
print(f"STEP RATIO STATISTICS (query_step / total_steps)")
print(f"{'='*70}")
print(f"  Min ratio: {min(step_ratios):.3f}")
print(f"  Max ratio: {max(step_ratios):.3f}")
print(f"  Mean ratio: {mean(step_ratios):.3f}")
print(f"  Median ratio: {median(step_ratios):.3f}")
print(f"  Std dev: {std_dev(step_ratios):.3f}")

# Count frequency of each step value
step_counts = Counter(steps)
print(f"\n{'='*70}")
print(f"STEP FREQUENCY")
print(f"{'='*70}")
for step in sorted(step_counts.keys()):
    bar = '█' * step_counts[step]
    print(f"  Step {step:2d}: {step_counts[step]:2d} occurrences  {bar}")

# Uniformity analysis
num_bins = 5
step_min, step_max = min(steps), max(steps)
bin_width = (step_max - step_min) / num_bins
bin_edges = [step_min + i * bin_width for i in range(num_bins + 1)]
bin_counts = [0] * num_bins

for s in steps:
    bin_idx = int((s - step_min) / bin_width)
    if bin_idx >= num_bins:
        bin_idx = num_bins - 1
    bin_counts[bin_idx] += 1

print(f"\n{'='*70}")
print(f"UNIFORMITY ANALYSIS ({num_bins} bins)")
print(f"{'='*70}")
print(f"  Bin edges: {[f'{x:.1f}' for x in bin_edges]}")
print(f"  Bin counts: {bin_counts}")
print(f"  Expected count per bin (uniform): {len(steps) / num_bins:.1f}")
print(f"  Std dev of bin counts: {std_dev(bin_counts):.2f}")
print(f"  CV (coefficient of variation): {std_dev(bin_counts) / mean(bin_counts):.3f}")

# Check step distribution by graph type and action
graph_types = [item['metadata']['graph_type'] for item in data]
actions = [item['metadata']['action'] for item in data]

print(f"\n{'='*70}")
print(f"BREAKDOWN BY METADATA")
print(f"{'='*70}")
print(f"\nGraph types: {dict(Counter(graph_types))}")
print(f"Actions: {dict(Counter(actions))}")

# Group steps by graph type
steps_by_graph_type = defaultdict(list)
steps_by_action = defaultdict(list)

for item in data:
    steps_by_graph_type[item['metadata']['graph_type']].append(item['steps'])
    steps_by_action[item['metadata']['action']].append(item['steps'])

print(f"\nSteps by graph type:")
for gtype, step_list in sorted(steps_by_graph_type.items()):
    print(f"  {gtype:8s}: mean={mean(step_list):5.2f}, std={std_dev(step_list):5.2f}, n={len(step_list):2d}")

print(f"\nSteps by action:")
for action, step_list in sorted(steps_by_action.items()):
    print(f"  {action:10s}: mean={mean(step_list):5.2f}, std={std_dev(step_list):5.2f}, n={len(step_list):2d}")

# Show detailed list sorted by steps
sorted_data = sorted(data, key=lambda x: x['steps'])

print(f"\n{'='*70}")
print(f"DETAILED LIST (sorted by steps)")
print(f"{'='*70}")
print(f"{'ID':<4} {'Step':<5} {'Total':<6} {'Ratio':<6} {'GraphType':<8} {'Action':<10} {'Nodes':<6} {'Edges':<6}")
print("-" * 70)
for item in sorted_data:
    ratio = item['steps'] / item['metadata']['total_dfs_steps']
    print(f"{item['id']:<4} {item['steps']:<5} {item['metadata']['total_dfs_steps']:<6} {ratio:<6.3f} "
          f"{item['metadata']['graph_type']:<8} {item['metadata']['action']:<10} "
          f"{item['metadata']['graph_nodes']:<6} {item['metadata']['graph_edges']:<6}")

print(f"\n{'='*70}")
print(f"ANALYSIS COMPLETE!")
print(f"{'='*70}")

# Summary assessment
print(f"\n{'='*70}")
print(f"UNIFORMITY ASSESSMENT")
print(f"{'='*70}")

# Calculate gaps between consecutive steps (sorted)
sorted_steps = sorted(steps)
gaps = [sorted_steps[i+1] - sorted_steps[i] for i in range(len(sorted_steps)-1)]
max_gap = max(gaps) if gaps else 0
avg_gap = mean(gaps) if gaps else 0

print(f"\nStep coverage:")
print(f"  Range: {min(steps)} - {max(steps)} ({max(steps) - min(steps) + 1} possible values)")
print(f"  Unique values: {len(set(steps))}")
print(f"  Max gap between consecutive steps: {max_gap}")
print(f"  Avg gap between consecutive steps: {avg_gap:.2f}")
print(f"\nDistribution quality:")
print(f"  CV of step frequencies: {std_dev([step_counts[s] for s in sorted(step_counts.keys())]) / mean([step_counts[s] for s in sorted(step_counts.keys())]):.3f}")
print(f"  (Lower CV = more uniform; CV < 0.5 is good)")

if std_dev(bin_counts) / mean(bin_counts) < 0.5:
    print(f"\n✓ Step distribution appears reasonably uniform")
else:
    print(f"\n⚠ Step distribution shows significant non-uniformity")
    print(f"  Consider rebalancing to achieve more even coverage")
