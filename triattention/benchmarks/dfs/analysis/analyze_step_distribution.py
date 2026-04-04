#!/usr/bin/env python3
"""
Analyze the step distribution in dfs_state_query_small.json dataset.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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

print(f"\n{'='*70}")
print(f"STEP STATISTICS")
print(f"{'='*70}")
print(f"  Min step: {min(steps)}")
print(f"  Max step: {max(steps)}")
print(f"  Mean step: {np.mean(steps):.2f}")
print(f"  Median step: {np.median(steps):.2f}")
print(f"  Std dev: {np.std(steps):.2f}")

print(f"\n{'='*70}")
print(f"STEP RATIO STATISTICS (query_step / total_steps)")
print(f"{'='*70}")
print(f"  Min ratio: {min(step_ratios):.3f}")
print(f"  Max ratio: {max(step_ratios):.3f}")
print(f"  Mean ratio: {np.mean(step_ratios):.3f}")
print(f"  Median ratio: {np.median(step_ratios):.3f}")
print(f"  Std dev: {np.std(step_ratios):.3f}")

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
step_bins = np.linspace(min(steps), max(steps), num_bins + 1)
bin_counts, _ = np.histogram(steps, bins=step_bins)

print(f"\n{'='*70}")
print(f"UNIFORMITY ANALYSIS ({num_bins} bins)")
print(f"{'='*70}")
print(f"  Bin edges: {[f'{x:.1f}' for x in step_bins]}")
print(f"  Bin counts: {bin_counts.tolist()}")
print(f"  Expected count per bin (uniform): {len(steps) / num_bins:.1f}")
print(f"  Std dev of bin counts: {np.std(bin_counts):.2f}")
print(f"  CV (coefficient of variation): {np.std(bin_counts) / np.mean(bin_counts):.3f}")

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
for gtype, step_list in steps_by_graph_type.items():
    print(f"  {gtype:8s}: mean={np.mean(step_list):5.2f}, std={np.std(step_list):5.2f}, n={len(step_list):2d}")

print(f"\nSteps by action:")
for action, step_list in steps_by_action.items():
    print(f"  {action:10s}: mean={np.mean(step_list):5.2f}, std={np.std(step_list):5.2f}, n={len(step_list):2d}")

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

# Visualize step distribution
print(f"\n{'='*70}")
print(f"GENERATING VISUALIZATIONS...")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram of steps
axes[0, 0].hist(steps, bins=range(min(steps), max(steps) + 2), edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_xlabel('Query Step', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Distribution of Query Steps', fontsize=12, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Scatter plot: step vs total_dfs_steps
axes[0, 1].scatter(total_dfs_steps, steps, alpha=0.6, s=50, color='coral')
axes[0, 1].plot([min(total_dfs_steps), max(total_dfs_steps)],
                [min(total_dfs_steps), max(total_dfs_steps)],
                'r--', alpha=0.3, label='y=x')
axes[0, 1].set_xlabel('Total DFS Steps', fontsize=11)
axes[0, 1].set_ylabel('Query Step', fontsize=11)
axes[0, 1].set_title('Query Step vs Total DFS Steps', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Histogram of step ratios
axes[1, 0].hist(step_ratios, bins=20, edgecolor='black', alpha=0.7, color='seagreen')
axes[1, 0].set_xlabel('Step Ratio (query_step / total_steps)', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('Distribution of Step Ratios', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Box plot of steps
box = axes[1, 1].boxplot([steps], vert=False, labels=['Query Steps'], patch_artist=True)
box['boxes'][0].set_facecolor('lightblue')
axes[1, 1].set_xlabel('Step Value', fontsize=11)
axes[1, 1].set_title('Box Plot of Query Steps', fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('step_distribution_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: step_distribution_analysis.png")
plt.close()

print(f"\n{'='*70}")
print(f"ANALYSIS COMPLETE!")
print(f"{'='*70}")
