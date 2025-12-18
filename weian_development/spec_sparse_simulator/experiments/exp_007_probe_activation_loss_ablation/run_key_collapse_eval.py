#!/usr/bin/env python3
"""
Evaluate collapse metrics for key experiments only.

Key experiments:
- Baseline (lambda_0_0 in ablation/)
- Best loss1 (lambda_0_1 in ablation/)
- Best loss2 (balance_lambda_0_005 in ablation_balance/)
"""

import json
import subprocess
import sys
from pathlib import Path


KEY_EXPERIMENTS = [
    # (name, checkpoint_path, output_dir)
    ("Baseline (no loss)",
     "output/ablation/lambda_0_0/checkpoints/final_model.pt",
     "output/ablation/lambda_0_0"),
    ("Loss1 λ=0.1",
     "output/ablation/lambda_0_1/checkpoints/final_model.pt",
     "output/ablation/lambda_0_1"),
    ("Loss2 λ=0.005 (best)",
     "output/ablation_balance/balance_lambda_0_005/checkpoints/final_model.pt",
     "output/ablation_balance/balance_lambda_0_005"),
]


def run_eval(name, checkpoint, output_dir):
    base_dir = Path(__file__).parent
    checkpoint_path = base_dir / checkpoint

    if not checkpoint_path.exists():
        print(f"  [SKIP] Checkpoint not found: {checkpoint_path}")
        return None

    cmd = [
        sys.executable,
        str(base_dir / "evaluate.py"),
        "--checkpoint", str(checkpoint_path),
    ]

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Checkpoint: {checkpoint}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    # Read results
    results_path = base_dir / output_dir / "evaluation_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def main():
    results = []

    for name, checkpoint, output_dir in KEY_EXPERIMENTS:
        data = run_eval(name, checkpoint, output_dir)
        if data and "collapse_metrics" in data:
            cm = data["collapse_metrics"]
            soft = cm.get("soft_metrics", {})
            hard = cm.get("hard_metrics", {})
            hit_rate = data.get("hit_rates", {}).get("1000", {}).get("hit_rate", "N/A")

            results.append({
                "name": name,
                "active_bins_soft": soft.get("active_bin_count", "N/A"),
                "active_bins_hard": hard.get("active_bin_count", "N/A"),
                "entropy": soft.get("entropy_normalized", "N/A"),
                "gini": soft.get("gini_coefficient", "N/A"),
                "hit_rate_1000": hit_rate,
            })

    # Print comparison table
    print(f"\n{'='*80}")
    print("COLLAPSE METRICS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Experiment':<30} {'Active(S)':<12} {'Active(H)':<12} {'Entropy':<12} {'Gini':<12} {'HitRate':<10}")
    print("-" * 88)

    for r in results:
        ent = f"{r['entropy']:.4f}" if isinstance(r['entropy'], (int, float)) else str(r['entropy'])
        gini = f"{r['gini']:.4f}" if isinstance(r['gini'], (int, float)) else str(r['gini'])
        hr = f"{r['hit_rate_1000']:.2f}" if isinstance(r['hit_rate_1000'], (int, float)) else str(r['hit_rate_1000'])

        print(f"{r['name']:<30} {str(r['active_bins_soft']):<12} {str(r['active_bins_hard']):<12} {ent:<12} {gini:<12} {hr:<10}")

    print(f"\nNote: Active(S)=soft metrics (probability), Active(H)=hard metrics (argmax)")
    print(f"      Higher entropy = more uniform, Lower gini = less concentrated")
    print(f"      Ideal: Active=128, Entropy=1.0, Gini=0.0")


if __name__ == "__main__":
    main()
