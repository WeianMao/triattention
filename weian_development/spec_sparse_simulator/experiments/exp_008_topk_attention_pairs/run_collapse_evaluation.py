#!/usr/bin/env python3
"""
Batch Collapse Evaluation Script

Evaluates bin collapse metrics across all trained models to compare
the effectiveness of different loss configurations.
"""

import json
import subprocess
import sys
from pathlib import Path


def find_all_models():
    """Find all trained models in the output directory."""
    output_dir = Path(__file__).parent / "output"
    models = []

    # Find all final_model.pt files
    for model_path in output_dir.rglob("final_model.pt"):
        checkpoint_dir = model_path.parent
        exp_dir = checkpoint_dir.parent

        # Parse experiment info
        exp_name = exp_dir.name
        ablation_type = exp_dir.parent.name

        models.append({
            "checkpoint_path": str(model_path),
            "exp_dir": str(exp_dir),
            "exp_name": exp_name,
            "ablation_type": ablation_type,
        })

    return models


def run_evaluation(checkpoint_path, exp_dir):
    """Run evaluation on a single model."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "evaluate.py"),
        "--checkpoint", checkpoint_path,
    ]

    print(f"\nEvaluating: {checkpoint_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return None

    # Read results
    results_path = Path(exp_dir) / "evaluation_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def main():
    models = find_all_models()
    print(f"Found {len(models)} trained models")

    # Group models by ablation type
    by_type = {}
    for m in models:
        ablation_type = m["ablation_type"]
        if ablation_type not in by_type:
            by_type[ablation_type] = []
        by_type[ablation_type].append(m)

    results_summary = []

    for ablation_type, models_in_type in sorted(by_type.items()):
        print(f"\n{'='*60}")
        print(f"Ablation Type: {ablation_type}")
        print(f"{'='*60}")

        for m in sorted(models_in_type, key=lambda x: x["exp_name"]):
            result = run_evaluation(m["checkpoint_path"], m["exp_dir"])

            if result and "collapse_metrics" in result:
                cm = result["collapse_metrics"]
                results_summary.append({
                    "ablation_type": ablation_type,
                    "exp_name": m["exp_name"],
                    "active_bins_soft": cm.get("active_bin_count", "N/A"),
                    "active_bins_hard": cm.get("active_bin_count_hard", "N/A"),
                    "entropy": cm.get("entropy_normalized", "N/A"),
                    "gini": cm.get("gini_coefficient", "N/A"),
                    "hit_rate_1000": result.get("hit_rates", {}).get("1000", {}).get("hit_rate", "N/A"),
                })

                print(f"\n  {m['exp_name']}:")
                print(f"    Active bins (soft): {cm.get('active_bin_count', 'N/A')}/128")
                print(f"    Active bins (hard): {cm.get('active_bin_count_hard', 'N/A')}/128")
                print(f"    Entropy: {cm.get('entropy_normalized', 'N/A'):.4f}" if isinstance(cm.get('entropy_normalized'), (int, float)) else f"    Entropy: {cm.get('entropy_normalized', 'N/A')}")
                print(f"    Gini: {cm.get('gini_coefficient', 'N/A'):.4f}" if isinstance(cm.get('gini_coefficient'), (int, float)) else f"    Gini: {cm.get('gini_coefficient', 'N/A')}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Ablation Type':<25} {'Exp Name':<20} {'Active(H)':<10} {'Entropy':<10} {'Gini':<10} {'HitRate':<10}")
    print("-" * 85)

    for r in results_summary:
        entropy = f"{r['entropy']:.4f}" if isinstance(r['entropy'], (int, float)) else str(r['entropy'])
        gini = f"{r['gini']:.4f}" if isinstance(r['gini'], (int, float)) else str(r['gini'])
        hit_rate = f"{r['hit_rate_1000']:.2f}" if isinstance(r['hit_rate_1000'], (int, float)) else str(r['hit_rate_1000'])

        print(f"{r['ablation_type']:<25} {r['exp_name']:<20} {str(r['active_bins_hard']):<10} {entropy:<10} {gini:<10} {hit_rate:<10}")

    # Save summary
    summary_path = Path(__file__).parent / "output" / "collapse_comparison.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
