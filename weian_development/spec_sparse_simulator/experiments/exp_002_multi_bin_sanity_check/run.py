#!/usr/bin/env python3
"""
Experiment 002: Multi-Bin Key Assignment Sanity Check

Verify Multi-Bin Key Assignment loss functions with softmax over keys (dim=0).
This allows keys to belong to multiple bins, unlike exp_001 where each key belongs to one bin.

Key Differences from exp_001:
- Softmax direction: dim=0 for keys (P[:, b] sums to 1 for each bin)
- TopK inference: Select TopK keys from predicted bin instead of all keys in bin
- Loss functions: Attraction Loss (NLL) and Bidirectional CE (with normalization)

Reference: docs/06_multi_bin_key_assignment.md
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# ============================================================================
# Directory Setup
# ============================================================================

EXP_DIR = Path(__file__).parent
OUTPUT_DIR = EXP_DIR / "output"


def setup_output_dirs():
    """Ensure output directories exist"""
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "mock_data").mkdir(parents=True, exist_ok=True)


# ============================================================================
# IMPL-002: SanityCheckModel with softmax over keys (dim=0)
# ============================================================================


class SanityCheckModel(nn.Module):
    """
    Multi-Bin Key Assignment sanity check model.

    Key difference from exp_001:
    - Key softmax is over keys (dim=0), not bins (dim=1)
    - This means P[:, b] sums to 1 (probability distribution over keys for each bin)
    - Allows keys to have high probability in multiple bins (multi-bin membership)

    Reference: docs/06_multi_bin_key_assignment.md Section 2
    """

    def __init__(self, num_queries: int, num_keys: int, num_bins: int):
        super().__init__()
        # Learnable parameters (logits before softmax)
        self.query_logits = nn.Parameter(torch.randn(num_queries, num_bins))
        self.key_logits = nn.Parameter(torch.randn(num_keys, num_bins))

    def forward(self):
        """
        Returns softmax and log_softmax distributions.

        Key difference from exp_001:
        - Query: softmax(dim=1) - each query selects one bin (p_q sums to 1 over bins)
        - Key: softmax(dim=0) - each bin has a distribution over keys (P[:, b] sums to 1 over keys)

        This allows keys to appear in multiple bins with different probabilities.
        """
        # Query: softmax over bins (dim=1) - same as exp_001
        p = F.softmax(self.query_logits, dim=1)  # (num_queries, num_bins)
        log_p = F.log_softmax(self.query_logits, dim=1)

        # Key: softmax over keys (dim=0) - DIFFERENT from exp_001
        # P[:, b] is the probability distribution over all keys for bin b
        P = F.softmax(self.key_logits, dim=0)  # (num_keys, num_bins)
        log_P = F.log_softmax(self.key_logits, dim=0)

        return p, P, log_p, log_P

    def get_topk_keys_for_bin(self, P: torch.Tensor, bin_id: int, K: int) -> torch.Tensor:
        """
        Select TopK keys for a given bin based on key probability scores.

        Reference: docs/06_multi_bin_key_assignment.md Section 2.3

        Args:
            P: (num_keys, num_bins) - key bin scores (softmax over keys for each bin)
            bin_id: bin index to select keys from
            K: number of top keys to select

        Returns:
            topk_indices: (K,) - indices of top K keys for this bin
        """
        # P[:, bin_id] is the probability distribution over all keys for this bin
        scores = P[:, bin_id]  # (num_keys,)
        topk_indices = torch.topk(scores, min(K, len(scores))).indices
        return topk_indices


# ============================================================================
# Mock Data Generation
# ============================================================================


def generate_mock_data(num_queries: int, num_keys: int):
    """
    Generate mock Q-K group relations.

    Same structure as exp_001:
    - First half: one-to-one (each query -> unique key)
    - Second half: two-to-one (2 queries share 1 key)

    Returns:
        query_to_key: (num_queries,) - argmax key for each query
        group_masks: (num_queries, num_keys) - True if (q, k) in same group
    """
    query_to_key = torch.zeros(num_queries, dtype=torch.long)
    group_masks = torch.zeros(num_queries, num_keys, dtype=torch.bool)

    half = num_queries // 2

    # One-to-one: query i -> key i (for i < half)
    for q in range(half):
        k = q
        query_to_key[q] = k
        group_masks[q, k] = True

    # Two-to-one: 2 queries share 1 key
    for q in range(half, num_queries):
        k = half + (q - half) // 2
        query_to_key[q] = k
        group_masks[q, k] = True

    return query_to_key, group_masks


def get_mock_data_path(num_queries: int, num_keys: int) -> Path:
    """Get mock data file path"""
    return OUTPUT_DIR / "mock_data" / f"mock_data_q{num_queries}_k{num_keys}.pt"


def save_mock_data(query_to_key: torch.Tensor, group_masks: torch.Tensor, path: Path):
    """Save mock data to disk"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "query_to_key": query_to_key,
        "group_masks": group_masks,
        "num_queries": len(query_to_key),
        "num_keys": group_masks.shape[1],
    }, path)
    print(f"Saved mock data to: {path}")


def load_mock_data(path: Path):
    """Load mock data from disk"""
    data = torch.load(path)
    print(f"Loaded mock data from: {path}")
    print(f"  num_queries: {data['num_queries']}, num_keys: {data['num_keys']}")
    return data["query_to_key"], data["group_masks"]


def get_or_create_mock_data(num_queries: int, num_keys: int, force_regenerate: bool = False):
    """Get or create mock data, ensuring all experiments use same ground truth"""
    path = get_mock_data_path(num_queries, num_keys)

    if path.exists() and not force_regenerate:
        return load_mock_data(path)
    else:
        print(f"Generating new mock data (num_queries={num_queries}, num_keys={num_keys})...")
        query_to_key, group_masks = generate_mock_data(num_queries, num_keys)
        save_mock_data(query_to_key, group_masks, path)
        return query_to_key, group_masks


# ============================================================================
# IMPL-003: Attraction Loss (NLL-based)
# ============================================================================


def attraction_loss_nll(
    p: torch.Tensor,
    P: torch.Tensor,
    query_to_key: torch.Tensor,
) -> torch.Tensor:
    """
    Attraction Loss using Negative Log Likelihood.

    Reference: docs/06_multi_bin_key_assignment.md Section 3.2

    Formula:
        match_prob[q] = sum_b p_q[b] * P[k(q), b]
        loss = -log(match_prob).mean()

    Intuition: Probability that query's predicted bin selects its argmax key.
    - p_q[b]: probability query q selects bin b
    - P[k, b]: probability bin b selects key k
    - match_prob: expected probability of selecting correct key

    Args:
        p: (num_queries, num_bins) - query bin distributions (softmax over bins)
        P: (num_keys, num_bins) - key bin scores (softmax over keys for each bin)
        query_to_key: (num_queries,) - argmax key index for each query

    Returns:
        loss: scalar tensor
    """
    # Get P[k(q), :] for each query's argmax key
    P_matched = P[query_to_key]  # (num_queries, num_bins)

    # Compute match probability: sum over bins of p_q[b] * P[k(q), b]
    match_prob = (p * P_matched).sum(dim=1)  # (num_queries,)

    # Negative log likelihood with epsilon for numerical stability
    # Reference: docs/06_multi_bin_key_assignment.md Section 3.3
    loss = -torch.log(match_prob + 1e-8).mean()

    return loss


# ============================================================================
# IMPL-004: Bidirectional CE Loss (alternative)
# ============================================================================


def bidirectional_ce_loss(
    p: torch.Tensor,
    P: torch.Tensor,
    log_P: torch.Tensor,
    log_p: torch.Tensor,
    query_to_key: torch.Tensor,
) -> torch.Tensor:
    """
    Bidirectional Cross-Entropy Loss with normalization.

    Reference: docs/06_multi_bin_key_assignment.md Section 3.4

    Key insight: P[k, :] does NOT sum to 1 (it's softmax over keys, not bins).
    We must normalize P[k, :] to get a valid probability distribution over bins.

    Formula:
        P_norm[k, :] = P[k, :] / sum_b P[k, b]  (normalize over bins)
        loss = CE(p_q, P_norm[k(q)]) + CE(P_norm[k(q)], p_q)

    Implementation uses log-space for numerical stability:
        log_P_norm = log_P - logsumexp(log_P, dim=1)

    Args:
        p: (num_queries, num_bins) - query bin distributions (softmax over bins)
        P: (num_keys, num_bins) - key bin scores (softmax over keys)
        log_P: (num_keys, num_bins) - log of key bin scores
        log_p: (num_queries, num_bins) - log of query bin distributions
        query_to_key: (num_queries,) - argmax key index for each query

    Returns:
        loss: scalar tensor
    """
    # Get P and log_P for each query's argmax key
    P_matched = P[query_to_key]  # (num_queries, num_bins)
    log_P_matched = log_P[query_to_key]  # (num_queries, num_bins)

    # Normalize P[k, :] over bins using log-space computation
    # P_norm[k, b] = P[k, b] / sum_b' P[k, b']
    # In log-space: log_P_norm = log_P - logsumexp(log_P, dim=1)
    log_sum = torch.logsumexp(log_P_matched, dim=1, keepdim=True)  # (num_queries, 1)
    log_P_norm = log_P_matched - log_sum  # (num_queries, num_bins)
    P_norm = torch.exp(log_P_norm)  # (num_queries, num_bins)

    # Bidirectional cross-entropy
    # CE(p_q, P_norm) = -sum_b p_q[b] * log(P_norm[b])
    ce1 = -(p * log_P_norm).sum(dim=1)  # (num_queries,)

    # CE(P_norm, p_q) = -sum_b P_norm[b] * log(p_q[b])
    ce2 = -(P_norm * log_p).sum(dim=1)  # (num_queries,)

    # Total loss: average bidirectional CE over all queries
    loss = (ce1 + ce2).mean()

    return loss


# ============================================================================
# IMPL-005: TopK-based Evaluation Metrics
# ============================================================================


def compute_topk_hit_rate(
    query_bins: torch.Tensor,
    P: torch.Tensor,
    query_to_key: torch.Tensor,
    K: int,
) -> float:
    """
    Compute TopK Hit Rate: fraction of queries whose argmax key is in TopK of predicted bin.

    Reference: docs/06_multi_bin_key_assignment.md Section 2.3

    Different from exp_001's Argmax Hit Rate:
    - exp_001: Check if Q and argmax K are in same bin (same argmax bin)
    - exp_002: Check if argmax K is in TopK keys of Q's predicted bin

    Args:
        query_bins: (num_queries,) - predicted bin ID for each query (argmax of p)
        P: (num_keys, num_bins) - key bin scores (softmax over keys)
        query_to_key: (num_queries,) - argmax key index for each query
        K: TopK parameter

    Returns:
        hit_rate: float in [0, 1]
    """
    num_queries = len(query_bins)
    hits = 0

    for q in range(num_queries):
        bin_q = query_bins[q].item()
        argmax_k = query_to_key[q].item()

        # Get TopK keys for this bin
        scores = P[:, bin_q]
        topk_indices = torch.topk(scores, min(K, len(scores))).indices

        # Check if argmax key is in TopK
        if argmax_k in topk_indices:
            hits += 1

    return hits / num_queries


def compute_keys_per_query(K: int) -> float:
    """
    Compute keys per query: fixed at K for TopK inference.

    Different from exp_001 where this was variable (dependent on bin size).
    In Multi-Bin Key Assignment, we always select exactly K keys.

    Args:
        K: TopK parameter

    Returns:
        K: keys per query (fixed)
    """
    return float(K)


def compute_computation_reduction(keys_per_query: float, num_keys: int) -> float:
    """Compute computation reduction: 1 - (keys_per_query / num_keys)"""
    return 1.0 - (keys_per_query / num_keys)


def compute_bin_statistics(bin_assignments: torch.Tensor, num_bins: int) -> dict:
    """Compute bin distribution statistics"""
    bin_counts = torch.zeros(num_bins)
    for b in range(num_bins):
        bin_counts[b] = (bin_assignments == b).sum()

    num_empty = (bin_counts == 0).sum().item()
    utilization = 1.0 - (num_empty / num_bins)
    variance = bin_counts.var().item()

    return {
        "num_empty_bins": int(num_empty),
        "bin_utilization": utilization,
        "bin_size_variance": variance,
        "bin_counts": bin_counts.tolist(),
    }


def compute_all_metrics(
    p: torch.Tensor,
    P: torch.Tensor,
    query_to_key: torch.Tensor,
    num_bins: int,
    K: int,
) -> dict:
    """
    Compute all evaluation metrics for Multi-Bin Key Assignment.

    Args:
        p: (num_queries, num_bins) - query bin distributions
        P: (num_keys, num_bins) - key bin scores (softmax over keys)
        query_to_key: (num_queries,) - argmax key for each query
        num_bins: number of bins
        K: TopK parameter

    Returns:
        dict with metrics
    """
    # Get bin assignments
    query_bins = p.argmax(dim=1)  # (num_queries,)

    # Core metrics
    hit_rate = compute_topk_hit_rate(query_bins, P, query_to_key, K)
    keys_per_query = compute_keys_per_query(K)
    comp_reduction = compute_computation_reduction(keys_per_query, P.shape[0])

    # Bin statistics for queries
    query_bin_stats = compute_bin_statistics(query_bins, num_bins)

    return {
        "topk_hit_rate": hit_rate,
        "keys_per_query": keys_per_query,
        "computation_reduction": comp_reduction,
        "query_bin_stats": query_bin_stats,
        "topk_k": K,
    }


# ============================================================================
# Training Loop
# ============================================================================


def train_experiment(
    model: SanityCheckModel,
    loss_fn,
    loss_name: str,
    query_to_key: torch.Tensor,
    epochs: int,
    lr: float,
    K: int,
    exp_name: str,
    log_interval: int = 100,
) -> dict:
    """
    Train single experiment.

    Args:
        model: SanityCheckModel instance
        loss_fn: loss function (attraction_loss_nll or bidirectional_ce_loss)
        loss_name: "attraction_nll" or "bidirectional_ce"
        query_to_key: (num_queries,) - argmax key for each query
        epochs: training epochs
        lr: learning rate
        K: TopK parameter for evaluation
        exp_name: experiment name for logging
        log_interval: logging interval

    Returns:
        history: dict with training history and final metrics
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "exp_name": exp_name,
        "loss_name": loss_name,
        "topk_k": K,
        "epochs": [],
        "loss": [],
        "topk_hit_rate": [],
        "keys_per_query": [],
    }

    num_bins = model.query_logits.shape[1]

    for epoch in range(epochs):
        optimizer.zero_grad()

        p, P, log_p, log_P = model()

        # Compute loss based on loss function type
        if loss_name == "attraction_nll":
            loss = loss_fn(p, P, query_to_key)
        else:  # bidirectional_ce
            loss = loss_fn(p, P, log_P, log_p, query_to_key)

        loss.backward()
        optimizer.step()

        # Record history
        history["epochs"].append(epoch)
        history["loss"].append(loss.item())

        # Compute metrics at intervals
        if epoch % log_interval == 0 or epoch == epochs - 1:
            with torch.no_grad():
                metrics = compute_all_metrics(p, P, query_to_key, num_bins, K)
            history["topk_hit_rate"].append(metrics["topk_hit_rate"])
            history["keys_per_query"].append(metrics["keys_per_query"])

            if epoch % log_interval == 0:
                print(
                    f"[{exp_name}] Epoch {epoch:4d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"TopK Hit Rate: {metrics['topk_hit_rate']:.4f}"
                )

    # Final metrics
    with torch.no_grad():
        p, P, log_p, log_P = model()
        final_metrics = compute_all_metrics(p, P, query_to_key, num_bins, K)

    history["final_metrics"] = final_metrics

    return history


# ============================================================================
# Visualization Functions
# ============================================================================


def plot_loss_curves(histories: dict, save_path: Path):
    """Plot loss curves for all experiments"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_name, hist in histories.items():
        ax.plot(hist["epochs"], hist["loss"], label=exp_name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_topk_hit_rate_comparison(histories: dict, save_path: Path):
    """Plot TopK Hit Rate comparison across K values and loss functions"""
    # Group by K value
    k_values = sorted(set(h["topk_k"] for h in histories.values()))
    loss_names = sorted(set(h["loss_name"] for h in histories.values()))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(k_values))
    width = 0.35

    for i, loss_name in enumerate(loss_names):
        hit_rates = []
        for K in k_values:
            exp_name = f"{loss_name}_K{K}"
            if exp_name in histories:
                hit_rates.append(histories[exp_name]["final_metrics"]["topk_hit_rate"])
            else:
                hit_rates.append(0)

        offset = width * (i - len(loss_names) / 2 + 0.5)
        bars = ax.bar([xi + offset for xi in x], hit_rates, width, label=loss_name)

        # Add value labels
        for bar, val in zip(bars, hit_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("TopK (K)")
    ax.set_ylabel("TopK Hit Rate")
    ax.set_title("TopK Hit Rate Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_bin_distribution_heatmap(
    p: torch.Tensor,
    P: torch.Tensor,
    query_to_key: torch.Tensor,
    K: int,
    exp_name: str,
    save_path: Path,
):
    """Plot bin distribution heatmaps"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Query bin distribution (same as exp_001)
    ax = axes[0]
    im = ax.imshow(p.detach().numpy(), aspect="auto", cmap="viridis")
    ax.set_xlabel("Bin ID")
    ax.set_ylabel("Query ID")
    ax.set_title(f"{exp_name}: Query Bin Probabilities")
    plt.colorbar(im, ax=ax)

    # Key bin scores (different interpretation from exp_001)
    ax = axes[1]
    im = ax.imshow(P.detach().numpy(), aspect="auto", cmap="viridis")
    ax.set_xlabel("Bin ID")
    ax.set_ylabel("Key ID")
    ax.set_title(f"{exp_name}: Key Bin Scores (softmax over keys)")
    plt.colorbar(im, ax=ax)

    # TopK Hit visualization
    ax = axes[2]
    query_bins = p.argmax(dim=1)
    num_queries = len(query_bins)

    hit_status = torch.zeros(num_queries)
    for q in range(num_queries):
        bin_q = query_bins[q].item()
        argmax_k = query_to_key[q].item()
        topk_indices = torch.topk(P[:, bin_q], min(K, P.shape[0])).indices
        hit_status[q] = 1.0 if argmax_k in topk_indices else 0.0

    ax.bar(range(min(100, num_queries)), hit_status[:100].numpy(), color=["green" if h else "red" for h in hit_status[:100]])
    ax.set_xlabel("Query ID (first 100)")
    ax.set_ylabel("Hit (1) / Miss (0)")
    ax.set_title(f"{exp_name}: TopK Hit Status (K={K})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(results: dict, save_path: Path):
    """Plot metrics comparison bar chart"""
    exp_names = list(results.keys())
    metrics = ["topk_hit_rate", "computation_reduction"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = []
        for exp in exp_names:
            val = results[exp]["final_metrics"].get(metric, 0)
            values.append(val)

        colors = plt.cm.tab10(range(len(exp_names)))
        bars = ax.bar(range(len(exp_names)), values, color=colors)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(exp_names, rotation=45, ha="right")

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Experiment Results Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_topk_comparison_table(results: dict):
    """Print comparison table for TopK sweep results"""
    # Group by loss function and K
    k_values = sorted(set(h["topk_k"] for h in results.values()))
    loss_names = sorted(set(h["loss_name"] for h in results.values()))

    print("\n" + "=" * 80)
    print("TopK SWEEP RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'K Value':<10} {'Loss Function':<20} {'Hit Rate':>12} {'Keys/Query':>12} {'Comp. Red.':>12}")
    print("-" * 80)

    for K in k_values:
        for loss_name in loss_names:
            exp_name = f"{loss_name}_K{K}"
            if exp_name in results:
                fm = results[exp_name]["final_metrics"]
                hit_rate = fm["topk_hit_rate"]
                keys_per_q = fm["keys_per_query"]
                comp_red = fm["computation_reduction"]
                print(f"{K:<10} {loss_name:<20} {hit_rate:>12.4f} {keys_per_q:>12.0f} {comp_red:>12.4f}")

    print("=" * 80)


# ============================================================================
# IMPL-006: Main Function with TopK Sweep
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Multi-Bin Key Assignment Sanity Check")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--num_queries", type=int, default=6000, help="Number of queries")
    parser.add_argument("--num_keys", type=int, default=6000, help="Number of keys")
    parser.add_argument("--num_bins", type=int, default=128, help="Number of bins")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--force_regenerate", action="store_true", help="Force regenerate mock data")
    parser.add_argument("--topk_k", type=int, nargs="+", default=None, help="TopK values to test")
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        args.num_queries = config.get("model", {}).get("num_queries", args.num_queries)
        args.num_keys = config.get("model", {}).get("num_keys", args.num_keys)
        args.num_bins = config.get("model", {}).get("num_bins", args.num_bins)
        args.epochs = config.get("training", {}).get("epochs", args.epochs)
        args.lr = config.get("training", {}).get("lr", args.lr)
        args.seed = config.get("experiment", {}).get("seed", args.seed)
        args.log_interval = config.get("training", {}).get("log_interval", args.log_interval)
        # TopK values from config
        if args.topk_k is None:
            args.topk_k = config.get("topk_k", [50, 500, 1000])

    # Default TopK values if not specified
    if args.topk_k is None:
        args.topk_k = [50, 500, 1000]

    # Setup
    setup_output_dirs()
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("Multi-Bin Key Assignment Sanity Check (exp_002)")
    print("=" * 70)
    print(f"num_queries: {args.num_queries}")
    print(f"num_keys: {args.num_keys}")
    print(f"num_bins: {args.num_bins}")
    print(f"epochs: {args.epochs}")
    print(f"lr: {args.lr}")
    print(f"seed: {args.seed}")
    print(f"topk_k values: {args.topk_k}")
    print("=" * 70)

    # Load or create mock data
    print("\nLoading/Creating mock data...")
    query_to_key, group_masks = get_or_create_mock_data(
        args.num_queries, args.num_keys, force_regenerate=args.force_regenerate
    )

    # Run experiments for each TopK value
    results = {}
    models = {}

    for K in args.topk_k:
        print(f"\n{'=' * 70}")
        print(f"TopK = {K}")
        print("=" * 70)

        # Experiment 1: Attraction Loss (NLL)
        print(f"\n--- Attraction Loss (NLL) with K={K} ---")
        torch.manual_seed(args.seed)
        model_nll = SanityCheckModel(args.num_queries, args.num_keys, args.num_bins)
        exp_name_nll = f"attraction_nll_K{K}"
        results[exp_name_nll] = train_experiment(
            model_nll, attraction_loss_nll, "attraction_nll",
            query_to_key, args.epochs, args.lr, K, exp_name_nll, args.log_interval
        )
        models[exp_name_nll] = model_nll

        # Experiment 2: Bidirectional CE Loss
        print(f"\n--- Bidirectional CE Loss with K={K} ---")
        torch.manual_seed(args.seed)
        model_ce = SanityCheckModel(args.num_queries, args.num_keys, args.num_bins)
        exp_name_ce = f"bidirectional_ce_K{K}"
        results[exp_name_ce] = train_experiment(
            model_ce, bidirectional_ce_loss, "bidirectional_ce",
            query_to_key, args.epochs, args.lr, K, exp_name_ce, args.log_interval
        )
        models[exp_name_ce] = model_ce

    # Print comparison table
    print_topk_comparison_table(results)

    # Generate visualizations
    print("\nGenerating visualizations...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Loss curves
    plot_loss_curves(results, OUTPUT_DIR / "figures" / f"loss_curves_{timestamp}.png")

    # TopK Hit Rate comparison
    plot_topk_hit_rate_comparison(results, OUTPUT_DIR / "figures" / f"topk_hit_rate_comparison_{timestamp}.png")

    # Metrics comparison
    plot_metrics_comparison(results, OUTPUT_DIR / "figures" / f"metrics_comparison_{timestamp}.png")

    # Per-experiment heatmaps (sample for first and last K)
    for K in [args.topk_k[0], args.topk_k[-1]]:
        for loss_name in ["attraction_nll", "bidirectional_ce"]:
            exp_name = f"{loss_name}_K{K}"
            if exp_name in models:
                model = models[exp_name]
                with torch.no_grad():
                    p, P, _, _ = model()
                plot_bin_distribution_heatmap(
                    p, P, query_to_key, K, exp_name,
                    OUTPUT_DIR / "figures" / f"heatmap_{exp_name}_{timestamp}.png"
                )

    # Save results to JSON
    results_path = OUTPUT_DIR / "results" / f"metrics_{timestamp}.json"
    results_json = {}
    for exp_name, res in results.items():
        results_json[exp_name] = {
            "exp_name": res["exp_name"],
            "loss_name": res["loss_name"],
            "topk_k": res["topk_k"],
            "final_loss": res["loss"][-1],
            "final_metrics": res["final_metrics"],
        }

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved results to: {results_path}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Figures saved to: {OUTPUT_DIR / 'figures'}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
