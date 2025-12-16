#!/usr/bin/env python3
"""
Experiment 001: Loss Function Sanity Check

验证 Module 2 Bin-based Sparse Attention 的 Loss Function 能否正确优化 bin 分布。
使用可学习参数直接代替神经网络输出，观察 loss 优化结果。

参考文档: docs/04_training_and_labels.md Section 5
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
    """确保输出目录存在"""
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)


# ============================================================================
# IMPL-002: SanityCheckModel and Mock Data Generation
# ============================================================================


class SanityCheckModel(nn.Module):
    """
    无输入的模拟模型，直接学习 bin 分布

    参考: docs/04_training_and_labels.md Section 5.1
    """

    def __init__(self, num_queries: int, num_keys: int, num_bins: int):
        super().__init__()
        # 可学习参数代替神经网络输出（logits）
        self.query_logits = nn.Parameter(torch.randn(num_queries, num_bins))
        self.key_logits = nn.Parameter(torch.randn(num_keys, num_bins))

    def forward(self):
        """
        返回 softmax 和 log_softmax 后的概率分布
        使用 log_softmax 解决数值稳定性问题
        """
        p = F.softmax(self.query_logits, dim=1)  # (num_queries, num_bins)
        r = F.softmax(self.key_logits, dim=1)    # (num_keys, num_bins)
        log_p = F.log_softmax(self.query_logits, dim=1)
        log_r = F.log_softmax(self.key_logits, dim=1)
        return p, r, log_p, log_r


def generate_mock_data(num_queries: int, num_keys: int):
    """
    生成模拟的 group 关系

    包含一对一和二对一关系：
    - 前半部分 query (50%): 一对一关系，每个 query 对应唯一的 key
    - 后半部分 query (50%): 二对一关系，每 2 个 query 共享 1 个 key

    Key 分配：
    - 一对一部分使用 key 0 ~ (half-1)
    - 二对一部分使用 key half ~ (half + half//2 - 1)
    - 剩余 25% 的 keys 未使用

    参考: docs/04_training_and_labels.md Section 5.2

    Returns:
        query_to_key: (num_queries,) 每个 query 的 argmax key
        group_masks: (num_queries, num_keys) True if same group
    """
    query_to_key = torch.zeros(num_queries, dtype=torch.long)
    group_masks = torch.zeros(num_queries, num_keys, dtype=torch.bool)

    half = num_queries // 2

    # 一对一关系：前半部分 query (0 ~ half-1) -> key (0 ~ half-1)
    for q in range(half):
        k = q
        query_to_key[q] = k
        group_masks[q, k] = True

    # 二对一关系：后半部分 query，每 2 个 query 共享 1 个 key
    # query (half ~ num_queries-1) -> key (half ~ half + half//2 - 1)
    for q in range(half, num_queries):
        # 每 2 个 query 共享 1 个 key
        k = half + (q - half) // 2
        query_to_key[q] = k
        group_masks[q, k] = True

    return query_to_key, group_masks


def get_mock_data_path(num_queries: int, num_keys: int) -> Path:
    """获取 mock data 文件路径"""
    return OUTPUT_DIR / "mock_data" / f"mock_data_q{num_queries}_k{num_keys}.pt"


def save_mock_data(query_to_key: torch.Tensor, group_masks: torch.Tensor, path: Path):
    """保存 mock data 到硬盘"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "query_to_key": query_to_key,
        "group_masks": group_masks,
        "num_queries": len(query_to_key),
        "num_keys": group_masks.shape[1],
    }, path)
    print(f"Saved mock data to: {path}")


def load_mock_data(path: Path):
    """从硬盘加载 mock data"""
    data = torch.load(path)
    print(f"Loaded mock data from: {path}")
    print(f"  num_queries: {data['num_queries']}, num_keys: {data['num_keys']}")
    return data["query_to_key"], data["group_masks"]


def get_or_create_mock_data(num_queries: int, num_keys: int, force_regenerate: bool = False):
    """
    获取或创建 mock data

    - 如果硬盘上已存在，加载并返回
    - 如果不存在或 force_regenerate=True，生成并保存

    确保所有实验使用相同的 ground truth
    """
    path = get_mock_data_path(num_queries, num_keys)

    if path.exists() and not force_regenerate:
        return load_mock_data(path)
    else:
        print(f"Generating new mock data (num_queries={num_queries}, num_keys={num_keys})...")
        query_to_key, group_masks = generate_mock_data(num_queries, num_keys)
        save_mock_data(query_to_key, group_masks, path)
        return query_to_key, group_masks


# ============================================================================
# IMPL-003: Loss Functions (Exp A, Exp B, Baseline)
# ============================================================================


def sanity_check_loss_exp_a(
    p: torch.Tensor,
    r: torch.Tensor,
    log_p: torch.Tensor,
    log_r: torch.Tensor,
    query_to_key: torch.Tensor,
    group_masks: torch.Tensor,
    lambda_repel: float = 1.0,
):
    """
    Experiment A: 双向交叉熵 + Linear Repel (归一化版本)

    参考: docs/04_training_and_labels.md Section 5.3

    Args:
        p: (num_queries, num_bins) - query soft bin distributions (softmax)
        r: (num_keys, num_bins) - key soft bin distributions (softmax)
        log_p: (num_queries, num_bins) - query log distributions (log_softmax)
        log_r: (num_keys, num_bins) - key log distributions (log_softmax)
        query_to_key: (num_queries,) - 每个 query 的 argmax key 索引
        group_masks: (num_queries, num_keys) - True if (q, k) in same group
        lambda_repel: repel loss 的权重

    Returns:
        total_loss, attract_loss, repel_loss
    """
    # 1. 拉近项：双向交叉熵
    r_matched = r[query_to_key]  # (num_queries, num_bins)
    log_r_matched = log_r[query_to_key]
    # 每个 query 的 attract loss
    attract_per_query = -(p * log_r_matched).sum(dim=1) - (r_matched * log_p).sum(dim=1)
    # 归一化：除以 num_pos
    num_pos_per_query = group_masks.float().sum(dim=1).clamp(min=1)
    attract = (attract_per_query / num_pos_per_query).sum()

    # 2. 推远项：Linear (p · r for non-group)
    s = torch.mm(p, r.T)  # (num_queries, num_keys)
    repel_matrix = s * (~group_masks).float()
    # 归一化：除以 num_neg
    num_neg_per_query = (~group_masks).float().sum(dim=1).clamp(min=1)
    repel_per_query = repel_matrix.sum(dim=1) / num_neg_per_query
    repel = repel_per_query.sum()

    total = attract + lambda_repel * repel
    return total, attract, repel


def sanity_check_loss_exp_b(
    p: torch.Tensor,
    r: torch.Tensor,
    log_p: torch.Tensor,
    log_r: torch.Tensor,
    query_to_key: torch.Tensor,
    group_masks: torch.Tensor,
    lambda_repel: float = 1.0,
):
    """
    Experiment B: 双向交叉熵 + Log Repel (归一化版本)

    参考: docs/04_training_and_labels.md Section 5.3

    Returns:
        total_loss, attract_loss, repel_loss
    """
    # 1. 拉近项：双向交叉熵
    r_matched = r[query_to_key]
    log_r_matched = log_r[query_to_key]
    # 每个 query 的 attract loss
    attract_per_query = -(p * log_r_matched).sum(dim=1) - (r_matched * log_p).sum(dim=1)
    # 归一化：除以 num_pos
    num_pos_per_query = group_masks.float().sum(dim=1).clamp(min=1)
    attract = (attract_per_query / num_pos_per_query).sum()

    # 2. 推远项：Log (p · log(r) for non-group)
    ce_matrix = torch.mm(p, log_r.T)  # (num_queries, num_keys)
    repel_matrix = ce_matrix * (~group_masks).float()
    # 归一化：除以 num_neg
    num_neg_per_query = (~group_masks).float().sum(dim=1).clamp(min=1)
    repel_per_query = repel_matrix.sum(dim=1) / num_neg_per_query
    repel = repel_per_query.sum()

    total = attract + lambda_repel * repel
    return total, attract, repel


def sanity_check_loss_baseline(
    p: torch.Tensor,
    r: torch.Tensor,
    log_p: torch.Tensor,
    log_r: torch.Tensor,
    query_to_key: torch.Tensor,
    group_masks: torch.Tensor,
):
    """
    Baseline: 仅双向交叉熵（无 repel 项，归一化版本）

    Returns:
        total_loss, attract_loss, repel_loss (repel_loss = 0)
    """
    r_matched = r[query_to_key]
    log_r_matched = log_r[query_to_key]
    # 每个 query 的 attract loss
    attract_per_query = -(p * log_r_matched).sum(dim=1) - (r_matched * log_p).sum(dim=1)
    # 归一化：除以 num_pos
    num_pos_per_query = group_masks.float().sum(dim=1).clamp(min=1)
    attract = (attract_per_query / num_pos_per_query).sum()
    return attract, attract, torch.tensor(0.0)


# ============================================================================
# IMPL-004: Evaluation Metrics
# ============================================================================


def compute_argmax_hit_rate(
    query_bins: torch.Tensor,
    key_bins: torch.Tensor,
    query_to_key: torch.Tensor,
) -> float:
    """
    计算 Argmax Hit Rate: Q 和其 argmax K 在同一 bin 的比例

    参考: docs/04_training_and_labels.md Section 5.5

    Args:
        query_bins: (num_queries,) - 每个 query 分配的 bin ID (argmax)
        key_bins: (num_keys,) - 每个 key 分配的 bin ID (argmax)
        query_to_key: (num_queries,) - 每个 query 的 argmax key

    Returns:
        hit_rate: float in [0, 1], 1.0 表示完美匹配
    """
    argmax_key_bins = key_bins[query_to_key]  # (num_queries,)
    hits = (query_bins == argmax_key_bins).float().sum()
    return (hits / len(query_bins)).item()


def compute_keys_per_query(
    query_bins: torch.Tensor,
    key_bins: torch.Tensor,
    num_bins: int,
) -> float:
    """
    计算每个 Query 参与 attention 的平均 Key 数量（即平均 bin 大小）

    Args:
        query_bins: (num_queries,) - 每个 query 分配的 bin ID
        key_bins: (num_keys,) - 每个 key 分配的 bin ID
        num_bins: bin 总数

    Returns:
        avg_keys_per_query: float
    """
    total_keys = 0
    for q_bin in query_bins:
        keys_in_bin = (key_bins == q_bin).sum().item()
        total_keys += keys_in_bin
    return total_keys / len(query_bins)


def compute_computation_reduction(keys_per_query: float, num_keys: int) -> float:
    """
    计算计算量减少比例: 1 - (keys_per_query / num_keys)
    """
    return 1.0 - (keys_per_query / num_keys)


def compute_bin_statistics(bin_assignments: torch.Tensor, num_bins: int) -> dict:
    """
    计算 bin 分布统计

    Returns:
        dict with: num_empty_bins, bin_utilization, bin_size_variance, bin_counts
    """
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
    r: torch.Tensor,
    query_to_key: torch.Tensor,
    num_bins: int,
) -> dict:
    """
    计算所有评估指标
    """
    # 获取 argmax bin 分配
    query_bins = p.argmax(dim=1)  # (num_queries,)
    key_bins = r.argmax(dim=1)    # (num_keys,)

    # 核心指标
    hit_rate = compute_argmax_hit_rate(query_bins, key_bins, query_to_key)
    keys_per_query = compute_keys_per_query(query_bins, key_bins, num_bins)
    comp_reduction = compute_computation_reduction(keys_per_query, len(key_bins))

    # Bin 统计
    key_bin_stats = compute_bin_statistics(key_bins, num_bins)
    query_bin_stats = compute_bin_statistics(query_bins, num_bins)

    return {
        "argmax_hit_rate": hit_rate,
        "keys_per_query": keys_per_query,
        "computation_reduction": comp_reduction,
        "key_bin_stats": key_bin_stats,
        "query_bin_stats": query_bin_stats,
    }


# ============================================================================
# IMPL-005: Training Loop
# ============================================================================


def train_experiment(
    model: SanityCheckModel,
    loss_fn,
    query_to_key: torch.Tensor,
    group_masks: torch.Tensor,
    epochs: int,
    lr: float,
    lambda_repel: float,
    exp_name: str,
    log_interval: int = 100,
) -> dict:
    """
    训练单个实验

    Args:
        model: SanityCheckModel 实例
        loss_fn: 损失函数 (exp_a, exp_b, or baseline)
        query_to_key: (num_queries,) 每个 query 的 argmax key
        group_masks: (num_queries, num_keys) True if same group
        epochs: 训练轮数
        lr: 学习率
        lambda_repel: repel loss 权重
        exp_name: 实验名称
        log_interval: 日志打印间隔

    Returns:
        history: dict with training history and final metrics
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "exp_name": exp_name,
        "epochs": [],
        "total_loss": [],
        "attract_loss": [],
        "repel_loss": [],
        "argmax_hit_rate": [],
        "keys_per_query": [],
    }

    num_bins = model.query_logits.shape[1]

    for epoch in range(epochs):
        optimizer.zero_grad()

        p, r, log_p, log_r = model()

        # 计算 loss
        if exp_name == "baseline":
            total, attract, repel = loss_fn(p, r, log_p, log_r, query_to_key, group_masks)
        else:
            total, attract, repel = loss_fn(
                p, r, log_p, log_r, query_to_key, group_masks, lambda_repel
            )

        total.backward()
        optimizer.step()

        # 记录 history
        history["epochs"].append(epoch)
        history["total_loss"].append(total.item())
        history["attract_loss"].append(attract.item())
        history["repel_loss"].append(repel.item() if isinstance(repel, torch.Tensor) else repel)

        # 计算 metrics（每 log_interval 次或最后一次）
        if epoch % log_interval == 0 or epoch == epochs - 1:
            with torch.no_grad():
                metrics = compute_all_metrics(p, r, query_to_key, num_bins)
            history["argmax_hit_rate"].append(metrics["argmax_hit_rate"])
            history["keys_per_query"].append(metrics["keys_per_query"])

            if epoch % log_interval == 0:
                print(
                    f"[{exp_name}] Epoch {epoch:4d} | "
                    f"Loss: {total.item():.4f} | "
                    f"Attract: {attract.item():.4f} | "
                    f"Repel: {repel.item() if isinstance(repel, torch.Tensor) else repel:.4f} | "
                    f"Hit Rate: {metrics['argmax_hit_rate']:.4f}"
                )

    # 最终 metrics
    with torch.no_grad():
        p, r, log_p, log_r = model()
        final_metrics = compute_all_metrics(p, r, query_to_key, num_bins)

    history["final_metrics"] = final_metrics

    return history


# ============================================================================
# IMPL-006: Visualization Functions
# ============================================================================


def plot_loss_curves(histories: dict, save_path: Path):
    """
    绘制所有实验的 loss 曲线对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total Loss
    ax = axes[0]
    for exp_name, hist in histories.items():
        ax.plot(hist["epochs"], hist["total_loss"], label=exp_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Attract Loss
    ax = axes[1]
    for exp_name, hist in histories.items():
        ax.plot(hist["epochs"], hist["attract_loss"], label=exp_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Attract Loss")
    ax.set_title("Attract Loss (Bidirectional CE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Repel Loss
    ax = axes[2]
    for exp_name, hist in histories.items():
        ax.plot(hist["epochs"], hist["repel_loss"], label=exp_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Repel Loss")
    ax.set_title("Repel Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_bin_distribution_heatmap(
    p: torch.Tensor,
    r: torch.Tensor,
    query_to_key: torch.Tensor,
    exp_name: str,
    save_path: Path,
):
    """
    绘制 bin 分布热力图，展示 Q 和 K 的 bin 概率分布
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Query bin distribution
    ax = axes[0]
    im = ax.imshow(p.detach().numpy(), aspect="auto", cmap="viridis")
    ax.set_xlabel("Bin ID")
    ax.set_ylabel("Query ID")
    ax.set_title(f"{exp_name}: Query Bin Probabilities")
    plt.colorbar(im, ax=ax)

    # Key bin distribution
    ax = axes[1]
    im = ax.imshow(r.detach().numpy(), aspect="auto", cmap="viridis")
    ax.set_xlabel("Bin ID")
    ax.set_ylabel("Key ID")
    ax.set_title(f"{exp_name}: Key Bin Probabilities")
    plt.colorbar(im, ax=ax)

    # Q-K alignment check: for each query, show if its argmax bin matches its key's bin
    ax = axes[2]
    query_bins = p.argmax(dim=1)  # (num_queries,)
    key_bins = r.argmax(dim=1)    # (num_keys,)
    matched_key_bins = key_bins[query_to_key]  # (num_queries,)

    # Create alignment matrix: 1 if Q and its argmax K in same bin, 0 otherwise
    alignment = (query_bins == matched_key_bins).float().unsqueeze(1)
    # Expand to show bin assignment
    alignment_viz = torch.zeros(len(query_bins), 3)
    alignment_viz[:, 0] = query_bins.float() / p.shape[1]  # Normalized query bin
    alignment_viz[:, 1] = matched_key_bins.float() / p.shape[1]  # Normalized key bin
    alignment_viz[:, 2] = alignment.squeeze()  # Match indicator

    im = ax.imshow(alignment_viz.detach().numpy(), aspect="auto", cmap="RdYlGn")
    ax.set_xlabel("(Q Bin, K Bin, Match)")
    ax.set_ylabel("Query ID")
    ax.set_title(f"{exp_name}: Q-K Bin Alignment")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Q Bin (norm)", "K Bin (norm)", "Match"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_sample_points(
    p: torch.Tensor,
    r: torch.Tensor,
    query_to_key: torch.Tensor,
    sample_indices: list,
    exp_name: str,
    save_path: Path,
    num_bins_to_show: int = 20,
):
    """
    绘制个别 Q-K pair 的详细 bin 分布

    展示用户虚拟的样本点和模型预测的 bin 概率分布
    """
    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, 2)

    # 只显示概率最高的 bins
    for idx, q_idx in enumerate(sample_indices):
        k_idx = query_to_key[q_idx].item()

        q_probs = p[q_idx].detach().numpy()
        k_probs = r[k_idx].detach().numpy()

        # 获取 top bins
        q_top_bins = q_probs.argsort()[-num_bins_to_show:][::-1]
        k_top_bins = k_probs.argsort()[-num_bins_to_show:][::-1]
        all_top_bins = sorted(set(q_top_bins) | set(k_top_bins))[:num_bins_to_show]

        # Query distribution
        ax = axes[idx, 0]
        ax.bar(range(len(all_top_bins)), [q_probs[b] for b in all_top_bins], alpha=0.7, label="Query")
        ax.set_xticks(range(len(all_top_bins)))
        ax.set_xticklabels([str(b) for b in all_top_bins], rotation=45)
        ax.set_ylabel("Probability")
        ax.set_title(f"Query {q_idx} Bin Distribution (argmax: {q_probs.argmax()})")
        ax.legend()

        # Key distribution
        ax = axes[idx, 1]
        ax.bar(range(len(all_top_bins)), [k_probs[b] for b in all_top_bins], alpha=0.7, color="orange", label="Key")
        ax.set_xticks(range(len(all_top_bins)))
        ax.set_xticklabels([str(b) for b in all_top_bins], rotation=45)
        ax.set_ylabel("Probability")
        ax.set_title(f"Key {k_idx} Bin Distribution (argmax: {k_probs.argmax()})")
        ax.legend()

    fig.suptitle(f"{exp_name}: Sample Q-K Pair Bin Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_bin_histogram(
    key_bins: torch.Tensor,
    query_bins: torch.Tensor,
    num_bins: int,
    exp_name: str,
    save_path: Path,
):
    """
    绘制 bin 占用直方图，用于检测 collapse
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Key bin histogram
    ax = axes[0]
    key_counts = torch.zeros(num_bins)
    for b in range(num_bins):
        key_counts[b] = (key_bins == b).sum()
    ax.bar(range(num_bins), key_counts.numpy(), alpha=0.7)
    ax.set_xlabel("Bin ID")
    ax.set_ylabel("Count")
    ax.set_title(f"{exp_name}: Key Bin Occupancy")
    ax.axhline(y=len(key_bins) / num_bins, color="r", linestyle="--", label="Uniform avg")
    ax.legend()

    # Query bin histogram
    ax = axes[1]
    query_counts = torch.zeros(num_bins)
    for b in range(num_bins):
        query_counts[b] = (query_bins == b).sum()
    ax.bar(range(num_bins), query_counts.numpy(), alpha=0.7, color="orange")
    ax.set_xlabel("Bin ID")
    ax.set_ylabel("Count")
    ax.set_title(f"{exp_name}: Query Bin Occupancy")
    ax.axhline(y=len(query_bins) / num_bins, color="r", linestyle="--", label="Uniform avg")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(results: dict, save_path: Path):
    """
    绘制指标对比柱状图
    """
    exp_names = list(results.keys())
    metrics = ["argmax_hit_rate", "keys_per_query", "computation_reduction", "bin_utilization"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = []
        for exp in exp_names:
            if metric == "bin_utilization":
                val = results[exp]["final_metrics"]["key_bin_stats"]["bin_utilization"]
            else:
                val = results[exp]["final_metrics"].get(metric, 0)
            values.append(val)

        bars = ax.bar(exp_names, values, color=["#2196F3", "#4CAF50", "#FF9800"])
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Experiment Results Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# IMPL-007: Main Script
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Loss Function Sanity Check Experiment")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--num_queries", type=int, default=64, help="Number of queries")
    parser.add_argument("--num_keys", type=int, default=32, help="Number of keys")
    parser.add_argument("--num_bins", type=int, default=128, help="Number of bins")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--lambda_repel", type=float, default=1.0, help="Repel loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--force_regenerate", action="store_true", help="Force regenerate mock data")
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Override args with config values
        args.num_queries = config.get("model", {}).get("num_queries", args.num_queries)
        args.num_keys = config.get("model", {}).get("num_keys", args.num_keys)
        args.num_bins = config.get("model", {}).get("num_bins", args.num_bins)
        args.epochs = config.get("training", {}).get("epochs", args.epochs)
        args.lr = config.get("training", {}).get("lr", args.lr)
        args.lambda_repel = config.get("training", {}).get("lambda_repel", args.lambda_repel)
        args.seed = config.get("experiment", {}).get("seed", args.seed)
        args.log_interval = config.get("training", {}).get("log_interval", args.log_interval)

    # Setup
    setup_output_dirs()
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Loss Function Sanity Check Experiment")
    print("=" * 60)
    print(f"num_queries: {args.num_queries}")
    print(f"num_keys: {args.num_keys}")
    print(f"num_bins: {args.num_bins}")
    print(f"epochs: {args.epochs}")
    print(f"lr: {args.lr}")
    print(f"lambda_repel: {args.lambda_repel}")
    print(f"seed: {args.seed}")
    print("=" * 60)

    # Load or create mock data (ensures all experiments use same ground truth)
    print("\nLoading/Creating mock data...")
    query_to_key, group_masks = get_or_create_mock_data(
        args.num_queries, args.num_keys, force_regenerate=args.force_regenerate
    )
    print(f"One-to-one queries: {args.num_queries // 2}")
    print(f"One-to-many queries: {args.num_queries - args.num_queries // 2} (all to key 0)")
    print(f"Mock data path: {get_mock_data_path(args.num_queries, args.num_keys)}")

    # Run experiments
    results = {}
    models = {}

    # Experiment A: Linear Repel
    print("\n" + "=" * 60)
    print("Running Experiment A: Linear Repel")
    print("=" * 60)
    model_a = SanityCheckModel(args.num_queries, args.num_keys, args.num_bins)
    results["exp_a"] = train_experiment(
        model_a, sanity_check_loss_exp_a, query_to_key, group_masks,
        args.epochs, args.lr, args.lambda_repel, "exp_a", args.log_interval
    )
    models["exp_a"] = model_a

    # Experiment B: Log Repel
    print("\n" + "=" * 60)
    print("Running Experiment B: Log Repel")
    print("=" * 60)
    torch.manual_seed(args.seed)  # Reset seed for fair comparison
    model_b = SanityCheckModel(args.num_queries, args.num_keys, args.num_bins)
    results["exp_b"] = train_experiment(
        model_b, sanity_check_loss_exp_b, query_to_key, group_masks,
        args.epochs, args.lr, args.lambda_repel, "exp_b", args.log_interval
    )
    models["exp_b"] = model_b

    # Baseline: No Repel
    print("\n" + "=" * 60)
    print("Running Baseline: No Repel")
    print("=" * 60)
    torch.manual_seed(args.seed)  # Reset seed for fair comparison
    model_baseline = SanityCheckModel(args.num_queries, args.num_keys, args.num_bins)
    results["baseline"] = train_experiment(
        model_baseline, sanity_check_loss_baseline, query_to_key, group_masks,
        args.epochs, args.lr, args.lambda_repel, "baseline", args.log_interval
    )
    models["baseline"] = model_baseline

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<15} {'Hit Rate':>12} {'Keys/Query':>12} {'Comp. Red.':>12} {'Bin Util.':>12}")
    print("-" * 60)
    for exp_name, res in results.items():
        fm = res["final_metrics"]
        hit_rate = fm["argmax_hit_rate"]
        keys_per_q = fm["keys_per_query"]
        comp_red = fm["computation_reduction"]
        bin_util = fm["key_bin_stats"]["bin_utilization"]
        print(f"{exp_name:<15} {hit_rate:>12.4f} {keys_per_q:>12.2f} {comp_red:>12.4f} {bin_util:>12.4f}")
    print("=" * 60)

    # Generate visualizations
    print("\nGenerating visualizations...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Loss curves
    plot_loss_curves(results, OUTPUT_DIR / "figures" / f"loss_curves_{timestamp}.png")

    # Metrics comparison
    plot_metrics_comparison(results, OUTPUT_DIR / "figures" / f"metrics_comparison_{timestamp}.png")

    # Per-experiment visualizations
    for exp_name, model in models.items():
        with torch.no_grad():
            p, r, _, _ = model()
            query_bins = p.argmax(dim=1)
            key_bins = r.argmax(dim=1)

        # Heatmap
        plot_bin_distribution_heatmap(
            p, r, query_to_key, exp_name,
            OUTPUT_DIR / "figures" / f"heatmap_{exp_name}_{timestamp}.png"
        )

        # Sample points (show 5 representative samples)
        sample_indices = [0, args.num_queries // 4, args.num_queries // 2,
                         3 * args.num_queries // 4, args.num_queries - 1]
        plot_sample_points(
            p, r, query_to_key, sample_indices, exp_name,
            OUTPUT_DIR / "figures" / f"sample_points_{exp_name}_{timestamp}.png"
        )

        # Bin histogram
        plot_bin_histogram(
            key_bins, query_bins, args.num_bins, exp_name,
            OUTPUT_DIR / "figures" / f"bin_histogram_{exp_name}_{timestamp}.png"
        )

    # Save results to JSON
    results_path = OUTPUT_DIR / "results" / f"metrics_{timestamp}.json"
    # Convert tensors to lists for JSON serialization
    results_json = {}
    for exp_name, res in results.items():
        results_json[exp_name] = {
            "exp_name": res["exp_name"],
            "final_loss": res["total_loss"][-1],
            "final_attract_loss": res["attract_loss"][-1],
            "final_repel_loss": res["repel_loss"][-1],
            "final_metrics": res["final_metrics"],
        }

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved results to: {results_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Figures saved to: {OUTPUT_DIR / 'figures'}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
