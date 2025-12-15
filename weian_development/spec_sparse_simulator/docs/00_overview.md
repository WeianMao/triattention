# Neural Sparse Attention - Overview

## 项目目标

基于神经网络的两阶段 Sparse Attention 算法，用于长序列 LLM 推理中的稀疏注意力计算。

---

## 背景：现有工作与动机

### 现有代码：`attention_pruning_case_study_hybrid_rounds_xtrace.py`

这是一个 **KV Cache 压缩** 的模拟器，用于评估基于频率的 Key 剪枝算法。

**核心机制**：
- 每 `round_window`（默认 128）次解码执行一次 cache maintenance
- 使用 **频率统计** 给每个 Key 打分（基于 RoPE 频段的 amplitude、phase 等特征）
- 保留得分最高的 `max_keys` 个 Key，丢弃其余

**关键函数**：
| 函数 | 功能 |
|------|------|
| `simulate_round_pruning()` | 模拟 round-based 的 Key 剪枝过程 |
| `score_keys_for_round()` | 基于频率统计计算 Key 得分 |
| `compute_pooled_attention()` | 计算和可视化 attention pattern |
| `invert_rope()` / `to_complex_pairs()` | RoPE 相关的向量处理工具 |

**局限性**：
- 使用手工设计的频率特征打分，缺乏学习能力
- 只能做 KV Cache 压缩（Drop KV），无法做 Sparse Attention

### 本项目的目标：Neural Sparse Attention

从 **KV Cache 压缩** 扩展到 **Sparse Attention**：

| 对比项 | KV Cache 压缩 (现有) | Sparse Attention (本项目) |
|--------|---------------------|--------------------------|
| **机制** | 丢弃 Key，减少显存 | 保留 Key，但只计算部分 attention |
| **max_keys** | 有硬限制 | **不存在**此限制 |
| **打分方式** | 手工频率特征 | **神经网络学习** |
| **计算节省** | 减少 KV 数量 | 减少 attention 计算量 |

### 复用与新增

**复用现有代码**：
- Round-based 框架（每 128 次解码执行一次 maintenance）
- `compute_pooled_attention()` 用于评估
- RoPE 相关工具函数

**需要新增**：
- 神经网络替换手工打分
- Bin-based sparse attention 机制
- 训练流程

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    每 128 次解码的 Round 开头                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────┐      ┌─────────────────────────────┐  │
│   │   Module 1          │      │   Module 2                  │  │
│   │   Key Pruning       │ ──→  │   Key Binning               │  │
│   │   (Drop KV)         │      │   (给剩余 K 分 bin)           │  │
│   └─────────────────────┘      └─────────────────────────────┘  │
│                                            │                    │
│                                            ▼                    │
│                                   ┌─────────────────┐           │
│                                   │   bin_index     │           │
│                                   │   {bin → [K]}   │           │
│                                   └─────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    每次解码 (128 次内)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   新 Query Q ──→ ┌─────────────────────┐                        │
│                  │   Query Routing     │                        │
│                  │   (Q 分 bin)         │                        │
│                  └──────────┬──────────┘                        │
│                             │                                   │
│                             ▼                                   │
│                  ┌─────────────────────┐                        │
│                  │   Sparse Attention  │                        │
│                  │   Q × K_{same_bin}  │                        │
│                  └─────────────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 两个模块概述

### Module 1: Key Pruning (Drop KV)

| 属性 | 描述 |
|------|------|
| **目标** | 预测并丢弃未来不会被 attend 的 Key |
| **执行时机** | 每 128 次解码的 round 开头 |
| **神经网络** | 较重（Kernel 层 + 1 层 MLP + Sigmoid） |
| **输出** | 二分类：保留 / 丢弃 |

详细设计见：[01_module1_key_pruning.md](./01_module1_key_pruning.md)

### Module 2: Bin-based Sparse Attention

| 属性 | 描述 |
|------|------|
| **目标** | 将 K 和 Q 分到 bin，Q 只与同 bin 的 K 做 attention |
| **执行时机** | Key Binning: round 开头；Query Routing: 每次解码 |
| **神经网络** | 轻量（Kernel 层 + Softmax，无 MLP） |
| **输出** | 多分类：128 个 bin |

详细设计见：[02_module2_bin_sparse_attention.md](./02_module2_bin_sparse_attention.md)

---

## 神经网络共享设计

两个模块的神经网络共享相同的输入编码方式（Kernel-based encoding），详见：
- [03_neural_network_architecture.md](./03_neural_network_architecture.md)

---

## Multi-head 处理策略

- **每个 Query head 独立**：每个 Query head 学习两个独立的神经网络（K 网络 + Q 网络）
- **GQA 情况**：多个 Query head 共享一个 KV head 时，同一个 K 会被多个不同的神经网络处理
- 这意味着同一个 K 在不同 Query head 视角下可能被分到不同的 bin

---

## 实验计划

两个模块**独立验证**，分别确认有效性后再组合：

| Phase | 内容 | 状态 |
|-------|------|------|
| A | Module 1 独立验证 (Key Pruning) | TODO |
| B | Module 2 独立验证 (Bin-based Sparse) | TODO |
| C | Module 1 + Module 2 联合验证 | TODO |

---

## 文档索引

| 文件 | 内容 |
|------|------|
| [00_overview.md](./00_overview.md) | 本文件，整体架构概览 |
| [01_module1_key_pruning.md](./01_module1_key_pruning.md) | Module 1: Key Pruning 详细设计 |
| [02_module2_bin_sparse_attention.md](./02_module2_bin_sparse_attention.md) | Module 2: Bin-based Sparse Attention 详细设计 |
| [03_neural_network_architecture.md](./03_neural_network_architecture.md) | 神经网络架构与 Kernel 编码 |
| [04_training_and_labels.md](./04_training_and_labels.md) | 训练数据与标签定义 |

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-12-14 | 初始化文档结构 |
| 2025-12-14 | 添加背景介绍：现有代码说明、项目动机 |
| 2025-12-15 | 修正 round_window 默认值为 128 |
