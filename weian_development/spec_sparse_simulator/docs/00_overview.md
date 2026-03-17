# Neural Sparse Attention - Overview

## 项目目标

基于神经网络的两阶段 Sparse Attention 算法，用于长序列 LLM 推理。

## 背景与动机

从 **KV Cache 压缩** 扩展到 **Sparse Attention**：

| 对比项 | KV Cache 压缩 (现有) | Sparse Attention (本项目) |
|--------|---------------------|--------------------------|
| 机制 | 丢弃 Key，减少显存 | 保留 Key，只计算部分 attention |
| max_keys | 有硬限制 | 无此限制 |
| 打分方式 | 手工频率特征 | 神经网络学习 |

## 系统架构

```
┌──────────────────────────────────────────────────────┐
│              每 128 次解码的 Round 开头                 │
├──────────────────────────────────────────────────────┤
│   Module 1: Key Pruning  →  Module 2: Key Scoring    │
│   (Drop KV)                 (给剩余 K 打分)            │
└──────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────┐
│              每次解码 (128 次内)                        │
├──────────────────────────────────────────────────────┤
│   Query → Query Routing → TopK Sparse Attention      │
│           (Q 选 bin)       (Q × TopK Keys in bin)     │
└──────────────────────────────────────────────────────┘
```

## 两个模块概述

| 模块 | 目标 | 执行时机 | 网络 | 输出 |
|------|------|----------|------|------|
| Module 1 | 丢弃未来不被 attend 的 Key | Round 开头 | Kernel + MLP + Position Scaling | 二分类 |
| Module 2 | K 多 bin 打分，Q 选 bin 后取 TopK | Round 开头(K)/每次解码(Q) | Kernel + Softmax | 128 个 bin |

## Multi-head 处理

- 每个 Query head 独立学习神经网络
- GQA 情况下，同一 K 在不同 Query head 视角下可能分到不同 bin

## 文档索引

| 文件 | 内容 |
|------|------|
| [01_architecture.md](./01_architecture.md) | 系统架构与模块详细设计 |
| [02_neural_network.md](./02_neural_network.md) | 神经网络架构（Kernel Encoding） |
| [03_loss_and_training.md](./03_loss_and_training.md) | Loss 设计与训练策略 |
| [04_evaluation_metrics.md](./04_evaluation_metrics.md) | 评估指标设计 |
| [05_experiment_conventions.md](./05_experiment_conventions.md) | 实验规范 |
