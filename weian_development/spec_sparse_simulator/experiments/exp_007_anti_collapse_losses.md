# Exp 007: Anti-Collapse Loss Design

## 背景 (Background)

基于 exp_006 和 exp_006a 的分析发现，Query Network 存在严重的 **bin collapse** 问题：99.96% 的 query 路由到单个 bin (Bin 37)。本文档定义两个新的 loss 来解决此问题。

## 术语定义 (Terminology)

原先使用的 "bin" 概念不够准确，重新定义为 **探针 (Probe)**：

- **探针 (Probe)**: 假设有 $N$ 个探针，Key Network 为每个探针维护一个对所有 key 的评分向量
- **探针机制**: 每个探针 $p_i$ 对所有 key 进行 softmax 评分，推理时对该探针的评分排序，取 Top-K 个 key
- **路由**: Query Network 输出一个 $N$ 维概率分布，选择概率最高的探针作为该 query 的路由目标

## Loss 1: 探针激活损失 (Probe Activation Loss)

### 目的

在训练过程中，某些探针可能因为 collapse 而被选中的概率趋近于 0，导致永远无法获得梯度更新。此 loss 用于激活这些 "死亡" 探针。

### 阈值定义

设探针总数为 $N$，定义允许的最小使用率为 $\alpha$（默认 $\alpha = 0.05$，即 5%）。

**死亡阈值**:
$$\tau = \frac{\alpha}{N}$$

在理想均衡状态下，每个探针被选中的概率为 $\frac{1}{N}$。允许不均衡到原来的 $\alpha$ 倍，即低于 $\frac{\alpha}{N}$ 的探针视为死亡探针。

### 死亡探针检测

给定一个 batch 中的所有 query，Query Network 输出概率矩阵 $\mathbf{P} \in \mathbb{R}^{B \times N}$，其中 $B$ 为 query 数量，$N$ 为探针数量。

1. 沿 query 维度取平均：
$$\bar{\mathbf{p}} = \frac{1}{B} \sum_{i=1}^{B} \mathbf{P}_{i,:} \in \mathbb{R}^{N}$$

2. 生成死亡探针掩码：
$$\mathbf{m}_{\text{dead}} = \mathbb{1}[\bar{\mathbf{p}} < \tau] \in \{0, 1\}^{N}$$

其中 $\mathbb{1}[\cdot]$ 为指示函数。

### 激活损失计算

设死亡探针集合为 $\mathcal{D} = \{i : m_{\text{dead},i} = 1\}$。

对于每个死亡探针 $d \in \mathcal{D}$，以当前 batch 中所有 query 的 **argmax key** 作为正样本集合 $\mathcal{K}^+$（与主损失 $\mathcal{L}_{\text{attraction}}$ 的正样本定义一致）。

**激活损失** (多标签交叉熵形式):
$$\mathcal{L}_{\text{activation}} = -\frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \frac{1}{|\mathcal{K}^+|} \sum_{k \in \mathcal{K}^+} \log \sigma_{d,k}$$

其中 $\sigma_{d,k}$ 为探针 $d$ 对 key $k$ 的 softmax 得分（来自 Key Network）。

**实现注意**: 计算 $\log \sigma_{d,k} = \log \text{softmax}(\mathbf{z}_d)_k$ 时，应使用 `log_softmax` 或 `logsumexp` API 避免数值精度问题，不要先算 softmax 再取 log。

**直观理解**: 对每个死亡探针，强制其学习将当前 batch 中的活跃 key 排在前面，从而"激活"该探针。

### 超参数

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 允许不均衡比例 | $\alpha$ | 0.05 | 控制死亡阈值的松弛程度 |
| 激活损失权重 | $\lambda_{\text{act}}$ | TBD | 需实验调整 |

## Loss 2: 负载均衡损失 (Load Balancing Loss)

### 目的

类似于 Mixture-of-Experts (MOE) 中的 load balancing loss，鼓励 Query Network 均匀使用所有探针。

### 公式

参考 Switch Transformer 的 auxiliary load balancing loss：

设一个 batch 中有 $B$ 个 query，Query Network 输出概率矩阵 $\mathbf{P} \in \mathbb{R}^{B \times N}$。

定义：
- **分配比例** $f_i$: 探针 $i$ 被分配到的 query 比例（硬分配，argmax）
$$f_i = \frac{1}{B} \sum_{j=1}^{B} \mathbb{1}[\text{argmax}(\mathbf{P}_{j,:}) = i]$$

- **概率质量** $P_i$: 所有 query 对探针 $i$ 的平均概率（软分配）
$$P_i = \frac{1}{B} \sum_{j=1}^{B} \mathbf{P}_{j,i}$$

**负载均衡损失**:
$$\mathcal{L}_{\text{balance}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

**直观理解**: 当某个探针同时具有高分配比例 $f_i$ 和高概率质量 $P_i$ 时，loss 会增大，从而抑制 rich-get-richer 的 collapse 动态。

### 超参数

| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| 负载均衡损失权重 | $\lambda_{\text{bal}}$ | TBD | 需实验调整 |

## 实验顺序 (Experiment Order)

1. **Phase 1**: 实现并验证 **探针激活损失** ($\mathcal{L}_{\text{activation}}$)
   - 评估对死亡探针的激活效果
   - 评估对整体 Hit Rate 的影响
   - 调整 $\alpha$ 和 $\lambda_{\text{act}}$ 超参数

2. **Phase 2**: 在 Phase 1 基础上添加 **负载均衡损失** ($\mathcal{L}_{\text{balance}}$)
   - 评估探针利用率的改善
   - 评估与激活损失的协同效果
   - 调整 $\lambda_{\text{bal}}$ 超参数

## 总损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{attraction}} + \lambda_{\text{act}} \cdot \mathcal{L}_{\text{activation}} + \lambda_{\text{bal}} \cdot \mathcal{L}_{\text{balance}}$$

其中 $\mathcal{L}_{\text{attraction}}$ 为原有的主损失。

## 相关文档

- [exp_006 ANALYSIS_REPORT](./exp_006_module2_reverse_cross_trace_validation/ANALYSIS_REPORT.md) - Bin collapse 详细分析
- [exp_006a README](./exp_006a_top2_bin_inference/README.md) - Top-2 推理实验
