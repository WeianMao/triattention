# Sparse Attention Bin Assignment Network Documentation

本文档详细介绍 `exp_012_stage1_shared_probe_q_scoring` 和 `exp_006_module2_reverse_cross_trace_validation` 两个实验的网络架构和训练方法。

---

## 目录

1. [概述](#概述)
2. [Exp_006: Von Mises Kernel 网络](#exp_006-von-mises-kernel-网络)
3. [Exp_012: Shared Probe 网络](#exp_012-shared-probe-网络)
4. [训练流程对比](#训练流程对比)
5. [损失函数](#损失函数)

---

## 概述

两个实验都旨在解决**稀疏注意力中的 Bin Assignment 问题**：将 Key 和 Query 向量分配到不同的 Bin 中，使得每个 Query 能快速定位到其 argmax Key 所在的 Bin。

| 特性 | Exp_006 | Exp_012 |
|------|---------|---------|
| 网络类型 | Von Mises Kernel | Shared Probe |
| K/Q 网络 | 独立参数 | 共享 Probe 层 |
| 总参数量 | ~147K | ~41K |
| RoPE 处理 | Kernel 中心偏移 | Probe 向量旋转 |
| 归一化 | 无 | L2 归一化 (可选) |

---

## Exp_006: Von Mises Kernel 网络

### 网络架构

```
输入: K/Q 向量 (num_vectors, head_dim=128)
          ↓
   KernelEncodingLayer (Von Mises Kernels)
          ↓
输出: logits (num_vectors, num_bins=128)
```

#### KernelEncodingLayer

**核心思想**: 使用 Von Mises 核函数对向量的相位信息进行编码。

**参数** (每个网络 ~73K):
- `mu`: (128, 64, 3) - 核中心位置
- `kappa`: (128, 64, 3) - 核宽度 (集中度)
- `weight`: (128, 64, 3) - 核权重
- `bias`: (128,) - 偏置

**前向传播**:

1. **复数表示**: 将 head_dim=128 的向量重塑为 64 个复数对
   ```python
   K_complex = K.view(num_keys, 64, 2)  # (num_keys, num_freqs, 2)
   ```

2. **提取幅度和相位**:
   ```python
   magnitude = torch.norm(K_complex, dim=2)      # (num_keys, 64)
   angle = torch.atan2(K_complex[..., 1], K_complex[..., 0])  # (num_keys, 64)
   ```

3. **RoPE 位置编码融合**:
   ```python
   # reference_angles 基于 round 的中点位置计算
   ref_position = round_start + round_window // 2
   omega = 1.0 / (10000 ** (2 * dim_indices / head_dim))
   reference_angles = ref_position * omega  # (64,)

   # 将 reference_angles 添加到 mu 中心
   mu_effective = self.mu + reference_angles.view(1, -1, 1)  # (128, 64, 3)
   ```

4. **Von Mises 核计算**:
   ```python
   # 归一化 Von Mises: exp(kappa * (cos(angle - mu) - 1))
   # 这确保了核值在 [0, 1] 范围内，当 angle = mu 时取最大值 1
   kernel = exp(kappa * (cos(angle - mu_effective) - 1))  # (num_keys, 128, 64, 3)
   ```

5. **加权求和**:
   ```python
   # 对 3 个核求加权和
   weighted_kernels = (kernel * weight).sum(dim=3)  # (num_keys, 128, 64)

   # 乘以幅度，对频率维度求和
   logits = (weighted_kernels * magnitude).sum(dim=2) + bias  # (num_keys, 128)
   ```

#### Module2Network 结构

```
Module2Network
├── key_network (Module2KeyNetwork)
│   └── kernel_layer (KernelEncodingLayer)  # 73,856 params
└── query_network (Module2QueryNetwork)
    └── kernel_layer (KernelEncodingLayer)  # 73,856 params (独立参数)

总参数: 147,712
```

**关键特点**:
- K 和 Q 网络使用**相同架构但独立参数**
- Softmax 在网络外部应用:
  - Key: `softmax(logits, dim=0)` - 每列 (bin) 的 key 概率和为 1
  - Query: `softmax(logits, dim=-1)` - 每行 (query) 的 bin 概率和为 1

---

## Exp_012: Shared Probe 网络

### 网络架构

```
                    SharedProbeLayer (128 learnable probes)
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                                      ↓
 Module2KeyNetwork                    Module2QueryNetwork
 (dot product + magnitude)            (distance-based scoring)
        ↓                                      ↓
  key_logits                              query_logits
```

#### SharedProbeLayer

**核心思想**: 使用共享的可学习 Probe 向量，通过 RoPE 旋转适应不同位置。

**参数**:
- `probes`: (128, 128) - 可学习的 Probe 向量 (16,384 params)

**Probe 旋转**:
```python
def get_rotated_probes(reference_angles, normalize=True):
    if normalize:
        probes_to_rotate = l2_normalize(self.probes)  # L2 归一化

    # RoPE 旋转: y = x * cos(theta) + rotate_half(x) * sin(theta)
    # rotate_half: [a, b] -> [-b, a]
    theta = ref_position * omega
    rotated_probes = probes * cos(theta) + rotate_half(probes) * sin(theta)
    return rotated_probes  # (128, 128)
```

#### Module2KeyNetwork

**得分公式**:
$$s_K^{(b)} = P_{\text{norm,rot}}^{(b)T} \cdot K + u_b^T \cdot m^K + k_{\text{bias}_b}$$

其中:
- $P_{\text{norm,rot}}^{(b)}$: L2 归一化后 RoPE 旋转的第 b 个 probe
- $K$: 原始 Key 向量 (不归一化)
- $m^K_f = \sqrt{K_{2f}^2 + K_{2f+1}^2}$: 幅度特征 (64 维)
- $u_b$: 幅度权重 (可学习)
- $k_{\text{bias}_b}$: 偏置

**参数**:
- `k_magnitude_weights`: (128, 64) - 8,192 params
- `k_bias`: (128,) - 128 params
- `temperature_raw`: (1,) - 可学习温度 (可选)
- `position_scaling`: (3,) - 位置依赖缩放 (可选)

**前向传播**:
```python
def forward(K, reference_angles):
    rotated_probes = self.probe_layer.get_rotated_probes(reference_angles, normalize=True)

    # 点积项
    dot_product_term = K @ rotated_probes.T  # (num_keys, 128)

    # 幅度特征
    K_magnitude = compute_magnitude_features(K, num_freqs=64)  # (num_keys, 64)
    magnitude_term = K_magnitude @ k_magnitude_weights.T  # (num_keys, 128)

    logits = dot_product_term + magnitude_term + k_bias

    # 可选: 温度缩放
    if use_key_temperature:
        logits = logits / softplus(temperature_raw)

    return logits  # (num_keys, 128)
```

#### Module2QueryNetwork

**得分公式**:
$$s_Q^{(b)} = \tilde{w}_b^T \cdot d_b + v_b^T \cdot m^Q_{\text{norm}} + c_b$$

其中:
- $d_{b,f} = \|P_{\text{norm,rot}}^{(b)}_f - Q_{\text{norm},f}\|$: 逐频率距离
- $\tilde{w}_b = -\text{softplus}(w_{\text{raw}_b})$: 负权重 (距离越小得分越高)
- $m^Q_f$: 归一化 Q 的幅度特征
- $v_b$: 幅度权重
- $c_b$: 偏置

**参数**:
- `q_weights_raw`: (128, 64) - 8,192 params
- `q_magnitude_weights`: (128, 64) - 8,192 params
- `q_bias`: (128,) - 128 params
- `q_error_weights`: (128, 128) - 16,384 params (可选)

**前向传播**:
```python
def forward(Q, rotated_probes):
    # L2 归一化 Q
    Q_normalized = l2_normalize(Q)

    # 分离实部和虚部
    Q_real, Q_imag = Q_normalized[:, :64], Q_normalized[:, 64:]
    P_real, P_imag = rotated_probes[:, :64], rotated_probes[:, 64:]

    # 逐频率距离
    error_real = P_real - Q_real.unsqueeze(1)  # (num_queries, 128, 64)
    error_imag = P_imag - Q_imag.unsqueeze(1)
    distance = sqrt(error_real**2 + error_imag**2 + eps)  # (num_queries, 128, 64)

    # 负 softplus 权重
    effective_weights = -softplus(q_weights_raw)  # (128, 64)

    # 距离项
    distance_term = (distance * effective_weights).sum(dim=-1)  # (num_queries, 128)

    # 幅度项
    Q_magnitude = compute_magnitude_features(Q_normalized, 64)
    magnitude_term = Q_magnitude @ q_magnitude_weights.T  # (num_queries, 128)

    scores = distance_term + magnitude_term + q_bias
    return scores
```

#### Module2Network 参数统计

```
Module2Network
├── shared_probe_layer.probes: 128 × 128 = 16,384
├── key_network
│   ├── k_magnitude_weights: 128 × 64 = 8,192
│   ├── k_bias: 128
│   ├── temperature_raw: 1 (可选)
│   └── position_scaling: 3 (可选)
└── query_network
    ├── q_weights_raw: 128 × 64 = 8,192
    ├── q_magnitude_weights: 128 × 64 = 8,192
    ├── q_bias: 128
    └── q_error_weights: 128 × 128 = 16,384 (可选)

基础参数: 41,216
带 error_term: 57,600
```

---

## 训练流程对比

### Exp_006 训练流程

```python
for epoch in range(epochs):
    for round_start in range(0, seq_len, round_window):
        if round_start == 0:
            continue  # 跳过第一个 round (无历史 keys)

        # 1. 获取历史 Keys
        historical_keys = K[:round_start]

        # 2. 计算参考角度
        reference_angles = model.compute_reference_angles(round_start, round_window)

        # 3. Key 网络前向 (一次性处理所有历史 keys)
        key_probs = softmax(key_network(historical_keys, reference_angles), dim=0)

        # 4. 检测空 bin
        empty_bin_mask = (key_probs.sum(dim=0) == 0)

        # 5. Query 网络前向 (分批处理)
        for batch in query_batches:
            query_bin_probs = softmax(query_network(queries, reference_angles), dim=-1)

            # 6. 计算 Attraction Loss
            loss = attraction_loss(key_probs, query_bin_probs, argmax_keys)

        # 7. 反向传播 (每个 round 一次)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Exp_012 训练流程 (V5 Batched)

```python
# 预处理: 预加载所有 traces 到 GPU
all_traces = preload_all_traces(trace_paths, layer, head, device)

# K-means 初始化 probes
init_probes = compute_kmeans_init_multi_trace(config, traces)

# 创建模型
model = create_model(config, init_probes=init_probes, use_l2_norm=True)

for epoch in range(epochs):
    for trace in all_traces:
        # 预计算所有 rounds 的标签 (首个 epoch)
        if 'cached_rounds' not in trace:
            attention = compute_attention_matrix(Q, K)
            rounds = extract_all_round_labels(attention)
            trace['cached_rounds'] = rounds
            del attention  # 释放内存

        # 批量处理 rounds (关键优化)
        for batch_start in range(0, len(rounds), round_batch_size):
            batch_rounds = rounds[batch_start:batch_end]

            # 1. 准备批量数据
            ref_positions = [r['round_start'] + window//2 for r in batch_rounds]
            key_lengths = [r['round_start'] for r in batch_rounds]

            # 2. 批量 Key 前向 (所有 rounds 共享 K 数据，不同旋转)
            key_probs_batch, key_mask = model.forward_keys_batched(
                K_shared, ref_positions, key_lengths
            )  # (batch_size, max_keys, num_bins)

            # 3. 批量 Query 前向
            bin_probs_batch = model.forward_queries_batched(
                Q_batch, ref_positions, empty_bin_mask_batch
            )  # (batch_size, max_queries, num_bins)

            # 4. 计算损失 (支持多种损失函数)
            if use_weighted_ce_loss:
                loss = weighted_ce_loss(key_probs, query_probs, argmax_keys)
            elif use_discrete_topk_loss:
                loss = discrete_topk_loss(key_logits, query_logits, argmax_keys)
            else:
                loss = attraction_loss(key_probs, query_probs, argmax_keys)

            # 5. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler_step_per_batch:
                scheduler.step()
```

### 训练优化对比

| 优化项 | Exp_006 | Exp_012 V5 |
|--------|---------|------------|
| 数据预加载 | 每 epoch 加载 | GPU 预加载所有 traces |
| Round 批处理 | 单 round 处理 | 多 round 批量处理 |
| 标签缓存 | 无 | 首 epoch 后缓存 |
| 多 trace | 单 trace | 多 trace 训练 |
| K-means 初始化 | 无 | 支持 |
| LR Scheduler | 无 | OneCycle/CosineRestart/Step |
| 损失函数 | Attraction | Attraction/WeightedCE/DiscreteTopK/Rank |

---

## 损失函数

### Attraction Loss (共用)

$$\mathcal{L}_{\text{attract}} = -\log\left(\sum_b p_Q(b) \cdot P(k^* | b) + \epsilon\right)$$

其中:
- $p_Q(b)$: Query 选择 bin $b$ 的概率
- $P(k^* | b)$: argmax key $k^*$ 在 bin $b$ 中的概率
- 目标: 让 Query 选择的 bin 包含其 argmax key

```python
def compute_attraction_loss(key_probs, query_bin_probs, argmax_keys, argmax_in_recent):
    # 排除 argmax 在当前 round 的 query (不需要路由)
    valid_mask = ~argmax_in_recent

    # 获取 argmax key 的 bin 概率
    P_matched = key_probs[argmax_keys[valid_mask]]  # (num_valid, num_bins)

    # 计算匹配概率
    match_prob = (query_bin_probs[valid_mask] * P_matched).sum(dim=1)

    # Attraction Loss
    loss = -torch.log(match_prob + eps).mean()
    return loss
```

### Weighted Cross-Entropy Loss (Exp_012)

$$\mathcal{L}_{\text{wce}} = -\sum_b p_Q(b) \cdot \log P(k^* | b)$$

**与 Attraction Loss 的区别**:
- Attraction: $-\log(\sum_b p_Q(b) \cdot P(k^* | b))$ — log 在求和外
- Weighted CE: $-\sum_b p_Q(b) \cdot \log P(k^* | b)$ — log 在求和内

根据 Jensen 不等式: $\mathcal{L}_{\text{wce}} \geq \mathcal{L}_{\text{attract}}$

### Discrete Top-K Loss (Exp_012)

```python
def compute_discrete_topk_loss(key_logits, query_logits, argmax_keys, top_k=50):
    # 1. 获取 Query 的 top-1 bin (离散选择)
    b_star = query_logits.argmax(dim=-1)

    # 2. 检查 argmax key 是否在该 bin 的 top-k keys 中
    _, topk_indices = key_logits[:, b_star].topk(top_k, dim=0)
    hit_mask = (topk_indices == argmax_keys).any(dim=0)

    # 3. 只对 miss cases 计算损失
    miss_mask = ~hit_mask
    if miss_mask.sum() == 0:
        return 0.0

    # 4. 计算 CE loss
    p_q_bstar = softmax(query_logits, dim=-1)[:, b_star]
    log_p_k_target = log_softmax(key_logits, dim=0)[argmax_keys, b_star]

    loss = (p_q_bstar * (-log_p_k_target) * miss_mask).sum() / miss_mask.sum()
    return loss
```

### Focal Loss (Exp_012 可选)

$$\mathcal{L}_{\text{focal}} = -(1 - p_{\text{match}})^\gamma \cdot \log(p_{\text{match}})$$

当 $\gamma > 0$ 时，降低易分样本的权重，聚焦于难分样本。

---

## 评估指标

### Top-K Hit Rate

对于每个 Query，检查其 argmax Key 是否在选中 bin 的 Top-K keys 中:

```python
# Top-1 bin, K=50
selected_bin = query_bin_probs.argmax()
top_k_keys = key_probs[:, selected_bin].topk(K).indices
hit = argmax_key in top_k_keys

# Top-8 bins, K=50 (每个 bin 取 top-K, 共 8*50=400 keys)
top_8_bins = query_bin_probs.topk(8).indices
hit = any(argmax_key in key_probs[:, bin].topk(K).indices for bin in top_8_bins)
```

### 评估配置

```yaml
evaluation:
  topk_K: [50, 500, 1000]  # 不同的 K 值
  round_window: 128
```

---

## 总结

| 方面 | Exp_006 (Von Mises) | Exp_012 (Shared Probe) |
|------|---------------------|------------------------|
| **设计理念** | 显式相位编码 | 距离/相似度学习 |
| **参数效率** | 较低 (~147K) | 较高 (~41K) |
| **位置编码** | 核中心偏移 | Probe 向量旋转 |
| **K/Q 关系** | 完全独立 | 共享 Probe 层 |
| **可解释性** | 中等 | 较好 (Probe 可视化) |
| **训练稳定性** | 一般 | 较好 (K-means 初始化) |
| **扩展性** | 有限 | 灵活 (多种损失函数) |
