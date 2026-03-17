# 实验 008: Top-K Attention Pairs 设计文档

## 1. 概述

**状态**: 已完成 ✅

**基准参考**: `exp_007_probe_activation_loss_ablation`

**核心改动**: 将 Ground Truth 的选择从 Top-1 扩展到 Top-K attention pairs。

**其他设置**: 除本文档描述的改动外，其他配置与 exp_007 保持一致，包括：
- 两个可选 Loss（`lambda_activation`, `lambda_balance`）**默认关闭**（均为 0.0）
- 模型架构：`Module2Network`，包含 Key/Query 两个网络
- 训练超参数：与 exp_007 相同

---

## 2. 背景：当前 Top-1 方法

### 2.1 问题设定

在 Attention 机制中：
- **Query** $Q_i$：想要关注相关的 Keys
- **Key** $K_j$：被关注的候选对象
- **Attention Score**：$\alpha_{i,j} = \text{softmax}(Q_i \cdot K_j / \sqrt{d})$

### 2.2 当前 Ground Truth 选择方式（Top-1）

对于每个 Query $Q_i$，我们只选择**一个** Key 作为 Ground Truth：

$$k^* = \arg\max_j \alpha_{i,j}$$

这个 Key $K_{k^*}$ 被称为 "argmax Key"——即从 Query $Q_i$ 获得最高 attention 的 Key。

### 2.3 当前 Loss 函数（Attraction Loss）

给定：
- $p^{(Q)}_i(b)$：Query $Q_i$ 选择 bin $b$ 的概率（对 bins 做 softmax）
- $P^{(K)}_j(b)$：Key $K_j$ 属于 bin $b$ 的概率（对 keys 做 softmax）

**当前 Attraction Loss**：

$$\mathcal{L}_{\text{attraction}} = -\frac{1}{N} \sum_{i=1}^{N} \log \left( \sum_{b=1}^{B} p^{(Q)}_i(b) \cdot P^{(K)}_{k^*_i}(b) \right)$$

其中：
- $N$：batch 中的 query 数量
- $B$：bin（探针）数量，默认 128
- $k^*_i$：query $i$ 的 argmax key 索引

**含义**：鼓励 Query $Q_i$ 选择与其 argmax Key $K_{k^*_i}$ 相同的 bin。

---

## 3. 新想法：Top-K Attention Pairs

### 3.1 动机

**Top-1 的问题**：
- 在实际 attention 中，一个 Query 通常会同时关注**多个 Keys**
- Top-1 只捕获了最强的 attention，忽略了其他重要的 Keys
- 这可能导致监督信号有偏差，无法反映真实的 attention 模式

**改进方案**：
- 引入超参数 $\mathbf{K}$（Top-K）来选择 Ground Truth
- 之前的 Top-1 是 $K=1$ 的特例
- 可选 $K \in \{1, 3, 5, 10, ...\}$

### 3.2 Top-K Key 选择算法

对于每个 Query $Q_i$：

1. **计算 Attention Scores**：
   $$\alpha_{i,j} = \frac{\exp(Q_i \cdot K_j / \sqrt{d})}{\sum_{l \leq i} \exp(Q_i \cdot K_l / \sqrt{d})}$$
   （使用 causal mask：只考虑位置 $j \leq i$ 的 Keys）

2. **选择 Top-K Keys**：
   $$\mathcal{K}^+_i = \text{TopK}_j(\alpha_{i,j}, K) = \{k_1, k_2, ..., k_K\}$$
   其中 $\alpha_{i,k_1} \geq \alpha_{i,k_2} \geq ... \geq \alpha_{i,k_K}$

3. **构建 Ground Truth Pairs**：
   对于 $k_m \in \mathcal{K}^+_i$ 中的每个 key，$(Q_i, K_{k_m})$ 都是一个 Ground Truth pair。

### 3.3 更新后的 Loss 函数（Top-K Attraction Loss）

**方案 A：等权重（推荐）**

所有 Top-K pairs 对 loss 贡献相等：

$$\mathcal{L}_{\text{attraction}}^{\text{TopK}} = -\frac{1}{N \cdot K} \sum_{i=1}^{N} \sum_{k \in \mathcal{K}^+_i} \log \left( \sum_{b=1}^{B} p^{(Q)}_i(b) \cdot P^{(K)}_k(b) \right)$$

**展开形式**：对于单个 Query $Q_i$ 及其 Top-K Keys $\{k_1, k_2, ..., k_K\}$：

$$\mathcal{L}_i = -\frac{1}{K} \sum_{m=1}^{K} \log \left( \sum_{b=1}^{B} p^{(Q)}_i(b) \cdot P^{(K)}_{k_m}(b) \right)$$

**方案 B：Attention 加权（备选）**

按 attention score 加权每个 pair（供后续探索）：

$$\mathcal{L}_{\text{attraction}}^{\text{weighted}} = -\frac{1}{N} \sum_{i=1}^{N} \frac{\sum_{k \in \mathcal{K}^+_i} \alpha_{i,k} \cdot \log \left( \sum_{b=1}^{B} p^{(Q)}_i(b) \cdot P^{(K)}_k(b) \right)}{\sum_{k \in \mathcal{K}^+_i} \alpha_{i,k}}$$

---

## 4. 实现细节

### 4.1 超参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `topk_gt` | int | 1 | 用作 Ground Truth 的 top attention keys 数量 |

### 4.2 修改函数：`extract_query_to_key_labels`

**当前签名**（exp_007）：
```python
def extract_query_to_key_labels(attention_trace, round_start, round_end, seq_len, exclude_tail=1000):
    """
    返回:
        dict:
            - query_indices: (num_valid_queries,)
            - argmax_keys: (num_valid_queries,)  # 每个 query 一个 key
            - argmax_in_recent: (num_valid_queries,)
    """
```

**新签名**（exp_008）：
```python
def extract_query_to_key_labels(attention_trace, round_start, round_end, seq_len, exclude_tail=1000, topk_gt=1):
    """
    返回:
        dict:
            - query_indices: (num_valid_queries,)
            - topk_keys: (num_valid_queries, topk_gt)  # 每个 query K 个 keys
            - topk_in_recent: (num_valid_queries, topk_gt)  # 布尔 mask
    """
```

**实现伪代码**：
```python
for q_idx in range(round_start, min(round_end, valid_end)):
    attn_weights = attention_trace[q_idx, :q_idx + 1]  # 已应用 causal mask

    # 选择 Top-K keys，而不是只选 argmax
    topk_values, topk_indices = torch.topk(attn_weights, k=min(topk_gt, len(attn_weights)))

    # 存储 Top-K key 索引
    topk_keys.append(topk_indices)

    # 检查哪些 keys 在当前 round（recent keys）
    topk_in_recent.append(topk_indices >= round_start)
```

### 4.3 修改函数：`compute_attraction_loss`

**新签名**（exp_008）：
```python
def compute_attraction_loss(key_probs, query_bin_probs, topk_keys, topk_in_recent, eps=1e-8):
    """
    参数:
        key_probs: (num_keys, num_bins)
        query_bin_probs: (num_queries, num_bins)
        topk_keys: (num_queries, topk_gt) 每个 query 的 Top-K key 索引
        topk_in_recent: (num_queries, topk_gt) 布尔 mask

    返回:
        标量 loss（对所有有效 Query-Key pairs 取平均）
    """
```

**实现伪代码**：
```python
total_loss = 0.0
valid_pairs = 0

for i in range(num_queries):
    query_prob = query_bin_probs[i]  # (num_bins,)

    for m in range(topk_gt):
        if topk_in_recent[i, m]:
            continue  # 跳过 recent keys

        key_idx = topk_keys[i, m]
        key_prob = key_probs[key_idx]  # (num_bins,)

        # 匹配概率
        match_prob = (query_prob * key_prob).sum()
        total_loss += -torch.log(match_prob + eps)
        valid_pairs += 1

return total_loss / valid_pairs if valid_pairs > 0 else 0.0
```

---

## 5. 数学总结

### 5.1 符号表

| 符号 | 含义 |
|------|------|
| $Q_i$ | 位置 $i$ 的 Query 向量 |
| $K_j$ | 位置 $j$ 的 Key 向量 |
| $\alpha_{i,j}$ | Query $i$ 到 Key $j$ 的 attention score |
| $K$ | 要选择的 top keys 数量（超参数 `topk_gt`） |
| $\mathcal{K}^+_i$ | Query $i$ 的 Top-K key 索引集合 |
| $p^{(Q)}_i(b)$ | Query $i$ 选择 bin $b$ 的概率 |
| $P^{(K)}_j(b)$ | Key $j$ 属于 bin $b$ 的概率 |
| $B$ | bin（探针）总数，默认 128 |
| $N$ | batch 中的 query 数量 |

### 5.2 完整算法

**输入**：Attention 矩阵 $A$，Query 概率 $p^{(Q)}$，Key 概率 $P^{(K)}$，参数 $K$

**输出**：Loss $\mathcal{L}$

```
1. 对于每个 query i：
   a. 获取 attention scores：α_i = A[i, :i+1]（causal）
   b. 选择 Top-K：K⁺_i = TopK(α_i, K)

2. 计算 loss：
   L = 0, count = 0
   对于每个 query i：
     对于 K⁺_i 中的每个 key k：
       如果 k 不在 recent round 中：
         match = Σ_b p^(Q)_i(b) · P^(K)_k(b)
         L += -log(match)
         count += 1

   返回 L / count
```

### 5.3 对比：Top-1 vs Top-K

| 方面 | Top-1（当前） | Top-K（新方案） |
|------|---------------|-----------------|
| Ground Truth | 单个最强 Key | K 个最强 Keys |
| Loss 公式 | $-\log(\text{match}_{k^*})$ | $-\frac{1}{K}\sum_{m=1}^{K}\log(\text{match}_{k_m})$ |
| 监督信号 | 集中在 argmax | 分布在多个重要 pattern |
| 边界情况 | 可能忽略重要 Keys | K 太大时可能引入噪声 |

---

## 6. 建议实验

### 6.1 消融实验

| 实验 | `topk_gt` | 说明 |
|------|-----------|------|
| Baseline | 1 | 当前行为 |
| Top-3 | 3 | 初步扩展 |
| Top-5 | 5 | 中等 |
| Top-10 | 10 | 激进 |

### 6.2 评估指标

- Hit Rate @ K=50/500/1000（与 exp_007 相同）
- 比较 loss 收敛曲线
- 分析 bin 利用率模式

---

## 7. 配置变更

```yaml
# exp_008 config.yaml 新增内容
training:
  topk_gt: 1  # 默认值：与 Top-1 向后兼容
              # 设置为 3, 5, 10 进行 Top-K 实验
```

---

## 8. 实验结果

**实验日期**：2024-12-18

### 8.1 Hit Rate 对比

| topk_gt | K=50 Hit Rate | K=500 Hit Rate | K=1000 Hit Rate |
|---------|---------------|----------------|-----------------|
| 1       | 98.62%        | 98.92%         | 99.16%          |
| 3       | 98.67%        | 99.03%         | 99.23%          |
| 5       | 98.58%        | 98.92%         | 99.17%          |
| **10**  | **98.68%**    | **99.08%**     | **99.26%**      |

### 8.2 Bin Collapse 指标对比

| topk_gt | Active Bins (soft) | Entropy (normalized) | Gini | Max Bin Usage |
|---------|-------------------|---------------------|------|---------------|
| 1       | 1/128             | 0.0083              | 0.9919 | 0.9951      |
| 3       | 11/128            | 0.2980              | 0.9728 | 0.5025      |
| 5       | 11/128            | 0.2788              | 0.9727 | 0.6231      |
| **10**  | **16/128**        | **0.4460**          | **0.9488** | **0.3204** |

### 8.3 关键发现

1. **Hit Rate**：所有配置在 K=50 时都达到 >98.5% 的命中率，差异不大（~0.1-0.2%）。`topk_gt=10` 在 K=1000 时达到最高命中率 99.26%。

2. **Bin Collapse**：这是最显著的差异：
   - `topk_gt=1`：严重崩溃（仅 1 个活跃 bin，Gini=0.9919，Entropy=0.0083）
   - `topk_gt=3`：分布改善（11 个活跃 bins，Entropy=0.2980）
   - `topk_gt=5`：与 3 类似（11 个活跃 bins，Entropy=0.2788）
   - `topk_gt=10`：最佳分布（16 个活跃 bins，Entropy=0.4460，Gini=0.9488）

3. **Training Loss**：随着 topk_gt 增加，最终 loss 值增大（因为需要同时满足更多 key 的匹配），但这不影响实际的 hit rate 性能。

### 8.4 结论

**推荐配置**：`topk_gt=10`

理由：
- 最高的 entropy（0.4460），表示更均匀的 bin 使用
- 最低的 Gini 系数（0.9488），表示更少的集中度
- 最低的 max bin usage（0.3204），表示更好的负载分布
- 略高的 hit rate 表现

**核心洞察**：增加 topk_gt 可以显著减少 bin 崩溃问题，同时保持相似的命中率。这验证了我们的假设：使用多个 attention pairs 作为 Ground Truth 可以提供更丰富的监督信号，帮助模型学习更均衡的 bin 分配。

---

## 9. 参考

- 基准实验：`exp_007_probe_activation_loss_ablation/`
- 当前 loss 实现：`exp_007_probe_activation_loss_ablation/train.py:159-200`
- 当前 label 提取：`exp_007_probe_activation_loss_ablation/train.py:108-156`
- 实验结果：`exp_008_topk_attention_pairs/output/ablation_topk/`
