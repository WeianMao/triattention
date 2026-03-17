# 新训练方案：基于排名的对齐推理目标

## 核心思想

**问题**：原 loss 用 softmax 概率，推理用 top-K 排名，不对齐。

**解决**：直接用排名相关的 loss，让 argmax_key 在某个 bin 内排进 top-K。

**关键设计**：完全基于 rank，不依赖数值 margin，避免模型通过调整 logit scale 来 hack loss。

---

## 方案概述

训练分两部分：

1. **Key 侧**：让 argmax_key 在目标 bin 内**排名进入 top-K'**（K' < K，留排名余量）
2. **Query 侧**：让 query **选中目标 bin(s)**

---

## Step 1：找目标 Bin（基于排名）

有两种方案可选：

### 方案 A：选排名最高的 Bin(s)

**规则**：argmax_key 在哪个 bin 的**排名最高**，哪个 bin 就是目标。

**多正样本情况**：如果多个 bin 排名并列最高，都算正样本。

**具体做法**：
```python
# 计算 argmax_key 在每个 bin 的排名（排名从 1 开始，1 表示第一名）
ranks = []
for b in range(num_bins):
    # 计算有多少 key 的 logit >= argmax_key 的 logit
    rank_b = (key_logits[:, b] >= key_logits[argmax_key, b]).sum().item()
    ranks.append(rank_b)
ranks = torch.tensor(ranks)

# 找排名最好（数值最小）的 bin(s)
best_rank = ranks.min()
positive_bins = (ranks == best_rank).nonzero().squeeze(-1)  # 可能有多个
```

**举例**：
- argmax_key 在 bin 0 排第 5，bin 1 排第 5，bin 2 排第 100
- `best_rank = 5`
- `positive_bins = [0, 1]`（两个都是正样本）

**特点**：
- 总是至少有一个正样本
- 即使 argmax_key 在所有 bin 排名都很差，也会选最好的那个

---

### 方案 B：选所有进入 top-K' 的 Bin(s)

**规则**：只要 argmax_key 在某个 bin 内**排名进入 top-K'**，这个 bin 就是正样本。

**多正样本情况**：可能有多个 bin 都满足条件，都是正样本。

**具体做法**：
```python
K_prime = 45  # 训练阈值

# 计算 argmax_key 在每个 bin 的排名
ranks = []
for b in range(num_bins):
    rank_b = (key_logits[:, b] >= key_logits[argmax_key, b]).sum().item()
    ranks.append(rank_b)
ranks = torch.tensor(ranks)

# 选所有排名 <= K' 的 bin
positive_bins = (ranks <= K_prime).nonzero().squeeze(-1)

# 如果没有任何 bin 满足条件，退化到方案 A（选排名最好的）
if positive_bins.numel() == 0:
    best_rank = ranks.min()
    positive_bins = (ranks == best_rank).nonzero().squeeze(-1)
```

**举例**：
- argmax_key 在 bin 0 排第 10，bin 1 排第 30，bin 2 排第 100
- K' = 45
- `positive_bins = [0, 1]`（排名 10 和 30 都 <= 45，都是正样本）

**特点**：
- 语义更直接：只要推理时"能命中"的 bin，都是正样本
- 如果 argmax_key 在多个 bin 都能命中，Query 选哪个都对
- 兜底：如果没有任何 bin 能命中，退化到方案 A

---

### 方案对比

| | 方案 A | 方案 B |
|---|---|---|
| 正样本定义 | 排名最高的 bin(s) | 所有能命中的 bin(s) |
| 正样本数量 | 通常较少（只有并列最高的） | 可能较多 |
| 语义 | "最好的 bin" | "所有够好的 bin" |
| 兜底 | 总有正样本 | 没有时退化到方案 A |

---

## Step 2：Key 侧 Loss（排名 Margin）

**目标**：让 argmax_key 的 logit 比目标 bin 内**第 K' 名**高。

**排名 Margin 思想**：
- 推理用 top-50
- 训练要求进 top-45（K' = 45）
- 留 5 个位置的排名余量，不用数值 margin

```python
K_prime = 45  # 训练时的 threshold，比推理的 K=50 更严格

# 取第一个正样本 bin 来算 key loss（多个正样本时取任意一个即可）
target_bin = positive_bins[0]

# threshold 是目标 bin 内第 K' 名的 logit（detach 避免不稳定梯度）
threshold = torch.topk(key_logits[:, target_bin], K_prime).values[-1].detach()

# Hinge loss，margin=0
L_key = torch.relu(threshold - key_logits[argmax_key, target_bin])
```

**解释**：
- 如果 argmax_key 已经进入 top-45，loss = 0
- 否则，loss > 0，把 argmax_key 往上推
- 不用数值 margin，排名余量通过 K' < K 实现

---

## Step 3：Query 侧 Loss（多正样本交叉熵）

**目标**：让 query 选中正样本 bin(s) 的概率之和最大。

**公式**：
```
L_bin = -log(sum(p[positive_bins]))
```

**数值稳定实现（用 logsumexp）**：
```python
query_logits  # (num_bins,)

# 取正样本 bins 的 logits
positive_logits = query_logits[positive_bins]  # (num_positive,)

# 用 logsumexp 计算 log(sum(exp(logits))) - log(sum(exp(all_logits)))
# = logsumexp(positive_logits) - logsumexp(query_logits)
L_bin = torch.logsumexp(query_logits, dim=0) - torch.logsumexp(positive_logits, dim=0)
```

**解释**：
- 如果只有一个正样本，退化为标准交叉熵
- 如果有多个正样本，只要 query 选中其中任意一个就算对

---

## 总 Loss

```python
loss = L_key + lambda * L_bin
```

---

## 完整伪代码

```python
def compute_loss(key_logits, query_logits, argmax_key, K_prime=45, lam=1.0):
    """
    key_logits: (num_keys, num_bins)
    query_logits: (num_bins,)
    argmax_key: int, ground truth key index
    K_prime: 训练时的 top-K 阈值（比推理的 K 更严格）
    """
    num_bins = key_logits.shape[1]

    # Step 1: 找正样本 bin(s)，基于排名
    argmax_key_logits = key_logits[argmax_key, :]  # (num_bins,)
    # 计算每个 bin 内的排名：有多少 key 的 logit >= argmax_key 的 logit
    ranks = (key_logits >= argmax_key_logits.unsqueeze(0)).sum(dim=0)  # (num_bins,)

    best_rank = ranks.min()
    positive_bins = (ranks == best_rank).nonzero().squeeze(-1)
    if positive_bins.dim() == 0:
        positive_bins = positive_bins.unsqueeze(0)

    # Step 2: Key loss (hinge, margin=0, 用 K' 实现排名余量)
    target_bin = positive_bins[0]
    threshold = torch.topk(key_logits[:, target_bin], K_prime).values[-1].detach()
    L_key = torch.relu(threshold - key_logits[argmax_key, target_bin])

    # Step 3: Query loss (多正样本交叉熵，用 logsumexp)
    positive_logits = query_logits[positive_bins]
    L_bin = torch.logsumexp(query_logits, dim=0) - torch.logsumexp(positive_logits, dim=0)

    # Total
    loss = L_key + lam * L_bin
    return loss
```

---

## 与原 Loss 的对比

| | 原 Loss | 新 Loss |
|---|---|---|
| Key 侧目标 | softmax 概率高 | 排名进 top-K' |
| Query 侧目标 | bin 分布匹配 | 选中正样本 bin(s) |
| 受 scale 影响 | 是（可被 hack） | 否（纯排名） |
| 超参 | margin（难调） | K'（语义清晰） |
| 与推理对齐 | 否 | 是 |

---

## 超参设置

| 超参 | 默认值 | 说明 |
|---|---|---|
| K（推理） | 50 | 推理时取 top-50 |
| K'（训练） | 45 | 训练时要求进 top-45，留 5 个排名余量 |
| lambda | 1.0 | Query loss 权重 |

---

## 监控指标

1. **Hit Rate (top-1 bin, K=50)**：最终推理指标
2. **Bin Accuracy**：query 选中的 top-1 bin 是否在 positive_bins 中
3. **Average Rank**：argmax_key 在 query 选中的 bin 内的平均排名（越小越好）
