# Top‑1 Bin 推理设置下的训练方案（对齐推理目标）

> 场景：推理时 **只取 Top‑1 bin**（`N=1`），然后在该 bin 内取 **Top‑K keys**，看 `argmax_key`（记作 \(k^*\)）是否被选中。  
> 目标：训练目标与推理指标（Hit@K with top1 bin）严格对齐。

---

## 0. Notation（符号）

- 历史 key 总数：\(L\)
- bin 总数：\(B\)
- 推理 Top‑K：\(K\)
- 对某个 query \(q\)：
  - Query 网络输出 bin logits：\(\{a_b\}_{b=1}^{B}\)（`query_logits[b]`）
  - Key 网络输出 key‑bin logits：\(\{s_{k,b}\}_{k=1..L,b=1..B}\)（`key_logits[k,b]`）
  - ground‑truth 最重要 key：\(k^*\)（`argmax_key`）

---

## 1. 推理目标写成数学条件（Top‑1 bin）

推理步骤（硬选择）：

1) 选 Top‑1 bin：
\[
\hat b \;=\;\arg\max_{b} a_b
\]

2) 在 bin \(\hat b\) 内选 Top‑K keys（按 \(s_{k,\hat b}\) 排序）

命中（hit）条件等价于：

- 定义 bin \(b\) 内第 \(K\) 名阈值（第 K 大 logit）：
\[
t_b \triangleq s_{(K),b}
\]
- 则 \(k^*\) 在 bin \(b\) 内进入 Top‑K 当且仅当：
\[
s_{k^*,b} \ge t_b
\]

所以 **Top‑1 bin 推理的命中事件**为：
\[
s_{k^*,\hat b} \ge t_{\hat b}
\]

> 训练必须围绕这一个事件来构造 surrogate loss。

---

## 2. 训练方案总览（推荐：最对齐且实现简单）

核心思想：在 top1 setting 下，训练要同时解决两件事：

1) **Bin 选择要正确**：让 query 的 top1 bin 变成“最适合命中”的目标 bin  
2) **Key 排名要正确**：在该目标 bin 内，让 \(k^*\) **超过 Top‑K 边界**（超过第 K 名阈值）

这对应两个 loss：

- Bin 分类 loss：\(L_{\text{bin}}\)
- Top‑K 边界 hinge loss：\(L_{\text{key}}\)

最终：
\[
L \;=\; L_{\text{key}} \;+\; \lambda\, L_{\text{bin}}
\]

---

## 3. 如何定义目标 bin \(b^*\)（没有“真实 bin label”的情况下）

由于 bin 是模型学习出来的，通常没有外部标注的“正确 bin”。  
因此我们用 **“在该 bin 内最容易命中 Top‑K”** 来定义伪标签（pseudo‑label）。

### 3.1 定义“命中裕量”（hit margin）

对每个 bin \(b\)，定义：
\[
r_b \triangleq s_{k^*,b} - t_b
\]

- 若 \(r_b \ge 0\)，说明在 bin \(b\) 内 \(k^*\) 已经进入 Top‑K  
- \(r_b\) 越大，说明 \(k^*\) 越稳地处于 Top‑K 前列

### 3.2 目标 bin 选择规则

最简单的定义：
\[
b^* \;=\;\arg\max_{b} r_b \;=\;\arg\max_{b}\big(s_{k^*,b}-t_b\big)
\]

工程注意：

- \(t_b=s_{(K),b}\) 通过 `topk` 得到，建议 **stop‑grad**：\(\tilde t_b=\text{stopgrad}(t_b)\)
- \(b^*\) 是离散 label，本来也不需要对它反传梯度

---

## 4. Key 侧 loss：Top‑K 边界 Margin（最对齐推理）

推理要求：\(s_{k^*,\hat b} \ge t_{\hat b}\)  
训练我们用更稳的 margin 版本（留安全边际）：

\[
L_{\text{key}} \;=\; \max\big(0,\; m + \tilde t_{b^*} - s_{k^*,b^*}\big)
\]

这等价于要求：
\[
s_{k^*,b^*} \ge \tilde t_{b^*} + m
\]

其中：

- \(m>0\) 是 margin（如 0.1 / 1.0，取决于 logit 尺度）
- \(\tilde t_{b^*}\) 是 stop‑grad 的第 K 名阈值

### 为什么 stop‑grad 合理？

`topk` 的“第 K 名是谁”是离散变化，强行对其求梯度往往更不稳定。  
我们真正希望的梯度方向是：**把正样本 \(s_{k^*,b^*}\) 往上推**。  
当 hinge 处于激活区间时：
\[
\frac{\partial L_{\text{key}}}{\partial s_{k^*,b^*}} = -1
\]
非常干净；当已满足 margin 时，梯度为 0，避免浪费优化。

---

## 5. Bin 侧 loss：让 Query 的 top1 bin 变成 \(b^*\)

定义：
\[
p(b\mid q)=\text{softmax}(a)_b
\]

用标准交叉熵：
\[
L_{\text{bin}} = -\log p(b^*\mid q)
\]

直觉：让 query 网络把概率质量集中到“最容易命中”的 bin 上，从而对齐 top1 推理。

---

## 6. 完整 Loss（单样本）

\[
\boxed{
L(q)=\max(0,\; m+\tilde t_{b^*}-s_{k^*,b^*})\;+\;\lambda\cdot\big(-\log p(b^*\mid q)\big)
}
\]

---

## 7. 最小可落地伪代码（PyTorch 风格）

```python
# key_logits: (L, B)   s_{k,b}
# query_logits: (B,)   a_b
# kstar: scalar index (argmax_key)

# 1) thresholds t_b = kth largest in each bin (stop-grad recommended)
t = []
for b in range(B):
    t_b = torch.topk(key_logits[:, b], K).values[K-1].detach()
    t.append(t_b)
t = torch.stack(t)  # (B,)

# 2) pseudo-label bin: b* = argmax_b (s_{k*,b} - t_b)
r = key_logits[kstar, :] - t          # (B,)
b_star = torch.argmax(r).item()       # discrete

# 3) key loss only on b*
L_key = torch.relu(margin + t[b_star] - key_logits[kstar, b_star])

# 4) bin CE loss: make query pick b*
p = torch.softmax(query_logits, dim=0)
L_bin = -torch.log(p[b_star] + 1e-8)

loss = L_key + lam * L_bin
```

---

## 8. 实验建议（Top‑1 setting 的常用超参起点）

- \(K\)：与推理一致（如 50）
- margin \(m\)：0.1 / 0.5 / 1.0 三档试一下（看 logit 尺度）
- \(\lambda\)：0.1 / 0.3 / 1.0 三档（控制 bin CE 强度）
- 监控指标：
  - **Hit@K (top1 bin)**：与最终推理一致的指标（最重要）
  - **Bin accuracy**：\(\hat b == b^*\) 的比例（伪标签一致性）
  - **Avg margin**：\(\mathbb{E}[s_{k^*,\hat b} - t_{\hat b}]\)（越大越稳）

---

## 9. 可选增强（仍保持“只取 top1 bin”的简单设置）

### 9.1 限定伪标签搜索范围（只在 query 的 Top‑M bins 内找 \(b^*\)）

early training 时，\(r_b\) 可能噪声大。可先取：
\[
\mathcal{B}_M(q)=\text{TopM}(a)
\]
再定义：
\[
b^*=\arg\max_{b\in\mathcal{B}_M(q)} (s_{k^*,b}-t_b)
\]
这样 pseudo‑label 更贴近“推理会选到的区域”。

### 9.2 近似阈值（当 L 非常大时）

用候选子集 \(\mathcal{S}_b\) 近似全量 keys：
\[
t_b \approx \text{KthLargest}\big(\{s_{k,b}\}_{k\in\mathcal{S}_b}\big)
\]
\(\mathcal{S}_b\) 可由 hard negatives + random negatives + \(\{k^*\}\) 组成。

---

## 10. Summary（一句话总结）

在 Top‑1 bin 推理设置下，训练要对齐的核心事件是：

\[
s_{k^*,\arg\max a_b} \ge s_{(K),\arg\max a_b}
\]

推荐的最简单对齐训练法是：

- 用 \(b^*=\arg\max_b(s_{k^*,b}-t_b)\) 作为伪标签  
- 用 **Top‑K 边界 hinge** 训练 key 网络  
- 用 **CE** 训练 query 网络把 top1 bin 对齐到 \(b^*\)

这能避免 softmax 份额带来的“概率值与排名不一致”的问题，并且梯度只作用在真正会影响 top1 推理命中的地方。
