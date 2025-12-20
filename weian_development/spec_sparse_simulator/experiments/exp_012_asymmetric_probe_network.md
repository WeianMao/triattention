# Exp 012: 非对称探针网络 - 基于距离的 Query 评分

## 背景

本文档是对 `exp_011_learnable_probe_vectors` 的改进方案的形式化描述。exp_011 使用带 RoPE 旋转的可学习探针向量来进行 Q/K 的 bin 分配。

### 当前架构 (exp_011)

**K 网络:**
- 探针向量: $\mathbf{P} \in \mathbb{R}^{B \times D}$ (B=128 bins, D=128 head_dim)
- 分数计算: $s_K^{(b)} = \mathbf{P}_b^\top \mathbf{K} + \text{bias}_b$

**Q 网络:**
- **独立的**探针向量: $\mathbf{P}^Q \in \mathbb{R}^{B \times D}$ (与 K 网络不共享)
- 分数计算: $s_Q^{(b)} = {\mathbf{P}^Q_b}^\top \mathbf{Q} + \text{bias}_b$

**核心问题:** Q 和 K 网络使用相同的点乘算法，但参数完全独立。这种设计缺乏参数共享，可能无法捕捉 Q 和 K 之间的几何关系。

---

## 重要说明：与 exp_011 的兼容性

**只有模型结构变化，其他保持不变：**
- Softmax 方向与 exp_011 完全一致：
  - K 侧: softmax over keys (dim=0)
  - Q 侧: softmax over bins (dim=-1)
- 训练流程、损失函数、数据加载等均不变
- 评估指标和评估流程不变

---

## 第一阶段: 基于距离的 Query 评分 + 共享探针

### 核心思想

1. **共享探针向量**: Q 和 K 网络使用同一组探针
2. **不同的评分算法**: K 侧使用点乘，Q 侧使用基于距离的特征

### 符号定义

- $B$: bin/探针数量 (默认: 128)
- $D$: head 维度 (默认: 128)
- $F$: 频段数量, $F = D/2$ (默认: 64)
- $\mathbf{P}_b \in \mathbb{R}^D$: 第 $b$ 个探针向量，可重塑为 $(F, 2)$ 进行逐频段操作
- $\mathbf{K} \in \mathbb{R}^D$: Key 向量 (经过 RoPE)
- $\mathbf{Q} \in \mathbb{R}^D$: Query 向量 (经过 RoPE)

### RoPE 旋转机制 (与 exp_011 保持一致)

探针向量 $\mathbf{P}$ 存储为**基础向量**（未旋转状态）。在前向传播时：

1. 计算参考位置: $\text{ref\_pos} = \text{round\_start} + \text{round\_window} / 2$
2. 将探针旋转到参考位置: $\mathbf{P}^{\text{rot}}_b = \text{RoPE}(\mathbf{P}_b, \text{ref\_pos})$
3. 输入的 K 和 Q 已经是 post-RoPE 的（由 LLM 旋转到各自的实际位置）

每一轮内，所有 K 和 Q 都使用同一组旋转后的探针进行计算。

### K 侧计算 (保持不变)

$$
s_K^{(b)} = {\mathbf{P}^{\text{rot}}_b}^\top \mathbf{K}_{\text{post}}
$$

其中 $\mathbf{P}^{\text{rot}}_b = \text{RoPE}(\mathbf{P}_b, \text{ref\_pos})$ 是旋转到参考位置的探针向量。

**注意:** K 侧偏置默认**不加**，是否添加作为消融实验选项。

### Q 侧计算 (新方法)

Q 侧同样使用旋转后的探针 $\mathbf{P}^{\text{rot}}_b$，但计算方式不同。

**步骤 1: 计算逐频段误差向量**

将 Q（post-RoPE）和旋转后的探针重塑为 $(F, 2)$ 的复数对形式:
$$
\mathbf{Q}_{\text{post}} \rightarrow \mathbf{Q}_{f} = (Q_{2f}, Q_{2f+1}) \in \mathbb{R}^2, \quad f = 0, 1, \ldots, F-1
$$
$$
\mathbf{P}^{\text{rot}}_b \rightarrow \mathbf{P}^{\text{rot}}_{b,f} = (P^{\text{rot}}_{b,2f}, P^{\text{rot}}_{b,2f+1}) \in \mathbb{R}^2
$$

每个频段的误差向量:
$$
\mathbf{e}_{b,f} = \mathbf{P}^{\text{rot}}_{b,f} - \mathbf{Q}_f \in \mathbb{R}^2
$$

**步骤 2: 计算误差向量的模长**
$$
d_{b,f} = \sqrt{\|\mathbf{e}_{b,f}\|_2^2 + \epsilon} = \sqrt{(P^{\text{rot}}_{b,2f} - Q_{2f})^2 + (P^{\text{rot}}_{b,2f+1} - Q_{2f+1})^2 + \epsilon}
$$

其中 $\epsilon = 10^{-8}$，用于保证数值稳定性。

这样对于每个探针 $b$，我们得到一个特征向量 $\mathbf{d}_b = (d_{b,0}, d_{b,1}, \ldots, d_{b,F-1}) \in \mathbb{R}^F$。

**步骤 3: 线性变换得到分数**

为确保权重为负值（距离越小分数越高），使用 -softplus 映射：
$$
\tilde{\mathbf{w}}_b = -\text{softplus}(\mathbf{w}_b^{\text{raw}}) = -\ln(1 + e^{\mathbf{w}_b^{\text{raw}}})
$$

最终分数：
$$
s_Q^{(b)} = \tilde{\mathbf{w}}_b^\top \mathbf{d}_b + c_b
$$

其中:
- $\mathbf{w}_b^{\text{raw}} \in \mathbb{R}^F$: 第 $b$ 个探针的**原始**可学习权重（未经映射）
- $\tilde{\mathbf{w}}_b \in \mathbb{R}^F$: 经过 -softplus 映射后的权重（保证为负）
- $c_b \in \mathbb{R}$: 第 $b$ 个探针的可学习偏置
- 每个探针有**自己独立的**线性层（不跨探针连接）
- 不同探针之间的线性层参数**不共享**

### 第一阶段参数量

| 组件 | 参数形状 | 数量 |
|------|----------|------|
| 共享探针 $\mathbf{P}$ | $B \times D$ | $128 \times 128 = 16,384$ |
| Q 线性层原始权重 $\mathbf{w}^{\text{raw}}$ | $B \times F$ | $128 \times 64 = 8,192$ |
| Q 线性层偏置 $c$ | $B$ | $128$ |
| **总计** | | **24,704** |

可选消融项:
| K 偏置（可选） | $B$ | $128$ |

对比 exp_011: $2 \times (16,384 + 128) = 33,024$ (两套独立的探针层)

### 第一阶段初始化

#### 探针向量 $\mathbf{P}$ 的 K-means 初始化（消融实验验证）

**核心思想：** 探针的基础向量需要在"相对空间"中聚类，因为前向传播时探针会被旋转到 ref_pos 后与 Q_post 比较。

**数学推导：**

点乘操作满足相对性：
$$
\text{RoPE}(\mathbf{P}, \text{ref\_pos}) \cdot \mathbf{Q}_{\text{post}} = \mathbf{P} \cdot \text{RoPE}(\mathbf{Q}_{\text{post}}, -\text{ref\_pos})
$$

因此，为了让 K-means 找到有意义的初始探针，应该在**相对空间**聚类：
$$
\mathbf{Q}_{\text{relative}} = \text{RoPE}(\mathbf{Q}_{\text{post}}, -\text{ref\_pos})
$$

**K-means 初始化流程：**

1. 遍历训练数据中所有 (Q, round) 样本
2. 对每个 Q：
   - 获取该轮的参考位置: $\text{ref\_pos} = \text{round\_start} + \text{round\_window} / 2$
   - 计算相对向量: $\mathbf{Q}_{\text{relative}} = \text{RoPE}(\mathbf{Q}_{\text{post}}, -\text{ref\_pos})$
3. 收集所有 $\mathbf{Q}_{\text{relative}}$ 向量
4. 运行 K-means 聚类得到 $B$ 个聚类中心
5. 用聚类中心初始化探针基础向量 $\mathbf{P}$

**代码伪代码：**
```python
def compute_kmeans_init(training_data):
    Q_relatives = []
    for Q_post, round_start, round_window in training_data:
        ref_pos = round_start + round_window // 2
        # 逆向旋转：相当于 RoPE(Q, -ref_pos)
        Q_relative = apply_rope_rotation(Q_post, -ref_pos)
        Q_relatives.append(Q_relative)

    Q_relatives = torch.stack(Q_relatives)  # (N, head_dim)

    # K-means 聚类
    kmeans = KMeans(n_clusters=num_bins)
    kmeans.fit(Q_relatives)

    return kmeans.cluster_centers_  # (num_bins, head_dim)
```

**需要消融实验验证：** K-means 初始化 vs. 随机初始化

#### 其他参数初始化

1. **K 偏置:** 默认不加（消融实验选项）

2. **Q 线性层原始权重 $\mathbf{w}_b^{\text{raw}}$:**
   - 目标：使映射后的权重 $\tilde{\mathbf{w}}_b = -\text{softplus}(\mathbf{w}_b^{\text{raw}}) = -1$
   - 推导：$\text{softplus}(w^{\text{raw}}) = 1 \Rightarrow \ln(1 + e^{w^{\text{raw}}}) = 1 \Rightarrow w^{\text{raw}} = \ln(e - 1) \approx 0.541$
   - **初始化值:** $\mathbf{w}_b^{\text{raw}} = \ln(e - 1) \approx 0.541$（对所有 $b, f$）

3. **Q 线性层偏置 $c_b$:** 初始化为 0

---

## 第二阶段: 模长感知评分

### 核心思想

将模长信息作为额外特征加入 K 和 Q 的评分计算。

### K 侧增强

**K 的逐频段模长:**
$$
m_f^K = \|\mathbf{K}_f\|_2 = \sqrt{K_{2f}^2 + K_{2f+1}^2}, \quad f = 0, \ldots, F-1
$$

**新的 K 分数:**
$$
s_K^{(b)} = \underbrace{{\mathbf{P}^{\text{rot}}_b}^\top \mathbf{K}}_{\text{点乘项}} + \underbrace{\mathbf{u}_b^\top \mathbf{m}^K}_{\text{模长项}}
$$

其中:
- $\mathbf{m}^K = (m_0^K, m_1^K, \ldots, m_{F-1}^K) \in \mathbb{R}^F$
- $\mathbf{u}_b \in \mathbb{R}^F$: 第 $b$ 个探针的模长权重（K 侧）

**注意:** K 侧偏置默认**不加**，可作为后续消融实验选项。

### Q 侧增强（默认不启用）

**注意:** Q 侧模长增强默认**不启用**，可作为后续消融实验选项。

**Q 的逐频段模长:**
$$
m_f^Q = \|\mathbf{Q}_f\|_2 = \sqrt{Q_{2f}^2 + Q_{2f+1}^2}, \quad f = 0, \ldots, F-1
$$

**启用时的 Q 分数:**
$$
s_Q^{(b)} = \underbrace{\tilde{\mathbf{w}}_b^\top \mathbf{d}_b}_{\text{距离项}} + \underbrace{\mathbf{v}_b^\top \mathbf{m}^Q}_{\text{模长项}} + c_b
$$

其中:
- $\mathbf{m}^Q = (m_0^Q, m_1^Q, \ldots, m_{F-1}^Q) \in \mathbb{R}^F$
- $\mathbf{v}_b \in \mathbb{R}^F$: 第 $b$ 个探针的模长权重（Q 侧）
- $\mathbf{v}_b$ 和 $\mathbf{u}_b$ 在 Q 和 K 之间**不共享**

### 第二阶段额外参数量

| 组件 | 参数形状 | 数量 |
|------|----------|------|
| K 模长权重 $\mathbf{u}$ | $B \times F$ | $128 \times 64 = 8,192$ |
| Q 模长权重 $\mathbf{v}$ | $B \times F$ | $128 \times 64 = 8,192$ |
| **第二阶段额外** | | **16,384** |
| **第一阶段 + 第二阶段总计** | | **41,216** |

### 第二阶段初始化

1. **K 模长权重 $\mathbf{u}_b$:** 初始化为 0
2. **Q 模长权重 $\mathbf{v}_b$:** 初始化为 0
3. **K 偏置:** 初始化为 0

**设计原则:** 初始化为 0 使第二阶段一开始等效于第一阶段，然后逐步学习模长信息的贡献。

---

## 算法流程总结

### 训练/推理过程

**输入:**
- K 向量 $(N, D)$：post-RoPE，已旋转到各自的实际位置
- Q 向量 $(M, D)$：post-RoPE，已旋转到各自的实际位置
- round_start, round_window：用于计算参考位置

**每轮开始时:**
1. 计算参考位置: $\text{ref\_pos} = \text{round\_start} + \text{round\_window} / 2$
2. 将探针基础向量旋转到参考位置: $\mathbf{P}^{\text{rot}} = \text{RoPE}(\mathbf{P}, \text{ref\_pos})$

**K 处理 (每轮一次):**
1. 使用旋转后的探针 $\mathbf{P}^{\text{rot}}$
2. 计算每个 key 的分数: $s_K^{(b)} = {\mathbf{P}^{\text{rot}}_b}^\top \mathbf{K}_n$
3. （可选-第二阶段）加入模长项: $s_K^{(b)} = {\mathbf{P}^{\text{rot}}_b}^\top \mathbf{K}_n + \mathbf{u}_b^\top \mathbf{m}^K_n$
4. 对 keys 做 softmax (dim=0)

**Q 处理 (每个 query):**
1. 使用同样的旋转后探针 $\mathbf{P}^{\text{rot}}$
2. 计算逐频段距离（注意 $\epsilon$）: $d_{b,f} = \sqrt{\|\mathbf{P}^{\text{rot}}_{b,f} - \mathbf{Q}_f\|^2 + \epsilon}$
3. 计算有效权重: $\tilde{\mathbf{w}}_b = -\text{softplus}(\mathbf{w}_b^{\text{raw}})$
4. 计算 Q 分数: $s_Q^{(b)} = \tilde{\mathbf{w}}_b^\top \mathbf{d}_b + c_b$
5. （可选-第二阶段）加入模长项: $s_Q^{(b)} = \tilde{\mathbf{w}}_b^\top \mathbf{d}_b + \mathbf{v}_b^\top \mathbf{m}^Q + c_b$
6. 对 bins 做 softmax (dim=-1)

---

## 待确认问题 / 潜在问题

### 1. 数值稳定性（重要）

**距离范数的梯度问题:**
- $d_{b,f} = \|\mathbf{e}_{b,f}\|_2$ 在 $\|\mathbf{e}\| \rightarrow 0$ 时梯度不稳定（除以零）
- **解决方案:** 使用 $d_{b,f} = \sqrt{\|\mathbf{e}_{b,f}\|_2^2 + \epsilon}$，其中 $\epsilon = 10^{-8}$

**实现提醒:**
> ⚠️ 实现时务必注意数值稳定性问题。涉及 sqrt、norm、除法等操作时，都需要考虑加 $\epsilon$ 防止梯度爆炸或 NaN。

### 2. 第二阶段模长项的解释
- 对于 K: 模长表示每个频率分量的"强度"
- 加入模长可能有助于区分方向相似但模长不同的 key
- **消融实验:** 先训练第一阶段，再加入第二阶段来衡量增量收益

### 3. 线性层的独立性
- 每个探针有独立的权重: Q 侧线性层共 $(B \times F)$ 个参数
- 不共享参数意味着探针之间无法泛化
- **权衡:** 更强的表达能力 vs. 更多的参数需要学习

---

## 消融实验计划

1. **基线:** exp_011 (独立 Q/K 探针 + 点乘)
2. **仅第一阶段:** 共享探针 + 基于距离的 Q 评分（无 K 偏置）
3. **第一阶段 + K 偏置:** 在 K 侧加入可学习偏置
4. **第一阶段 + 模长 (仅 Q 侧):** 在 Q 侧加入模长项
5. **第一阶段 + 模长 (仅 K 侧):** 在 K 侧加入模长项
6. **第一阶段 + 第二阶段 (完整):** Q 和 K 侧都加入模长项
7. **初始化对比:** K-means 初始化 vs. 随机初始化

---

## 当前实现范围（第一阶段默认配置）

**本次只实现第一阶段的默认配置：**
- 共享探针向量（Q/K 共用同一组基础向量）
- K 侧：探针点乘（无偏置）
- Q 侧：基于距离的评分 + -softplus 映射的线性层
- 随机初始化（K-means 初始化作为后续消融选项）
- 注意数值稳定性：距离计算使用 $\sqrt{\|\cdot\|^2 + \epsilon}$

---

## 后续可扩展功能（消融实验选项）

以下功能**暂不实现**，作为后续消融实验选项，方便与同事沟通：

| 功能 | 描述 | 优先级 |
|------|------|--------|
| K-means 初始化 | 在相对空间聚类初始化探针 | 高 |
| K 侧偏置 | 在 K 分数计算中加入可学习偏置 | 中 |
| K 侧模长项（第二阶段） | 加入 $\mathbf{u}_b^\top \mathbf{m}^K$ | 中 |
| Q 侧模长项（第二阶段） | 加入 $\mathbf{v}_b^\top \mathbf{m}^Q$ | 低 |
| K 侧偏置（第二阶段） | 第二阶段的 K 偏置 | 低 |
