# SpecKV + Similarity Deduplication: 算法设计与开发要求

> 作者: weian
> 日期: 2025-12-11
> 状态: 待开发

---

## 1. 背景与动机

### 1.1 问题描述

当前 SpecKV 算法使用 **frequency scoring**（频率预测）来评估 KV cache 中每个 token 的重要性，然后通过 TopK 选择保留最重要的 token。这个设计类似于 R-KV 中的 **Attention Score** 组件。

然而，SpecKV **缺少去除冗余的能力**。R-KV 算法使用了 **Similarity Score** 来识别冗余 token：如果两个 key 向量非常相似（余弦相似度高），说明它们携带的信息是冗余的，只需要保留一个。

### 1.2 算法对比

| 组件 | R-KV | SpecKV (当前) | SpecKV (目标) |
|------|------|--------------|---------------|
| **重要性评估** | Attention Score | Frequency Score | Frequency Score |
| **冗余去除** | Similarity Score | ❌ 无 | ✅ Similarity Score |
| **最终分数** | `attn * λ - sim * (1-λ)` | `freq_score` | `freq * λ - sim * (1-λ)` |

### 1.3 核心思想

将 R-KV 的 Similarity Score 组件整合到 SpecKV 中，使得 SpecKV 同时具备：
1. **预测未来重要性**（frequency scoring）
2. **去除冗余 token**（similarity scoring）

---

## 2. R-KV 参考资料（必读）

> **开发前必须先理解 R-KV 的 Similarity Deduplication 是如何工作的！**

### 2.1 R-KV 算法入口

**运行脚本**：
```
R-KV/weian_script/aime24_official_sampled8/run_rkv_aime24_official_sampled8.sh
```

这是 R-KV 冗余去除算法的**算法入口**，通过这个脚本可以运行完整的 R-KV 实验。

### 2.2 R-KV 算法详解文档

**详细解释文档**：
```
R-KV/docs/R-KV_algorithm_explanation.md
```

这个文档包含：
- Similarity Score 的计算原理（余弦相似度）
- `cal_similarity` 函数的详细解释
- `final_score = attn * λ - sim * (1-λ)` 公式的来源
- 所有超参数（`threshold`, `retain_ratio`, `retain_direction` 等）的含义
- 完整的压缩流程图

**强烈建议**：在开始编码前，务必完整阅读 `R-KV_algorithm_explanation.md`，确保理解 Similarity Score 的工作机制。

---

## 3. 算法设计

### 3.1 对齐 R-KV 的设计（方案 1）

为了最大程度对齐 R-KV 的做法，我们采用 **单阶段组合分数** 的方式：

```python
# R-KV 原始公式
final_score = attn_cache * mix_lambda - similarity_cos * (1 - mix_lambda)

# SpecKV + Similarity 的公式
final_score = freq_score * mix_lambda - similarity_cos * (1 - mix_lambda)
```

**核心原则**：
- `freq_score` 高 + `similarity` 低 = 保留优先级高（重要且独特）
- `freq_score` 低 或 `similarity` 高 = 保留优先级低（不重要或冗余）

### 3.2 Similarity Score 计算

直接复用 R-KV 的 `cal_similarity` 函数（位于 `HuggingFace/rkv/utils.py`）：

```python
def cal_similarity(key_states, threshold=0.5, retain_ratio=0.2, retain_direction="last"):
    k = key_states[0]  # shape: [num_heads, seq_len, head_dim]

    # L2 归一化
    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)

    # 余弦相似度矩阵
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))

    # 对角线置零
    for h in range(num_heads):
        similarity_cos[h].fill_diagonal_(0.0)

    # 阈值过滤 + 方向选择
    similarity_mask = similarity_cos > threshold
    # ... (详见 R-KV 实现)

    return similarity_cos.mean(dim=1).softmax(dim=-1)  # [num_heads, seq_len]
```

### 3.3 计算复杂度分析

Similarity 计算是 O(n²)，但由于 KV budget 有限，实际开销可接受：

| budget | O(n²) 计算量 | 评估 |
|--------|-------------|------|
| 2048 | ~4M | 可忽略 |
| 4096 | ~16M | 很小 |
| 8192 | ~67M | 可接受 |

核心操作是矩阵乘法，GPU 上非常快。

---

## 4. 开发要求

### 4.1 兼容性要求（重要）

> **绝对不能修改或破坏已有实验**

开发过程中必须保证以下脚本的行为**完全不变**：
- `R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8_norm.sh`
- 其他现有的 SpecKV / R-KV 实验脚本

**实现方式**：通过 **新增参数开关** 控制是否启用 Similarity Deduplication

```yaml
# 默认关闭（兼容现有实验）
sparse_use_similarity: false

# 打开时启用去冗余
sparse_use_similarity: true
sparse_similarity_mix_lambda: 0.1  # 需要搜索的超参
```

### 4.2 参数设计

#### 4.2.1 需要新增的参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `sparse_use_similarity` | bool | `false` | 是否启用 Similarity Deduplication |
| `sparse_similarity_mix_lambda` | float | `0.1` | freq_score 的权重（需要搜索） |

#### 4.2.2 超参数处理策略

| 超参数 | 来源 | 处理方式 |
|--------|------|----------|
| `mix_lambda` | R-KV 默认 0.1 | **需要搜索**：建议实验 0.1, 0.3, 0.5, 0.7, 0.9 |
| `threshold` | R-KV 默认 0.5 | **直接沿用**：不搜索，使用 R-KV 默认值 |
| `retain_ratio` | R-KV 默认 0.1 | **直接沿用**：不搜索 |
| `retain_direction` | R-KV 默认 "last" | **直接沿用**：不搜索 |
| `kernel_size` | R-KV 默认 7 | **不适用**：这是 R-KV attention smoothing 用的，SpecKV 不需要 |

> **注意**：如果在实现过程中发现还有其他需要设置的超参数，请**立即联系 weian 确认处理方式**，不要自行决定。

### 4.3 代码修改范围

主要修改文件：`weian_development/speckv/sparse_round_pruner_prefill_keep.py`

修改位置：`_select_keep_indices` 方法

```python
def _select_keep_indices(self, past_key_values, key_positions, keep_count, start_index=0):
    # 现有代码：计算 frequency scores
    per_head_scores = self._compute_head_scores(...)
    combined = per_head_scores.max(dim=0).values

    # ===== 新增：Similarity Deduplication =====
    if self.use_similarity:
        # 提取 key states
        key_states = self._extract_key_states(past_key_values, key_positions, start_index)

        # 计算 similarity scores（复用 R-KV 的 cal_similarity）
        similarity_cos = cal_similarity(
            key_states,
            threshold=0.5,        # 沿用 R-KV 默认值
            retain_ratio=0.1,     # 沿用 R-KV 默认值
            retain_direction="last"
        )

        # 组合分数
        combined = combined * self.similarity_mix_lambda - similarity_cos * (1 - self.similarity_mix_lambda)
    # ===== 新增结束 =====

    # 现有代码：TopK 选择
    ...
```

### 4.4 配置文件与脚本

#### 4.4.1 新增配置文件

创建 `R-KV/weian_script/configs/sample8_speckv_similarity_aime24_official.yaml`：

```yaml
experiment:
  name: sample8_speckv_similarity_aime24_official
  # ... (复制 sample8_sparseprefillkeep_aime24_official.yaml 的内容)
  runner_args:
    # ... (现有参数)
    sparse_use_similarity: true
    sparse_similarity_mix_lambda: 0.1  # 或其他要搜索的值
```

#### 4.4.2 新增运行脚本

为不同的 `mix_lambda` 值创建对应脚本：

```
R-KV/weian_script/aime24_official_sampled8/
├── run_speckv_similarity_lambda01.sh  # mix_lambda=0.1
├── run_speckv_similarity_lambda03.sh  # mix_lambda=0.3
├── run_speckv_similarity_lambda05.sh  # mix_lambda=0.5
├── run_speckv_similarity_lambda07.sh  # mix_lambda=0.7
└── run_speckv_similarity_lambda09.sh  # mix_lambda=0.9
```

---

## 5. 实验设计

### 5.1 Baseline

实验对比的 baseline 是：
```
R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8_norm.sh
```

即**启用了 score normalization 的 SpecKV**（`--sparse-normalize-scores`）。

### 5.2 实验矩阵

| 实验名 | use_similarity | mix_lambda | 对比目的 |
|--------|----------------|------------|----------|
| baseline | false | - | 原始 SpecKV |
| sim_λ=0.1 | true | 0.1 | 接近 R-KV 默认 |
| sim_λ=0.3 | true | 0.3 | 更重视 freq_score |
| sim_λ=0.5 | true | 0.5 | 平衡 |
| sim_λ=0.7 | true | 0.7 | 更重视 freq_score |
| sim_λ=0.9 | true | 0.9 | 几乎只用 freq_score |

### 5.3 评估指标

使用与现有实验相同的评估方式（AIME24 准确率）。

---

## 6. 注意事项

### 6.1 Checklist

开发完成前请确认：

- [ ] `run_speckv_aime24_official_sampled8_norm.sh` 运行结果与修改前完全一致
- [ ] 新参数 `sparse_use_similarity=false` 时行为与原来一致
- [ ] 新参数 `sparse_use_similarity=true` 时正确启用 Similarity Deduplication
- [ ] `cal_similarity` 函数正确导入并复用
- [ ] 所有新增超参数都有合理的默认值
- [ ] 创建了所有 `mix_lambda` 搜索实验的脚本
- [ ] 如发现未列出的超参数，已联系 weian 确认

### 6.2 参考文件

- **R-KV 算法入口**：`R-KV/weian_script/aime24_official_sampled8/run_rkv_aime24_official_sampled8.sh`
- **R-KV 算法解释**：`R-KV/docs/R-KV_algorithm_explanation.md`（必读！）
- R-KV Similarity 实现：`R-KV/HuggingFace/rkv/utils.py` → `cal_similarity()`
- R-KV 主算法：`R-KV/rkv/compression/r1_kv.py` → `R1KV.update_kv()`
- SpecKV 主算法：`R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py`

### 6.3 联系方式

如有任何疑问，特别是：
- 发现未列出的超参数需要设置
- 实现过程中遇到设计决策问题
- 不确定某个行为是否会影响已有实验

请**立即联系 weian** 确认，不要自行决定。

---

## 7. 附录：R-KV cal_similarity 函数签名

```python
def cal_similarity(
    key_states,           # shape: [batch, num_heads, seq_len, head_dim]
    threshold=0.5,        # 余弦相似度阈值
    retain_ratio=0.2,     # 保留比例
    retain_direction="last",  # 保留方向: "last", "first", "last_percent", "first_percent"
) -> torch.Tensor:        # shape: [num_heads, seq_len]
    """
    计算 key 之间的相似度分数。
    返回每个 token 的平均冗余分数（经过 softmax 归一化）。
    分数越高表示该 token 与其他 token 越相似（越冗余）。
    """
```
