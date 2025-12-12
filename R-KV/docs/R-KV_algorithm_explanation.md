# R-KV 算法详解：冗余 KV Cache 去除机制

> 基于 `/data/rbg/users/weian/project/rl/dc/R-KV` 代码库的分析

## 1. 算法核心思想

R-KV 使用**两个信号的组合**来评估哪些 KV 对应该被保留：
- **Attention Score（注意力分数）**：衡量 token 的重要性
- **Similarity Score（相似度分数）**：衡量 token 之间的冗余程度

最终公式 (`rkv/compression/r1_kv.py:79-81`)：
```python
final_score = attn_cache * mix_lambda - similarity_cos * (1 - mix_lambda)
```

**核心原则：注意力分数高 + 相似度低 = 保留优先级高**

---

## 2. 如何评估两个 K 是否冗余

核心实现在 `HuggingFace/rkv/utils.py:42-94` 的 `cal_similarity` 函数：

```python
def cal_similarity(key_states, threshold=0.5, retain_ratio=0.2, retain_direction="last"):
    k = key_states[0]  # shape: [num_heads, seq_len, head_dim]

    # Step 1: L2 归一化
    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)

    # Step 2: 计算余弦相似度矩阵 [num_heads, seq_len, seq_len]
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))

    # Step 3: 对角线置零（自己和自己不比较）
    for h in range(num_heads):
        similarity_cos[h].fill_diagonal_(0.0)

    # Step 4: 找到相似度超过阈值的token对
    similarity_mask = similarity_cos > threshold  # 默认 threshold=0.5
```

### 冗余判断逻辑

1. **余弦相似度计算**：对每个 head，计算所有 token 对之间的余弦相似度
2. **阈值过滤**：如果两个 K 向量的**余弦相似度 > 0.5**（默认阈值），则认为它们冗余
3. **方向选择**：对于每个 token，通过 `retain_direction` 参数决定保留最前/最后的相似 token
   - `"last"`：保留最后出现的相似 token
   - `"first"`：保留最先出现的相似 token
   - `"last_percent"`：保留最后 k% 的相似 token
   - `"first_percent"`：保留最先 k% 的相似 token

### 最终输出

```python
return similarity_cos.mean(dim=1).softmax(dim=-1)  # shape: [num_heads, seq_len]
```

返回每个 token 的**平均冗余分数**（经过 softmax 归一化），分数越高表示该 token 与其他 token 越相似（越冗余）。

---

## 3. 如何去除冗余 KV

核心实现在 `rkv/compression/r1_kv.py:39-183` 的 `update_kv` 方法：

```python
def update_kv(self, key_states, query_states, value_states):
    kv_cache_len = key_states.shape[-2]

    if kv_cache_len < self.budget:
        return key_states, value_states  # 未超过预算，不压缩

    # Step 1: 计算 attention scores
    attn_weights = compute_attention_scores(query_states, key_states)
    attn_weights_sum = F.softmax(
        attn_weights[:, :, -self.window_size:, :-self.window_size],
        dim=-1
    ).mean(dim=-2)  # 最近 window_size 个 token 对历史 token 的注意力

    # Step 2: Max Pooling 平滑注意力分数
    attn_cache = F.max_pool1d(
        attn_weights_sum,
        kernel_size=self.kernel_size,  # 默认 7
        padding=self.kernel_size // 2,
        stride=1
    )

    # Step 3: 计算相似度分数
    similarity_cos = cal_similarity(key_states, ...)[:, :-self.window_size]

    # Step 4: 组合两个分数
    final_score = attn_cache * mix_lambda - similarity_cos * (1 - mix_lambda)
    # mix_lambda 默认 0.1，即：10% attention + 90% (负)similarity

    # Step 5: TopK 选择保留的 token 索引
    indices = final_score.topk(self.budget - self.window_size, dim=-1).indices

    # Step 6: 使用 gather 提取保留的 KV
    k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
    v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)

    # Step 7: 拼接保留的历史 KV + 最近 window_size 个 KV
    key_states = torch.cat([k_past_compress, k_cur], dim=2)
    value_states = torch.cat([v_past_compress, v_cur], dim=2)

    return key_states, value_states
```

### 压缩流程详解

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 计算注意力分数 | 最近 `window_size` 个 query 对历史 key 的注意力 |
| 2 | Max Pooling | 使用滑动窗口（kernel_size=7）平滑注意力分数，捕获局部重要性 |
| 3 | 计算相似度分数 | 通过余弦相似度评估 token 之间的冗余程度 |
| 4 | 组合分数 | `final = attn * 0.1 - sim * 0.9`（注意力 10%，负相似度 90%） |
| 5 | TopK 选择 | 选择分数最高的 `budget - window_size` 个 token |
| 6 | Gather 操作 | 根据索引提取对应的 K 和 V |
| 7 | 拼接输出 | 压缩后的历史 KV + 最近 window_size 个 KV |

---

## 4. 关键参数解释

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `budget` | 2048 (配置文件) | KV cache 的最大容量 |
| `window_size` | 8 | 最近的 8 个 token 永远保留（不参与压缩） |
| `mix_lambda` | 0.1 | 注意力权重占比（0.1 = 注意力 10%，相似度 90%） |
| `kernel_size` | 7 | Max pooling 窗口大小，平滑注意力分数 |
| `retain_ratio` | 0.1 | 相似度计算时保留的比例 |
| `threshold` | 0.5 | 余弦相似度超过此值认为冗余 |
| `fp32_topk` | false | 是否使用 FP32 精度进行 TopK 计算 |

### 配置文件示例

来自 `weian_script/configs/sample8_rkv_aime24_official.yaml`：

```yaml
runner_args:
  method: rkv
  kv_budget: 2048
  attn_implementation: flash_attention_2
  load_dtype: bfloat16
  reset_cache_each_batch: false
  fp32_topk: false
  num_samples: 8
  temperature: 0.6
  top_p: 0.95
```

---

## 5. 算法流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    KV Cache 压缩流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: K, Q, V (seq_len > budget)                         │
│                                                             │
│  ┌──────────────────┐     ┌──────────────────┐             │
│  │  Attention Score │     │  Similarity Score │             │
│  │  (重要性)         │     │  (冗余度)          │             │
│  └────────┬─────────┘     └────────┬─────────┘             │
│           │                        │                        │
│           │ Q·K^T / √d            │ cos(k_i, k_j)          │
│           │                        │                        │
│           ▼                        ▼                        │
│  ┌──────────────────────────────────────────┐              │
│  │    Final Score = attn * λ - sim * (1-λ)  │              │
│  │    λ = 0.1 (注意力10%, 负相似度90%)       │              │
│  └────────────────────┬─────────────────────┘              │
│                       │                                     │
│                       ▼                                     │
│  ┌──────────────────────────────────────────┐              │
│  │         TopK 选择 (budget - window_size)  │              │
│  └────────────────────┬─────────────────────┘              │
│                       │                                     │
│                       ▼                                     │
│  ┌──────────────────────────────────────────┐              │
│  │   Gather 保留的 KV + 最近 window_size KV  │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
│  Output: 压缩后的 K, V (长度 = budget)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 直观理解

### 为什么这样做有效？

1. **低注意力分数的 token** → 对生成影响小 → 可以删除
2. **高相似度的 token** → 信息冗余 → 保留一个即可
3. **两者结合** → 保留"重要且独特"的 token，删除"不重要或重复"的 token

### 公式中的负号

```python
final_score = attn_cache * mix_lambda - similarity_cos * (1 - mix_lambda)
```

`- similarity_cos` 表示：
- 相似度越高 → 最终分数越低 → 越容易被删除
- 相似度越低 → 最终分数越高 → 越容易被保留

### 为什么 mix_lambda = 0.1？

- 默认设置中，相似度占 90% 的权重
- 这说明算法**优先去除冗余 token**，而不仅仅是去除不重要的 token
- 即使一个 token 注意力分数较低，如果它是独特的（与其他 token 不相似），仍可能被保留

---

## 7. 与其他方法的对比

R-KV 项目中还实现了其他 KV cache 压缩方法：

| 方法 | 文件 | 核心思想 |
|------|------|----------|
| **R-KV (R1KV)** | `r1_kv.py` | 注意力 + 相似度双信号 |
| SnapKV | `snapkv.py` | 基于注意力分数 |
| H2O | `h2o.py` | Heavy-Hitter Oracle |
| StreamingLLM | `streamingllm.py` | 保留初始 token + 最近 token |

R-KV 的独特之处在于引入了**相似度信号**来识别冗余 token，而不仅仅依赖注意力分数。

---

## 8. 源代码路径

- 主算法：`/data/rbg/users/weian/project/rl/dc/R-KV/rkv/compression/r1_kv.py`
- 工具函数：`/data/rbg/users/weian/project/rl/dc/R-KV/HuggingFace/rkv/utils.py`
- 运行脚本：`/data/rbg/users/weian/project/rl/dc/R-KV/weian_script/aime24_official_sampled8/run_rkv_aime24_official_sampled8.sh`
- 配置文件：`/data/rbg/users/weian/project/rl/dc/R-KV/weian_script/configs/sample8_rkv_aime24_official.yaml`
