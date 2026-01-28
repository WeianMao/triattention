# Baseline Algorithms

## 1. R-KV (Redundancy-aware KV Cache)

### 核心思想

R-KV使用**两个信号的组合**来评估哪些KV对应该被保留：
- **Attention Score（注意力分数）**: 衡量token的重要性
- **Similarity Score（相似度分数）**: 衡量token之间的冗余程度

**核心原则**: 注意力分数高 + 相似度低 = 保留优先级高

### 评分公式

```python
final_score = attn_cache * mix_lambda - similarity_cos * (1 - mix_lambda)
```

其中 `mix_lambda` 默认为0.1，即：
- 注意力占10%
- 负相似度占90%（相似度越高，分数越低）

这说明算法**优先去除冗余token**，而不仅仅是去除不重要的token。即使一个token注意力分数较低，如果它是独特的（与其他token不相似），仍可能被保留。

### 相似度计算

```python
def cal_similarity(key_states, threshold=0.5):
    # L2归一化
    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)

    # 余弦相似度矩阵
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))

    # 对角线置零（自己和自己不比较）
    similarity_cos.fill_diagonal_(0.0)

    # 返回平均冗余分数
    return similarity_cos.mean(dim=1).softmax(dim=-1)
```

### 压缩流程

1. 计算注意力分数：最近`window_size`个query对历史key的注意力
2. Max Pooling平滑：使用滑动窗口(kernel_size=7)捕获局部重要性
3. 计算相似度分数：通过余弦相似度评估token冗余
4. 组合分数：`final = attn * 0.1 - sim * 0.9`
5. TopK选择：选择分数最高的`budget - window_size`个token
6. Gather操作：根据索引提取对应的K和V
7. 拼接输出：压缩后的历史KV + 最近window_size个KV

### 关键参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `budget` | 2048 | KV cache的最大容量 |
| `window_size` | 8 | 最近的8个token永远保留 |
| `mix_lambda` | 0.1 | 注意力权重占比 |
| `kernel_size` | 7 | Max pooling窗口大小 |

### 代码位置

- 主算法：`R-KV/rkv/compression/r1_kv.py`
- 工具函数：`R-KV/HuggingFace/rkv/utils.py`

---

## 2. FullKV (No Compression Baseline)

### 核心思想

FullKV**不进行任何压缩**，保留所有KV cache。作为评估压缩算法性能损失的baseline。

### 配置特点

```yaml
method: fullkv
kv_budget: null  # 无限制
```

### 用途

- 衡量其他压缩算法的性能损失
- 提供upper bound参考
- 测量在无内存限制下的模型能力

---

## 3. 算法对比表

| 特性 | SpecKV | R-KV | FullKV |
|------|--------|------|--------|
| 压缩方式 | 频率域预测 | 注意力+相似度 | 无压缩 |
| 计算来源 | 预计算统计 | 实时计算 | N/A |
| Prefill处理 | 保留 | 保留 | 保留 |
| Per-head支持 | 是 | 否（全局） | N/A |
| 内存使用 | budget固定 | budget固定 | 无限增长 |

---

## 4. R-KV框架中的其他方法

项目中还实现了其他KV cache压缩方法：

| 方法 | 文件 | 核心思想 |
|------|------|----------|
| SnapKV | `rkv/compression/snapkv.py` | 基于注意力分数的裁剪 |
| H2O | `rkv/compression/h2o.py` | Heavy-Hitter Oracle |
| StreamingLLM | `rkv/compression/streamingllm.py` | 保留初始token + 最近token |

这些方法都采用类似的forward patch机制集成到HF generate流程中。
