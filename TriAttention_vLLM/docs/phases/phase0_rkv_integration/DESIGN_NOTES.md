# Phase 0 设计补充说明（vLLM 集成）

基于 R-KV/vLLM 代码分析的技术细节和设计决策。

---

## 1. R-KV/vLLM 压缩架构详解

### 1.1 R1KV 在 vLLM v1 中的集成方式

**文件**：`R-KV/vLLM/vllm/v1/attention/backends/flash_attn.py`

```python
class FlashAttentionImpl:
    def __init__(self, ...):
        # 行 431：初始化压缩器
        self.kvcompressor = R1KV(budget=VLLM_V1_R_KV_BUDGET)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, ...):
        # 行 547-588：压缩逻辑
        for i in range(attn_metadata.num_reqs):
            if attn_metadata.seq_lens[i] < VLLM_V1_R_KV_BUDGET + VLLM_V1_R_KV_BUFFER:
                continue  # 序列太短，不压缩

            # 提取当前序列的 KV cache
            current_key_cache = key_cache.view(...)[occupied_slot_mapping[...], ...]
            current_value_cache = value_cache.view(...)[occupied_slot_mapping[...], ...]

            # 调用压缩器
            compressed_key, compressed_value = self.kvcompressor.update_kv(
                current_key_cache,
                current_query,
                current_value_cache,
            )

            # 回写压缩后的 KV
            key_cache.view(...)[...] = compressed_key.transpose(0, 1)
            value_cache.view(...)[...] = compressed_value.transpose(0, 1)

            # 记录压缩统计
            num_dropped = current_kv_len - compressed_kv_len
            attn_metadata.num_dropped_tokens_list[i] = num_dropped
```

### 1.2 触发条件

```python
# vLLM/vllm/envs.py
VLLM_V1_R_KV_BUDGET: int = 64   # 保留的 token 数
VLLM_V1_R_KV_BUFFER: int = 64   # 触发压缩的缓冲区

# 触发条件
if seq_len >= VLLM_V1_R_KV_BUDGET + VLLM_V1_R_KV_BUFFER:
    # 执行压缩
```

**含义**：当序列长度 >= 128 (64+64) 时触发压缩，压缩后保留 64 个 token。

### 1.3 KV Cache 布局

vLLM 使用 PagedAttention，KV cache 是分块存储的：

```python
# kv_cache 形状：[num_blocks, block_size, num_heads, head_dim]
# 通过 slot_mapping 访问特定序列的 token

# 提取当前序列的 KV
current_key = key_cache.view(num_blocks * block_size, num_heads, head_dim)
current_key = current_key[occupied_slot_mapping[start:end], ...]
# 结果形状：[seq_len, num_heads, head_dim]

# 需要转置为 R1KV 期望的格式
current_key = current_key.transpose(0, 1).unsqueeze(0)
# 结果形状：[1, num_heads, seq_len, head_dim]
```

---

## 2. R1KV 接口分析

### 2.1 update_kv() 接口

**文件**：`R-KV/rkv/compression/r1_kv.py`

```python
def update_kv(
    self,
    key_states: torch.Tensor,      # [batch, num_kv_heads, seq_len, head_dim]
    query_states: torch.Tensor,    # [batch, num_heads, seq_len, head_dim]
    value_states: torch.Tensor,    # [batch, num_kv_heads, seq_len, head_dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回压缩后的 (key_states, value_states)
    输出形状：[batch, num_kv_heads, compressed_len, head_dim]
    其中 compressed_len <= self.budget
    """
```

### 2.2 R1KV 算法核心

```python
# 1. 计算注意力权重
attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
attn_cache = attn_weights.mean(dim=-2)  # [batch, heads, seq_len]

# 2. 计算 token 相似度（用于去重）
similarity = cal_similarity(key_states)

# 3. 混合评分
final_score = attn_cache * mix_lambda - similarity * (1 - mix_lambda)

# 4. TopK 选择
keep_indices = final_score.topk(budget - window_size, dim=-1).indices

# 5. 保留最近 window_size 个 token
k_past = key_states.gather(dim=2, index=keep_indices)
k_cur = key_states[:, :, -window_size:, :]
key_states = torch.cat([k_past, k_cur], dim=2)
```

---

## 3. SpeckV vLLM 适配要点

### 3.1 与 R1KV 的关键区别

| 方面 | R1KV | SpeckV |
|-----|------|--------|
| 评分依据 | attention + similarity | 频率统计 |
| 是否需要 query | 是 | 否（使用预计算统计） |
| 额外输入 | 无 | stats 文件 |
| 窗口保留 | 是（window_size） | 可选 |
| RoPE 处理 | 无 | 需要 RoPE 反演 |

### 3.2 SpeckV 需要适配的地方

1. **Stats 加载**
   ```python
   def __init__(self, budget, stats_path, ...):
       self.stats = torch.load(stats_path)
       # stats 包含：q_mean_complex, freq_scale_sq, sampled_heads 等
   ```

2. **避免 RoPE 反演**（重要优化）
   - **不需要**对 key 做 RoPE 反演
   - 直接用存储的 $K_{rot}$ 计算，通过相位校正等价
   - 详见 `docs/design/optimization.md` 第 2 节
   ```python
   # 优化后：直接用 K_rot 计算
   Re = q_r * k_r + q_i * k_i
   Im = q_i * k_r - q_r * k_i
   A_coef = freq_scale_sq * Re  # 无需三角函数
   B_coef = freq_scale_sq * Im
   ```

3. **忽略 query_states**
   - SpeckV 不使用 query 打分，但接口需要保持兼容
   ```python
   def update_kv(self, key_states, query_states, value_states):
       # query_states 被忽略，保持接口兼容
       scores = self._compute_scores(key_states)  # 不使用 query
       ...
   ```

### 3.3 位置信息解决方案

> 详细设计见 `docs/implementation/data_structures.md` 和 `docs/implementation/fill_in_place.md`

**问题**：SpeckV 打分需要知道每个 token 的绝对位置，但 KV cache 压缩后 token 变成乱序。

**解决方案**：维护 `position_indices` tensor

```python
# 与 KV cache 同步存储，形状与 PagedAttention 对齐
position_indices: torch.Tensor  # [num_blocks, block_size]

# 每个 token 一个位置值，记录其原始序列位置
# 例如压缩后：position_indices = [0, 1, 16385, 4, 32769, ...]
#                                  ↑  ↑  ↑      ↑  ↑
#                                  原始位置（可能乱序）
```

**更新时机**：
| 事件 | position_indices 操作 |
|-----|----------------------|
| Prefill | 初始化为 `[0, 1, 2, ..., prefill_len-1]` |
| Decode | 追加新位置 `append(current_seq_len)` |
| Prune（回填） | 保留的 token 位置不变，新位置填入空槽 |

**显存开销**：极小（< 0.04% of KV cache），详见 `docs/implementation/data_structures.md` 第 3 节

### 3.4 打分函数优化

> 详细设计见 `docs/design/optimization.md`

Phase 0 应实现以下优化（PyTorch 版本），为 Phase 1 Triton 实现打基础：

**优化 1：避免 RoPE 反演**（已在 3.2 说明）

**优化 2：单次读取多位置打分**
```python
# 只加载 K 一次，在寄存器中迭代所有 offset
k = load_from_memory()  # 只读取一次
A_coef, B_coef = compute_coefficients(k)

for offset in offsets:  # 在寄存器中迭代，无显存访问
    score[offset] = dot(A_coef, C[offset]) - dot(B_coef, S[offset]) + extra_term
```

**优化 3：共享三角函数查找表（LUT）**
```python
# 每轮打分开始时预计算（所有 token 共享）
C = cos(offsets * omega)  # [num_offsets, freq_count]
S = sin(offsets * omega)  # [num_offsets, freq_count]

# 显存占用仅 ~4 KB
```

**批量矩阵形式**（GPU 友好）：
$$\mathbf{Scores} = \mathbf{A}_{all} \cdot \mathbf{C}^T - \mathbf{B}_{all} \cdot \mathbf{S}^T + \mathbf{E}$$

---

## 4. GQA 处理

### 4.1 背景

Qwen2.5-7B 使用 GQA：
- `num_attention_heads` = 28
- `num_key_value_heads` = 4
- `num_key_value_groups` = 7

### 4.2 vLLM 中的 GQA 处理

```python
# flash_attn.py 中，KV cache 使用 num_kv_heads
key_states.shape = [batch, num_kv_heads, seq_len, head_dim]  # [1, 4, seq, 128]

# query 使用 num_attention_heads
query_states.shape = [batch, num_heads, seq_len, head_dim]   # [1, 28, seq, 128]
```

### 4.3 SpeckV 的 GQA 适配

SpeckV 的 `sampled_heads` 是 attention head 索引，需要映射到 KV heads：

```python
# sampled_heads: [(layer, attn_head), ...]
# 需要映射到 kv_head

def map_to_kv_head(attn_head, num_kv_groups):
    return attn_head // num_kv_groups

# 例如：attn_head=14 -> kv_head=14//7=2
```

---

## 5. 环境变量设计

### 5.1 新增环境变量

| 变量名 | 默认值 | 说明 |
|-------|--------|------|
| `VLLM_COMPRESSION_ALGO` | `"r1kv"` | 压缩算法选择 |
| `VLLM_SPECKV_STATS_PATH` | `None` | SpeckV stats 文件路径 |
| `VLLM_SPECKV_PRUNING_MODE` | `"per_head"` | pruning 模式 |

### 5.2 在 envs.py 中添加

```python
# vLLM/vllm/envs.py

VLLM_COMPRESSION_ALGO: str = "r1kv"
VLLM_SPECKV_STATS_PATH: Optional[str] = None
VLLM_SPECKV_PRUNING_MODE: str = "per_head"

environment_variables: Dict[str, Callable[[], Any]] = {
    ...
    "VLLM_COMPRESSION_ALGO": lambda: os.getenv("VLLM_COMPRESSION_ALGO", "r1kv"),
    "VLLM_SPECKV_STATS_PATH": lambda: os.getenv("VLLM_SPECKV_STATS_PATH", None),
    "VLLM_SPECKV_PRUNING_MODE": lambda: os.getenv("VLLM_SPECKV_PRUNING_MODE", "per_head"),
}
```

---

## 6. 测试策略

### 6.1 单元测试优先级

1. **接口兼容性**：`update_kv()` 输入输出格式正确
2. **边界条件**：`seq_len <= budget` 时直接返回
3. **压缩比**：输出长度 <= budget
4. **数值正确性**：与 HF 路径对比

### 6.2 与 HF 路径对比验证

```python
def test_consistency_with_hf():
    """验证 vLLM 路径与 HF 路径结果一致"""
    # 准备相同的输入
    key = torch.randn(1, 4, 1000, 128)
    value = torch.randn(1, 4, 1000, 128)

    # HF 路径
    hf_compressor = SpeckVRKVStyle(...)
    hf_key, hf_value = hf_compressor.compress(key, value)

    # vLLM 路径
    vllm_compressor = SpeckVvLLM(...)
    vllm_key, vllm_value = vllm_compressor.update_kv(key, None, value)

    # 对比
    assert torch.allclose(hf_key, vllm_key, rtol=1e-4)
    assert torch.allclose(hf_value, vllm_value, rtol=1e-4)
```

---

## 7. 已知限制

### 7.1 Phase 0 限制

| 限制 | 原因 | Phase 1 解决方案 |
|-----|------|-----------------|
| 无 Triton 优化 | Phase 0 重点是正确性 | Phase 1 实现 Triton kernel |
| 仅支持 per_head 模式 | per_layer 需要跨层信息 | Phase 1 重新设计接口 |

### 7.2 与 HF 路径的功能差异

| 功能 | HF 路径 | vLLM Phase 0 |
|-----|--------|--------------|
| per_head pruning | ✓ | ✓ |
| per_layer pruning | ✓ | ✗（需跨层信息） |
| per_layer_perhead | ✓ | ✗（需跨层信息） |
| 位置追踪 | ✓ | ✓（通过 position_indices） |
| prefill 保护 | ✓ | 待实现 |
| 打分优化（LUT） | ✗ | ✓（Phase 0 即实现） |

---

## 8. 文件修改检查清单

### 8.1 新建文件

- [ ] `rkv/compression/speckv_vllm.py` - SpeckV vLLM 接口实现
- [ ] `rkv/compression/position_tracker.py` - position_indices 管理（可选，可合并到 speckv_vllm.py）
- [ ] `tests/test_speckv_vllm.py` - 单元测试

### 8.2 修改文件

- [ ] `rkv/compression/__init__.py` - 导出 SpeckVvLLM
- [ ] `vLLM/vllm/v1/attention/backends/flash_attn.py` - 算法选择分支 + position_indices 更新
- [ ] `vLLM/vllm/envs.py` - 环境变量

### 8.3 关键数据结构

- [ ] `position_indices`: `[num_blocks, block_size]` 与 KV cache 对齐
- [ ] `trig_cos/sin`: `[num_offsets, freq_count]` 共享三角表（~4 KB）
- [ ] Stats 从文件加载：`q_mean_complex`, `freq_scale_sq`, `omega` 等

### 8.3 验证默认行为不变

```bash
# 不设置 VLLM_COMPRESSION_ALGO，确认使用 R1KV
python -c "from vllm.envs import VLLM_COMPRESSION_ALGO; print(VLLM_COMPRESSION_ALGO)"
# 应输出: r1kv
```

---

*创建日期：2025-01-31*
*更新日期：2025-01-31（重写为 vLLM 集成路径）*
