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

## 4. 关键设计决策（Review 反馈）

> 本节针对 review 中指出的关键问题给出明确决策。

### 4.1 模块路径问题

**问题**：文档写 `rkv/compression/speckv_vllm.py`，但 `flash_attn.py` 实际 import 的是：
```python
from rkv.modeling import R1KV  # 来自 R-KV/HuggingFace/rkv/modeling.py
```

**决策**：SpeckV 放入 **HuggingFace rkv 包**，保持 import 一致性。

```
# 文件位置
R-KV/HuggingFace/rkv/compression/speckv_vllm.py  ← SpeckV 实现
R-KV/HuggingFace/rkv/compression/__init__.py     ← 添加 SpeckVvLLM export
R-KV/HuggingFace/rkv/modeling.py                 ← 添加 SpeckVvLLM import

# flash_attn.py 中的 import
from rkv.modeling import R1KV, SpeckVvLLM  # 新增 SpeckVvLLM
```

### 4.2 有状态算法 vs 全局单例问题

**问题**：`FlashAttentionImpl.kvcompressor` 是全局单例，但 SpeckV 需要按 request 跟踪状态（`cache_positions`, `absolute_position`）。

**决策**：使用 **per-request 状态字典**，压缩器本身无状态。

```python
class SpeckVvLLM:
    def __init__(self, ...):
        # 压缩器无状态，所有 request 共享
        self.stats = self._load_stats(stats_path)

        # 状态由外部管理，通过 request_id 索引
        # 不在压缩器内部存储 per-request 状态

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        position_indices,     # 外部传入：当前 KV 的位置信息
        request_id=None,      # 可选：用于日志/调试
    ):
        # 所有状态信息通过参数传入，不依赖内部状态
        ...
```

**position_indices 管理责任**：
- **谁维护**：`flash_attn.py` 或 `attn_metadata`，不是 SpeckV 压缩器
- **更新时机**：prefill/decode 时写入，prune 时同步更新
- **存储位置**：与 KV cache 同布局 `[num_blocks, block_size]`

### 4.3 position_indices 在 vLLM 中的维护

**问题**：HF 路径靠 `cache_positions` 跟踪位置，vLLM 没有现成的位置追踪。

**决策**：在 `flash_attn.py` 中显式维护 `position_indices`，与 KV cache 同步。

```python
# flash_attn.py 修改（伪代码）

class FlashAttentionImpl:
    def __init__(self, ...):
        self.kvcompressor = SpeckVvLLM(...) if algo == "speckv" else R1KV(...)

        # 新增：position_indices 存储（与 KV cache 同布局）
        # 实际实现可能放在 attn_metadata 或单独的数据结构中
        self.position_indices = None  # [num_blocks, block_size]

    def forward(self, ...):
        # Prefill: 初始化 position_indices
        if is_prefill:
            self.position_indices[block_ids] = torch.arange(prefill_len)

        # Decode: 追加新位置
        if is_decode:
            self.position_indices[new_slot] = current_seq_len

        # Prune: 同步更新
        if need_compress:
            # 提取当前序列的 position_indices
            current_positions = self.position_indices[slot_mapping]

            # 调用压缩器，传入 position_indices
            compressed_k, compressed_v = self.kvcompressor.update_kv(
                current_key, current_query, current_value,
                position_indices=current_positions,
            )

            # 回写时同步更新 position_indices
            # （保留的 token 位置不变）
```

### 4.4 Prefill 处理规则

**问题**：HF 路径有 `include_prefill_in_budget` / `allow_prefill_compression`，vLLM 路径未明确。

**决策**：Phase 0 默认 **prefill 不参与压缩**（保护前缀）。

| 配置 | 值 | 说明 |
|-----|-----|------|
| `protect_prefill` | `True`（默认） | Prefill token 不参与裁剪 |
| `prefill_budget` | 与 `budget` 分开 | Prefill 有独立配额（可选） |

**实现方式**：
```python
def update_kv(self, key_states, ..., position_indices, prefill_len=0):
    if prefill_len > 0:
        # 分离 prefill 和 decode 部分
        prefill_k = key_states[:, :, :prefill_len, :]
        decode_k = key_states[:, :, prefill_len:, :]

        # 只对 decode 部分打分和裁剪
        scores = self._compute_scores(decode_k, position_indices[prefill_len:])
        keep_indices = self._select_tokens(scores, budget - prefill_len)

        # 拼接：prefill（全保留）+ decode（裁剪后）
        compressed_k = torch.cat([prefill_k, decode_k.gather(..., keep_indices)], dim=2)
```

**与 HF 基线对比**：确保使用相同的 prefill 策略，否则结果不可比。

### 4.5 per_head 模式限制（跨层聚合）

**问题**：vLLM 按层调用 `update_kv`，无法实现"全局 per_head"（跨层聚合打分）。

**决策**：Phase 0 仅支持 **per_layer_perhead**（每层独立、每 head 独立）。

| 模式 | HF 路径 | vLLM Phase 0 | 原因 |
|-----|---------|-------------|------|
| `per_head` (跨层) | ✓ | ✗ | 需要所有层的 KV 才能聚合打分 |
| `per_layer` | ✓ | ✗ | 需要同层所有 head 聚合 |
| `per_layer_perhead` | ✓ | ✓ | 每层每 head 独立，无跨层依赖 |

**配置映射**：
```python
# 环境变量
VLLM_SPECKV_PRUNING_MODE = "per_layer_perhead"  # 唯一支持的模式

# 如果用户指定其他模式，警告并 fallback
if pruning_mode != "per_layer_perhead":
    logger.warning(f"vLLM Phase 0 只支持 per_layer_perhead，忽略 {pruning_mode}")
    pruning_mode = "per_layer_perhead"
```

### 4.6 Stats 校验与 Model Config

**问题**：Stats 需要与模型配置匹配（RoPE/num_heads/dtype/prompt），但文档只有 `stats_path`。

**决策**：添加 model config 校验，stats 必须包含元数据。

**Stats 文件结构**：
```python
stats = {
    # 核心数据
    "q_mean_complex": ...,
    "freq_scale_sq": ...,
    "sampled_heads": ...,

    # 元数据（用于校验）
    "metadata": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "rope_theta": 1000000.0,
        "rope_scaling": None,
        "dtype": "bfloat16",
        "prompt_template": "plain",  # plain / chat
    }
}
```

**校验逻辑**：
```python
def _load_stats(self, stats_path, model_config=None):
    stats = torch.load(stats_path)

    if model_config is not None:
        meta = stats.get("metadata", {})

        # 必须匹配的项
        assert meta.get("num_kv_heads") == model_config.num_kv_heads
        assert meta.get("head_dim") == model_config.head_dim

        # 警告但不阻止的项
        if meta.get("rope_theta") != model_config.rope_theta:
            logger.warning(f"Stats rope_theta 不匹配: {meta.get('rope_theta')} vs {model_config.rope_theta}")

    return stats
```

**环境变量补充**：
```python
VLLM_SPECKV_MODEL_CONFIG = os.getenv("VLLM_SPECKV_MODEL_CONFIG", None)  # 可选：model config 路径
```

---

## 5. GQA 处理

### 5.1 背景

Qwen2.5-7B 使用 GQA：
- `num_attention_heads` = 28
- `num_key_value_heads` = 4
- `num_key_value_groups` = 7

### 5.2 vLLM 中的 GQA 处理

```python
# flash_attn.py 中，KV cache 使用 num_kv_heads
key_states.shape = [batch, num_kv_heads, seq_len, head_dim]  # [1, 4, seq, 128]

# query 使用 num_attention_heads
query_states.shape = [batch, num_heads, seq_len, head_dim]   # [1, 28, seq, 128]
```

### 5.3 SpeckV 的 GQA 适配

SpeckV 的 `sampled_heads` 是 attention head 索引，需要映射到 KV heads：

```python
# sampled_heads: [(layer, attn_head), ...]
# 需要映射到 kv_head

def map_to_kv_head(attn_head, num_kv_groups):
    return attn_head // num_kv_groups

# 例如：attn_head=14 -> kv_head=14//7=2
```

---

## 6. 环境变量设计

### 6.1 新增环境变量

| 变量名 | 默认值 | 说明 |
|-------|--------|------|
| `VLLM_COMPRESSION_ALGO` | `"r1kv"` | 压缩算法选择 |
| `VLLM_SPECKV_STATS_PATH` | `None` | SpeckV stats 文件路径 |
| `VLLM_SPECKV_PRUNING_MODE` | `"per_head"` | pruning 模式 |

### 6.2 在 envs.py 中添加

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

## 7. 测试策略

### 7.1 单元测试优先级

1. **接口兼容性**：`update_kv()` 输入输出格式正确
2. **边界条件**：`seq_len <= budget` 时直接返回
3. **压缩比**：输出长度 <= budget
4. **数值正确性**：与 HF 路径对比

### 7.2 与 HF 路径对比验证

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

## 8. 已知限制

### 8.1 Phase 0 限制

| 限制 | 原因 | Phase 1 解决方案 |
|-----|------|-----------------|
| 无 Triton 优化 | Phase 0 重点是正确性 | Phase 1 实现 Triton kernel |
| 仅支持 per_layer_perhead 模式 | vLLM 按层调用，无法跨层聚合 | Phase 1 重新设计接口 |

### 8.2 与 HF 路径的功能差异

| 功能 | HF 路径 | vLLM Phase 0 | 说明 |
|-----|--------|--------------|------|
| per_head pruning (跨层) | ✓ | ✗ | 需要全局聚合，见 4.5 节 |
| per_layer pruning | ✓ | ✗ | 需要同层所有 head 聚合 |
| per_layer_perhead | ✓ | ✓ | **Phase 0 唯一支持的模式** |
| 位置追踪 | ✓ | ✓ | 通过 position_indices，见 4.3 节 |
| prefill 保护 | ✓ | ✓ | 默认保护，见 4.4 节 |
| 打分优化（LUT） | ✗ | ✓ | Phase 0 即实现 |
| Stats 校验 | ✓ | ✓ | 见 4.6 节 |

---

## 9. 文件修改检查清单

### 9.1 新建文件

- [ ] `R-KV/HuggingFace/rkv/compression/speckv_vllm.py` - SpeckV vLLM 接口实现
- [ ] `tests/test_speckv_vllm.py` - 单元测试

### 9.2 修改文件

- [ ] `R-KV/HuggingFace/rkv/compression/__init__.py` - 导出 SpeckVvLLM
- [ ] `R-KV/HuggingFace/rkv/modeling.py` - 添加 SpeckVvLLM import（保持与 R1KV 一致的 import 方式）
- [ ] `R-KV/vLLM/vllm/v1/attention/backends/flash_attn.py` - 算法选择分支 + position_indices 更新
- [ ] `R-KV/vLLM/vllm/envs.py` - 环境变量

### 9.3 关键数据结构

- [ ] `position_indices`: `[num_blocks, block_size]` 与 KV cache 对齐
- [ ] `trig_cos/sin`: `[num_offsets, freq_count]` 共享三角表（~4 KB）
- [ ] Stats 从文件加载：`q_mean_complex`, `freq_scale_sq`, `omega` 等

### 9.4 验证默认行为不变

```bash
# 不设置 VLLM_COMPRESSION_ALGO，确认使用 R1KV
python -c "from vllm.envs import VLLM_COMPRESSION_ALGO; print(VLLM_COMPRESSION_ALGO)"
# 应输出: r1kv
```

---

*创建日期：2025-01-31*
*更新日期：2025-01-31（重写为 vLLM 集成路径）*
