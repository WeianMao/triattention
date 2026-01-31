# Phase 0 设计补充说明

基于代码验证发现的补充说明和修正。

> **重要说明**：Phase 0 采用**方案 A（monkey patch model.forward）**，复用现有 `speckv_rkv_style.py` 实现，**不新建 SpeckVRKV 类，不修改 `rkv/` 核心文件**。
> 本文档中关于 `SpeckVRKV` 类的代码示例仅作为 R-KV 接口机制的参考说明，不代表 Phase 0 的实际实现路径。

---

## 1. Compression Flag 三态机制（重要补充）

R-KV 的压缩触发有**三种状态**，不是简单的 True/False：

### 1.1 状态定义

| compression 值 | 行为 | 使用场景 |
|---------------|------|---------|
| `None` | 总是执行压缩 | 每个 attention call 都压缩 |
| `True` | 先更新缓存再压缩 | 有条件压缩（step_length 或 newline 触发） |
| `False` | 不压缩，仅更新缓存 | 非压缩 step |

### 1.2 代码位置

**状态设置**：`modeling.py` CausalLM_forward() 中

```python
# 判断是否触发压缩
if self.config.divide_method == "newline":
    is_newline = predicted_token_ids[0].cpu().item() in self.newline_token_ids
elif self.config.divide_method == "step_length":
    is_newline = self.length % self.config.divide_length == 0

# 设置所有层的 compression flag
for layer in self.model.layers:
    layer.self_attn.config.compression = is_newline  # True or False
```

**状态使用**：`modeling.py` Attention.forward() 中

```python
if self.config.compression is None:
    # 路径 A: 总是压缩
    k_compress, v_compress = self.kv_cluster.update_kv(...)

elif self.config.compression is True:
    # 路径 B: 先更新缓存再压缩
    key_states, value_states = past_key_value.update(...)
    k_compress, v_compress = self.kv_cluster.update_kv(...)
    past_key_value.key_cache[self.layer_idx] = k_compress
    past_key_value.value_cache[self.layer_idx] = v_compress

else:  # False
    # 路径 C: 不压缩
    key_states, value_states = past_key_value.update(...)
```

### 1.3 对 SpeckV-RKV 实现的影响

SpeckV-RKV 需要正确处理这三种状态：

```python
class SpeckVRKV:
    def update_kv(self, key_states, query_states, value_states):
        """
        注意：此方法仅在以下情况被调用：
        1. config.compression is None（总是）
        2. config.compression is True（被触发时）

        当 config.compression is False 时，此方法不会被调用
        """
        # 检查是否需要压缩
        if key_states.shape[-2] <= self.config.budget:
            return key_states, value_states

        # 执行压缩...
```

---

## 2. Query States 来源澄清

### 2.1 文档原描述（不完全准确）

```python
query_states: torch.Tensor,    # [batch, num_q_heads, window_size, head_dim]
```

### 2.2 实际来源

`query_states` 不是当前 decode step 的 query，而是**缓存的 queries**：

```python
# modeling.py L136-143
# Query cache 初始化（prefill 时）
if self.layer_idx not in past_key_value.query_cache:
    past_key_value.query_cache[self.layer_idx] = query_states[
        :, :, -self.config.method_config["window_size"] :, :
    ]

# Query cache 更新（decode 时）
else:
    past_key_value.query_cache[self.layer_idx] = torch.cat(
        (past_key_value.query_cache[self.layer_idx], query_states), dim=2
    )
    # 裁剪到 window_size
    if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
        past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
            self.layer_idx
        ][:, :, -window_size:, :]
```

### 2.3 对 SpeckV-RKV 的影响

**关键区别**：SpeckV 不使用 query_states 进行打分！

```python
class SpeckVRKV:
    def update_kv(self, key_states, query_states, value_states):
        """
        SpeckV 与 R1KV 的关键区别：

        R1KV: 使用 query_states 计算 attention scores
        SpeckV: 使用预计算的 Q 频率统计，忽略 query_states

        参数 query_states 保留在接口中是为了与 R-KV 框架兼容，
        但 SpeckV 实现中应该忽略它。
        """
        # ❌ 不使用 query_states
        # scores = compute_attention_scores(query_states, key_states)

        # ✅ 使用预计算统计
        scores = self._compute_frequency_scores(key_states)
        ...
```

---

## 3. GQA（分组查询注意力）处理

### 3.1 背景

现代模型（Qwen2.5, LLaMA3）使用 GQA：
- `num_attention_heads` (Q heads): 28-32
- `num_key_value_heads` (KV heads): 4-8

### 3.2 R-KV 的 GQA 处理

**位置**：`utils.py` compute_attention_scores()

```python
def compute_attention_scores(query_states, key_states, pooling="max"):
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    query_group_size = q_heads // kv_heads

    if query_group_size == 1:
        # 标准 MHA: Q heads == KV heads
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    else:
        # GQA: Q heads > KV heads
        # Reshape Q: [batch, kv_heads, group_size, q_len, head_dim]
        query_states = query_states.view(
            batch_size, kv_heads, query_group_size, q_len, head_dim
        )
        # Expand K: [batch, kv_heads, 1, kv_len, head_dim]
        key_states = key_states.unsqueeze(2)

        # Compute attention: [batch, kv_heads, group_size, q_len, kv_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(3, 4))

        # Pool over query groups
        if pooling == "mean":
            attn_weights = attn_weights.mean(dim=2)
        elif pooling == "max":
            attn_weights = attn_weights.max(dim=2).values
```

### 3.3 对 SpeckV-RKV 的影响

SpeckV 不使用 attention 打分，但**打分结果需要与 KV heads 对齐**：

```python
class SpeckVRKV:
    def _compute_frequency_scores(self, key_states):
        """
        key_states: [batch, num_kv_heads, seq_len, head_dim]

        SpeckV 的 sampled_heads 是 (layer, attention_head) 对，
        需要映射到 KV heads。
        """
        batch_size, num_kv_heads, seq_len, head_dim = key_states.shape

        # 获取 GQA 配置
        if hasattr(self, 'num_key_value_groups'):
            group_size = self.num_key_value_groups
        else:
            group_size = 1

        scores_per_kv_head = []
        for kv_head in range(num_kv_heads):
            # 找到属于这个 KV head 的 sampled attention heads
            attn_heads = [h for l, h in self.sampled_heads
                         if h // group_size == kv_head]

            # 计算每个 attention head 的打分
            head_scores = []
            for attn_head in attn_heads:
                score = self._score_single_head(key_states[:, kv_head], attn_head)
                head_scores.append(score)

            # 聚合（层内取 max，跨层取 mean）
            aggregated = self._aggregate_head_scores(head_scores)
            scores_per_kv_head.append(aggregated)

        return torch.stack(scores_per_kv_head, dim=1)  # [batch, num_kv_heads, seq_len]
```

---

## 4. 现有 SpeckV 实现架构

### 4.1 三层架构

```
rkv/compression/speckv.py  (Wrapper)
    │
    └──→ weian_development/speckv/rkv_speckv_generate.py  (Generate Patch)
            │
            └──→ weian_development/speckv/sparse_round_pruner_prefill_keep.py  (Core Logic)
```

### 4.2 各层职责

| 层 | 文件 | 职责 |
|---|------|------|
| Wrapper | `speckv.py` | 导出 `apply_speckv_generate_patch` |
| Generate Patch | `rkv_speckv_generate.py` | 修改 model.generate() 流程 |
| Core Logic | `sparse_round_pruner_prefill_keep.py` | 打分、裁剪、位置追踪 |

### 4.3 Phase 0 的定位（方案 A）

**Phase 0 采用方案 A：复用现有实现，不新建类**。

```
Phase 0 实现路径：
  apply_speckv_rkv_style_patch() → SpeckVRKVStyle → round_pruning_utils.py

现有代码位置：
  weian_development/speckv/speckv_rkv_style.py     # 主压缩器
  weian_development/speckv/round_pruning_utils.py  # 打分函数
  weian_development/speckv/rkv_speckv_generate.py  # Generate patch
```

**Phase 0 目标**：
- 理解现有代码的工作方式
- 验证三种 pruning mode 的正确性
- 创建使用文档
- **不重写代码，不修改 `rkv/` 核心文件**

---

## 5. reset_compression_state 调用点

### 5.1 实际用途

用于多样本评估时重置状态（如 token tracking）。

### 5.2 调用位置

**未在 modeling.py 中调用**，需要在评估脚本中手动调用：

```python
# 评估脚本示例
for sample in dataset:
    # 重置压缩器状态
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'kv_cluster'):
            if hasattr(layer.self_attn.kv_cluster, 'reset_compression_state'):
                layer.self_attn.kv_cluster.reset_compression_state()

    # 运行推理
    output = model.generate(...)
```

### 5.3 SpeckV-RKV 实现

```python
class SpeckVRKV:
    def reset_compression_state(self) -> None:
        """重置状态用于新样本"""
        self.prefill_length = 0
        self.cache_positions = []

        # 如果追踪 token 索引
        if hasattr(self, 'kept_token_indices'):
            self.kept_token_indices = []
            self.evicted_token_num = 0
```

---

## 6. 触发语义对齐问题（重要发现）

### 6.1 问题描述

验证 Agent 发现 SpeckV 的压缩触发机制与 R-KV 的三态 compression flag **完全独立**：

| 维度 | R-KV | SpeckV |
|-----|------|--------|
| **触发判断** | `compression` flag (None/True/False) | `absolute_position % divide_length == 0` |
| **决策者** | CausalLM.forward 设置标志 | SpeckVRKVStyle 自行判断 |
| **语义** | 三态：总是/触发时/禁止 | 二态：位置对齐时压缩 |

### 6.2 潜在风险

1. **`compression=False` 时 SpeckV 仍可能压缩**
   - R-KV 明确禁止压缩时，SpeckV 仍在位置对齐点压缩
   - 违反 R-KV 的设计意图

2. **压缩时机不可预测**
   - R-KV 和 SpeckV 可能在不同时间点触发
   - 缓存大小控制可能与预期不符

### 6.3 影响评估

**对 Phase 0 验证的影响**：
- 如果只运行 SpeckV 脚本（不混用 R-KV 其他算法），问题不会表现
- 现有三个脚本应该仍能正常运行
- 但结论的可信度需要额外验证

**建议**：
- Phase 0 验证时，添加日志记录实际压缩时机
- 确认压缩仅发生在预期位置
- 如果发现异常，在 Phase 1 实现中修复

### 6.4 Stats 兼容性（已验证通过）

✓ `stats_utils.py` 包含完整的 `validate_stats_metadata()` 校验
✓ 校验字段：prompt_template, use_chat_template, dtype, attn_implementation, rope_style/type
✓ 不匹配时立即抛出 `ValueError`
✓ `speckv_rkv_style.py` 在初始化时强制调用校验

---

## 7. Phase 0 验证清单（基于 Review 反馈）

### 7.1 Stats 元数据校验（✓ 已实现）

**验证结果**：现有代码已包含完整的 stats 元数据校验机制。

`validate_stats_metadata()` 在 `speckv_rkv_style.py` 初始化时自动调用，校验：
- prompt_template, use_chat_template, system_prompt
- attn_implementation, dtype, rope_style, rope_type

无需额外操作，如果配置不匹配会自动报错。

### 7.2 GQA 映射校验（建议）

验证 sampled_heads 到 KV heads 的映射正确：

```python
# 检查每个 KV head 覆盖的 sampled heads 数量
for kv_head in range(num_kv_heads):
    covered_heads = [h for l, h in sampled_heads if h // group_size == kv_head]
    print(f"KV head {kv_head}: {len(covered_heads)} sampled heads")
```

### 7.3 reset_compression_state 调用位置

**已确认**：在 `rkv_sharded_eval.py` 的评估循环中，每个样本前需重置状态。
现有脚本应已包含此逻辑，需验证。

### 7.4 进程命名规范

长时间任务必须使用 `PD-L1_binder` 进程名前缀：
- 通过 `rkv_sharded_runner.py` wrapper 自动设置
- 或在脚本中使用 `setproctitle`

### 7.5 输出目录命名

避免覆盖基线结果：
```bash
--output-dir outputs/speckv_perhead_aime24_$(date +%Y%m%d)
```

### 7.6 Phase 0 → Phase 1 迁移说明

| 方面 | Phase 0 (R-KV) | Phase 1 (vLLM) | 迁移难度 |
|-----|----------------|----------------|---------|
| KV Layout | `[batch, heads, seq, dim]` | PagedAttention blocks | 高 |
| 压缩触发 | `absolute_position % divide_length` | 需要新设计 | 中 |
| 位置追踪 | `cache_positions` list | `position_indices` tensor | 中 |
| 打分算法 | 可复用 | 可复用 | 低 |

---

## 8. 文件职责（只读参考）

| 文件 | 用途 | Phase 0 操作 |
|-----|------|-------------|
| `weian_development/speckv/speckv_rkv_style.py` | 主压缩器 | 只读、理解 |
| `weian_development/speckv/round_pruning_utils.py` | 打分函数 | 只读、理解 |
| `weian_development/speckv/stats_utils.py` | Stats 验证 | 只读、使用 |
| `weian_development/speckv/rkv_speckv_generate.py` | Generate patch | 只读、理解 |
| `weian_development/speckv/README.md` | 使用文档 | **新建** |

---

*创建日期：2025-01-31*
*更新日期：2025-01-31（基于 Review 反馈修正，移除与方案 A 矛盾的内容）*
