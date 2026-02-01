# Phase 0 设计补充说明

## 1. R1KV 集成点分析

**位置**：`flash_attn.py` 第 431 行、547-588 行

**触发条件**：`seq_len >= VLLM_V1_R_KV_BUDGET + VLLM_V1_R_KV_BUFFER`（默认 64+64=128）

**KV 布局**：PagedAttention，通过 `occupied_slot_mapping` 索引

---

## 2. SpeckV 适配要点

### 2.1 与 R1KV 的区别

| 方面 | R1KV | SpeckV |
|-----|------|--------|
| 评分依据 | attention + similarity | 频率统计（预计算） |
| 需要 query | 是 | 否 |
| 额外输入 | 无 | stats 文件、position_indices |
| RoPE 处理 | 无 | 优化版（直接用 K_rot） |

### 2.2 GQA 映射

```python
kv_head = attn_head // num_kv_groups  # 28 heads → 4 kv_heads
```

### 2.3 全局模式约束

vLLM PagedAttention 要求所有层共享相同 KV 布局 → **只能用全局模式**（所有层、所有 head 保留相同 token）

---

## 3. 关键设计决策

### 3.1 模块路径

统一入口：`from rkv.modeling import SpeckVvLLM`

### 3.2 position_indices 管理

- 形状：`[num_blocks, block_size]`（与 KV cache 对齐）
- 由 `FlashAttentionImpl` 维护，压缩器只读
- 通过 `occupied_slot_mapping` 实现 per-request 隔离

### 3.3 Lazy Init

layer_idx 在首次 forward 时从 `layer.layer_idx` 获取，stats 使用类级别缓存。

### 3.4 prefill_len 来源

首次压缩时的 seq_len 作为 prefill_len（假设：prefill >= budget + buffer）。

**局限**：若 prefill 未达阈值，会高估 prefill_len。影响：略保守，不影响正确性。

---

## 4. Stats 文件格式

```python
{
    "metadata": {
        "sampled_heads": [(layer, attn_head), ...],
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
    },
    "stats": {
        "layer00_head05": {
            "q_mean_real": Tensor,
            "q_mean_imag": Tensor,
            "q_abs_mean": Tensor,
        },
        ...
    }
}
```

---

*更新日期：2025-01-31*
