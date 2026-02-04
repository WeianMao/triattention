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

### 2.3 模式选择 [已确认]

**三种模式都可支持**，直接按 HF 实现即可：

| 模式 | 说明 | 可行性 |
|-----|------|--------|
| per_head | 每个 head 独立选 token | ✓ R-KV 已证明 |
| per_layer | 每层选同样的 token | ✓ per_head 的简化版 |
| per_layer_perhead | 每层每 head 独立选 | ✓ 和 per_head 本质相同 |

**原理**：R-KV 在 vLLM 里已实现 per-head 选择。KV cache 布局 `[batch, num_kv_heads, seq_len, head_dim]` 天然支持每个 head 用不同的 indices，通过 `gather(dim=2, index=indices)` 实现。

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

通过 `FlashAttentionMetadata.prefill_lens` 传递真实的 prefill_len：
- 数据来源：`req_states.prefill_len`（vLLM 已有）
- 传递方式：在 metadata builder 中填充，压缩时通过 `attach_prefill_length()` 设置
- 隔离保证：R1KV 等现有算法不使用此字段，不受影响

---

## 4. Stats 文件（关键字段）

最少需包含：
- 统计数据：`q_mean_complex` / `freq_scale_sq` / `sampled_heads`
- 元信息：`num_attention_heads` / `num_kv_heads` / `head_dim` / `rope` / `prompt_template`

具体字段以现有 `stats_utils` 为准，避免硬编码假格式。

---

## 5. 待确认 / 待验证

- ~~模式选择：global vs per_layer_perhead~~ [已解决：三种模式都可支持]
- ~~优化版打分与 HF 原版数学等价性~~ [已验证 ✓]
- ~~prefill_len 的真实获取方式~~ [已解决：通过 AttentionMetadata 传递]
- ~~position_indices 的 reset 机制~~ [已解决：双重保险，见下文]
- stats 字段完整性与加载校验规则

### 5.1 position_indices reset 机制 [已确定]

采用**双重保险**策略，两种机制同时生效：

**机制 1：Slot 复用时 reset**
- 在分配 slot 给新 request 时，清零对应的 position_indices
- 实现位置：slot 分配逻辑中

**机制 2：Scheduler hook**
- 在 request 结束时，通过回调通知 attention 层
- 主动清理该 request 占用的 position_indices

**双重保险的好处：**
- 任一机制失效，另一机制兜底
- 防止边界情况（如 request 异常终止）导致状态泄漏
- 调试时更容易定位问题

---

*更新日期：2026-01-31*
