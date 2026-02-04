# Phase 0 决策历史

> **归档说明**：本文档合并自 `phase0_plan_review.md` 和 `phase0_doc_fixlog.md`，记录了文档演进过程中的重要决策。
> 当前设计请参考 `../README.md` 和 `../SPEC.md`。

---

## 1. 早期审查反馈（已解决）

### 1.1 架构方案矛盾 → 已统一为 vLLM update_kv 方案

**原问题**：README 选择 monkey patch，DESIGN_NOTES 选择 update_kv，两者冲突。

**决策**：统一为 vLLM `update_kv()` 接口方案，因为：
- R1KV 已在 vLLM v1 实现此接口
- 可复用现有触发机制
- Phase 0 仅支持 `per_layer_perhead` 模式（无跨层依赖）

### 1.2 隔离开发原则 → 参数隔离

**原问题**：文档写"不能修改 rkv/ 核心代码"，但实际需要修改。

**决策**：允许修改核心文件，但必须：
- 通过参数/环境变量控制
- 默认行为不变（默认使用 R1KV）
- 禁止无参数隔离的核心修改

### 1.3 触发条件 → 复用 vLLM 现有机制

**原问题**：SpeckV 触发条件与 R-KV 三态 flag 可能不一致。

**决策**：复用 vLLM 的 `seq_len >= budget + buffer` 触发条件，不引入新触发逻辑。

### 1.4 Stats 校验 → 由入口保证

**原问题**：Stats 元数据校验过于绝对。

**决策**：完整校验由入口脚本保证（传入 `metadata_expectations`），压缩器只做最小默认校验。

### 1.5 reset_compression_state → 自动重置

**原问题**：是否需要手动调用 reset。

**决策**：SpeckVvLLM 无状态设计，无需手动 reset。

---

## 2. 关键设计决策迭代

### 2.1 position_indices 维护

| 迭代 | 方案 | 问题 | 解决 |
|-----|------|------|------|
| v1 | per-request dict | flash_attn.py 无 request_ids | 改用 slot_mapping |
| v2 | 全局张量 + slot_mapping | - | **最终方案** |

**最终决策**：
- 全局 `position_indices` 张量 `[num_blocks, block_size]`
- 通过 `occupied_slot_mapping` 实现 per-request 隔离
- 由 `FlashAttentionImpl` 维护，不暴露给外部

### 2.2 layer_idx 获取

| 迭代 | 方案 | 问题 | 解决 |
|-----|------|------|------|
| v1 | `__init__` 传入 | vLLM 不一定传 layer_idx | 改用 lazy init |
| v2 | lazy init | - | **最终方案** |

**最终决策**：首次 `forward` 时从 `layer.layer_idx` 获取。

### 2.3 prefill_len 来源

| 迭代 | 方案 | 问题 | 解决 |
|-----|------|------|------|
| v1 | 修改 attn_metadata | 违反隔离原则 | 改用本地记录 |
| v2 | 首次压缩时记录 | - | **最终方案** |

**最终决策**：`FlashAttentionImpl` 内部维护 `prefill_lens` dict，首次压缩时记录。

### 2.4 update_kv 返回值

| 迭代 | 方案 | 问题 | 解决 |
|-----|------|------|------|
| v1 | 返回 `(K, V, position_indices)` | 破坏 R1KV 接口兼容性 | 改为只返回 (K, V) |
| v2 | 只返回 `(K, V)` | - | **最终方案** |

**最终决策**：
- `update_kv()` 只返回 `(K, V)`
- 保留索引通过 `get_last_keep_indices()` 获取

---

## 3. 文档修订记录

### 修订 1：隔离开发口径统一
- 位置：README 2.2/2.3
- 改动：明确参数隔离策略

### 修订 2：压缩触发条件
- 位置：README 1.2
- 改动：说明使用 vLLM 现有触发逻辑

### 修订 3：reset_compression_state
- 位置：DESIGN_NOTES 5.2/7.3
- 改动：说明无状态设计，无需手动 reset

### 修订 4：Stats 元数据校验
- 位置：DESIGN_NOTES 6.4/7.1
- 改动：明确校验由入口保证

### 修订 5：vLLM API 命名
- 位置：全文
- 改动：`slot_mapping` → `occupied_slot_mapping`

---

*归档日期：2025-01-31*
