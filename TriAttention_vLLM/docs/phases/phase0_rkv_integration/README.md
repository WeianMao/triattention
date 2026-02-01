# Phase 0: R-KV/vLLM 框架内 SpeckV 集成

## 1. 目标

在 R-KV 的 vLLM fork 中实现 SpeckV 压缩，复用 R1KV 的接口和触发机制。

| 目标 | 验收标准 |
|-----|---------|
| 实现 `update_kv()` 接口 | 与 R1KV 接口兼容 |
| 准确率 | 与 HF 路径差异 < 1% |
| 隔离性 | 默认行为仍为 R1KV |

---

## 2. 架构

### 2.1 集成点

```
vLLM/vllm/v1/attention/backends/flash_attn.py
    ├── __init__: self.kvcompressor = SpeckVvLLM(...) or R1KV(...)
    └── forward: if seq_len >= budget + buffer: 调用 update_kv()
```

### 2.2 与 HF 路径的关键区别

| 方面 | HF 路径 | vLLM 路径 |
|-----|--------|----------|
| 触发条件 | `pos % divide_length == 0` | `seq_len >= budget + buffer` |
| 支持模式 | per_head, per_layer, global | **仅 global** |
| RoPE 处理 | 反演后计算 | **优化版（直接用 K_rot）** |

---

## 3. 核心设计

### 3.1 SpeckVvLLM 类

```python
class SpeckVvLLM:
    def update_kv(self, key, query, value, position_indices, prefill_len) -> (K, V)
    def get_last_keep_indices() -> Tensor  # 供外部更新 position_indices
```

**关键点**：
- 接口与 R1KV 一致，只返回 `(K, V)`
- `position_indices` 由 `FlashAttentionImpl` 维护
- **全局模式**：所有层共享相同 keep_indices（PagedAttention 约束）

### 3.2 打分算法（优化版）

**核心优化**（详见 `docs/design/optimization.md`）：
1. **避免 RoPE 反演**：直接用 K_rot，通过相位校正等价
2. **位置无关系数**：A = s²·Re, B = s²·Im（无需 atan2/sqrt）
3. **共享三角表**：C = cos(t·ω), S = sin(t·ω)
4. **批量矩阵乘法**：`scores = A @ C.T - B @ S.T`

**与 HF 对齐**：
- Union-based selection（每个 head 独立选 top-k → 并集 → 从并集选 top-k）
- Score normalization（per-head z-score）
- Tie-breaking noise（1e-6 量级，TODO: 未来移除）

### 3.3 Position Indices 追踪

| 事件 | 操作 |
|-----|------|
| Prefill | 写入 `[0, 1, ..., prefill_len-1]` |
| Decode | 写入新 token 位置 |
| 压缩后 | 保留 `kept_positions` |
| Request 结束 | 置为 -1 |

---

## 4. 环境配置

| 变量 | 默认值 | 说明 |
|-----|--------|------|
| `VLLM_COMPRESSION_ALGO` | `r1kv` | 算法选择 |
| `VLLM_SPECKV_STATS_PATH` | - | Stats 文件路径 |
| `VLLM_SPECKV_MODEL_PATH` | - | 模型路径（RoPE 配置） |

**Conda 环境**：`rkv_vllm`

**Stats 文件**：`R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/`

---

## 5. 实现步骤

1. 创建 `rkv/compression/speckv_vllm.py`
2. 导出到 `rkv/modeling.py`
3. 修改 `flash_attn.py` 添加算法选择
4. 单元测试 + AIME24 准确率验证

---

## 6. 文件清单

| 类型 | 文件 |
|-----|------|
| 新建 | `rkv/compression/speckv_vllm.py` |
| 修改 | `flash_attn.py`, `rkv/__init__.py` |
| 参考 | `speckv_rkv_style.py`, `round_pruning_utils.py` |

---

## 7. 待验证事项

- [ ] 优化版打分与 HF 原版的数学等价性（需单元测试）
- [ ] prefill_len 的获取准确性（当前假设：首次压缩时 seq_len = prefill_len）

---

*更新日期：2025-01-31*
