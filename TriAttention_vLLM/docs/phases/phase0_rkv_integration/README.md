# Phase 0: R-KV/vLLM 框架内 SpeckV 集成

## 概述

在 R-KV 的 **vLLM fork** 中集成 SpeckV 压缩算法，复用 R1KV 已有的压缩接口和触发机制。

> **注意**：HuggingFace/transformers 路径（`apply_speckv_rkv_style_patch`）是**已完成的工作**，不属于 Phase 0。
> Phase 0 的目标是在 vLLM 的 attention backend 里接入 SpeckV。

---

## 1. 架构概述

### 1.1 R-KV/vLLM 的压缩架构

R1KV 已在 vLLM v1 中实现，集成点在 `flash_attn.py`：

```
vLLM/vllm/v1/attention/backends/flash_attn.py
    │
    ├── FlashAttentionImpl.__init__()
    │   └── self.kvcompressor = R1KV(budget=VLLM_V1_R_KV_BUDGET)
    │
    └── FlashAttentionImpl.forward()
        └── if seq_len >= budget + buffer:
                compressed_k, compressed_v = self.kvcompressor.update_kv(k, q, v)
```

### 1.2 Phase 0 目标架构

```
vLLM/vllm/v1/attention/backends/flash_attn.py
    │
    ├── FlashAttentionImpl.__init__()
    │   └── if compression_algo == "speckv":
    │           self.kvcompressor = SpeckV(budget=..., stats_path=...)
    │       else:
    │           self.kvcompressor = R1KV(budget=...)
    │
    └── FlashAttentionImpl.forward()
        └── [复用现有压缩调用逻辑，无需修改]
```

### 1.3 与 HF 路径的关键区别

| 方面 | HF 路径（已完成） | vLLM 路径（Phase 0） |
|-----|------------------|---------------------|
| 框架 | HuggingFace transformers | vLLM v1 |
| 集成方式 | monkey patch `model.forward()` | 实现 `update_kv()` 接口 |
| 压缩触发 | `absolute_position % divide_length` | `seq_len >= budget + buffer` |
| KV 布局 | `[batch, heads, seq, dim]` | PagedAttention blocks |
| Batch 支持 | batch=1 | batch > 1 |
| 代码位置 | `weian_development/speckv/` | `rkv/compression/` + `vLLM/vllm/v1/` |

---

## 2. 目标与约束

### 2.1 目标

| 目标 | 验收标准 |
|-----|---------|
| 实现 SpeckV 的 `update_kv()` 接口 | 与 R1KV 接口兼容 |
| 集成到 vLLM v1 flash_attn backend | 可通过参数选择 SpeckV |
| 准确率验证 | 与 HF 路径结果差异 < 1% |
| 不影响 R1KV | 默认行为不变 |

### 2.2 约束

| 约束 | 说明 |
|-----|------|
| 复用 R1KV 触发机制 | 使用 `VLLM_V1_R_KV_BUDGET` + `VLLM_V1_R_KV_BUFFER` |
| 隔离开发 | 通过参数选择算法，默认仍为 R1KV |
| 无 Triton | Phase 0 使用 PyTorch，Triton 留给 Phase 1 |

### 2.3 隔离开发原则（重要）

**核心原则**：允许修改 vLLM 核心文件，但必须通过**参数隔离**，默认路径对 R1KV 零影响。

#### 2.3.1 允许的操作

| 操作 | 示例 | 说明 |
|-----|------|------|
| 新建 SpeckV 压缩类 | `R-KV/HuggingFace/rkv/compression/speckv_vllm.py` | 实现 `update_kv()` 接口 |
| 在 flash_attn.py 添加算法选择 | `if algo == "speckv": ...` | 参数隔离分支 |
| 添加环境变量 | `VLLM_COMPRESSION_ALGO` | 控制算法选择 |
| 添加 SpeckV 特定配置 | `VLLM_SPECKV_STATS_PATH` | SpeckV 需要的额外参数 |

#### 2.3.2 禁止的操作

| 禁止操作 | 原因 |
|---------|------|
| 修改 R1KV 的 `update_kv()` 实现 | 影响现有算法 |
| 改变默认压缩算法 | 必须默认为 R1KV |
| 修改触发条件（budget + buffer） | 影响所有压缩算法 |

#### 2.3.3 验证隔离性

```bash
# 1. 默认配置：R1KV 行为不变
python -m vllm.entrypoints.openai.api_server --model xxx
# 应使用 R1KV

# 2. 指定 SpeckV
export VLLM_COMPRESSION_ALGO=speckv
export VLLM_SPECKV_STATS_PATH=/path/to/stats.pt
python -m vllm.entrypoints.openai.api_server --model xxx
# 应使用 SpeckV
```

---

## 3. 现有代码分析

### 3.1 R1KV 实现参考

**文件**：`R-KV/rkv/compression/r1_kv.py`

```python
class R1KV:
    def __init__(self, budget=128, window_size=8, ...):
        self.budget = budget
        self.window_size = window_size

    def update_kv(
        self,
        key_states,      # [batch, num_kv_heads, seq_len, head_dim]
        query_states,    # [batch, num_heads, seq_len, head_dim]
        value_states,    # [batch, num_kv_heads, seq_len, head_dim]
    ) -> tuple[Tensor, Tensor]:
        """返回压缩后的 (key_states, value_states)"""
        if key_states.shape[2] <= self.budget:
            return key_states, value_states

        # 计算得分 → topk 选择 → 返回压缩后的 K/V
        ...
```

### 3.2 vLLM 集成点

**文件**：`R-KV/vLLM/vllm/v1/attention/backends/flash_attn.py`

```python
# 第 431 行：初始化压缩器
self.kvcompressor = R1KV(budget=VLLM_V1_R_KV_BUDGET)

# 第 547-588 行：压缩调用
for i in range(attn_metadata.num_reqs):
    if seq_lens[i] < VLLM_V1_R_KV_BUDGET + VLLM_V1_R_KV_BUFFER:
        continue

    # 提取当前序列的 KV
    current_key = key_cache[...]
    current_value = value_cache[...]

    # 压缩
    compressed_key, compressed_value = self.kvcompressor.update_kv(
        current_key, current_query, current_value
    )

    # 回写
    key_cache[...] = compressed_key
    value_cache[...] = compressed_value
```

### 3.3 SpeckV 现有实现（HF 路径，仅供参考）

**文件**：`R-KV/weian_development/speckv/speckv_rkv_style.py`

SpeckV 的核心打分逻辑可以从这里提取：
- `score_keys_for_round()` - 基于频率统计的打分
- `_select_per_head_independent()` 等 - 不同 pruning mode 的选择策略

---

## 4. 设计方案

### 4.1 SpeckV vLLM 类设计

> 详细设计见 `docs/implementation/data_structures.md` 和 `docs/design/optimization.md`

```python
# R-KV/HuggingFace/rkv/compression/speckv_vllm.py
# (import path: from rkv.compression.speckv_vllm import SpeckVvLLM)

class SpeckVvLLM:
    """SpeckV 压缩器 - vLLM 接口版本"""

    def __init__(
        self,
        budget: int = 128,
        stats_path: str = None,           # 预计算统计文件
        pruning_mode: str = "per_head",   # per_head / per_layer / per_layer_perhead
        **kwargs,
    ):
        self.budget = budget
        self.stats = self._load_stats(stats_path)
        self.pruning_mode = pruning_mode

        # 位置追踪（与 KV cache 同步）
        self.position_indices = None  # [num_blocks, block_size]

        # 共享三角函数表（每轮打分时更新）
        self.trig_cos = None  # [num_offsets, freq_count]
        self.trig_sin = None

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        position_indices: torch.Tensor = None,  # 可选：外部传入位置
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        SpeckV 压缩逻辑（优化版，无 RoPE 反演）：
        1. 直接用 K_rot 计算打分系数（避免 RoPE 反演）
        2. 使用共享三角表计算多位置得分
        3. TopK 选择
        4. 返回压缩后的 K/V（同步更新 position_indices）
        """
        if key_states.shape[2] <= self.budget:
            return key_states, value_states

        scores = self._compute_scores_optimized(key_states, position_indices)
        keep_indices = self._select_tokens(scores)

        compressed_key = self._gather(key_states, keep_indices)
        compressed_value = self._gather(value_states, keep_indices)

        return compressed_key, compressed_value

    def _compute_scores_optimized(self, key_states, position_indices):
        """
        优化的打分计算（无三角函数调用）：
        1. 计算位置无关系数 A_coef, B_coef（只需乘法加法）
        2. 使用预计算的共享三角表 C, S
        3. score = A_coef · C - B_coef · S + extra_term
        """
        ...

    def _select_tokens(self, scores):
        """根据 pruning_mode 选择 token"""
        ...
```

### 4.2 vLLM 集成修改

```python
# vLLM/vllm/v1/attention/backends/flash_attn.py

from rkv.compression import R1KV
from rkv.compression.speckv_vllm import SpeckVvLLM  # 新增

# 环境变量
VLLM_COMPRESSION_ALGO = os.getenv("VLLM_COMPRESSION_ALGO", "r1kv")
VLLM_SPECKV_STATS_PATH = os.getenv("VLLM_SPECKV_STATS_PATH", None)

class FlashAttentionImpl:
    def __init__(self, ...):
        # 根据配置选择压缩算法
        if VLLM_COMPRESSION_ALGO == "speckv":
            self.kvcompressor = SpeckVvLLM(
                budget=VLLM_V1_R_KV_BUDGET,
                stats_path=VLLM_SPECKV_STATS_PATH,
            )
        else:  # 默认 r1kv
            self.kvcompressor = R1KV(budget=VLLM_V1_R_KV_BUDGET)
```

---

## 5. 实现步骤

### Step 1: 创建 SpeckVvLLM 类

```bash
# 文件：R-KV/HuggingFace/rkv/compression/speckv_vllm.py
# 内容：实现 update_kv() 接口的 SpeckV
```

**子任务**：
- [ ] 从 `speckv_rkv_style.py` 提取打分逻辑
- [ ] 适配 `[batch, heads, seq, dim]` 输入格式
- [ ] 实现 `update_kv()` 返回压缩后的 K/V
- [ ] 添加 stats 加载和验证

### Step 2: 更新 rkv/compression/__init__.py

```python
from .speckv_vllm import SpeckVvLLM
__all__ = [..., "SpeckVvLLM"]
```

### Step 3: 修改 flash_attn.py

**子任务**：
- [ ] 添加环境变量读取
- [ ] 在 `__init__` 中添加算法选择分支
- [ ] 确保默认仍为 R1KV

### Step 4: 单元测试

```python
# tests/test_speckv_vllm.py
def test_update_kv_interface():
    """验证接口兼容性"""
    compressor = SpeckVvLLM(budget=128, stats_path="...")
    k, v = compressor.update_kv(key, query, value)
    assert k.shape[2] <= 128
    assert v.shape[2] <= 128

def test_compression_quality():
    """验证压缩质量"""
    # 对比 HF 路径结果
```

### Step 5: 端到端验证

```bash
# 运行 vLLM 服务
export VLLM_COMPRESSION_ALGO=speckv
export VLLM_SPECKV_STATS_PATH=/path/to/stats.pt
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B

# 发送测试请求
curl http://localhost:8000/v1/completions -d '{"prompt": "...", "max_tokens": 100}'
```

---

## 6. 测试计划

### 6.1 单元测试

| 测试 | 验证内容 | 预期结果 |
|-----|---------|---------|
| 接口兼容性 | `update_kv()` 输入输出格式 | 与 R1KV 一致 |
| 边界条件 | `seq_len <= budget` | 直接返回原始 K/V |
| 压缩比 | 输出长度 | `<= budget` |

### 6.2 集成测试

| 测试 | 场景 | 预期结果 |
|-----|------|---------|
| 默认配置 | 不设置环境变量 | 使用 R1KV |
| SpeckV 配置 | 设置 `VLLM_COMPRESSION_ALGO=speckv` | 使用 SpeckV |
| 算法切换 | 修改环境变量 | 正确切换 |

### 6.3 准确率测试

| 测试 | 数据集 | 预期结果 |
|-----|-------|---------|
| SpeckV vLLM vs HF | AIME24 | 差异 < 1% |
| SpeckV vs R1KV | AIME24 | 记录对比 |

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|-----|------|------|------|
| PagedAttention 布局不兼容 | 中 | 高 | 参考 R1KV 的处理方式 |
| Stats 文件格式不匹配 | 低 | 中 | 复用现有 stats_utils 验证 |
| GQA head 映射问题 | 中 | 中 | 与 HF 路径对比验证 |
| 性能不达预期 | 中 | 低 | Phase 0 重点是正确性 |

---

## 8. Phase 0 完成标准

```
□ SpeckVvLLM 类实现 update_kv() 接口
□ 集成到 flash_attn.py，通过环境变量选择
□ 默认配置下 R1KV 行为不变
□ 单元测试通过
□ AIME24 准确率与 HF 路径差异 < 1%
□ 代码通过 review
```

---

## 9. 文件清单

### 9.1 新建文件

| 文件 | 说明 |
|-----|------|
| `R-KV/HuggingFace/rkv/compression/speckv_vllm.py` | SpeckV vLLM 接口实现 |
| `tests/test_speckv_vllm.py` | 单元测试 |

### 9.2 修改文件

| 文件 | 修改内容 |
|-----|---------|
| `rkv/compression/__init__.py` | 导出 SpeckVvLLM |
| `vLLM/vllm/v1/attention/backends/flash_attn.py` | 添加算法选择分支 |
| `vLLM/vllm/envs.py` | 添加 SpeckV 相关环境变量 |

### 9.3 参考文件（只读）

| 文件 | 用途 |
|-----|------|
| `rkv/compression/r1_kv.py` | R1KV 实现参考 |
| `weian_development/speckv/speckv_rkv_style.py` | SpeckV 打分逻辑参考 |
| `weian_development/speckv/round_pruning_utils.py` | 打分函数参考 |

---

*创建日期：2025-01-31*
*更新日期：2025-01-31（重写为 vLLM 集成路径）*
