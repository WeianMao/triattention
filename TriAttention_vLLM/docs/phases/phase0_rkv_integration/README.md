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
    │   ├── self.kvcompressor = SpeckVvLLM(...) or R1KV(...)  # 根据配置选择
    │   └── self.position_indices = None  # SpeckV 需要的位置追踪（统一管理）
    │
    └── FlashAttentionImpl.forward()
        ├── [复用现有压缩触发逻辑]
        └── [SpeckV 需额外维护 position_indices：压缩时同步更新]
```

> **注意**：SpeckV 与 R1KV 的区别在于需要维护 `position_indices` 来追踪 token 的原始位置。
> 这个追踪由 `FlashAttentionImpl` 统一管理，压缩器本身无状态。

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

**文件**：`R-KV/HuggingFace/rkv/compression/r1_kv.py`

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
# (统一导入：from rkv.modeling import SpeckVvLLM，不要直接从 compression 导入)

class SpeckVvLLM:
    """SpeckV 压缩器 - vLLM 接口版本

    设计原则：
    1. update_kv() 接口与 R1KV 完全一致，只返回 (K, V)
    2. position_indices 由 FlashAttentionImpl 维护，通过参数传入打分
    3. 保留索引通过 get_last_keep_indices() 暴露，供外部更新位置表
    4. layer_idx 采用 lazy init（首次 forward 时初始化）
    """

    _shared_stats = None  # 类级别 stats 缓存（避免重复加载）

    def __init__(
        self,
        budget: int = 128,
        stats_path: str = None,           # 预计算统计文件
        pruning_mode: str = "per_layer_perhead",  # Phase 0 仅支持 per_layer_perhead
        **kwargs,
    ):
        self.budget = budget
        self.stats_path = stats_path
        self.pruning_mode = pruning_mode

        # lazy init：首次 forward 时从 layer.layer_idx 获取
        self.layer_idx = None
        self._initialized = False

        # 本次压缩保留的索引（供外部获取）
        self._last_keep_indices = None

        # 共享三角函数表（惰性初始化）
        self.trig_cos = None
        self.trig_sin = None

    def _lazy_init(self, layer_idx: int):
        """首次 forward 时调用，初始化 layer 相关状态"""
        if self._initialized:
            return

        self.layer_idx = layer_idx
        full_stats = self._load_shared_stats(self.stats_path)

        # 只索引当前层的 stats
        self.q_mean = full_stats["q_mean_complex"][layer_idx]
        self.freq_scale_sq = full_stats["freq_scale_sq"][layer_idx]
        self._initialized = True

    @classmethod
    def _load_shared_stats(cls, stats_path):
        """类级别 stats 加载（所有层共享，避免重复 I/O）"""
        if cls._shared_stats is None:
            cls._shared_stats = torch.load(stats_path)
        return cls._shared_stats

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        position_indices: torch.Tensor = None,  # 打分用，由 FlashAttentionImpl 传入
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        SpeckV 压缩逻辑。

        接口与 R1KV 完全一致：只返回 (compressed_key, compressed_value)。
        保留索引通过 get_last_keep_indices() 获取。

        Args:
            key_states: [batch, num_kv_heads, seq_len, head_dim]
            query_states: [batch, num_heads, seq_len, head_dim]
            value_states: [batch, num_kv_heads, seq_len, head_dim]
            position_indices: [seq_len] 原始位置（打分用）

        Returns:
            (compressed_key, compressed_value) - 与 R1KV 接口一致
        """
        self._last_keep_indices = None  # 重置

        if key_states.shape[2] <= self.budget:
            return key_states, value_states

        scores = self._compute_scores_optimized(key_states, position_indices)
        keep_indices = self._select_tokens(scores)

        # 保存保留索引，供外部更新 position_indices
        self._last_keep_indices = keep_indices

        compressed_key = self._gather(key_states, keep_indices)
        compressed_value = self._gather(value_states, keep_indices)

        return compressed_key, compressed_value

    def get_last_keep_indices(self) -> torch.Tensor:
        """获取上一次压缩保留的索引（只读）

        用途：FlashAttentionImpl 调用此方法更新 position_indices
        """
        return self._last_keep_indices

    def _compute_scores_optimized(self, key_states, position_indices):
        """优化的打分计算（无三角函数调用）"""
        ...

    def _select_tokens(self, scores):
        """根据 pruning_mode 选择 token"""
        ...
```

### 4.2 vLLM 集成修改（occupied_slot_mapping 方案）

> **注意**：flash_attn.py 中没有 `request_ids` 可用，使用 `occupied_slot_mapping` 实现 per-request 隔离。
> `seq_starts_ends_indices` 由 `seq_lens` 的前缀和得到（长度 = `num_reqs + 1`），用于切分每个请求的 slot 区间。

```python
# vLLM/vllm/v1/attention/backends/flash_attn.py

from rkv.modeling import R1KV, SpeckVvLLM

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

        # SpeckV 专用：全局 position_indices（与 KV cache 同布局）
        self.position_indices = None  # [num_blocks, block_size]，惰性初始化
        self.prefill_lens = {}        # slot_key -> prefill_len

    def forward(self, layer, ...):
        # SpeckV lazy init
        if isinstance(self.kvcompressor, SpeckVvLLM):
            self.kvcompressor._lazy_init(layer.layer_idx)
            if self.position_indices is None:
                self.position_indices = torch.full(
                    (num_blocks, block_size), -1, dtype=torch.long, device=device
                )

        # 压缩循环（与现有 R1KV 结构一致）
        for i in range(attn_metadata.num_reqs):
            seq_len = seq_lens[i]

            if seq_len < VLLM_V1_R_KV_BUDGET + VLLM_V1_R_KV_BUFFER:
                continue

            # 通过 occupied_slot_mapping 获取当前序列的 slots
            slots = occupied_slot_mapping[seq_starts_ends_indices[i]:seq_starts_ends_indices[i+1]]
            slot_key = slots[0].item()  # 用起始 slot 作为序列标识

            # 首次压缩：记录 prefill_len，初始化 position_indices
            if slot_key not in self.prefill_lens:
                self.prefill_lens[slot_key] = seq_len
                self._write_positions(slots, torch.arange(seq_len, device=device))

            # 提取当前序列的位置
            current_positions = self._read_positions(slots)

            # 压缩（接口与 R1KV 一致，只返回 K, V）
            compressed_key, compressed_value = self.kvcompressor.update_kv(
                current_key,
                current_query,
                current_value,
                position_indices=current_positions,  # 打分用
            )

            # SpeckV：用保留索引更新位置表
            if isinstance(self.kvcompressor, SpeckVvLLM):
                keep_indices = self.kvcompressor.get_last_keep_indices()
                if keep_indices is not None:
                    kept_positions = current_positions[keep_indices]
                    # 更新（需配合 vLLM block 管理）

            # 回写 KV cache
            key_cache[...] = compressed_key
            value_cache[...] = compressed_value

    def _read_positions(self, slots):
        """从全局 position_indices 读取当前序列的位置"""
        ...

    def _write_positions(self, slots, positions):
        """写入全局 position_indices"""
        ...
```

**清理策略**：
- `position_indices`：全局张量，block 回收后下次使用时覆盖，无需显式清理
- `prefill_lens`：以 `slot_key` 作为 key，**必须在 slot 复用时重置**，避免复用旧请求的 prefill_len
  - 最小实现：当检测到该 slot 尚未初始化（如 `position_indices` 为 -1）时，覆盖 `prefill_lens[slot_key]` 并重写位置表
  - 可选：在 scheduler 通知 request 完成时清理（hook 回调）

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
# R-KV/vLLM/tests/test_speckv_vllm.py

def test_update_kv_interface():
    """验证接口兼容性（与 R1KV 一致）"""
    compressor = SpeckVvLLM(budget=128, stats_path="...")
    compressor._lazy_init(layer_idx=0)  # 必须先初始化

    # 准备测试数据
    seq_len = 256
    key = torch.randn(1, 8, seq_len, 64)
    query = torch.randn(1, 8, seq_len, 64)
    value = torch.randn(1, 8, seq_len, 64)
    position_indices = torch.arange(seq_len)  # 显式传入

    # 测试：传入 position_indices
    k, v = compressor.update_kv(key, query, value, position_indices=position_indices)
    assert k.shape[2] <= 128
    assert v.shape[2] <= 128

    # 验证 keep_indices 可获取
    keep_indices = compressor.get_last_keep_indices()
    assert keep_indices is not None
    assert len(keep_indices) == k.shape[2]

def test_update_kv_fallback():
    """验证 position_indices=None 时的 fallback 行为"""
    compressor = SpeckVvLLM(budget=128, stats_path="...")
    compressor._lazy_init(layer_idx=0)

    key = torch.randn(1, 8, 256, 64)
    query = torch.randn(1, 8, 256, 64)
    value = torch.randn(1, 8, 256, 64)

    # position_indices=None 时，内部 fallback 到 torch.arange(seq_len)
    k, v = compressor.update_kv(key, query, value, position_indices=None)
    assert k.shape[2] <= 128

def test_compression_quality():
    """验证压缩质量（与 HF 路径对比）"""
    # 对比 HF 路径结果，需相同 stats、相同输入
```

> **position_indices 行为说明**：
> - 传入 `position_indices`：使用传入的位置进行打分（生产环境）
> - 传入 `None`：fallback 到 `torch.arange(seq_len)`（测试/debug 用）

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

**测试条件（确保可复现）**：

| 条件 | 值 | 说明 |
|-----|-----|------|
| Prompt 模板 | plain（无 chat wrapper） | 与 stats 采样时一致 |
| temperature | 0.0 | 确定性输出 |
| top_p | 1.0 | 不做 nucleus sampling |
| seed | 固定值（如 42） | 随机种子一致 |
| budget | 相同值 | vLLM 与 HF 路径使用相同 budget |
| 模型 | 同一 checkpoint | 避免版本差异 |

> **注意**：若 HF 路径用 chat 模板（如 Qwen2.5-Chat）而 vLLM 用 plain 模板，
> 结果差异可能超过 1%，这是模板差异而非算法差异。

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
| `R-KV/vLLM/tests/test_speckv_vllm.py` | 单元测试 |

### 9.2 修改文件

| 文件 | 修改内容 |
|-----|---------|
| `R-KV/HuggingFace/rkv/compression/__init__.py` | 导出 SpeckVvLLM |
| `vLLM/vllm/v1/attention/backends/flash_attn.py` | 添加算法选择分支 |
| `vLLM/vllm/envs.py` | 添加 SpeckV 相关环境变量 |

### 9.3 参考文件（只读）

| 文件 | 用途 |
|-----|------|
| `R-KV/HuggingFace/rkv/compression/r1_kv.py` | R1KV 实现参考 |
| `weian_development/speckv/speckv_rkv_style.py` | SpeckV 打分逻辑参考 |
| `weian_development/speckv/round_pruning_utils.py` | 打分函数参考 |

---

*创建日期：2025-01-31*
*更新日期：2025-01-31（重写为 vLLM 集成路径）*
