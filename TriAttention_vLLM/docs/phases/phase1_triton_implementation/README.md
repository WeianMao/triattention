# Phase 1: TriAttention 独立 Triton 实现

## 概述

完全独立的 SpeckV (TriAttention) 实现，使用 Triton kernel 优化核心操作，目标达到生产级效率。

---

## 1. 目标与约束

### 1.1 目标

| 目标 | 说明 | 优先级 |
|-----|------|--------|
| **Triton 级别效率** | Phase 1 以正确性为主；2-3x 目标留到 Phase 2（引入 Triton TopK/Gather 后评估） | P0 |
| **Batch Size > 1** | 支持批量推理，提高吞吐量 | P0 |
| **独立开发** | 不依赖 R-KV 框架，独立代码库 | P0 |
| vLLM 集成 | 与 vLLM 0.15.x 非侵入式集成 | P1 |
| CUDA Graph 兼容 | 支持 CUDA Graph 优化（Phase 2 预留） | P2 |

### 1.2 约束

| 约束 | 说明 |
|-----|------|
| 仅支持 RoPE 模型 | Qwen, LLaMA, DeepSeek, Mistral |
| 不修改 vLLM 核心 | 所有代码在 `TriAttention_vLLM/` 目录 |
| Triton 2.0+ | 依赖 Triton 的 JIT 编译和自动调优 |

### 1.3 与 Phase 0 的对比

| 方面 | Phase 0 | Phase 1 |
|-----|---------|---------|
| 框架 | R-KV/HuggingFace | TriAttention 独立 |
| Batch Size | = 1 | > 1 |
| 核心操作 | PyTorch 原生 | Triton kernel |
| 效率 | 不追求 | 2-3x 于 R-KV |
| 集成 | HuggingFace generate | vLLM 主路径 |

---

## 1.4 阶段范围定义（核心 vs 边界）

**Phase 1 = 核心实现（常见场景）**：
- 在 TriAttention_vLLM 目录内完成 Triton 版本核心实现
- 覆盖最常见的推理路径（稳定、常见场景）
- 不处理复杂边界情况（见 Phase 2）

**Phase 2 = 边界情况与鲁棒性**：
- prefill > budget
- 混合 prefill/decode（含 chunked prefill）
- request 取消 / slot 复用 / position_indices 重置
- 内存触发压缩（preemption 之前介入）
- CUDA Graph 兼容性与长时间稳定性

---

## 1.5 关键文档依赖（必须遵守）

Phase 1 的实现必须对齐以下文档中的需求与方案：

- `TriAttention_vLLM/docs/README.md`（总体目标与范围）
- `TriAttention_vLLM/docs/design/algorithm.md`（算法逻辑）
- `TriAttention_vLLM/docs/design/optimization.md`（打分优化必须应用）
- `TriAttention_vLLM/docs/implementation/`（数据结构与集成细节）
- `TriAttention_vLLM/docs/project/`（项目级规划与关键决策）
- `TriAttention_vLLM/docs/r-kv-analysis/`（可复用内容与约束对齐）

若 Phase 1 计划与上述内容冲突，以上述文档为准，并在本计划中显式记录偏差与理由。

---

## 1.6 与关键文档对齐的要点（摘要）

- **架构约束**（`docs/README.md`）：非侵入式、命名为 TriAttention、目标 vLLM 0.15.x。
- **对齐基线**（`project/roadmap.md`）：严格对齐 3 个 R-KV 脚本（打分公式、offsets、divide_length、默认参数）。
- **阶段 1 必须项**（`project/key_decisions.md` / `r-kv-analysis/`）：
  - batch > 1 必须支持（或显式报错）。
  - **打分使用 Triton**；TopK/Gather 阶段 1 先用 `torch.topk/torch.gather`。
  - 不使用 query cache；不使用 noise injection。
  - 提供状态重置接口（与 R-KV 行为对齐）。
**数据结构**（`implementation/data_structures.md`）：
  - `position_indices` 必需，阶段 1 使用 bf16/int32（优先 int32）。
  - per-head 需要 2D 位置表（按 head 维度）。
  - 额外数据结构需按 page/block 组织，以便后续 vLLM 集成。
- **集成约束**（`implementation/vllm_integration.md` / `fill_in_place.md`）：
  - prefill 阶段不压缩，prefill 结束后再裁剪。
  - 预算/overflow 的填充遵循 Fill-in-Place 思路。
  - RoPE 使用真实序列位置（`actual_seq_len`）。

---

## 1.7 已记录的偏差（需确认）

**Prefill > Budget 处理**：`docs/README.md` 要求 prefill 结束后立即裁剪；  
但 `project/CLARIFICATIONS_NEEDED.md` 将该场景标记为“阶段 2 处理”。  
**处理方式**：需明确阶段 1 是否覆盖该场景；若不覆盖，需同步更新需求文档。

---

## 2. 架构设计

### 2.1 整体架构

```
TriAttention_vLLM/
├── triattention/
│   ├── __init__.py
│   ├── config.py               # 配置类
│   ├── compressor.py           # 主压缩器类
│   ├── scoring.py              # 打分逻辑（Python wrapper）
│   ├── state.py                # 状态管理
│   ├── utils.py                # 工具函数
│   │
│   ├── kernels/                # Triton kernels
│   │   ├── __init__.py
│   │   ├── scoring_kernel.py   # 打分 kernel（Phase 1 必需）
│   │   ├── topk_kernel.py      # TopK kernel（Phase 2 评估）
│   │   ├── gather_kernel.py    # Gather kernel（Phase 2 评估）
│   │   └── fused_kernel.py     # 融合 kernel（Phase 2 评估）
│   │
│   └── integration/            # 框架集成
│       ├── __init__.py
│       ├── hf_integration.py   # HuggingFace 集成
│       └── vllm_integration.py # vLLM 基础集成（Phase 1 主路径）
│
├── stats/                      # 预计算统计
│   └── loader.py               # Stats 加载器
│
└── tests/                      # 测试套件
    ├── test_kernels.py
    ├── test_compressor.py
    └── benchmarks/
        └── benchmark_kernels.py
```

### 2.2 核心类设计

#### 2.2.1 配置类

```python
# triattention/config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Literal
import torch

@dataclass
class TriAttentionConfig:
    """TriAttention 配置"""

    # ===== 必需参数 =====
    budget: int                                    # KV 缓存预算
    stats_path: Path                               # 预计算统计文件
    model_path: Path                               # 模型路径（用于 RoPE）

    # ===== 基础参数 =====
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    dtype: torch.dtype = torch.float16             # 计算精度
    divide_length: int = 128                       # 压缩间隔

    # ===== 裁剪模式 =====
    pruning_mode: Literal["global", "per_head", "per_layer", "per_layer_per_head"] = "per_head"

    # ===== 打分参数 =====
    offset_max_length: int = 65536                 # 最大 offset
    score_aggregation: Literal["mean", "max"] = "mean"
    disable_top_n_high_freq: int = 0               # 禁用高频分量
    disable_mlr: bool = False                      # 禁用 MLR 项
    disable_trig: bool = False                     # 禁用三角项

    # ===== 可选功能 =====
    protect_prefill: bool = False                  # 保护 prefill token
    window_size: int = 0                           # 硬保留最近 N 个 token（0=不保护）
    normalize_scores: bool = False                 # Z-score 标准化
    seed: Optional[int] = None                     # 随机种子

    # ===== Triton 参数 =====
    triton_block_size: int = 128                   # Triton 块大小
    use_fused_kernel: bool = True                  # 使用融合 kernel

    # ===== RoPE 参数（自动检测） =====
    rope_style: Literal["half", "interleaved"] = "half"
    head_dim: Optional[int] = None                 # 自动从模型检测
    num_kv_heads: Optional[int] = None             # 自动从模型检测

    def __post_init__(self):
        if isinstance(self.stats_path, str):
            self.stats_path = Path(self.stats_path)
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
```

#### 2.2.2 主压缩器类

```python
# triattention/compressor.py

import torch
from typing import Tuple, Optional, Dict
from .config import TriAttentionConfig
from .state import CompressionState
from .scoring import compute_scores
from .kernels import fused_topk_gather, topk_select, gather_kv

class TriAttentionCompressor:
    """TriAttention KV 缓存压缩器"""

    def __init__(self, config: TriAttentionConfig):
        self.config = config
        self.state = CompressionState(config)

        # 加载预计算统计
        self._load_stats()

        # 初始化 RoPE
        self._init_rope()

        # 预计算频率缩放
        self._precompute_freq_scale()

    def compress(
        self,
        key_states: torch.Tensor,      # [batch, num_kv_heads, seq_len, head_dim]
        value_states: torch.Tensor,    # [batch, num_kv_heads, seq_len, head_dim]
        cache_positions: torch.Tensor, # [batch, seq_len] 或 [seq_len]（广播）
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行 KV 缓存压缩

        Args:
            key_states: K 缓存
            value_states: V 缓存
            cache_positions: 每个 token 的原始位置（用于 RoPE）

        Returns:
            (compressed_keys, compressed_values, new_positions)
        """
        batch_size, num_kv_heads, seq_len, head_dim = key_states.shape

        # 1. 检查是否需要压缩
        if seq_len <= self.config.budget:
            return key_states, value_states, cache_positions

        # 2. 计算打分
        scores = compute_scores(
            key_states=key_states,
            cache_positions=cache_positions,
            head_stats=self.head_stats,
            omega=self.omega,
            offsets=self.offsets,
            freq_scale_sq=self.freq_scale_sq,
            config=self.config,
        )  # [batch, num_kv_heads, seq_len] 或 [batch, seq_len]

        # 3. 选择保留的 token 并 Gather
        if self.config.use_fused_kernel:
            # 融合 TopK + Gather
            k_compressed, v_compressed, new_positions = fused_topk_gather(
                key_states, value_states, cache_positions, scores,
                k=self.config.budget,
                window_size=self.config.window_size,
            )
        else:
            # 分离操作
            keep_indices = topk_select(
                scores,
                k=self.config.budget - self.config.window_size,
                window_size=self.config.window_size,
            )
            k_compressed, v_compressed = gather_kv(
                key_states, value_states, keep_indices
            )
            new_positions = cache_positions.gather(-1, keep_indices)

        return k_compressed, v_compressed, new_positions

    def reset(self) -> None:
        """重置状态（新序列）"""
        self.state.reset()

    # ===== 私有方法 =====

    def _load_stats(self) -> None:
        """加载预计算统计"""
        from ..stats.loader import load_head_frequency_stats
        self.metadata, self.head_stats = load_head_frequency_stats(
            self.config.stats_path, self.config.device
        )

    def _init_rope(self) -> None:
        """初始化 RoPE"""
        # 从模型配置获取 inv_freq
        pass

    def _precompute_freq_scale(self) -> None:
        """预计算频率缩放因子"""
        pass
```

#### 2.2.3 状态管理

```python
# triattention/state.py

import torch
from typing import Optional, List, Dict, Tuple
from .config import TriAttentionConfig

class CompressionState:
    """压缩状态管理"""

    def __init__(self, config: TriAttentionConfig):
        self.config = config

        # 全局状态
        self.absolute_position: int = 0
        self.compression_count: int = 0
        self.prefill_length: int = 0

        # 位置追踪（根据 pruning_mode 选择）
        self.cache_positions: Optional[torch.Tensor] = None
        self.cache_positions_per_head: Optional[torch.Tensor] = None
        self.cache_positions_per_layer: Optional[Dict[int, torch.Tensor]] = None

    def reset(self) -> None:
        """重置状态"""
        self.absolute_position = 0
        self.compression_count = 0
        self.prefill_length = 0
        self.cache_positions = None
        self.cache_positions_per_head = None
        self.cache_positions_per_layer = None

    def should_compress(self, current_len: int) -> bool:
        """判断是否应该触发压缩"""
        effective_len = current_len
        if not self.config.protect_prefill:
            pass  # 使用全部长度
        else:
            effective_len = max(0, current_len - self.prefill_length)

        return (
            effective_len > self.config.budget
            and self.absolute_position % self.config.divide_length == 0
        )

    def update_positions(
        self,
        keep_indices: torch.Tensor,
        original_positions: torch.Tensor,
    ) -> torch.Tensor:
        """更新位置追踪"""
        # 根据 pruning_mode 更新不同的位置数据结构
        pass
```

---

## 3. Triton Kernel 设计

### 3.1 设计原则

1. **融合操作**：减少全局内存往返
2. **块大小自适应**：使用 `@triton.autotune`
3. **数值稳定**：使用 FP32 进行关键计算
4. **批量处理**：原生支持 batch 维度

### 3.2 打分 Kernel

```python
# triattention/kernels/scoring_kernel.py

import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_F': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_F': 32}, num_warps=8),
        triton.Config({'BLOCK_N': 256, 'BLOCK_F': 64}, num_warps=8),
    ],
    key=['seq_len', 'freq_count'],
)
@triton.jit
def speckv_scoring_kernel(
    # 输入
    K_ptr,                    # [batch, num_heads, seq_len, head_dim]
    positions_ptr,            # [batch, seq_len] 或 [seq_len]
    q_mean_real_ptr,          # [num_sampled_heads, freq_count]
    q_mean_imag_ptr,          # [num_sampled_heads, freq_count]
    q_abs_mean_ptr,           # [num_sampled_heads, freq_count]
    omega_ptr,                # [freq_count]
    offsets_ptr,              # [num_offsets]
    freq_scale_sq_ptr,        # [freq_count]
    cos_table_ptr,            # [max_pos, freq_count]
    sin_table_ptr,            # [max_pos, freq_count]
    # 输出
    scores_ptr,               # [batch, num_heads, seq_len]
    # 维度
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    freq_count,
    num_offsets,
    round_start,
    # 常量
    BLOCK_N: tl.constexpr,    # 序列块大小
    BLOCK_F: tl.constexpr,    # 频率块大小
    DISABLE_TRIG: tl.constexpr,
):
    """
    SpeckV 打分 kernel

    对每个 (batch, head) 计算所有 token 的重要性分数
    """
    # 获取程序 ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_seq = tl.program_id(2)

    # 计算当前块处理的序列范围
    seq_start = pid_seq * BLOCK_N
    seq_offsets = seq_start + tl.arange(0, BLOCK_N)
    seq_mask = seq_offsets < seq_len

    # 1. 加载 K 值并执行 RoPE 反演
    # ... (RoPE inversion in-kernel)

    # 2. 转换到复数域
    # k_complex = to_complex_pairs(k_unrot)

    # 3. 加载 Q 统计
    # q_mean_complex, q_abs_mean

    # 4. 计算 amp 和 phi
    # amp = |q_mean| * |k|
    # phi = atan2(...)

    # 5. 计算位置相关评分
    # base_delta = round_start - positions
    # phase = base_delta * omega + phi
    # base_scores = (amp * freq_scale_sq * cos(phase)).sum()

    # 6. 添加位置无关项
    # additive = (extra * freq_scale_sq).sum()

    # 7. 聚合并存储
    # scores = base_scores + additive (mean over offsets)
    # tl.store(scores_ptr + ..., scores, mask=seq_mask)


def compute_scores_triton(
    key_states: torch.Tensor,
    cache_positions: torch.Tensor,
    head_stats: dict,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    config,
) -> torch.Tensor:
    """
    Python wrapper for Triton scoring kernel
    """
    batch_size, num_heads, seq_len, head_dim = key_states.shape
    freq_count = omega.numel()
    num_offsets = offsets.numel()

    # 分配输出
    scores = torch.empty(batch_size, num_heads, seq_len,
                        device=key_states.device, dtype=torch.float32)

    # 计算 grid
    grid = (batch_size, num_heads, triton.cdiv(seq_len, 128))

    # 调用 kernel
    speckv_scoring_kernel[grid](
        key_states, cache_positions,
        # ... 其他参数
        batch_size, num_heads, seq_len, head_dim, freq_count, num_offsets,
        round_start=config.absolute_position,
        BLOCK_N=128,
        BLOCK_F=32,
        DISABLE_TRIG=config.disable_trig,
    )

    return scores
```

### 3.3 候选 Triton TopK + Gather Kernel（Phase 2 评估）

> 说明：Phase 1 采用 PyTorch TopK/Gather，这里的 Triton 融合方案作为 Phase 2 的性能优化备选。

```python
# triattention/kernels/fused_kernel.py

import triton
import triton.language as tl
import torch

@triton.jit
def fused_topk_gather_kernel(
    # 输入
    K_ptr,                    # [batch, num_heads, seq_len, head_dim]
    V_ptr,                    # [batch, num_heads, seq_len, head_dim]
    positions_ptr,            # [batch, seq_len]
    scores_ptr,               # [batch, num_heads, seq_len] 或 [batch, seq_len]
    # 输出
    K_out_ptr,                # [batch, num_heads, budget, head_dim]
    V_out_ptr,                # [batch, num_heads, budget, head_dim]
    positions_out_ptr,        # [batch, budget]
    # 维度
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    budget,
    window_size,
    # 常量
    BLOCK_K: tl.constexpr,    # TopK 块大小
    BLOCK_D: tl.constexpr,    # head_dim 块大小
    PER_HEAD: tl.constexpr,   # 是否 per-head 选择
):
    """
    融合 TopK 选择 + K,V Gather

    步骤：
    1. 加载 scores 块
    2. 块内 partial sort 找 top-k
    3. 直接 gather K, V 到输出
    4. 更新位置索引
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1) if PER_HEAD else 0

    # 1. 分块读取 scores
    # 使用 bitonic sort 或 radix sort 实现块内 topk

    # 2. 跨块合并找全局 top-k
    # 需要多轮迭代或使用 atomics

    # 3. 根据选择的索引 gather K, V
    # 流水线化：compute → load → store

    # 4. 处理 window_size（保护最近 token）
    # 最后 window_size 个 token 无条件保留


def fused_topk_gather(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_positions: torch.Tensor,
    scores: torch.Tensor,
    k: int,
    window_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Python wrapper for fused TopK + Gather kernel
    """
    batch_size, num_heads, seq_len, head_dim = key_states.shape

    # 分配输出
    k_out = torch.empty(batch_size, num_heads, k, head_dim,
                       device=key_states.device, dtype=key_states.dtype)
    v_out = torch.empty_like(k_out)
    positions_out = torch.empty(batch_size, k,
                               device=key_states.device, dtype=torch.int32)

    # 计算 grid
    grid = (batch_size, num_heads)

    # 调用 kernel
    fused_topk_gather_kernel[grid](
        key_states, value_states, cache_positions, scores,
        k_out, v_out, positions_out,
        batch_size, num_heads, seq_len, head_dim, k, window_size,
        BLOCK_K=256,
        BLOCK_D=64,
        PER_HEAD=True,
    )

    return k_out, v_out, positions_out
```

### 3.4 TopK 选择策略（Phase 1 决策）

**当前明确决策**：Phase 1 **TopK/Gather 使用 PyTorch（`torch.topk` / `torch.gather`）**。  
Triton TopK/Gather 是否实现，放到 Phase 2，根据性能收益再决定。

由于 Triton 没有内置的高效 TopK，需要实现自定义方案：

#### 方案 A: 分块 Partial Sort

```python
@triton.jit
def block_partial_sort(
    data_ptr,
    indices_ptr,
    n,
    k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    块内 partial sort 实现 top-k

    1. 每个块内排序
    2. 保留每个块的 top-k 候选
    3. 归约找全局 top-k
    """
    pass
```

#### 方案 B: Threshold-based 过滤

```python
@triton.jit
def threshold_topk(
    scores_ptr,
    k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    基于阈值的 top-k 选择

    1. 估计第 k 大的阈值（通过采样或直方图）
    2. 并行过滤 > threshold 的元素
    3. 处理边界情况
    """
    pass
```

#### 方案 C: 混合方案（推荐）

```python
def hybrid_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    混合 TopK 策略

    - 小规模 (seq_len < 4096): 使用 torch.topk
    - 大规模 (seq_len >= 4096): 使用 Triton 分块排序（Phase 2 决定）
    """
    if scores.shape[-1] < 4096:
        return torch.topk(scores, k, dim=-1).indices
    else:
        return triton_block_topk(scores, k)
```

---

## 4. 性能优化策略

### 4.1 内存访问优化

| 优化 | 说明 | 实现 |
|-----|------|------|
| 合并访问 | K, V 连续读取 | 调整 layout 或使用 `tl.load` 合并 |
| 共享内存 | 频繁使用的数据缓存 | `tl.constexpr` 数组 |
| 流水线 | 计算-访存重叠 | `tl.load` 的 `num_stages` |

### 4.2 计算优化

| 优化 | 说明 | 实现 |
|-----|------|------|
| FP32 精度 | TopK 使用 FP32 | `scores.to(tl.float32)` |
| 向量化 | 使用 SIMD 指令 | Triton 自动向量化 |
| 融合操作 | 减少 kernel launch | `fused_topk_gather_kernel` |

### 4.3 自动调优

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=4),
    ],
    key=['seq_len', 'head_dim'],
    warmup=25,
    rep=50,
)
```

---

## 5. 与 PyTorch 对比

### 5.1 R-KV PyTorch 实现（当前）

```python
# 6-8 次全局内存往返
scores = compute_attention_scores(Q, K)      # 读 Q, K，写 scores
scores_pooled = max_pool1d(scores)           # 读 scores，写 pooled
similarity = cal_similarity(K)               # 读 K，写 similarity
final_score = mix_scores(scores, similarity) # 读 2 个，写 1 个
indices = torch.topk(final_score, k)         # 读 scores，写 indices
k_out = K.gather(indices)                    # 读 K, indices，写 k_out
v_out = V.gather(indices)                    # 读 V, indices，写 v_out
```

### 5.2 TriAttention Triton 实现（目标）

```python
# 2-3 次全局内存往返
# Kernel 1: 打分（读 K, stats，写 scores）
scores = speckv_scoring_kernel(K, stats, ...)

# Kernel 2: PyTorch TopK + Gather
indices = torch.topk(scores, k, dim=-1).indices
k_out = K.gather(dim=2, index=indices)
v_out = V.gather(dim=2, index=indices)
```

### 5.3 预期加速（Phase 2 目标）

> 说明：Phase 1 使用 PyTorch TopK/Gather，以下 2-3x 目标是引入 Triton TopK/Gather 后的预期。

| 操作 | PyTorch | Triton | 加速比 |
|-----|---------|--------|-------|
| 打分 | 2-3 kernels | 1 kernel | 1.5-2x |
| TopK + Gather | 3 kernels | 1 kernel | 2-3x |
| **整体** | 6-8 往返 | 2-3 往返 | **2-3x** |

---

## 6. 测试计划

### 6.1 正确性测试

```python
# tests/test_kernels.py

def test_scoring_kernel_correctness():
    """验证 Triton 打分与 PyTorch 实现一致"""
    # 使用相同输入，比较输出
    # 允许 FP32 精度内的误差 (rtol=1e-4, atol=1e-5)

def test_topk_gather_correctness():
    """验证 TopK + Gather 正确性"""
    # 比较 Triton vs torch.topk + gather

def test_batch_processing():
    """验证批处理正确性"""
    # batch_size > 1 时结果与逐个处理一致

def test_edge_cases():
    """边界情况测试"""
    # seq_len < budget, seq_len == budget
    # k == seq_len (全部保留)
    # window_size > 0
```

### 6.2 性能测试

```python
# tests/benchmarks/benchmark_kernels.py

def benchmark_scoring():
    """打分 kernel 性能测试"""
    configs = [
        (1, 32, 2048, 128),   # batch=1, heads=32, seq=2048, dim=128
        (1, 32, 8192, 128),   # 长序列
        (8, 32, 2048, 128),   # batch=8
        (16, 32, 4096, 128),  # 大 batch + 长序列
    ]
    # 比较 Triton vs PyTorch 耗时

def benchmark_topk_gather():
    """TopK + Gather 性能测试"""
    # 不同 seq_len, k 值的性能

def benchmark_end_to_end():
    """端到端压缩性能"""
    # 完整 compress() 流程
```

### 6.3 集成测试

```python
def test_hf_integration():
    """HuggingFace 集成测试"""
    # 加载模型，执行生成，验证输出质量

def test_vllm_integration():
    """vLLM 集成测试（Phase 1 主路径）"""
    pass
```

---

## 7. 实现步骤

### Step 1: 基础框架

1. 创建目录结构
2. 实现配置类 `TriAttentionConfig`
3. 实现状态管理 `CompressionState`
4. 实现 Stats 加载器

### Step 2: Triton 打分 Kernel

1. 实现 RoPE 反演（in-kernel）
2. 实现频率统计计算
3. 实现位置相关打分
4. 添加自动调优

### Step 3: PyTorch TopK + Gather（Phase 1）

1. `torch.topk` 选择（保证正确性）
2. `torch.gather` 先跑通
3. 处理 window_size 保护
4. 与打分 kernel 联调

### Step 4: 集成与优化

1. 集成到主压缩器
2. 性能调优（autotune 参数）
3. HuggingFace 集成

### Step 5: 测试与验证

1. 正确性测试
2. 性能 benchmark
3. AIME 数据集评估

---

## 8. 风险与缓解

### 8.1 技术风险

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| TopK/Gather 性能不理想 | 加速比不达预期 | Phase 2 再评估 Triton TopK/Gather |
| RoPE in-kernel 复杂 | 开发周期延长 | 先用预计算表，后续优化 |
| 数值精度问题 | 结果不一致 | FP32 打分 + 详细测试 |

### 8.2 备选方案

1. **打分 Kernel**：先实现简化版本，再逐步引入优化
2. **TopK/Gather**：Phase 1 用 PyTorch；Phase 2 评估是否 Triton 化
3. **融合 Kernel**：如果融合效果不好，分离为独立 kernels

---

## 9. 与 Phase 2（边界情况）的接口预留

### 9.1 vLLM 集成点

```python
# triattention/integration/vllm_integration.py (Phase 1 基础集成，Phase 2 扩展边界处理)

class TriAttentionVLLMBackend:
    """vLLM Attention Backend 集成"""

    def __init__(self, compressor: TriAttentionCompressor):
        self.compressor = compressor

    def forward_decode(self, q, k, v, ...):
        """解码阶段前向传播"""
        # 在 attention 计算前插入压缩
        pass

    def forward_prefill(self, q, k, v, ...):
        """预填充阶段前向传播"""
        pass
```

### 9.2 CUDA Graph 兼容

```python
# 预留 CUDA Graph 兼容接口
class TriAttentionCompressor:
    def capture_graph(self):
        """捕获 CUDA Graph"""
        pass

    def replay_graph(self):
        """重放 CUDA Graph"""
        pass
```

---

*创建日期：2025-01-31*
