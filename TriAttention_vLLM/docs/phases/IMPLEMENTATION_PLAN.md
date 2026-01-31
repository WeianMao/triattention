# SpeckV/TriAttention 实现计划

## 概述

本文档定义 Phase 0 和 Phase 1 的具体实现计划和任务分解。

> **重要说明**：HuggingFace/transformers 路径（`apply_speckv_rkv_style_patch`）是**已完成的工作**。
> 当前 Phase 0 的目标是在 **R-KV/vLLM** 的 vLLM 框架里做快速集成验证。

---

## 0. 架构决策

### Phase 0 架构：vLLM attention backend 集成

**决策**：Phase 0 在 R-KV 的 vLLM fork 中集成 SpeckV，复用 R1KV 已有的 `update_kv()` 接口。

**集成点**：`vLLM/vllm/v1/attention/backends/flash_attn.py`

**原因**：
- R1KV 已在 vLLM v1 中实现，留有压缩接口
- 可以复用现有的触发机制和 KV cache 管理
- 为后续 Phase 1/2 的深度集成打基础

**影响**：
- 需要实现 `SpeckVvLLM` 类，提供 `update_kv()` 接口
- 通过环境变量选择压缩算法（默认仍为 R1KV）
- Phase 0 仅支持 `per_head` 模式（跨层模式留给 Phase 1）

---

## 1. 总体路线图

```
已完成 (HF 路径)             Phase 0 (vLLM 集成)         Phase 1 (Triton 实现)
        ↓                           ↓                           ↓
  speckv_rkv_style.py        SpeckVvLLM 类               高效独立版本
  monkey patch HF            update_kv() 接口            batch>1, Triton
  三种 pruning mode          per_head 模式               新接口设计
        ↓                           ↓                           ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                  Phase 2 (高级功能)                           │
    │           vLLM PagedAttention 深度集成、CUDA Graph            │
    └─────────────────────────────────────────────────────────────┘
```

**阶段关系**：
- **已完成**：HF 路径验证了 SpeckV 算法正确性
- **Phase 0**：在 vLLM 中快速集成验证，复用 R1KV 接口
- **Phase 1**：独立 Triton 实现，不依赖 R-KV 框架

---

## 2. Phase 0 实现计划（vLLM 集成）

### 2.1 目标

| 目标 | 验收标准 |
|-----|---------|
| 实现 `SpeckVvLLM` 类 | 提供与 R1KV 兼容的 `update_kv()` 接口 |
| 集成到 flash_attn.py | 可通过环境变量选择 SpeckV |
| 默认行为不变 | 不设置环境变量时使用 R1KV |
| 准确率验证 | 与 HF 路径结果差异 < 1% |

### 2.2 策略说明

**Phase 0 策略**：
1. 从 HF 路径（`speckv_rkv_style.py`）提取打分逻辑
2. 适配 vLLM 的 `update_kv(key, query, value)` 接口
3. 通过环境变量选择算法，保持隔离开发
4. 仅支持 `per_head` 模式（最简单，无跨层依赖）

### 2.3 任务分解

#### Task 0.1: 理解 R1KV vLLM 集成

```
子任务：
□ 阅读 vLLM/vllm/v1/attention/backends/flash_attn.py
□ 理解 R1KV 的 update_kv() 调用链
□ 理解 KV cache 的 PagedAttention 布局
□ 确认 attn_metadata 中可用的信息
```

**关键代码位置**：
- 压缩器初始化：`flash_attn.py` 第 431 行
- 压缩调用：`flash_attn.py` 第 547-588 行
- 环境变量：`vLLM/vllm/envs.py`

---

#### Task 0.2: 实现 SpeckVvLLM 类

**目标**：创建 `rkv/compression/speckv_vllm.py`

```
子任务：
□ 从 speckv_rkv_style.py 提取打分逻辑
□ 实现 __init__(budget, stats_path, ...)
□ 实现 update_kv(key_states, query_states, value_states)
□ 添加 stats 加载和验证
□ 处理 GQA head 映射
```

**接口要求**：
```python
class SpeckVvLLM:
    def update_kv(
        self,
        key_states: torch.Tensor,      # [batch, num_kv_heads, seq_len, head_dim]
        query_states: torch.Tensor,    # [batch, num_heads, seq_len, head_dim] (可忽略)
        value_states: torch.Tensor,    # [batch, num_kv_heads, seq_len, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回压缩后的 (key_states, value_states)"""
```

---

#### Task 0.3: 添加环境变量

**目标**：在 `vLLM/vllm/envs.py` 中添加 SpeckV 配置

```
新增变量：
□ VLLM_COMPRESSION_ALGO (默认 "r1kv")
□ VLLM_SPECKV_STATS_PATH (默认 None)
□ VLLM_SPECKV_PRUNING_MODE (默认 "per_head")
```

---

#### Task 0.4: 修改 flash_attn.py

**目标**：在压缩器初始化处添加算法选择

```
子任务：
□ 导入 SpeckVvLLM
□ 读取 VLLM_COMPRESSION_ALGO 环境变量
□ 添加 if/else 分支选择压缩器
□ 确保默认仍为 R1KV
```

**修改位置**：`vLLM/vllm/v1/attention/backends/flash_attn.py` 第 431 行附近

---

#### Task 0.5: 更新 rkv/compression/__init__.py

**目标**：导出 SpeckVvLLM

```python
from .speckv_vllm import SpeckVvLLM
__all__ = [..., "SpeckVvLLM"]
```

---

#### Task 0.6: 单元测试

**目标**：创建 `tests/test_speckv_vllm.py`

```
测试用例：
□ test_update_kv_interface() - 接口兼容性
□ test_no_compression_when_short() - seq_len <= budget 时不压缩
□ test_compression_ratio() - 输出长度 <= budget
□ test_consistency_with_hf() - 与 HF 路径结果对比
```

---

#### Task 0.7: 集成测试

**目标**：验证端到端功能

```
测试场景：
□ 默认配置下 R1KV 行为不变
□ 设置 VLLM_COMPRESSION_ALGO=speckv 后使用 SpeckV
□ vLLM 服务正常启动和响应
```

**命令**：
```bash
# 测试 SpeckV
export VLLM_COMPRESSION_ALGO=speckv
export VLLM_SPECKV_STATS_PATH=/path/to/stats.pt
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B
```

---

#### Task 0.8: 准确率验证

**目标**：与 HF 路径结果对比

```
验证内容：
□ 在 AIME24 上运行 SpeckV vLLM
□ 与 HF 路径结果对比
□ 差异应 < 1%
```

---

### 2.4 Phase 0 Checklist

```
Phase 0 完成标准（vLLM 集成）：
□ SpeckVvLLM 类实现 update_kv() 接口
□ 集成到 flash_attn.py，通过环境变量选择
□ 默认配置下 R1KV 行为不变
□ 单元测试通过
□ AIME24 准确率与 HF 路径差异 < 1%

隔离开发验证：
□ 不设置环境变量时使用 R1KV
□ 设置 VLLM_COMPRESSION_ALGO=speckv 时使用 SpeckV
□ R1KV 的测试/benchmark 结果不变
```

---

## 3. Phase 1 实现计划

### 3.1 目标

| 目标 | 验收标准 |
|-----|---------|
| Triton kernel 实现打分 | 与 PyTorch 结果一致 (rtol=1e-4) |
| Triton kernel 实现 TopK | 正确性 100% |
| 端到端性能 | 1.3-1.7x 加速 |
| 支持 batch > 1 | batch=8 正常运行 |

### 3.2 任务分解

#### Task 1.1: 创建项目结构

```
子任务：
□ 创建 triattention/ 目录结构
□ 创建 config.py - TriAttentionConfig
□ 创建 state.py - CompressionState
□ 创建 utils.py - 工具函数
□ 创建 stats/loader.py - Stats 加载器
```

---

#### Task 1.2: PyTorch 参考实现

**目的**：作为 Triton 实现的对照和 fallback

```
子任务：
□ 实现 compressor.py - TriAttentionCompressor
□ 实现 scoring.py - compute_scores_pytorch()
□ 实现 topk/gather 的 PyTorch 版本
□ 单元测试
```

---

#### Task 1.3: Triton 打分 Kernel

**依赖**：Task 1.2

```
子任务：
□ 实现 kernels/scoring_kernel.py
□ 实现 RoPE 反演（in-kernel 或预计算）
□ 实现频率统计计算
□ 实现位置相关打分
□ 添加 @triton.autotune
□ 与 PyTorch 版本对比验证
```

**关键设计决策**：
- cos/sin 表：预计算还是 in-kernel？
- 复数运算：实部/虚部分离
- 精度：打分用 FP32，存储用 FP16

---

#### Task 1.4: Triton TopK Kernel

**依赖**：Task 1.2

```
子任务：
□ 实现 kernels/topk_kernel.py
□ 评估不同 TopK 策略性能：
  - 方案 A: tl.sort() + 取前 k
  - 方案 B: 分块 partial sort
  - 方案 C: 混合策略（小规模 PyTorch，大规模 Triton）
□ 选择最优方案
□ 与 torch.topk 对比验证
```

---

#### Task 1.5: Triton Gather Kernel（可选）

**依赖**：Task 1.4

```
子任务：
□ 评估是否需要自定义 Gather（torch.gather 可能已足够快）
□ 如需要，实现 kernels/gather_kernel.py
□ 考虑与 TopK 融合
```

**决策点**：如果 torch.gather 性能足够，跳过此任务。

---

#### Task 1.6: 集成与优化

**依赖**：Task 1.3, Task 1.4

```
子任务：
□ 集成到 TriAttentionCompressor
□ 添加 kernel 选择逻辑（Triton vs PyTorch fallback）
□ 性能调优（autotune 参数、块大小）
□ 内存优化（减少中间结果）
```

---

#### Task 1.7: 测试与 Benchmark

**依赖**：Task 1.6

```
子任务：
□ 正确性测试（与 PyTorch 对比）
□ 边界情况测试
□ 性能 benchmark：
  - 不同 seq_len: 2K, 4K, 8K, 16K, 32K
  - 不同 batch_size: 1, 2, 4, 8
  - 不同 budget: 256, 512, 1024, 2048
□ 生成 benchmark 报告
```

---

#### Task 1.8: HuggingFace 集成

**依赖**：Task 1.6

```
子任务：
□ 实现 integration/hf_integration.py
□ 创建使用示例
□ 在 AIME24 上评估
```

---

### 3.3 Phase 1 Checklist

```
Phase 1 完成标准：
□ triattention/ 目录结构完整
□ Triton 打分 kernel 通过正确性测试
□ Triton TopK kernel 通过正确性测试
□ batch > 1 正常运行
□ 端到端性能 >= 1.3x（vs PyTorch baseline）
□ AIME24 准确率与 Phase 0 差异 < 0.5%
□ 代码通过 lint 检查
□ 有 benchmark 报告
```

---

## 4. 依赖关系图

```
Phase 0（理解 + 验证，无代码改动）:
Task 0.1 ──→ Task 0.2 ──→ Task 0.4 ──→ Task 0.6 ──→ Task 0.7
         └──→ Task 0.3 ──┘     │
                               └──→ Task 0.5 ──┘
                                        │
                                        └──→ Task 0.8 (可选)

Phase 1（Triton 重写）:
Task 1.1 ──→ Task 1.2 ──→ Task 1.3 ──→ Task 1.6 ──→ Task 1.7
                     └──→ Task 1.4 ──┘         └──→ Task 1.8
                     └──→ Task 1.5 ──┘ (可选)
```

---

## 5. 文件清单

### 5.1 Phase 0 文件（只读 + 新建文档）

| 文件 | 操作 | 说明 |
|-----|------|------|
| `weian_development/speckv/speckv_rkv_style.py` | 只读 | 理解主压缩器 |
| `weian_development/speckv/round_pruning_utils.py` | 只读 | 理解打分函数 |
| `weian_development/speckv/rkv_speckv_generate.py` | 只读 | 理解集成方式 |
| `weian_development/speckv/README.md` | **新建** | 使用文档 |
| `weian_script/aime_sampled8/speckv/aime24/*.sh` | 只读 | 运行验证 |

### 5.2 Phase 1 新建文件

| 文件 | 说明 |
|-----|------|
| `triattention/__init__.py` | 包初始化 |
| `triattention/config.py` | 配置类 |
| `triattention/compressor.py` | 主压缩器 |
| `triattention/scoring.py` | 打分逻辑 |
| `triattention/state.py` | 状态管理 |
| `triattention/utils.py` | 工具函数 |
| `triattention/kernels/__init__.py` | Kernel 包 |
| `triattention/kernels/scoring_kernel.py` | 打分 kernel |
| `triattention/kernels/topk_kernel.py` | TopK kernel |
| `triattention/kernels/gather_kernel.py` | Gather kernel |
| `triattention/integration/hf_integration.py` | HF 集成 |
| `triattention/stats/loader.py` | Stats 加载 |
| `tests/test_kernels.py` | Kernel 测试 |
| `tests/test_compressor.py` | 压缩器测试 |
| `tests/benchmarks/benchmark_kernels.py` | 性能测试 |

---

## 6. 风险与缓解

### 6.1 Phase 0 风险

| 风险 | 概率 | 影响 | 缓解 |
|-----|------|------|------|
| 现有脚本无法运行 | 低 | 高 | 检查环境、依赖、路径 |
| 代码过于复杂难以理解 | 中 | 中 | 分模块阅读，做笔记 |
| Stats 文件缺失或损坏 | 低 | 高 | 确认文件存在和格式 |
| 历史准确率记录缺失 | 中 | 低 | 重新运行获取基准 |

### 6.2 Phase 1 风险

| 风险 | 概率 | 影响 | 缓解 |
|-----|------|------|------|
| Triton TopK 效率不足 | 中 | 中 | 混合策略 |
| 性能不达 1.3x | 低 | 高 | 接受或降级 |
| 复数运算精度 | 低 | 高 | FP32 中间计算 |

---

*创建日期：2025-01-31*
