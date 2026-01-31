# 实施路线图

本文档描述 TriAttention 的实施阶段和代码结构规划。

---

## 1. 目录结构

```
TriAttention_vLLM/
├── docs/                           # 文档
│   ├── README.md                   # 项目概述与索引
│   ├── design/                     # 设计文档
│   ├── implementation/             # 实现文档
│   └── project/                    # 项目管理
├── triattention/                   # 核心实现
│   ├── __init__.py
│   ├── config.py                   # TriAttentionConfig
│   ├── compressor.py               # 主压缩器类
│   ├── scoring.py                  # 打分函数
│   ├── rope_utils.py               # RoPE 工具
│   ├── stats_utils.py              # Stats 加载
│   ├── paged_kv_manager.py         # PagedAttention KV 管理
│   └── kernels/
│       ├── __init__.py
│       ├── scoring_kernel.py       # Triton 打分
│       ├── pruning_kernel.py       # Triton 裁剪
│       └── fill_in_place_kernel.py # Triton Fill-in-Place
├── integration/                    # vLLM 集成
│   ├── __init__.py
│   ├── vllm_plugin.py              # vLLM 集成
│   ├── attention_wrapper.py        # Attention 包装器
│   └── memory_monitor.py           # 内存监控
├── test/                           # 测试
│   ├── __init__.py
│   ├── test_correctness.py
│   ├── test_scoring.py
│   ├── test_pruning.py
│   ├── test_paged_kv.py
│   ├── test_performance.py
│   └── fixtures/
├── benchmarks/                     # 性能测试
│   ├── run_accuracy_benchmark.py
│   ├── run_throughput_benchmark.py
│   └── configs/
└── scripts/                        # 脚本
    ├── run_triattention.sh
    └── collect_stats.py
```

---

## 2. 实施阶段

### 阶段 1：基础实现

**目标**：CPU 参考实现，验证正确性

| 任务 | 描述 | 依赖 |
|-----|------|-----|
| 1.1 | 建立目录结构 | - |
| 1.2 | 实现 `TriAttentionConfig` | 1.1 |
| 1.3 | 移植打分逻辑到 `scoring.py` | 1.2 |
| 1.4 | 实现基本压缩器 `compressor.py` | 1.3 |
| 1.5 | 实现 `protect_prefill` 参数 | 1.4 |
| 1.6 | 创建正确性测试 | 1.4 |

**交付物**：
- 可运行的 Python 参考实现
- 与 R-KV 输出对比的测试

---

### 阶段 2：Triton Kernel

**目标**：高效的 GPU kernel 实现

| 任务 | 描述 | 依赖 |
|-----|------|-----|
| 2.1 | 实现 `scoring_kernel.py`（单次读取多位置） | 1.4 |
| 2.2 | 实现 `pruning_kernel.py`（topk 选择） | 2.1 |
| 2.3 | 实现 `fill_in_place_kernel.py`（KV 填充） | 2.2 |
| 2.4 | Kernel 正确性测试 | 2.3 |
| 2.5 | Kernel 性能 benchmark | 2.4 |
| 2.6 | `triton.autotune` 优化 | 2.5 |

**交付物**：
- 优化的 Triton kernel
- 性能 benchmark 报告

---

### 阶段 3：vLLM 集成

**目标**：与 vLLM 0.15.x 集成

| 任务 | 描述 | 依赖 |
|-----|------|-----|
| 3.1 | 实现 `attention_wrapper.py` | 2.3 |
| 3.2 | 实现 `paged_kv_manager.py` | 3.1 |
| 3.3 | 与 PagedAttention 集成 | 3.2 |
| 3.4 | CUDA Graph 兼容处理 | 3.3 |
| 3.5 | vLLM serving 测试 | 3.4 |

**交付物**：
- 可用的 vLLM 插件
- 集成测试

---

### 阶段 4：高级功能

**目标**：内存触发、计算优化

| 任务 | 描述 | 依赖 |
|-----|------|-----|
| 4.1 | 实现 `memory_monitor.py` | 3.3 |
| 4.2 | 内存触发压缩逻辑 | 4.1 |
| 4.3 | 避免 RoPE 反转优化 | 2.1 |
| 4.4 | 位置索引存储 | 4.3 |

**交付物**：
- 内存触发压缩功能
- 优化的打分实现

---

### 阶段 5：Benchmark 与完善

**目标**：完整测试，文档完善

| 任务 | 描述 | 依赖 |
|-----|------|-----|
| 5.1 | 完整 benchmark 套件 | 4.2 |
| 5.2 | 多模型测试（LLaMA/Qwen/DeepSeek） | 5.1 |
| 5.3 | 边界情况处理 | 5.2 |
| 5.4 | 文档完善 | 5.3 |
| 5.5 | 性能调优 | 5.4 |

**交付物**：
- Benchmark 报告
- 完整文档

---

## 3. 关键文件参考

### 3.1 源文件（R-KV）

| 文件 | 用途 |
|------|-----|
| `R-KV/weian_development/speckv/speckv_rkv_style.py` | 主实现 |
| `R-KV/weian_development/speckv/round_pruning_utils.py` | 打分、RoPE |
| `R-KV/weian_development/tests/` | 测试套件 |
| `R-KV/HuggingFace/evaluation/` | 评估脚本 |

### 3.2 启动脚本

| 变种 | 脚本路径 |
|-----|---------|
| per-head | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh` |
| per-layer-per-head | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_layer_perhead.sh` |
| per-layer | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perlayer.sh` |

### 3.3 目标文件（vLLM 0.15.x）

| 文件 | 用途 |
|------|-----|
| `vllm/attention/layer.py` | 统一 Attention 层 |
| `vllm/v1/attention/backends/triton_attn.py` | Triton 后端 |
| `vllm/v1/attention/ops/triton_*.py` | Kernel 示例 |

---

## 4. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|-----|------|
| PagedAttention 集成复杂 | 高 | 详细分析 block 机制 |
| CUDA Graph 不兼容 | 中 | 提供 eager 回退 |
| 数值精度问题 | 中 | FP16/BF16 广泛测试 |
| 性能未达预期 | 中 | 持续 profiling |

---

## 5. 成功标准

| 指标 | 目标 |
|-----|------|
| 正确性 | 输出差异 < 1% 困惑度 |
| 吞吐量 | >= 1.5x（2048 budget） |
| 延迟开销 | < 10% |
| 稳定性 | 24h 零崩溃 |
| 模型覆盖 | LLaMA/Qwen/DeepSeek/Mistral |

---

## 6. 延后需求

| 需求 | 状态 |
|-----|------|
| 多 GPU (TP/PP) | 延后 |
| 多 vLLM 版本 | 延后 |
| 非 RoPE 模型 | 不支持 |
| Stats 收集脚本 | P1 |

---

*文档版本：1.0*
*创建日期：2025-01-30*
