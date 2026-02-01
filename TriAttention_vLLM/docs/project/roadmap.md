# 实施路线图

本文档描述 TriAttention 的实施阶段和代码结构规划。

---

## 0. 开发准则

### 0.1 最小修改原则

- **复用优先**：尽可能复用 vLLM 官方实现，不自行搭建平行系统
- **最小侵入**：只在必要处做修改和添加，避免大规模重构
- **扩展而非替换**：优先通过继承、包装、hook 等方式扩展，而非替换原有代码
- **Overflow Pages 设计**：
  - 不大改 vLLM engine，在外面包一层实现 overflow 管理
  - 复用 vLLM 现有的 block allocator 和 KV cache 管理
  - 修改量最小，只添加必要的 budget/overflow 跟踪逻辑

### 0.2 第一阶段：严格对齐 R-KV 脚本

第一阶段实现必须**严格对齐**以下三个参考脚本的行为：

| 变种 | 参考脚本 |
|-----|---------|
| per-head | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh` |
| per-layer-per-head | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_layer_perhead.sh` |
| per-layer | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perlayer.sh` |

**对齐要求（包括但不限于）**：
- 打分公式、聚合策略（mean/max）
- Offsets 几何序列生成
- 裁剪触发条件（divide_length、budget）
- Per-head/per-layer 的 token 选择逻辑
- RoPE 处理方式
- 所有配置参数的默认值

### 0.3 阶段兼容性

- 第一阶段的实现方案**不得阻碍**后续阶段的开发
- 不需要刻意预留接口，但要提前考虑扩展点
- 设计决策要考虑后续功能（如内存触发压缩）的可行性

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

### 阶段 1：核心实现（常见场景）

**目标**：在 TriAttention_vLLM 框架内完成 Triton 版本的核心实现，并覆盖最常见推理路径。  
**范围**：只处理主路径（稳定、常见场景）；边界情况明确推迟到阶段 2。

| 任务 | 描述 | 依赖 |
|-----|------|-----|
| 1.1 | 建立目录结构 | - |
| 1.2 | 实现 `TriAttentionConfig` | 1.1 |
| 1.3 | 实现状态/统计加载（`CompressionState`、stats loader） | 1.2 |
| 1.4 | 实现 Triton 打分 kernel（基础版本） | 1.3 |
| 1.5 | 使用 `torch.topk/torch.gather` 跑通 TopK/Gather | 1.4 |
| 1.6 | 集成到 `compressor.py` 并完成正确性测试 | 1.5 |
| 1.7 | vLLM 基础集成（PagedAttention 主路径） | 1.6 |
| 1.8 | 基线性能验证（非极限优化） | 1.7 |

**交付物**：
- 可运行的 Triton 版本（GPU）
- vLLM 主路径可用（常见场景）
- 与 R-KV 输出对比的正确性测试

---

### 阶段 2：边界情况与鲁棒性

**目标**：覆盖实际部署中的边界情况与稳定性问题。  
**原则**：阶段 1 不覆盖的情况在此阶段统一处理。

| 任务 | 描述 | 依赖 |
|-----|------|-----|
| 2.1 | Prefill > budget 处理策略 | 1.7 |
| 2.2 | 混合 prefill/decode（含 chunked prefill） | 1.7 |
| 2.3 | request 取消 / slot 复用 / position_indices 重置 | 1.7 |
| 2.4 | 内存触发压缩（preemption 之前介入） | 1.7 |
| 2.5 | CUDA Graph 兼容性处理 | 1.7 |
| 2.6 | 评估 Triton TopK/Gather 性能收益（可选） | 1.8 |
| 2.7 | 长时间运行与多 request 稳定性测试 | 2.1 |

**交付物**：
- 边界情况处理完备
- 稳定性测试报告

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

*文档版本：2.0*
*创建日期：2025-01-30*
*更新日期：2025-01-31（添加开发准则）*
