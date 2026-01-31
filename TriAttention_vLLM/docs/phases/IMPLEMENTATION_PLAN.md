# SpeckV/TriAttention 实现计划

## 概述

本文档定义 Phase 0 和 Phase 1 的具体实现计划和任务分解。

---

## 0. 架构决策

### Phase 0 架构：Monkey Patch Model.forward

**决策**：Phase 0 采用 **monkey patch model.forward** 方式，而不是 R-KV 的 `update_kv` 接口。

**原因**：
- SpeckV 的 per_layer 和 per_layer_perhead 模式需要**跨层信息**
- R-KV 的 `update_kv` 是 per-layer 接口，压缩器只能看到当前层
- 为了完整支持三种 pruning mode，必须使用全局压缩器

**影响**：
- Phase 0 **复用现有代码**（`speckv_rkv_style.py`），不重写
- 不创建新的 `SpeckVRKV` 类
- 重点是**理解和验证**，而不是重新实现

---

## 1. 总体路线图

```
Phase 0 (R-KV 集成)          Phase 1 (Triton 实现)
        ↓                           ↓
    理解 + 验证              高效独立版本
  复用现有代码              batch>1, Triton
  monkey patch               新接口设计
        ↓                           ↓
    ┌───────────────────────────────┐
    │     Phase 2 (高级功能)          │
    │   vLLM 集成、CUDA Graph        │
    └───────────────────────────────┘
```

**关键决策**：Phase 0 和 Phase 1 可以**并行开发**，因为：
- Phase 0 是理解和验证，不依赖 Phase 1
- Phase 1 是独立实现，不依赖 Phase 0
- Phase 0 为 Phase 1 提供参考基准

---

## 2. Phase 0 实现计划

### 2.1 目标

| 目标 | 验收标准 |
|-----|---------|
| 理解现有代码 | 能清晰解释打分公式、三种 pruning mode 的差异 |
| 验证三种脚本 | 三个脚本全部能正常运行 |
| 创建使用文档 | README.md 完整记录使用方法 |
| 准确率验证 | AIME24 准确率与历史结果一致 |

### 2.2 策略说明

**重要**：Phase 0 **不重写代码**，而是：
1. 复用现有的 `speckv_rkv_style.py` 实现
2. 理解其工作原理
3. 验证三种 pruning mode 的正确性
4. 创建使用文档

### 2.3 任务分解

#### Task 0.1: 环境验证

```
子任务：
□ 激活 rkv conda 环境
□ 验证依赖完整（transformers, torch, flash-attn）
□ 确认 stats 文件存在
□ 验证 stats 元数据与配置一致（见下方）
□ 运行一个简单的测试脚本
```

**命令**：
```bash
conda activate rkv
cd /data/rbg/users/weian/project/rl/dc/R-KV
ls weian_development/speckv/
ls outputs/repository/sample8_fullkv_aime25_official_qwen/stats/
```

**Stats 元数据校验（必须）**：
```python
# 验证 stats 与当前配置一致
# 需检查：model_path, prompt_style, attention_impl, dtype, rope_scaling
from weian_development.speckv.stats_utils import validate_stats_metadata
# 或手动检查 stats 文件的元数据字段
```

---

#### Task 0.2: 代码阅读 - 压缩器初始化

**目标**：理解 `SpeckVRKVStyle.__init__()` 的流程

```
阅读文件：weian_development/speckv/speckv_rkv_style.py

关注点：
□ Config 参数如何解析
□ Stats 如何加载（load_head_frequency_stats）
□ RoPE 如何初始化（build_rotary, inv_freq, cos/sin 表）
□ freq_scale_sq 如何计算
□ sampled_heads 的过滤逻辑
```

**输出**：笔记文档，记录关键数据流

---

#### Task 0.3: 代码阅读 - 打分函数

**目标**：理解 `score_keys_for_round()` 的数学公式

```
阅读文件：weian_development/speckv/round_pruning_utils.py

关注点：
□ invert_rope() - RoPE 反演公式
□ compute_frequency_statistics_from_means() - amp, phi, extra 计算
□ score_keys_for_round() - 位置相关打分公式
□ 聚合方式（mean/max over offsets）
```

**输出**：打分公式的数学描述

---

#### Task 0.4: 代码阅读 - 三种 Pruning Mode

**目标**：理解三种模式的差异

```
阅读文件：weian_development/speckv/speckv_rkv_style.py

关注点：
□ _select_union_based() - global 模式
□ _select_per_head_independent() - per_head 模式
□ _select_per_layer_independent() - per_layer 模式
□ _select_per_layer_perhead_independent() - per_layer_perhead 模式
□ 各模式的 keep_indices 形状
□ 位置追踪的差异
```

**输出**：三种模式的对比表

---

#### Task 0.5: 代码阅读 - 集成方式

**目标**：理解 monkey patch 的工作方式

```
阅读文件：weian_development/speckv/rkv_speckv_generate.py

关注点：
□ apply_speckv_rkv_style_patch() 如何工作
□ speckv_rkv_forward() 的触发逻辑
□ should_compress() 的条件
□ KV cache 如何被压缩
```

**输出**：调用流程图

---

#### Task 0.6: 验证三种脚本

**目标**：确认三种 pruning mode 都能正常运行

```
子任务：
□ 运行 per_head 脚本，检查输出
□ 运行 per_layer_perhead 脚本，检查输出
□ 运行 per_layer 脚本，检查输出
□ 对比三种模式的准确率
□ 验证 reset_compression_state 是否在每个样本前被调用
□ 确认进程名使用 PD-L1_binder 前缀
```

**命令**：
```bash
# Per-Head 模式
bash weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh

# Per-Layer-Per-Head 模式
bash weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_layer_perhead.sh

# Per-Layer 模式
bash weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perlayer.sh
```

**验证点**：
- `reset_compression_state()` 必须在每个样本前调用，防止状态泄露
- 长时间任务通过 `rkv_sharded_runner.py` wrapper 保证进程名规范

---

#### Task 0.7: 创建使用文档

**目标**：创建 `weian_development/speckv/README.md`

```
文档内容：
□ 快速开始指南
□ 配置参数完整说明
□ 三种 pruning mode 的使用场景
□ 常见问题与解答
□ 打分公式的简要说明
```

---

#### Task 0.8: GQA 映射验证（建议）

**目标**：验证 sampled_heads 到 KV heads 的映射正确

```
验证内容：
□ 统计每个 KV head 覆盖的 sampled heads 数量
□ 确认映射逻辑与 GQA group_size 一致
□ 检查是否有 KV head 没有被任何 sampled head 覆盖
```

**验证代码**：
```python
num_kv_heads = model.config.num_key_value_heads
group_size = model.config.num_attention_heads // num_kv_heads
for kv_head in range(num_kv_heads):
    covered = [h for l, h in sampled_heads if h // group_size == kv_head]
    print(f"KV head {kv_head}: {len(covered)} sampled heads")
```

---

#### Task 0.9: 边界测试（可选）

**目标**：验证边界情况的处理

```
测试场景：
□ seq_len < budget（不应压缩）
□ prefill_length > budget（应正确处理）
□ 多轮压缩（位置追踪正确）
```

---

### 2.4 Phase 0 Checklist

```
Phase 0 完成标准：
□ 三个脚本全部能正常运行
□ 理解打分公式（有笔记）
□ 理解三种 pruning mode 的差异（有对比表）
□ 理解 monkey patch 集成方式（有调用流程图）
□ 创建使用文档 README.md
□ 准确率与历史结果一致

验证清单（基于 Review 反馈）：
□ Stats 元数据校验通过（model/rope/prompt/attn/dtype）
□ reset_compression_state 调用位置已确认
□ GQA 映射验证通过（每个 KV head 有对应的 sampled heads）
□ 进程命名规范（PD-L1_binder）已确认
□ 输出目录命名不覆盖基线结果
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
