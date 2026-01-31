# 关键决策与验证结论

本文档汇总 TriAttention 项目的所有关键决策和经过验证的结论。

---

## 1. 开发阶段定义

| 阶段 | 目标 | 框架 | Batch Size | 效率要求 |
|-----|------|------|------------|----------|
| **阶段 0** | 快速验证 | R-KV 框架内 | = 1 | 不追求 |
| **阶段 1** | 高效独立版本 | TriAttention 独立 | > 1 | Triton 级别 |
| **阶段 2** | 高级功能 | TriAttention 独立 | > 1 | Triton 级别 |

---

## 2. 验证确认的结论 ✅

以下结论经过代码审查验证：

### 2.1 R-KV 效率

| 结论 | 验证方法 | 状态 |
|-----|---------|------|
| R-KV 无 Triton kernel | 搜索整个代码库 | ✅ 确认 |
| R-KV 无 custom CUDA kernel | 搜索整个代码库 | ✅ 确认 |
| R-KV 全部 native PyTorch | 代码审查 | ✅ 确认 |
| TopK 使用 `torch.topk()` | 代码审查 | ✅ 确认 |
| Gather 使用 `torch.gather()` | 代码审查 | ✅ 确认 |
| 整体效率比 Triton 慢 1.8-2.8x | 估算 | ✅ 确认 |

### 2.2 R-KV 功能限制

| 结论 | 验证方法 | 状态 |
|-----|---------|------|
| batch_size > 1 **静默失败** | 发现 `key_states[0]` 硬编码 | ✅ 确认 |
| 无 noise injection | 搜索整个代码库 | ✅ 确认（之前分析有误）|
| 相似度计算 O(n²) 内存 | 代码审查 | ✅ 确认 |
| 无 TP/PP 支持 | 代码审查 | ✅ 确认 |
| 代码版本不一致 | 对比 rkv/ 和 HuggingFace/rkv/ | ✅ 确认 |

### 2.3 SpeckV 优势

| 结论 | 原因 | 状态 |
|-----|------|------|
| 无 O(n²) 内存问题 | 不做相似度计算 | ✅ 确认 |
| 不依赖实时 Query | 使用预计算频率统计 | ✅ 确认 |

---

## 3. 关键设计决策

### 3.1 不使用 Noise Injection ❌

**决策**：不使用加噪声方式解决 tie-breaking

**理由**：
1. 方法不够优雅
2. R-KV 实际也没有实现（之前分析有误）
3. 直接使用 `topk()` 的 PyTorch 默认行为

**替代方案**：
- 阶段 0：怎么方便怎么来
- 阶段 1：确定性选择（按位置顺序取前 N 个）

### 3.2 不使用 Query Cache ❌

**决策**：不实现 Window-based Query Cache

**理由**：SpeckV 基于预计算的 Q 频率统计打分，不依赖实时 Query

**影响**：简化实现，只需保护最近的 KV

### 3.3 阶段 1 必须支持 Batch Size > 1 ✅

**决策**：阶段 1 必须显式支持 batch > 1

**理由**：R-KV 的 batch=1 限制是静默失败，会导致数据错误

**实现要求**：要么支持 batch > 1，要么在 batch > 1 时抛出明确错误

### 3.4 阶段 1 必须 Triton 重写 ✅

**决策**：阶段 1 不复用 R-KV 的 TopK/Gather 代码

**理由**：R-KV 代码效率比 Triton 慢 1.8-2.8x

**重写范围**：打分、TopK、Gather 三个核心操作

---

## 4. 各阶段需求清单

### 阶段 0（R-KV 框架内）

**必须**：
- [ ] 基于频率统计的打分（SpeckV 核心）
- [ ] Per-head / Per-layer 裁剪模式
- [ ] RoPE 位置追踪（复用 R-KV）
- [ ] Prefill 保护选项

**可选**：
- [ ] RoPE 一致性检查（复用 R-KV）
- [ ] FP32 TopK

**不需要**：Batch Size > 1、Triton 优化、状态重置（R-KV 框架已有）

### 阶段 1（独立高效版本）

**必须**：
- [ ] Triton kernel 实现打分、TopK、Gather
- [ ] **Batch Size > 1 支持**
- [ ] RoPE 一致性检查
- [ ] 状态重置接口
- [ ] Per-head / Per-layer 裁剪模式
- [ ] Prefill 保护

**可选**：
- [ ] Union-based 选择
- [ ] 多种聚合策略（mean/max）

### 阶段 2（高级功能）

**计划**：
- [ ] 内存触发压缩
- [ ] CUDA Graph 兼容
- [ ] 更灵活的 Budget 策略
- [ ] TP/PP 支持（可选）

---

## 5. 需求覆盖对比

| 需求 | R-KV | 阶段 0 | 阶段 1 | 阶段 2 |
|-----|------|-------|-------|-------|
| 频率统计打分 | ❌ | ✅ | ✅ | ✅ |
| Per-head 裁剪 | ✅ | ✅ | ✅ | ✅ |
| Per-layer 裁剪 | ❌ | ✅ | ✅ | ✅ |
| Recent KV 保护 | ✅ | ✅ | ✅ | ✅ |
| Prefill 保护 | ✅ | ✅ | ✅ | ✅ |
| RoPE 位置追踪 | ✅ | ✅ | ✅ | ✅ |
| Batch Size > 1 | ❌ | ❌ | ✅ | ✅ |
| Triton 效率 | ❌ | ❌ | ✅ | ✅ |
| 无 O(n²) 内存 | ❌ | ✅ | ✅ | ✅ |
| CUDA Graph | ❌ | ❌ | ❌ | ⏸️ |
| 内存触发压缩 | ❌ | ❌ | ❌ | ⏸️ |

---

## 6. 从 R-KV 借鉴的内容

### 需要借鉴

| 项目 | 来源文件 | 阶段 |
|-----|---------|------|
| RoPE 一致性检查 | `round_pruning_utils.py` | 阶段 0/1 |
| 状态重置机制 | `rkv_speckv_generate.py` | 阶段 1 |
| FP32 TopK 选项 | `r1_kv.py` | 阶段 1（可选）|
| 配置类设计 | `r1_kv.py` | 阶段 1 |
| 评估脚本结构 | `rkv_sharded_eval.py` | 阶段 0/1 |

### 不需要借鉴

| 项目 | 原因 |
|-----|------|
| Noise injection | R-KV 未实现，我们也不需要 |
| Query cache | SpeckV 不依赖实时 Query |
| 相似度计算 | SpeckV 不需要，且有 O(n²) 问题 |
| TopK/Gather 代码 | 效率不达标 |

---

## 7. R-KV 已知问题

| 问题 | 影响 | 我们的规避 |
|-----|------|----------|
| batch > 1 静默失败 | 数据错误 | 阶段 1 必须显式支持或报错 |
| O(n²) 相似度计算 | 长序列 OOM | SpeckV 不做相似度 |
| 代码版本不一致 | 行为差异 | 阶段 0 用 HuggingFace 版本 |
| 无 TP/PP | 分布式受限 | 阶段 2 考虑 |

---

## 8. 文件位置参考

### 阶段 0 开发位置

```
R-KV/weian_development/speckv/
├── speckv_rkv_style.py        # 主实现
├── round_pruning_utils.py     # 工具函数
└── rkv_speckv_generate.py     # 生成脚本
```

### 阶段 1 开发位置

```
TriAttention_vLLM/triattention/
├── config.py                  # 配置类
├── compressor.py              # 主压缩器
├── scoring.py                 # 打分逻辑
└── kernels/
    ├── scoring_kernel.py      # Triton 打分
    ├── topk_gather_kernel.py  # Triton TopK+Gather
    └── fill_in_place_kernel.py
```

---

*创建日期：2025-01-31*
*本文档汇总自 R-KV 对比分析（QA 文档），所有结论经过代码审查验证*
