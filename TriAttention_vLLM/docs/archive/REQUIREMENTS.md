# TriAttention 需求文档

## 1. 项目概述

### 1.1 背景

在 torch 和 HuggingFace 的 backend 下实现了一个 KV cache 压缩算法（原名 SpeckV），现需要移植到 vLLM 并用 Triton 实现。

### 1.2 目标

将 KV 压缩算法在 vLLM 上实现，并使用 Triton kernel 优化。

### 1.3 源代码位置

**三种算法变种**：

| 变种 | 脚本路径 |
|-----|---------|
| per-head（默认） | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh` |
| per-layer-per-head | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_layer_perhead.sh` |
| per-layer | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perlayer.sh` |

**vLLM 推理引擎**：`vllm/` 文件夹

**目标 vLLM 版本**：0.15.x

---

## 2. 架构约束

| ID | 约束 | 描述 |
|----|-----|------|
| A-01 | 非侵入式实现 | 所有代码在 `TriAttention_vLLM/` 文件夹，不修改其他文件夹 |
| A-02 | 算法命名 | 重命名为 "TriAttention" |
| A-03 | 文档归档 | `TriAttention_vLLM/docs/` 记录整理信息 |
| A-04 | 测试目录 | `TriAttention_vLLM/test/` 专门负责测试 |

---

## 3. 功能需求

### 3.1 核心功能

| ID | 需求 | 优先级 | 描述 |
|----|-----|-------|------|
| F-01 | KV Cache 压缩 | P0 | 基于频率统计的打分，保留 top-k token |
| F-02 | 三种裁剪粒度 | P0 | per-head / per-layer-per-head / per-layer |
| F-03 | 内存触发压缩 | P1 | PagedAttention page 用尽时自动激活压缩，避免爆显存 |

### 3.2 裁剪触发条件

1. Overflow 满了（达到 divide_length）
2. **且**当前 KV 总量超过 budget

如果 budget 还没满，不触发裁剪，直接合并 overflow 到 budget。

### 3.3 Prefill 处理

**Prefill > budget 的情况**：Prefill 完成后立即触发裁剪，将 KV 压缩到 budget 以内。

**`protect_prefill` 参数**（默认 `False`）：
- `False`：prefill token 参与裁剪竞争，可能被裁掉
- `True`：prefill token 被保护不参与裁剪

### 3.4 算法变种

| 变种 | 参数 | 描述 |
|-----|------|------|
| per-head | `pruning_mode="per_head"` | 每个 KV head 全局独立选择 token |
| per-layer-per-head | `pruning_mode="per_layer_per_head"` | 每个 (layer, head) 独立选择 |
| per-layer | `pruning_mode="per_layer"` | 同层所有 head 共享 token 选择 |

---

## 4. 计算优化需求

### 4.1 避免 RoPE 反转

显存中保存 RoPE 之后的 K，同时记录该 K 的原始位置。通过调整打分公式中的 cos 参数实现数学等价，避免反转 RoPE 的计算开销。

详见：`COMPUTATION_OPTIMIZATION.md`

### 4.2 单次读取多位置打分

打分函数对未来多个位置打分时，只从显存读取 K 一次，避免每个位置都重新读取导致的 memory bottleneck。

### 4.3 位置无关计算分离

打分公式中大部分计算与目标位置无关，只有 cos 中的值随位置变化。实现位置无关部分只算一次，位置相关的 cos 表预计算共享。

详见：`TRIG_TABLE_OPTIMIZATION.md`

---

## 5. 模型支持

### 5.1 支持范围

**仅支持 RoPE 位置编码模型**：

| 模型系列 | 具体模型 | 优先级 |
|---------|---------|-------|
| Qwen | Qwen2, Qwen2.5, Qwen3 | P0 |
| LLaMA | LLaMA2, LLaMA3, CodeLlama | P0 |
| DeepSeek | DeepSeek-V2, DeepSeek-R1 | P0 |
| Mistral | Mistral, Mixtral | P1 |

### 5.2 RoPE 风格

- **half**（主要）：前后两半配对（Qwen, LLaMA）
- **interleaved**（次要）：奇偶交替配对

---

## 6. 配置与 Stats 文件

### 6.1 配置参数

默认使用脚本中的配置方式，参数包括：
- `budget`: KV cache 上限
- `divide_length`: 每 N 步检查一次
- `pruning_mode`: 裁剪粒度
- `stats_path`: 频率统计文件路径

### 6.2 Stats 文件

**当前测试模型**：使用脚本中指定的校准文件

**Stats 文件位置示例**：
```
R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/
```

Stats 文件包含：
- `Q_mean_real`, `Q_mean_imag`: 平均 query 的实部虚部
- `freq_scale_sq`: 频率缩放因子
- `extra_coef`: 位置无关项系数

---

## 7. 测试需求

### 7.1 正确性测试

| 测试项 | 描述 |
|-------|------|
| 输出对比 | 同样 prompt 下，与 R-KV 实现输出接近（允许少量误差） |
| 打分验证 | 裁剪时的打分结果是否相同 |
| 裁剪索引 | 保留的 token 索引匹配 |

### 7.2 性能测试

| 测试项 | 描述 |
|-------|------|
| 吞吐量 | Batch size=32 下，Full attention vs TriAttention |
| 解码延迟 | 解码一个 token 多久 |
| 打分延迟 | 打分一轮多久 |
| 瓶颈分析 | 如果慢的话，慢在哪 |

### 7.3 Benchmark

复用 R-KV 中的评估 benchmark：
- 测试新模型吞吐
- 测试精度（AIME24 等）

---

## 8. 延后需求

| 需求 | 状态 |
|-----|------|
| 多 GPU (TP/PP) | 延后 |
| 多 vLLM 版本 | 延后 |
| 非 RoPE 模型 | 不支持 |

---

## 9. 文档列表

| 文档 | 内容 |
|-----|------|
| REQUIREMENTS.md | 本文档 |
| TODO.md | 待敲定细节和待完成任务 |
| PLAN.md | 实施计划 |
| IMPLEMENTATION_DETAILS.md | Fill-in-Place 策略实现细节 |
| COMPUTATION_OPTIMIZATION.md | 打分计算优化（避免 RoPE 反转等） |
| TRIG_TABLE_OPTIMIZATION.md | 共享三角函数表优化 |
| STORAGE_REQUIREMENTS.md | 额外存储需求 |
| KV_CACHE_LAYOUT_ANALYSIS.md | KV cache 布局分析 |

---

*文档版本：2.0*
*创建日期：2025-01-30*
*更新：整合原始需求，添加 TODO*
