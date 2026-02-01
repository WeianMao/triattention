# TriAttention for vLLM

将 SpeckV KV cache 压缩算法移植到 vLLM，使用 Triton kernel 优化。

---

## 文档结构

```
docs/
├── README.md                      # 本文档：项目概述与索引
├── design/                        # 设计文档
│   ├── algorithm.md               # 算法设计：打分公式、裁剪逻辑
│   └── optimization.md            # 优化设计：RoPE 优化、三角表共享
├── implementation/                # 实现文档
│   ├── fill_in_place.md           # Fill-in-Place 策略详解
│   ├── data_structures.md         # 数据结构：position_indices、stats
│   └── vllm_integration.md        # vLLM PagedAttention 集成分析
├── project/                       # 项目管理
│   ├── key_decisions.md           # ⭐ 关键决策与验证结论汇总
│   ├── roadmap.md                 # 实施路线图与开发准则
│   ├── todo.md                    # 待办事项
│   └── CLARIFICATIONS_NEEDED.md   # 设计问题澄清记录
├── r-kv-analysis/                 # R-KV/vLLM 对比分析（详细）
│   ├── README.md                  # 分析文档索引与核心结论速查
│   ├── Q1_requirement_coverage.md # 需求覆盖对比
│   ├── Q2_pros_cons_analysis.md   # 优缺点分析
│   └── Q3_reusable_code.md        # 可复用代码分析
└── archive/                       # 归档（原始文档，供参考）
```

---

## 1. 项目概述

### 1.1 背景

在 torch 和 HuggingFace 的 backend 下实现了 KV cache 压缩算法（原名 SpeckV），现需移植到 vLLM 并用 Triton 实现。

### 1.2 目标

- 将 KV 压缩算法在 vLLM 0.15.x 上实现
- 使用 Triton kernel 优化打分和裁剪
- 重命名为 "TriAttention"

### 1.3 架构约束

| ID | 约束 | 描述 |
|----|-----|------|
| A-01 | 非侵入式 | 所有代码在 `TriAttention_vLLM/` 文件夹，不修改其他文件夹 |
| A-02 | 算法命名 | 重命名为 "TriAttention" |
| A-03 | 目标版本 | vLLM 0.15.x |

---

## 2. 源代码位置

### 2.1 原始实现（R-KV）

**三种算法变种**：

| 变种 | 脚本路径 |
|-----|---------|
| per-head（默认） | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh` |
| per-layer-per-head | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_layer_perhead.sh` |
| per-layer | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perlayer.sh` |

**核心实现文件**：

| 文件 | 用途 |
|------|-----|
| `R-KV/weian_development/speckv/speckv_rkv_style.py` | 主实现 |
| `R-KV/weian_development/speckv/round_pruning_utils.py` | 打分、RoPE |
| `R-KV/weian_development/tests/` | 测试套件 |

### 2.2 Stats 文件位置

```
R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/
```

---

## 3. 核心功能

### 3.1 KV Cache 压缩

基于频率统计的打分机制，保留 top-k token 的 KV cache。

### 3.2 三种裁剪粒度

| 变种 | 参数 | 描述 |
|-----|------|------|
| per-head | `pruning_mode="per_head"` | 每个 KV head 全局独立选择 token |
| per-layer-per-head | `pruning_mode="per_layer_per_head"` | 每个 (layer, head) 独立选择 |
| per-layer | `pruning_mode="per_layer"` | 同层所有 head 共享 token 选择 |

### 3.3 裁剪触发条件

1. Overflow 满了（达到 `divide_length`）
2. **且** 当前 KV 总量超过 `budget`

如果 budget 还没满，不触发裁剪，直接合并 overflow 到 budget。

### 3.4 Prefill 处理

- **Prefill > budget**：阶段 2 才处理（阶段 1 不覆盖）
- **`protect_prefill` 参数**（默认 `False`）：
  - `False`：prefill token 参与裁剪竞争
  - `True`：prefill token 被保护不参与裁剪

---

## 4. 模型支持

**仅支持 RoPE 位置编码模型**：

| 模型系列 | 具体模型 | 优先级 |
|---------|---------|-------|
| Qwen | Qwen2, Qwen2.5, Qwen3 | P0 |
| LLaMA | LLaMA2, LLaMA3, CodeLlama | P0 |
| DeepSeek | DeepSeek-V2, DeepSeek-R1 | P0 |
| Mistral | Mistral, Mixtral | P1 |

**RoPE 风格**：
- **half**（主要）：前后两半配对（Qwen, LLaMA）
- **interleaved**（次要）：奇偶交替配对

---

## 5. 配置参数

| 参数 | 说明 | 示例值 |
|-----|------|-------|
| `budget` | KV cache 上限 | 2048, 4096, 8192 |
| `divide_length` | 每 N 步检查一次 | 64, 128 |
| `pruning_mode` | 裁剪粒度 | `per_head`, `per_layer_per_head`, `per_layer` |
| `stats_path` | 频率统计文件路径 | 见 2.2 |
| `protect_prefill` | 是否保护 prefill token | `False`（默认） |

---

## 6. 快速导航

### 核心文档（优先阅读）

| 想了解... | 阅读文档 |
|----------|---------|
| ⭐ **关键决策与验证结论** | [project/key_decisions.md](project/key_decisions.md) |
| 开发阶段与需求清单 | [project/key_decisions.md](project/key_decisions.md) |
| 实施路线图和开发准则 | [project/roadmap.md](project/roadmap.md) |

### 设计文档

| 想了解... | 阅读文档 |
|----------|---------|
| 打分公式和裁剪逻辑 | [design/algorithm.md](design/algorithm.md) |
| 计算优化（RoPE、三角表） | [design/optimization.md](design/optimization.md) |

### 实现文档

| 想了解... | 阅读文档 |
|----------|---------|
| Fill-in-Place 工作流程 | [implementation/fill_in_place.md](implementation/fill_in_place.md) |
| position_indices 等数据结构 | [implementation/data_structures.md](implementation/data_structures.md) |
| vLLM PagedAttention 集成 | [implementation/vllm_integration.md](implementation/vllm_integration.md) |

### 项目管理

| 想了解... | 阅读文档 |
|----------|---------|
| 待办事项 | [project/todo.md](project/todo.md) |
| 设计问题澄清记录 | [project/CLARIFICATIONS_NEEDED.md](project/CLARIFICATIONS_NEEDED.md) |

### R-KV 对比分析（详细）

| 想了解... | 阅读文档 |
|----------|---------|
| R-KV 分析概述 | [r-kv-analysis/README.md](r-kv-analysis/README.md) |
| 需求覆盖对比 | [r-kv-analysis/Q1_requirement_coverage.md](r-kv-analysis/Q1_requirement_coverage.md) |
| 优缺点分析 | [r-kv-analysis/Q2_pros_cons_analysis.md](r-kv-analysis/Q2_pros_cons_analysis.md) |
| 可复用代码分析 | [r-kv-analysis/Q3_reusable_code.md](r-kv-analysis/Q3_reusable_code.md) |

---

*文档版本：4.0*
*创建日期：2025-01-30*
*更新日期：2025-01-31（重组文档结构，添加关键决策汇总）*
