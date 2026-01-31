# Phase 0: R-KV 框架内 SpeckV 集成

## 概述

在 R-KV 仓库内实现 SpeckV 压缩算法，**使用 monkey patch model.forward 方式**（与现有实现一致），完整支持三种 pruning mode。

---

## 1. 架构决策

### 1.1 方案选择：Monkey Patch Model.forward

**选择原因**：

| R-KV 的 update_kv 接口 | SpeckV 需求 |
|----------------------|------------|
| Per-layer 压缩器实例 | 全局压缩器，需要访问所有层 |
| 只能看到当前层的 K, V | per_layer/per_layer_perhead 模式需要跨层信息 |

SpeckV 的三种 pruning mode 中，有两种（per_layer, per_layer_perhead）需要跨层打分聚合，无法用 R-KV 的 per-layer 接口实现。

**因此**：采用与现有 `speckv_rkv_style.py` 相同的架构——通过 patch `model.forward()` 实现全局压缩。

### 1.2 架构图

```
                    ┌─────────────────────────────────────┐
                    │     SpeckVRKVStyle（全局压缩器）      │
                    │                                     │
                    │  - 可以访问所有层的 KV cache          │
                    │  - 支持三种 pruning mode             │
                    │  - 独立的位置追踪系统                 │
                    └──────────────┬──────────────────────┘
                                   │
                                   │ patch
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                        model.forward()                           │
│                                                                  │
│  1. 执行正常 forward（所有 Attention 层）                          │
│  2. 检查是否需要压缩（absolute_position % divide_length == 0）     │
│  3. 如果需要：调用 compressor.compute_keep_indices()              │
│  4. 根据 keep_indices 压缩所有层的 KV cache                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.3 与 R-KV 其他算法的区别

| 方面 | R1KV/SnapKV/H2O | SpeckV |
|-----|----------------|--------|
| 压缩器位置 | 每个 Attention 层内 | model.forward 外部 |
| 接口 | `update_kv(k, q, v)` | `compute_keep_indices(pkv_tuple)` |
| 触发机制 | R-KV 的 compression flag | 独立的 absolute_position 追踪 |
| 数据可见性 | 单层 | 所有层 |

---

## 2. 目标与约束

### 2.1 目标

| 目标 | 说明 |
|-----|------|
| 完整支持三种 pruning mode | per_head, per_layer, per_layer_perhead |
| 复用现有实现 | 基于 `speckv_rkv_style.py`，最小改动 |
| 在 AIME24 上评估 | 准确率与现有实现一致 |

### 2.2 约束

| 约束 | 说明 |
|-----|------|
| Batch Size = 1 | R-KV 框架限制 |
| 无 Triton | 使用 PyTorch 原生操作 |
| 隔离开发 | 不修改 R-KV 核心代码（rkv/、HuggingFace/rkv/） |

---

## 3. 现有代码分析

### 3.1 核心文件

| 文件 | 行数 | 用途 |
|-----|------|------|
| `weian_development/speckv/speckv_rkv_style.py` | ~1255 | 主压缩器类 |
| `weian_development/speckv/round_pruning_utils.py` | ~373 | 打分函数、RoPE 处理 |
| `weian_development/speckv/stats_utils.py` | ~93 | Stats 加载验证 |
| `weian_development/speckv/rkv_speckv_generate.py` | ~200 | Generate wrapper |

### 3.2 调用链

```
weian_script/aime_sampled8/speckv/aime24/run_speckv_*.sh
    ↓
rkv_sharded_dispatch.py (分发器)
    ↓
rkv_sharded_runner.py (进程名包装)
    ↓
rkv_sharded_eval.py (评估主逻辑)
    ↓
apply_speckv_rkv_style_patch(model, compressor)
    ↓
SpeckVRKVStyle.compute_keep_indices()
```

### 3.3 三个脚本的参数差异

| 脚本 | 关键参数 | 对应配置 |
|-----|---------|---------|
| `*_perhead.sh` | `--per-head-pruning` | `per_head_pruning=True` |
| `*_layer_perhead.sh` | `--per-layer-perhead-pruning` | `per_layer_perhead_pruning=True` |
| `*_perlayer.sh` | `--per-layer-pruning` | `per_layer_pruning=True` |

**共同参数**：
```bash
--sparse-normalize-scores         # normalize_scores=True
--include-prefill-in-budget       # include_prefill_in_budget=True
--rkv-style-compression           # 使用 R-KV 风格压缩
--rkv-style-slack-trigger         # use_slack_trigger=True
--divide-length 128               # divide_length=128
```

---

## 4. 设计方案

### 4.1 开发策略

**策略**：复用现有代码，不重写。

现有的 `speckv_rkv_style.py` 已经是完整、经过验证的实现。Phase 0 的目标是：

1. **理解**现有代码的工作方式
2. **整理**调用入口，使其更易使用
3. **验证**三种 pruning mode 的正确性
4. **文档化**使用方法

### 4.2 文件组织

```
R-KV/
├── weian_development/
│   └── speckv/
│       ├── speckv_rkv_style.py       # 现有：主压缩器（不修改）
│       ├── round_pruning_utils.py    # 现有：打分函数（不修改）
│       ├── stats_utils.py            # 现有：Stats 工具（不修改）
│       ├── rkv_speckv_generate.py    # 现有：Generate wrapper（不修改）
│       └── README.md                 # 新增：使用文档
│
├── weian_script/
│   └── aime_sampled8/
│       └── speckv/
│           └── aime24/
│               ├── run_speckv_aime24_qwen_norm_aligned_perhead.sh
│               ├── run_speckv_aime24_qwen_norm_aligned_layer_perhead.sh
│               └── run_speckv_aime24_qwen_norm_aligned_perlayer.sh
│
└── HuggingFace/
    └── rkv/
        └── compression/
            └── speckv.py             # 现有：wrapper（可选扩展）
```

### 4.3 SpeckVRKVStyleConfig 完整参数

```python
@dataclass
class SpeckVRKVStyleConfig:
    """SpeckV-RKV 风格配置（完整参数列表）"""

    # ===== 必需参数 =====
    stats_path: Path                      # 预计算统计文件路径
    model_path: Path                      # 模型路径
    device: torch.device                  # 计算设备
    dtype: torch.dtype                    # 计算精度
    budget: int                           # KV 缓存预算

    # ===== 打分参数 =====
    offset_max_length: int = 65536        # 最大 offset
    score_aggregation: str = "mean"       # 聚合方式: "mean" 或 "max"

    # ===== 裁剪模式（三选一） =====
    per_head_pruning: bool = False        # 每 KV head 独立
    per_layer_perhead_pruning: bool = False  # 每 (layer, KV head) 独立
    per_layer_pruning: bool = False       # 每层独立，同层 heads 共享

    # ===== 聚合参数 =====
    layer_perhead_aggregation: str = "max"   # per_layer_perhead 聚合: "max" 或 "mean"
    per_layer_aggregation: str = "max"       # per_layer 聚合: "max", "mean", "pure_mean"

    # ===== Prefill 处理 =====
    include_prefill_in_budget: bool = False  # prefill tokens 计入 budget
    allow_prefill_compression: bool = False  # 允许压缩 prefill tokens

    # ===== 触发机制 =====
    divide_length: int = 128              # 压缩间隔
    use_slack_trigger: bool = False       # 宽松触发（budget + divide_length）

    # ===== 可选功能 =====
    normalize_scores: bool = False        # Z-score 标准化
    use_rank_aggregation: bool = False    # 使用排名而非分数
    seed: Optional[int] = None            # 随机种子
    head_limit: Optional[int] = None      # 采样头数限制

    # ===== 高级参数 =====
    disable_top_n_high_freq: int = 0      # 禁用高频分量
    disable_mlr: bool = False             # 禁用 MLR 项
    disable_trig: bool = False            # 禁用三角项
```

---

## 5. 实现步骤

### Step 1: 环境验证

```bash
# 1. 激活 rkv 环境
conda activate rkv

# 2. 验证现有脚本可运行
cd /data/rbg/users/weian/project/rl/dc/R-KV
bash weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh
```

### Step 2: 代码阅读与理解

**目标**：理解现有实现的核心逻辑。

**阅读顺序**：
1. `speckv_rkv_style.py` 的 `__init__()` — 初始化流程
2. `speckv_rkv_style.py` 的 `compute_keep_indices()` — 主压缩逻辑
3. `round_pruning_utils.py` 的 `score_keys_for_round()` — 打分函数
4. `rkv_speckv_generate.py` 的 `apply_speckv_rkv_style_patch()` — 集成方式

**记录**：
- 关键数据流
- 三种 pruning mode 的差异
- 位置追踪机制

### Step 3: 创建使用文档

**文件**：`weian_development/speckv/README.md`

**内容**：
- 快速开始指南
- 配置参数说明
- 三种 pruning mode 的使用场景
- 常见问题

### Step 4: 验证三种 Pruning Mode

**测试脚本**：
```bash
# Per-Head 模式
bash weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh

# Per-Layer-Per-Head 模式
bash weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_layer_perhead.sh

# Per-Layer 模式
bash weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perlayer.sh
```

**验证项**：
- 脚本能正常运行
- 输出格式正确
- 准确率在合理范围

### Step 5: 数值对比测试

**目标**：确认打分逻辑正确。

```python
# 创建测试脚本
# tests/test_scoring_consistency.py

def test_scoring_output():
    """验证打分输出的形状和数值范围"""
    # 加载相同的 stats
    # 使用相同的输入
    # 检查打分输出
```

### Step 6: 整理与优化（可选）

如果现有代码有明显可改进的地方：
- 添加类型注解
- 优化日志输出
- 添加断言检查

**原则**：最小改动，不影响功能。

---

## 6. 测试计划

### 6.1 功能测试

| 测试 | 验证内容 | 预期结果 |
|-----|---------|---------|
| per_head 模式 | 脚本运行 + 输出 | 正常完成，有输出文件 |
| per_layer_perhead 模式 | 脚本运行 + 输出 | 正常完成，有输出文件 |
| per_layer 模式 | 脚本运行 + 输出 | 正常完成，有输出文件 |

### 6.2 准确率测试

| 测试 | 数据集 | 预期结果 |
|-----|-------|---------|
| per_head @budget=2048 | AIME24 | 与历史结果一致 |
| per_layer_perhead @budget=2048 | AIME24 | 与历史结果一致 |
| per_layer @budget=2048 | AIME24 | 与历史结果一致 |

### 6.3 边界测试

| 测试 | 场景 | 预期行为 |
|-----|------|---------|
| 短序列 | seq_len < budget | 不压缩，直接返回 |
| 长 prefill | prefill_length > budget | 正确处理 |
| 多轮压缩 | 连续触发多次压缩 | 位置追踪正确 |

---

## 7. Phase 0 完成标准

```
□ 三个脚本全部能正常运行
□ 理解现有代码的核心逻辑（有笔记/文档）
□ 创建使用文档 (README.md)
□ 准确率与历史结果一致
□ 边界测试通过
```

---

## 8. 后续：Phase 1 的关系

Phase 0 完成后，Phase 1 将：

1. **提取核心算法**：从 `speckv_rkv_style.py` 提取打分和裁剪逻辑
2. **Triton 重写**：用 Triton kernel 实现高效版本
3. **新接口设计**：设计适合 vLLM 的接口（不是 R-KV 的 update_kv）
4. **支持 batch > 1**：突破 R-KV 的 batch=1 限制

Phase 0 的主要价值是**理解算法**和**验证正确性**，为 Phase 1 提供参考基准。

---

*创建日期：2025-01-31*
*更新日期：2025-01-31（架构决策：使用 monkey patch 方案）*
