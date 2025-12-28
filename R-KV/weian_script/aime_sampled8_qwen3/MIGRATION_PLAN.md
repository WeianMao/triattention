# Qwen3 迁移计划文档

> 本文档记录将 `aime_sampled8` 实验从 Qwen2.5 (DeepSeek-R1-Distill-Qwen-7B) 迁移到 Qwen3 (DeepSeek-R1-0528-Qwen3-8B) 的完整计划、进度和开发规范。

**创建日期**: 2025-12-28
**最后更新**: 2025-12-28
**负责人**: (待填写)

---

## 目录

1. [项目背景](#1-项目背景)
2. [模型架构差异](#2-模型架构差异)
3. [环境依赖分析](#3-环境依赖分析)
4. [代码隔离要求](#4-代码隔离要求)
5. [文件清单](#5-文件清单)
6. [Flag 功能说明](#6-flag-功能说明)
7. [关键问题与解决方案](#7-关键问题与解决方案)
8. [开发规范](#8-开发规范)
9. [TODO List](#9-todo-list)
10. [变更日志](#10-变更日志)

---

## 1. 项目背景

### 1.1 目标

将 `aime_sampled8` 实验脚本从 Qwen2.5 模型迁移到 Qwen3 模型，以便在新模型上进行 R-KV/SpeckV KV cache 压缩实验。

### 1.2 源目录与目标目录

| 类型 | 路径 |
|------|------|
| 源目录 (Qwen2.5) | `R-KV/weian_script/aime_sampled8/` |
| 目标目录 (Qwen3) | `R-KV/weian_script/aime_sampled8_qwen3/` |
| 源配置 | `R-KV/weian_script/configs/aime_sampled8_*.yaml` |
| 目标配置 | `R-KV/weian_script/configs/aime_sampled8_qwen3_*.yaml` |

### 1.3 模型路径

| 模型 | 路径 |
|------|------|
| Qwen2.5 (7B) | `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B` |
| Qwen3 (8B) | `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B` |

---

## 2. 模型架构差异

### 2.1 架构对比表

| 属性 | Qwen2.5 (7B) | Qwen3 (8B) | 影响 |
|------|-------------|------------|------|
| 架构类型 | `Qwen2ForCausalLM` | `Qwen3ForCausalLM` | 需要不同的 monkeypatch |
| 隐藏层数 (`num_hidden_layers`) | 28 | 36 | Stats 文件不兼容 |
| 注意力头数 (`num_attention_heads`) | 28 | 32 | Stats 文件不兼容 |
| KV 头数 (`num_key_value_heads`) | 4 | 8 | GQA 比例变化 |
| 隐藏维度 (`hidden_size`) | 3584 | 4096 | 内存占用增加 |
| 中间维度 (`intermediate_size`) | 18944 | 12288 | FFN 结构不同 |
| 头维度 (`head_dim`) | 128 | 128 | 相同 |
| 最大位置编码 | 131072 | 131072 | 相同 |
| RoPE 类型 (`rope_type`) | `default` | `yarn` | 位置编码不同 |
| RoPE theta | 10000 | 1000000 | 位置编码不同 |
| 词表大小 | 152064 | 151936 | Tokenizer 不同 |

### 2.2 关键差异总结

1. **层数增加**: 28 → 36 (+8层)
2. **注意力头增加**: 28 → 32 (+4头)
3. **GQA 比例变化**: 28:4=7:1 → 32:8=4:1
4. **RoPE 编码**: 从 default 变为 yarn，theta 从 10000 变为 1000000
5. **模型参数量**: ~7B → ~8B

---

## 3. 环境依赖分析

### 3.1 当前环境状态

```bash
# rkv 环境
conda activate rkv
transformers version: 4.48.1
```

### 3.2 Qwen3 依赖要求

```bash
# Qwen3 模型 config.json 中指定
"transformers_version": "4.51.0"
```

### 3.3 兼容性测试结果

```python
# 测试命令
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

# 结果: ModuleNotFoundError: No module named 'transformers.models.qwen3'
```

### 3.4 环境升级方案

**方案 A: 原地升级 rkv 环境**
```bash
conda activate rkv
pip install transformers>=4.51.0
```
- 优点: 简单直接
- 风险: 可能影响 Qwen2.5 实验的复现性

**方案 B: 克隆 rkv 环境为 rkv1** ✅ 已采用
```bash
conda create --name rkv1 --clone rkv
conda activate rkv1
pip install transformers>=4.51.0
```
- 优点: 完全隔离，不影响原有 rkv 环境
- 缺点: 维护两套环境

**当前决定**: ✅ 已采用方案 B

### 3.5 环境配置结果 (2025-12-28)

| 环境 | 用途 | transformers 版本 | Qwen2 支持 | Qwen3 支持 |
|------|------|-------------------|------------|------------|
| `rkv` | Qwen2.5 实验 (不修改) | 4.48.1 | ✅ | ❌ |
| `rkv1` | Qwen3 实验 | 4.57.3 | ✅ | ✅ |

**验证结果**:
```bash
# rkv1 环境验证 (2025-12-28)
conda activate rkv1
python -c "from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM; print('✓ Qwen2 OK')"
python -c "from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM; print('✓ Qwen3 OK')"
# 输出: ✓ Qwen2 OK, ✓ Qwen3 OK
```

**环境记录位置**: 已更新至项目根目录 `CLAUDE.md`

---

## 4. 代码隔离要求

### 4.1 核心原则

> **所有 Qwen3 相关的修改不能影响 Qwen2.5 的代码和实验可复现性**

### 4.2 隔离策略

#### 4.2.1 脚本隔离

| 类型 | Qwen2.5 | Qwen3 |
|------|---------|-------|
| 脚本目录 | `aime_sampled8/` | `aime_sampled8_qwen3/` |
| 配置文件 | `aime_sampled8_*.yaml` | `aime_sampled8_qwen3_*.yaml` |
| 输出目录 | `outputs/aime_sampled8/` | `outputs/aime_sampled8_qwen3/` |
| 日志目录 | `logs/aime_sampled8/` | `logs/aime_sampled8_qwen3/` |

#### 4.2.2 代码隔离

现有代码已通过模型路径判断实现隔离 (`rkv_sharded_eval.py:461-466`):

```python
if "qwen3" in args.model_path.lower():
    replace_qwen3(compression_config)
elif "qwen" in args.model_path.lower():
    replace_qwen2(compression_config)
```

**原则**:
- 不修改 `replace_qwen2()` 的任何逻辑
- `replace_qwen3()` 可以独立修改
- 公共函数如需修改，必须保持向后兼容

#### 4.2.3 Stats 文件隔离

| 模型 | Stats 路径 |
|------|-----------|
| Qwen2.5 | `outputs/repository/sample8_fullkv_*_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt` |
| Qwen3 | `outputs/repository/sample8_fullkv_*_official_qwen3/stats/deepseek_r1_qwen3_8b_plain_stats.pt` |

### 4.3 禁止的操作

- ❌ 修改 `aime_sampled8/` 目录下的任何文件
- ❌ 修改 `aime_sampled8_*.yaml` 配置文件
- ❌ 修改 `replace_qwen2()` 函数
- ❌ 删除或重命名 Qwen2.5 的 stats 文件
- ❌ 修改公共函数导致 Qwen2.5 行为变化

### 4.4 允许的操作

- ✅ 在 `aime_sampled8_qwen3/` 目录下修改文件
- ✅ 创建新的 `aime_sampled8_qwen3_*.yaml` 配置文件
- ✅ 修改 `replace_qwen3()` 函数
- ✅ 添加新的 Qwen3 专用函数
- ✅ 升级 transformers 版本（如果验证不影响 Qwen2.5）

---

## 5. 文件清单

### 5.1 Shell 脚本 (12个)

#### FullKV (2个)
| 文件 | 数据集 | 状态 |
|------|--------|------|
| `fullkv/aime24/run_fullkv_aime24_qwen.sh` | AIME24 | ✅ 已修改 |
| `fullkv/aime25/run_fullkv_aime25_qwen.sh` | AIME25 | ✅ 已修改 |

#### R-KV (2个)
| 文件 | 数据集 | 状态 |
|------|--------|------|
| `rkv/aime24/run_rkv_aime24_qwen.sh` | AIME24 | ⬜ 待修改 |
| `rkv/aime25/run_rkv_aime25_qwen.sh` | AIME25 | ⬜ 待修改 |

#### SpeckV (8个)
| 文件 | 数据集 | Flag | 状态 |
|------|--------|------|------|
| `speckv/aime24/run_speckv_aime24_qwen_norm.sh` | AIME24 | norm | ⬜ 待修改 |
| `speckv/aime24/run_speckv_aime24_qwen_rank.sh` | AIME24 | rank | ⬜ 待修改 |
| `speckv/aime24/run_speckv_aime24_qwen_norm_aligned.sh` | AIME24 | norm+aligned | ⬜ 待修改 |
| `speckv/aime24/run_speckv_aime24_qwen_rank_aligned.sh` | AIME24 | rank+aligned | ⬜ 待修改 |
| `speckv/aime25/run_speckv_aime25_qwen_norm.sh` | AIME25 | norm | ⬜ 待修改 |
| `speckv/aime25/run_speckv_aime25_qwen_rank.sh` | AIME25 | rank | ⬜ 待修改 |
| `speckv/aime25/run_speckv_aime25_qwen_norm_aligned.sh` | AIME25 | norm+aligned | ⬜ 待修改 |
| `speckv/aime25/run_speckv_aime25_qwen_rank_aligned.sh` | AIME25 | rank+aligned | ⬜ 待修改 |

### 5.2 YAML 配置文件 (12个待创建)

| 源配置文件 | 目标配置文件 | 状态 |
|-----------|-------------|------|
| `aime_sampled8_fullkv_aime24_qwen.yaml` | `aime_sampled8_qwen3_fullkv_aime24.yaml` | ✅ 已创建 |
| `aime_sampled8_fullkv_aime25_qwen.yaml` | `aime_sampled8_qwen3_fullkv_aime25.yaml` | ✅ 已创建 |
| `aime_sampled8_rkv_aime24_qwen.yaml` | `aime_sampled8_qwen3_rkv_aime24.yaml` | ⬜ 待创建 |
| `aime_sampled8_rkv_aime25_qwen.yaml` | `aime_sampled8_qwen3_rkv_aime25.yaml` | ⬜ 待创建 |
| `aime_sampled8_speckv_aime24_qwen_norm.yaml` | `aime_sampled8_qwen3_speckv_aime24_norm.yaml` | ⬜ 待创建 |
| `aime_sampled8_speckv_aime24_qwen_rank.yaml` | `aime_sampled8_qwen3_speckv_aime24_rank.yaml` | ⬜ 待创建 |
| `aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml` | `aime_sampled8_qwen3_speckv_aime24_norm_aligned.yaml` | ⬜ 待创建 |
| `aime_sampled8_speckv_aime24_qwen_rank_aligned.yaml` | `aime_sampled8_qwen3_speckv_aime24_rank_aligned.yaml` | ⬜ 待创建 |
| `aime_sampled8_speckv_aime25_qwen_norm.yaml` | `aime_sampled8_qwen3_speckv_aime25_norm.yaml` | ⬜ 待创建 |
| `aime_sampled8_speckv_aime25_qwen_rank.yaml` | `aime_sampled8_qwen3_speckv_aime25_rank.yaml` | ⬜ 待创建 |
| `aime_sampled8_speckv_aime25_qwen_norm_aligned.yaml` | `aime_sampled8_qwen3_speckv_aime25_norm_aligned.yaml` | ⬜ 待创建 |
| `aime_sampled8_speckv_aime25_qwen_rank_aligned.yaml` | `aime_sampled8_qwen3_speckv_aime25_rank_aligned.yaml` | ⬜ 待创建 |

---

## 6. Flag 功能说明

### 6.1 norm (`--sparse-normalize-scores`)

**功能**: 在聚合前对每个头的稀疏分数进行归一化

**实现位置**: `rkv_sharded_eval.py` → `apply_speckv_generate_patch()` / `apply_speckv_rkv_style_patch()`

**技术细节**:
- 将每个注意力头的分数进行 z-score 标准化
- 使分数在不同头之间具有可比性
- 有助于平衡不同头的贡献

**模型相关性**: ❌ 无关，可直接沿用

### 6.2 rank (`--use-rank-aggregation`)

**功能**: 使用基于排名的聚合策略

**实现位置**: `rkv_sharded_eval.py` → SpeckV 配置

**技术细节**:
- 默认策略: z-score + max-pooling
- rank 策略: 将分数转换为排名，使用 min-pooling 聚合
- min-pooling: 保留所有头中排名最高（最重要）的 token

**模型相关性**: ❌ 无关，可直接沿用

### 6.3 aligned 变体

**功能**: 与 R-KV 论文对齐的实验设置

**特殊配置**:
| 参数 | 非 aligned | aligned |
|------|-----------|---------|
| `sparse_round_window` | 128 | 32 |
| `--include-prefill-in-budget` | false | true |
| `--rkv-style-compression` | false | true |
| stats 来源 | 同数据集 | 交叉数据集 |

**交叉 stats 策略**:
- AIME24 测试使用 AIME25 的 stats（避免数据泄露）
- AIME25 测试使用 AIME24 的 stats

**模型相关性**: ⚠️ 需要新的 stats 文件

---

## 7. 关键问题与解决方案

### 7.1 问题: Transformers 版本不兼容

**现状**: rkv 环境 transformers 4.48.1，Qwen3 需要 ≥4.51.0

**解决方案**:
1. 升级 transformers: `pip install transformers>=4.51.0`
2. 验证升级后 Qwen2.5 实验仍能正常运行
3. 如有问题，回滚并创建新环境

**验证步骤**:
```bash
# 升级前先记录当前版本
pip freeze | grep transformers > /tmp/transformers_version_backup.txt

# 升级
pip install transformers>=4.51.0

# 验证 Qwen2.5
python -c "from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM; print('Qwen2 OK')"

# 验证 Qwen3
python -c "from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM; print('Qwen3 OK')"
```

### 7.2 问题: SpeckV Stats 文件不兼容

**现状**:
- 现有 stats: 28层 × 28头 = 784 个 head 统计
- Qwen3 需要: 36层 × 32头 = 1152 个 head 统计

**解决方案**: 分阶段执行

**阶段 1**: 先运行 FullKV
- FullKV 不需要 stats 文件
- 生成 Qwen3 的 trace 输出

**阶段 2**: 生成 Qwen3 Stats
- 使用 FullKV 输出的 trace 生成 stats
- 参考脚本: `weian_development/attention_qk_analysis/` 目录

**阶段 3**: 运行 SpeckV
- 使用新生成的 Qwen3 stats
- 更新配置文件中的 `sparse_stats_path`

### 7.3 问题: RoPE 编码差异

**现状**:
- Qwen2.5: `rope_type=default`, `theta=10000`
- Qwen3: `rope_type=yarn`, `theta=1000000`

**影响**:
- Stats 文件中记录了 `rope_style` 和 `rope_type`
- SpeckV 使用 RoPE 逆变换进行频率分析

**解决方案**:
- 确保 stats 生成时使用正确的 RoPE 配置
- `round_pruning_utils.py` 中的 `determine_rope_style()` 已能自动检测

---

## 8. 开发规范

### 8.1 命名规范

#### 8.1.1 文件命名

| 类型 | 格式 | 示例 |
|------|------|------|
| Shell 脚本 | `run_{method}_{dataset}_qwen3.sh` | `run_fullkv_aime25_qwen3.sh` |
| YAML 配置 | `aime_sampled8_qwen3_{method}_{dataset}[_{variant}].yaml` | `aime_sampled8_qwen3_speckv_aime25_norm.yaml` |
| Stats 文件 | `deepseek_r1_qwen3_8b_{template}_stats.pt` | `deepseek_r1_qwen3_8b_plain_stats.pt` |

#### 8.1.2 输出路径命名

```
outputs/aime_sampled8_qwen3/{method}/{dataset}/[variant/]
logs/aime_sampled8_qwen3/{method}/{dataset}/[variant/]
```

### 8.2 配置文件规范

#### 8.2.1 必须修改的字段

```yaml
experiment:
  name: aime_sampled8_qwen3_{method}_{dataset}  # 添加 qwen3 标识
  conda_env: rkv1                                # 使用 rkv1 环境 (非 rkv)
  log_dir: R-KV/logs/aime_sampled8_qwen3/...    # 路径包含 qwen3
  method_output_dir: R-KV/outputs/aime_sampled8_qwen3/...
  runner_args:
    model_path: /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B
    output_dir: R-KV/outputs/aime_sampled8_qwen3/.../shards
    sparse_stats_path: R-KV/outputs/repository/sample8_fullkv_{dataset}_official_qwen3/stats/...  # SpeckV only
```

**重要**: Qwen3 实验必须使用 `conda_env: rkv1`，不能使用 `rkv`

### 8.3 测试规范

#### 8.3.1 dry-run 测试

每个脚本修改后，先执行 dry-run 验证:
```bash
bash run_xxx_qwen3.sh --dry-run
```

#### 8.3.2 单样本测试

使用 `--max-examples 1` 进行快速验证:
```bash
bash run_xxx_qwen3.sh --max-examples 1 --num-shards 1
```

### 8.4 Git 提交规范

```
feat(qwen3): add fullkv scripts for Qwen3 model

- Create aime_sampled8_qwen3 experiment directory
- Add YAML configs for fullkv on AIME24/25
- Update shell scripts with new model path

Refs: MIGRATION_PLAN.md
```

---

## 9. TODO List

### 阶段 0: 准备工作

| ID | 任务 | 负责人 | 状态 | 完成日期 | 备注 |
|----|------|--------|------|----------|------|
| P0.1 | 复制 aime_sampled8 目录到 aime_sampled8_qwen3 | Claude | ✅ 已完成 | 2025-12-28 | |
| P0.2 | 创建 MIGRATION_PLAN.md 文档 | Claude | ✅ 已完成 | 2025-12-28 | |
| P0.3 | 确定环境升级方案 | Claude | ✅ 已完成 | 2025-12-28 | 采用方案 B: 克隆为 rkv1 |
| P0.4 | 确定 SpeckV 处理方案 | | ⬜ 待确定 | | 先跑 FullKV |

### 阶段 1: 环境准备

| ID | 任务 | 负责人 | 状态 | 完成日期 | 备注 |
|----|------|--------|------|----------|------|
| P1.1 | 克隆 rkv 环境为 rkv1 | Claude | ✅ 已完成 | 2025-12-28 | `conda create --name rkv1 --clone rkv` |
| P1.2 | 升级 rkv1 的 transformers 到 ≥4.51.0 | Claude | ✅ 已完成 | 2025-12-28 | 4.48.1 → 4.57.3 |
| P1.3 | 验证 Qwen2 导入正常 | Claude | ✅ 已完成 | 2025-12-28 | rkv1 环境测试通过 |
| P1.4 | 验证 Qwen3 导入正常 | Claude | ✅ 已完成 | 2025-12-28 | rkv1 环境测试通过 |
| P1.5 | 更新 CLAUDE.md 记录新环境 | Claude | ✅ 已完成 | 2025-12-28 | |
| P1.6 | 运行 Qwen2.5 FullKV 回归测试 | | ⬜ 未开始 | | 可选：确保 rkv 环境不受影响 |

### 阶段 2: FullKV 配置与脚本

| ID | 任务 | 负责人 | 状态 | 完成日期 | 备注 |
|----|------|--------|------|----------|------|
| P2.1 | 创建 YAML: aime_sampled8_qwen3_fullkv_aime24.yaml | Claude | ✅ 已完成 | 2025-12-28 | |
| P2.2 | 创建 YAML: aime_sampled8_qwen3_fullkv_aime25.yaml | Claude | ✅ 已完成 | 2025-12-28 | |
| P2.3 | 修改脚本: fullkv/aime24/run_fullkv_aime24_qwen.sh → qwen3 | Claude | ✅ 已完成 | 2025-12-28 | |
| P2.4 | 修改脚本: fullkv/aime25/run_fullkv_aime25_qwen.sh → qwen3 | Claude | ✅ 已完成 | 2025-12-28 | |
| P2.5 | dry-run 测试 FullKV AIME24 | Claude | ✅ 已完成 | 2025-12-28 | 使用 rkv1 环境 |
| P2.6 | dry-run 测试 FullKV AIME25 | Claude | ✅ 已完成 | 2025-12-28 | 使用 rkv1 环境 |

### 阶段 3: R-KV 配置与脚本

| ID | 任务 | 负责人 | 状态 | 完成日期 | 备注 |
|----|------|--------|------|----------|------|
| P3.1 | 创建 YAML: aime_sampled8_qwen3_rkv_aime24.yaml | | ⬜ 未开始 | | |
| P3.2 | 创建 YAML: aime_sampled8_qwen3_rkv_aime25.yaml | | ⬜ 未开始 | | |
| P3.3 | 修改脚本: rkv/aime24/run_rkv_aime24_qwen.sh → qwen3 | | ⬜ 未开始 | | |
| P3.4 | 修改脚本: rkv/aime25/run_rkv_aime25_qwen.sh → qwen3 | | ⬜ 未开始 | | |
| P3.5 | dry-run 测试 R-KV AIME24 | | ⬜ 未开始 | | |
| P3.6 | dry-run 测试 R-KV AIME25 | | ⬜ 未开始 | | |

### 阶段 4: 运行 FullKV 实验

| ID | 任务 | 负责人 | 状态 | 完成日期 | 备注 |
|----|------|--------|------|----------|------|
| P4.1 | 运行 FullKV AIME24 完整实验 | | ⬜ 未开始 | | 生成 trace |
| P4.2 | 运行 FullKV AIME25 完整实验 | | ⬜ 未开始 | | 生成 trace |
| P4.3 | 验证 FullKV 输出格式正确 | | ⬜ 未开始 | | |

### 阶段 5: 生成 Qwen3 Stats

| ID | 任务 | 负责人 | 状态 | 完成日期 | 备注 |
|----|------|--------|------|----------|------|
| P5.1 | 确认 stats 生成脚本路径 | | ⬜ 未开始 | | 参考 attention_qk_analysis/ |
| P5.2 | 修改/创建 Qwen3 stats 生成脚本 | | ⬜ 未开始 | | 确保隔离 |
| P5.3 | 生成 AIME24 Qwen3 stats | | ⬜ 未开始 | | |
| P5.4 | 生成 AIME25 Qwen3 stats | | ⬜ 未开始 | | |
| P5.5 | 验证 stats 文件格式 | | ⬜ 未开始 | | 检查 sampled_heads 等 |

### 阶段 6: SpeckV 配置与脚本

| ID | 任务 | 负责人 | 状态 | 完成日期 | 备注 |
|----|------|--------|------|----------|------|
| P6.1 | 创建 8 个 SpeckV YAML 配置 | | ⬜ 未开始 | | norm/rank × aligned × aime24/25 |
| P6.2 | 修改 8 个 SpeckV Shell 脚本 | | ⬜ 未开始 | | |
| P6.3 | 更新 sparse_stats_path 到 Qwen3 stats | | ⬜ 未开始 | | |
| P6.4 | dry-run 测试所有 SpeckV 脚本 | | ⬜ 未开始 | | |

### 阶段 7: 完整实验运行

| ID | 任务 | 负责人 | 状态 | 完成日期 | 备注 |
|----|------|--------|------|----------|------|
| P7.1 | 运行 R-KV AIME24 完整实验 | | ⬜ 未开始 | | |
| P7.2 | 运行 R-KV AIME25 完整实验 | | ⬜ 未开始 | | |
| P7.3 | 运行 SpeckV norm AIME24 | | ⬜ 未开始 | | |
| P7.4 | 运行 SpeckV rank AIME24 | | ⬜ 未开始 | | |
| P7.5 | 运行 SpeckV norm_aligned AIME24 | | ⬜ 未开始 | | |
| P7.6 | 运行 SpeckV rank_aligned AIME24 | | ⬜ 未开始 | | |
| P7.7 | 运行 SpeckV norm AIME25 | | ⬜ 未开始 | | |
| P7.8 | 运行 SpeckV rank AIME25 | | ⬜ 未开始 | | |
| P7.9 | 运行 SpeckV norm_aligned AIME25 | | ⬜ 未开始 | | |
| P7.10 | 运行 SpeckV rank_aligned AIME25 | | ⬜ 未开始 | | |

### 阶段 8: 结果验证与清理

| ID | 任务 | 负责人 | 状态 | 完成日期 | 备注 |
|----|------|--------|------|----------|------|
| P8.1 | 验证所有实验结果完整性 | | ⬜ 未开始 | | |
| P8.2 | 运行 eval_math_multi.py 评估 | | ⬜ 未开始 | | |
| P8.3 | 对比 Qwen2.5 vs Qwen3 结果 | | ⬜ 未开始 | | |
| P8.4 | 更新 MIGRATION_PLAN.md | | ⬜ 未开始 | | 标记完成 |
| P8.5 | 清理临时文件 | | ⬜ 未开始 | | |

---

## 10. 变更日志

| 日期 | 修改人 | 修改内容 |
|------|--------|----------|
| 2025-12-28 | Claude | 创建初始文档，完成分析和规划 |
| 2025-12-28 | Claude | 创建 rkv1 环境 (克隆自 rkv)，升级 transformers 4.48.1 → 4.57.3 |
| 2025-12-28 | Claude | 验证 rkv1 环境支持 Qwen2/Qwen3，更新 CLAUDE.md |
| 2025-12-28 | Claude | 测试 Qwen3 模型加载和推理，验证环境可用 |
| 2025-12-28 | Claude | 完成阶段 2: 创建 FullKV YAML 配置和脚本，dry-run 测试通过 |
| | | |

---

## 附录

### A. 相关文件快速参考

```bash
# 核心脚本
R-KV/weian_development/rkv_sharded_dispatch.py  # 调度器
R-KV/weian_development/rkv_sharded_eval.py      # 实际 runner
R-KV/HuggingFace/rkv/monkeypatch.py             # 模型 patch

# Stats 生成
weian_development/attention_qk_analysis/capture_qk_distributed.py

# SpeckV 实现
R-KV/weian_development/speckv/speckv_rkv_style.py
R-KV/rkv/compression/speckv.py
```

### B. 常用命令

```bash
# 检查环境
conda activate rkv
python -c "import transformers; print(transformers.__version__)"

# dry-run 测试
bash run_xxx.sh --dry-run

# 单样本快速测试
bash run_xxx.sh --max-examples 1 --num-shards 1 --gpus 0

# 查看 stats 文件内容
python -c "import torch; s=torch.load('xxx_stats.pt'); print(s['metadata'])"
```

### C. 问题反馈

如遇到问题，请在本文档 [变更日志](#10-变更日志) 中记录，或联系负责人。
