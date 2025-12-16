# Experiment Conventions

## 概述

本文档定义实验目录结构和文件组织规范，所有后续实验必须遵守。

---

## 默认配置与参考实现

### 1. 默认 Head 索引

所有实验默认使用以下文件中定义的 head 进行实验：

```
weian_development/spec_sparse_simulator/hybrid_sample_heads_lowret_top10.json
```

除非实验明确说明使用其他 head，否则一律使用此文件中的索引。

### 2. 参考实现

没有在文档中明确规定的实现细节，默认遵守以下脚本（这个脚本任何情况下都不允许修改）中的实现：

```
weian_development/spec_sparse_simulator/attention_pruning_case_study_hybrid_rounds_xtrace.py
```

**包括但不限于**：
- Round window 大小（默认 128）
- Attention 计算方式
- RoPE 相关处理
- 数据加载方式
- 可视化方法

### 3. 修改原则

| 情况 | 处理方式 |
|------|----------|
| 文档明确规定 | 按文档执行 |
| 文档未规定，参考实现有 | 遵守参考实现 |
| 用户明确要求修改 | 按用户要求修改 |
| **不确定 / 不清楚** | **必须询问用户或警告用户** |

### 4. 不确定情况处理

> **重要**：对于不确定或不清楚的细节，**禁止擅自做决定**。

**必须采取以下措施之一**：
1. **询问用户**：明确提出问题，等待用户确认
2. **警告用户**：在代码中添加注释或日志，说明当前采用的假设

```python
# ⚠️ WARNING: 此处假设 xxx，未在文档中明确规定
# 如需修改，请参考 attention_pruning_case_study_hybrid_rounds_xtrace.py
# 或联系用户确认
```

**典型需要确认的情况**：
- 超参数选择（如 learning rate、batch size）
- 数据预处理方式
- 评估指标的具体计算方式
- 边界情况处理

---

## 目录结构规范

### 1. 每个实验一个独立文件夹

每个实验必须在 `weian_development/spec_sparse_simulator/experiments/` 下创建独立文件夹。

```
weian_development/spec_sparse_simulator/
├── docs/                          # 文档（本目录）
├── experiments/                   # 实验根目录
│   ├── exp_001_sanity_check/      # 实验 1
│   ├── exp_002_module1_oracle/    # 实验 2
│   ├── exp_003_module2_binning/   # 实验 3
│   └── ...
└── src/                           # 共享代码（可选）
```

### 2. 实验文件夹命名规范

```
exp_{序号}_{简短描述}/
```

| 组成部分 | 规范 | 示例 |
|----------|------|------|
| 序号 | 3 位数字，递增 | `001`, `002`, `010` |
| 简短描述 | 小写字母 + 下划线，简明扼要 | `sanity_check`, `module1_oracle` |

**示例**：
- `exp_001_sanity_check/` - Loss function sanity check
- `exp_002_module1_oracle/` - Module 1 oracle upper bound
- `exp_003_module2_random_baseline/` - Module 2 random binning baseline

### 3. 实验文件夹内部结构

```
exp_001_sanity_check/
├── README.md              # 实验说明（必须）
├── config.yaml            # 配置文件（可选）
├── run.py                 # 主运行脚本
├── *.py                   # 其他脚本
├── output/                # 输出目录（必须）
│   ├── logs/              # 日志
│   ├── checkpoints/       # 模型检查点
│   ├── results/           # 结果文件
│   └── figures/           # 图表
└── data/                  # 实验专用数据（可选，大文件用 symlink）
```

---

## 输出规范

### 1. 所有输出必须在 `output/` 目录下

**禁止**将输出文件放在实验文件夹根目录或其他位置。

```python
# ✓ 正确
output_dir = "output/results/"
log_file = "output/logs/train.log"
checkpoint_path = "output/checkpoints/model_epoch_10.pt"

# ✗ 错误
output_dir = "results/"  # 不在 output/ 下
log_file = "../logs/train.log"  # 跑到实验文件夹外
```

### 2. 输出子目录划分

| 子目录 | 用途 | 文件类型 |
|--------|------|----------|
| `output/logs/` | 训练日志、运行日志 | `.log`, `.txt` |
| `output/checkpoints/` | 模型检查点 | `.pt`, `.pth` |
| `output/results/` | 评估结果、指标 | `.json`, `.csv`, `.pkl` |
| `output/figures/` | 可视化图表 | `.png`, `.pdf`, `.svg` |

### 3. 输出文件命名

包含时间戳或版本号，便于追溯：

```
output/results/metrics_20251215_143052.json
output/checkpoints/model_v1_epoch_100.pt
output/figures/loss_curve_run3.png
```

---

## README.md 规范

每个实验文件夹必须包含 `README.md`，内容包括：

```markdown
# 实验名称

## 目标
简述实验目的（1-2 句话）

## 方法
简述实验方法

## 运行方式
\`\`\`bash
python run.py --config config.yaml
\`\`\`

## 结果摘要
- 关键指标 1: xxx
- 关键指标 2: xxx

## 结论
实验结论（1-2 句话）

## 相关文档
- [01_module1_key_pruning.md](../../docs/01_module1_key_pruning.md)
```

---

## 脚本规范

### 1. 脚本位置

所有实验脚本必须放在实验文件夹内，**禁止**放在其他位置。

```
# ✓ 正确
exp_001_sanity_check/run.py
exp_001_sanity_check/train.py
exp_001_sanity_check/evaluate.py

# ✗ 错误
scripts/exp_001_run.py  # 不在实验文件夹内
```

### 2. 输出路径配置

脚本中使用相对路径，基于实验文件夹：

```python
import os
from pathlib import Path

# 获取实验根目录
EXP_DIR = Path(__file__).parent
OUTPUT_DIR = EXP_DIR / "output"

# 确保输出目录存在
(OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
```

### 3. 配置文件

推荐使用 YAML 配置文件管理实验参数：

```yaml
# config.yaml
experiment:
  name: "sanity_check"
  seed: 42

model:
  num_bins: 128
  num_kernels: 3

training:
  epochs: 100
  lr: 1e-3
  batch_size: 32

output:
  save_every: 10
  log_every: 1
```

---

## 共享代码

如果多个实验共享代码，放在 `src/` 目录：

```
weian_development/spec_sparse_simulator/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── kernel_encoding.py
│   │   ├── module1_network.py
│   │   └── module2_network.py
│   ├── utils/
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── data/
│       └── trace_loader.py
└── experiments/
    └── exp_001_sanity_check/
        └── run.py  # from src.models import ...
```

---

## 快速创建实验模板

```bash
# 创建新实验目录
exp_name="exp_004_new_experiment"
mkdir -p experiments/${exp_name}/output/{logs,checkpoints,results,figures}
touch experiments/${exp_name}/README.md
touch experiments/${exp_name}/run.py
touch experiments/${exp_name}/config.yaml
```

---

## Git 版本控制规范

### 1. 只 commit 代码文件

**必须 commit**:
- `*.py` - Python 脚本
- `*.yaml`, `*.json` - 配置文件（小型）
- `README.md` - 文档
- 其他小型文本文件

**禁止 commit**:
- `output/` 目录下的所有文件（实验结果、日志、图表、checkpoints）
- 大型数据文件（`.pt`, `.pth`, `.pkl` 超过 1MB）
- 临时文件、缓存文件（`__pycache__/`, `*.pyc`）

### 2. 推荐的 .gitignore 配置

在实验目录或项目根目录添加：

```gitignore
# 实验输出（禁止 commit）
experiments/*/output/
**/output/

# 大型数据文件
*.pt
*.pth
*.pkl
*.npy
*.npz

# Python 缓存
__pycache__/
*.pyc
*.pyo

# 临时文件
*.tmp
*.log
```

### 3. Commit 原则

| 文件类型 | 是否 commit | 原因 |
|----------|-------------|------|
| 实验脚本 (`run.py`) | ✓ | 代码需要版本控制 |
| 配置文件 (`config.yaml`) | ✓ | 记录实验配置 |
| README.md | ✓ | 实验文档 |
| 结果 JSON (`output/results/*.json`) | ✗ | 实验结果可重复生成 |
| 图表 (`output/figures/*.png`) | ✗ | 可视化可重复生成 |
| 模型文件 (`*.pt`) | ✗ | 文件过大 |
| 日志文件 (`*.log`) | ✗ | 运行日志不需要版本控制 |

---

## Checklist

开始新实验前，确认以下事项：

- [ ] 在 `experiments/` 下创建了独立文件夹
- [ ] 文件夹命名符合 `exp_{序号}_{描述}/` 格式
- [ ] 创建了 `output/` 子目录结构
- [ ] 创建了 `README.md`
- [ ] 所有脚本都在实验文件夹内
- [ ] 输出路径都指向 `output/` 目录
- [ ] **确认 `output/` 目录不会被 commit**

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2025-12-15 | 初始化文档，定义实验目录结构和文件组织规范 |
| 2025-12-15 | 添加默认配置：指定默认 head 索引文件、参考实现脚本、修改原则、不确定情况处理规范 |
| 2025-12-15 | 添加 Git 版本控制规范：禁止 commit 大文件和实验结果文件 |
