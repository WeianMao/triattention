# 数据集

## 决策状态：已确认

## 决策

不 release 任何数据集文件，README 中提供 HuggingFace 下载链接，用户自行下载。

决策理由：
- AIME 系列是竞赛题，版权属于 MAA，不宜直接分发
- MATH-500 虽然 MIT 许可，但为统一处理方式也不放进 repo
- 三个数据集在 HuggingFace 上都有可靠的公开来源

## 实验使用的数据集

| 数据集 | 样本数 | 推荐 HuggingFace 来源 | License | 用途 |
|--------|--------|----------------------|---------|------|
| **AIME 2024** | 30 | `HuggingFaceH4/aime_2024` | 未明确标注 | 主实验（80+ 配置引用） |
| **AIME 2025** | 30 | `MathArena/aime_2025` | CC BY-NC-SA 4.0 | 主实验（40+ 配置引用） |
| **MATH-500** | 500 | `HuggingFaceH4/MATH-500` | MIT | speckv_experiments |

## 备选下载源

### AIME 2024
- `HuggingFaceH4/aime_2024`（推荐，HF 官方团队维护）
- `Maxwell-Jia/AIME_2024`
- `math-ai/aime24`
- `di-zhang-fdu/AIME_1983_2024`（历史 AIME 1983-2024 合集）

### AIME 2025
- `MathArena/aime_2025`（推荐，ETH Zurich MathArena 团队维护）
- `MathArena/aime_2025_I`（仅 AIME I，15 题）
- `MathArena/aime_2025_II`（仅 AIME II，15 题）
- `math-ai/aime25`
- `opencompass/AIME2025`

### MATH-500
- `HuggingFaceH4/MATH-500`（推荐，HF 官方团队维护）
- `openai/prm800k`（原始来源，定义了 500 题子集）
- `RLHFlow/Deepseek-MATH500-Test`

## 数据一致性验证

已通过 agent 下载 HuggingFace 数据并与本地文件逐条对比，结果：

| 数据集 | 本地文件 | HF 来源 | 匹配状态 |
|--------|---------|---------|---------|
| AIME24 | `R-KV/HuggingFace/data/aime24.jsonl` | `HuggingFaceH4/aime_2024` | 完全一致 |
| AIME25 | `/datasets/aime25.jsonl` | `MathArena/aime_2025` | 完全一致 |
| MATH-500 | `R-KV/HuggingFace/data/math.jsonl` | `HuggingFaceH4/MATH-500` | 完全一致 |

## 字段名映射

本地 JSONL 和 HuggingFace 版本的字段名有差异，release 代码需要处理：

| 数据集 | 本地字段名 | HuggingFace 字段名 | 需要适配 |
|--------|-----------|-------------------|---------|
| AIME24 | `question` / `answer` | `problem` / `answer` | 是 — `question` → `problem` |
| AIME25 | `question` / `answer` | `problem` / `answer` | 是 — `question` → `problem` |
| MATH-500 | `problem` / `answer` | `problem` / `answer` | 否 — 已一致 |

**处理方案**：release 版本的 data_loader 统一使用 `problem` / `answer` 字段名，与 HuggingFace 格式对齐。

## 数据集自动下载（已确认）

**不在 README 里写下载链接，也不放数据文件进 repo。** 改为在代码中实现自动下载：

- data_loader 在第一次运行时自动从 HuggingFace 下载数据集
- 下载后缓存到本地 `./data/` 目录，后续运行不重复下载
- 下载链接硬编码在 data_loader 代码中

```python
# data_loader.py 中的自动下载逻辑（伪代码）
DATASET_SOURCES = {
    "aime24": "HuggingFaceH4/aime_2024",
    "aime25": "MathArena/aime_2025",
    "math500": "HuggingFaceH4/MATH-500",
}

def load_data(dataset_name, data_dir="./data"):
    local_path = f"{data_dir}/{dataset_name}.jsonl"
    if not os.path.exists(local_path):
        # 自动从 HuggingFace 下载并转换为 JSONL
        ds = load_dataset(DATASET_SOURCES[dataset_name])
        save_as_jsonl(ds, local_path)
    return load_jsonl(local_path)
```

这样用户 clone 后直接运行脚本即可，无需手动下载数据。

### 对比 RKV 官方做法

| 做法 | RKV 官方 | 我们 |
|------|---------|------|
| AIME24 | 直接 commit 到 repo | 代码自动下载 |
| MATH-500 | 直接 commit 到 repo | 代码自动下载 |
| AIME25 | 未公布 | 代码自动下载 |
| 用户操作 | clone 即可用 | clone + 首次运行自动下载 |

## 字段名兼容性（已确认无问题）

用户从 HuggingFace 下载的数据字段名（`problem`）与本地 JSONL（AIME 用 `question`）不一致，但代码中**已有 3 层防护机制**，无需额外处理：

### 第 1 层：`dataset2key` 显式映射

`run_math.py` 和 `rkv_sharded_eval.py` 中定义了每个数据集的字段名：
```python
dataset2key = {
    "aime24": ["question", "answer"],
    "aime25": ["question", "answer"],
    "math": ["problem", "answer"],
    "math500": ["problem", "answer"],
}
```
代码读取对应字段后统一存为 `example["question"]`。

### 第 2 层：`extract_question_from_record()` fallback

`weian_development/speckv/prompt_utils.py` 中按 `["question", "problem"]` 顺序尝试，任一字段存在即可。

### 第 3 层：eval 代码 fallback

`eval_math_multi.py` 中：
```python
if "question" not in sample and "problem" in sample:
    sample["question"] = sample["problem"]
```

### 结论

用户从 HuggingFace 下载数据（字段名 `problem`），代码自动 fallback，**无需用户做任何字段名转换**。Release 时保留现有机制即可。

## 不 release 的数据集相关内容

- BruMO 2025 — 实验脚本中未实际使用，不需要提供
- 校准语料 — 不 release（见 [03_scope_exclude.md](03_scope_exclude.md)）
- Parquet 结果文件 — 运行时产物，不 release
