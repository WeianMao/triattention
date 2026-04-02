# 开发环境背景（接手人须知）

## 决策状态：已确认

## Replica 信息

- 当前开发在 **dc1**（`/data/rbg/users/weian/project/rl/dc1`），是 dc 的 parallel copy
- 原始项目：`/data/rbg/users/weian/project/rl/dc`

## 共享 Symlink 目录（不应 release）

以下目录是 dc 和 dc1 之间的共享 symlink，包含运行时产物，不应进入 release：
- `R-KV/logs/`
- `R-KV/outputs/`
- `R-KV/vLLM/`
- `R-KV/SGLang/`

## Conda 环境

### 现有环境

| 环境名 | 用途 | 备注 |
|--------|------|------|
| `dc1-env` | DeepConf 开发环境 | **未用于** R-KV/SpeckV 实验 |
| `trivllm1` | TriAttention_vLLM vLLM V1 后端开发 | |
| `lazy_evict` | LazyEviction 子项目 | |
| `rkv` | R-KV 压缩实验（Qwen2.5） | 绑定 dc 目录，非 dc1；后期被升级过（transformers 4.57.6） |
| `rkv1` | R-KV 压缩实验（Qwen3） | 协作者使用，本地可能未完整配置 |

### 论文实验实际使用的环境

经调查（2026-04-02），论文实验**均未使用 `dc`/`dc1-env`**，而是：

| 实验 | conda 环境 | 证据 |
|------|-----------|------|
| DS-Qwen-7B, DS-Llama-8B（所有 AIME/MATH） | `rkv` | yaml `conda_env: rkv`，日志 python3.10 |
| Qwen3 fullkv (weian_script) | `rkv1` | yaml `conda_env: rkv1` |
| Qwen3 speckv/消融 (speckv_experiments) | `rkv` | runner_defaults.yaml 默认值，日志确认 |
| GPT-OSS-20B | 协作者环境（未知） | Phase 1 搁置 |

实验从 `dc`（原始目录）运行，非 `dc1`。`rkv` 环境绑定 dc 目录。

### Release 验证环境（待创建）

**决策：需要新建一个 conda 环境**，cover 除 GPT-OSS 外的所有模型（DS-Qwen-7B, DS-Llama-8B, Qwen3-8B）。

- 环境名：`triattention`（建议）
- Python: 3.10（与 rkv 环境一致）
- 基于 `rkv` 环境的已知工作版本创建
- `requirements.txt` 必须与此环境对齐
- 用于：release 代码验证测试（Level 1/2 单元测试 + Level 3 端到端）
- 必须能跑通所有 3 个非 GPT-OSS 模型（DS-Qwen-7B, DS-Llama-8B, Qwen3-8B）

#### 创建步骤（草案）

```bash
# 1. 创建基础环境
conda create -n triattention python=3.10 -y
conda activate triattention

# 2. 安装 PyTorch（根据本机 CUDA 版本选择）
# 如果是 CUDA 12.1:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121
# 如果是 CUDA 11.8:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 flash-attn（需要匹配 torch + CUDA 版本）
pip install flash-attn==2.5.8

# 4. 安装核心依赖
pip install transformers>=4.48.1 datasets huggingface-hub accelerate
pip install numpy pyyaml tqdm matplotlib scipy einops sentencepiece

# 5. 安装评估管线依赖
pip install pebble sympy regex latex2sympy2 word2number
pip install antlr4-python3-runtime==4.7.2  # latex2sympy2 要求精确版本

# 6. 安装 release 包本身（editable mode）
cd /data/rbg/users/weian/project/rl/dc1
pip install -e .

# 7. 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from flash_attn import flash_attn_func; print('flash-attn OK')"
python -c "import transformers; print(transformers.__version__)"
```

#### 关键版本参考（来自 rkv 环境）

| 包 | rkv 实际版本 | requirements.txt 建议 |
|---|---|---|
| Python | 3.10.19 | `>=3.10` |
| torch | 2.6.0+cu118 | `>=2.3.1`（注释说明 CUDA 安装） |
| flash-attn | 2.5.8 | `>=2.5.8` |
| transformers | 4.57.6 | `>=4.48.1` |
| numpy | 1.26.4 | `>=1.26` |
| datasets | 4.4.1 | `>=4.0` |
| huggingface-hub | 0.36.0 | `>=0.35` |
| pyyaml | 6.0.3 | `>=6.0` |
| sympy | 1.13.1 | `>=1.13` |
| pebble | 5.1.3 | `>=5.0` |
| antlr4-python3-runtime | — | `==4.7.2`（精确pinned） |

#### 注意事项

- `rkv` 环境后期升级过 transformers（从 4.48.1 → 4.57.6），Qwen3 需要较新版本
- `flash-attn` 安装较复杂，依赖 CUDA toolkit 版本和 torch 版本的匹配
- 环境创建后需跑 Level 1+2 单元测试验证基本功能

### 模型权重位置

模型权重在 `/data/rbg/users/weian/project/rl/datasets/`（dc 和 dc1 共享上级目录）：

| 模型 | 本地路径 | HuggingFace hub 名 |
|------|---------|-------------------|
| DS-Qwen-7B | `../datasets/DeepSeek-R1-Distill-Qwen-7B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| DS-Llama-8B | `../datasets/DeepSeek-R1-Distill-Llama-8B` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| Qwen3-8B | `../datasets/DeepSeek-R1-0528-Qwen3-8B` | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` |
| GPT-OSS-20B | `../datasets/gpt-oss-20b` | `openai/gpt-oss-20b`（Phase 1 搁置） |

## GPU 占座进程（PD-L1）

本机上有一个名为 **PD-L1** 的常驻进程，功能是 GPU "占座"：

- **作用**：在我们不使用 GPU 时把显卡打满，防止其他用户抢占
- **自动让位**：PD-L1 能检测到其他进程需要 GPU，会自动释放资源
- **使用方式**：需要跑 GPU 任务时，直接往被 PD-L1 占着的卡上提交任务即可，**不需要手动 kill PD-L1**，它会自己让出来
- **相关代码**：`scripts/gpu_occupier.py`（不公布）

## 其他排除目录

以下目录为开发过程中间产物或个人工具，不应 release：
- `weian_development/` -- 个人开发脚本和工具
- `scripts/gpu_occupier.py` 及测试脚本 -- GPU 占领工具
- `.claude/`、`.workflow/` -- 开发工具配置

以下目录**已确认**不 release：
- `paper_visualizations/` -- 论文可视化脚本（不公布）
- `experiments/` -- 第一阶段不公布，第二阶段视情况
