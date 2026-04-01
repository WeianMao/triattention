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

| 环境名 | 用途 |
|--------|------|
| `dc1-env` | DeepConf 开发环境 |
| `trivllm1` | TriAttention_vLLM vLLM V1 后端开发 |
| `lazy_evict` | LazyEviction 子项目 |
| `rkv` | R-KV 压缩实验（Qwen2.5） |
| `rkv1` | R-KV 压缩实验（Qwen3） |

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
