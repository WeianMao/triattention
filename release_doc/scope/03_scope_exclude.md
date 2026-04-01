# 排除清单

## 校准相关

- :x: 校准语料（calibration data）
- :x: 校准脚本（calibration scripts）
- :x: 任何涉及在 AIME 数据集上校准的信息
- :white_check_mark: 校准结果文件可以 release，但文件名/路径中**不能出现 "aime" 字样**（需要重命名或泛化）

## 敏感内容

- :x: 数据敏感内容
- :x: 进程伪装相关代码（`PD-L1_binder`、`mask_process_command`、`process_utils.py` 等）
- :x: 内部服务器路径（`/data/rbg/`、`weian`、CSAIL 基础设施引用）— 详见 [../code_cleanup/06_path_cleanup.md](../code_cleanup/06_path_cleanup.md)

## 历史遗留

- :x: `deepconf/` — 历史遗留文件夹，与当前开发无关
- :x: `TriAttention/` — 空目录

## 实验性 Flag 和代码

- :x: 试验过但后来没有采用的 flag 及其对应代码实现
- 本地代码可以保留这些，但 release 版本中要去掉
- 需要后续步骤：对比起点脚本使用的 flag vs 代码中所有 flag，识别实验性的，列出确认后删除

## 不公布的文件

- `sparse_round_pruner_prefill_keep.py` — 废弃旧实现
- `rkv_speckv_generate.py` — 废弃旧实现
- `analysiskv.py` — 内部分析工具
- 校准脚本和校准语料
- 进程伪装代码（PD-L1_binder, mask_process_command 等）
- `weian_development/` 中的个人开发工具
- `scripts/gpu_occupier.py` 及测试脚本 — GPU 占领工具
- `.claude/`、`.workflow/` — 开发工具配置

## 已确认不公布的目录

- `paper_visualizations/` — 论文可视化脚本（已确认不公布）
- `experiments/` — 第一阶段不公布，第二阶段视情况（已确认）

## 共享 Symlink 目录（不应 release）

以下目录是 dc 和 dc1 之间的共享 symlink，包含运行时产物：
- `R-KV/logs/`
- `R-KV/outputs/`
- `R-KV/vLLM/`
- `R-KV/SGLang/`

## 敏感关键词扫描清单

在 clean-room 阶段（阶段 2），需要全局扫描以下关键词确保不遗漏：

| 关键词 | 为什么敏感 |
|--------|-----------|
| `aime` | 校准数据集引用 |
| `weian` | 个人用户名 |
| `/data/rbg` | 内部服务器路径 |
| `PD-L1` | 进程伪装 |
| `mask_process` | 进程伪装 |
| `binder` | 进程伪装 |
| `csail` | MIT 内部基础设施 |
| `gpu_occupier` | GPU 占领工具 |
