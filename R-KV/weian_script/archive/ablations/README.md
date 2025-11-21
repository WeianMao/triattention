# Sharded Ablation Launchers (AIME24)

便于分别观测三个因素的影响：注意力后端（flash_attn2 vs sdpa）、`fp32_topk`、`reset_cache_each_batch`。所有脚本均使用 8 shards 与 AIME24 数据集，输出到各自目录，不会覆盖已有结果。

快捷对照（均依赖 `rkv` conda 环境）：
- `run_flash_fp32topk.sh`：flash_attn2 + bf16 + `fp32_topk=True`（无 reset）。  
- `run_flash_reset.sh`：flash_attn2 + bf16 + `reset_cache_each_batch=True`（无 fp32_topk）。  
- `run_sdpa_bf16.sh`：sdpa + bf16（无 fp32_topk、无 reset），只改后端。  
- `run_sdpa_bf16_fp32topk_reset.sh`：sdpa + bf16 + `fp32_topk=True` + `reset_cache_each_batch=True`（与现有 fp16 版本区分开，隔离 dtype 影响）。

使用示例：`bash run_flash_fp32topk.sh`（脚本会自动写入日志/输出到独立目录）。
