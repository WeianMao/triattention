# AIME24 官方设置脚本（64 抽样）

运行位置：在仓库根目录（`/data/rbg/users/weian/project/rl/dc`）执行命令，示例：
- FullKV：`bash R-KV/weian_script/wei/run_fullkv_aime24_official64.sh`
- SnapKV：`bash R-KV/weian_script/wei/run_snapkv_aime24_official64.sh`
- R-KV：`bash R-KV/weian_script/wei/run_rkv_aime24_official64.sh`

通用特性：
- 官方设置 = flash_attn2 + bfloat16，默认不启用 reset_cache_each_batch / fp32_topk。
- 64 次采样，温度 0.6、top_p 0.95，max_length 32768，eval_batch_size 固定为 1。
- 使用 `conda run -n rkv` 触发的 Python（由调度器内部处理），日志名仍以 `rkv_aime24_shardXX.log` 命名但不影响实际方法。
- 输出目录位于 `R-KV/outputs/sample64_*_aime24_official/`（各方法对应的子目录）。

可选参数：脚本透传额外参数到 `rkv_sharded_dispatch.py`，例如指定 GPU `--gpus 0,1`，或跳过已完成分片 `--skip-existing`。***
