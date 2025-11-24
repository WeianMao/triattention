# SpecKV 迁移与 quick 小实验运行记录（最新修订）

## 迁移与实现概述
- 目标：把 LazyEviction 的 `sparse_prefill_keep` 算法移植到 R-KV，并改名 **SpecKV**，用于与 R-KV 官方配置公平对比（prompt 不压缩、kv_budget 对齐、flash_attn2 + bf16）。
- 实现点：
  - 在 `rkv_sharded_eval.py` 增加方法枚举 `speckv`（等价之前的 sparse_prefill_keep），引入 `SparseRoundPruner` 流程，保持 prompt 全量保留，解码阶段裁剪。
  - 引入聊天模板开关、稀疏统计路径、round_window 等参数，保持和 FullKV 统计一致。
  - 修正 resume 判定逻辑（由他人提交，已合入）：完成判定基于“题目数 × 抽样数”而非仅行数，避免残缺 shard 被视为完成；`--output-dir` 可覆盖 YAML 的 runner 输出目录，便于切换干净目录。
  - 删除 quick_v2 临时测试配置，保持与 R-KV 基线一致的 chat template。
- 统计阶段（已完成）：使用 FullKV trace 3 条生成 SpecKV 统计文件  
  `R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt`  
  脚本：`R-KV/weian_development/rkv_sparse_round_calibrate.py`。

## 相关脚本/配置
- 官方 8/64 抽样脚本（已改名 speckv）：  
  - `R-KV/weian_script/aime24_official_sampled8/run_sparseprefillkeep_aime24_official_sampled8.sh` → `configs/sample8_sparseprefillkeep_aime24_official.yaml`（内部已改 method=speckv）  
  - `R-KV/weian_script/wei/run_sparseprefillkeep_aime24_official64.sh` → `configs/sample64_sparseprefillkeep_aime24_official.yaml`
- 快速小实验配置：`R-KV/weian_script/configs/sample8_speckv_aime24_quick.yaml`（6 题 × 8 抽样，max_len=16k）
- 快速脚本：`R-KV/weian_script/quick_tests/run_speckv_aime24_quick.sh`（同上 YAML）

## 本次运行需求
- 运行 SpecKV 快速小实验（AIME24，6 题 × 8 抽样，max_len=16k，chat template 对齐 R-KV），验证 resume 逻辑。
- 避开繁忙 GPU；删除旧输出；保留 `--skip-existing` 测试 resume；使用全新输出/日志目录。
- 已知 resume 修复已合入（参考 `R-KV/docs/resume_fix_plan.md`）。

## 环境/配置
- 仓库：`/data/rbg/users/weian/project/rl/dc`
- Conda 环境：`rkv`
- 配置文件：`R-KV/weian_script/configs/sample8_speckv_aime24_quick.yaml`
- 输出目录：`R-KV/outputs/sample8_speckv_aime24_quick_clean`
- 日志目录：`R-KV/logs/sample8_speckv_aime24_quick_clean`
- GPU：空闲 1/2/3/4（避开卡 0，其他卡空闲）

## 操作步骤
1. 清理旧目录  
   ```bash
   rm -rf R-KV/outputs/sample8_speckv_aime24_quick \
          R-KV/outputs/sample8_speckv_aime24_quick_clean \
          R-KV/logs/sample8_speckv_aime24_quick \
          R-KV/logs/sample8_speckv_aime24_quick_clean
   ```
2. GPU 余量检查  
   ```bash
   nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
   ```
   选用 1/2/3/4。
3. dry-run 验证（检查命令/分片映射，未执行）  
   ```bash
   PYTHONPATH=/data/rbg/users/weian/project/rl/dc \
   conda run -n rkv python R-KV/weian_development/rkv_sharded_dispatch.py \
     --config R-KV/weian_script/configs/sample8_speckv_aime24_quick.yaml \
     --gpus 1,2,3,4,5,6 --num-shards 4 --skip-existing --dry-run \
     --method-output-dir R-KV/outputs/sample8_speckv_aime24_quick_clean \
     --log-dir R-KV/logs/sample8_speckv_aime24_quick_clean \
     --output-dir R-KV/outputs/sample8_speckv_aime24_quick_clean/shards
   ```
   输出显示每 shard 缺失 run 列表，命令行、日志路径正确。
4. 正式运行（新目录空，`--skip-existing` 不跳过）  
   ```bash
   PYTHONPATH=/data/rbg/users/weian/project/rl/dc \
   conda run -n rkv python R-KV/weian_development/rkv_sharded_dispatch.py \
     --config R-KV/weian_script/configs/sample8_speckv_aime24_quick.yaml \
     --gpus 1,2,3,4,5,6 --num-shards 4 --skip-existing \
     --method-output-dir R-KV/outputs/sample8_speckv_aime24_quick_clean \
     --log-dir R-KV/logs/sample8_speckv_aime24_quick_clean \
     --output-dir R-KV/outputs/sample8_speckv_aime24_quick_clean/shards
   ```
   - 调度：shard0→GPU1, shard1→GPU2, shard2→GPU3, shard3→GPU4。
   - 自动 merge+eval。

## 运行结果
- 合并：`R-KV/outputs/sample8_speckv_aime24_quick_clean/merged/merged.jsonl`，48 条（6 题 × 8 抽样），无缺失。
- 评测：`acc=4.2`（按论文定义：同题 8 次采样取均值后再对题目取均值；旧版评测只看首抽样，现已修正）。  
  评测文件：`R-KV/outputs/sample8_speckv_aime24_quick_clean/eval/sample8_speckv_aime24_quick_clean/aime24/default-default_math_multi_eval.jsonl`
- 日志：`R-KV/logs/sample8_speckv_aime24_quick_clean/rkv_aime24_shard0[0-3].log`

## 备注/后续
- quick_v2 已删除，保持 chat template 与基线一致（目录仍保留旧 shard 便于对比）。
- resume 修复已生效（行数判定基于题目数×抽样数，`--output-dir` 可覆盖 YAML），本次运行未出现跳过。
- 若需提高准确率，需检查生成质量（shard 日志、merged 内容）、prompt 或 seed 等设置；评测指标已对齐论文（采样均值）。 
