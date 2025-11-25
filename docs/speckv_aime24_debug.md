# SpecKV AIME24 Debug Notes (handoff)

## 背景
- 任务：将 LazyEviction 的 `sparse_prefill_keep`（现名 SpecKV）移植到 R-KV，并与官方 RKV / SnapKV 在 AIME24 (8 draws) 公平对比。
- 现象：SpecKV 精度极低（quick 跑 4%），与预期、与 SnapKV 不符。

## 已发现/修复的问题
1) **RoPE 适配错误**  
   - 之前对 Llama3 模型使用了 Qwen3 RoPE，频率/缩放错配。  
   - 修复：`build_rotary` 依据 model_type 使用 `LlamaRotaryEmbedding`，并用该实现重跑校准。
   - 重新校准命令（已执行）：  
     ```bash
     export PYTHONPATH=/data/rbg/users/weian/project/rl/dc
     conda run -n rkv python R-KV/weian_development/rkv_sparse_round_calibrate.py \
       --trace-root R-KV/outputs/sample8_fullkv_aime24_official \
       --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B \
       --output-path R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt \
       --num-traces 3 --use-chat-template --dtype float16
     ```
   - 校准文件已覆盖：`R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt`

2) **剪枝窗口过小**  
   - SpeckV YAML 误设 `window_size`/`sparse_round_window` 为 8（原设计为百级）。  
   - 已改为 128（与 LazyEviction 同级）：  
     - `R-KV/weian_script/configs/sample8_speckv_aime24_quick.yaml`  
     - `R-KV/weian_script/configs/sample8_sparseprefillkeep_aime24_official.yaml`  
     - `R-KV/weian_script/configs/sample64_sparseprefillkeep_aime24_official.yaml`

3) **max_length 被覆盖**  
   - `rkv_sharded_eval.py` 之前无条件用数据集默认覆盖 YAML，导致解码过长。  
   - 已修正：只有未设置或 ≤0 时才落到 `dataset2max_length`。

4) **官方 8 抽样脚本补齐**  
   - 新增：`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`  
   - 与同目录其它脚本风格一致，输出目录沿用官方命名。

## 仍需验证/可能剩余问题
- 重新跑官方 8 抽样 SpeckV（修复后的 RoPE + 128 窗口 + 新 stats + 正确 max_length）看精度是否恢复到与 RKV/SnapKV 同级。  
  示例运行（可透传 `--gpus/--num-shards` 等）：  
  ```bash
  bash R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh --gpus 0,1,2,3 --num-shards 4
  ```
- 生成长度/停止条件：如仍长尾导致“最后数字”噪声，可在 YAML 调低 `max_length` 或添加基于 “Final answer” 的 stop words。
- 如果精度仍异常，需进一步对比：  
  - FlashAttention/position_ids 与 pruner 的 absolute_position 同步性；  
  - 头映射（32 attention heads / 8 KV heads）是否有模型侧差异；  
  - 评测解析是否受长输出影响（Math 多采样按列均值评估）。

## 代码变更摘要（关键点）
- `weian_development/hf_offline_runner_sparse/round_pruning_utils.py`: Llama 模型使用 `LlamaRotaryEmbedding`，Qwen 仍用 Qwen3/Qwen2。  
- `R-KV/weian_development/rkv_sharded_eval.py`: max_length 不再无条件覆盖。  
- SpeckV YAML: `window_size`/`sparse_round_window` 调至 128。  
- 新脚本：`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`

## 路径速查
- 校准脚本：`R-KV/weian_development/rkv_sparse_round_calibrate.py`
- 校准结果：`R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt`
- SpeckV 官方 8 抽样脚本：`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`
- SpeckV 官方 8 配置：`R-KV/weian_script/configs/sample8_sparseprefillkeep_aime24_official.yaml`
- max_length 约定：官方 YAML 默认 32768（sample8/sample64）；quick YAML 默认 16000（仅 smoke）。
