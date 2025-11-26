# SpecKV AIME24 Debug Notes (handoff)

## 背景
- 任务：将 LazyEviction 的 `sparse_prefill_keep`（现名 SpecKV）移植到 R-KV，并与官方 RKV / SnapKV 在 AIME24 (8 draws) 公平对比。
- 现象：SpecKV 精度极低（quick 跑 4%），与预期、与 SnapKV 不符。

## 2025-02-xx 当前排查（本轮对话）
- 用户要求：对齐 R-KV baseline 的生成方式，算法仍用 SpecKV，但“实现/设置”尽量一致；若 RKV 用 `model.generate`，SpecKV 未超预算时也应走 generate；prompt 模板也要对齐。
- 发现的问题：
  1) 直通逻辑失效导致短序列也走手写采样：此前在 `example_offline_hf_serialized.py` 判断 `input_len + max_new_tokens <= kv_budget`，而 `max_new_tokens` 从 max_length(32k) 算出，直通永远不触发；即便总长 < 2k 也用自写循环，输出偏离 baseline。
  2) 自写采样与 RKV 生成不同：只用了温度+top-p，停条件只有 EOS；RKV baseline 直接用 HF generate（带默认处理），prompt 也不同（SpecKV 之前用了 chat_template）。
  3) 我曾尝试把 `max_new_tokens` 截断到 kv_budget=2048 以强制“预算内直通”，但与 RKV 行为不一致（RKV 不截断 max_new_tokens，而是在注意力里按预算裁剪）。
- 本轮已做调整：
  - 去掉 `max_new_tokens=min(..., kv_budget)` 截断，保持与 RKV 一样由 max_length/上下文上限决定（约 32k）。
  - 保留“预算内走 generate”判定（`max_new_tokens <= kv_budget`），但在当前上限下几乎不会触发，行为更接近 RKV。
  - 统一 YAML 提示模板为非 chat（`use_chat_template:false`），与 RKV baseline 提示一致。
- 运行状态：
  - 用户跑官方脚本时日志为空；调度器日志在进程结束前不 flush，看起来像卡住。
  - 我在 6 号卡跑最小 smoke（1 题 1 抽样，max_length=4096，kv_budget=2048，FA2，chat_template 关）120 秒超时未落盘，只见一条 FA2 提示；可能需要更长时间或换 SDPA/缩短生成长度。
- 待办/建议：
  - 如需快速验证流程，先用较小 `max_length` 或 `attn_implementation=sdpa`，或加 stop words，确保能产出结果。
  - 若要恢复“预算内直通 generate”，需另行设计不依赖截断 max_new_tokens 的判定，否则当前上限 32k 直通几乎不生效。

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

## 2025-02-xx 额外排查记录
- **FA2 位置对齐**：在 `weian_development/hf_offline_runner_sparse/example_offline_hf_serialized.py` 增量解码时显式传 `cache_position`，避免 flash-attn 路径用裁剪后长度推位置导致 pruner/模型坐标系错位。  
- **短序列直通**：若 `prompt_len + max_new_tokens <= max_keys`，直接走 `model.generate` 全量 KV，不触发剪枝（保证长度远小于 kv_budget 时与 FullKV 等价）。  
- **统计重算（FA2）**：`R-KV/weian_development/rkv_sparse_round_calibrate.py` 增加 `--attn-implementation`（默认 FA2），并用 FA2 重采 3 条 trace 生成 `R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt`（元数据已记录 attn 实现、chat_template=true）。  
- **配置恢复**：此前误把 YAML 的 `use_chat_template` 改成 false，已恢复为 true（sample8/64 official + quick）。

## 2025-02-xx 待执行计划（当前代理断点记录）
- 背景：用户要求 SpeckV 在 R-KV 下完全对齐官方 baseline（prompt/tokenization/generate/stop 条件等），弃用 LazyEviction 的手写采样路径；校准与采样的 `use_chat_template` 必须一致；若此前为适配 R-KV 改动了 `weian_development/hf_offline_runner_sparse`（原 Lazy 实现），需回退这些改动，避免污染 Lazy。
- 计划步骤：
  1) **Git 历史核查**：检查 `weian_development/hf_offline_runner_sparse` 是否被改动以适配 R-KV；若发现 Lazy 路径被改写，先定位原始版本并恢复为 Lazy-only（R-KV 侧将新建原生实现）。
  2) **SpeckV 原生化**（R-KV）：在 `R-KV/weian_development` 下实现 SpeckV 的 R-KV 版本，复用基线的 prompt/tokenization（`add_special_tokens=True`）、`model.generate` 流程、stop 条件、`reset_cache_each_batch` 等开关，并通过 R-KV monkeypatch 接入裁剪。
  3) **校准/运行对齐**：确保 SpeckV 运行用的 `use_chat_template` 与校准脚本一致；必要时调整 SpeckV YAML 或重跑校准（保持相同模板）。
  4) **配置与文档**：更新 SpeckV 相关 YAML（chat_template 与 baseline 对齐）、脚本调用路径，以及本文件的现状说明；记录验证建议（小样本 smoke + 校准一致性检查）。
  5) **验证建议**：完成后跑一次小规模 smoke（少题/少抽样、短 max_length、FA2 或 SDPA），确认输出落盘和评测正常。

（以上为断点记录，后续执行从步骤 1 开始）

## 2025-02-xx 本轮进展
- 已回退 Lazy 路径中为 R-KV 适配加入的短路 + `cache_position` 变更，保持 `weian_development/hf_offline_runner_sparse` 纯 Lazy 行为。
- SpeckV 在 R-KV 下改为原生实现：`R-KV/weian_development/rkv_sharded_eval.py` 内置采样/裁剪（`speckv_generate_sequence`），不再依赖 Lazy 的 `run_sparse_generation`；tokenization 使用 `add_special_tokens=True`，支持 `reset_cache_each_batch`，提示/采样开关与基线一致。
- SpeckV YAML 统一 `use_chat_template: true` 以匹配现有校准文件（chat 模板统计仍沿用 `deepseek_r1_llama8b_chat_stats.pt`；尚未产出非 chat stats）。涉及：`sample8/64_sparseprefillkeep_aime24_official.yaml`、`sample8_speckv_aime24_quick.yaml`。
- 试图用非 chat 模板重跑校准时报错：  
  - `PYTHONPATH=/data/rbg/users/weian/project/rl/dc conda run -n rkv python R-KV/weian_development/rkv_sparse_round_calibrate.py ...` → `No such file or directory`（conda run 版本位于 `/data/rbg/users/weian/env/miniconda3/bin/conda`）。  
  - 直接 `/data/rbg/users/weian/env/miniconda3/envs/rkv/bin/python R-KV/weian_development/rkv_sparse_round_calibrate.py ...` → `ModuleNotFoundError: No module named 'weian_development'`（缺 PYTHONPATH）。  
  - 目前未拿到非 chat stats；仍使用 chat stats。
- 试图在 GPU3 上跑 quick smoke：`PYTHONPATH=... conda run -n rkv python R-KV/weian_development/rkv_sharded_dispatch.py --config ... --gpus 3 --num-shards 1 --skip-existing --method-output-dir ... --log-dir ... --output-dir ... --no-eval` 同样报 `No such file or directory`，未能验证新路径；可能与当前 shell/conda run 触发的路径或可执行依赖缺失有关。
- 下一步建议：  
  1) 优先修复 `conda run`/解释器调用问题（可尝试：用完整 python 路径 + 导出 `PYTHONPATH=/data/rbg/users/weian/project/rl/dc`；或在 shell 直接 `source activate rkv` 后运行相同命令），确认能调用 `R-KV/weian_development/rkv_sparse_round_calibrate.py` 与 `rkv_sharded_dispatch.py`。  
  2) 与运行模板一致地重跑校准（chat 或非 chat）；若要非 chat 版本，生成新 stats 并更新 SpeckV YAML。  
  3) 选空闲 GPU（3 或 6 当前空闲）跑一次小样本 smoke（可用 quick YAML，加 `--no-eval`，短 `max_length`/`max_examples`/`num_samples=1`），确认落盘和评测链路正常。  
  4) 完成后更新本文件记录结果和剩余问题。
