# SpecKV-Qwen 差异草稿与脚本设计

## 对照表（核心要素）
| 维度 | Qwen 可视化基线<br>`attention_pruning_case_study_hybrid_rounds_xtrace.py` | LazyEviction SparseRound (Qwen7B)<br>`run_sparse_prefill_keep_sharded_eval.sh` → `lazy_eviction_sparse_evaluation_prefill_keep.py` | R-KV SpeckV (Llama8B)<br>`run_speckv_aime24_official_sampled8.sh` |
| --- | --- | --- | --- |
| 模型/加载 | Qwen3 stats only（离线 qk.pt，可视化模拟）；dtype 默认 float32；无生成。 | `DeepSeek-R1-Distill-Qwen-7B`，`torch_dtype` fp16，`attn_implementation=sdpa`，`use_cache=True`。 | `DeepSeek-R1-Distill-Llama-8B`，`torch_dtype` bfloat16，`attn_implementation=flash_attention_2`，`use_cache=True`。 |
| 数据/Prompt | 仅读 trace；无 prompt。 | AIME 基准，经 `<|im_start|>system/user/assistant<think>` 聊天模板；`temperature=0.0`，`top_p=1.0`，`max_new_tokens` 动态裁剪。 | AIME24 问答 JSON，使用 SpeckV plain prompt（无 chat），`temperature=0.6`，`top_p=0.95`，`max_length=32768`。 |
| Stats/采样 | `stats_trace` qk.pt + `head_sample_file`；`round_window=64` 默认；`max_keys`=2048；offset_max_length=65536；aggregation 默认 mean；seed 可选。 | `sparse_stats_path=distill_qwen7b_qid9001_trace00_stats.pt`；`max_kv_capacity=1492`；`sparse_round_window=decoding_recent_size=363`；aggregation=mean；head_limit=-1；seed=0。 | `sparse_stats_path=...llama8b...stats.pt`；`kv_budget=2048`；`sparse_round_window=window_size=128`；aggregation=mean；head_limit=-1；seed=0；可选 `normalize_scores`（默认关）。 |
| 前向/位置 | 纯模拟：`round_start=q_idx` 绝对位置；RoPE 用 Qwen3 Rotary + `attention_scaling`；保留未来窗口：`keep_capacity=max_keys-upcoming`. | Pruner 绑定 HF 生成循环；`absolute_position` 递增，未覆写 `cache_position`；prefix 全保留；`keep_capacity=max_keys-round_window`；RoPE via `build_rotary`(Qwen3) + `freq_scale_sq`。 | HF generate patch覆写：`position_ids` = pruner 绝对位置，`cache_position` 压紧缓存，忽略旧 `attention_mask`；prefix 全保留；循环 `start_next_round` until round_window；超 budget 再 `ensure_capacity`；RoPE 对齐检查 + rope_style 支持。 |
| 评分/裁剪 | `score_keys_for_round` 无 freq_scale；无归一化；topk 按 head-score max + random tie noise。 | 评分乘 `freq_scale_sq`；无归一化；head union -> topk；噪声 1e-6。 | 同 LazyEviction 但可选标准化（z-score）；验证 stats metadata（prompt/attn/dtype/rope_style/type/kv_budget）。 |
| 生成/采样 | 无生成。 | `do_sample=False`（温度 0）；batch=1；`compute_available_new_tokens` 适配上下文。 | `do_sample=True`（温度 0.6/top_p 0.95）；batch=1；固定 `max_length`。 |

## 关键不一致/风险（需在 Qwen 适配时修正或标警告）
- 模型与 stats：LazyEviction/Qwen 基线 vs R-KV/Llama（stats+RoPE 不兼容）→ 必须切回 Qwen 模型与对应 stats。  
- Prompt 与数据：LazyEviction 使用 Qwen chat + `<think>`，R-KV 用 plain prompt（固定 “You are given a math problem...Final answer: \\boxed{}”）；模板差异会影响统计与推理→ 实现阶段按 R-KV 其它 Qwen 脚本的 Prompt 配置执行，不沿用 LazyEviction 配置，但需记录高风险差异。  
- 超参差异：`kv_budget/round_window`（1492/363 vs 2048/128 vs 2048/64）为核心算法参数，当前不一致；实现先沿用 R-KV 设置，收尾再评估是否需对齐。  
- 位置处理：R-KV SpeckV 通过 `position_ids`/`cache_position` 压紧缓存；LazyEviction HF 路径未覆写 `cache_position`，依赖默认缓存布局→ 需确认两者在 Qwen 上等价，否则需调整。  
- 采样/温度：R-KV SpeckV 走采样 (0.6/0.95) 而 LazyEviction 基线温度 0；属于允许差异但需在对比文档标注。  
- Stats 校验：R-KV SpeckV 强校验 `prompt_template/use_chat_template/rope_style/type/dtype/kv_budget`，LazyEviction 未校验→ 替换 Qwen 版时需提供匹配 metadata 的 stats，否则会直接报错。
- R-KV SpeckV 强制 plain prompt：`rkv_sharded_eval.py` 中禁止 `use_chat_template=True`，与 LazyEviction 的 chat+`<think>` 模板不兼容；若需 chat 必须改代码并重算 stats。  
- AIME24/AIME25 SpeckV Qwen 配置已切到 `flash_attention_2` + `bfloat16`；对应 stats (`R-KV/outputs/sample8_fullkv_aime24_official_qwen/...`、`...aime25...`) 已重新以 flash_attn2/bf16 + plain prompt 生成（2 traces）。

## 运行脚本与校准指引（Qwen SpeckV）
- 模型：`/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B`。  
- 脚本：  
  - AIME24：`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh` → `sample8_speckv_aime24_official_qwen.yaml`（flash_attn2 + bf16, plain prompt）。  
  - AIME25：`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime25_official_sampled8_qwen.sh` → `sample8_speckv_aime25_official_qwen.yaml`（flash_attn2 + bf16, plain prompt）。  
- 校准步骤（示例，需在 `conda run -n rkv` 环境下、指定空闲 GPU）：  
  - AIME24 stats：`env CUDA_VISIBLE_DEVICES=3 conda run -n rkv python R-KV/weian_development/rkv_sparse_round_calibrate.py --trace-root R-KV/outputs/sample8_fullkv_aime24_official_qwen --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B --output-path R-KV/outputs/sample8_fullkv_aime24_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt --attn-implementation flash_attention_2 --dtype bfloat16 --kv-budget 2048 --num-traces 2`  
  - AIME25 stats：`env CUDA_VISIBLE_DEVICES=4 conda run -n rkv python R-KV/weian_development/rkv_sparse_round_calibrate.py --trace-root R-KV/outputs/sample8_fullkv_aime25_official_qwen --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B --output-path R-KV/outputs/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt --attn-implementation flash_attention_2 --dtype bfloat16 --kv-budget 2048 --num-traces 2`  
- 运行/校准均需保持 plain prompt（`use_chat_template=False`）；切换 attn/dtype 后必须重算对应 stats。
## 必须一致 vs 允许差异（用于后续自检）
- **必须一致**：  
  - 模型/Tokenizer = Qwen（含 rope_scaling/attention_scaling/rope_style/type）与生成该 stats 的模型完全匹配。  
  - Stats 文件：与运行时 prompt_template/use_chat_template/system_prompt/attn_impl/dtype/kv_budget 对齐；RoPE 元数据（rope_style/type）一致。  
  - kv_budget & round_window/keep_capacity 逻辑：round_window 用于 keep_capacity=max_keys-round_window，超限再 enforce；prefix 全保留。  
  - 位置/RoPE 处理：绝对 position_ids 用于旋转，cache_position 紧致写入（按 R-KV patch 方式）；pruner absolute_position 同步递增。  
  - Pruner触发/评分：round_boundary 判定、head union + topk、freq_scale_sq 加权、一致的噪声注入；head 映射（kv_head vs attn_head）与 sampled_heads 一致。  
  - Prompt/数据源：采用 R-KV Qwen 脚本的 plain prompt 语料（非 LazyEviction chat），模板与 stats 同源；数据基准保持 AIME24/25 JSONL。  
  - 模型加载/attn 实现：与 stats 记录一致的 attn_implementation、dtype（bf16/fp16），use_cache=True，batch=1。  
  - 采样集：sampled_heads 与 stats 文件一致，head_limit 仅可截断前缀。
- **允许差异（需记录）**：  
  - 采样温度/top_p（默认 R-KV 0.6/0.95 vs LazyEviction 0）。  
  - logprob 存储开关、输出/日志目录、GPU 列表与 shard 数。  
  - head_limit 调优、score 归一化（normalize_scores 开关）。  
  - 数据抽样范围（max_examples/max_samples）若不影响 prompt 模板。

## 脚本设计草稿
- **SpecKV for `run_rkv_aime25_official_sampled8_qwen.sh`（骨架不变）**  
  - 切换 method→`speckv` 并补齐 SpeckV 配置：`kv_budget`/`window_size` 与 LazyEviction 基线对齐（候选：1492/363 或 2048/128? 需确认目标），新增 `sparse_stats_path` 指向 Qwen stats（`distill_qwen7b_qid9001_trace00_stats.pt`），`sparse_round_window`、`sparse_offset_max_length`、`sparse_score_aggregation`、`sparse_head_limit`、`sparse_seed`。  
  - 保持模型/attn/dtype 跟 Qwen 基线一致：`model_path`/`tokenizer` = DeepSeek-R1-Distill-Qwen-7B，`attn_implementation`、dtype 与 stats metadata 相符；若保留 flash_attn2 需确认 stats 亦为 flash_attn2/bf16，否则改 sdpa/fp16。  
  - Prompt/数据：沿用现有 dataset 路径（AIME25 JSONL），但需决定是否切换到 LazyEviction chat 模板；若保持 plain，需要在差异文档注明不可比。  
  - 生成参数：保留现有温度/Top-p 或改为与基线一致（温度 0）；在设计稿中标注为“允许差异”项。  
  - 运行器侧：`rkv_sharded_eval` SpeckV 分支已经支持 Qwen（position_ids/cache_position），需验证 stats metadata 满足校验；必要时补 `use_chat_template=false` 等 flag。

- **SpecKV Qwen 版 for `run_speckv_aime24_official_sampled8.sh`（模型从 Llama→Qwen）**  
  - 模型/Tokenizer：改为 `DeepSeek-R1-Distill-Qwen-7B`（或 Qwen3-8B if target），更新 `load_dtype`/`attn_implementation` 与 stats 匹配；替换 `sparse_stats_path` 为 Qwen 统计（现有 distill_qwen7b_*），并确保 metadata 校验项（prompt_template/use_chat_template/rope_style/type/dtype/kv_budget）一致。  
  - Prompt/数据：若仍跑 AIME24，需决定模板（plain vs chat）；若使用 chat，需调整 prompt 构造（当前 SpeckV 强制 plain）并考虑 stats 兼容性；若保留 plain，需要在对比中说明与 LazyEviction 基线存在模板差异。  
  - 超参：调整 `kv_budget`/`sparse_round_window`/`window_size` 至与 Qwen 基线匹配（目前 2048/128 源自 Llama 版）；确认 `max_length` 与 Qwen 上下文限制。  
- 其他：保留 sharding/seed/num_samples 结构；评估是否开启 `sparse_normalize_scores`（默认 False）以保持与 LazyEviction 对齐；更新日志/输出目录避免覆盖 Llama 结果。***

## 脚本设计细化（当前执行）
- `run_rkv_aime25_official_sampled8_qwen.sh` → SpecKV 方案  
  - 新建 SpeckV 配置（`sample8_speckv_aime25_official_qwen.yaml`，日志/输出目录加 `_speckv_qwen` 区分），脚本继续调用 `rkv_sharded_dispatch.py`。  
  - `runner_args`：`method=speckv`，沿用 `kv_budget=2048`、`window_size=128`（标注与 LazyEviction 1492/363 的差异待后评估）；`attn_implementation=sdpa` + `load_dtype=float16`（对齐 LazyEviction，避开 flash_attn2 缺失）。  
  - SpeckV 必填：`sparse_stats_path` 指向 Qwen plain-prompt stats（已用 sdpa+fp16 生成，占位 `R-KV/outputs/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt`）；`sparse_round_window=128`（若与 window 不同需说明）、`sparse_offset_max_length=65536`、`sparse_score_aggregation=mean`、`sparse_head_limit=-1`、`sparse_seed=0`、`sparse_normalize_scores=False`。  
  - 数据/Prompt：保持 AIME25 JSONL（`/data/rbg/users/weian/project/rl/dc/aime25.jsonl`），SpeckV 强制 `use_chat_template=False`、`chat_system_prompt=""`；温度/Top-p 先沿用 0.6/0.95（允许差异，与 LazyEviction 温度 0 需在对比文档注明）。  
  - 其他：保持 `reset_cache_each_batch=false`、`fp32_topk=false`、`num_samples=8`，确认路径不覆盖 Llama 或 R-KV rkv 版本输出。
- `run_speckv_aime24_official_sampled8.sh` → Qwen 切换方案  
  - 新建 Qwen SpeckV YAML（`sample8_speckv_aime24_official_qwen.yaml`，日志/输出目录加 `_speckv_aime24_qwen`），脚本改指向新 YAML，原 Llama 配置移至 archive 或保留备查。  
  - `runner_args`：`model_path`/`tokenizer` 切换至 DeepSeek-R1-Distill-Qwen-7B，`method=speckv` 保持；`kv_budget/window_size` 先沿用 2048/128（与 LazyEviction 1492/363 差异需记录），`attn_implementation=sdpa` + `load_dtype=float16`（与 LazyEviction 一致，flash_attn2 缺失时可运行）。  
  - SpeckV 字段：`sparse_stats_path` 指向 Qwen plain stats（已用 sdpa+fp16 生成，占位 `R-KV/outputs/sample8_fullkv_aime24_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt`），`sparse_round_window` 默认同 window，`sparse_offset_max_length=65536`，`sparse_score_aggregation=mean`，`sparse_head_limit=-1`，`sparse_seed=0`，`sparse_normalize_scores=False`。  
  - 数据/Prompt：保持 AIME24 JSONL 与 SpeckV plain prompt（`use_chat_template=False`，`chat_system_prompt=""`），温度/Top-p 沿用 0.6/0.95（允许差异），输出/日志路径避免覆盖 Llama 结果。

## 核心逻辑与 Prompt 复核（当前执行）
- SpeckV 生成绑定：`apply_speckv_generate_patch` 通过绝对 `position_ids` + 紧致 `cache_position` 写入缓存，前缀全保留；仅在超出 `max_keys - round_window` 时 prune，匹配“保持未来窗口后再 enforce”思路。  
- RoPE/模型校验：pruner 初始化用 `AutoConfig` 推断 `rope_style/type`，`validate_stats_metadata` 硬校验 prompt_template/use_chat_template/system_prompt/attn_impl/dtype/kv_budget/rope_style/type；`verify_rotary_alignment` 强制模型 RoPE 与 stats 内部旋转一致（Qwen/Llama 不匹配会直接报错）。  
- 评分细节：`SparseRoundPruner` 使用 freq_scale_sq 加权、head union + topk（含 1e-6 噪声），支持 `normalize_scores` 但默认关闭；kv_head 映射考虑 num_key_value_groups，兼容 Qwen/Llama 不同 kv 头数。  
- Prompt/数据：`rkv_sharded_eval` 在 method=speckv 时强制 `use_chat_template=False`，加载数据时也用 plain prompt（`PROMPT_TEMPLATE`）；若要改 chat 模板需改代码并重算 stats，当前保持 plain 并在差异文档标注与 LazyEviction chat 的不一致。

## 校准/统计生成备注
- R-KV Llama 版使用 `R-KV/weian_development/rkv_sparse_round_calibrate.py` 生成 stats（固定 plain prompt、禁止 chat、记录 rope_style/type/kv_budget/attn_impl/dtype）。  
- 适配 Qwen 时需在该脚本基础上改用 Qwen 模型/Tokenizer、Qwen 数据/Prompt 模板，保持 plain prompt 与运行时一致；确保 attn_impl/dtype 与目标脚本配置一致，以生成可用的 Qwen stats。  
- 不可直接复用 Llama stats 到 Qwen（或反之），因为 inv_freq/attention_scaling/rope_style 不同会导致频谱不匹配、评分失真。

## 待完成后集中复核的高风险点（需明确结论）
- Prompt 模板差异（高风险）：R-KV SpeckV 使用 plain prompt（`weian_development/speckv/prompt_utils.py`，固定 “You are given a math problem... Final answer: \\boxed{}”），LazyEviction 基线使用 Qwen chat 模板 `<|im_start|>system/user/assistant<think>`（带系统提示、角色标签、think 标签）。本次实现按 R-KV Qwen 脚本惯例走 plain prompt，收尾需确认是否需要统一或重新生成匹配模板的 stats。  
- Pruner 绑定方式（高风险）：LazyEviction 在 HF 生成循环内嵌 pruner（未覆写 `cache_position`），R-KV SpeckV 采用 HF generate patch 覆写 `position_ids`/`cache_position` 压紧缓存。本次按 patch 方案实现，后续需确认两者在 Qwen 上的等价性（位置/缓存布局）。  
- RoPE/Stats 兼容性（高风险）：Llama 模型 + Llama stats 的 inv_freq/attention_scaling 与 Qwen 不同，频率表不兼容会导致评分失真；必须使用 Qwen 模型并配套 Qwen stats（且 rope_style/type 对应）。  
- kv_budget/round_window（高风险）：1492/363 vs 2048/128 vs 2048/64 不一致；先沿用 R-KV 2048/128（及 2048/128 系列）落地，收尾再评估是否需对齐 LazyEviction 1492/363。  
- 位置处理（高风险）：R-KV patch 使用紧致 `cache_position`，LazyEviction 默认缓存布局；需最终确认对生成长度与剪枝触发的影响。  
- 采样/温度（需确认）：R-KV SpeckV 用温度 0.6/top_p 0.95，LazyEviction 温度 0；暂不改，收尾时决定是否统一。  
- Prompt/数据源（需确认）：执行时采用 R-KV 其它 Qwen 脚本的配置，不沿用 LazyEviction 配置；需记录模板差异风险以备后续讨论。
