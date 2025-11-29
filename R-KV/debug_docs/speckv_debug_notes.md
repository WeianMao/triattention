## SpeckV 调研记录（避免重复排查）

以下为本轮排查已核实的点、使用的命令及结论，方便后续同事复用。

### 1) 模板/提示对齐性
- **核查路径**：`R-KV/weian_development/rkv_sharded_eval.py` 加载数据后调用 `build_prompt(...)`，实现位于 `R-KV/weian_development/speckv/prompt_utils.py`。
- **逻辑**：
  - `prompt_use_chat` 在 `rkv_sharded_eval.py` 中被硬编码为 `False`（即使传入 `--use_chat_template` 也不生效）。
  - `build_prompt` 在 `use_chat_template=False` 时返回纯文本 `PROMPT_TEMPLATE.format(question=...)`；`True` 时才走 `tokenizer.apply_chat_template`。
  - SpeckV 推理（method=speckv）与基线方法共用同一 `build_prompt`，因此默认都是纯文本模板。
- **校准脚本**：`R-KV/weian_development/rkv_sparse_round_calibrate.py` 使用 `build_prompt_with_response`，且脚本开头直接禁止 chat（`if bool(args.use_chat_template): raise ValueError(...)`）。校准也固定纯文本模板。
- **结论**：基线 / SpeckV 推理 / SpeckV 校准三处模板逻辑对齐，默认纯文本。旧的 merged.jsonl 出现 chat 前缀是历史产物。

### 2) 统计文件与配置匹配
- **检查命令**：
  ```bash
  python -c "import torch; meta=torch.load('R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt',map_location='cpu')['metadata']; print(meta)"
  python -c "import torch; meta=torch.load('R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt',map_location='cpu')['metadata']; print(meta.get('num_traces'), meta.get('trace_root'), meta.get('head_dim'))"
  ```
- **读取结果要点**：`use_chat_template: False`, `system_prompt: ""`, `kv_budget: 2048`, `attn_implementation: flash_attention_2`, `dtype: bfloat16`, `rope_style: half`, `rope_type: llama3`，`num_traces: 3`。
- **用户确认**：stats 已多次重算，num_traces=3 不是当前问题，无需再跑校准。

### 3) RoPE/频率相关实现检查
- 反旋逻辑：`R-KV/weian_development/speckv/round_pruning_utils.py` 中 `determine_rope_style` 对 Llama 返回 `"half"`；`invert_rope` 支持 `style`；`compute_frequency_statistics_from_means`、`score_keys_for_round` 均使用 `freq_scale_sq`。
- 校准：`rkv_sparse_round_calibrate.py` 使用同一工具，`compute_rotary_tables` 返回 `(cos,sin,inv_freq,freq_scale,style)`，生成 stats 时携带 `rope_style`/`rope_type` 元数据。
- 运行期：`apply_speckv_generate_patch` 构建 pruner 时同样使用上述工具，并调用 `verify_rotary_alignment` 检查模型 RoPE 与 pruner 一致（对比 inv_freq）。

### 4) KV 剪枝流程对齐性与潜在 MQA 误配
- **LazyEviction 逻辑**：前缀固定，按 round_window 滚动窗口，在每轮开始时把动态 KV 剪到 `keep_capacity = max_keys - round_window`，按 head 频率评分。
- **SpeckV 运行期**：`apply_speckv_generate_patch` 中的 `SparseRoundPruner`：
  - 前缀通过 `attach_initial_cache` 保持不动，动态部分计入 `tokens_in_round`，`start_next_round` 同样按 `keep_capacity=max_keys-round_window` 剪动态 KV。
  - 超预算时再调用 `ensure_capacity`，也是仅动动态区。
  - head/kv head 映射：若存在 kv_group，head 映射为 `head // num_key_value_groups`。
- **重要疑点（MQA 映射不一致）**：DeepSeek-R1-Llama-8B 配置 `num_attention_heads=32`, `num_key_value_heads=8`。stats 采样与 q_mean 统计基于 *attention head*（0..31），运行期在 `_compute_head_scores` 中将 head 映射到 KV 组 `kv_head = head // num_key_value_groups`，再用 KV 头的 K 与“attention head”统计配对。MQA 下可能导致统计与实际 KV 粒度不一致，剪枝打分失真。
  - **冒烟验证**（确认映射错位风险）：
    ```bash
    python -c "import sys,torch;from pathlib import Path;sys.path.insert(0,str(Path('R-KV').resolve()));from weian_development.speckv.sparse_round_pruner_prefill_keep import SparsePruningConfig,SparseRoundPruner;from transformers import AutoConfig;model=Path('/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B');stats=Path('R-KV/outputs/sample8_fullkv_aime24_official/stats/deepseek_r1_llama8b_chat_stats.pt');cfg=AutoConfig.from_pretrained(model,trust_remote_code=True);pr=SparseRoundPruner(SparsePruningConfig(stats_path=stats,model_path=model,device=torch.device('cpu'),dtype=torch.float32,max_keys=2048,round_window=128,offset_max_length=1024,score_aggregation='mean'));kv_groups=pr.num_key_value_groups or 1;bucket={};\\nfor layer,head in pr.sampled_heads:\\n kv=head//kv_groups; bucket.setdefault(kv,[]).append((layer,head));\\nprint('model heads',cfg.num_attention_heads,'kv',cfg.num_key_value_heads);print('sampled_heads',len(pr.sampled_heads),'kv_heads',pr.num_key_value_heads,'kv_groups',kv_groups);\\nfor kv,heads in sorted(bucket.items()):\\n print(f'kv_head {kv}: {len(heads)} stats, examples {heads[:4]}')\" 
    ```
    输出：`kv_head 0: 9 stats; kv_head 1:16; kv_head 2:10; kv_head 3:14; kv_head 4:11; kv_head 5:17; kv_head 6:12; kv_head 7:11`。多个 attention 头共用同一 KV 头但使用各自统计，存在粒度错位风险（KV=head 的模型上不会暴露）。

### 5) 已确认的“不是问题”的点
- 旧的 merged.jsonl 带 chat 前缀是历史产物，与当前代码不符。
- 模板、chat 开关、prompt_template 在运行/校准/基线三处一致。
- stats 元数据与当前 SpeckV 配置字段匹配（kv_budget、attn_impl、dtype、chat 开关等）；用户已确认 stats 曾多次重算，num_traces=3 不是当前问题，无需再重跑校准脚本。

### 后续待验证方向（优先级）
- **MQA 统计/运行期对齐**：构造最小复现，打印 pruner 内 head→kv_head 映射与所用 stats 头索引，确认是否用到了跨组统计；若成立，需要将统计与运行期一致地按 KV 组重采样/重映射。
- **flash_attn2 兼容性**：单步运行 flash_attn2 + speckv，检查 cache_position/position_ids 与输出长度是否一致，并确认剪枝前后 past_key_values 长度变化符合预期（防止剪枝被跳过或长度错位）。
- **剪枝触发性**：在 decode 循环中观察动态 cache 长度和 tokens_in_round，确认 round_window 触发、start_next_round 和 ensure_capacity 都被执行（排除逻辑被绕过）。

> 以上为已检查项及结论，如需复现检查可直接按命令重跑。
