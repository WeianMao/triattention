# FullKV（R-KV vs LazyEviction）差异梳理与消融计划

> 该文档完整吸纳自 `R-KV/docs/rkv_fullkv_vs_lazy_ablation_plan.md`，确保无信息遗漏。LazyEviction 作为参照，**只改 R-KV 侧**；所有改动需 cfg 开关（默认旧行为），必要时代码隔离。长跑由用户执行，Agent 仅做短 smoke。

> 默认约定：每个消融方法对应 R-KV 里的一个可配置选项（cfg/flag），**默认关闭 = 原行为**；开启后将该差异点对齐 LazyEviction 的实现。未经特别说明，均按此约定设计与实现。

## 关键实现差异（证据位）
- **数据/Prompt**  
  - R-KV：`aime25.jsonl`，字段 `{question, answer}`，plain prompt（无 chat、无 `<think>`），`use_chat_template=False`。  
  - LazyEviction：`datasets/aime/test.jsonl`，`messages`+`cot`，Qwen chat 模板（system+user+assistant+`<think>`）。
- **解码策略**  
  - R-KV：`do_sample=True`，`temperature=0.6`，`top_p=0.95`，每题 8 draws，`eval_math_multi.py` 取 pass@1 均值。  
  - LazyEviction：`do_sample=False`（温度 0），单次输出，字符串解析评估。
- **长度/窗口**  
  - R-KV：`max_length=32768`（prompt+生成总长）。  
  - LazyEviction：`max_new_tokens=16384`（生成长度），prompt 未限。
- **注意力/精度**  
  - R-KV：`attn_implementation=flash_attention_2`，`torch_dtype=bfloat16`。  
  - LazyEviction：`attn_implementation=sdpa`，`torch_dtype=float16`。
- **评估口径**  
  - R-KV：多样本 pass@1 平均（`eval_math_multi.py`）。  
  - LazyEviction：单样本准确率。

## 消融优先级与操作（默认旧逻辑，需 cfg 控制）
1. **数据/Prompt 对齐（最高优先）**  
   - 动作：cfg 允许切换到 LazyEviction 数据+chat prompt（system+user+`<think>`）；默认保持 plain + AIME25.jsonl。  
   - 目的：验证模板/数据差异是否主因。
2. **解码方式：采样 vs 贪心**  
   - 动作：cfg `do_sample` 开关（默认 True）；若 False，温度=0/top_p=1。  
   - 目的：排除温度采样导致的波动。
3. **多样本聚合策略**  
   - 动作：cfg `num_samples`（可设 1）或 `eval_mode`（avg / best-of / 投票）；默认 8 + pass@1 均值。  
   - 目的：检查 8-draw 平均是否稀释精度。
4. **Attn/DType 对齐**  
   - 动作：cfg `attn_implementation`/`load_dtype` 可切 `sdpa`+fp16，默认 flash-attn2+bf16。  
   - 目的：排除内核/精度差异。
5. **长度/截断**  
   - 动作：cfg 提供 `max_new_tokens` 或放宽 `max_length`；默认 32768。记录截断率。  
   - 目的：确保生成未被过早截断。
6. **信任远程代码 / tokenizer 设置**  
   - 动作：cfg `trust_remote_code` 等，默认旧值；一次性排查兼容性。  
   - 目的：排除加载差异。
7. **Seed 与缓存复位**  
   - 动作：cfg `seed`（如对齐 42）、`reset_cache_each_batch`（默认 False）。  
   - 目的：提高稳定性/可重复性。

## 建议实验顺序
1) 数据 + Lazy chat prompt 对齐（保留其他默认）。  
2) 在 (1) 基础切贪心解码（`do_sample=False`）。  
3) 在 (2) 基础设 `num_samples=1`（或改评估聚合）。  
4) 在 (3) 基础对齐 `sdpa`+fp16。  
5) 检查长度截断，必要时调整 `max_length` / `max_new_tokens`。  
6) 若差距仍大，补充 trust_remote_code / seed / reset_cache 等次要项。

## 继承/安全提醒
- 消融脚本集中在 `R-KV/weian_script/ablations_fullkv/`，不改旧脚本。  
- 每个 cfg 默认为旧行为；未配置时现有实验不受影响。  
- 长跑由用户执行；Agent 仅做短 smoke。  
- 如需改动核心文件，优先隔离或确保默认路径不变；发现关键不可比差异须先告警。
