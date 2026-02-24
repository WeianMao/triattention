# V2 HF 对齐审计清单（进行中）

- 开始时间：2026-02-24
- 目的：系统性检查当前 V2 实现与 HF 参考实现（重点 `per_head` / `per_layer_per_head`）在算法语义上是否等价，并区分：
  - 真正 bug
  - 合理实现差异（理论等价）
  - 可能导致结果差异但不一定是 bug 的系统/后端差异
- 约束：本轮先做审计与证据整理，默认不改代码（除非后续确认明显 bug）

## 结论摘要（当前阶段）

1. **性能问题已解决**：decode 热路径利用率恢复到接近原始 vLLM（实测 full run 活跃 GPU 常见 `98-99%`）。
2. **`per_layer strict` 这条历史线的结果已回到正常区间**：本轮 full-run `acc=42.9`，从经验上看与历史参考值接近。
3. **仍不能据此宣称“目标模式 HF 对齐已完成”**：
   - 本轮 full-run 配置是 `per_layer`（历史 strict 配置）；
   - 用户最高优先级目标仍是 `per_head` / `per_layer_per_head`；
   - `compare_results.py` 报告可用于观察差异，但不是采样场景下的最终“算法对齐证明”。

## 审计原则（本次采用）

1. 先核对“算法语义层”是否等价（选择逻辑、保留集合、布局与写回语义）。
2. 再核对“运行时语义层”是否等价（position/effective length/slot mapping）。
3. 再核对“实验配置与生成口径”是否一致（避免把采样差异误判为算法 bug）。
4. 允许实现不同，只要理论语义等价。
5. 用户已明确可接受的小差异（暂不作为阻塞）：
   - prefill 保护边界的小 off-by-one；
   - recent window 127/128/129 的小范围差别；
   - 时序不同但数学等价/算法等价。

## 用户后续补充约束（2026-02-24，已记录）

1. `per_layer` **不是目标模式**；后续一旦用户路径/配置触发 `per_layer`，应改为**显式报错**，不能继续执行（避免误用）。
2. 当前 HF 对齐参照以脚本为准：
   - `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`
3. 对齐范围只看该脚本实际使用到的参数/语义：
   - 脚本里没有用到的历史 ablation 参数（如失败实验遗留开关）本轮可忽略；
   - 脚本里有的参数/语义必须对齐。
4. 与该脚本唯一预期差异是后续评测支持额外数据集（AIME25），该项属于评测/调度层，不属于本轮推理算法对齐核心。
5. `disable_mlr` / `disable_trig` / `disable_top_n_high_freq` 不属于当前 `norm_aligned_perhead` 参照脚本的对齐范围（先忽略）。

## 本轮开发修复（2026-02-24，已落地）

1. `per_layer` 模式门禁：
   - V2 默认 `pruning_mode` 已切到 `per_head`；
   - 若显式设置 `per_layer` 且未开启 `allow_per_layer_mode=True`，运行时 selector 构建会直接报错；
   - 同时保留 `allow_per_layer_mode` 显式放行接口，便于后续经批准后继续使用 `per_layer`。
2. `per_layer_per_head` GQA 聚合语义修复：
   - 在 `stats_heads != runtime_heads` 时，`per_layer_per_head` 不再走“先缩并统计再打分”的路径；
   - 改为与 HF 语义一致的方向：先按 attention-head 打分，再按 KV-head 分组聚合。
3. 配置接口预留：
   - 新增 `layer_perhead_aggregation` / `per_layer_aggregation` / `allow_per_layer_mode` 接口；
   - 当前默认实验仍以 `per_head` 为主，不要求立刻跑 `per_layer_per_head` 实验。
4. 实验入口（runner CLI/env）已同步暴露上述接口，便于后续直接配置。

## 审计清单（总表）

状态说明：
- `DONE`：已完成审计并有证据
- `PARTIAL`：已完成部分核查，仍需补证据
- `PENDING`：尚未审计
- `N/A`：与当前问题无关或不影响算法等价

### A. 目标与口径确认

- [x] `DONE` 明确当前最高优先级目标模式是 `per_head` / `per_layer_per_head`（`per_layer` 非最终目标）
- [x] `DONE` 明确本轮 full-run 使用的是 `per_layer strict` 历史配置，仅能作为性能+历史线 sanity check
- [x] `DONE` 明确 `compare_results.py` 在采样场景下不是最终“算法对齐证明”口径

### B. 实验配置对齐（HF vs vLLM）

- [x] `DONE` 核对本轮 full strict v2 配置（`triattention_v2_aime24_hf_strict.yaml`）关键生成参数
- [x] `DONE` 核对 HF 参考 sample8 qwen 配置（`sample8_speckv_aime24_official_qwen.yaml`）关键生成参数
- [ ] `PENDING` 做逐项参数差异表（区分“应对齐项” vs “允许差异项”；当前已确认本轮 `per_layer strict` 的生成/评测口径与 HF sample8 参考线为同量级对照，但仍缺完整参数表）
- [ ] `PENDING` 为目标模式（`per_head` / `per_layer_per_head`）建立对应的 HF 参考配置映射表

### C. 选择逻辑（HF 语义层）

- [x] `DONE` 审核 V2 selector 支持 `per_head` / `per_layer_per_head`
- [x] `DONE` 审核 `per_head` 的 `hf_aligned_global_per_head` 语义入口是否存在并默认可配
- [x] `DONE` 审核 `per_layer_per_head` 不会错误走“跨层全局 per_head 聚合”路径（已有专门单测）
- [x] `DONE` 跑 selector/planner/layout 相关定点测试（通过）
- [x] `DONE` 跑 `hf_aligned_global_per_head` 关键 selector 单测（dense/paged 对齐，通过）
- [ ] `PENDING` 人工逐项核对 HF 参考代码中 top-k 聚合/tie 语义与 V2 `selector_hf.py` 的一致性（源码级）

### C1. 打分函数等价性（严重差异项，单列）

- [x] `DONE` 核对 HF 参考打分公式入口：`R-KV/weian_development/speckv/round_pruning_utils.py::score_keys_for_round`
- [x] `DONE` 核对 V2 实际打分入口：`TriAttention_vLLM/triattention_v2/selector_hf.py` -> `TriAttention_vLLM/triattention/scoring.py::compute_scores_triton`
- [x] `DONE` 确认当前 V2 Triton 打分路径对 `disable_trig` / `disable_top_n_high_freq` 不具备 HF 等价支持（见“已确认差异”）
- [x] `DONE` 确认 `selector_hf` 直接调用 `compute_scores_triton`，在 `disable_mlr=True` 下不会走 `compute_scores()` 的 PyTorch fallback（见“已确认差异”）
- [x] `DONE` 确认 TriAttention PyTorch fallback 的 `disable_trig` 语义与 HF 不等价（见“已确认差异”）
- [ ] `PENDING` 对默认开关（`disable_trig=false`, `disable_top_n_high_freq=0`, `disable_mlr=false`）补一组跨实现数值 spot-check（非必要但可增强信心）

### D. 布局与压缩执行（实现不同但应等价）

- [x] `DONE` 确认 V2 使用 low-move `fill-hole` 主线（不要求物理保序）
- [x] `DONE` 确认“物理不保序”原则上不改变注意力数学（只要 keep 集合与 K/V 配对正确）
- [x] `DONE` 相关布局/压缩单测已通过（`test_layout_engine.py`, `test_kv_compaction.py`）
- [ ] `PENDING` 针对 `per_head` / `per_layer_per_head` 的端到端 keep 集合一致性 spot-check（真实运行路径）

### E. 运行时语义适配（position / effective len / slot）

- [x] `DONE` 已加 request-local 状态语义标记并收敛 runtime override 主逻辑（此前代码收敛已完成）
- [x] `DONE` 已加多层 fail-fast 护栏，防止 A/B 槽位错改静默发生
- [x] `DONE` 相关定点测试通过（`input_patch_ops`, `input_patch_vllm_backend`, `effective_overrides`, `input_adapter`）
- [ ] `PENDING` 在目标模式真实运行路径下做端到端验证（确认护栏不触发且结果稳定）
- [ ] `PENDING` 用证据确认/否定“历史 HF 不对齐是否由地址映射错位导致”

### F. 生成后端差异（不一定是 bug）

- [x] `DONE` 确认 HF 与 vLLM 都是采样生成路径，本身会产生输出分布差异
- [x] `DONE` 确认采样输出 token-level 低匹配率不能单独作为算法不对齐证据
- [ ] `PENDING` 统一口径评测对照（同一 evaluator、同一 config 映射）并整理“可能解释差异”的非 bug 因素清单

### G. 目标模式端到端验收（最高优先级）

- [ ] `PENDING` `per_head`：端到端对齐核验（至少 1 个 anchor 配置）
- [ ] `PENDING` `per_layer_per_head`：端到端对齐核验（至少 1 个 anchor 配置）
- [ ] `PENDING` 若发现差异：分类为 bug / 等价差异 / 后端采样差异，并形成结论

## 已确认差异（需要报告）

以下差异不属于“实现形式不同但算法等价”的范畴，需要明确记录。

### 1) 当前 full strict 跑的并不是目标模式验收（`per_layer` vs 用户目标 `per_head/per_layer_per_head`）

- 本轮 full-run 使用配置：`TriAttention_vLLM/evaluation/dispatch/configs/triattention_v2_aime24_hf_strict.yaml`
- 其中 `pruning_mode: per_layer`（历史 strict 配置）
- 因此它可以作为：
  - 性能恢复验证（已通过）
  - 历史 `per_layer` 路线 sanity check（已通过）
- 但不能替代用户最高优先级目标（`per_head` / `per_layer_per_head`）的最终对齐验收。

结论：
- **不是 bug**，但属于“验收覆盖不足”，必须单独补目标模式验证。

### 2) TriAttention Triton 打分路径对部分 HF 打分开关不等价（严重差异）

证据：
- HF 参考打分函数支持并显式实现：
  - `disable_top_n_high_freq`
  - `disable_trig`
  见 `R-KV/weian_development/speckv/round_pruning_utils.py:277`，尤其 `:307-318`
- V2 实际调用的 Triton 打分包装只传入：
  - `aggregation`
  - `disable_mlr`
  未传递 `disable_trig` / `disable_top_n_high_freq`
  见 `TriAttention_vLLM/triattention/scoring.py:178-193`

影响：
- 当 `disable_trig=True` 或 `disable_top_n_high_freq>0` 时，V2 Triton 路径与 HF 参考**不等价**。
- 当前默认配置通常为 `disable_trig=false`, `disable_top_n_high_freq=0`，所以在默认线下**不一定影响结果**。

结论：
- **这是严重差异（功能/语义缺口）**。
- 对当前默认配置可能不构成问题；对相关 ablation/对齐实验会构成问题。

### 3) TriAttention PyTorch fallback 的 `disable_trig` 语义与 HF 参考不一致（严重差异）

证据：
- HF 参考：`disable_trig=True` 时使用 `combined = additive`（完全去掉 position-dependent base term）
  - `R-KV/weian_development/speckv/round_pruning_utils.py:318`
- TriAttention PyTorch fallback：`disable_trig=True` 时改成“去掉 cos，但保留 magnitude position term”
  - `TriAttention_vLLM/triattention/scoring.py:326-330`
  - 随后仍会再加 `extra_term`（`:344-357`）

影响：
- 这是**公式级不等价**，不是实现细节差异。
- 若未来进入 PyTorch fallback 且启用 `disable_trig=True`，会与 HF 行为偏离。
- 同时，TriAttention PyTorch fallback 中也未见 HF `disable_top_n_high_freq` 的对应实现分支（HF 参考见 `round_pruning_utils.py:307-313`）。

结论：
- **严重差异（bug/公式不一致）**，但仅在相关开关与 fallback 条件下触发。

### 3.1) `selector_hf` 在 `disable_mlr=True` 时无法使用 HF 等价 fallback（严重差异 / 能力缺口）

证据：
- `TriAttention` 通用打分入口 `compute_scores()` 在 `disable_mlr=True` 时会改走 PyTorch fallback
  - `TriAttention_vLLM/triattention/scoring.py:54`
- 但 V2 `selector_hf` 实际直接调用的是 `compute_scores_triton(...)`（绕过 `compute_scores()`）
  - `TriAttention_vLLM/triattention_v2/selector_hf.py`（`_compute_layer_scores_raw()`）
- Triton kernel wrapper 明确声明 `disable_mlr=True` 不支持，并抛异常
  - `TriAttention_vLLM/triattention/kernels/triton_scoring.py:543-546`

影响：
- 当用户尝试用 HF 的 `disable_mlr=True` ablation 时，V2 `selector_hf` 不会得到 HF 等价行为；
- 更可能出现直接报错（而不是自动回退到 PyTorch 等价实现）。

结论：
- **严重差异（能力缺口）**，默认配置 `disable_mlr=false` 时不影响当前主线结果。

### 4) `per_layer_per_head` 路径存在潜在不等价风险：V2 在非 `per_head:hf_aligned_global_per_head` 时会先做 GQA 统计均值缩并

证据（V2）：
- `_resolve_layer_score_inputs()` 中仅对 `requested_pruning_mode == "per_head" and per_head_semantics == "hf_aligned_global_per_head"` 启用 `use_hf_group_max`
  - `TriAttention_vLLM/triattention_v2/selector_hf.py:229-233`
- 否则当 `stats_heads != runtime_heads` 时，会调用 `_build_reduced_layer_stats()` 先把 q 统计和 `freq_scale_sq` 按组求均值缩并到 `runtime_heads`
  - `TriAttention_vLLM/triattention_v2/selector_hf.py:243-247`
  - `TriAttention_vLLM/triattention_v2/selector_hf.py:149-179`

证据（HF 参考）：
- HF `per_layer_per_head` 逻辑是先按采样 attention heads 打分，再按 KV head 分组做 `max/mean` 聚合后 top-k
  - `R-KV/weian_development/speckv/speckv_rkv_style.py:548-572`

风险判断：
- “先平均 stats 再打分” 与 “先打分再按 head 聚合”一般情况下**不等价**。
- 是否实际触发该分支，取决于 V2 selector 里 `stats_heads` 与 `runtime_heads` 的关系（通常在 GQA 模型上很可能触发）。
- 当前目标模型 `DeepSeek-R1-Distill-Qwen-7B` 为 GQA（`num_attention_heads=28`, `num_key_value_heads=4`），因此该风险在目标模式上具备现实触发条件。

结论：
- **高优先级风险项（很可能是实质不等价）**。
- 需要用目标模式（`per_layer_per_head`）做端到端或定点数值对照确认。

状态更新（2026-02-24）：
- 该风险已完成代码级修复（V2 selector 在 `per_layer_per_head` 下改走“先打分再按组聚合”的路径）。
- 仍需后续 `per_layer_per_head` 端到端实验做最终验收（当前按用户要求先不跑该模式实验）。

### 5) V2 当前未暴露 HF `per_layer_per_head` / `per_layer` 的聚合方式开关（`max/mean/...`）

证据：
- HF 参考 `SpeckVRKVStyleConfig` 支持：
  - `layer_perhead_aggregation`
  - `per_layer_aggregation`
  - 且有多个取值（如 `max`, `mean`, 部分路径还有 `pure_mean`）
  - `R-KV/weian_development/speckv/speckv_rkv_style.py:54-61`
- V2 配置目前仅有 `sparse_score_aggregation`（offset 聚合维度），未见独立 `layer_perhead_aggregation/per_layer_aggregation` 配置
  - `TriAttention_vLLM/triattention_v2/config.py:49`

影响：
- 当 HF 参考实验使用了非默认层内/层头聚合方式时，V2 可能无法表达同一语义。

结论：
- **能力差异（可能导致不对齐）**，是否构成当前问题取决于所对齐的 HF 配置是否使用这些开关。

## 本轮已完成审计证据（2026-02-24）

### 1. Full strict（`per_layer`）全量跑通 + 性能恢复

- 运行目录：
  - `TriAttention_vLLM/evaluation/outputs/v2_strict_monkeypatch_probe/strict_monkeypatch_probe_20260224_004238`
- 结果：
  - 8 shards 全部完成（`240` records merged）
  - `eval_math_multi` 输出 `acc=42.9`
- 观察：
  - 推理期间活跃 GPU 长时间 `98-99%`
  - 说明此前“GPU 打不满”的主因（wrapper/subclass 接入）已修复

### 2. `compare_results.py` 报告已生成，但仅作差异观察

- 报告：
  - `TriAttention_vLLM/evaluation/outputs/v2_strict_monkeypatch_probe/strict_monkeypatch_probe_20260224_004238/hf_compare_report_vs_speckv_qwen.txt`
- 说明：
  - 该报告反映采样输出差异与 strict grader 统计，不等价于“算法语义层是否完全对齐”的最终证明。

### 2.1 同口径 HF baseline 评测（补充）

- 使用同一个 `eval_math_multi.py` 对 HF baseline `sample8_speckv_aime24_official_qwen` 复评：
  - `acc=41.2`
- 对比本轮 V2 `per_layer strict`：
  - `acc=42.9`

结论：
- 这支持“当前 V2 `per_layer strict` 结果已回到正常区间，并可能略高于该 HF 参考线”的判断。
- 但不改变“目标模式端到端验收（`per_head` / `per_layer_per_head`）仍待完成”的结论。

### 2.2 评测打分函数口径（与用户示例相关）

- 当前对 `42.9` 的判断基于 `eval_math_multi.py` 输出（`default-default_math_multi_eval*.json*`）。
- 同样的 evaluator 已用于对 HF baseline 输出做复评（得到 `41.2`）。

结论：
- **目前没有证据表明“评测打分函数不一致”是导致差异的原因**。
- 结果差异更可能来自生成/采样轨迹或算法实现差异，而非 evaluator 口径不一致。

### 3. 语义层关键单测（与 HF 对齐相关）已补跑

- 运行命令（`PYTHONPATH=TriAttention_vLLM`, `conda env=trivllm`）：
  - selector/planner/layout 组合：`8 passed`
  - `hf_aligned_global_per_head` 关键 selector 用例：`3 passed`
- 结论：
  - 至少在定点输入上，V2 语义选择与布局关键路径行为与预期一致；
  - 仍需补目标模式端到端实验作为最终证据。

## 已知“可能导致结果更好/不同，但不一定是 bug”的候选原因（待进一步归因）

1. HF 与 vLLM 采样后端实现差异（即使参数名一致）
2. 不同内核/后端数值路径导致的采样轨迹差异（FlashAttention / vLLM runtime）
3. 一些实现细节虽不完全同形，但在数学上等价（例如布局与写回顺序）

## 明确暂不判 bug 的差异（按用户要求）

1. prefill 保护边界小范围 off-by-one（例如 127/128/129）
2. recent window 边界小范围差异（若不改变总体算法语义）
3. 时序不同但数学等价/算法等价

## 下一步（按优先级）

1. 做 `per_head` 端到端 anchor 对齐核验（P0）
2. 做 `per_layer_per_head` 端到端 anchor 对齐核验（P0）
3. 补“HF 参考配置 vs V2 配置”逐项差异表，标注哪些是非 bug 差异（P1）
4. 只有在出现明确不对齐证据时，才进入代码修复（避免无效改动）
