## 多模型多算法对齐与可视化/指标计划

### 背景与目标
- 需要统一跑通并可视化 5 条压缩/基线路径，覆盖 LLaMA 与 Qwen 两类模型，便于对比性能与可视化差异。
- 可视化与命中率指标沿用老版 hybrid rounds xtrace 的 benchmark 逻辑；压缩逻辑保持各自脚本原样（含潜在 bug），只做旁路分析。

### 模型与入口
- **LLaMA（DeepSeek-R1-Distill-Llama-8B）**：`/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B`，在 R-KV sample8/speckv/snapkv/rkv/h2o 等 YAML 中均指向该路径（例：`R-KV/weian_script/configs/sample8_sparseprefillkeep_aime24_official.yaml`）。
- **Qwen（DeepSeek-R1-Distill-Qwen-7B，LazyEviction 路径）**：`/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B`，见 `LazyEviction/weian_script/configs/sparse_prefill_keep_aime.yaml`。
- **Qwen（R-KV 官方适配的 14B 版本）**：在 R-KV 的 configs/shell 中未找到显式 Qwen14B 配置/脚本，需要同事补充具体入口或配置；暂留空位待确认。

### 算法/脚本矩阵（需统一接入可视化）
1) **LazyEviction sparse_prefill_keep**  
   - 入口：`weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py` + `LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh`（配置 `configs/sparse_prefill_keep_aime.yaml`）。  
   - 关键差异：需要“同一 token、同一层的所有 KV head 同步保留/删除”（非 per-head 独立）。当前 LLaMA 可视化脚本是 per-head 打分/选择，未实现多头同步淘汰，需新增支持。
2) **SpeckV（R-KV）**  
   - 入口：`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`（模型 LLaMA 8B）。
3) **SpeckV-norm 变种**  
   - 入口：`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8_norm.sh`（模型 LLaMA 8B）。
4) **SnapKV（R-KV）**  
   - 入口：`R-KV/weian_script/aime24_official_sampled8/run_snapkv_aime24_official_sampled8.sh`（模型 LLaMA 8B）。
5) **R-KV baseline**  
   - 入口：`R-KV/weian_script/aime24_official_sampled8/run_rkv_aime24_official_sampled8.sh`（模型 LLaMA 8B）。
6) **当前 LLaMA 可视化基线**  
   - 入口：`weian_development/rkv_debug/attention_pruning_llama_xtrace.py`（使用捕获 Q/K 旁路；per-head 打分；patch=32 默认写死）。

### 待办与拆解
- **A. 头同步保留/删除支持（LazyEviction 对齐）**
  - 在可视化/指标脚本中新增“同一 token 同层全头同步”模式：对每 token 的打分/选择先跨头聚合（如 head-wise max/mean）再产生统一的 prune_mask，或直接对 prune_mask 做 head-wise AND/OR，同步保留/删除。参考 `LazyEviction/docs/sparse_prefill_keep_sharded_eval.md` 的“同一 token、同一层的所有 KV head 同步保留/删除”要求。
  - 确认当前 LLaMA 脚本：**现状是 per-head 独立打分/淘汰，不符合同步要求**，需要新增模式开关。
  - 产物：新增脚本或参数（如 `--sync-heads true`），并在文档中说明默认行为与 LazyEviction 要求的差异。
- **B. 模型适配**
  - LLaMA：沿用现有捕获与脚本，无需改模型路径。
  - Qwen-7B（LazyEviction）：复用现有 capture/benchmark 逻辑，注意 attn 实现/rope 差异，保持 benchmark 不改。确认 capture 路径或新增 Q/K 捕获脚本（若未有）。
  - Qwen-14B（R-KV 官方）：需同事提供具体 YAML/脚本或模型路径；计划留空待补。
- **C. 算法跑数与对齐**
  - SpeckV/SpeckV-norm/SnapKV/RKV：按原脚本参数跑，开启旁路 capture（保持默认 patch=32）；生成指标/可视化。
  - LazyEviction sparse_prefill_keep：在同步模式下生成可视化/指标，并对比 per-head 模式差异（记录偏差原因）。
- **D. 文档与复现**
  - 在本文件持续记录命令、输出目录、采样头文件位置（如 `/tmp/llama_heads_100.json` 等），以及同步模式的配置说明。
  - 对 Qwen 14B 入口缺失之处做 TODO 标记，待补。

### 当前状态检查
- LLaMA 可视化脚本（`attention_pruning_llama_xtrace.py`）现默认 patch_size=32/min=32，per-head 打分/淘汰，**未实现多头同步保留/删除**，需在 LazyEviction 场景下改造或新增模式。
- 已有捕获样本：`outputs/qk_capture_fullkv/shard00/run001_sample00061/{qk.pt,metadata.json}` 可用于以上算法的离线复刻（LLaMA）。
- 采样头文件：`weian_development/rkv_debug/llama_sample_heads.json`（默认 8 个）；`/tmp/llama_heads_100.json`（最新 100 头随机）可重复使用或删除重采样。

### 待确认/需求
- R-KV 官方 Qwen14B 入口/模型路径未找到，请提供具体脚本或 YAML。
- LazyEviction 路径下若已有 Q/K 捕获或可视化管线，请指明可复用部分，否则按 LLaMA 脚本适配 Qwen。
