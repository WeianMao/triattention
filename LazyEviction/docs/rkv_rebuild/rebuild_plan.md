# RKV → LazyEviction 重构计划与风险

> 目的：在 LazyEviction 体系内重构 RKV，实现公平对比。计划阶段仅梳理接口/风险，不改动现有算法；每个差异点需具备可配置方案并保持默认不变。

## 1. 重构目标
- 复制 RKV 算法核心（kv 选择/压缩策略），但以 LazyEviction 的调度、配置、输出规范实现。
- 首版对齐 `run_sparse_prefill_keep_sharded_eval.sh` 的运行外壳（分片、日志、配置路径、PD-L1 前缀），形成一键脚本。
- 提供与 LazyEviction 其他方法可比的配置：KV budget 定义一致、prefill 不压缩且不计入预算、数据/提示词/解码策略可切换。

## 2. 关键差异与“潜在作弊”检查项
- **KV budget 计数**：RKV 当前的 `kv_budget=2048` 是否按 token 维、按层均分、是否包含 prefills/padding/新轮 KV；需与 LazyEviction 计量一致，避免因定义差异获得更多可用 KV。
- **Prefill 处理**：确认 RKV 是否在 prefills 上执行压缩或计入 budget；LazyEviction 约定为“完整前缀+不占 budget”，如有出入需提供对齐选项。
- **数据与提示词**：RKV 使用 plain prompt + `aime25.jsonl`；LazyEviction 使用 chat 模板与 `datasets/aime/test.jsonl`。需要可切换的数据/模板，以免提示词差异影响对比。
- **解码策略**：RKV 用采样（temp 0.6/top_p 0.95、num_samples=8 + pass@1 平均）；LazyEviction 默认为贪心单样本。需提供 `do_sample/temperature/top_p/num_samples/aggregation` 的开关。
- **长度与截断**：RKV `max_length=32768`（总长）；LazyEviction 通常用 `max_new_tokens`（16384）。需记录截断率并可配置两种模式。
- **Attn/dtype/实现细节**：RKV 默认 flash-attn2 + bf16 + `reset_cache_each_batch=false` + `fp32_topk=false`；LazyEviction 多用 sdpa+fp16。需要可配置以排除内核/精度差异。
- **分片/调度/输出**：确认 shard 切分、合并、日志路径与 LazyEviction 方式一致，避免某侧额外复用缓存或避开开销。
- **评估口径**：RKV 8 样本 pass@1 平均 vs LazyEviction 单样本准确率；需要可切换或记录成独立指标。

## 3. 实施路线（阶段）
1) **盘点与接口设计**  
   - 解析 `R-KV/weian_development/rkv_sharded_dispatch.py`、`rkv_sharded_runner.py` 与 `sample8_rkv_aime25_official_qwen.yaml`，列出必需参数与默认值。  
   - 定义 LazyEviction 版 cfg schema（yaml + argparse），保持默认值与 LazyEviction 习惯一致（尤其 budget、prompt/数据、解码默认）。  
   - 确认需要的依赖模块是否可迁移/复用，避免跨仓库 import。
2) **调度与 Runner 设计**  
   - 参考 `run_sparse_prefill_keep_sharded_eval.sh`/`lazy_eviction_sparse_prefill_keep_dispatch.py`，设计新的 RKV dispatch 与 runner；隔离路径（输出/日志/配置前缀独立）。  
   - 加入安全检查：缺失统计文件/头部采样/预算不合理时早报错；prefill 计数逻辑可调。  
3) **公平性对齐开关**  
   - 数据/模板选择（plain vs chat）、解码/聚合（采样/贪心、多样本平均）、attn/dtype、长度、trust_remote_code、seed/reset_cache。  
   - KV budget 计数方式（含不计前缀、计 or 不计 padding/chunk 边界）需可控，并在日志中打印实际可用 KV。  
4) **验证计划**  
   - Smoke：单 shard、1-2 样本、最小 kv_budget，验证入口/输出结构、budget 统计、prefill 处理。  
   - 对比：在相同数据/提示词/贪心配置下，与 LazyEviction 其他方法跑小规模对照，确认公平性与性能初步趋势。  
5) **收尾与文档**  
   - 更新本目录与新脚本注释，记录默认值与差异点；准备长跑命令模板。

## 4. 依赖与资源假设
- 模型：DeepSeek-R1-Distill-Qwen-7B（已有路径）。  
- 数据：`aime25.jsonl` 与 LazyEviction 生成的 `datasets/aime/test.jsonl`。  
- 环境：`lazy_evict` conda；需 flash-attn2 或 sdpa 按配置切换。  
- 进程命名：保持 `PD-L1_binder`。

## 5. 判定标准（完成态）
- LazyEviction 下有独立的 RKV 调度/runner + YAML + 一键脚本；默认行为不影响现有方法。  
- KV budget/Prefill 处理、数据/模板、解码、长度等关键影响因素可配置对齐；默认遵循 LazyEviction 规则。  
- 文档与 TODO 更新，列出任何仍存在的潜在不公平或待决问题。
