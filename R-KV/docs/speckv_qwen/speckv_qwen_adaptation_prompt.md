# SpecKV 适配 Qwen 任务梳理（来自原始 Prompt）

## 背景
- 最早的 SpecKV/KV 压缩模拟在 `weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py`（模型：Qwen）；数学推理效果良好。
- 同事在 LazyEviction 路线用 `LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh` 评测，表现也好。
- 另一位同事在 R-KV 重构出 `R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`（模型：Llama，已按 R-KV style 适配、包含多处改动）。但该版本效果很差，原因不明：可能重构引入 bug/不合适的 setting，也可能模型差异（Llama vs Qwen）。

## 当前目标
- 在 **R-KV 框架与风格** 下，将 SpecKV 算法适配 **Qwen** 模型，所有代码停留在 `R-KV/`。
- 需要完成两组实验的目标脚本（各自以指定脚本为骨架去修改，改动仅限描述的差异，其余尽量保持一致），修改后即可在各自实验组内做对比：
  - **实验组 1（基于 `run_rkv_aime25_official_sampled8_qwen.sh`）**  
    - 目标脚本：在 `R-KV/weian_script/aime24_official_sampled8/` 下新增/改出 SpecKV 版本，整体 follow `run_rkv_aime25_official_sampled8_qwen.sh` 的结构与配置，**唯一核心差异是算法从 rkv 换成 speckv**。  
    - 修改后，该 SpecKV 脚本应与 `LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh` 在关键逻辑上非常接近（仅允许采样参数等次要差异），可互为参考；同时可与原始 rkv 版（同名 Qwen 脚本）做对比。  
  - **实验组 2（基于 `run_speckv_aime24_official_sampled8.sh`）**  
    - 目标脚本：继续用 SpecKV 算法，但**把模型从 Llama 换成 Qwen**，其余流程尽量保持原脚本风格；完成后放在同目录。  
    - 修改后，该 Qwen 版 SpecKV 脚本可与 R-KV 的其他 Qwen 脚本进行对比，比如 `R-KV/weian_script/aime24_official_sampled8/run_fullkv_aime24_official_sampled8_qwen.sh`、`R-KV/weian_script/aime24_official_sampled8/run_rkv_aime24_official_sampled8_qwen.sh`，保证同模型/基准、算法不同的横向评测。
- 需要额外产出一份 MD 文档，列出实验组 1 的 SpecKV 脚本与 `run_sparse_prefill_keep_sharded_eval.sh` 之间允许的次要差异（如采样参数）及实际差异清单；发现关键元素（模型、数据、pruner/position 逻辑、kv_budget、prompt 模板等）不一致时，必须标记为问题并告警。同时要核实文档中声称的可比性假设是否成立：实现者需检查对比对象是否真能做到“公平比较”或仅有可接受的差异；若发现设置完全不一致或有忽略的坑，必须及时警告。
- 目标是让 SpecKV（Qwen/R-KV 实现）与 LazyEviction 参考脚本形成**近似苹果对苹果**的可比性：模型、基准、数据保持一致；任何关键实现差异必须立即告警并停下。
- 实现时可大幅参考 R-KV 版脚本的组织方式（路径、风格、调度），但算法逻辑必须与早期 Qwen 版本等价。

## 必须参考的实现
- Qwen 基准逻辑：`weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py`（最早模拟脚本，算法正确性基线）。  
- Qwen LazyEviction 评测：`LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh`（实际评测表现好的版本，含分片调度、pruner、手写采样实现）。  
- R-KV 重构版本：`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`（R-KV style、目前用 Llama，效果差，可能含 bug/配置漂移，用作目录/风格/接口参考，但不盲信算法）。
- 对 LazyEviction 与 R-KV 的差异要敏感：实现 SpecKV 时要对照 LazyEviction 版本，确认关键逻辑（pruner状态、RoPE/position处理、round窗口、kv_budget、prompt模板、模型/数据选择）没有偏差；一旦发现不一致需立刻报告并暂停。

## 核心要求
- 三个版本在算法上**应当等价**：SpecKV 核心逻辑（pruner状态同步、round裁剪触发、prefix 保留、kv_budget使用方式、position/rope 处理、采样与生成接口）需与早期 Qwen 参考实现一致。  
- R-KV/Qwen 实现中，任何可能影响算法的差异都必须核对：  
  - 模型与数据：使用 Qwen + AIME 基准，与 LazyEviction 版本一致；差异仅限采样参数等非关键配置。  
  - 前向 patch/裁剪：行为、位置同步与 LazyEviction/模拟版一致，避免 Llama 重构中的潜在偏差。  
  - Prompt/模板：保持与早期 Qwen 版本一致（或显式说明并重算统计），避免因模板漂移导致性能差异。  
- 发现关键逻辑不一致（相对早期 Qwen 或 LazyEviction）的情况，必须**优先报警并暂停**，不要继续实现。  
- 代码组织、脚本风格遵循 R-KV 规范；路径和依赖全部落在 `R-KV/` 内。  
- 可以复用/借鉴 R-KV 现有脚本框架，但不盲信其算法正确性；以早期表现良好的 Qwen/LazyEviction 版本为准确认逻辑。

## 可能的性能差异来源（需留意）
- 重构过程引入的 bug 或不合理配置。
- 模型差异：当前 R-KV 版使用 Llama，而早期表现好的版本使用 Qwen；需验证 SpecKV 在 Qwen 上的表现。
