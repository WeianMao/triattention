## 计划：对齐 `run_speckv_aime24_official_sampled8.sh` 的可视化与命中率调试

### 目标与硬性约束
- 目的：将 `weian_development/rkv_debug/docs/kv_pruning_trace_upstream.md` 里用于 Qwen 仿真的可视化与指标收集方法，逐项移植到 LLaMA 路径 `R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh` 的真实跑数上，用于定位性能劣化原因。
- 约束：**所有执行路径、校准、裁剪、采样等行为必须与原始 `run_speckv_aime24_official_sampled8.sh` 完全一致，连同潜在 bug 一并保留**；新增的可视化/指标只能是旁路记录，不得改动原有决策逻辑或参数。
- 环境：默认使用 conda env `dc`；进程名保持 `PD-L1_binder` 约定；HF/PIP cache 路径沿用脚本中已有导出。

### 现有资料与输入
- 入口脚本：`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`（调用 `R-KV/weian_development/rkv_sharded_dispatch.py`，使用 `configs/sample8_sparseprefillkeep_aime24_official.yaml`；固定 8 draws）。
- 上游调试文档：`weian_development/rkv_debug/docs/kv_pruning_trace_upstream.md`（Qwen 仿真，依赖 `weian_development/attention_qk_analysis/capture_qk_distributed.py` 与 `online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py`）。
- 已采样 Reasoning Trace：`R-KV/outputs/sample8_fullkv_aime24_official/shards/shard00.jsonl` 的第 2 行（`"id": 61`）。调试时需要锁定该样本，确保与仿真一致。
- 相关配置：`R-KV/weian_script/configs/sample8_snapkv_aime24.yaml`（小样本烟雾测试），`sample64_snapkv_aime24.yaml`（完整运行）。

### 工作分块与可交付物
- [X] **A. 基线重放与对齐确认**
  - [X] 从 `run_speckv_aime24_official_sampled8.sh` 读取实际命令行与 env，生成最小可重放指令。最小重放示例：`bash R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh --dry-run`（默认 auto GPU/num_shards=8，配置 `R-KV/weian_script/configs/sample8_sparseprefillkeep_aime24_official.yaml`，输出到 `R-KV/outputs/sample8_sparseprefillkeep_aime24_official/shards`，合并目录 `R-KV/outputs/sample8_sparseprefillkeep_aime24_official/merged`，日志 `R-KV/logs/sample8_sparseprefillkeep_aime24_official`）。
  - [X] 交付物：一份 README 片段或注释，记录“原样”重放命令、依赖的 YAML、期望输出目录（不修改代码）。本脚本固定 8 draws，不读取 `SAMPLES` 环境变量。
  - [X] 测试：`python -m compileall R-KV/weian_development R-KV/weian_script` 确认语法（已在当前工作区执行，成功）。

- [X] **B. 仿真数据链移植（旁路捕获 Q/K 或中间态）**
  - [X] 参考 `capture_qk_distributed.py` 的 pre-hook 方式，在 LLaMA 推理路径插入只读钩子，捕获与 KV 决策相关的张量（Q/K 为主）。新增模块：`weian_development/rkv_debug/qk_capture.py`；在 `R-KV/HuggingFace/rkv/modeling.py` 的 LlamaAttention prefill 阶段调用 `maybe_capture_qk`，不影响原逻辑。
  - [X] 运行时通过环境变量开启，默认关闭：`RKV_QK_CAPTURE_DIR=/path/to/out RKV_QK_CAPTURE_SAMPLES=61 bash R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh --qids 61 --max-workers 1`（如无 `--qids`，需在 YAML 或输入过滤保证仅跑目标样本）。`RKV_QK_CAPTURE_SAMPLES` 为空则对所有样本捕获；目录结构：`<DIR>/shardXX/runYYY_sampleZZZZZ/{metadata.json,qk_layerNN.pt}`。
  - [X] 保持 run 脚本配置（校准、裁剪、采样）不变，仅在旁路中读取；元数据写入 `metadata.json`（model_path/dataset_path/kv_budget/window_size/attn_implementation/load_dtype/temperature/top_p/prefill_length 等）。
  - [X] 交付物：捕获模块及调用说明（见上），落盘示例：`RKV_QK_CAPTURE_DIR=outputs/qk_capture` 时，会生成 `outputs/qk_capture/shard00/run000_sample00061/qk_layer00.pt` 等。
  - [X] 测试：`python -m compileall R-KV/weian_development R-KV/HuggingFace/rkv`（已执行，成功）；后续烟雾需实际跑 `"id": 61` 单条验证文件生成。

- [ ] **C. 命中率与裁剪指标复刻**
  - [ ] 从 Qwen 仿真指标（命中率、被裁剪 token 占比、位置分布等）抽取公式，映射到 LLaMA 输出的旁路数据；如缺失字段，记录需要从哪一层新增观测但仍不改写原逻辑。
  - [ ] 指标计算脚本应独立于主推理（离线分析），输入为 B 步骤落盘的张量/元数据。
  - [ ] 交付物：指标计算脚本 + 指标定义文档（列出与 Qwen 的一一对应关系、可能的偏差原因）。
  - [ ] 测试：用 `"id": 61` 的捕获结果跑指标脚本，输出 JSON/CSV 摘要；检查运行成功即可，不对数值阈值做断言。

- [ ] **D. 可视化复刻**
  - [ ] 参照 `online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py` 的可视化流程，绘制与 SnapKV 相关的 heatmap/trajectory；如果维度不匹配（头数/层数），在文档中注明映射规则，保持绘图代码只读原始捕获数据。
  - [ ] 交付物：可视化脚本、示例截图路径、运行命令。
  - [ ] 测试：对 `"id": 61` 数据生成一张示例图，确认脚本无异常退出。

- [ ] **E. 差异与潜在缺口记录**
  - [ ] 列出 Qwen 仿真与 LLaMA 实跑在框架、前处理、KV 管理策略上的差异，标记哪些可能导致性能差异（但不在本轮修复），供后续定位。
  - [ ] 交付物：差异对照表，标注“观察/待验证/需代码比对”。
  - [ ] 测试：无代码测试，人工核对文档完整性。

### 单元测试与烟雾检查分配（给执行同事）
- [ ] 语法检查：`python -m compileall deepconf R-KV/weian_development R-KV/weian_script`。
- [ ] 捕获旁路烟雾：`bash R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh --qids 61 --max-workers 1`（如有 `--qids` 开关；若无则在配置中单独挑选 shard，并设置 `RKV_QK_CAPTURE_*` 环境变量），确认输出的 `qk_layer*.pt`/`metadata.json` 生成。
- [ ] 指标脚本：`python <metrics_script> --input <capture_dir_for_id_61> --output tmp_metrics.json`，检查文件生成。
- [ ] 可视化脚本：`python <viz_script> --input <capture_dir_for_id_61> --out-dir tmp_viz`，确认图片生成。
- [ ] 任何失败都不得通过修改主推理逻辑修复，只能修复调试脚本自身或补充文档。

### 备注与后续协作提示
- 如果需要新增模块，优先放在 `weian_development/`（遵守复用约定），并保持 ASCII。
- 长跑任务如需后台执行，记得在调度脚本中保持 `PD-L1_binder` 前缀。
- 大文件（捕获张量、可视化图片）不纳入版本控制，落在已有忽略目录（如 `outputs/`）下。
- 后续开发均在 `weian_development/rkv_debug` 路径下完成，本次调整已遵守此约定。

### 发现的问题与修正（2025-02）
- SpeckV 路径未经过 `replace_llama` monkeypatch，导致 `maybe_capture_qk` 不会被调用，`qk_capture.py` 环节根本无效，GPU 负载也无法体现拷贝；需要在 SpeckV/FullKV 实跑时额外打补丁。
- 现已在 `weian_development/rkv_debug/qk_capture.py` 提供 `patch_llama_attention_for_capture()`，在 `R-KV/weian_development/rkv_sharded_eval.py` 内部，当设置 `RKV_QK_CAPTURE_DIR` 时自动调用；若补丁失败会给出 stderr 提示但不影响原逻辑。
- 开启捕获的正确方式（保持原脚本参数不变）：`RKV_QK_CAPTURE_DIR=outputs/qk_capture RKV_QK_CAPTURE_SAMPLES=61 bash R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh --max-workers 1`。捕获仅发生在 prefill，无 past cache 时，每层一次。落盘路径示例：`outputs/qk_capture/shard00/run000_sample00061/qk_layer00.pt`。
- 如果需要确认补丁生效，可先 dry-run：`RKV_QK_CAPTURE_DIR=/tmp/qk RKV_QK_CAPTURE_SAMPLES=61 python -m compileall R-KV/weian_development/rkv_sharded_eval.py`（仅验证导入和补丁不报错），然后实际跑单条 smoke 验证文件生成。

### FullKV 单条样本 Q/K 捕获（对齐 R-KV prompt 与设置）
- 新脚本：`weian_development/rkv_debug/capture_fullkv_qk.py`。直接重放 dataset 中的指定 `id`（默认 61），使用 R-KV 的 `build_prompt` 与相同 seed/温度/Top-p/attn_implementation/load_dtype，前向一次（无自回归采样），在 LlamaAttention pre-hook 中复制投影+RoPE 计算并保存原始 Q/K。
- 输出：`<output-dir>/shardXX/runYYY_sample00061/{qk.pt,metadata.json}`，其中 `qk.pt` 包含全层全长的原始张量（无任何统计聚合）。
- 单卡运行示例（仅 GPU 7，prefill 捕获）：  
  ```
  CUDA_VISIBLE_DEVICES=7 conda run -n rkv \
    python weian_development/rkv_debug/capture_fullkv_qk.py \
      --dataset-path R-KV/HuggingFace/data/aime24.jsonl \
      --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B \
      --output-dir outputs/qk_capture_fullkv \
      --sample-id 61 \
      --shard-id 0 --run-id 0 \
      --max-length 32768 \
      --temperature 0.6 --top-p 0.95 \
      --attn-implementation flash_attention_2 \
      --load-dtype bfloat16
  ```
- 注意：只进行一次前向，Q/K 捕获发生在 prefill（无 past cache）；若 flash_attn2 环境缺失，可将 `--attn-implementation sdpa` 作为降级（与官方配置有差异）。
