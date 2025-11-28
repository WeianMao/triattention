# SpecKV 在 R-KV 重构计划（详细要求 + TODO）

## 背景与硬性要求
- 用户原始要求（需完整落实）：  
  1) 旧算法在 LazyEviction 下，入口 `LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh`，希望迁移到 R-KV 框架。  
  2) 迁移后必须遵循 R-KV 的标准/Setting/方式（调度、配置、生成流程），能复用 R-KV 代码就复用，但不能影响自有算法逻辑。  
  3) 所有代码、依赖、统计必须放在 `R-KV/` 内，禁止依赖 `R-KV` 之外的文件。  
  4) 同事的半成品入口 `R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh`，任务记录 `docs/speckv_aime24_debug.md`；但实现存在复制粘贴、外部依赖等不符合上述标准的问题，需要重新规划。  
  5) 目前任务是出一份问题清单与分步修复计划（不是直接修代码），让后续同事按步骤执行。
- 旧实现链路（LazyEviction）：`LazyEviction/weian_script/run_sparse_prefill_keep_sharded_eval.sh` → `weian_development/lazy_eviction_sparse_prefill_keep_dispatch.py` → `weian_development/lazy_eviction_sparse_evaluation_prefill_keep.py`，包含分片调度、pruner、手写采样。
- 半成品链路（R-KV）：`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh` → `R-KV/weian_development/rkv_sharded_eval.py`，问题记录见 `docs/speckv_aime24_debug.md`。
- 基线参考（对齐行为）：`rkv/compression/*`（R-KV/SnapKV/H2O 的 forward 内裁剪 + HF generate）、官方 YAML `R-KV/weian_script/configs/sample*_rkv_aime24*.yaml`（flash_attn2 + bf16，kv_budget=2048，prompt 不压缩，generate 默认 top_k=50/默认 eos）。
- 非侵入式硬约束：所有 SpeckV 代码、统计、脚本驻留 `R-KV/`；不得破坏/修改 R-KV 现有算法逻辑；SpeckV 与其他方法隔离（独立模块/monkeypatch），确保 baseline 行为不变。

## 当前问题（精确列举）
1) **依赖跨仓库根**  
   - `R-KV/weian_development/rkv_sharded_eval.py`、`R-KV/weian_development/rkv_sparse_round_calibrate.py` 引用根目录的 `weian_development/hf_offline_runner_sparse/sparse_round_pruner_prefill_keep.py`、`round_pruning_utils.py`、`attention_qk_analysis`、`process_utils`。违反“全部在 R-KV”。
2) **实现双轨且与 R-KV 范式不符**  
   - `method=sparse_round_prefill_keep` 分支走手写采样循环（`speckv_generate_sequence`），采样参数与 HF generate 不同（缺省 top_k=0，仅温度+top-p），kv_budget 裁剪在 pruner 内；  
   - `method=speckv` 分支走 `apply_speckv_generate_patch` + HF generate，裁剪发生在 forward patch。两套路径并存，行为不一致。
3) **模板/统计不一致风险**  
   - 统计脚本默认 `use_chat_template=True`（系统 + 用户消息），运行 YAML 可关闭 chat；校准与运行可能不匹配。  
   - LazyEviction 数据读取多消息 JSON，R-KV 基线是单 question 模板，提示构造未统一；统计文件存放在根目录 `weian_development/hf_offline_runner_sparse/stats/*`。
4) **配置/命名漂移**  
   - `method` 存在 `speckv` 与 `sparse_round_prefill_keep` 两个名字；路径混用根目录模块。  
   - dispatcher/runner 仍依赖根目录的 `process_utils`。
5) **跨路径耦合 / 污染风险**  
   - `R-KV/weian_development/rkv_sharded_eval.py` 强行把仓库根 (`REPO_ROOT`) 放入 `sys.path`，脚本也设置 `PYTHONPATH=${PROJECT_ROOT}`，即使迁移代码也会优先解析根目录模块，违背“仅依赖 R-KV”要求。  
   - `R-KV/HuggingFace/run_math.py` 现版本从根 `weian_development` 引入 `rkv_cache_utils.py`（官方备份 `R-KV-backup-20250219/HuggingFace/run_math.py` 无此 import，`max_length` 也不同），说明文件已被改动并引入外部依赖，需要纠正。  
   - `rkv_sparse_round_calibrate.py` 默认的 `head-sample-file` 和 `round_pruning_utils` 也指向根目录，未迁移到 `R-KV/`。
   - `run_math.py` 已被改动：引入根目录 `rkv_cache_utils`（问题）、改为采样多抽样（正确，需保留）、`aime24` 上下文 32768（正确，需保留）、默认 bf16（应保持）、`mix_lambda=0.1`（正确，需保留）。

## 目标状态（完成后应达成）
- SpeckV 仅以一种方式存在：forward 内裁剪 + HF generate，接口和行为与 R-KV 其他方法一致；无手写逐 token 采样分支。
- 所有 SpeckV 代码和依赖、统计文件均在 `R-KV/`，`PYTHONPATH` 只需包含 `R-KV/`。
- 提示模板、统计、运行保持一致性：同一 chat 开关、同一 system prompt；若切换模板必须同步重算 stats。
- 非侵入式：对 R-KV 现有方法（rkv/snapkv/h2o/streamingllm 等）的代码路径零改动；SpeckV 的 monkeypatch/forward patch 与其他方法隔离，不共享状态。

## 参考对照文件（开发规范）
- R-KV 基线行为参考：`rkv/compression/snapkv.py`、`r1_kv.py`（forward 内 update_kv/裁剪）、`R-KV/weian_development/rkv_sharded_eval.py` 中非 SpeckV 分支的 generate 调用。
- 提示/数据模板参考：`R-KV/weian_development/rkv_sharded_eval.py` 的 prompt_template 与 `dataset_path=R-KV/HuggingFace/data/aime24.jsonl`；chat 模板参考 `AutoTokenizer.apply_chat_template` 使用方式。
- 统计生成参考：`R-KV/weian_development/rkv_sparse_round_calibrate.py`（需迁移依赖后自包含）。
- 调度/合并/评测流程参考：`R-KV/weian_development/rkv_sharded_dispatch.py`、`R-KV/weian_development/merge_rkv_shards.py`、`R-KV/HuggingFace/evaluation/eval_math_multi.py`。

## 拆解式修复步骤（含具体要求 + TODO 勾选）
按序执行，每一步完成后把 [ ] 改为 [x]。

1) 依赖迁移（自包含）  
   - [ ] 在 `R-KV/weian_development/` 下建立 SpeckV 子目录（或放入 `rkv/compression/`），复制并重命名下列文件：  
     - `weian_development/hf_offline_runner_sparse/sparse_round_pruner_prefill_keep.py`、`round_pruning_utils.py`、`stats/*`；  
     - `weian_development/rkv_speckv_generate.py`；  
     - 校准依赖的 `weian_development/attention_qk_analysis/capture_qk_distributed.py` 及必要模块；  
     - 最小化版 `weian_development/process_utils.py`（仅保留 `mask_process_command` 等必需函数）。  
     - `weian_development/rkv_cache_utils.py`（当前被 `run_math.py`/`rkv_sharded_eval.py` 引用）。  
   - [ ] 调整所有 import 指向新路径，确保 `PYTHONPATH` 只需 `R-KV/`，`sys.path` 不再插入仓库根。
   - [ ] 校准脚本的默认路径（如 `head-sample-file`）改到 `R-KV/`，不再落到根目录。
   - [ ] `R-KV/HuggingFace/run_math.py`：去除对根目录的依赖（迁移 `rkv_cache_utils` 后改 import），同时**保留**已有采样多抽样逻辑（do_sample/num_samples/temperature/top_p/输出 draw_idx）、`aime24=32768` 和 `mix_lambda=0.1`，默认 dtype 仍用 bfloat16。

2) 按 R-KV 范式重写 SpeckV 集成  
   - [ ] 合并双轨：删除/废弃手写采样分支，仅保留 “forward 内裁剪 + HF generate” 路径。  
   - [ ] 在 SpeckV 专用 forward patch 内实现：保持 prefix 固定，decode 轮次按 round_window 触发裁剪，kv_budget 不修改 max_length；采样参数使用 HF generate 默认（含 top_k=50、eos）并允许 YAML 透传温度/top_p。  
   - [ ] 将 SpeckV 注册到与 R-KV 其他方法相同的挂钩方式（可新增 `rkv/compression/speckv.py` 或专用 patch），确保对其他方法零侵入。  
   - [ ] 移除/合并 `method=sparse_round_prefill_keep` 与 `method=speckv` 的重复代码，保留单一名称（建议 `speckv`）。

3) 模板与统计一致性  
   - [ ] 抽象统一的 prompt 构造函数（chat 与非 chat），校准与运行共用；在运行时校验统计文件的元数据（chat 开关、system prompt）与当前配置一致，不一致则直接报错。  
   - [ ] 额外校验 stats 元数据中的 `attn_implementation`、`dtype`、`kv_budget` 等关键字段，防止用到 FA2/SDPA 或精度不一致的统计文件。  
   - [ ] 将统计文件生成与读取路径改到 `R-KV/outputs/.../stats/*.pt`，并在 SpeckV README/注释中写清如何重算（命令示例）。  
   - [ ] 如需兼容 LazyEviction 多消息 JSON，加入转换层（消息 → question/prompt），保证评测输入与 R-KV baseline 一致。

4) 配置与脚本收敛  
   - [ ] 统一 YAML：`method` 使用 `speckv`，所有路径改为 `R-KV/...`，去掉对根目录模块的隐式依赖。  
   - [ ] 更新脚本（`run_speckv_aime24_official_sampled8.sh`、`quick_tests/run_speckv_aime24_quick.sh` 等）指向新的模块路径；若需要额外说明，补充 SpeckV 专用 README。  
   - [ ] dispatcher/runner 使用迁移后的 `process_utils`，确认 merge/eval 流程不变。

5) 验证矩阵（完成即打勾）  
   - [ ] 烟囱测试：`num_shards=1`，`max_examples≈2`，`num_samples=1`，`max_length≈2048`，`attn_implementation=sdpa`，确认能写出 shard→merge→eval。  
   - [ ] 官方对齐测试：`run_speckv_aime24_official_sampled8.sh --gpus ... --num-shards 4`，输出行数=题目数×抽样数，评测正常。  
   - [ ] 模板/统计一致性：切换 chat 开关或模型后，重跑校准并在日志打印所用 stats 路径，验证调用的是新文件。  
   - [ ] 回归检查：R-KV 其他方法（rkv/snapkv/h2o/streamingllm）仍可通过相同 dispatcher/runner 运行。

## 预期单元/集成测试
- 快速自测：在 SpeckV 模块添加最小单测（如可用 pytest）验证 pruner 对小型假 cache 的裁剪逻辑、round_window 触发、前缀保留（可放到 `R-KV/tests/`，命名 `test_speckv_pruner.py`）。  
- 集成烟囱：上述“烟囱测试”即最小端到端验证（含 generate + merge + eval）。  
- 统计一致性：校准脚本运行后验证输出 metadata（chat 模板、head 采样信息）与运行配置匹配。

## 非侵入式开发提醒
- 不要修改 `rkv/compression/*` 现有方法逻辑；SpeckV 新增文件/patch 与其他方法隔离。  
- 不要改动 `R-KV/weian_development` 里与其他方法共用的代码路径；如需共用工具，复制到 SpeckV 自己的子目录并显式 import。  
- 不回退/改写已有 YAML、脚本的其他方法配置；仅调整 SpeckV 相关项。  
- 保持 PD-L1_binder 进程名需求，继续使用本仓内的 `mask_process_command` 副本。
- 避免污染 LazyEviction：如曾为 R-KV 适配改动了 `LazyEviction/` 或根 `weian_development/` 下的文件，应恢复成 Lazy 专用版本；R-KV 侧所需代码请自成一份，且通过 `PYTHONPATH`/`sys.path` 限定为 `R-KV/` 内部。

完成上述 TODO 后，SpeckV 将实现：行为对齐 R-KV 基线、代码完全驻留于 `R-KV/`、模板与统计一致、对现有算法零侵入，便于后续同事直接跟踪勾选进度。 

## 追加发现（当前计划未覆盖的缺口）
- **路径隔离未闭环**：`R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8.sh` 仍向 `PYTHONPATH` 注入仓库根；`rkv_sharded_runner.py`/`rkv_sharded_dispatch.py`/`rkv_sparse_round_calibrate.py` 也将根目录塞入 `sys.path`。即便迁移了代码，依然会优先解析根目录模块，违反“仅依赖 R-KV”要求。→ 补充明确动作：这些脚本/runner 改为仅插入 `R-KV/`（或不改 `sys.path`），并移除根目录注入。
- **默认文档路径仍指向外部**：`R-KV/weian_script/README.md` 的 head-sample 默认路径指向根目录 `weian_development/hf_offline_runner_sparse/stats/deepseek_r1_llama8b_heads.json`。计划虽提到复制 stats，但未要求更新文档与默认路径，容易再次引用外部。→ 补充动作：更新 README 和任何默认参数到 `R-KV/weian_development/...` 下的新副本。
- **SpeckV forward patch 语义差异**：`weian_development/rkv_speckv_generate.py` 当前在 `_sync_pruner_positions` 将 `cache_positions` 重置为 0..n，`absolute_position` 设为当前长度，且 prune 后未显式传递 `cache_position`/`position_ids`。与 LazyEviction 手写循环中用 `pruner.absolute_position` 驱动 RoPE 的逻辑不一致，FA2/SDPA 下可能导致 RoPE 坐标与统计文件错位。→ 补充动作：对齐 Lazy 版的状态同步（绝对位置、tokens_in_round 递增、cache_position 传递），确保裁剪与统计一致。
