# V2 全面审计计划（HF 对齐 + 性能/潜在 Bug）

- 开始时间：2026-02-24
- 审计目标：
  1. 检查 TriAttention V2 是否与 HF 参考脚本对齐（以 `per_head` 为主目标；`protect_prefill` 不做严格要求）
  2. 检查是否存在未发现 bug 或实现不当导致的吞吐/运行速度问题
- HF 对齐参照脚本：
  - `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`

## 审计边界（本轮）

1. 主目标模式：`per_head`
2. 额外支持模式：`per_layer_per_head`（检查实现正确性与风险，不要求本轮立即跑 full experiment）
3. `per_layer`：
   - 保留能力
   - 默认禁止误用（未经批准应报错）
4. 非严格要求项（本轮不作为阻塞）：
   - `protect_prefill` 是否完全一致
   - 小范围 off-by-one（例如 127/128/129）
   - 时序不同但数学等价/算法等价

## 审计产出要求

1. 仅报告“有意义差异”：
   - 真的 bug
   - 高风险潜在 bug
   - 与 HF 脚本不一致且可能影响结果/吞吐的实现差异
2. 不报告：
   - 实现形式不同但理论等价
   - 极小且用户明确可接受的差异
3. 每个发现需标注：
   - `HF对齐` / `性能` / `稳定性`
   - `严重程度`
   - `证据（文件/测试/实验）`
   - `建议（修复/接受/后续验证）`

## 执行计划（阶段）

### 阶段 A：口径与范围确认（P0）

- [x] 确认 HF 脚本实际生效参数（只审脚本用到的参数）
- [x] 确认 V2 `per_head` anchor 配置映射是否与脚本一致（脚本未用参数已降级）
- [x] 列出“必须对齐项”与“允许差异项”

### 阶段 B：HF 对齐链路审计（P0）

从输入参数到压缩执行的全链路，重点看 `per_head`：

- [x] `config.py` / runner 参数映射（是否能表达 HF 脚本语义）
- [x] `selector_hf.py`（per_head 语义、GQA 头处理、聚合/normalize）
- [x] `selection_planner.py` / `planner.py` / `plan_models.py`（保留集合规划）
- [x] `layout_engine.py` / `kv_compaction.py`（keep 集合落地、布局与搬运）
- [x] `hook_group_pipeline.py` / `hook_impl.py`（组级执行与调用顺序）
- [x] `runner.py` / `runner_compression_actions.py` / `runner_state_updates.py`（触发与状态闭环）
- [x] `effective_overrides.py` / `input_adapter.py` / `input_patch_*`（runtime 语义适配）
- [x] `worker_reclaim_sync.py`（回收同步是否破坏正确性）

### 阶段 C：性能与吞吐风险审计（P0/P1）

重点看 decode 热路径、Python 开销、同步点、冗余 metadata：

- [x] `integration_monkeypatch.py`（接入方式是否保持轻量）
- [x] `scheduler.py` / `effective_len_tracker.py`（每步额外工作量）
- [x] `worker.py` / `runner.py`（每步 wrapper/代理是否会拖慢）
- [x] `effective_overrides.py` / `input_patch_*`（是否仍有 decode 热路径高开销）
- [x] `kv_compaction.py` / `layout_engine.py`（压缩触发点的 CPU/GPU 开销热点）
- [x] `perf_profile.py` 与已有 profiling 路径（是否还存在易误导/遗漏）

### 阶段 D：证据复核与汇总（P0）

- [x] 跑关键 pytest（对本轮审计涉及模块；部分环境阻塞时采用已有通过记录+补充 compileall）
- [x] 如需要，补最小定点测试（不做大范围实验）
- [x] 产出最终报告：仅列有意义差异/风险与建议（见 `V2_HF_ALIGNMENT_AUDIT_CHECKLIST_2026-02-24.md`）

## 模块审计记录（滚动更新）

状态说明：
- `TODO` 未开始
- `IN_PROGRESS` 进行中
- `DONE` 完成
- `RISK` 已发现问题待修/待确认

### HF 对齐链路

- `config.py` / runner 参数映射：`DONE`
- `selector_hf.py`：`DONE`
- `selection_planner.py`：`DONE`
- `planner.py`：`DONE`
- `plan_models.py`：`DONE`
- `layout_engine.py`：`DONE`
- `kv_compaction.py`：`DONE`
- `kv_group_resolver.py`：`DONE`（已修复跨 backend layout 歧义；需纳入最终报告）
- `hook_group_pipeline.py`：`DONE`
- `hook_impl.py`：`DONE`
- `hook_preflight.py`：`DONE`
- `hook_runtime_context.py`：`DONE`
- `runner.py`：`DONE`
- `runner_compression_actions.py`：`DONE`
- `runner_state_updates.py`：`DONE`
- `effective_overrides.py`：`DONE`
- `input_adapter.py`：`DONE`
- `input_patch_backend.py`：`DONE`
- `input_patch_state.py`：`DONE`
- `input_patch_ops.py`：`DONE`
- `input_patch_vllm_backend.py`：`DONE`
- `worker_reclaim_sync.py`：`DONE`

### 性能与吞吐

- `integration_monkeypatch.py`：`DONE`
- `scheduler.py`：`DONE`
- `effective_len_tracker.py`：`DONE`
- `worker.py`：`DONE`
- `runner.py`：`DONE`（补修默认关闭 profiling 时的每步计时开销）
- `effective_overrides.py`：`DONE`
- `input_patch_*`：`DONE`
- `kv_compaction.py`：`DONE`（压缩触发点允许 host-side校验/同步；不属于 decode 热路径）
- `layout_engine.py`：`DONE`
- `perf_profile.py`：`DONE`（确认 env gated；新增 runner/bridge fast-path 避免默认每步计时）

## 初始已知信息（复用历史结论）

1. 已知性能大问题（wrapper/subclass 接入拖慢 decode 热路径）已修复，当前 monkeypatch 接入可恢复接近原生 vLLM 性能。
2. 已知 `per_layer_per_head` GQA 聚合高风险不等价路径已做代码修复，但仍需在本轮审计中复核逻辑与证据。
3. 已知跨 backend KV layout 歧义（`num_blocks == 2`）已修复，并已加测试覆盖（Flash/Triton + 歧义 shape + resolver 自动注册 hint）。

## 本轮新增发现（简表）

1. `per_head` 参照脚本的关键参数映射在 V2 anchor 配置中已覆盖，`disable_trig/disable_top_n_high_freq/disable_mlr` 等非脚本项可忽略。
2. `per_layer_per_head` 的 GQA 聚合顺序风险已在代码中修复（先打分再按 KV-head 聚合），但仍建议后续单独做端到端验收。
3. decode 热路径存在一处默认关闭 profiling 时仍执行 `perf_counter` 的实现不当，已修复（不改变算法语义）。
4. `worker_reclaim_sync.py` 中的 `torch.cuda.synchronize()` 仅在 debug 校验开关开启时触发，不构成默认吞吐风险。
