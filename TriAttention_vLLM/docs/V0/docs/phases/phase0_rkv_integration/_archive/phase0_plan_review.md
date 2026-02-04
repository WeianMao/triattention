# Phase 0 计划审查（R-KV 框架内 SpeckV 集成）

> 目标：评估 Phase 0 计划是否合理，指出缺失/潜在风险，并给出方向性解决建议。
> 约束重点：在 R-KV 框架内验证，且不干扰/修改他人算法行为。

---

## 一、结论概览（简版）

Phase 0 计划总体方向正确（先在 R-KV 快速验证），但存在架构选择不一致、隔离原则被破坏、运行语义不对齐、统计/模板/触发条件未校验等关键风险。若不补足，可能导致：

- 实验跑通但结论不可信
- 无意修改 R-KV 代码路径，影响他人算法
- 与 Phase 1 的迁移假设不一致

---

## 二、关键问题与潜在风险（按严重程度排序）

### 1) 架构方案自相矛盾（高风险）

- `phase0/README.md` 明确选择 monkey patch model.forward（全局压缩器），并强调“不修改 R-KV 核心代码”。
- `phase0/DESIGN_NOTES.md` 又提出新建 `rkv/compression/speckv_rkv.py`、修改 `rkv/modeling.py`、`run_math.py` 等核心入口，并走 update_kv 方式。

风险：两个实现路径语义不同，且 update_kv 无法满足跨层聚合（per_layer/per_layer_perhead），与 README 的选择相冲突。

建议：
- 明确只保留一个方案。
  - 若坚持跨层聚合 + 不改核心代码 → 必须走 `apply_speckv_rkv_style_patch` 的 forward patch 方案。
  - 若要走 update_kv → 必须明确只支持 per_head，并修正文档。
- 统一文档：README 与 DESIGN_NOTES 必须讲同一架构。

### 2) 隔离开发原则需要显式边界（高风险）

Phase 0 约束写的是“不修改 R-KV 核心代码”，但 DESIGN_NOTES 明确要求改：

- `rkv/compression/__init__.py`
- `rkv/modeling.py`
- `run_math.py`

这类修改如果没有明确隔离边界，可能影响 R-KV 默认行为，违反“保护他人算法”的核心要求。

建议：
- 如果允许修改核心文件，必须明确“仅通过参数/开关启用”，默认路径对其他算法零影响。
- 为 SpeckV 增加独立的 method/flag，并确保默认不触发 SpeckV 路径。
- 在文档中写清楚隔离策略（哪些文件允许改、如何保证不影响其他算法）。

### 3) 触发语义与 R-KV 现有机制可能不一致（高风险）

R-KV 有三态 compression flag（`None/True/False`），而 Phase 0 计划只按 `absolute_position % divide_length` 判断。

风险：触发时机不同会导致剪枝逻辑错位，结果与历史 SpeckV/基线不一致。

建议：
- 补充明确的触发对齐策略：
  - `compression=False` 时必须完全跳过压缩
  - `compression=True` 时才允许调用压缩
- 明确是否启用 `use_slack_trigger` 以及与 budget 的关系，并在计划里写死，避免行为漂移。

### 4) 统计文件/Prompt/Attn/Dtype 兼容性未校验（高风险）

SpeckV 依赖 stats 文件，stats 与以下要严格匹配：

- prompt 模板（plain vs chat）
- 模型版本
- RoPE scaling
- attention 实现（sdpa / flash-attn2）
- dtype

计划中只写了“跑脚本验证”，没有明确 stats 一致性检查。

建议：
- 在 Phase 0 计划里补充 stats 元数据校验：使用 `weian_development/speckv/stats_utils.py` 做一致性验证。
- 明确 “SpeckV 仅支持 plain prompt” 的前提是否与基线对齐。

### 5) reset_compression_state 未加入执行链（中高风险）

DESIGN_NOTES 提到了重置压缩器，但 Phase 0 README 没将其纳入执行流程。

风险：多样本评测时状态泄露，导致统计错误或剪枝异常。

建议：
- 在评估脚本（如 `rkv_sharded_eval.py`）中强制重置，或在 patch 内自动重置。

### 6) GQA 映射风险未纳入测试计划（中风险）

DESIGN_NOTES 提到 GQA 映射，但 Phase 0 测试计划只做“运行成功/准确率”。

风险：KV head 与 sampled head 的映射错误会导致 silent bug，结果可能“看似正常”但不可信。

建议：
- 增加 GQA 专项小测试：统计每个 KV head 覆盖的 sampled heads 数量是否与预期一致。

### 7) 长任务命名规范未纳入计划（中风险）

执行规范要求长任务使用 `PD-L1_binder` 进程名，但 Phase 0 计划没有涉及。

建议：
- 明确：长时间脚本应通过 `rkv_sharded_runner.py` 或指定 wrapper 保证进程名符合规范。

### 8) Phase 0 与 Phase 1 的迁移假设未明确（中风险）

Phase 0 的方案如果是 forward patch + global compressor，这与 Phase 1 的 vLLM PagedAttention 接入有较大结构差异。

建议：
- 在 Phase 0 输出中明确“哪些行为可迁移、哪些不能迁移”。
- 记录与 vLLM 0.15.x 的核心差异（KV layout、cache_position、kernel 实现）。

---

## 三、建议补充到 Phase 0 计划中的内容（可直接增补为 Checklist）

- 明确唯一架构选择（forward patch vs update_kv），删除另一套文档。
- 在所有 SpeckV 入口上强制隔离，不修改 `rkv/` 核心。
- 补充 stats metadata 校验流程（model/rope/prompt/attn/dtype）。
- 加入 compression flag 三态对齐测试。
- 加入 GQA 映射校验测试。
- 规定 reset_compression_state 的调用位置。
- 所有长任务确保 `PD-L1_binder` 命名。
- 明确 `output_dir` 命名防止覆盖基线结果。

---

## 四、值得保留的计划亮点

- 明确三种 pruning mode 作为主目标（合理）。
- 以 R-KV 先验证算法正确性，减少 Phase 1 风险（合理）。
- 明确“不使用 Triton、Batch=1”避免复杂度膨胀（合理）。

---

## 五、建议的最小修改方向（不破坏他人代码）

如果目标是“快速验证 + 零侵入”：

1. 保留 `weian_development/speckv/speckv_rkv_style.py` 与 `apply_speckv_rkv_style_patch` 作为唯一入口
2. 仅在 `weian_development/` 下新增 README 与 wrapper 脚本
3. 不改 `rkv/` 核心文件
4. 在 runner 脚本中显式 `method=speckv` + `rkv_style_compression`
5. 在运行前做 stats metadata 校验
6. 每次样本前 reset_compression_state

---

## 附：涉及文件（便于修改同事定位）

- `TriAttention_vLLM/docs/phases/phase0_rkv_integration/README.md`
- `TriAttention_vLLM/docs/phases/phase0_rkv_integration/DESIGN_NOTES.md`
- `R-KV/weian_development/speckv/speckv_rkv_style.py`
- `R-KV/weian_development/speckv/rkv_speckv_generate.py`
- `R-KV/weian_development/speckv/stats_utils.py`
