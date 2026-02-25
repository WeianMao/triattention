# TriAttention_vLLM 当前实现总览（默认版本）

- 更新时间：2026-02-25
- 适用版本：vLLM `0.15.x`
- 说明：当前默认实现的内部兼容目录名仍为 `triattention_v2/`，这是为了降低重构风险；对外入口已切换为无 `V2` 命名。

## 1. 默认入口（建议使用）

1. Dispatch：
   - `TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py`
   - 默认配置：`TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24.yaml`
2. Runner：
   - `TriAttention_vLLM/evaluation/runner/vllm_triattention_runner.py`
   - 兼容转发到现有实现：`TriAttention_vLLM/evaluation/runner/vllm_triattention_v2_runner.py`

## 2. 模块边界（代码即文档）

1. 集成接入（保性能的 monkey patch 路线）
   - `TriAttention_vLLM/triattention_v2/integration_monkeypatch.py`
   - 作用：在原生 vLLM `Scheduler/Worker` 上做最小方法替换，避免 wrapper/subclass 热路径开销
2. 调度触发与有效长度跟踪
   - `TriAttention_vLLM/triattention_v2/scheduler.py`
   - `TriAttention_vLLM/triattention_v2/planner.py`
   - `TriAttention_vLLM/triattention_v2/effective_len_tracker.py`
3. Worker/Runner 执行编排
   - `TriAttention_vLLM/triattention_v2/worker.py`
   - `TriAttention_vLLM/triattention_v2/runner.py`
   - `TriAttention_vLLM/triattention_v2/runner_compression_actions.py`
   - `TriAttention_vLLM/triattention_v2/runner_state_updates.py`
   - `TriAttention_vLLM/triattention_v2/runner_output_bridge.py`
4. HF 对齐选择逻辑（当前主目标 `per_head`）
   - `TriAttention_vLLM/triattention_v2/selector_hf.py`
   - `TriAttention_vLLM/triattention_v2/selection_planner.py`
5. 布局整理与压缩/回收
   - `TriAttention_vLLM/triattention_v2/layout_engine.py`
   - `TriAttention_vLLM/triattention_v2/kv_compaction.py`
   - `TriAttention_vLLM/triattention_v2/kv_group_resolver.py`
   - `TriAttention_vLLM/triattention_v2/worker_reclaim_sync.py`
6. 运行时输入语义适配（effective len / positions / slot mapping）
   - `TriAttention_vLLM/triattention_v2/effective_overrides.py`
   - `TriAttention_vLLM/triattention_v2/input_adapter.py`
   - `TriAttention_vLLM/triattention_v2/input_patch_backend.py`
   - `TriAttention_vLLM/triattention_v2/input_patch_vllm_backend.py`
   - `TriAttention_vLLM/triattention_v2/input_patch_ops.py`

## 3. 当前默认语义（重要）

1. 默认 `pruning_mode` 为 `per_head`
   - 配置定义：`TriAttention_vLLM/triattention_v2/config.py:48`
2. `per_layer` 默认禁止误用（需显式 opt-in）
   - 校验：`TriAttention_vLLM/triattention_v2/config.py:204`
   - selector 门禁：`TriAttention_vLLM/triattention_v2/selector_hf.py:30`
3. 当前主线对齐语义：
   - `per_head`（HF 对齐）
   - `per_layer_per_head`（已修 GQA 聚合顺序风险，待端到端实验验收）

## 4. 维护建议（简版）

1. 优先改对外入口与配置，不要轻易大规模重命名 `triattention_v2/` 内部目录。
2. decode 热路径改动要极少，性能问题先检查：
   - `integration_monkeypatch.py`
   - `worker.py`
   - `runner.py`
   - `runner_output_bridge.py`
3. 如需理解对齐逻辑，先读 `selector_hf.py` 和 `selection_planner.py`，再看 `layout_engine.py`。
