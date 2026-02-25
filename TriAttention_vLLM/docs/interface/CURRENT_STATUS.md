# TriAttention_vLLM 当前状态（默认实现）

- 更新时间：2026-02-25
- 状态：Active
- 适用范围：vLLM `0.15.x`

## 1. 当前结论（简版）

1. 当前默认实现已经是以 `per_head` 为主目标的版本（对外入口已不再使用 `V2` 命名）。
2. 推理吞吐主问题（decode 热路径 wrapper/subclass 接入拖慢）已修复，当前主线采用 monkeypatch 最小接入。
3. `per_head` 全量 anchor（AIME24 sampled8）已跑通，结果在 HF 参考线附近。
4. `per_layer_per_head` 代码路径已修复关键 GQA 聚合风险，但端到端实验验收可后置。

## 2. 默认使用入口（对外）

1. Dispatch：
   - `TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py`
   - 默认配置：`TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24.yaml`
2. Runner：
   - `TriAttention_vLLM/evaluation/runner/vllm_triattention_runner.py`

说明：
- 内部实现目录名为 `TriAttention_vLLM/triattention_runtime/`。
- `TriAttention_vLLM/triattention_runtime/` 仅作为兼容导入包保留（薄转发层）。

## 3. 代码主链路（按职责）

1. 集成接入（性能优先）
   - `TriAttention_vLLM/triattention_runtime/integration_monkeypatch.py`
2. 调度与触发
   - `TriAttention_vLLM/triattention_runtime/scheduler.py`
   - `TriAttention_vLLM/triattention_runtime/planner.py`
   - `TriAttention_vLLM/triattention_runtime/effective_len_tracker.py`
3. 执行编排
   - `TriAttention_vLLM/triattention_runtime/worker.py`
   - `TriAttention_vLLM/triattention_runtime/runner.py`
   - `TriAttention_vLLM/triattention_runtime/runner_output_bridge.py`
4. HF 对齐选择与布局/压缩
   - `TriAttention_vLLM/triattention_runtime/selector_hf.py`
   - `TriAttention_vLLM/triattention_runtime/selection_planner.py`
   - `TriAttention_vLLM/triattention_runtime/layout_engine.py`
   - `TriAttention_vLLM/triattention_runtime/kv_compaction.py`

## 4. 模式状态

1. `per_head`
   - 当前主目标模式
   - 默认模式（见 `TriAttention_vLLM/triattention_runtime/config.py:48`）
2. `per_layer_per_head`
   - 需要支持
   - 关键代码路径已修（GQA 聚合顺序风险）
   - 端到端实验可后置
3. `per_layer`
   - 保留能力但默认禁止误用
   - 未显式放行会报错（见 `TriAttention_vLLM/triattention_runtime/config.py:204`）

## 5. 文档入口（当前）

1. 实现总览：`TriAttention_vLLM/docs/interface/IMPLEMENTATION_OVERVIEW.md`
2. HF 对齐状态：`TriAttention_vLLM/docs/interface/HF_ALIGNMENT_STATUS.md`
3. 接手导读：`TriAttention_vLLM/docs/interface/GUIDED_TOUR.md`

历史 debug / 审计流水文档已清理；如需追溯旧设计讨论，请看 `docs/backend/` 与 `docs/archive/`。
