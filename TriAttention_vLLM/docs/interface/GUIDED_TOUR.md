# TriAttention_vLLM 接手导读（当前默认版本）

- 更新时间：2026-02-25
- 状态：Active

## 目标（15~30 分钟内接手）

你需要先搞清楚三件事：
1. 当前默认入口怎么跑
2. HF 对齐状态到哪一步
3. 核心代码模块各自负责什么

## Step 1（5 分钟）先看目标与当前实现

1. `TriAttention_vLLM/docs/interface/PROJECT_GOAL.md`
2. `TriAttention_vLLM/docs/interface/IMPLEMENTATION_OVERVIEW.md`

重点确认：
1. 当前默认目标模式是 `per_head`
2. `per_layer_per_head` 需要支持（代码路径已修关键风险）
3. 内部实现目录已重命名为 `triattention_runtime/`，`triattention_v2/` 仅作兼容导入层

## Step 2（5 分钟）看 HF 对齐状态

阅读：`TriAttention_vLLM/docs/interface/HF_ALIGNMENT_STATUS.md`

重点确认：
1. 当前对齐参照脚本是哪一个
2. `per_head` 全量 anchor 的结果与结论
3. 哪些项当前不作为阻塞（脚本未用到的 ablation 参数）

## Step 3（5~10 分钟）直接看代码入口

1. Dispatch：
   - `TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py`
   - 默认配置：`TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24.yaml`
2. Runner：
   - `TriAttention_vLLM/evaluation/runner/vllm_triattention_runner.py`
   - 兼容实现：`TriAttention_vLLM/evaluation/runner/vllm_triattention_v2_runner.py`
3. 集成接入与主链路：
   - `TriAttention_vLLM/triattention_runtime/integration_monkeypatch.py`
   - `TriAttention_vLLM/triattention_runtime/scheduler.py`
   - `TriAttention_vLLM/triattention_runtime/worker.py`
   - `TriAttention_vLLM/triattention_runtime/runner.py`

## Step 4（按任务深入）

1. 对齐问题先看：
   - `TriAttention_vLLM/triattention_runtime/selector_hf.py`
   - `TriAttention_vLLM/triattention_runtime/selection_planner.py`
2. 布局/回收问题先看：
   - `TriAttention_vLLM/triattention_runtime/layout_engine.py`
   - `TriAttention_vLLM/triattention_runtime/kv_compaction.py`
   - `TriAttention_vLLM/triattention_runtime/worker_reclaim_sync.py`
3. 运行时语义/映射问题先看：
   - `TriAttention_vLLM/triattention_runtime/effective_overrides.py`
   - `TriAttention_vLLM/triattention_runtime/input_adapter.py`
   - `TriAttention_vLLM/triattention_runtime/input_patch_vllm_backend.py`
