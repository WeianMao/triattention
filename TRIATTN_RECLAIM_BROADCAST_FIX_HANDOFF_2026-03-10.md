# TriAttention Reclaim Broadcast Fix Handoff (2026-03-10)

## 1. 背景与问题

- 目标：修复长 prefill + reclaim 开启时的崩溃：
  - `ValueError: could not broadcast input array from shape (7376,) into shape (4,)`
- 现象：压缩和回收已经触发，但后续调度分配仍按逻辑长度补大量 block，导致 worker block table 追加语义冲突并崩溃。

## 2. 本次修复（保守最小改动）

- 新增每请求的轻量状态同步：
  - `effective_kv_offset = logical_num_computed - effective_kv_len`
  - `effective_num_computed = logical_num_computed - effective_kv_offset`
- 只在两个边界点使用：
  1. scheduler 每步前为运行中请求刷新 `effective_num_computed`；
  2. `KVCacheManager.allocate_slots` 内部临时用 `effective_num_computed` 做分配计算，结束后恢复原 `num_computed_tokens`。
- 非压缩请求走原路径；不改核心前向/采样逻辑。

## 3. 代码改动

- `TriAttention_vLLM/triattention_runtime/kv_allocation_sync.py` (new)
- `TriAttention_vLLM/triattention_runtime/scheduler.py`
- `TriAttention_vLLM/triattention_runtime/integration_monkeypatch.py`
- `TriAttention_vLLM/tests_runtime/test_kv_allocation_sync.py` (new)
- `TriAttention_vLLM/tests_runtime/test_scheduler.py`
- `TriAttention_vLLM/tests_runtime/test_integration_monkeypatch.py`

## 4. 验证结果

### 4.1 单测

- 命令：
  - `conda run -n dc env PYTHONPATH=TriAttention_vLLM python -m pytest -q TriAttention_vLLM/tests_runtime/test_kv_allocation_sync.py TriAttention_vLLM/tests_runtime/test_scheduler.py TriAttention_vLLM/tests_runtime/test_integration_monkeypatch.py`
  - `conda run -n dc env PYTHONPATH=TriAttention_vLLM python -m pytest -q TriAttention_vLLM/tests_runtime/test_runner.py TriAttention_vLLM/tests_runtime/test_runner_compression_actions.py TriAttention_vLLM/tests_runtime/test_worker_reclaim_sync.py`
- 结果：`16 passed` + `13 passed`

### 4.2 最早完整验证实验（A/B/C 三组）重跑

- 路径：`weian_development/demo_debug/runtime_debug_check/root_cause_check_fullrerun/`

1. A 组（async + reclaim on）
- 输出：`weian_development/demo_debug/runtime_debug_check/root_cause_check_fullrerun/async_run/shard00/run000.jsonl`
- debug：`.../async_debug.jsonl`
- 结果：完成，无 broadcast 崩溃  
- 统计：`SIGNAL=59, APPLY=56, RECLAIM=56, reclaimed_blocks_total=7376`

2. B 组（sync flag + reclaim on）
- 输出：`weian_development/demo_debug/runtime_debug_check/root_cause_check_fullrerun/sync_run/shard00/run000.jsonl`
- debug：`.../sync_debug.jsonl`
- 结果：完成，无 broadcast 崩溃  
- 统计：`SIGNAL=59, APPLY=56, RECLAIM=56, reclaimed_blocks_total=7376`

3. C 组（reclaim off 对照）
- 输出：`weian_development/demo_debug/runtime_debug_check/root_cause_check_fullrerun/noreclaim_run/shard00/run000.jsonl`
- debug：`.../noreclaim_debug.jsonl`
- 结果：完成  
- 统计：`reclaim_count=0, reclaimed_blocks_total=0`（符合预期）

## 5. 性能与风险说明

- 该修复不引入 GPU 同步、无额外 dense tensor 拷贝，额外开销是请求级整数状态维护。
- 改动面集中在“分配边界”，默认逻辑与非压缩路径保持原样。
- 建议后续补一组吞吐对比（同卡同参数）做定量确认。

## 6. 后续建议

1. 用同样的 A/B/C 脚本在同事机器再跑一次（仅换 `CUDA_VISIBLE_DEVICES`）。
2. 如果要做 release 说明，直接引用本文件 + `root_cause_check_fullrerun` 产物目录即可。
3. 如需性能结论，再补一组 `fullkv vs triattention` 延迟/吞吐对比。
