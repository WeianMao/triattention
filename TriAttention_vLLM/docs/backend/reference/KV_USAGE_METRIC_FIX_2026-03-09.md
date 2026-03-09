# TriAttention KV Usage 指标修复说明（2026-03-09）

## 背景

同事反馈：压缩触发后，`vllm:kv_cache_usage_perc` 仍持续上升，看起来像“压缩没有生效”。

我们确认了两件事：

1. 压缩链路本身是触发的（signal -> runner 压缩 -> reclaim 事件）；
2. 指标上报时机有偏差：vLLM 原生先产出 `scheduler_stats`，TriAttention reclaim 在其后执行，导致该步上报值偏向“压缩前”口径。

## 根因

在 monkeypatch 路径中：

- 先调用原始 `Scheduler.update_from_output` 产出 `outputs`（其中已包含 `scheduler_stats.kv_cache_usage`）
- 然后才应用 TriAttention reclaim 事件

因此在该步导出的 usage 指标可能不是 reclaim 后状态。

## 修复策略（保守最小改动）

只修“指标刷新口径”，不改压缩触发逻辑/算法：

1. 在 `integration_monkeypatch.py` 中新增 `_refresh_scheduler_stats_kv_usage(...)`；
2. 在 `_patched_scheduler_update_from_output(...)` 中，压缩事件应用后，用 `self.kv_cache_manager.usage` 刷新 `outputs` 内所有 `scheduler_stats.kv_cache_usage`；
3. 在 `scheduler.py` 的 `TriAttentionScheduler.update_from_output(...)` 路径也做同样刷新，保持两条集成路径一致。

## 代码变更

- `TriAttention_vLLM/triattention_runtime/integration_monkeypatch.py`
- `TriAttention_vLLM/triattention_runtime/scheduler.py`
- `TriAttention_vLLM/tests_runtime/test_integration_monkeypatch.py`（新增）

## 验证

### 单元/回归测试

```bash
PYTHONPATH=TriAttention_vLLM python -m pytest -q \
  TriAttention_vLLM/tests_runtime/test_planner.py \
  TriAttention_vLLM/tests_runtime/test_scheduler.py \
  TriAttention_vLLM/tests_runtime/test_integration_monkeypatch.py
```

结果：`11 passed`

### 修复有效性快速检查

通过最小 fake 调用 `_patched_scheduler_update_from_output` 验证：

- 有压缩事件时，`_apply_compression_events` 会调用；
- 返回输出中的 `scheduler_stats.kv_cache_usage` 被刷新为 `kv_cache_manager.usage`（而非保留旧值）。

## 结论

1. 本次修复目标是“修正 usage 指标口径时机”，已完成；
2. 压缩触发逻辑与阈值逻辑保持不变；
3. 该修复为低侵入、可回滚的小改动，适合继续用于 demo 与协作联调。

## 相关提交

- `ff72aee0` `chore: checkpoint before kv usage metric refresh fix`
- `d6f87e7d` `fix: refresh kv usage metric after triattention reclaim`

