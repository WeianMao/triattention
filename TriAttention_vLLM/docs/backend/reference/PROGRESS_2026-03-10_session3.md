# Progress: TriAttention V1 Async Fix & Demo A/B Testing (2026-03-10 Session 3)

## 1. 本次目标

修复 vLLM V1 async path 下压缩事件丢失问题，并完成 TriAttention vs Baseline 的 A/B 对比测试。

## 2. 已完成的修复

### 2.1 `**kwargs` 前向兼容修复

**文件**: `triattention_runtime/integration_monkeypatch.py`

**问题**: upstream commit `cf4d34a1` 的 `_patched_kv_cache_allocate_slots` 硬编码了参数列表，但本地 vLLM 版本有额外的 `num_external_computed_tokens` 参数，导致 `TypeError`。

**修复**: 将函数签名改为 `**kwargs`，所有内部调用统一使用 `**kwargs` 转发，确保兼容任何 vLLM allocate_slots 签名。

**commit**: `0498acd8`

### 2.2 V1 Async 事件 Side-Channel（已验证生效）

**机制**: vLLM V1 的 `execute_model()` 返回 `None`（async 模式），压缩事件无法附着到 output 上。修复方案将事件附着到 `scheduler_output` 对象上（同一 Python 对象贯穿 schedule → execute → update_from_output 全程）。

**验证结果**: 日志确认事件通过 `scheduler_output` 成功传递：
```
attach_events: output=None, attached 1 events (1 applied) to scheduler_output
TriAttention update_from_output: received 1 events (1 applied) via scheduler_output
```

### 2.3 `protect_prefill=false` 长 prefill 适配

**问题**: 30KB 文档 prefill (~15K tokens) + `protect_prefill=True`（默认）+ `kv_budget=2048` → `prefill_exceeds_budget`，压缩永远无法执行。

**根因**: `protect_prefill=True` 会保护所有 prefill token 不被压缩，但 prefill 长度已远超 budget，没有空间留给 decode token。

**修复**: 在 `swap_backend.sh` 中设置 `TRIATTN_RUNTIME_PROTECT_PREFILL=false`。

**commit**: `46a0746a`

### 2.4 KV_BUDGET 提升至 8192

**问题**: budget=2048 对于 30KB 文档场景过小，压缩后质量可能受损。

**修复**: 默认 `KV_BUDGET=8192`。

**commit**: `750b4dbc`

## 3. 排查过程：0 applied 问题

### 3.1 现象

所有压缩事件均为 `0 applied`——信号触发但压缩从未执行。

### 3.2 排查步骤

1. 确认 side-channel 修复生效（事件通过 scheduler_output 传递 ✓）
2. 确认 runner proxy 安装成功（`TriAttentionWorker lazily injected runner proxy` ✓）
3. 确认环境变量正确（STATS_PATH、KV_BUDGET、REQUIRE_TRITON_SCORING 等 ✓）
4. 发现 skip reason 日志在 DEBUG 级别不可见——添加临时 INFO 级别日志
5. 揭示真实 skip reason 分布：
   - **456 `prefill_exceeds_budget`**（主因）
   - **458 `batch_queue_dedup`**（级联）
   - **1 `under_budget`**
   - **1 `req_state_not_found`**

### 3.3 根因

`protect_prefill=True` + prefill (~15K) > budget (2048) → `build_keep_token_indices` 返回 None → 每次压缩都 skip。由于压缩从未成功，effective length 不重置，threshold 条件每步都满足，导致每步都触发信号但都 skip。

## 4. 验证结果

### 4.1 TriAttention 模式（protect_prefill=false, budget=2048 测试）

```
Step  5: compression applied before=6144 → after=2048, reclaimed_blocks=384
Step 133: compression applied before=2175 → after=2048, reclaimed_blocks=254
Step 261: compression applied before=2175 → after=2048, reclaimed_blocks=8
```

- KV cache usage 稳定在 **~13%**（不再单调递增）
- 生成吞吐 **~29 tok/s**（无 stutter）
- 压缩周期 = `divide_length=128` tokens（符合预期）

### 4.2 Baseline 模式（KV 16K 物理上限）

openclaw 发送 30KB 文档 → 多轮对话积累上下文：

```
KV cache usage: 97.0% → 99.8% → preempt
Running: 1 → Running: 0, Waiting: 1
throughput: 28.6 tok/s → 0.0 tok/s
```

- KV 达到物理上限后请求被 preempt
- 进入 preempt → re-prefill → preempt 死循环
- **Baseline 在长上下文场景下无法完成任务**（符合预期）

## 5. 当前状态

| 组件 | 状态 |
|------|------|
| vLLM 后端 | 运行中，baseline 模式，端口 8002 |
| Demo 代理 | 运行中，端口 8010 |
| SSH 隧道 | 本地 8010 → 远端 8010 |
| 模型 | Qwen3-32B-INT4, max_model_len=32768, gpu_util=0.95 |

### swap_backend.sh 已知问题

`_current_mode()` 检测逻辑有 bug：检查 `/proc/PID/environ` 中的 `ENABLE_TRIATTENTION=true`，但 triattention 模式下 `run_vllm_serve.sh` 不 export 这个变量（使用 shell 变量而非环境变量），导致 triattention 被误报为 baseline。

## 6. 待完成

1. 切换到 triattention (budget=8192) 模式，跑 openclaw 完整对比测试
2. 收集 TriAttention 模式下的完整输出，与 baseline 的失败结果对比
3. 清理临时 INFO 诊断日志（`runner_compression_actions.py`）
4. 修复 `_current_mode()` 检测 bug
5. 可选：补充吞吐/延迟量化对比数据

## 7. 关键文件改动

| 文件 | 改动 |
|------|------|
| `integration_monkeypatch.py` | `**kwargs` 前向兼容 |
| `runner_compression_actions.py` | 临时 INFO 诊断日志 |
| `swap_backend.sh` | `PROTECT_PREFILL=false`, `KV_BUDGET=8192` |

## 8. Commits (本次 session)

| Hash | 说明 |
|------|------|
| `0498acd8` | fix: use **kwargs in patched allocate_slots |
| `baec54cd` | debug: add INFO-level skip reason logs |
| `46a0746a` | fix: disable protect_prefill for long-prefill demo |
| `750b4dbc` | fix: increase default KV_BUDGET to 8192 |
