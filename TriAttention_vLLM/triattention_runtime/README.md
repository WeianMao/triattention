# TriAttention v2

该目录是 Runtime 新工程目录，目标是通过 vLLM 可配置扩展点实现非侵入式接入。

## 当前范围（Phase 1 骨架）

1. `scheduler.py`: 继承 vLLM scheduler，生成按请求维度的压缩触发信号。
2. `worker.py`: 继承 vLLM GPU worker，注入 runner 代理。
3. `runner.py`: 包装原生 model runner，维护请求生命周期状态并消费触发信号。
   - 支持通过 runner hook 执行压缩：`triattention_apply_compression(...)`
4. `config.py`: 从环境变量读取统一配置。
5. `planner.py`: 触发策略（长度触发 + 可选 KV usage 触发）。
6. `state.py`: 请求级状态存储（req_id 唯一键）。
7. `executor.py`: 压缩执行器抽象与默认 hook executor。
8. `effective_len_tracker.py`: scheduler 侧有效缓存长度跟踪。

## 启动方式（class path）

```bash
--worker-cls triattention_runtime.worker.TriAttentionWorker \
--scheduler-cls triattention_runtime.scheduler.TriAttentionScheduler
```

## 环境变量

```bash
TRIATTN_RUNTIME_KV_BUDGET=2048
TRIATTN_RUNTIME_DIVIDE_LENGTH=128
TRIATTN_RUNTIME_PROTECT_PREFILL=true
TRIATTN_RUNTIME_ENABLE_KV_USAGE_TRIGGER=false
TRIATTN_RUNTIME_KV_USAGE_TRIGGER=0.98
TRIATTN_RUNTIME_KV_USAGE_RELEASE=0.90
TRIATTN_RUNTIME_PER_HEAD_SELECTION_SEMANTICS=legacy_layer_local
TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION=false
TRIATTN_RUNTIME_LOG_DECISIONS=true
```

`TRIATTN_RUNTIME_PER_HEAD_SELECTION_SEMANTICS`：

1. `legacy_layer_local`：历史 Runtime 行为（每层独立做 per-head 选择）。
2. `hf_aligned_global_per_head`：HF 对齐行为（跨层聚合后做 per-head 选择，同一组 per-head 索引应用到组内各层）。

## 说明

本阶段不改动 vLLM attention 计算路径，也不直接改写旧版 `triattention/` 代码。

## Runner Hook 协议（Phase 1B）

如果底层 runner 实现了以下方法，runtime runner 会在触发命中时调用：

```python
triattention_apply_compression(req_id: str, signal: CompressionSignal, scheduler_output) -> bool | dict
```

返回值约定：

1. `True/False`
2. `{"applied": bool, "reason": str, "cache_len_after": int | None}`

若 hook 缺失或执行失败，系统自动降级为 no-op，不中断主推理流程。

默认 `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION=false`：

1. 只做压缩计划（plan-only），不直接修改底层 KV cache。
2. 开启后才执行实验性 in-place compaction（仅 Phase 1B 原型能力）。
3. 当前 experimental compaction 已支持多 KV cache group 的基础映射（best-effort）。
4. 若某个 group 无可压缩 attention tensor，会跳过该 group 并继续其他 group。

## Phase 1 冒烟回归

```bash
python tests_runtime/run_smoke.py
```

该脚本不依赖 pytest，适合作为本地最小回归门禁。

## 快速对齐实验（Runtime）

```bash
# 仅跑 Runtime quick 小样本（默认 2 题 x 1 sample）
TriAttention_vLLM/evaluation/scripts/run_hf_alignment_quick.sh

# 先做 dry-run（仅检查命令链路）
TriAttention_vLLM/evaluation/scripts/run_hf_alignment_quick.sh --dry-run

# 跑 Runtime quick 并与 HF 结果做对比报告
TriAttention_vLLM/evaluation/scripts/run_hf_alignment_quick.sh /path/to/hf_merged.jsonl
```
