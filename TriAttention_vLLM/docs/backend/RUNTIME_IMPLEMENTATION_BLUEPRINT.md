# V2 实现蓝图（代码映射）

- 更新时间：2026-02-13
- 状态：Active
- 适用范围：vLLM 0.15.x（V1 Engine）

---

## 1. 目录映射

新增目录：

`TriAttention_vLLM/triattention_runtime/`

模块映射：

1. `config.py`
- 职责：统一读取/校验 V2 环境配置。
- 输出：`TriAttentionRuntimeConfig`。

2. `planner.py`
- 职责：根据长度/KV usage 计算压缩触发信号。
- 输出：`CompressionSignal`。

3. `scheduler.py`
- 职责：继承 vLLM Scheduler，给 `SchedulerOutput` 挂载 `triattention_signals`。
- 输入：vLLM 原生调度状态 + `kv_cache_manager.usage`。

4. `runner.py`
- 职责：runner 代理层，消费 `triattention_signals`，维护 req_id 状态。
- 约束：不改动原生 forward 主路径。
- 补充：Phase 1B 通过 `executor.py` 调用可选 runner hook 执行压缩。

5. `worker.py`
- 职责：继承 vLLM GPU Worker，在 `init_device()` 后注入 runner 代理。

6. `state.py`
- 职责：请求状态存储与生命周期 API。
- 键：`req_id`（唯一键）。

7. `executor.py`
- 职责：压缩执行抽象层，默认走 `triattention_apply_compression` hook。
- 目标：在不侵入 vLLM runner 主逻辑的前提下，完成执行闭环与失败降级。

8. `hook_impl.py`
- 职责：为 base runner 安装默认 hook，并处理 plan-only / experimental compaction 路径。

9. `kv_compaction.py`
- 职责：提供 request 级 keep-index 构造与 in-place KV compaction 原型。
- 说明：当前为 Phase 1B 原型，默认不启用真实 compaction。

11. `V2_RECLAIM_STRATEGY.md`
- 职责：定义“物理回收”从逻辑压缩到 block/page 回收的半侵入继承层落地路径。
- 说明：该文档是回收阶段实现与评审的主入口。

10. `effective_len_tracker.py`
- 职责：维护 scheduler 侧“有效缓存长度”语义，避免仅使用 `num_computed_tokens` 带来触发偏差。

---

## 2. 当前数据流

1. Scheduler `schedule()` 先执行原生调度。
2. TriAttention scheduler 通过 effective len tracker 估计请求有效缓存长度，并结合可选 KV usage 生成信号：
   `triattention_signals: dict[req_id, CompressionSignal]`
3. 信号附加到 `SchedulerOutput`，由 vLLM 正常下发到 worker。
4. Runner proxy 在 `execute_model()` 前消费信号，更新请求状态。
5. Runner proxy 调用原生 runner 执行推理。
6. Runner 在 `execute_model()` 优先挂载压缩事件到 `ModelRunnerOutput`（side-channel，`sample_tokens()` 仅兼容回退）。
7. Scheduler 在 `update_from_output()` 消费事件并更新 effective len tracker；实验开关开启时同时应用 `block_reclaim` 执行 tail block 回收。

---

## 3. 与后续压缩执行的衔接点

当前尚未接入真实 KV gather/scatter，后续在 `runner.py` 追加：

1. 读取 signal 命中请求（`should_compress=true`）。
2. 调用压缩执行器（Phase 1B 已接入 hook executor）。
3. 默认：plan-only（不改 KV）；实验开关开启后执行 in-place compaction。
4. 成功：`state.mark_compressed(...)`。
5. 失败：回退 no-op，记录结构化日志。

---

## 4. 测试映射

目录：`TriAttention_vLLM/tests_runtime/`

1. `test_config.py`：配置解析与参数约束。
2. `test_planner.py`：长度触发、KV usage 触发、滞回行为。
3. `test_state.py`：request 生命周期状态正确性。
4. `test_executor.py`：hook executor 的输入输出契约。
5. `test_runner.py`：runner 触发/执行/降级闭环。
6. `test_kv_compaction.py`：KV compaction 内核原型。
7. `test_hook_impl.py`：默认 hook 安装与 plan-only/experimental 行为。
8. `test_effective_len_tracker.py`：有效缓存长度语义校验。
