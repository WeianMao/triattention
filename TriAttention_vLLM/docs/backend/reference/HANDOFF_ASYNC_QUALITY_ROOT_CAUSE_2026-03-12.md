# TriAttention vLLM Async 质量问题交接（2026-03-12）

## TL;DR

1. 之前“严重胡言乱语/重复”问题，当前证据基本收敛到 **async 调度路径**（而不是模型/统计文件不匹配本身）。
2. `8B + 对齐 stats + 12k budget` 下：
   - `sync(debug 强制)`：输出可读，仍有分叉但不“炸”；
   - `async(默认)`：可复现后段重复/质量明显变差。
3. 这轮“看起来变好”主要因为测试条件不同（debug+sync），**不是**核心算法 bug 已完全修复。
4. 我本轮代码级实改只有一项：`run_vllm_serve.sh` 的 baseline 防污染清理（不直接提升压缩质量）。

---

## 关键问题来龙去脉

### A. 为什么会出现“我之前跑过同配置还是不对”

这个疑问是正确的。后面复盘确认：

1. 我之前“看起来正常”的对照组用了 `TRIATTN_RUNTIME_DEBUG_MODE=1`，并且开了 `--debug-force-sync-scheduling true`。
2. 这会把路径切到 sync，和默认生产/演示常走的 async 路径不同。
3. 补跑同配置的 async（非 debug）后，问题可复现：后段重复明显。

结论：之前结论偏乐观的根因是“实验条件不一致”（sync vs async），不是你记忆有误。

### B. 8B + 对齐 stats + 12k budget 的结论（你特别关注）

在 OpenClaw-like 样本上（6 个 demo 文档拼接 + W9 任务）：

1. `8B + 对齐 stats + 12k + sync(debug)`：能说人话，结构完整。
2. `8B + 对齐 stats + 12k + async(默认)`：可复现后段重复/退化。

所以同一组超参数下，**调度路径（async/sync）是决定性因素**。

---

## 本轮关键实验（最小集）

> 数据：`/tmp/tri_diag/openclaw_like_dataset.jsonl`  
> 模型：`/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B`  
> stats：`.../deepseek_r1_qwen3_8b_plain_stats.pt`

1. `fullkv baseline`（async 默认）  
   文件：`openclaw_baseline_fullkv_async_default_output.jsonl`

2. `tri 12k`（async 默认，非 debug）  
   文件：`openclaw_tri12k_async_nodebug_output.jsonl`  
   现象：后段重复明显（你关心的问题可复现）

3. `tri 12k`（sync，debug 强制）  
   文件：`openclaw_tri12k_sync_debug_output.jsonl`  
   现象：明显比 async 稳定，仍有分叉但不“炸”

4. `tri 14k`（sync，debug 强制）  
   文件：`openclaw_tri14k_sync_debug_output.jsonl`  
   现象：不触发压缩时可与 baseline 对齐

---

## 我到底改了什么

### 改了（本轮）

1. `TriAttention_vLLM/linxi_dev/run_vllm_serve.sh`
   - `--no-triattention` 分支会清理 `TRIATTN_RUNTIME_*`；
   - 会从 `VLLM_PLUGINS` 中移除 `triattention`；
   - 作用：防止 baseline 被残留插件污染。

### 没改（本轮）

1. 没有新增“修复 async 质量退化”的核心算法补丁。
2. 所以 async 下的问题目前仍在，只是我们现在已定位更清楚。

---

## 被证伪/走弯路的点（保留结论，不保留临时代码）

1. “仅仅是 cascade 开关导致”  
   - 证据不支持：开关后在同路径下并不能解释全部异常。

2. “只是指标变量显示错，真实质量不受影响”  
   - 部分指标确有时序问题，但 async 路径下质量退化是真实存在，不是纯显示问题。

3. “模型和 stats 对齐后问题应自动消失”  
   - 对齐能避免额外灾难，但不能消除 async 路径问题。

---

## 当前建议

1. 对外 demo 先保守：
   - 先用更稳配置（如更高 budget 或 sync 验证路径）确保可演示。
2. 修复优先级：
   - 优先收敛 async 路径的压缩触发/应用一致性问题，再做性能优化。

---

## 清理说明

本轮已清理大量临时产物（debug json/log、中间输出目录、弯路验证脚本/配置），并把易膨胀目录加入 `.gitignore`，避免后续再次污染工作区。
