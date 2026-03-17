# TriAttention 压缩触发后输出混乱问题 — 完整记录

**日期**: 2026-02-22 ~ 2026-03-10（跨多个 session 累积定位）
**优先级**: P0
**状态**: 核心根因已定位并修复，残留问题仍在收敛中

---

## 一、问题现象

触发 KV cache 压缩后，模型输出出现严重质量退化：

| 症状 | 描述 |
|------|------|
| **乱码** | 输出包含无意义 token 序列、随机字符组合 |
| **无限重复** | 同一段文本/pattern 反复输出直到 max_length |
| **异常长输出** | `output_tokens` 接近 `max_length`，正常应远短于此 |
| **准确率断崖** | 评测精度从 ~45% 跌至 ~10%（AIME24 sample8） |

不触发压缩时（fullKV baseline），输出完全正常。

---

## 二、根因分析

### 2.1 主因：Zero-Tailing 污染 Softmax（已修复）

**OPEN_ISSUES #1.3**

旧实现在 in-place compaction 后将被丢弃 token 的 KV 位置**置零**，但**不缩短逻辑序列长度** (`total_tokens`)。

```
压缩前: [tok_0, tok_1, tok_2, tok_3, tok_4, tok_5, tok_6, tok_7]  (total_tokens=8)
                        ↓ 保留 {0,1,4,5}, 丢弃 {2,3,6,7}
旧实现: [tok_0, tok_1, 0000, 0000, tok_4, tok_5, 0000, 0000]  (total_tokens 仍=8)
```

**后果**：attention softmax 仍按 total_tokens=8 计算，4 个零 K 向量与 query 的点积 ≈ 0，对应 softmax 权重 ≈ `exp(0) = 1`。大量无效项抬高分母，有效 token 的注意力权重被稀释 → 输出随机化。

**修复**（`kv_compaction.py`）：改为全量 permutation（`[kept..., dropped...]`），不置零尾部。dropped token 的原始 KV 数据保留在物理位置上，但后续 attention 不再读取它们（通过 effective length 控制）。

```python
# kv_compaction.py:366-368
"""
We intentionally avoid writing zero tails while request logical length is
still `total_tokens`, otherwise dropped entries continue participating in
attention softmax as zero-K tokens and corrupt generation quality.
"""
```

### 2.2 次因：Fill-in-Place 打乱时序（残留问题）

**OPEN_ISSUES #1.6**

当启用物理 block 回收 (`enable_experimental_block_reclaim=true`)，走 `preserve_dropped_tokens=False` 快路径时：

```
原始顺序: [tok_0, tok_1, tok_2, tok_3, tok_4, tok_5]
                        ↓ 保留 {0,3,4}, 丢弃 {1,2,5}
fill-hole: [tok_0, tok_4, tok_3, -, -, -]   ← tok_3 和 tok_4 顺序反了！
```

保留 token 的时间顺序被打乱 → attention 的位置-内容对应关系失真 → 类似乱码症状。

**当前状态**：已回退到保序前缀写入（correctness stopgap），搬运量 ≈ `keep_count`，性能待优化。

### 2.3 辅因：seq_len / effective_len 不同步（残留问题）

**OPEN_ISSUES #1.7**

vLLM V1 用同一个 `num_computed_tokens` 同时构造：
- `positions`（绝对位置，用于 RoPE）
- `seq_lens`（attention 上下文窗口长度）

压缩后需要 positions 继续单调增长，但 seq_lens 应按压缩后 effective length 计算。两者混用导致：
- attention 仍按未压缩长度读取（读到无效尾部）
- 或位置 id 被错误回退（语义错误）

**当前修复**：`gpu_seq_len_patch.py` 在 decode 热路径做 override（过渡方案），长期需重构为持久状态 + 薄适配层。

---

## 三、修复时间线

| 日期 | 事件 | Commit |
|------|------|--------|
| 2026-02-01 | 发现压缩触发逻辑 bug（AND vs threshold），修复 | — |
| 2026-02-22 | 定位 zero-tailing 为主因，实现全量 permutation 修复 | — |
| 2026-02-22 | 发现 fill-in-place 乱序问题，回退到保序写入 | — |
| 2026-02-22 | 定位 seq_len 双重语义冲突，实现 gpu_seq_len_patch | — |
| 2026-02-23 | 达成方案共识：低搬运 fill-hole + 持久状态适配层 | — |
| 2026-03-09 | 修复 V1 async 路径事件丢失（4 个子 bug） | `6c56438d`, `d4f275fb` |
| 2026-03-09 | 修复 KV usage 指标口径 | `d6f87e7d` |
| 2026-03-10 | protect_prefill=false + budget=8192，长文档 demo 跑通 | `46a0746a`, `750b4dbc` |

---

## 四、当前验证状态

### 4.1 已验证正常

| 链路 | 状态 |
|------|------|
| Scheduler 信号触发（每 128 步） | ✅ |
| Runner 消费信号并执行压缩 | ✅ |
| Triton kernel KV compaction | ✅ |
| Worker block table reclaim | ✅ |
| V1 async 事件 side-channel | ✅ |
| Scheduler free_blocks | ✅ |
| KV usage 指标刷新 | ✅ |
| 长生成（4096 tokens, 16 次压缩）不崩溃 | ✅ |
| 长文档 demo（30KB, budget=8192）KV 稳定在 ~13% | ✅ |

### 4.2 残留问题

| 问题 | 优先级 | 状态 |
|------|--------|------|
| 保序前缀写入搬运量过大（≈keep_count） | P1 | Open |
| seq_len/effective_len 补丁仍在 decode 热路径 | P0 | In Progress |
| HF 等价性验证未完成 | P0 | Open |
| strict reclaim 仍可能出现 runaway | P0 | Open |

---

## 五、核心代码位置

| 文件 | 职责 |
|------|------|
| `triattention_runtime/kv_compaction.py` | KV 压缩核心（permutation 逻辑、fill-hole 逻辑） |
| `triattention_runtime/hook_impl.py` | 压缩 hook 执行入口（score → select → compact） |
| `triattention_runtime/gpu_seq_len_patch.py` | seq_lens/slot_mapping override 补丁 |
| `triattention_runtime/runner_compression_actions.py` | Runner 侧压缩执行与去重 |
| `triattention_runtime/worker_reclaim_sync.py` | Worker 侧 block table 回收 |
| `triattention_runtime/state.py` | 请求级压缩状态管理 |
| `triattention_runtime/integration_monkeypatch.py` | vLLM 集成 monkeypatch 层 |

---

## 六、复现方法

### 6.1 触发输出混乱（旧代码）

1. 使用 `preserve_dropped_tokens=False`（或旧版 zero-tailing 实现）
2. 设置 `KV_BUDGET` 远小于输入 + 生成长度（如 2048）
3. 发送足够长的 prompt 使 KV 超过 budget，触发压缩
4. 观察输出：乱码 / 重复 / 异常长

### 6.2 验证修复

```bash
# 启动 TriAttention 模式
export ENABLE_TRIATTENTION=true
export TRIATTN_RUNTIME_KV_BUDGET=8192
export TRIATTN_RUNTIME_PROTECT_PREFILL=false
# 使用 preserve_dropped_tokens=True（默认）
# 观察压缩日志和输出质量
```

---

## 七、经验教训

1. **KV cache 压缩不能只改数据、不改长度语义**：任何在 attention 视野内的位置都会参与 softmax，zero-K 不等于"不存在"。
2. **in-place 操作的隐含假设极易出错**：物理位置、逻辑长度、绝对位置三者的一致性需要显式保证。
3. **vLLM V1 async 架构增加了事件传递复杂度**：`execute_model()` 返回 None，side-channel 是必须的。
4. **保序是硬约束**：attention 对 KV 的时间顺序有隐式依赖，乱序等同于语义破坏。
5. **问题通常是多因叠加**：zero-tailing + 乱序 + seq_len 不同步，需要逐一排除。
