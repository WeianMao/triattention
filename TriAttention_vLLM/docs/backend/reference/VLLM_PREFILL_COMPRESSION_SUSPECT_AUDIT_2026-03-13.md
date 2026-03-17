# vLLM Prefill Compression Suspect Audit

Updated: 2026-03-14
Status: Initial audit completed; see 2026-03-14 findings for validated top suspect
Owner: Codex

Related validated findings:

- `TriAttention_vLLM/docs/backend/reference/VLLM_PREFILL_COMPRESSION_FINDINGS_2026-03-14.md`

## 1. Purpose

本文件的目标不是直接给出根因，而是完成第一阶段的全面梳理：

1. 当前有哪些 vLLM 侧嫌疑点？
2. 哪些嫌疑点在算法逻辑上真的可能导致“压缩后输出坏掉”？
3. 哪些只是实现风格不同、时机不同、或者 HF 模拟不严谨，不足以构成有效嫌疑？
4. 这些嫌疑按优先级应该怎么排？

这份文档只做审计和排序，不把“嫌疑”写成“结论”。

## 2. Audit principle

这次审计遵循下面几个原则：

1. 以 `per-head` 为主。
2. 以 HF 侧这次能正常工作的 prefill probe 作为参考标准。
3. 但不会机械要求 vLLM 与 HF probe 逐行一致。
4. 如果某个差异只是因为 HF probe 本身为了做判别实验而故意简化，那么它不算 vLLM 嫌疑。
5. 只有那些在逻辑上真的可能导致“读错 KV / 选错 KV / 用错位置 / 错算可见长度”的地方，才会进入高优先级嫌疑清单。

## 3. Current evidence baseline

在看嫌疑点之前，先把已经比较清楚的证据放在前面：

### 3.1 已有强证据

1. `vLLM + TriAttention` 的长 prefill 压缩链路，确实能出现明显输出退化。
2. `HF fullkv baseline` 没有出现同等级别的灾难性坏输出。
3. `HF prefill compress_once` 在同一 32B INT4 模型上也没有出现同等级别的坏输出。

### 3.2 这意味着什么

这说明当前最优先的方向，不是继续怀疑“算法一压就坏”，而是优先怀疑：

1. vLLM 的 keep 选择实现
2. vLLM 的 KV 搬运/压缩实现
3. vLLM 的有效长度/位置覆盖实现
4. vLLM 的压缩后回收与后续 decode 语义衔接

## 4. High-priority suspects

以下嫌疑点，都是“逻辑上真的可能导致当前这类坏输出”的。

### 4.1 Suspect A: reclaim 相关 compaction 改变了后续实际读到的 KV 语义

相关文件：

1. `TriAttention_vLLM/triattention_runtime/layout_engine.py`
2. `TriAttention_vLLM/triattention_runtime/kv_compaction.py`

关键事实：

1. `layout_engine.py` 中，只要开启 block reclaim 且压缩后 block 数下降，就会把 `preserve_dropped_tokens` 设成 `False`。
2. `kv_compaction.py` 中，`preserve_dropped_tokens=False` 会走 fill-hole 路径。
3. fill-hole 路径的注释已经明确写了：
   - 它为了少搬运数据，允许 prefix 内部发生 permutation。

为什么这是高嫌疑：

1. 单纯的 KV permutation，如果后续 attention 读到的还是同一组配对正确的 `(K, V)`，数学上不一定会导致结果出错。
2. 真正危险的是：fill-hole + reclaim 之后，系统后续实际读到的 KV 集合、K/V 对应关系、追加写入位置，已经不再等于“我们以为保留下来的那一组”。
3. 也就是说，这里的重点不是“顺序本身”，而是“顺序变化是否进一步引发了语义漂移”。
4. 这与 HF probe 当前更接近的“保守单次压缩”仍然存在本质区别，所以它依旧是高嫌疑，但要按更严格的标准去验证。

当前判断：

- 这是目前最强的高嫌疑之一。

### 4.2 Suspect B: effective seq base / position delta 覆盖逻辑与物理 compaction 结果不完全对齐

相关文件：

1. `TriAttention_vLLM/triattention_runtime/effective_overrides.py`
2. `TriAttention_vLLM/triattention_runtime/input_patch_vllm_backend.py`
3. `TriAttention_vLLM/triattention_runtime/runner_output_bridge.py`

关键事实：

1. vLLM runtime 压缩后，没有直接天然理解“当前请求的有效历史长度已经变了”。
2. 现在的做法是额外构造 effective base 和 position delta，再去 patch vLLM backend 的 `seq_lens` 和 `positions`。
3. 这条链路依赖多个中间状态：
   - request state
   - scheduler output
   - compression event
   - active override window

为什么这是高嫌疑：

1. 只要 effective length 和真实保留的 KV 不一致，attention 就可能看错上下文窗口。
2. 只要 position delta 和真实“当前 token 应该对应的绝对位置”不一致，RoPE 语义就可能错。
3. 这类错误很符合“压缩一触发，输出开始越来越怪”的现象。
4. 这也属于会直接改模型可见语义的错误，不是轻微时机偏差。

当前判断：

- 这是目前最强的高嫌疑之一。

### 4.3 Suspect C: paged-streaming selector 与 dense/HF selector 的 keep 结果可能并不等价

相关文件：

1. `TriAttention_vLLM/triattention_runtime/selector_hf.py`
2. `TriAttention_vLLM/triattention_runtime/selection_planner.py`

关键事实：

1. vLLM 跑的不是简单 dense 路径，而是 paged-streaming 选择路径。
2. `selector_hf.py` 里有专门的 debug compare 开关：
   - `TRIATTN_DEBUG_COMPARE_PAGED_DENSE_KEEP=1`
3. 这说明代码本身也承认：paged keep 和 dense keep 是否一致，是一个需要被专门核验的问题。

为什么这是高嫌疑：

1. 如果 selector 选错了 token，后面 compaction、reclaim、override 即使全都做对，也还是会坏。
2. 这是直接改变“哪些 token 进入 attention”的问题，影响最本质。
3. HF probe 能工作，至少说明“算法在一个更直接的 dense 语义下可以工作”；那么 paged selector 是否等价，就必须严肃怀疑。

当前判断：

- 这是第三个高优先级嫌疑。

## 5. Medium-priority suspects

这些点有一定逻辑风险，但目前不如前三个直接。

### 5.1 Suspect D: reclaim 后 scheduler / worker 的 block 视图与后续追加路径可能不完全一致

相关文件：

1. `TriAttention_vLLM/triattention_runtime/scheduler.py`
2. `TriAttention_vLLM/triattention_runtime/runner.py`
3. `TriAttention_vLLM/triattention_runtime/worker_reclaim_sync.py`

关键事实：

1. worker 侧会先更新 `num_blocks_per_row` 和 `req_state.block_ids`。
2. scheduler 侧随后也会根据 compression event 去更新 `req_to_blocks`。
3. `runner.py` 里还有一层 `_patch_scheduler_output_for_compressed_reqs()`，会在必要时直接 trim `new_block_ids`，避免后续 append overflow。

为什么可疑：

1. 这说明压缩后“谁认为当前还剩多少 block、下一批新 token 应该往哪里追加”这件事，本身就是一条需要修补的链路。
2. 如果这条链路有轻微不一致，往往不会在压缩当下立刻表现为乱码，而是会在继续 decode、继续追加 KV 一段之后才逐渐出问题。
3. 这非常符合“刚压完看起来还行，后面跑一段开始重复”的现象形态。

为什么不是最高优先级：

1. 它更偏向“压缩后继续追加”的状态同步问题。
2. 当前最本质的高嫌疑，仍然是 keep/order/position 这三类直接改语义的点。

### 5.2 Suspect E: selection fallback / mode fallback 在某些场景 silently 降级

相关文件：

1. `TriAttention_vLLM/triattention_runtime/selection_planner.py`
2. `TriAttention_vLLM/triattention_runtime/selector_hf.py`

为什么可疑：

1. 当前 `per_head` 路径有 group-global 语义，也有 layer-local/普通 fallback。
2. 如果某一步因为输入形态、异常、或者 selector 返回 `None`，悄悄走到了 fallback 路径，可能就不再是预期算法语义。

为什么不是最高优先级：

1. 这更像“为什么选错模式”。
2. 它的重要性很高，但比起“顺序直接被打乱”或“位置直接被改错”，仍稍弱一层。

### 5.3 Suspect F: trigger / effective token accounting 与真实 runtime 状态有偏差

相关文件：

1. `TriAttention_vLLM/triattention_runtime/scheduler.py`
2. `TriAttention_vLLM/triattention_runtime/hook_runtime_context.py`

为什么可疑：

1. runtime 会基于 `estimated_cache_len`、`scheduled_tokens`、`prefill_len` 等估计值决定何时压缩。
2. 这些量如果偏差太大，可能会让压缩时机与真实状态错开。

为什么不是最高优先级：

1. 这更容易导致“压缩时机不理想”或“指标看起来怪”。
2. 但单靠它本身，不太容易解释“内容明显胡言乱语”。
3. 除非它进一步把错误输入喂给了后续 override / reclaim 链路，否则更像次级因素。

### 5.4 Suspect G: async / side-channel 事件传递存在时序或覆盖问题

相关文件：

1. `TriAttention_vLLM/triattention_runtime/runner_output_bridge.py`
2. `TriAttention_vLLM/triattention_runtime/scheduler.py`

为什么可疑：

1. compression event 既可能挂在 output 上，也可能挂在 scheduler_output 上。
2. 这条链路如果有时序问题，会导致状态刷新不一致。

为什么现在不是最高优先级：

1. 历史上它确实出过问题，但当前阶段不能只因为“以前出过问题”就默认它还是主因。
2. 结合目前 HF 判别实验，更像要先把“选择/压缩/位置语义”这几个更本质的点查清。

## 6. Low-priority or currently non-actionable differences

以下内容目前不应被当成高价值嫌疑点。

### 6.1 HF probe 的简化本身

例如：

1. HF 只做了一次手动 prefill 压缩。
2. HF 没完整复刻 vLLM 的在线 runtime。
3. HF 某些 chat/template 细节没有做到完全线上等价。

这些是实验边界，不是 vLLM bug 证据。

### 6.2 纯“看起来不一样”但不直接改变语义的实现风格差异

例如：

1. 某个中间变量名称不同
2. 某个状态缓存时机不同
3. 某个 helper 切分方式不同

只要它不改变：

1. keep 集合
2. keep 顺序
3. 有效长度
4. 位置编码语义

它就不该进入高优先级嫌疑。

## 7. Ranked suspect list

当前建议的排查优先级如下：

1. `A` fill-hole compaction 在 reclaim 下破坏保序
2. `B` effective length / position override 与物理 compaction 语义不对齐
3. `C` paged-streaming selector 与 dense/HF selector 不等价
4. `D` reclaim 后 scheduler / worker / append 路径状态不同步
5. `E` selection fallback / mode fallback 导致语义 silently 降级
6. `F` trigger / effective token accounting 偏差
7. `G` async / side-channel 时序问题

## 8. Suggested validation order for phase 2

第二阶段验证时，建议按下面顺序做：

1. 先验证 `A`
2. 再验证 `B`
3. 再验证 `C`
4. 再验证 `D`
5. 最后再看 `E/F/G`

原因很简单：

1. `A/B/C` 都会直接改变“模型到底看到了哪些 token、以什么顺序看、以什么位置看”。
2. 这些点最容易直接把输出语义搞坏。
3. `D/E/F` 更像链路放大器、触发器、或同步问题，重要但优先级略低。

## 9. What to keep in mind for phase 2

进入验证阶段前，需要明确几条纪律：

1. 当前这些都还只是嫌疑，不是结论。
2. 不能看到第一个高嫌疑点就停下来。
3. 必须靠 debug-only 实验去验证。
4. debug 代码不能影响非 debug 模式的默认行为和效率。
5. 如果某个修正只在 HF 简化实验下成立，但无法解释 vLLM 真实坏输出，那它就不是最终根因。

## 10. Current bottom line

截至本次审计结束，可以比较稳地说：

1. 当前方向已经从“算法是否本身不 work”切换到“vLLM 哪个实现环节出了问题”。
2. 目前最值得优先怀疑的，不是纯采样参数，也不是单纯 trigger 时机，而是：
   - keep 选得对不对
   - compaction 后顺序对不对
   - effective length / position 覆盖是否真的和物理状态一致
3. 下一阶段应该进入验证实验，而不是直接修。
