# vLLM Prefill Compression Findings

Updated: 2026-03-14
Status: High-confidence structural finding, not yet repaired
Owner: Codex

## 1. Current milestone

截至这一步，已有证据已经明显从“怀疑算法”转向了“怀疑 vLLM runtime 接口兼容层”。

更准确地说：

1. `HF prefill compression probe` 支持“算法本身不是一压就坏”。
2. `vLLM` 这边当前最强的根因方向，不是 selector 细节，也不是单纯 reclaim，而是：
3. `effective override` 这条链在当前真实执行的 runner 路径上，没有正确接到位。

这还不是最终修复，但已经是一个新的 milestone。

## 2. What we verified

### 2.1 当前环境默认跑的是哪条 runner 路径

当前 `trivllm` 环境里的 vLLM 默认配置是：

- `VLLM_USE_V2_MODEL_RUNNER = False`

这意味着 worker 默认构造的是：

- `vllm.v1.worker.gpu_model_runner.GPUModelRunner`

而不是：

- `vllm.v1.worker.gpu.model_runner.GPUModelRunner`

这两条路径都属于 vLLM V1 engine，但它们的内部接口不同。

## 2.2 两条 runner 路径的接口差异

### 默认实际使用的路径

`vllm.v1.worker.gpu_model_runner.GPUModelRunner`

其核心请求状态更接近：

1. `base_runner.requests`
2. `base_runner.input_batch`
3. `base_runner.input_batch.req_id_to_index`

它没有当前 TriAttention override 链路假设的 `base_runner.req_states`。

### 当前 override / patch 代码假设的路径

`vllm.v1.worker.gpu.model_runner.GPUModelRunner`

这条路径才有：

1. `base_runner.req_states`
2. `req_states.req_id_to_index`
3. `prepare_pos_seq_lens(...)`
4. `gpu.block_table.BlockTables.compute_slot_mappings(...)`

## 2.3 install_runtime_input_patch 当前实际 patch 到了哪里

已验证 `install_runtime_input_patch_hooks()` 的行为是：

1. 会 patch `vllm.v1.worker.gpu.model_runner.prepare_pos_seq_lens`
2. 会 patch `vllm.v1.worker.gpu.block_table.BlockTables.compute_slot_mappings`
3. 但不会 patch 当前默认实际使用的：
   - `vllm.v1.worker.gpu_model_runner.GPUModelRunner._prepare_inputs`
   - `vllm.v1.worker.block_table.MultiGroupBlockTable.compute_slot_mapping`

也就是说：

`当前 input patch 安装成功，并不代表它真的 patch 到了当前真实执行路径。`

## 2.4 TriAttentionModelRunner 当前还会主动跳过 patch 安装

`TriAttentionModelRunner._ensure_runtime_input_patch_if_needed()` 里有这样一个保护条件：

1. 如果 `base_runner.req_states is None`
2. 就直接 return
3. 不安装 runtime input patch

而当前默认 V1 runner 正好没有 `req_states`

所以结果是：

`即使 need_effective_overrides=True，patch 安装也会被跳过。`

## 2.5 effective override builder 在当前默认 V1 runner 上会直接变成空操作

已做过一个最小复现实验：

1. 给一个 V1-style base runner：
   - 有 `requests`
   - 有 `input_batch.req_id_to_index`
   - 没有 `req_states`
2. 给一个 active compressed request state
3. 调用 `build_effective_sparse_overrides(...)`

结果返回：

- `(None, None, None, 0)`

这说明：

当前 override builder 不是“算错了”，而是“在默认 V1 runner 结构上直接退化成空 override”。

## 2.6 强行切到 V2 model runner 会触发另一个独立兼容性问题

为了做判别，又跑了一组：

- `VLLM_USE_V2_MODEL_RUNNER=1`

结果不是恢复正常，而是更早崩在：

- `scheduler.update_from_output`
- `KeyError: req_id not in model_runner_output.req_id_to_index`

这说明当前 TriAttention runtime 不是“同时兼容两套 runner 路径”。

它现在的状态更像是：

1. 默认 V1 路径下，override 没有真正接上
2. 强切 V2 路径下，又暴露出另一条 `req_id_to_index` / async 输出兼容性 bug

## 2.7 debug-only V1 compatibility probe 的结果

为了验证“是不是这条接口错配真的会影响当前坏输出”，又做了一步严格隔离的
debug-only probe：

1. 不修改默认行为
2. 只在显式打开 `TRIATTN_DEBUG_ENABLE_V1_OVERRIDE_PATH=1` 时生效
3. 让 override builder 可以从当前 V1 runner 的
   `input_batch.req_id_to_index` 取行号
4. 同时把最小 patch 挂到默认真实执行路径
   `vllm.v1.worker.gpu_model_runner.GPUModelRunner._prepare_inputs`

这个 probe 的关键结果是：

### 旧 trace（修补前）

在同一条长 case 上，压缩触发后虽然：

1. `need_effective_overrides = true`
2. `active_compressed = true`

但实际看到的是：

1. `req_id_to_index_present = false`
2. `prepare_effective_input_overrides -> seq_base_count = 0`
3. `prepare_effective_input_overrides -> pos_delta_count = 0`

也就是：

`override 逻辑被判定需要，但真正构建出来的是空。`

### 新 trace（debug-only V1 probe）

打开 debug V1 probe 后，在同一条 case 上，压缩触发后看到的是：

1. `req_index_source = "input_batch"`
2. `req_id_to_index_present = true`
3. `effective_override_row` 持续出现
4. `seq_base_count = 1`
5. `pos_delta_count = 1`
6. `single_seq_base / single_pos_delta` 会随着 decode 持续更新

也就是说：

`之前空掉的 override，在默认 V1 路径上已经真正非空并持续生效了。`

这一步非常关键，因为它说明：

1. 之前那条“接口错配”线索不是假线索
2. 它不是只影响静态结构，而是确实改变了 runtime 是否能构建 override

### 当前局限

这两条 debug probe 在生成阶段后，又出现了一个新的次级现象：

1. request 的生成阶段已经基本跑完
2. 但 engine 进入了长时间 `no scheduled items` 的空转收尾
3. 因此这轮还没有拿到最终落盘的 json 输出文件

所以这一步给出的结论是：

1. 已经高置信度证明了“默认 V1 路径下，override 接口错配是真问题”
2. 但还没有完成“修补后最终输出完全恢复正常”的最后闭环

## 3. Why this matters logically

这组证据说明的不是一个“小接口问题”，而是一个直接影响算法语义的问题：

1. 压缩之后，系统需要把“物理上缩短后的有效 cache 长度”和“当前位置语义”同步给 vLLM 输入准备链路。
2. 这一步本来要靠 effective override 完成。
3. 但当前默认真实路径上，这个 override 要么根本没构建出来，要么就算构建出来也没有 patch 到真实执行函数。

于是后果就是：

1. 压缩已经真实发生了
2. 但后续输入准备仍按“未压缩”的逻辑长度/位置语义走
3. 这会直接导致后续 attention 使用错误的有效上下文语义

这类错误完全可以解释：

1. 为什么不是立刻乱码
2. 但压缩触发后输出会持续退化
3. 且和 HF prefill probe 的结果明显不一致

## 4. Re-prioritized suspect ranking

基于现在的证据，嫌疑优先级应该更新为：

### Highest priority

1. `runtime override / input patch 接口错配`
   - 当前最强主嫌疑

### High priority

2. `压缩后 V1 runner 真实输入准备路径的 positions / seq_lens / slot_mapping 没被正确覆盖`
   - 这是上面主嫌疑在实现层面的直接展开

3. `reclaim / compaction 后续语义漂移`
   - 仍然是高嫌疑，但已不是第一位

### Medium priority

4. `paged selector vs dense selector keep 不等价`
   - 当前没有证据支持它是主因

5. `async 时序问题`
   - 历史上确实存在，但这一步的新证据说明它不是当前最先该查的东西

## 5. Most likely repair direction

当前更保守、更合理的修复方向不是强推 V2 runner，而是：

1. 继续支持当前默认 `GPUModelRunnerV1`
2. 让 effective override 真正兼容这条实际 runner 路径

也就是说：

1. override 构建要能从 `requests + input_batch.req_id_to_index` 取到行索引
2. runtime input patch 要能 patch 到 `gpu_model_runner` 这一套真实 V1 输入准备路径
3. debug 验证通过后，再考虑是否抽象成同时兼容 V1/V2 的统一接口

这样做的理由是：

1. 修改面更小
2. 风险更低
3. 更贴近当前 demo 实际运行路径
4. 不需要先解决 V2 runner 那条新的 `KeyError` 兼容性问题

## 6. What is still not proven

虽然现在证据已经很强，但仍然要把边界写清楚：

1. 还没有完成“修上这条路径后，输出就恢复正常”的最终闭环验证
2. 所以目前最准确的表述仍然是：
   - 这是高置信度根因方向
   - 但还不是最终修复完成的证明

下一步最自然的验证就是：

1. 继续排查 debug probe 后出现的 `no scheduled items` 空转收尾问题
2. 拿到修补后的最终输出文件
3. 再确认同一条 32B case 是否从“明显坏掉”恢复到“正常讲人话”
