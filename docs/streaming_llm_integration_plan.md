# Streaming-LLM 集成计划

## 1. 目标
- 在现有 DeepConf + vLLM 框架上集成长上下文的“流式”推理能力，同时继续复用 vLLM 的执行引擎。
- 保留最早的 N 个提示 token，并仅让 KV 缓存持有最近的 M 个生成 token，实现与 streaming-llm 相同的“首段 + 最近”策略。
- 避免依赖 FlashAttention v3，从而保证方案能在当前的 RTX A6000（SM 8.6）集群上运行。
- 尽量减少对 vLLM 的侵入式修改，保持代码路径和上游保持一致，便于长期维护。

## 2. 关键信息与已确认事项
- **streaming-llm 的现状**：`streaming_llm.enable_streaming_llm` 直接改写 HuggingFace 模型并依赖 `past_key_values`；但在 vLLM 中 KV 管理隐藏在 `PagedAttention` 之后，无法直接复用该 helper。
- **当前 vLLM 行为**：`CacheConfig.sliding_window`（参见 `vllm/config/cache.py`）会裁剪 KV，但采用简单的 FIFO 逻辑，无法保护前缀，不满足“保留起始块、滑动最近块”的组合策略。
- **Prompt 必须常驻**：针对数学题等任务，question/prompt 需要始终可见，流式裁剪只作用于新生成的 token，不得影响原始 prompt。
- **局部注意力层**：部分 GPT-OSS 等模型存在 per-layer sliding window（局部注意力层）。这些层保持模型自带的局部窗口，不叠加全局 streaming 裁剪。
- **可行性判断**：可以给 `BlockTable` 增加“固定前缀块数量”的能力，只要在 `CacheConfig`、`EngineArgs`、`SelfAttnBlockSpaceManager` 和 `BlockTable` 中传递新的配置即可。
- **FlashAttention 前提**：`vllm/vllm_flash_attn/flash_attn_interface.py` 明确指出 FA3 在计算能力 8.6 上不可用；保留 FA2 并不影响实现计划。
- **DeepConf 封装**：`DeepThinkLLM` 初始化 vLLM 时通过 kwargs 传参，只需增加新的流式配置即可让 YAML 自定义这些参数，示例脚本无需大改。

## 3. 架构方案
1. **配置层**
   - 在 YAML 中新增可选的 `streaming` 字段，例如：
     ```yaml
     streaming:
       enabled: true
       pinned_tokens: 512
       recent_tokens: 4096
     ```
   - 在 `run_dispatch_serialized.py` 暴露对应的 CLI 参数，并将其注入示例脚本命令行。
   - `DeepThinkLLM` 读取 YAML 后，将配置转换为 vLLM 初始化参数。

2. **vLLM 本地增强**
   - 在 `CacheConfig` 与 `EngineArgs` 新增 `streaming_pinned_tokens`、`streaming_recent_tokens` 字段（默认 `None`）。
   - Streaming 开启时，让 `sliding_window = streaming_recent_tokens`；根据 block 大小计算 `pinned_block_count = ceil(streaming_pinned_tokens / block_size)` 并传递给 `SelfAttnBlockSpaceManager` 及 `BlockTable`。
   - 修改 `BlockTable` 的淘汰逻辑：索引小于 `pinned_block_count` 的块永远保留，从第一个非固定块开始回收；同时记录每个序列的 prompt 长度，确保裁剪只作用于 prompt 之后的新 token。
   - 构造 per-layer 元数据（worker 侧）时，对模型中已声明局部注意力窗口的层跳过 streaming 裁剪，继续使用模型原生窗口参数。
   - 工作线程侧的 `ModelRunner` 在构造 FlashAttention 元数据时继续使用 FA2，无需额外调整。

3. **运行时策略**
   - 关闭 streaming 时行为完全不变，原有前缀缓存可照常使用。
   - 开启 streaming 时自动禁用 prefix caching，避免与“固定前缀”策略冲突。
   - Speculative decoding、beam search 暂不支持，保持现有实现。
   - 若后续需要按请求动态切换，可在 `DeepThinkLLM.deepthink()` 中扩展一个 `streaming_overrides` 参数。

4. **测试与验证**
   - 在 `examples/` 下新增 `example_streaming.py` 之类的脚本：
     - 构造超长 prompt，验证当生成长度超过 `pinned + recent` 后显存占用趋于稳定。
     - 与基线（关闭 streaming）比对尾部 token，确认输出连续性。
   - 运行 `python -m compileall deepconf examples streaming-llm` 作为语法检查。
   - 在单卡上执行一次开启 streaming 的 offline 作业，验证完整流程稳定。

## 4. 文件级改动概览
- **deepconf/wrapper.py**：在 `DeepThinkLLM.__init__` 接收 `streaming_config`，并在启用时向 vLLM 传入 `streaming_pinned_tokens`、`streaming_recent_tokens`，同时禁用 prefix caching；把最终配置写入 `output.config`。
- **deepconf/utils.py**：若需要，增加 YAML `streaming` 字段解析与校验帮助函数。
- **scripts/config_loader.py**：扩展 schema 校验，允许 `streaming` 出现在 `common`/`offline` 等段落，并支持模式化覆盖。
- **scripts/yaml_runs_serialized/run_dispatch_serialized.py**：加入 `--streaming-enabled`、`--streaming-pinned`、`--streaming-recent` 三个新参数，构造命令行时附带。
- **development/example_offline_serialized.py**：暴露同名 CLI，并将参数传给 `DeepThinkLLM`。
- **vllm/config/cache.py & vllm/engine/arg_utils.py**：注册新字段，增加范围校验（例如 `pinned_tokens ≤ recent_tokens ≤ max_model_len`）。
- **vllm/core/block_manager.py & vllm/core/block/block_table.py**：持久化固定前缀块元数据并在淘汰时跳过这些块，同时依据 prompt 长度控制裁剪范围。
- **vllm/model_executor/**：在构建注意力元数据时识别 per-layer sliding window，保留模型原生设置。
- **vllm/worker/model_runner.py**：确保构造时读取新配置，其余逻辑沿用原有滑动窗口实现。
- **docs/**：补充运行参数、限制（启用后 prefix caching 无法使用、FA3 非必需、Speculative/beam 暂不支持）等说明。

## 5. 潜在风险与待确认事项
- **与 prefix caching 的兼容性**：需要明确 streaming 开启时如何优雅地关闭 prefix caching，避免冲突。
- **局部注意力层处理**：必须确认模型 config 中的 per-layer sliding window 信息来源（如 `layer_types`、`sliding_window` 字段等），实现时要确保局部层完全跳过 streaming 裁剪，避免语义退化。
- **块引用计数**：多采样场景共享 `BlockTable`，要确认固定前缀块的标记在共享场景下行为一致（虽然当前不处理 beam search，但仍要防止共享引用出现回收错误）。
- **未来升级**：上游若提供原生 streaming 支持，应尽量把改动限制在少量文件，方便后续 rebase。

Speculative decoding 与 beam search 明确不在当前支持范围内，后续若需要再单独规划。

## 6. 后续步骤
1. 在独立测试脚本中原型化 `BlockTable` 的固定前缀淘汰逻辑，确保数学推导正确且 prompt 永不裁剪。
2. 落地配置打通（YAML → DeepThink → vLLM），并增加必要的断言。
3. 修改 vLLM 缓存管理并在单 GPU 上用合成数据验证，重点覆盖含局部注意力层的模型。
4. 补充文档与计划说明，准备后续代码实现。
5. 与基础设施团队沟通，重建包含补丁的 vLLM wheel 并发到 `dc` 环境。
