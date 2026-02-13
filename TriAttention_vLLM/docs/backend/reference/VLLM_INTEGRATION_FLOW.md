# TriAttention 与 vLLM 集成流程详解

本文档详细追踪 TriAttention 如何与 vLLM 集成，从入口点到压缩执行的完整数据流。

## 目录

1. [入口点：Runner 初始化](#1-入口点runner-初始化)
2. [Patch 过程：猴子补丁机制](#2-patch-过程猴子补丁机制)
3. [运行时 Hook：推理时的拦截](#3-运行时-hook推理时的拦截)
4. [压缩流程：数据在 KV Cache 中的流动](#4-压缩流程数据在-kv-cache-中的流动)
5. [状态管理：Per-Request 隔离](#5-状态管理per-request-隔离)

---

## 1. 入口点：Runner 初始化

### 1.1 文件位置

```
TriAttention_vLLM/evaluation/runner/vllm_triattention_runner.py
```

### 1.2 初始化流程

**步骤 1：创建 TriAttention 配置**

```python
# Line 293-315
def setup_triattention_config(args: argparse.Namespace):
    """从命令行参数创建 TriAttention 配置"""
    from triattention import TriAttentionConfig

    stats_path = resolve_path(args.sparse_stats_path) if args.sparse_stats_path else None

    config = TriAttentionConfig(
        stats_path=stats_path,           # 预计算的频率统计文件
        kv_budget=args.kv_budget,        # KV token 预算（如 2048）
        divide_length=args.divide_length, # 压缩间隔（如 128）
        pruning_mode=args.pruning_mode,  # per_head/per_layer
        window_size=args.window_size,    # 保护最近的 token（如 128）
        # ... 其他参数
    )
    return config
```

**关键参数说明：**
- `stats_path`: 频率统计文件路径（必需），包含模型 Q 分布统计
- `kv_budget`: 压缩后的最大 token 数（如 2048）
- `divide_length`: 触发压缩的间隔（当 cache 达到 `budget + divide_length` 时触发）
- `window_size`: 保护最近的 N 个 token 不被压缩

**步骤 2：初始化 vLLM 引擎**

```python
# Line 318-362
def setup_vllm_engine(args: argparse.Namespace, tri_config=None):
    """初始化 vLLM 引擎并应用 TriAttention 补丁"""
    from vllm import LLM, SamplingParams

    # 1. 创建 vLLM LLM 实例
    llm = LLM(
        model=args.model_path,
        dtype=args.load_dtype,              # bfloat16/float16
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
        max_model_len=max_model_len,
        enforce_eager=True,                 # 必须使用 eager 模式
    )

    tri_wrapper = None

    # 2. 如果启用压缩，应用猴子补丁
    if tri_config is not None and not args.disable_compression:
        from triattention.vllm_integration import patch_vllm_attention, TriAttentionWrapper

        # 创建 wrapper（管理压缩状态和逻辑）
        tri_wrapper = TriAttentionWrapper(tri_config)

        # 访问 vLLM 内部模型并打补丁
        try:
            # 深入 vLLM 内部获取实际的 model
            model = llm.llm_engine.model_executor.driver_worker.model_runner.model

            # 应用补丁（这是核心魔法！）
            patch_vllm_attention(model, tri_wrapper)

            print(f"[TriAttention] 猴子补丁已启用: kv_budget={tri_config.kv_budget}")
        except Exception as e:
            print(f"[TriAttention] 警告: 补丁失败: {e}")
            tri_wrapper = None

    return llm, tri_wrapper
```

**关键点：**
- vLLM 必须使用 `enforce_eager=True`，因为 CUDA graphs 与动态 KV cache 修改不兼容
- 通过 `llm.llm_engine.model_executor.driver_worker.model_runner.model` 获取实际的 PyTorch 模型
- `patch_vllm_attention()` 是核心集成点

---

## 2. Patch 过程：猴子补丁机制

### 2.1 文件位置

```
TriAttention_vLLM/triattention/vllm_integration.py
```

### 2.2 Patch 核心逻辑

**步骤 1：定位 Attention 层**

```python
# Line 543-619
def patch_vllm_attention(
    model,
    tri_wrapper: TriAttentionWrapper,
    layer_name_pattern: str = "model.layers",
    model_config=None,
    cache_config=None,
) -> None:
    """对 vLLM 模型的 attention 层打补丁"""

    # 1. 提取模型配置信息
    model_info = _extract_model_info(model, model_config, cache_config)
    # model_info = {
    #     'block_size': 16,        # vLLM paged cache 的 block 大小
    #     'num_kv_heads': 8,       # KV heads 数量（GQA）
    #     'head_dim': 128,         # head 维度
    # }

    # 2. 自动配置 wrapper
    if tri_wrapper.config.num_kv_heads is None:
        tri_wrapper.config.num_kv_heads = model_info['num_kv_heads']
    if tri_wrapper.config.head_dim is None:
        tri_wrapper.config.head_dim = model_info['head_dim']

    # 3. 找到所有 transformer layers（支持多种模型结构）
    layers = None
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            layers = model.model.layers  # Llama, Qwen, DeepSeek 等
        elif hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
            layers = model.model.decoder.layers  # OPT, BART decoder

    if layers is None:
        raise ValueError("无法找到 transformer layers")

    # 4. 遍历每一层，找到 attention 实现并打补丁
    num_patched = 0
    for layer_idx, layer in enumerate(layers):
        # 找到 attention 实现（FlashAttentionImpl）
        attn_impl = None

        if hasattr(layer, "self_attn"):
            attn_layer = layer.self_attn
            if hasattr(attn_layer, "impl"):
                attn_impl = attn_layer.impl  # FlashAttentionImpl

        if attn_impl is None:
            continue

        # 保存原始 forward 方法
        original_forward_func = attn_impl.forward.__func__

        # 创建包装后的 forward（带压缩逻辑）
        wrapped_forward = make_wrapped_forward(original_forward_func, layer_idx)

        # 替换 forward 方法
        attn_impl.forward = types.MethodType(wrapped_forward, attn_impl)
        num_patched += 1

    print(f"[TriAttention] 成功对 {num_patched} 个 attention 层打补丁")
```

**步骤 2：包装 Attention Forward**

```python
# Line 644-679
def make_wrapped_forward(orig_func, layer_index):
    """创建包装后的 forward 方法（闭包正确捕获 layer_idx）"""
    def wrapped_forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,    # 这是 vLLM 的 paged KV cache
        attn_metadata,              # 包含 block_tables, seq_lens 等
        output: Optional[torch.Tensor] = None,
    ):
        # 1. 先调用原始 attention forward（正常计算 attention）
        result = orig_func(
            self, layer, query, key, value, kv_cache, attn_metadata, output
        )

        # 2. 在 decode 步骤之后应用压缩
        # V1 API: 使用 max_query_len（decode: 1, prefill: >1）
        # V0 API: 使用 num_decode_tokens
        is_decode = getattr(attn_metadata, 'max_query_len', None) == 1 or \
                    getattr(attn_metadata, 'num_decode_tokens', 0) > 0

        if kv_cache.numel() > 0 and is_decode:
            try:
                # 应用 TriAttention 压缩（核心逻辑）
                _apply_triattention_compression(
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    tri_wrapper=tri_wrapper,
                    layer_idx=layer_index,
                    model_info=model_info,
                )
            except Exception as e:
                print(f"[TriAttention] 警告: 压缩失败 layer {layer_index}: {e}")

        return result
    return wrapped_forward
```

**关键时机：**
1. **Prefill 阶段**：`is_decode = False`，不触发压缩（只缓存 KV）
2. **Decode 阶段**：`is_decode = True`，每次生成新 token 后检查是否需要压缩
3. **压缩时机**：当 cache 长度达到 `budget + divide_length` 时触发

---

## 3. 运行时 Hook：推理时的拦截

### 3.1 压缩触发逻辑

**入口函数：`_apply_triattention_compression()`**

```python
# Line 780-965
def _apply_triattention_compression(
    kv_cache: torch.Tensor,      # vLLM paged cache
    attn_metadata,                # FlashAttentionMetadata
    tri_wrapper: TriAttentionWrapper,
    layer_idx: int,
    model_info: dict,
) -> None:
    """在 vLLM 的 paged KV cache 上应用 TriAttention 压缩"""

    # 1. 提取 decode metadata（支持 V0 和 V1 API）
    decode_meta = getattr(attn_metadata, 'decode_metadata', None)

    if decode_meta is not None:
        # V0 API
        block_tables = decode_meta.block_tables
        seq_lens_tensor = decode_meta.seq_lens_tensor
    else:
        # V1 API
        block_tables = getattr(attn_metadata, 'block_table', None)
        seq_lens_tensor = getattr(attn_metadata, 'seq_lens', None)

    # 2. 提取模型配置
    block_size = model_info['block_size']      # 16
    num_kv_heads = model_info['num_kv_heads']  # 8
    head_dim = model_info['head_dim']          # 128

    # 3. 处理 cache 格式（可能是扁平化的）
    # vLLM cache 格式:
    #   预期: [2, num_blocks, block_size, num_kv_heads, head_dim]
    #   扁平: [2, num_blocks, block_size * num_kv_heads * head_dim]

    if kv_cache.dim() == 3 and kv_cache.shape[0] == 2:
        # 扁平化格式，需要 reshape
        num_blocks = kv_cache.shape[1]
        kv_cache = kv_cache.view(2, num_blocks, block_size, num_kv_heads, head_dim)

    # 4. 分离 key 和 value cache
    key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache = kv_cache[1]

    # 5. 处理 batch 中的每个 sequence
    batch_size = block_tables.shape[0]

    for batch_idx in range(batch_size):
        seq_len = seq_lens_tensor[batch_idx].item()
        block_table = block_tables[batch_idx]

        # 生成 request_id（用于状态隔离）
        request_id = f"decode_{batch_idx}"

        # 6. 检查是否需要压缩
        should_compress = tri_wrapper.should_compress(layer_idx, seq_len, request_id)

        if not should_compress:
            continue

        # 7. 获取该 request 的 compressor
        compressor = tri_wrapper.get_compressor(layer_idx, request_id)

        # 8. 从 paged cache 中 gather KV 到 dense 格式
        keys, values = _gather_kv_from_paged_cache(
            key_cache, value_cache, block_table, seq_len, block_size
        )
        # keys: [1, num_kv_heads, seq_len, head_dim]

        # 9. 创建 position indices
        cache_positions = torch.arange(seq_len, device=key_cache.device, dtype=torch.int32)

        # 10. 执行压缩
        compressed_keys, compressed_values, new_positions = compressor.compress(
            key_states=keys,
            value_states=values,
            cache_positions=cache_positions,
            layer_idx=layer_idx,
        )

        # 11. 将压缩后的 KV scatter 回 paged cache
        _scatter_kv_to_paged_cache(
            compressed_keys, compressed_values,
            key_cache, value_cache,
            block_table, new_positions, block_size
        )

        new_seq_len = compressed_keys.shape[2]
        print(f"[TriAttention] 压缩完成: {seq_len} -> {new_seq_len} tokens")
```

### 3.2 Paged Cache 操作

**Gather 操作（从分页 cache 收集到连续内存）：**

```python
# Line 693-741
def _gather_kv_from_paged_cache(
    key_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,
    block_table: torch.Tensor,  # [num_seq_blocks] - 该序列使用的物理 block 索引
    seq_len: int,               # 序列实际长度
    block_size: int,            # 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从 paged cache 中收集 KV 到 dense 格式"""

    # 计算需要多少个完整 block 和剩余 token
    num_full_blocks = seq_len // block_size  # 例如 seq_len=130 -> 8 个完整 block
    remaining = seq_len % block_size          # 剩余 2 个 token

    gathered_keys = []
    gathered_values = []

    # 收集完整 blocks
    for block_idx in range(num_full_blocks):
        physical_block = block_table[block_idx].item()
        gathered_keys.append(key_cache[physical_block])    # [block_size, num_kv_heads, head_dim]
        gathered_values.append(value_cache[physical_block])

    # 收集最后一个不完整 block 的剩余 tokens
    if remaining > 0:
        physical_block = block_table[num_full_blocks].item()
        gathered_keys.append(key_cache[physical_block, :remaining])
        gathered_values.append(value_cache[physical_block, :remaining])

    # 拼接所有 tokens
    keys = torch.cat(gathered_keys, dim=0)  # [seq_len, num_kv_heads, head_dim]
    values = torch.cat(gathered_values, dim=0)

    # 转置为标准格式
    keys = keys.transpose(0, 1).unsqueeze(0)    # [1, num_kv_heads, seq_len, head_dim]
    values = values.transpose(0, 1).unsqueeze(0)

    return keys, values
```

**Scatter 操作（将压缩后的 KV 写回分页 cache）：**

```python
# Line 744-777
def _scatter_kv_to_paged_cache(
    compressed_keys: torch.Tensor,    # [1, num_kv_heads, budget, head_dim]
    compressed_values: torch.Tensor,
    key_cache: torch.Tensor,          # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    new_positions: torch.Tensor,      # [budget] - 压缩后保留的 token 索引
    block_size: int,
) -> None:
    """将压缩后的 KV 写回 paged cache"""

    # 去掉 batch 维度并转置
    keys = compressed_keys.squeeze(0).transpose(0, 1)  # [budget, num_kv_heads, head_dim]
    values = compressed_values.squeeze(0).transpose(0, 1)
    budget = keys.shape[0]

    # 按顺序写回 blocks
    for token_idx in range(budget):
        block_idx = token_idx // block_size
        slot_in_block = token_idx % block_size
        physical_block = block_table[block_idx].item()

        # 原地修改 cache
        key_cache[physical_block, slot_in_block] = keys[token_idx]
        value_cache[physical_block, slot_in_block] = values[token_idx]
```

**关键点：**
- vLLM 使用 paged attention，KV cache 不是连续存储
- 每个序列有一个 `block_table`，映射逻辑 block → 物理 block
- 压缩时需要 gather → compress → scatter 三步操作
- Scatter 操作是**原地修改**（in-place），直接更新 vLLM 的 cache

---

## 4. 压缩流程：数据在 KV Cache 中的流动

### 4.1 完整压缩管线

```
vLLM Paged Cache (fragmented)
         ↓
    [Gather] _gather_kv_from_paged_cache()
         ↓
Dense KV [1, num_kv_heads, seq_len, head_dim]
         ↓
    [Compress] compressor.compress()
         ├─ compute_scores() → Triton kernel
         ├─ normalize_scores()
         ├─ topk() → PyTorch
         └─ gather_kv_by_indices()
         ↓
Compressed KV [1, num_kv_heads, budget, head_dim]
         ↓
    [Scatter] _scatter_kv_to_paged_cache()
         ↓
vLLM Paged Cache (updated in-place)
```

### 4.2 核心压缩逻辑

**文件位置：`triattention/compressor.py`**

```python
# Line 162-229
def compress(
    self,
    key_states: torch.Tensor,     # [1, num_kv_heads, seq_len, head_dim]
    value_states: torch.Tensor,
    cache_positions: Optional[torch.Tensor] = None,  # DEPRECATED
    layer_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """压缩 KV cache"""

    # 1. 懒初始化（首次调用时加载统计信息）
    self._lazy_init()

    batch_size, num_kv_heads, seq_len, head_dim = key_states.shape

    # 2. 检查是否需要压缩
    if not self.state.should_compress(seq_len):
        # 不需要压缩，直接返回
        keep_indices = torch.arange(seq_len, device=key_states.device)
        return key_states, value_states, keep_indices

    # 3. 计算重要性分数
    scores = self._compute_scores(
        key_states=key_states,
        layer_idx=layer_idx,
    )
    # scores: [1, num_kv_heads, seq_len] (per_head 模式)
    #     或: [1, seq_len] (per_layer 模式)

    # 4. 标准化分数（可选）
    if self.config.sparse_normalize_scores:
        scores = normalize_scores(scores)

    # 5. 保护窗口 tokens（最近的 N 个 token）
    if self.config.window_size > 0:
        scores = protect_window_tokens(scores, self.config.window_size)

    # 6. TopK 选择
    keep_indices = self._select_topk(scores, self.config.kv_budget)
    # keep_indices: [1, num_kv_heads, budget] (per_head)

    # 7. Gather 压缩后的 KV
    compressed_keys = gather_kv_by_indices(key_states, keep_indices, dim=2)
    compressed_values = gather_kv_by_indices(value_states, keep_indices, dim=2)

    # 8. 更新压缩状态
    new_cache_len = compressed_keys.shape[2]
    self.state.update_after_compression(new_cache_len)

    return compressed_keys, compressed_values, keep_indices
```

### 4.3 Scoring 计算（Triton 加速）

**文件位置：`triattention/scoring.py`**

```python
# Line 77-106
def compute_scores_triton(
    key_states: torch.Tensor,      # [1, num_kv_heads, seq_len, head_dim]
    cache_positions: Optional[torch.Tensor],
    head_stats: Dict[str, torch.Tensor],
    omega: torch.Tensor,           # [freq_count] - RoPE 角频率
    offsets: torch.Tensor,         # [num_offsets] - 多参考位置
    freq_scale_sq: torch.Tensor,   # [num_kv_heads, freq_count]
    config: TriAttentionConfig,
    round_start: Optional[int] = None,
) -> torch.Tensor:
    """使用 Triton kernel 计算分数"""

    from .kernels.triton_scoring import speckv_scoring

    # 1. 提取 Q 统计信息（从预计算的 stats 文件）
    q_mean_complex = head_stats['q_mean_complex']  # [num_kv_heads, freq_count, 2]
    q_mean_real = q_mean_complex[..., 0].contiguous()
    q_mean_imag = q_mean_complex[..., 1].contiguous()
    q_abs_mean = head_stats['q_abs_mean'].contiguous()

    # 2. 确定 round_start（当前解码位置）
    if round_start is None:
        round_start = seq_len - 1

    # 3. 调用 Triton kernel 计算分数
    # SpeckV 公式: score = ||K_rot||^2 * freq_scale_sq * ||Q_mean||^2
    #                    - 2 * Re(K_rot^* · Q_mean) * freq_scale_sq
    # 其中 phase 通过 RoPE 解旋计算: phi = atan2(K_rot_imag, K_rot_real) - t * omega
    scores = speckv_scoring(
        K_rot=key_states.contiguous(),
        position_indices=None,                   # 不再需要
        q_mean_real=q_mean_real.to(key_states.dtype),
        q_mean_imag=q_mean_imag.to(key_states.dtype),
        q_abs_mean=q_abs_mean.to(key_states.dtype),
        freq_scale_sq=freq_scale_sq.contiguous().to(key_states.dtype),
        omega=omega.contiguous().to(key_states.dtype),
        offsets=offsets.contiguous().to(key_states.dtype),
        round_start=round_start,
        agg_mode=config.score_aggregation,      # 'mean' 或 'max'
        pruning_mode=config.pruning_mode,       # 'per_head' 或 'per_layer'
    )

    return scores
```

**Triton Kernel 优势：**
- **融合计算**：将 RoPE 解旋、复数运算、聚合操作融合在一个 kernel 中
- **高效访存**：减少 GPU 内存读写次数
- **自动调优**：通过 `@triton.autotune` 自动选择最优 block size

---

## 5. 状态管理：Per-Request 隔离

### 5.1 状态隔离机制

**文件位置：`triattention/vllm_integration.py`**

```python
# Line 29-67
class TriAttentionWrapper:
    """管理多个 request 的压缩状态"""

    def __init__(self, config: TriAttentionConfig, enabled_layers: Optional[set] = None):
        self.config = config
        self.enabled_layers = enabled_layers

        # Per-request compressor 存储: {request_id: {layer_idx: compressor}}
        self.request_compressors: Dict[str, Dict[int, TriAttentionCompressor]] = {}

        # 默认 request ID（向后兼容）
        self._default_request_id = "__default__"

    def register_request(self, request_id: str) -> None:
        """注册新 request 并初始化状态"""
        if request_id in self.request_compressors:
            self._reset_request(request_id)
        else:
            self.request_compressors[request_id] = {}

    def unregister_request(self, request_id: str) -> None:
        """注销 request 并清理状态（重要！）"""
        if request_id in self.request_compressors:
            for compressor in self.request_compressors[request_id].values():
                compressor.reset()
            del self.request_compressors[request_id]

    def get_compressor(
        self, layer_idx: int, request_id: Optional[str] = None
    ) -> TriAttentionCompressor:
        """获取或创建特定 request-layer 的 compressor"""
        if request_id is None:
            request_id = self._default_request_id

        # 自动注册
        if request_id not in self.request_compressors:
            self.register_request(request_id)

        # 获取或创建 compressor
        if layer_idx not in self.request_compressors[request_id]:
            self.request_compressors[request_id][layer_idx] = TriAttentionCompressor(
                self.config
            )

        return self.request_compressors[request_id][layer_idx]
```

### 5.2 CompressionState 状态跟踪

**文件位置：`triattention/state.py`**

```python
# Line 23-80
class CompressionState:
    """管理单个 request 的压缩状态"""

    def __init__(self, config: TriAttentionConfig):
        self.config = config

        # 状态变量
        self.absolute_position: int = 0       # 当前绝对位置（单调递增）
        self.compression_count: int = 0       # 已执行的压缩次数
        self.prefill_length: int = 0          # 初始 prefill 长度
        self.tokens_in_round: int = 0         # 当前 round 的 token 数
        self.current_cache_len: int = 0       # 当前 cache 长度
        self.last_prune_step: int = 0         # 上次压缩的位置
```

**压缩触发判断：**

```python
# Line 82-125
def should_compress(self, current_len: int) -> bool:
    """判断是否触发压缩

    R-KV slack mode 逻辑:
    - 触发条件: cache 达到 (budget + divide_length)
    - 压缩目标: 压缩到 budget
    - 结果: cache 在 [budget, budget + divide_length] 区间波动
    """

    # 更新内部状态
    if self.absolute_position == 0:
        # 首次调用 - 从 prefill 初始化
        self.initialize(current_len)
        effective_cache_len = current_len
    else:
        # 计算新增 tokens
        new_tokens = current_len - self.absolute_position
        if new_tokens > 0:
            self.append_tokens(new_tokens)
        # 使用内部跟踪的 cache 长度
        effective_cache_len = self.current_cache_len

    # 计算有效大小（排除受保护的 prefill）
    if self.config.protect_prefill:
        effective_size = max(0, effective_cache_len - self.prefill_length)
    else:
        effective_size = effective_cache_len

    # R-KV slack mode: 触发阈值为 budget + divide_length
    trigger_threshold = self.config.kv_budget + self.config.divide_length

    return effective_size >= trigger_threshold
```

**关键点：**
- **Prefill 保护**：如果 `protect_prefill=True`，prefill tokens 不参与压缩判断
- **Slack Mode**：允许 cache 在 `[budget, budget + divide_length]` 区间波动，避免频繁压缩
- **状态更新**：压缩后更新 `current_cache_len = budget`，为下一轮做准备

---

## 总结：完整数据流

```
1. Runner 初始化
   └─ setup_vllm_engine()
      ├─ 创建 vLLM LLM 实例
      ├─ 创建 TriAttentionWrapper
      └─ patch_vllm_attention() 打补丁

2. Patch 过程
   └─ 遍历所有 transformer layers
      └─ 包装 FlashAttentionImpl.forward()
         ├─ 调用原始 forward（正常 attention）
         └─ 在 decode 阶段后检查并压缩

3. Decode 推理循环
   └─ 每生成一个 token
      ├─ FlashAttention 计算 output
      ├─ KV cache 更新（vLLM 自动）
      └─ Hook 触发 _apply_triattention_compression()
         ├─ 检查: should_compress()?
         ├─ Gather: paged cache → dense
         ├─ Compress:
         │  ├─ Triton kernel 计算分数
         │  ├─ PyTorch TopK 选择
         │  └─ PyTorch Gather 收集
         └─ Scatter: dense → paged cache

4. 状态管理
   ├─ Per-request 隔离（不同 request 独立状态）
   ├─ Per-layer 独立（每层独立压缩）
   └─ 自动清理（request 完成时调用 unregister_request）
```

**核心优势：**
1. **非侵入式**：所有代码在 TriAttention 模块中，无需修改 vLLM 源码
2. **高效集成**：通过猴子补丁在 attention 后插入压缩逻辑
3. **状态隔离**：每个 request 独立状态，支持 batch 推理
4. **原地修改**：直接更新 vLLM 的 paged cache，无额外内存开销

**注意事项：**
- 必须使用 `enforce_eager=True`（CUDA graphs 不兼容）
- 需要预计算频率统计文件（`stats_path`）
- Request 完成后必须调用 `unregister_request()` 避免内存泄漏
