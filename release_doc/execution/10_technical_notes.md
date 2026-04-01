# 技术发现记录

## KV Cache 峰值对齐

经过多轮代码审查确认：

**RKV 和 TriAttention（SpeckV rkv-style）的 KV cache 峰值是对齐的，都是 ~2176（budget + divide_length）。**

### 原因

- RKV 官方实现在实验框架中也受 `divide_length` 门控（`CausalLM_forward` 中 `self.length % self.config.divide_length == 0`）
- 不是每个 token 都压缩，而是每 `divide_length`（默认 128）步才触发一次
- `r1_kv.py` 的 `update_kv` 虽然每次都检查，但上层 `modeling.py` 的 `CausalLM_forward` 通过 `config.compression` flag 控制是否真正执行压缩
- `compression = True` → 压缩；`compression = False` → 不压缩，cache 自由增长；`compression = None` → 初始状态

### 关键代码位置

- 门控逻辑：`R-KV/HuggingFace/rkv/modeling.py` line 638: `is_newline = self.length % self.config.divide_length == 0`
- Flag 传播：`R-KV/HuggingFace/rkv/modeling.py` line 647: `layer.self_attn.config.compression = is_newline`
- Attention 层三分支：`R-KV/HuggingFace/rkv/modeling.py` lines 146-191

### 常见误解

> :warning: 之前的分析曾错误认为 RKV 峰值是 2049（每 token 压缩），这是因为只看了 `r1_kv.py` 底层实现，没追踪到 `CausalLM_forward` 的门控逻辑。

## 脚本对比

| 脚本 | 方式 | 峰值 | 和 RKV 对齐？ |
|------|------|------|--------------|
| `norm_aligned_perhead`（脚本 #1） | `--rkv-style-compression` + `--rkv-style-slack-trigger` | ~2176 | :white_check_mark: 对齐 |
| `norm_aligned_budget_perhead`（脚本 #2） | `--rkv-aligned-budget`（generate wrapper 风格） | ~2080 | :x: 不对齐 |
| 官方 RKV | 原生 attention layer 压缩 + divide_length 门控 | ~2176 | — |
