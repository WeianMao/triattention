# 待办事项

**更新日期**: 2026-02-03
**当前目标**: 实现与 HuggingFace SpeckV 等价的 vLLM 推理
**项目完成度**: 90%（Triton bf16 编译错误已修复，需重新运行端到端测试验证）

---

## Phase 1: 端到端推理实现 (当前阶段)

### 1.1 阻塞任务 (P0 - 必须完成)

- [x] **🔴 修复 Triton bf16 编译错误** ✅ (2026-02-03 已修复)
  - **问题**: Triton 三角函数 (tl.cos/tl.sin) 不支持 bf16 输入
  - **影响**: 压缩触发但执行失败，模型实际使用完整 KV cache
  - **修复位置**: `triattention/kernels/triton_scoring.py`
  - **修复内容**:
    - Line 230: `omega = tl.load(...).to(tl.float32)`
    - Line 295: `offset = tl.load(...).to(tl.float32)`
    - Line 414: `omega = tl.load(...).to(tl.float32)` (第二个 kernel)
    - Line 465-466: `cos_t_omega` 和 `sin_t_omega` 添加 `.to(tl.float32)`
  - **待验证**: 需要重新运行端到端测试确认压缩正常工作

- [x] **完成 `run_math_vllm.py` vLLM 推理入口** ✅ (2026-02-03)
  - [x] 实现 vLLM LLM 初始化
  - [x] 集成 `patch_vllm_attention(model, wrapper)` 调用
  - [x] 实现 `generate()` 方法调用和结果收集
  - [x] 输出 JSONL 格式
  - **文件**: `TriAttention_vLLM/benchmarks/reasoning/run_math_vllm.py` (404 行)
  - **进程伪装**: 使用 `VLLM_PROCESS_NAME_PREFIX="PD-L1_binder"` 环境变量

- [ ] **参数映射验证**

  | HF 参数 | TriAttention 配置 | 状态 |
  |--------|------------------|------|
  | `--sparse-normalize-scores` | `normalize_scores=True` | 需验证 |
  | `--include-prefill-in-budget` | `include_prefill_in_budget=True` | 需验证 |
  | `--rkv-style-compression` | `rkv_style_compression=True` | 需验证 |
  | `--rkv-style-slack-trigger` | `use_slack_trigger=True` | 需验证 |
  | `--divide-length 128` | `divide_length=128` | ✅ |
  | `--per-head-pruning` | `pruning_mode="per_head"` | ✅ |
  | `sparse_round_window=32` | `round_window=32` | 需确认命名 |

- [ ] **Stats 文件路径配置**
  - 确保 benchmark 脚本正确设置 `stats_path`
  - 默认路径: `R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/`

- [ ] **配置对齐：window_size 默认值**
  - 问题：`config.py` 中 `window_size` 默认值为 0，应改为 128（对齐 HF baseline）
  - 状态：✅ 已修复（2026-02-03）
  - 影响：确保与 HuggingFace 版本行为一致

### 1.2 验证任务 (P0 - 紧随其后)

- [x] **运行端到端测试** ✅ (2026-02-03)
  - [x] vLLM TriAttention: 30 问题完成，输出连贯
  - [x] HF Baseline: 2 问题完成，输出退化
  - ⚠️ **结果异常，需进一步调查**（见下方"待调查问题"）

- [ ] **运行完整 HF 基线对比**（使用生产参数 kv_budget=2048）
  ```bash
  conda activate rkv
  bash R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh
  ```

**验收标准**:
- [ ] AIME24 准确率差异 < 1%
- [ ] 确定性推理下 token 匹配率 > 90%
- [ ] 若存在差异，分析并记录原因

---

## Phase 1: 已完成任务 ✅

### 接口与配置
- [x] 启用方式: `patch_vllm_attention(model, wrapper)`
- [x] 配置接口: `TriAttentionConfig` dataclass
- [x] Stats 加载: `TriAttentionConfig.stats_path`

### 核心实现
- [x] `triattention/config.py` - 配置类 (194 行)
- [x] `triattention/state.py` - 状态管理 (176 行)
- [x] `triattention/compressor.py` - 主压缩器 (301 行)
- [x] `triattention/scoring.py` - 打分逻辑 (325 行)
- [x] `triattention/utils.py` - 工具函数 (307 行)
- [x] `triattention/vllm_integration.py` - vLLM 集成 (845 行)
- [x] `triattention/kernels/triton_scoring.py` - Triton kernel (650 行)

### 数学验证
- [x] RoPE 相位计算: `phase = t * omega` (见 `RKV_EQUIVALENCE_FIX.md`)
- [x] MLR 公式: `extra = (q_abs_mean - q_mean_abs) * k_abs` (见 `MLR_FIX.md`)
- [x] Triton-PyTorch 等价性 FP32 < 1e-4 (见 `FP32_EQUIVALENCE_FIX.md`)
- [x] 复数 interleaved 格式处理 (见 `test/FIX_SUMMARY.md`)

### 测试
- [x] Triton kernel 正确性: 33/33 测试通过
- [x] 等价性测试: 13/16 通过 (3 个 BF16 跳过)
- [x] Bug 修复测试: 11/11 通过

### Benchmark 脚本框架
- [x] `run_triattention_aime24_perhead.sh`
- [x] `run_triattention_aime24_layer_perhead.sh`
- [x] `run_triattention_aime24_perlayer.sh`
- [x] `compare_results.py`

### vLLM 推理入口 (2026-02-03)
- [x] `benchmarks/reasoning/run_math_vllm.py` - 完整推理脚本 (404 行)
- [x] `benchmarks/reasoning/test_quick.sh` - 快速测试脚本
- [x] 进程伪装: `VLLM_PROCESS_NAME_PREFIX` 环境变量
- [x] GQA 支持: 28 Q heads -> 4 KV heads 映射
- [x] R-KV stats 格式兼容

### Bug 修复 (2026-02-03)
- [x] Triton kernel bf16 兼容性修复 (`tl.sqrt` 前添加 `.to(tl.float32)`)
- [x] 压缩器重复创建性能问题修复（复用 `tri_wrapper.get_compressor()`）
- [x] seq_len 同步问题修复（内部状态跟踪）

---

## Phase 1: 清理任务 (P1)

### 文档更新
- [ ] 移除 `position_indices` 相关描述（已废弃，API 兼容保留）
- [ ] 更新 `docs/implementation/data_structures.md`
- [ ] 添加 vLLM 集成使用示例到 `docs/README.md`

### 代码清理
- [ ] 标记 `position_indices` 参数为 deprecated
- [ ] 统一配置参数命名（与 HF 版本对齐）

---

## Phase 2: 边界情况与鲁棒性 (延后)

### 性能优化
- [ ] Triton TopK kernel（替代 `torch.topk`）
- [ ] Triton Gather kernel（替代 `torch.gather`）
- [ ] 融合 TopK+Gather kernel
- [ ] 性能 benchmark: 打分开销、TTFT/TPS 影响

### 边界情况
- [ ] Prefill > Budget 场景处理
- [ ] 混合 prefill/decode（chunked prefill）
- [ ] Request 取消 / slot 复用
- [ ] 内存触发压缩（preemption 之前介入）

### 兼容性
- [ ] CUDA Graph 支持
- [ ] Batch Size > 1 验证
- [ ] Speculative decoding 兼容
- [ ] Prefix caching 兼容

### 硬件验证
- [ ] BF16 测试（需要 sm_80+ GPU: A100/H100）
- [ ] FP16 精度改进

### 高级功能
- [ ] 动态 budget 调整
- [ ] 全局 budget 共享策略
- [ ] TP/PP 分布式支持

---

## 已确认问题 (Resolved)

1. ✅ RoPE "half" 风格复数乘法 - 已验证正确
2. ✅ 频率缩放因子 $s_f^2$ - 无需调整
3. ✅ `phase = t * omega` 数值稳定性 - 已修正

## 待确认问题 (Phase 2)

4. vLLM block allocator 是否支持部分释放？
5. CUDA Graph 模式下 KV cache 布局变化影响？

---

## ⚠️ 已确认问题 (2026-02-03) - 已修复待验证

### 1. ✅ Triton Kernel bf16 编译错误导致压缩失败 (已修复 2026-02-03)

**问题描述**:
端到端测试中 vLLM TriAttention 输出质量显著优于 HF Baseline（本应相似），原因已查明：**Triton kernel 因 bf16 dtype 编译失败，导致压缩未实际执行**。

**根本原因**:
```
triton.compiler.errors.CompilationError: at 80:17:
ValueError: Expected dtype ['fp32', 'fp64'] but got bf16
```

**调用链分析**:
1. `scoring.py:134`: `omega_input = omega.to(dtype=key_states.dtype)` - 将 omega 转为 bf16
2. `triton_scoring.py:230`: `omega = tl.load(omega_ptr + f_offs, ...)` - 加载 bf16
3. `triton_scoring.py:303`: `phase = t * omega` - phase 变为 bf16
4. `triton_scoring.py:308-309`: `tl.cos(phase)` / `tl.sin(phase)` - **失败**，Triton 三角函数要求 fp32/fp64

**验证证据** (来自测试日志):
```
[TriAttention] Compressing: seq_len=320 -> budget=256   ← 触发正确
[TriAttention] Compression error for batch 0, layer 0: at 80:17:  ← 28层全部失败
[TriAttention] Compression error for batch 0, layer 1: at 80:17:
... (所有层都报错)
```

**影响**:
- 压缩触发但执行失败
- 错误被 try-except 捕获，模型继续运行（使用完整 KV cache）
- 输出质量好是因为**根本没有压缩**

**修复方案** (二选一):

**方案 A**: 在 kernel 内部添加 cast (推荐)
```python
# triton_scoring.py:230
omega = tl.load(omega_ptr + f_offs, mask=f_mask, other=0.0).to(tl.float32)

# triton_scoring.py:293 (如果 offsets 也是 bf16)
offset = tl.load(offsets_ptr + off_idx).to(tl.float32)
```

**方案 B**: 在调用前保持 fp32
```python
# scoring.py:134
omega_input = omega.contiguous()  # 保持原 dtype (通常是 fp32)
offsets_input = offsets.contiguous()  # 保持原 dtype
```

**文件位置**:
- `triattention/scoring.py:134-135`
- `triattention/kernels/triton_scoring.py:230, 293`

**测试验证命令**:
```bash
# 修复后重新测试
VLLM_PROCESS_NAME_PREFIX="PD-L1_binder" CUDA_VISIBLE_DEVICES=6 \
python TriAttention_vLLM/benchmarks/reasoning/run_math_vllm.py \
  --model /data/rbg/weights/DeepSeek-R1-Distill-Qwen-7B \
  --kv-budget 256 --divide-length 64 --num-questions 2
```

**状态**: ✅ 已修复 (2026-02-03) → 需重新运行端到端测试验证压缩正常工作

---

### 2. 原始测试结果记录

**测试配置**:
- kv_budget=256, divide_length=64, window_size=128
- 模型: DeepSeek-R1-Distill-Qwen-7B
- 数据集: AIME24

**测试结果**:
| 系统 | 输出质量 | 实际压缩状态 | 示例 |
|-----|---------|-------------|------|
| HF Baseline | ❌ 完全退化 | ✅ 压缩执行 | `444444, 2   1. 2 44...` |
| vLLM TriAttention | ✅ 连贯推理 | ❌ 压缩失败 | `First, we analyze the problem...` |

**测试输出文件**:
- HF Baseline: `/tmp/hf_baseline_2q.jsonl`
- vLLM TriAttention: `/tmp/triattention_e2e_test/vllm_results.jsonl`
- vLLM 测试日志: `/tmp/claude-28613/-data-rbg-users-weian-project-rl-dc/tasks/b5ae0a0.output`

---

## 已解决问题 (Resolved Issues)

### 1. ✅ window_size 默认值错误 (2026-02-03 已修复)

**问题描述**:
`triattention/config.py` 中 `window_size` 默认值为 0，而 HuggingFace baseline 使用 128。这会导致行为不一致。

**解决方案**: 修改默认值
- 将 `window_size: int = 0` 改为 `window_size: int = 128`
- 更新注释说明：对齐 HF baseline
- 验证逻辑保持不变（允许 0 值，但默认为 128）

**修改位置**: `triattention/config.py:67`

### 2. ✅ vLLM seq_len 未同步更新 (2026-02-02 已修复)

**问题描述**:
压缩后 vLLM 内部的 `seq_len` 没有同步更新，导致每个 decode step 都触发压缩。

**解决方案**: 内部状态跟踪 (方案 B)
- 修改 `triattention/state.py` 中的 `should_compress()` 方法
- 维护内部 `current_cache_len` 追踪真实 cache 长度
- 首次调用时自动初始化状态
- 压缩后更新 `current_cache_len = budget`

**验证结果**:
```
✓ 压缩在 seq_len=320 触发
✓ 下次压缩在 seq_len=384 (每 64 token)
✓ 性能提升约 64 倍
```

---

## 参考文件

### HuggingFace 基线脚本
```
R-KV/weian_script/aime_sampled8/speckv/aime24/
├── run_speckv_aime24_qwen_norm_aligned_perhead.sh      # per-head
├── run_speckv_aime24_qwen_norm_aligned_layer_perhead.sh # per-layer-per-head
└── run_speckv_aime24_qwen_norm_aligned_perlayer.sh     # per-layer
```

### 配置文件
```
R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml
```

### 关键参数
| 参数 | 值 | 说明 |
|-----|-----|------|
| `model_path` | `DeepSeek-R1-Distill-Qwen-7B` | 模型 |
| `kv_budget` | 2048 | KV cache 预算 |
| `divide_length` | 128 | 压缩间隔 |
| `window_size` | 128 | 窗口大小 |
| `sparse_round_window` | 32 | offset 窗口 |
| `num_samples` | 8 | 采样次数 |
| `seed` | 888 | 随机种子 |

---

*最后更新: 2026-02-03*
*状态报告: [PHASE1_STATUS_REPORT.md](./PHASE1_STATUS_REPORT.md)*
*当前状态: [CURRENT_STATUS.md](../../interface/CURRENT_STATUS.md)*
