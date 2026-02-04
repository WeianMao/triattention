# 待办事项

**更新日期**: 2026-02-03
**当前目标**: Phase 1 - 完成 Attention 接口集成，实现与 HuggingFace SpeckV 等价的 vLLM 推理

---

## Phase 1: 端到端推理实现 (当前阶段)

### 1.1 阻塞任务 (P0 - 必须完成)

- [x] **修复 Triton bf16 编译错误** ✅ (2026-02-03)
  - **问题**: Triton 三角函数 (tl.cos/tl.sin) 不支持 bf16 输入，导致压缩执行失败
  - **修复**: `triattention/kernels/triton_scoring.py` 添加 `.to(tl.float32)` cast
  - **待验证**: 需重新运行端到端测试确认压缩正常工作

- [x] **完成 `run_math_vllm.py` vLLM 推理入口** ✅ (2026-02-03)
  - 实现 vLLM LLM 初始化、`patch_vllm_attention()` 集成、JSONL 输出
  - **文件**: `TriAttention_vLLM/benchmarks/reasoning/run_math_vllm.py` (404 行)
  - **进程伪装**: `VLLM_PROCESS_NAME_PREFIX="PD-L1_binder"`

- [ ] **实现 TriAttentionBackend（继承 FlashAttentionBackend）**（延后）
  - 创建 `triattention/backends/triattention_backend.py`
  - 支持 `--attention-backend triattention` 命令行参数
  - **预估工作量**: ~200 行代码

- [ ] **实现 TriAttentionImpl（继承 FlashAttentionImpl）**（延后）
  - 创建 `triattention/backends/triattention_impl.py`
  - **预估工作量**: ~150 行代码

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

### 1.2 验证任务 (P0 - 紧随其后)

- [ ] **运行 HF 基线获取参考结果**
  ```bash
  # 切换到 rkv 环境
  conda activate rkv
  bash R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh
  ```
  - 输出目录: `R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_perhead/`

- [ ] **运行 vLLM 版本**
  ```bash
  bash TriAttention_vLLM/benchmarks/reasoning/run_triattention_aime24_perhead.sh
  ```
  - 注意: 需要 `enforce_eager=True`（CUDA Graph 不兼容）

- [ ] **结果对比**
  ```bash
  python TriAttention_vLLM/benchmarks/reasoning/compare_results.py \
    --hf-output <hf_result.jsonl> \
    --vllm-output <vllm_result.jsonl>
  ```

**验收标准**:
- [ ] AIME24 准确率差异 < 1%
- [ ] 确定性推理下 token 匹配率 > 90%
- [ ] 若存在差异，分析并记录原因

### 1.3 集成 R-KV 测试框架 (P0 - 端到端验证)

**目标**: 复用 R-KV 项目的测试框架（上游分发 + 下游评估），仅替换中游推理引擎为 vLLM TriAttention。

**背景**:
- 原测试脚本: `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`
- 不能修改原脚本（其他项目在复用）
- 需要拷贝副本到 TriAttention_vLLM 项目，替换中游引擎

**三层架构**:

| 层级 | 原文件 | 功能 | 处理方式 |
|-----|-------|------|---------|
| **上游** | `rkv_sharded_dispatch.py` | 分发任务到 8 GPU 并行 | 拷贝，修改 runner_path |
| **中游** | `rkv_sharded_eval.py` | HF 推理 + SpeckV 压缩 | **替换为 vLLM 推理脚本** |
| **下游** | `merge_rkv_shards.py` + `eval_math_multi.py` | 合并结果 + 计算准确率 | 拷贝，保持不变 |

**数据流**:
```
aime24.jsonl → 分发(8 GPU) → vLLM推理 → shards/*.jsonl → 合并 → 评估 → 准确率
```

**需要拷贝的文件**:

```
TriAttention_vLLM/evaluation/          # 新建目录
├── dispatch/
│   ├── rkv_sharded_dispatch.py        # 上游分发器
│   └── configs/                       # 配置文件
│       └── triattention_aime24.yaml   # 新配置（指向 vLLM 推理）
├── runner/
│   └── vllm_triattention_runner.py    # 新建：vLLM 推理脚本（替代 rkv_sharded_eval.py）
├── merge/
│   └── merge_rkv_shards.py            # 下游合并
└── eval/
    ├── eval_math_multi.py             # 下游评估
    ├── evaluate.py
    ├── parser.py
    ├── python_executor.py
    ├── grader.py
    ├── math_utils.py
    ├── data_loader.py
    └── latex2sympy/                   # LaTeX 解析库
```

**vLLM 推理脚本适配要点**:
1. 接受相同命令行参数：`shard_id`, `num_shards`, `model_path`, `output_dir` 等
2. 输出格式兼容：`sample_idx`, `draw_idx`, `output`, `prefill_tokens`, `output_tokens`
3. 生成 `.meta.json` 文件标记完成状态
4. 支持断点续传（检查已完成的 sample_idx）
5. Seed 管理：`seed + run_id * 1_000_000 + sample_idx`

**执行步骤**:
- [ ] 创建 `TriAttention_vLLM/evaluation/` 目录结构
- [ ] 拷贝上游分发器和配置模板
- [ ] 拷贝下游合并和评估脚本
- [ ] 创建 `vllm_triattention_runner.py`（核心工作）
- [ ] 创建 `triattention_aime24.yaml` 配置文件
- [ ] 本地测试单 shard 运行
- [ ] 全量测试 8 GPU 并行

**验收标准**:
- [ ] 使用与 HF baseline 相同的配置（kv_budget=2048, divide_length=128）
- [ ] AIME24 准确率与 HF baseline 差异 < 1%
- [ ] 输出格式与下游评估脚本完全兼容

---

## Phase 1: 已完成任务 ✅

### 接口与配置
- [x] 启用方式（方案 B）: `patch_vllm_attention(model, wrapper)` - Monkey Patching
- [ ] 启用方式（方案 A）: `--attention-backend triattention` - 继承 FlashAttentionImpl（待实现）
- [x] 配置接口: `TriAttentionConfig` dataclass
- [x] Stats 加载: `TriAttentionConfig.stats_path`
- [x] 确认 vLLM 配置方式: 无 `VLLM_ATTENTION_BACKEND` 环境变量，统一使用 `--attention-backend` 参数

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

## Phase 2: Scheduler 接口集成与鲁棒性 (下一阶段)

### Scheduler 接口集成（内存触发压缩）
- [ ] Hook `Scheduler.schedule()` 获取 block 使用率
- [ ] 实现动态压缩触发：block 使用率 ≥ 98% 时触发
- [ ] 传递 block 使用率信息到 Attention 层（通过 ForwardContext）
- [ ] 在 `should_compress()` 中根据使用率决策
- [ ] 验证内存触发压缩功能
- **参考**: 决策 3.6（DESIGN_DECISIONS.md）
- **预估工作量**: ~450 行代码，7-11 天

### 性能优化
- [ ] Triton TopK kernel（替代 `torch.topk`）
- [ ] Triton Gather kernel（替代 `torch.gather`）
- [ ] 融合 TopK+Gather kernel
- [ ] 性能 benchmark: 打分开销、TTFT/TPS 影响

### 边界情况
- [ ] Prefill > Budget 场景处理
- [ ] 混合 prefill/decode（chunked prefill）
- [ ] Request 取消 / slot 复用

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

## 已解决问题 (Resolved Issues)

### 1. ✅ vLLM seq_len 未同步更新 (2026-02-02 已修复)

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
*状态报告: [PHASE1_STATUS_REPORT.md](../backend/reference/PHASE1_STATUS_REPORT.md)*
