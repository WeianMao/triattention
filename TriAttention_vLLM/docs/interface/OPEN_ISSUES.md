# 待办事项

**更新日期**: 2026-02-08
**当前目标**: Phase 1 - 完成 Attention 接口集成，实现与 HuggingFace SpeckV 等价的 vLLM 推理

---

## Phase 1: 端到端推理实现 (当前阶段)

### 1.0 🔴 切换到官方继承方案 (P0 - 最高优先级)

> **决策日期**: 2026-02-04
> **详细决策**: [DESIGN_DECISIONS.md - 决策 3.7](../backend/DESIGN_DECISIONS.md#37-切换到官方继承方案monkey-patching-作为备份归档-)

**背景**：当前 Monkey Patching 实现不符合 vLLM v1 引擎规范，需切换到官方继承方案。

**当前状态**：
- ✅ `triattention/v1_backend.py` - 官方继承方案已实现
- ✅ `triattention/plugin.py` - Plugin 注册已实现
- ✅ `evaluation/` pipeline - 已切换到官方继承方案 (2026-02-05)

**执行步骤**：

- [ ] **1. 归档当前实现**
  - 将当前 `evaluation/` 结果和配置备份
  - 记录 Monkey Patching 方案的 AIME24 结果：41.7%

- [x] **2. 调整 Runner 组件** ✅ (2026-02-05)
  - 文件：`evaluation/runner/vllm_triattention_runner.py`
  - 移除 `patch_vllm_attention()` 调用
  - 改用 `setup_triattention()` + 环境变量 `VLLM_ATTENTION_BACKEND=TRIATTENTION`

- [x] **3. 调整 Dispatch 组件** ✅ (2026-02-05)
  - 文件：`evaluation/dispatch/triattention_sharded_dispatch.py`
  - 添加环境变量设置 `VLLM_ATTENTION_BACKEND=TRIATTENTION`
  - 确认 conda 环境正确安装 triattention 包

- [x] **4. 检查 Merge/Eval 组件** ✅ (2026-02-05)
  - 文件：`evaluation/merge/`, `evaluation/eval/`
  - 确认输出格式兼容，无需修改

- [x] **5. 运行端到端测试** ✅ (2026-02-05) - **结果异常**
  - 使用官方继承方案运行 AIME24 评估
  - 结果：**45.0%**（预期 ~41.7%，差距过大）
  - 需要调查原因，见 1.0.1

- [x] **6. 确认 Plugin 注册正常** ✅ (2026-02-05)
  - `pyproject.toml` 配置正确：`triattention = "triattention.plugin:register"`
  - vLLM 能识别 `TRIATTENTION` backend

**验收标准**：
- 使用 `attention_backend="TRIATTENTION"` 或环境变量方式启用
- AIME24 准确率与 Monkey Patching 结果一致（差异 < 0.5%）
- 无 `patch_vllm_attention()` 调用

---

### 1.0.1 调查准确率异常 (P0 - 诊断已修正)

**历史结果记录**：
| 方案 | 准确率 | 日期 | 压缩是否生效 | 备注 |
|------|--------|------|-------------|------|
| HF baseline | 42.5% | - | ✅ | R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_perhead/ |
| Monkey Patching (vLLM 0.7.x) | 41.7% | 2026-02-04 | ✅ | patch_vllm_attention() |
| V1 Backend 第1次 (vLLM 0.15) | 45.0% | 2026-02-05 | ❌ 未触发 | v1_backend.py 中 _maybe_compress 是 pass |
| V1 Backend 第2次 (vLLM 0.15) | 46.7% | 2026-02-05 | ❌ 未触发 | 同上，3 GPU 运行 |
| V1 Backend 第3次 (vLLM 0.15) | **28.3%** | 2026-02-06 | ✅ 已触发 | 压缩生效但存在未知 bug |

**根因分析**：

**第1-2次（45.0% / 46.7%）**：`v1_backend.py` 的 `_maybe_compress_kv_cache()` 只有 `pass` 占位符，压缩从未执行，等同于 fullkv 模式。

**第3次（28.3%）**：重写 v1_backend.py 后压缩确实触发了（日志可确认），准确率大幅下降。

**⚠️ 诊断修正 (2026-02-07)**：之前认为 seq_lens 未同步是根因，但经深入调查后发现：
- **V0 Monkey Patching 方案（41.7%）同样不更新 seq_lens**（代码确认：`vllm_integration.py` 的 `_apply_triattention_compression()` 从未修改 seq_lens）
- V0 和 V1 都执行相同操作：scatter 压缩 tokens 到 positions 0-budget，不修改 seq_lens，attention 读取所有 positions（包括 budget~old_len 的"被淘汰"tokens）
- 因此 **seq_lens 未同步不是 28.3% 的根因**（否则 V0 也应该大幅下降，但实际 V0 获得 41.7%）
- 真正的根因需要进一步调查，见 1.0.2

**状态**：⚠️ 原诊断已修正，转入 1.0.2 重新调查

---

### 1.0.2 ✅ V1 Backend 压缩准确率异常 (P0 - 已解决)

**问题描述**：
V1 Backend 压缩后准确率（28.3%）远低于 V0 Monkey Patching（41.7%）。

**根因确认 (2026-02-08)**：**GQA 统计数据未正确平均**

#### 根因：`config.num_kv_heads=None` 导致 GQA stats 未平均

DeepSeek-R1-Distill-Qwen-7B 使用 GQA：`num_attention_heads=28`, `num_key_value_heads=4`, `gqa_ratio=7`。

频率统计文件包含 28 个 Q 头的数据。加载时需要按 GQA 分组平均到 4 个 KV 头。

| 步骤 | V0 (41.7%) | V1 (28.3%) |
|------|-----------|-----------|
| Config 创建 | `num_kv_heads=None` | `num_kv_heads=None` |
| 设置 num_kv_heads | `patch_vllm_attention()` 从模型提取 → 设为 4 | **没有人设置** → 保持 None |
| 加载统计数据 | `load_frequency_stats(..., num_kv_heads=4)` | `load_frequency_stats(..., num_kv_heads=None)` |
| GQA 映射 | `gqa_ratio=28/4=7` → 28 Q头平均到 4 KV头 | `gqa_ratio=28/28=1` → 不做 GQA 平均 |
| Stats 形状 | `freq_scale_sq: [4, 64]` ✅ | `freq_scale_sq: [28, 64]` ❌ |

**影响**：Triton 打分核用 head_idx 0-3（来自 4 个 KV 头）索引 28 条 stats：
- KV head 0 → stats[0] = Q head 0 个体统计（应为 Q heads 0-6 的平均）
- KV head 1 → stats[1] = Q head 1 个体统计（**错误** — Q1 属于 KV head 0 的 GQA 组）
- KV head 2 → stats[2] = Q head 2 个体统计（**错误** — Q2 属于 KV head 0 的 GQA 组）
- KV head 3 → stats[3] = Q head 3 个体统计（**错误** — Q3 属于 KV head 0 的 GQA 组）

4 个 KV 头全部使用错误的统计数据 → token 重要性打分错误 → 保留了错误的 tokens → 28.3%。

**关键代码证据**：

V0（正确）`vllm_integration.py:594`：
```python
if tri_wrapper.config.num_kv_heads is None:
    tri_wrapper.config.num_kv_heads = model_info['num_kv_heads']  # = 4
```

V1（缺失）`v1_backend.py:194`（修复前）：
```python
self._wrapper = TriAttentionWrapper(config)  # config.num_kv_heads = None
```

`utils.py:163-164`（当 num_kv_heads=None 时）：
```python
if num_kv_heads is None:
    num_kv_heads = num_attention_heads  # = 28，不做 GQA 平均
```

#### 修复 (2026-02-08)

在 `v1_backend.py` 的 `_get_wrapper()` 中，创建 wrapper 前从 `self._model_info` 获取模型信息：
```python
# Propagate model info to config for proper GQA handling
# in stats loading (mirrors V0's patch_vllm_attention logic)
if config.num_kv_heads is None:
    config.num_kv_heads = self._model_info["num_kv_heads"]
if config.head_dim is None:
    config.head_dim = self._model_info["head_dim"]
```

#### 修复后验证结果

| 方案 | Seed | AIME24 准确率 | 状态 |
|------|------|-------------|------|
| HF 无压缩基线 | - | 42.5% | 参考 |
| V0 Monkey-Patching | 888 | 41.7% | 参考 |
| V1 Backend (修复前) | 888 | 28.3% | ❌ GQA stats bug |
| **V1 Backend (修复后)** | **888** | **40.0%** | ✅ |
| **V1 Backend (修复后)** | **42** | **37.9%** | ✅ |

修复后两次实验平均 ~39.0%，与 V0 基线 (41.7%) 差距约 2.7%，在 30 题 × 8 samples 的正常方差范围内（±3%）。

**状态**：✅ 根因已确认并修复，准确率恢复正常

<details>
<summary>调查过程中排除的假设（参考）</summary>

| 假设 | 结论 | 原因 |
|------|------|------|
| seq_lens 未同步 | ❌ 非根因 | V0 也不同步，但获得 41.7% |
| KV cache 格式 NHD 不匹配 | ❌ 完全匹配 | stride_order = identity |
| Layer index 溢出 | ❌ 恰好 num_layers 次 | 全局计数器正确 |
| layer_idx 使用错误统计 | ❌ 会抛 ValueError | 有显式越界检查 |
| is_decode 保护缺失 | ❌ 影响 <1% | AIME24 prompt ~200 tokens，不会在 prefill 触发 |
| 请求状态污染 | 次要问题 | V0 也有同样行为 |
| block_table 修改 | ❌ 不可行 | kernel 用 seqused_k 控制迭代 |
| Zero-fill 被淘汰 tokens | ❌ 不可行 | softmax(0) = exp(0) = 1 |
| 修改 max_seqlen_k | ❌ 不可行 | 仅用于 workspace allocation |

</details>

<details>
<summary>vLLM V1 架构参考信息</summary>

**Flash Attention 核心参数**（`flash_attn.py:719-741`）：
- `seqused_k` = `attn_metadata.seq_lens`：主要控制，每个请求读取多少 KV tokens
- `max_seqlen_k` = workspace allocation，非执行控制
- `block_table` = 逻辑块到物理块映射，决定从哪里读

**seq_lens 生命周期**：
```
Request.num_computed_tokens → prepare_pos_seq_lens() → CommonAttentionMetadata(seq_lens=...) → forward()
```

**vLLM 社区 KV 压缩现状**：无官方 API，所有方案要么 fork vLLM 要么在 scheduler 层面操作。

**长期方案（Phase 2）**：需要在 scheduler 层面集成压缩，使 seq_lens 正确反映压缩后的长度。

</details>

**备注：seq_lens 同步参考信息**

以下信息来自之前的调查，虽然 seq_lens 不是 28.3% 的根因，但长期来看仍然是需要解决的问题：

<details>
<summary>seq_lens 数据流（参考）</summary>

```
Request.num_computed_tokens
  → prepare_pos_seq_lens() Triton kernel:
    seq_len = num_computed_tokens + query_len
  → CommonAttentionMetadata(seq_lens=...)
    → FlashAttentionMetadata(seq_lens=...)
      → forward() 中使用
```

| 组件 | 文件 | 关键行 |
|------|------|--------|
| Request state | `vllm/v1/request.py:133` | `num_computed_tokens = 0` |
| seq_lens 计算 | `vllm/v1/worker/gpu/input_batch.py:206-233` | Triton kernel |
| Metadata 构建 | `vllm/v1/worker/gpu/attn_utils.py:155-196` | CommonAttentionMetadata |
| Metadata 定义 | `vllm/v1/attention/backend.py:299-301` | dataclass, 有 replace() |

长期方案（Phase 2）：需要在 scheduler 层面集成压缩，使 seq_lens 正确反映压缩后的长度。参考 RFC #12254 CachePolicy 框架。
</details>

---

### 1.1 阻塞任务 (P0 - 必须完成)

- [x] **修复 Triton bf16 编译错误** ✅ (2026-02-03)

### 1.2 待清理 (P2 - 非阻塞)

- [ ] **删除 `sparse_round_window` 参数** (记录于 2026-02-04)
  - **原因**: 参数存在但从未使用，TriAttention 只用 `divide_length` (slack mode)
  - **位置**: `triattention/config.py:33`, 以及相关配置文件
  - **影响**: 无功能影响，仅清理代码

- [ ] **可选: 实现 `protect_prefill` debug 模式**
  - **现状**: 配置存在但 `get_effective_budget()` 未被调用
  - **用途**: Debug 时对比保护 prefill 前后效果
  - **优先级**: 低，可让 R-KV 团队提供对比数据
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

### 6. CUDA Graph 支持分析 (记录于 2026-02-04)

**背景**：官方继承方案是否更容易支持非 eager 模式？

**结论**：官方继承方案是**必要条件但非充分条件**。

**改进点**：
- ✅ `TriAttentionImpl.forward()` 在模型构建时注册，CUDA Graph 捕获时能"看到"
- ✅ 不再是运行时劫持，处于正确的框架位置

**仍存在的挑战**：
- ❌ 压缩有动态分支 `if should_compress()`
- ❌ 压缩改变 cache tensor shape

**可选解决方案**（Phase 2 评估）：
| 方案 | 说明 |
|------|------|
| Graph Break | 压缩时退出图模式，压缩后重新捕获 |
| Padding 策略 | 保持 cache shape 固定，用 mask 标记有效区域 |
| 分离捕获 | 只捕获 attention 核心，压缩逻辑在图外执行 |

**优先级**：Phase 2，先完成官方方案切换

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

*最后更新: 2026-02-08*
*状态报告: [PHASE1_STATUS_REPORT.md](../backend/reference/PHASE1_STATUS_REPORT.md)*
