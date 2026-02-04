# Phase 1 项目状态报告

**生成日期**: 2026-02-01
**目标**: 实现与 `run_speckv_aime24_qwen_norm_aligned_perhead.sh` 等价的 vLLM 推理评测

---

## 1. 执行摘要

| 维度 | 状态 | 说明 |
|-----|------|------|
| **核心库实现** | ✅ 完成 | 2,827 行代码，所有模块已实现 |
| **Triton Kernel** | ✅ 完成 | 打分 kernel 通过 33/33 测试 |
| **vLLM 集成** | ✅ 框架完成 | `patch_vllm_attention()` 可用 |
| **端到端验证** | ❌ 未完成 | benchmark 脚本未实际运行 |
| **HF 等价性** | ❌ 未验证 | vLLM vs HuggingFace 精度对比未完成 |

**结论**: 核心代码已完成，但**端到端验证未完成**。距离 vLLM 等价推理还需完成集成测试和精度验证。

---

## 2. 已完成任务 ✅

### 2.1 核心库 (`triattention/`)

| 文件 | 行数 | 状态 | 说明 |
|-----|------|------|------|
| `config.py` | 194 | ✅ | 30+ 配置参数，完整验证 |
| `state.py` | 176 | ✅ | 压缩状态管理，双重重置机制 |
| `compressor.py` | 301 | ✅ | 主压缩器，惰性初始化 |
| `scoring.py` | 325 | ✅ | 打分逻辑，Triton + PyTorch 双路径 |
| `utils.py` | 307 | ✅ | 工具函数，stats 加载 |
| `vllm_integration.py` | 845 | ✅ | vLLM hook 集成 |
| `kernels/triton_scoring.py` | 650 | ✅ | Triton 打分 kernel |

### 2.2 数学验证

| 验证项 | 状态 | 参考文档 |
|-------|------|---------|
| RoPE 相位计算公式 | ✅ | `RKV_EQUIVALENCE_FIX.md` |
| MLR 公式修正 | ✅ | `MLR_FIX.md` |
| Triton-PyTorch 等价性 (FP32) | ✅ | `FP32_EQUIVALENCE_FIX.md` |
| 复数格式 interleaved 处理 | ✅ | `test/FIX_SUMMARY.md` |

### 2.3 测试套件

| 测试类别 | 测试数 | 通过率 | 说明 |
|---------|-------|-------|------|
| Triton Kernel | 33 | 100% | `test_scoring_kernel.py` |
| 等价性测试 | 13/16 | 81% | 3 个 BF16 跳过（硬件限制） |
| Bug 修复测试 | 11 | 100% | `test_bug_fixes.py` |
| **总计** | 46/49 | 93.9% | 3 个跳过 |

### 2.4 Benchmark 脚本框架

- ✅ `benchmarks/reasoning/run_triattention_aime24_perhead.sh`
- ✅ `benchmarks/reasoning/run_triattention_aime24_layer_perhead.sh`
- ✅ `benchmarks/reasoning/run_triattention_aime24_perlayer.sh`
- ✅ `benchmarks/reasoning/run_math_vllm.py` (框架)
- ✅ `benchmarks/reasoning/compare_results.py`

---

## 3. 未完成任务 ❌

### 3.1 高优先级 (P0)

| 任务 | 状态 | 阻塞原因 | 预计工作量 |
|-----|------|---------|----------|
| **vLLM 端到端推理** | ❌ | `run_math_vllm.py` 有 TODO | 2-3 天 |
| **HF vs vLLM 精度验证** | ❌ | 需要先完成端到端推理 | 1-2 天 |
| **AIME24 准确率对比** | ❌ | 依赖精度验证 | 1 天 |

### 3.2 中优先级 (P1)

| 任务 | 状态 | 说明 |
|-----|------|------|
| Batch Size > 1 验证 | ⚠️ | 代码限制 batch=1，未充分测试 |
| RoPE 一致性检查 | ⚠️ | utils 中有函数但未集成 |
| Prefill 保护验证 | ⚠️ | 配置存在但未端到端测试 |
| 状态重置接口测试 | ⚠️ | `state.py` 实现但未验证 |

### 3.3 低优先级 (P2)

| 任务 | 状态 | 说明 |
|-----|------|------|
| Triton TopK/Gather | ⏸️ | Phase 2 评估 |
| CUDA Graph 兼容 | ⏸️ | Phase 2 |
| BF16 测试 | ⏸️ | 需要 sm_80+ GPU |

---

## 4. 问题清单

### 4.1 阻塞性问题 (Blocker)

#### 问题 1: `run_math_vllm.py` 未完整实现

**位置**: `TriAttention_vLLM/benchmarks/reasoning/run_math_vllm.py`

**现状**:
- 脚本框架存在
- vLLM 引擎初始化代码有 TODO 标记
- 无法实际运行推理

**影响**: 无法进行端到端验证

**建议修复**:
```python
# 需要补充：
1. vLLM LLM 初始化
2. 调用 patch_vllm_attention()
3. 运行 generate() 并收集结果
4. 输出 JSONL 格式结果
```

#### 问题 2: 参数映射不完整

**HF 脚本关键参数 vs TriAttention 配置**:

| HF 参数 | TriAttention 配置 | 状态 |
|--------|------------------|------|
| `--sparse-normalize-scores` | `normalize_scores` | ✅ |
| `--include-prefill-in-budget` | `include_prefill_in_budget` | ⚠️ 未验证 |
| `--rkv-style-compression` | `rkv_style_compression` | ⚠️ 需确认等价 |
| `--rkv-style-slack-trigger` | `use_slack_trigger` | ⚠️ 需确认等价 |
| `--divide-length 128` | `divide_length` | ✅ |
| `--per-head-pruning` | `pruning_mode="per_head"` | ✅ |
| `sparse_round_window=32` | `round_window` | ⚠️ 需确认命名 |

### 4.2 设计问题

#### 问题 3: Stats 文件路径硬编码

**现状**:
- HF 脚本使用 `R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/`
- vLLM 集成需要通过 `TriAttentionConfig.stats_path` 传入
- 但 benchmark 脚本可能没有正确设置

**建议**: 在 benchmark 脚本中显式设置 stats_path

#### 问题 4: batch_size 限制

**现状**:
- `compressor.py` 强制 `batch_size == 1`
- vLLM 默认可能尝试批量处理

**影响**: 多请求场景下可能出错

**代码位置**: `compressor.py:compress()` 方法开头

### 4.3 新暴露的问题

#### 问题 5: position_indices 设计已废弃

**现状**:
- 文档中多处提到 `position_indices` 作为核心数据结构
- 实际实现中已标记为 deprecated
- Triton kernel 不再使用此参数

**影响**: 文档与代码不一致

**建议**: 更新文档，移除对 position_indices 的依赖描述

#### 问题 6: vLLM Hook 触发条件不确定

**现状**:
- `patch_vllm_attention()` 基于 method replacement
- 不确定在 vLLM 0.15.x 的所有代码路径下都能正确触发

**建议**: 需要在真实 vLLM 环境下验证 hook 触发

#### 问题 7: CUDA Graph 不兼容

**现状**:
- 文档明确要求 `enforce_eager=True`
- vLLM 默认启用 CUDA Graph

**影响**: 性能可能不达预期

---

## 5. 与 HF 脚本的差距分析

### 5.1 目标脚本分析

**脚本**: `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`

**关键参数**:
```bash
--sparse-normalize-scores          # Z-score 标准化
--include-prefill-in-budget        # prefill 计入预算
--rkv-style-compression            # R-KV 风格压缩
--rkv-style-slack-trigger          # slack 触发（budget+128 时压缩）
--divide-length 128                # 每 128 步检查
--per-head-pruning                 # 按头独立剪枝
```

**配置文件参数**:
```yaml
kv_budget: 2048
window_size: 128
sparse_round_window: 32
sparse_offset_max_length: 65536
sparse_score_aggregation: mean
num_samples: 8
seed: 888
```

### 5.2 差距对照表

| 功能点 | HF 实现 | vLLM 实现 | 差距 |
|-------|--------|----------|------|
| 压缩算法核心 | `speckv_rkv_style.py` | `triattention/` | ✅ 算法等价 |
| 打分公式 | PyTorch | Triton | ✅ 已验证等价 |
| 三种 pruning mode | ✅ | ✅ | ✅ 配置存在 |
| stats 加载 | ✅ | ✅ | ✅ |
| RoPE 反演 | 显式反演 | 优化版（K_rot） | ✅ 数学等价 |
| 位置追踪 | cache_positions_per_head | CompressionState | ⚠️ 需验证等价 |
| 触发条件 | slack_trigger | use_slack_trigger | ⚠️ 需验证等价 |
| **推理入口** | `run_math.py` + HF generate | `run_math_vllm.py` + vLLM | ❌ 未完成 |
| **结果格式** | JSONL | JSONL | ✅ 兼容 |

---

## 6. 实现 vLLM 等价推理的路线图

### Phase 1.1: 完成推理入口 (预计 2-3 天)

```
任务:
1. 完善 run_math_vllm.py
   - 添加 vLLM LLM 初始化
   - 集成 patch_vllm_attention()
   - 实现 generate() 调用
   - 输出 JSONL 结果

2. 参数对齐
   - 映射所有 HF 参数到 TriAttentionConfig
   - 确保 stats_path 正确传递

3. 环境配置
   - 添加 enforce_eager=True
   - 设置正确的 CUDA 设备
```

### Phase 1.2: 端到端验证 (预计 1-2 天)

```
任务:
1. 运行 HF 基线
   bash R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh

2. 运行 vLLM 版本
   bash TriAttention_vLLM/benchmarks/reasoning/run_triattention_aime24_perhead.sh

3. 结果对比
   python compare_results.py --hf-output <hf_result> --vllm-output <vllm_result>

验收标准:
- 准确率差异 < 1%
- token 匹配率 > 90% (确定性推理下应更高)
```

### Phase 1.3: 问题修复 (视情况)

```
根据验证结果修复:
- 参数映射错误
- 触发条件差异
- 数值精度问题
```

---

## 7. 建议的 TODO 重写

**文件**: `TriAttention_vLLM/docs/project/todo.md`

### 7.1 Phase 1 剩余任务 (P0)

- [ ] **完成 `run_math_vllm.py` vLLM 推理入口**
  - [ ] vLLM LLM 初始化代码
  - [ ] `patch_vllm_attention()` 集成
  - [ ] `generate()` 调用和结果收集
  - [ ] JSONL 输出格式对齐 HF 版本

- [ ] **参数映射验证**
  - [ ] 确认 `sparse_round_window` → `round_window` 映射
  - [ ] 确认 `rkv_style_compression` → 对应配置
  - [ ] 确认 `rkv_style_slack_trigger` → `use_slack_trigger`

- [ ] **端到端精度验证**
  - [ ] 运行 HF 基线 (AIME24, per-head mode)
  - [ ] 运行 vLLM 版本
  - [ ] 对比准确率，要求差异 < 1%

### 7.2 Phase 1 清理任务 (P1)

- [ ] **文档更新**
  - [ ] 移除 position_indices 相关描述（已废弃）
  - [ ] 更新架构图反映当前实现
  - [ ] 添加 vLLM 集成使用示例

- [ ] **代码清理**
  - [ ] 移除 position_indices 参数（保持 API 兼容但标记 deprecated）
  - [ ] 统一配置参数命名

### 7.3 Phase 2 待处理 (延后)

- [ ] Triton TopK/Gather 优化
- [ ] CUDA Graph 兼容
- [ ] Batch Size > 1 支持
- [ ] BF16 硬件验证 (需要 A100/H100)

---

## 8. 附录

### A. 文件结构

```
TriAttention_vLLM/
├── triattention/                    # 核心库 (2,827 行)
│   ├── config.py                    # ✅ 完成
│   ├── state.py                     # ✅ 完成
│   ├── compressor.py                # ✅ 完成
│   ├── scoring.py                   # ✅ 完成
│   ├── utils.py                     # ✅ 完成
│   ├── vllm_integration.py          # ✅ 完成
│   └── kernels/
│       └── triton_scoring.py        # ✅ 完成
├── test/                            # 测试套件 (20+ 文件)
│   └── ...
├── benchmarks/
│   └── reasoning/
│       ├── run_triattention_aime24_*.sh  # ✅ 框架
│       ├── run_math_vllm.py              # ❌ 未完成
│       └── compare_results.py            # ✅ 完成
└── docs/
    └── ...
```

### B. 关键依赖

```
vLLM >= 0.15.x
Triton >= 2.0
PyTorch >= 2.0
CUDA >= 11.8
```

### C. 测试命令

```bash
# 运行核心测试
cd TriAttention_vLLM
pytest test/test_scoring_kernel.py -v

# 运行等价性测试
pytest test/test_triton_pytorch_equivalence.py -v

# 运行全部测试
pytest test/ -v --ignore=test/benchmarks
```

---

*报告生成: 2026-02-01*
*下次更新: 完成 Phase 1.1 后*
