# Triton BF16 编译错误修复

**日期**: 2026-02-03
**状态**: 已修复，待验证

---

## 问题描述

端到端测试中 vLLM TriAttention 输出质量显著优于 HF Baseline（本应相似），原因是 Triton kernel 因 bf16 dtype 编译失败，导致压缩未实际执行。

### 错误信息

```
triton.compiler.errors.CompilationError: at 80:17:
ValueError: Expected dtype ['fp32', 'fp64'] but got bf16
```

### 调用链分析

1. `scoring.py:134`: `omega_input = omega.to(dtype=key_states.dtype)` - 将 omega 转为 bf16
2. `triton_scoring.py:230`: `omega = tl.load(omega_ptr + f_offs, ...)` - 加载 bf16
3. `triton_scoring.py:303`: `phase = t * omega` - phase 变为 bf16
4. `triton_scoring.py:308-309`: `tl.cos(phase)` / `tl.sin(phase)` - **失败**

**根本原因**: Triton 三角函数 (`tl.cos`, `tl.sin`) 只支持 fp32/fp64，不支持 bf16。

---

## 影响

- 压缩触发但执行失败（被 try-except 捕获）
- 模型继续运行但使用**完整 KV cache**（无压缩）
- 输出质量好是因为没有压缩

### 验证证据

```
[TriAttention] Compressing: seq_len=320 -> budget=256   ← 触发正确
[TriAttention] Compression error for batch 0, layer 0: at 80:17:  ← 所有层失败
[TriAttention] Compression error for batch 0, layer 1: at 80:17:
... (28层全部报错)
```

---

## 修复方案

在 Triton kernel 内部添加 `.to(tl.float32)` 类型转换。

### 修改文件

`triattention/kernels/triton_scoring.py`

### 修改内容

**1. Line 230 - omega 加载 (主 kernel)**
```python
# 修复前
omega = tl.load(omega_ptr + f_offs, mask=f_mask, other=0.0)

# 修复后
omega = tl.load(omega_ptr + f_offs, mask=f_mask, other=0.0).to(tl.float32)
```

**2. Line 295 - offset 加载 (主 kernel)**
```python
# 修复前
offset = tl.load(offsets_ptr + off_idx)

# 修复后
offset = tl.load(offsets_ptr + off_idx).to(tl.float32)
```

**3. Line 414 - omega 加载 (预计算表 kernel)**
```python
omega = tl.load(omega_ptr + f_offs, mask=f_mask, other=0.0).to(tl.float32)
```

**4. Lines 465-466 - cos/sin 表加载**
```python
cos_t_omega = tl.load(cos_table_ptr + ...).to(tl.float32)
sin_t_omega = tl.load(sin_table_ptr + ...).to(tl.float32)
```

---

## 验证命令

```bash
VLLM_PROCESS_NAME_PREFIX="PD-L1_binder" CUDA_VISIBLE_DEVICES=6 \
python TriAttention_vLLM/benchmarks/reasoning/run_math_vllm.py \
  --model /data/rbg/weights/DeepSeek-R1-Distill-Qwen-7B \
  --kv-budget 256 --divide-length 64 --num-questions 2
```

### 预期结果

- 看到 `[TriAttention] Compressed: 320 -> 256 tokens`（成功信息）
- vLLM 输出质量应与 HF Baseline 相似（kv_budget=256 下都会退化）

---

## 测试输出文件

- HF Baseline: `/tmp/hf_baseline_2q.jsonl`
- vLLM TriAttention: `/tmp/triattention_e2e_test/vllm_results.jsonl`
- 测试日志: `/tmp/claude-28613/-data-rbg-users-weian-project-rl-dc/tasks/b5ae0a0.output`

---

## 相关文件

- `triattention/kernels/triton_scoring.py` - Triton kernel 实现
- `triattention/scoring.py` - 打分逻辑入口
- `docs/backend/reference/fixes/FP32_EQUIVALENCE_FIX.md` - 相关精度问题
