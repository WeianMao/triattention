# SpeckV-RKV Cache 用量对齐计划

## 用户需求

在 `--rkv-style-compression` 激活时，实现 SpeckV 和 R-KV 在**实际 cache 用量**上的对齐，同时：
- **不修改**不激活此 arg 时的代码和实现
- 实现代码隔离的开发

---

## 当前状态分析

### R-KV 的行为（HuggingFace 实现）

1. **外层控制**（`CausalLM_forward`）：
   - 通过 `divide_length=128` 控制压缩频率
   - 每 128 步设置 `compression = True`

2. **内层执行**（`Attention_forward`）：
   - 只有当 `compression = True` 时才调用 `update_kv`
   - 其他步骤只添加新 token，不压缩

3. **`update_kv`**：
   - 触发条件：`kv_cache_len >= budget`
   - 压缩后大小：`budget`

4. **实际 cache 用量**：
   - 波动范围：`budget ~ budget + divide_length`
   - 默认：`2048 ~ 2176`（divide_length=128）

### SpeckV 当前行为（`--rkv-style-compression`）

1. **`speckv_rkv_forward`**：
   - 每步检查 `effective_size >= budget`
   - 每步都调用 `update_kv`

2. **`update_kv`**：
   - 压缩后大小：`budget`

3. **实际 cache 用量**：
   - 始终 = `budget`（每步都压缩）

### 差异总结

| 项目 | R-KV | SpeckV (当前) |
|------|------|---------------|
| 压缩频率 | 每 128 步 | 每步 |
| cache 波动 | budget ~ budget+128 | 始终 = budget |
| 平均 cache | ~budget+64 | budget |

---

## 对齐方案

### 方案：添加 `divide_length` 参数

在 `speckv_rkv_style.py` 中添加类似 R-KV 的 `divide_length` 控制逻辑：

1. **新增配置参数**：
   - `divide_length: int = 128`（默认和 R-KV 一致）

2. **修改 `speckv_rkv_forward`**：
   ```python
   # 添加步数计数器
   comp.step_count = getattr(comp, 'step_count', 0) + step

   # 只有每 divide_length 步才压缩
   should_compress = (comp.step_count % comp.divide_length == 0)

   if effective_size >= comp.budget and should_compress:
       # 执行压缩
   ```

3. **对齐后行为**：
   - 压缩频率：每 128 步
   - cache 波动：`budget ~ budget + 128`
   - 和 R-KV 一致

### 代码隔离

- 只修改 `speckv_rkv_style.py`
- 不影响 `apply_speckv_generate_patch`（非 aligned 路径）
- 新参数通过现有的 arg 传递链传入

### 需要修改的文件

1. `R-KV/weian_development/speckv/speckv_rkv_style.py`
   - `SpeckVRKVStyleConfig`: 添加 `divide_length` 字段
   - `SpeckVRKVStyle`: 添加 `step_count` 状态
   - `speckv_rkv_forward`: 添加压缩频率控制逻辑
   - `apply_speckv_rkv_style_patch`: 添加 `divide_length` 参数

2. `R-KV/weian_development/rkv_sharded_eval.py`
   - 添加 `--divide-length` 参数
   - 传递给 `apply_speckv_rkv_style_patch`

3. `R-KV/weian_development/rkv_sharded_dispatch.py`
   - 添加 `--divide-length` 参数转发

---

## 确认事项

1. `divide_length` 默认值设为 128（和 R-KV 一致），是否正确？
2. 是否需要支持 `divide_method = "newline"`（遇到换行符才压缩）？
3. 其他需要考虑的对齐点？

---

请确认上述理解和计划是否正确，确认后我将开始实现。
