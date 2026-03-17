# Chat Template 一致性分析

## 结论

**FullKV、RKV、SpeckV 三种方法在 chat template 处理上是一致的 — 都不使用 chat template。**

## 配置对比

| 方法 | 配置文件 | `use_chat_template` 设置 | 实际使用值 |
|------|---------|-------------------------|-----------|
| FullKV | `aime_sampled8_fullkv_aime24_qwen.yaml` | 未设置 | `False`（硬编码） |
| RKV | `aime_sampled8_rkv_aime24_qwen.yaml` | 未设置 | `False`（硬编码） |
| SpeckV | `aime_sampled8_speckv_aime24_qwen_norm.yaml` | `false` | `False`（硬编码） |

## 代码分析

### 问题位置

`R-KV/weian_development/rkv_sharded_eval.py:379`

```python
prompt_use_chat = False   # <-- 硬编码，忽略了 args.use_chat_template
prompts, test_data = load_dataset(
    ...
    use_chat_template=prompt_use_chat,  # 总是 False
    ...
)
```

### 影响

- `args.use_chat_template` 参数存在但**未被使用**于 prompt 构建
- 该参数仅在 SpeckV 分支用于验证（必须为 False），见第 371-372 行：
  ```python
  if bool(args.use_chat_template):
      raise ValueError("SpeckV uses the plain R-KV prompt; use_chat_template must be False.")
  ```

## 实际使用的 Prompt 模板

三种方法都使用 `R-KV/weian_development/speckv/prompt_utils.py` 中的 plain prompt：

```
You are given a math problem.

Problem: {question}

You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.

Provide the final answer in the format: Final answer:  \boxed{}
```

不调用 HuggingFace 的 `tokenizer.apply_chat_template()`。

## 评估

| 评估维度 | 状态 | 说明 |
|---------|------|------|
| 三方法一致性 | ✅ | prompt 格式完全一致，可公平对比 |
| 模型最优使用 | ⚠️ | DeepSeek-R1-Distill-Qwen-7B 是 instruct 模型，不用 chat template 可能次优 |
| 实验有效性 | ✅ | 三者受影响程度相同，对比实验仍有效 |

## 建议

如需启用 chat template，需修改 `rkv_sharded_eval.py:379`：

```python
# 当前（硬编码）
prompt_use_chat = False

# 修改为（使用配置参数）
prompt_use_chat = args.use_chat_template
```

同时需要：
1. 移除 SpeckV 分支的 `use_chat_template` 强制检查（第 371-372 行）
2. 重新校准 SpeckV 统计文件（使用相同的 chat template 设置）
