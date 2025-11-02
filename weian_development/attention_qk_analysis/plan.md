# DeepSeek 离线 Trace Q/K 捕获计划

## 目标
- 复现 DeepSeek-R1 Qwen3-8B 离线推理时的提示构造，确保重放历史 trace 时无需修改 `text-generation-inference` 相关后端代码。
- 从 `outputs/deepseek_r1_qwen3_8b/offline_reasoning_json` 中随机挑选 32 个不同问题，每题只取一条 trace，将完整文本重新送入模型，截获 RoPE 之后的注意力 Q/K 张量。
- 将所有层的 Q/K（连同必要的元数据）序列化到 `.pt` 文件，后续分析脚本直接离线读取即可。

## 拟定流程
1. **Trace 抽样与元数据整理**
   - 遍历 `outputs/deepseek_r1_qwen3_8b/offline_reasoning_json` 下所有 JSON，按 `qid` 去重。
   - 使用固定随机种子（如 20250107）抽取 32 个独立问题，并为每个问题随机选取一条 trace。
   - 记录 `(qid, source_file, trace_index)` 到 manifest，存放在 `weian_development/attention_qk_analysis` 目录。

2. **Prompt 重建**
   - 载入 `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B` 的 tokenizer（`trust_remote_code=True`）。
   - 按 `development/example_offline_serialized.py` 中 `model_type="deepseek"` 的流程拼接系统提示 + 用户问题，通过 `apply_chat_template` 补上生成前缀。
   - 将抽样 trace 的完整文本（含 `<think>` 标签）直接追加在提示后，作为模型输入。

3. **模型 Hook 布置**
   - 在新目录内使用 HuggingFace `AutoModelForCausalLM.from_pretrained` 实例化模型，避免触碰后端。
   - 检查各层自注意力模块（如 `model.model.layers[i].self_attn`），在 RoPE 应用之后的位置注册前向 hook，捕获处理后的 `query_states`、`key_states`。
   - 确认张量形状为 `(batch=1, num_heads, seq_len, head_dim)`，为节约存储量可以转换为 `float16`。

4. **前向推理与数据收集**
   - 将重建文本 `tokenizer` 成张量，`return_tensors="pt"`，不添加额外 special token，优先放在 GPU（若可用）。
   - 在 `torch.no_grad()` 下执行一次前向推理，hook 自动收集各层 Q/K。
   - 将层列表堆叠成张量，移到 CPU，并补充辅助信息（token ids、attention mask、prompt 长度等）。

5. **结果序列化**
   - 每条 trace 构造结构化 payload：`{"qid": int, "source_file": str, "trace_index": int, "prompt_tokens": int, "total_tokens": int, "q": Tensor[L,H,T,D], "k": Tensor[L,H,T,D]}`。
   - 32 条数据合并为字典（以 `qid` 为键），使用 `torch.save` 写入 `weian_development/attention_qk_analysis/qk_captures.pt`，并生成 JSON 辅助文件记录随机种子、模型版本、tokenizer revision 等。

6. **校验与日志**
   - 重新加载 `.pt` 文件，检查层数、头数、序列长度是否与模型配置一致，确认抽样结果稳定。
   - 输出汇总信息（token 数、最大序列长度等）到 markdown 或 JSON 报告，便于追踪。

7. **进程规范**
   - 所有脚本位于 `weian_development/attention_qk_analysis/`，在运行时调用与 `mask_process_command` 同类的机制，将进程名伪装为 `PD-L1_binder`，符合集群约定。
   - 不修改 `text-generation-inference` 等 backend 文件；仅在用户目录下新增脚本与数据。

## 待确认关键点
- **RoPE 位置**：目前计划保存 RoPE 之后的 Q/K，请确认此偏好是否最终决定。
- **输出形式**：默认使用单一 `.pt` + JSON manifest。若更希望每条 trace 单独成文件，请告知。
- **额外数据**：是否需要同时提取注意力权重（attention score）或只保留 Q/K？
- **设备使用**：默认使用 `cuda:0`。如需 CPU-only 或指定 GPU，请提前说明。

确认上述事项后，我将按该计划继续实现与执行。
