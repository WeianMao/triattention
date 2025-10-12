# HuggingFace StreamingLLM 集成开发计划

## 总体目标
- 在保持现有 DeepConf 推理与评估接口不变的前提下，新增一套使用 HuggingFace + StreamingLLM 的离线推理与评估链路。
- 对齐原有 DeepThink 离线路径的配置、prompt 构造、序列化格式和分析脚本输入，便于直接比较结果。
- 全程与现有 vLLM 流程隔离开发，避免回归；每个阶段结束需有可执行的最小验证案例。

## 开发步骤与测试目标
1. **环境探测与模型加载验证**
   - 准备事项：确认 `streaming-llm` 子模块可用、当前环境能加载目标模型（如 DeepSeek-R1-Qwen3-8B）。
   - 主要工作：编写独立的 notebook/脚本（临时文件即可），调用 `AutoModelForCausalLM` + `enable_streaming_llm` 完成一次简短推理。
   - 验证目标：`python tmp_streaming_smoke.py --prompt "1+1?"` 可输出正常文本，日志显示 StreamingLLM 被启用。

2. **Prompt 与序列化流程复刻**
   - 主要工作：在 `development/` 新增 `example_offline_streaming_hf.py`（名称可调整但不得覆盖旧文件），复用 `development/example_offline_serialized.py` 中的 prompt 函数、Sampling 参数与 msgpack 序列化逻辑。
   - 验证目标：对单个 `--qid` 运行脚本，生成的 msgpack 文件结构与旧版一致（可用 `development/example_analyze_offline_serialized.py --output_dir ... --max_qid 0` 验证能否解析）。

3. **Streaming 推理循环实现**
   - 主要工作：在新脚本中实现批量 trace 生成，集成 `StartRecentKVCache`，确保 Streaming 模式下 logprob、confs 等字段与旧版输出对齐；日志需明确提示启用 Streaming。
   - 验证目标：`python development/example_offline_streaming_hf.py --qids 0 --budget 4` 单卡运行成功，输出 token 数、耗时等统计；对比旧版结果字段一致。

4. **运行脚本与配置隔离**
   - 主要工作：在 `scripts/` 下新增 `yaml_runs_streaming/run_offline_deepseek_streaming_hf.sh` 以及对应 YAML（如 `scripts/configs/streaming/deepseek_r1_qwen3_8b_streaming_64.yaml`），参数命名与旧版一致并新增 Streaming 专属字段（start/recent window）。
   - 验证目标：`bash scripts/yaml_runs_streaming/run_offline_deepseek_streaming_hf.sh --rid test --gpus 0 --qids 0` 可驱动新脚本，无需修改原有 YAML 或 shell。

5. **评估脚本对齐**
   - 主要工作：若旧评估脚本无需改动，则在 `scripts/analyze/` 新增 wrapper（如 `run_offline_msgpack_analysis_streaming.sh`）指向现有 `development/example_analyze_offline_serialized.py`；如需特殊指标，则新建隔离分析脚本。
   - 验证目标：运行新分析脚本可读取步骤 3 产出的数据并输出与旧流程相同的统计字段。

6. **回归确认与文档**
   - 主要工作：执行 `python -m compileall deepconf development` 确认语法；运行旧版 vLLM 流程 smoke test 确保未受影响；在 `docs/` 新增说明文件记录使用与配置。
   - 验证目标：
     - 旧脚本 `scripts/yaml_runs_serialized/run_offline_deepseek_msgpack_64.sh --qids 0` 能正常运行。
     - 新文档列出 Streaming 参数说明、注意事项。

## 关键提醒
- 严格杜绝修改原有脚本和配置，只能复制扩展；共用的工具函数若必须调整，需评估对旧流程的影响并同步测试。
- HuggingFace 推理默认不支持 tensor parallel；若 POC 单卡无法容纳模型，可考虑先选更小模型或激活 4-bit/8-bit 加载，再计划多卡方案。
- Streaming 调参（start_size/recent_size）需提供合理默认值，并在日志提示当前配置。
- 评估阶段需关注答案准确率是否因窗口限制下降；如有差异，应在文档中记录并解释。
