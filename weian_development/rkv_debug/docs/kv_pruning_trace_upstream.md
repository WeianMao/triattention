## KV 压缩可视化上游数据链

- 下游脚本：`weian_development/online_k_pruning_viz/attention_pruning_case_study_hybrid_rounds_xtrace.py` 在 `main()` 中读取 `input_root/<trace>/qk.pt` 和 `metadata.json` 作为模拟与可视化输入。
- 产出来源：`weian_development/attention_qk_analysis/capture_qk_distributed.py` 在前向 pre-hook 里捕获 Qwen3 的 post-RoPE Q/K，并落盘 `qk.pt`（`{"q": Tensor[L,H,T,D], "k": Tensor[L,H,T,D]}`）与 `metadata.json`。
- 路径示例：`outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34/qk.pt`（同目录有 `metadata.json`）。
- 生成流程：
  1) 构建 manifest：`python weian_development/attention_qk_analysis/build_manifest.py outputs/deepseek_r1_qwen3_8b/offline_reasoning_json --output weian_development/attention_qk_analysis/trace_manifest.json --count 8 --seed 20250107 --verbose`
  2) 捕获 Q/K：`python weian_development/attention_qk_analysis/capture_qk_distributed.py weian_development/attention_qk_analysis/trace_manifest.json outputs/deepseek_r1_qwen3_8b/qk_bf16_traces --gpus 0,1 --precision bfloat16 --verbose`
- 关联说明：`weian_development/attention_qk_analysis/README.md` 描述了上述命令、输出结构以及资源提示；同文件 258-318 行包含写盘 `qk.pt`/`metadata.json` 的核心逻辑。`attention_pruning_case_study_hybrid_rounds_xtrace.py` 644-687 行检查并加载这些文件。***
