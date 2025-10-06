# Development Workspace

This directory collects ad-hoc utilities that should not touch the main codebase.

- `offline_analysis/`: tooling for investigating offline DeepThink traces. See the nested README for details.
- `example_offline_serialized.py`: 离线示例副本，默认使用 msgpack+gzip 保存结果，兼容原始 pickle。
- `serialization_utils.py`: 将 DeepThink 结果结构化为可快速解包的 msgpack，同时保留所有原始信息。
- `offline_result_loader.py`: 命令行工具，快速加载 `.msgpack.*` 或 `.pkl` 结果、可选导出回 pickle。
- `example_analyze_offline_serialized.py`: 离线结果分析脚本，支持直接读取 msgpack 压缩并输出统计报表。
- 序列化版调度脚本挪到 `scripts/yaml_runs_serialized/`，与原 YAML 流程并行使用。
