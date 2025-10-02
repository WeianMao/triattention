# DeepConf 新旧版本差异与迁移建议

## 1. 项目结构概览
- **新版（仓库根目录）**：对外提供 `deepconf` Python 包，核心入口是 `DeepThinkLLM` 类，并附带 `examples/` 下的示例脚本。核心文件：`deepconf/wrapper.py:24`、`deepconf/utils.py:16`、`deepconf/outputs.py:13`、`deepconf/processors.py:58`。
- **旧版（`deepconf_old/`）**：仍沿用官方发布时的脚本式结构，并在此基础上进行了大量定制开发（YAML 配置、调度脚本等）。代表文件：`deepconf_old/development/deepconf-online.py:200`、`deepconf_old/development/deepconf-offline.py:60`、`deepconf_old/helper.py:7`、`deepconf_old/weian_script/yaml_runs/run_offline_deepseek_dispatch.py:21`。

## 2. 核心算法与脚本差异
- **统一包装器取代多脚本**  
  新版将在线/离线逻辑封装为 `DeepThinkLLM` 的 `deepthink()` 方法（`deepconf/wrapper.py:69`），调用者只需根据模式传参；旧版则分别使用 `deepconf-online.py`、`deepconf-offline.py`、`deepconf-baseline.py` 三个脚本管理全流程。
- **置信度控制实现方式调整**  
  旧版脚本通过向 `SamplingParams.extra_args` 写入 `enable_conf/threshold` 等字段触发早停；新版改为注册自定义 logits processor（`deepconf/processors.py:58`），在官方 vLLM 上即可使用 `params.extra_args={'conf_threshold': ...}` 控制在线置信度截断。
- **输出与统计数据结构升级**  
  新版将推理结果封装为 `DeepThinkOutput` 数据类（`deepconf/outputs.py:13`），默认提供 `to_dict()` 与 `print_summary()`；旧版脚本直接 dump Python dict（例如在线脚本保存 `pickle`，字段名如 `voted_answer`、`token_stats` 等）。
- **工具函数与多重投票扩展**  
  新版统一了答案抽取、置信度计算、投票策略等工具函数，集中在 `deepconf/utils.py:16`，并默认支持多种投票策略（含 top10 过滤、底窗口加权）；旧版逻辑散落在 `helper.py` 与分析脚本中，默认仅做简单加权多数。
- **官方示例与 CLI**  
  新版的 `examples/example_online.py:205` 等脚本展示了如何结合 `DeepThinkLLM`、`dynasor` 评估以及输出分析；旧版示例主要在 `README-online.md` / `README-offline.md`，需要运行对应脚本获得结果。

## 3. 兼容性评估（迁移风险点）
- **vLLM 配置**  
  旧版脚本假定 vLLM 支持 `extra_args['enable_conf']` 与 `threshold` 等字段；迁移时需确认目标环境的官方 vLLM 版本已加载 `ConfPerReqLogitsProcessor`，并在配置层插入 `conf_threshold`、`conf_group_size`、`conf_topk`。
- **结果文件格式**  
  旧版 `outputs/` 目录保存的多为 `.pkl`，字段与新版 `DeepThinkOutput.to_dict()` 输出存在差异。迁移时，所有下游分析工具（如 `analysis_online.py`、`analysis_offline.py`）需适配新字段名与结构。
- **配置体系**  
  旧版的 YAML 加载器（`deepconf_old/development/config_loader.py:19`）及 `deepseek_r1_qwen3_8b_all.yaml`（`deepconf_old/development/configs/deepseek_r1_qwen3_8b_all.yaml:1`）自定义了参数格式与命名，新版默认通过 Python 调用传参；若继续沿用 YAML，需要编写新的绑定层，把 YAML 字段映射到 `DeepThinkLLM.deepthink()` 的参数。
- **运行/调度脚本**  
  如多 GPU 调度器 `run_offline_deepseek_dispatch.py`（`deepconf_old/weian_script/yaml_runs/run_offline_deepseek_dispatch.py:21`）假定被调用脚本为 `development/deepconf-*.py`，且输出文件名遵循旧版规则。迁移后要保证它调用的新入口脚本能维持原有的文件命名与 exit code 约定。

## 4. 迁移建议步骤
1. **包装新版核心接口**  
   在 `deepconf_old/development/` 下创建新的桥接脚本（可命名为 `deepconf_v2_online.py` 等），内部调用 `DeepThinkLLM` 并接受原有 YAML/CLI 参数，保持旧脚本的 CLI 选项与输出文件命名，这样现有调度与分析工具无需立即改造。
2. **统一 vLLM 配置**  
   确认目标机器上安装的官方 vLLM 支持自定义 logits processor；在 YAML 或 CLI 中，将原先的 `extra_args={'enable_conf': True, 'threshold': ...}` 改写为传入 `conf_threshold`、`conf_group_size`（等于窗口大小）、`conf_topk`（默认 20）。
3. **适配输出与分析链路**  
   将新的 `DeepThinkOutput.to_dict()` 序列化结果保存为 JSON/PKL，并在分析脚本（如 `analysis_online.py`）中增加新版字段解析逻辑，保证统计指标（精度、token 数、耗时）与旧版一致。
4. **迁移配置与调度器**  
   更新 `load_config()` 逻辑，使其既能驱动旧版脚本也能驱动新版包装器；调度器调用的新脚本需支持旧的 `--qid/--rid`、GPU 绑定方式以及输出目录层次，以便直接替换。
5. **端到端验证**  
   选取代表性数据集（如 `brumo_2025` 与 `aime25`），在同一题目上分别运行旧版与包装后的新版脚本，核对：  
   - voting 答案是否一致；  
   - token 消耗与生成时间是否在可接受范围；  
   - 早停是否触发、触发统计是否一致。  
   确认一致性后，再逐步切换生产环境。

## 5. 后续验证与清理建议
- **回归测试**：建立最小化 smoke 流程（在线/离线各 1 道题）作为 CI 或手动验收脚本，确保迁移过程中新增代码保持可运行。
- **文档更新**：在内部 wiki 或 README 中记录新版调用方式、配置项对照表、常见问题排查步骤。
- **逐步淘汰旧脚本**：待新版脚本稳定后，将 `deepconf_old/development/*` 与 `weian_script/original` 中不再使用的脚本标记为 deprecated，并规划最终删除时间，避免维护成本。

## 6. `examples/` 目录脚本说明
- `examples/README.md`：汇总示例脚本的运行方式、基准结果与数据准备流程，可作为快速上手指南。
- `example_online.py`：演示新版在线模式的完整流程，涵盖 prompt 构造、`DeepThinkLLM` 推理、投票结果评估以及控制台报告输出。
- `example_online_baseline.py`：提供无置信度早停的对照实验，方便对比在线模式的 token 消耗与准确度。
- `example_offline.py`：示范离线批量生成的使用方法，生成并保存全量推理 trace，适合做投票或可视化分析。
- `example_analyze_online.py`：对在线模式的运行结果目录做汇总统计（准确率、token 使用、耗时），支持多次运行聚合。
- `example_analyze_online_baseline.py`：与上一脚本对应，用于基准模式结果的批量分析。
