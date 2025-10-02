# 深入定制内容检查清单

> 勾选需要在新版中重新实现或迁移的能力。

- [需要] **YAML 配置体系** (`deepconf_old/development/config_loader.py`, `development/configs/*.yaml`)
  - 支持多层默认值合并、按模式（baseline/offline/online）拆分配置。
- [不需要，按照新的版本结构来] **新版主脚本骨架** (`development/deepconf-online.py`, `development/deepconf-offline.py`, `development/deepconf-baseline.py`)
  - CLI 传参 + 配置读取 + 统一结果结构 + 预置阈值。
- [需要] **批量调度与 GPU 派发脚本** (`weian_script/yaml_runs/run_offline_deepseek_dispatch.py` 等)
  - 自动分配 QID、跳过已完成题、注入 `CUDA_VISIBLE_DEVICES`。
- [不需要] **预设置信度阈值与运行文档** (`development/deepconf-online.py:46`, `development/trace_truncation.md`, `development/weian_commit_history.md`)
  - BRUMO 题目置信度表、早停逻辑说明、自研迭代记录。
- [不需要] **核心辅助函数** (`deepconf_old/helper.py`)
  - 答案抽取、`dynasor` 判断、prompt 构造、置信度计算、批处理。
- [不需要] **分析与统计脚本** (`analysis_offline.py`, `analysis_online.py`, `analysis/context_vs_decode_speed.py`)
  - 汇总准确率、token、时间，辅助可视化或对比分析。
- [需要，重新实现的时候需要按照一模一样的模版来，但是可能要根据新版项目提供的入口调整每个脚本对应的实验] **运行脚本模板** (`weian_script/original/`, `weian_script/yaml_runs/*.sh`)
  - 快捷启动单题或全量测试的 Shell 脚本集合。

