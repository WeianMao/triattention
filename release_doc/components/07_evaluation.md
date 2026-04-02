# 评估管线

## 决策状态：已确认

## 公布的评估文件（13个 + latex2sympy/）

| 文件 | 来源 | 作用 |
|------|------|------|
| `evaluate.py` | 社区+自定义 | 核心评估函数 |
| `eval_math.py` | 自定义 | 单样本评估 CLI 入口 |
| `eval_math_multi.py` | 自定义 | 多样本 pass@k 评估（被启动器直接调用） |
| `grader.py` | 社区 | 数学等价性判断 |
| `parser.py` | 社区+自定义 | 答案提取（有 AIME 引用需清理） |
| `math_utils.py` | 社区+自定义 | sympy 数学工具 |
| `python_executor.py` | 自定义 | 安全代码执行 |
| `data_loader.py` | 自定义 | 数据集加载 |
| `utils.py` | 自定义 | 工具函数（有 AIME 引用需清理） |
| `trajectory.py` | 自定义 | chain-of-thought 解析 |
| `examples.py` | 社区 | few-shot 示例 |
| `model_utils.py` | 社区 | 模型加载/生成 |
| `rm_maj_eval.py` | 自定义 | majority voting（清理 __main__ 硬编码路径） |
| `run_math.py` | 自定义 | 推理入口脚本（需重构 import weian_development.* → 使用 release 包名） |
| `latex2sympy/` | 第三方 | LaTeX 解析库 |

## 不公布的评估文件

| 文件 | 原因 |
|------|------|
| `length_eval.py` | 硬编码内部目录结构，外部用户无法使用 |
| `CHANGELOG_weian.md` | 内部开发日志，含 PD-L1_binder 等敏感信息 |
| `evaluation/aime24/` 等结果缓存 | 运行时产物 |
| `.DS_Store`、`__pycache__` | 系统文件 |

## 需要清理的内容

- `parser.py`、`utils.py` 中的 AIME 引用需泛化
- `run_math.py` 中 `import weian_development.*` 需重构
- `rm_maj_eval.py` 中 `__main__` 硬编码路径需清理
- `model_utils.py` L448 中的 `../models/codellama_7b/v1-16k` 相对路径需清理

## 第三方代码归属（Attribution）

Release 为 Apache 2.0，所有上游许可证均兼容。需按 `grader.py` 格式在文件头添加来源声明。

### 需要添加 attribution header 的文件

| 文件 | 来源 | 上游许可证 |
|------|------|-----------|
| `parser.py` | ToRA + DeepSeek-Math | MIT |
| `examples.py` | ToRA + DeepSeek-Math | MIT |
| `python_executor.py` | ToRA (`microsoft/ToRA/blob/main/src/eval/python_executor.py`) | MIT |
| `data_loader.py` | ToRA + DeepSeek-Math | MIT |
| `utils.py` | ToRA + DeepSeek-Math | MIT |
| `evaluate.py` | ToRA + DeepSeek-Math | MIT |

### 已有 attribution 但需补全

| 文件 | 来源 | 现状 | 需补充 |
|------|------|------|--------|
| `model_utils.py` | allenai/open-instruct | 仅有 URL（L2） | 补充许可证声明（Apache 2.0） |
| `grader.py` | ToRA + DeepSeek-Math + Hendrycks MATH + CRITIC + PRM800K | ✅ 完整 | 无需修改（作为格式参考） |

### 原创文件（无需 attribution）

`math_utils.py`, `trajectory.py`, `rm_maj_eval.py`, `eval_math.py`, `eval_math_multi.py`

### 上游项目参考 URL

- ToRA: `https://github.com/microsoft/ToRA` (MIT)
- DeepSeek-Math: `https://github.com/deepseek-ai/DeepSeek-Math` (MIT)
- open-instruct: `https://github.com/allenai/open-instruct` (Apache 2.0)
- latex2sympy: MIT（已内含许可证）
