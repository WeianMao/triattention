# 第二阶段：kvpress 相关代码

## 决策状态：待确认（第一阶段完成后执行）

kvpress 相关代码目前只在 `dc1/rebuttal` 分支，需先转移到 main 上，第一阶段公布完后再检查并公布。

## 转移到 main 的内容（triattention_press/）

- `triattention_press.py` -- 主算法
- `triattention_press_v2.py` -- V2 生产版本
- `scoring.py` -- 频域评分引擎
- `__init__.py` -- 导出接口
- `extra_weight.py`（improvements/ 中唯一有活跃依赖的）
- 测试文件（在主分支上通过后才 release）
- 校准结果 .pt 文件（改名去 aime）
- 最终版实验脚本：
  - `run_longbench_qwen3_final_v2.sh`（LongBench 最终结果）
  - `run_clnorm_only_full_12subtask.py`（LongBench 最新最优）
  - `run_ruler_norm_v2.py`（RULER 最终结果）

## 不转移 / 不公布

- `improvements/` 其他文件（功能已合并到主类，中间产物）
- 废弃的 `score_norm.py`
- 消融/历史实验脚本（ablation A/B/D/F 等探索性实验）
- 校准脚本（`calibrate_qwen3.py` 等）
- `kvpress/` 外部库（作为依赖让用户自己安装，不包含在 repo 中）

## 最终报告的配置

| 场景 | 配置 | 得分 |
|------|------|------|
| LongBench 最优 | `aligned_max_a1.5_w8`（alpha=1.5, max+max, window=8） | 42.35 |
| RULER 最优 | `aligned_max_a1.5_w8_norm`（加 normalize_scores=True） | 66.08 |

## 第二阶段代码整理要求

与第一阶段相同的原则（参见 [../code_cleanup/04_naming.md](../code_cleanup/04_naming.md)）：**公布的代码是全新整理过的，不是内部代码的直接 copy。**

具体要求：
- **去掉实验性功能**：没有在最终版本使用的功能代码要删除（如废弃的 score_norm.py、kernel_pooling 如果最终未采用等）
- **命名规范化**：内部开发过程中的不规范命名要改成正式名字（如 `extra_weight` 改为正式术语）。不能暴露中间实验过程的命名
- **文件整理**：中间开发文件不直接公布，该合并的合并（如 extra_weight.py 的逻辑合并到主类）、该删的删
- **improvements/ 目录不公布**：功能已内化到主类，目录本身是开发过程产物
- **同样遵守重构原则**：能小改就小改，重构前报告，重构后 AB 测试 + 单元测试
