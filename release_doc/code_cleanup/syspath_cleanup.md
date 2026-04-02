# sys.path 清理方案

## 决策状态：已确认

### 调查结果

全项目 363 处 sys.path 调用，release 范围内约 60 处需修。

### 分级处理

| 优先级 | 数量 | 问题 | 处理方法 |
|--------|------|------|---------|
| CRITICAL | 1 | `R-KV/SGLang/eval.py` 硬编码 `/tmp/kewan/...` | 删除或改为环境变量 |
| HIGH | 16 | R-KV/ 和 TriAttention_vLLM/ 中的项目根目录 hack | 完善 setup.py，用 `pip install -e .` 代替 |
| MEDIUM | 18 | latex2sympy 相对导入 hack（3 个副本各 7 处） | 添加 `__init__.py`，正规包导入 |
| LOW | 13+ | 文档构建/第三方代码 | 保留不动 |

### 执行方案

代码清理阶段统一处理：
1. 完善 `setup.py` / `pyproject.toml` 包声明（使用 `find_packages()`）
2. 删除所有 release 范围内的 sys.path hack
3. 用户通过 `pip install -e .` 安装
4. 预计工作量：~1 天
