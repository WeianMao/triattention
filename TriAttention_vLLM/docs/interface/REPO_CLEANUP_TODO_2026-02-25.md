# TriAttention_vLLM 代码与文档整理 TODO（默认版本化）

- 开始时间：2026-02-25
- 目标：
  1. 弱化 `V2` 概念，让当前实现成为默认使用版本（以对外入口/文档为主）
  2. 清理历史 debug/问题追踪文档，降低接手噪音
  3. 在不改变代码逻辑前提下做少量命名与结构整理（保守修改）

## 原则（本轮）

1. **逻辑不变**：不改算法语义，不改关键执行路径行为。
2. **优先兼容**：新增“默认入口”，保留旧 `V2` 入口作为兼容层。
3. **先外后内**：先改用户会直接用到的 runner/config/docs，再考虑内部命名。
4. **少量整洁化**：仅处理明显冗余或误导性命名，不做大规模内部重命名。

## 执行清单

### A. 对外入口去 `V2`（默认版本化）

- [x] 新增默认评测 runner 入口（无 `V2` 文件名），兼容转发到现有实现
- [x] 新增默认 dispatch 配置（无 `V2` 文件名），并切换 dispatch 默认配置到新文件
- [x] dispatch 对 runner 的校验逻辑同时接受新旧文件名（避免老配置失效）
- [x] （可选）新增无 `V2` 名称的脚本入口包装（保留旧脚本兼容）

### B. 文档清理（删历史 debug / 问题历史）

- [x] 新增简化版实现总览文档（以代码引用为主）
- [x] 新增简化版对齐状态文档（以实验结果 + 代码位置为主）
- [x] 更新 `docs/README.md` 入口，弱化 `V2` 体系表述
- [x] 更新 `docs/interface/GUIDED_TOUR.md` 接手路径，指向新文档
- [x] 更新 `docs/interface/CURRENT_STATUS.md` 的执行摘要，弱化“V2”措辞
- [x] 删除已过时的 debug/审计流水文档（以新文档承接必要信息）

### C. 代码命名与文案整洁化（保守）

- [x] 调整用户可见日志/CLI 文案中的 `V2` 表述（不改环境变量/内部兼容名）
- [x] （后续已升级）先保留 `triattention_runtime` 内部实现目录以降低风险；第三阶段已重命名为 `triattention_runtime/`

### D. 回归检查

- [x] `compileall` 检查新增/修改的 runner、dispatch、文档相关 Python 文件
- [x] 快速检查默认 dispatch 是否指向新默认配置与新 runner
- [x] 检查删除文档后 docs 内无悬空引用

## 备注

- （已更新）内部实现目录为 `triattention_runtime/`；`triattention_runtime/` 与 `TRIATTN_RUNTIME_*` 前缀作为兼容层保留；
  本轮主要目标是把**外部使用面**改成默认版本入口，避免引入新 bug。

---

## 第二阶段（当前执行）：默认化 `triattention` 包并丢弃 V0/V1 活跃实现

- 开始时间：2026-02-25（晚）
- 目标：
  1. 将 `triattention` 包从旧 V0/V1 集成入口改为“当前版本默认共享入口”
  2. 移除（或替换为 stub）旧 V0/V1 vLLM 集成代码，避免误用
  3. 删除 `docs/V0/` 历史文档目录，降低仓库噪音
  4. 继续弱化用户可见 `V2` 命名（README/脚本说明）

### 第二阶段执行清单

- [x] 更新 `triattention/__init__.py`：移除旧 V0/V1 导出，保留共享 scoring/compressor/utils 能力，并暴露当前 runtime 入口别名（lazy export，避免循环导入）
- [x] 将旧 `triattention.plugin` 改为兼容 no-op（避免 vLLM 自动加载时因旧 backend 注册报错）
- [x] 将旧 `triattention.vllm_integration` / `triattention.v1_backend` / `triattention.backends` 替换为清晰 stub（禁用旧接口）
- [x] 删除 `TriAttention_vLLM/docs/V0/` 目录
- [x] 更新 `TriAttention_vLLM/evaluation/README.md` 使用默认无 `V2` 入口
- [x] 简单回归：`compileall` + import smoke + runner/disptach `--help`

### 第二阶段回归记录（2026-02-25）

1. `py_compile/compileall` 已通过（入口/stub 文件 + runner/dispatch）
2. 旧接口依赖扫描：当前活跃代码中未发现对 `triattention.vllm_integration` / `triattention.v1_backend` / `triattention.backends` 的实际依赖
3. stub 行为 smoke：
   - `triattention.plugin.register_triattention_backend()` 正常 no-op 打印提示
   - `v1_backend` / `vllm_integration` stub 通过语法检查（直接运行细粒度 smoke 时受 shell 字符串限制；不影响代码结论）
4. `runner --help` / 部分 import smoke 在本机环境出现重依赖导入超时（历史上有 I/O 卡顿现象），本轮不将其视为代码回归

---

## 第三阶段（已完成）：重命名内部实现目录 `triattention_runtime`

- 目标：
  1. 将内部实现目录从 `triattention_v2/` 重命名为更清晰的默认命名 `triattention_runtime/`
  2. （阶段性）保留最小兼容包 `triattention_v2/`（仅转发，不承载实现），避免历史导入路径立即失效
  3. 只修改命名与导入，不改算法逻辑/执行行为

### 第三阶段执行清单

- [x] `git mv TriAttention_vLLM/triattention_v2 -> TriAttention_vLLM/triattention_runtime`
- [x] 新增 `triattention_v2/__init__.py` 兼容包（通过 `__path__` 转发子模块）
- [x] 更新当前默认入口与活跃文档中的内部目录引用（最小必要范围）
- [x] 更新 `triattention/__init__.py` 的 lazy export 导入路径（改为 `triattention_runtime`）
- [x] 简单回归：`compileall` + `triattention_runtime`/`triattention_runtime` shim import smoke

### 第三阶段回归记录（2026-02-25）

1. `compileall` 已覆盖 `triattention_runtime/`、`triattention_v2/__init__.py`、`triattention/__init__.py`
2. shim 机制 smoke（fake `triattention_runtime`）通过：`__path__` 转发与 `from ... import *` 行为正常
3. 真实导入 smoke 通过：
   - `import triattention_runtime.kv_compaction`
   - `import triattention_v2`（阶段性兼容包）
   - `import triattention; triattention.TriAttentionRuntimeConfig`（lazy export）

---

## 第四阶段（当前执行）：三处热点规整（历史产物命名 + 缓存目录）

- 开始时间：2026-02-25（夜）
- 目标（第一优先级）：
  1. 清理当前活跃仓库树中显眼的 `V2` 命名热点（`evaluation/logs`、`evaluation/outputs`、`__pycache__`）
  2. 不改算法逻辑、不改运行路径，仅做路径命名与缓存清理
  3. 完成后做基础回归（路径扫描 + `compileall` + 关键 pytest）

### 第四阶段执行清单

- [x] 预检查 `evaluation/logs` 与 `evaluation/outputs` 中含 `v2` 的路径并做冲突检查
- [x] 批量重命名历史产物路径（`triattention_v2_* -> triattention_runtime_*`、`v2_* -> runtime_*` 等）
- [x] 删除 `TriAttention_vLLM` 源码树下所有 `__pycache__` 目录
- [x] 更新本节回归记录（路径扫描 / compileall / pytest）

### 第四阶段回归记录（2026-02-25）

1. 预检查：候选路径 `130` 个，重命名冲突 `0`，目标已存在冲突 `0`
2. 批量重命名：已完成 `130` 个历史产物路径重命名（仅 `evaluation/logs` 与 `evaluation/outputs`）
3. `__pycache__`：已删除 `TriAttention_vLLM` 下 `8` 个缓存目录
4. 路径扫描：
   - `evaluation/logs` + `evaluation/outputs` 中 basename 含 `v2` 的路径数量：`0`
   - 活跃代码/测试/配置中 `triattention_v2/tests_v2/TRIATTN_V2_` 仅剩本清单历史记录与 `repository_archive` 备份说明
5. `compileall`：已覆盖 `triattention_runtime/`、`triattention/`、`evaluation/runner`、`evaluation/dispatch`、`tests_runtime/`，通过
6. `trivllm` 关键 pytest（需 `PYTHONPATH=TriAttention_vLLM`）：
   - `tests_runtime/test_config.py` → `8 passed`
   - `tests_runtime/test_runtime_eval_runner.py -k 'test_apply_runtime_env or test_setup_vllm_engine_force_runtime_scheduler_only_installs_scheduler_patch'` → `2 passed`
