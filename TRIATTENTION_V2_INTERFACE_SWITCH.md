# TriAttention 接口切换说明（V1 -> V2）

更新时间：2026-03-08  
适用分支：`master`、`linxi-dev`

## 1. 问题是什么

这次 demo 里出现的 streaming 卡顿，根因不是“当前 V2 主线算法突然有 bug”，而是接口接线走成了旧 V1 路径：

1. 旧接线是 `vllm serve ... --attention-backend CUSTOM`（V1 custom backend 时代）。
2. 当前仓库里该路径已经退休（`triattention/plugin.py` 之前是 legacy no-op）。
3. 同时，历史 V1 路径本身在压缩触发时有 token 级 Python gather/scatter 热点，会引入秒级顿挫。

所以本质是“接口使用错代际 + 命中旧路径已知热点”。

## 2. 应该改成什么接口

统一改为 V2 runtime 接口：

1. 不再依赖 `--attention-backend CUSTOM`。
2. 通过插件加载后安装 runtime monkeypatch（scheduler/worker）。
3. 配置统一走 `TRIATTN_RUNTIME_*`（兼容桥接 `TRIATTENTION_*`）。

## 3. 这次已落地的改动

### 3.1 插件入口改为 V2 激活

文件：`TriAttention_vLLM/triattention/plugin.py`

1. 默认 `TRIATTENTION_INTERFACE=runtime` 时，自动安装 runtime monkeypatch：
   - `triattention_runtime.integration_monkeypatch.install_vllm_integration_monkeypatches`
2. 自动把 legacy 变量桥接到 runtime：
   - `TRIATTENTION_STATS_PATH -> TRIATTN_RUNTIME_SPARSE_STATS_PATH`
   - `TRIATTENTION_KV_BUDGET -> TRIATTN_RUNTIME_KV_BUDGET`
   - `TRIATTENTION_DIVIDE_LENGTH -> TRIATTN_RUNTIME_DIVIDE_LENGTH`
   - 其它常用项同理
3. 若显式设置 `TRIATTENTION_INTERFACE=legacy_custom`，仅打印退役提示，不再尝试注册 V1 backend。

### 3.2 启动脚本默认改为 V2

文件：`TriAttention_vLLM/linxi_dev/run_vllm_serve.sh`

1. 默认接口模式：`TRIATTENTION_INTERFACE=runtime`
2. runtime 模式下：
   - 保留 `VLLM_PLUGINS=triattention`
   - 注入 `TRIATTN_RUNTIME_*` 关键变量
   - 不再给 `vllm serve` 追加 `--attention-backend CUSTOM`
3. 仅在 `TRIATTENTION_INTERFACE=legacy_custom` 时，才走旧 `--attention-backend` 参数路径（兼容保底）。

### 3.3 Demo 文档同步

文件：`demo/DEMO_STARTUP.md`

1. 启动命令改为 runtime 模式（去掉 `--attention-backend CUSTOM`）。
2. 验证日志改为 V2 标志：
   - `Runtime (V2) plugin activated`
   - `Installed TriAttention runtime monkeypatch integration`

## 4. 如何使用（最小变更）

如果你是从旧命令迁移，按下面改：

1. 删除 `--attention-backend CUSTOM`。
2. 增加：
   - `TRIATTENTION_INTERFACE=runtime`
   - `VLLM_PLUGINS=triattention`
3. 确保 runtime 核心参数存在：
   - `TRIATTN_RUNTIME_SPARSE_STATS_PATH`
   - `TRIATTN_RUNTIME_KV_BUDGET`
   - `TRIATTN_RUNTIME_DIVIDE_LENGTH`
   - `TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION=true`

## 5. 快速验收

启动后检查 backend log（例如 `demo/vllm/logs/triattention_backend.log`）：

1. 有 `Runtime (V2) plugin activated`
2. 有 `Installed TriAttention runtime monkeypatch integration`
3. 运行到长 decode 后，能看到压缩相关日志（如 `TriAttention compression applied req=`）

满足以上 1/2/3，说明接口已经走到 V2，不再是旧 V1 CUSTOM backend 路径。
