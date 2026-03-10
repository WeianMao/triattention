# TriAttention 热插拔 & OpenClaw 对比测试进度

**日期**: 2026-03-09（第二轮 session）
**环境**: zju_230 (RTX 4090 24GB), vLLM V1, Qwen3-32B-INT4
**接续**: HANDOFF_2026-03-09.md（第一轮 session 修复交接）

---

## 一、完成的工作

### 1.1 热插拔脚本 `swap_backend.sh`

**文件**: `TriAttention_vLLM/linxi_dev/swap_backend.sh`

实现 TriAttention / Baseline vLLM 后端的一键切换：

```bash
./linxi_dev/swap_backend.sh status        # 查看当前后端
./linxi_dev/swap_backend.sh baseline      # 切到 vanilla vLLM
./linxi_dev/swap_backend.sh triattention  # 切到 TriAttention
```

功能：
- 自动 kill 旧进程、等待健康检查（180s 超时）
- TriAttention 模式自动设置所有 `TRIATTN_RUNTIME_*` 环境变量
- Baseline 模式清除所有 TriAttention 相关环境变量 + `ENABLE_TRIATTENTION=false`
- 两种模式统一 `--enforce-eager`（4090 显存限制，torch.compile 额外占 ~1GB）
- `VLLM_RELAXED_KV_CHECK=1` 绕过 KV cache 启动时内存检查

### 1.2 Baseline 模式插件隔离

**文件**: `TriAttention_vLLM/triattention/plugin.py`

**问题**: TriAttention 作为 vllm.general_plugins entry_point 注册，即使 `--no-triattention` 也会自动加载，导致 baseline 模式崩溃（`TRIATTN_FATAL_TRITON_SCORING_REQUIRED:stats_path_not_set`）。

**修复**: 在 `register_triattention_backend()` 开头检查 `ENABLE_TRIATTENTION` 环境变量，为 `false` 时直接 return。

### 1.3 KV Cache 内存检查放松

**问题**: vLLM 启动时检查 `max_model_len` 是否能完全放入 KV cache 物理内存。4090 上 KV cache 约 4.14 GiB，只够 ~16K tokens。但 TriAttention 压缩后实际占用远低于 token 数暗示的量。

**修复（两层）**:
1. **TriAttention monkeypatch**（`integration_monkeypatch.py`）: patch `vllm.v1.core.kv_cache_utils._check_enough_kv_cache_memory`，将 ValueError 改为 warning
2. **vLLM 源码 patch**（远端直接修改）: 在 `_check_enough_kv_cache_memory` 中增加 `VLLM_RELAXED_KV_CHECK` 环境变量检查，设置时跳过 raise

### 1.4 Demo Server completions 代理

**文件**: `demo/vllm/server.py`

**问题**: OpenClaw 使用 `openai-completions` API（`/v1/completions`），但 demo server 只代理了 `/v1/chat/completions`，导致 openclaw 请求 404 → 卡死。

**修复**: 新增 `/v1/completions` 透传端点，支持 streaming 和非 streaming。

### 1.5 OpenClaw 配置调整

**文件**: `~/.openclaw/openclaw.json`（custom2 provider）

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| contextWindow | 16384 | 32768 | 匹配 TriAttention 32K max_model_len |
| maxTokens | 2048 | 4096 | 允许更长的生成输出 |

### 1.6 测试文档集

**位置**: `~/tmp/claw-doc/`（6 份文档，共 30KB ~700 行）

围绕虚构项目 DocMind 生成，包含会议纪要、选型报告、技术方案、评审意见、周报、Bug 报告。文档间有交叉引用，用于测试 agent 的 long prefilling 和 long decoding 能力。

## 二、对比测试结果

### 2.1 小文档集（5KB，6 份简短文档）

| | Baseline 16K | TriAttention 16K |
|---|---|---|
| 结果 | 成功 | 成功 |
| 输出质量 | 完整总结 | 完整总结 |

### 2.2 大文档集（30KB，6 份扩充文档）

| | Baseline 16K | Baseline 32K | TriAttention 32K |
|---|---|---|---|
| 结果 | Context overflow（openclaw 客户端拒绝） | 成功 | 成功 |
| 输出质量 | N/A | 完整总结 | 完整总结，5 个风险点 + 建议 |

### 2.3 Baseline 32K + gpu_memory_utilization=0.90

**状态**: 未完成，openclaw 卡在 demo proxy 问题（`/v1/completions` 404），修复后尚未重新测试。

## 三、未完成 / 待解决

### 3.1 OpenClaw 卡死问题

OpenClaw agent 在某些情况下启动后无输出、无超时提示：
- 根因已定位：demo server 缺少 `/v1/completions` 端点 → 404 → openclaw 无限等待
- 已修复 demo server，但因 session 进度问题未来得及重新测试

### 3.2 swap_backend.sh 模式检测 bug

`_current_mode()` 通过 `/proc/PID/environ` 检查 `ENABLE_TRIATTENTION=true`，但 `run_vllm_serve.sh` 的 triattention 分支没有 `export ENABLE_TRIATTENTION`（只用 shell 变量），导致检测失败。切换时需要先手动 kill 再启动。

### 3.3 降低 gpu_memory_utilization 的对比测试

将 `gpu_memory_utilization` 从 0.96 降至 0.90，制造更紧张的 KV cache 压力，验证 TriAttention 压缩在极限条件下的优势。**尚未完成**。

### 3.4 诊断日志清理

上一轮 session 遗留的 `[DIAG]` 日志仍在以下文件中：
- `scheduler.py`
- `integration_monkeypatch.py`
- `runner_output_bridge.py`

## 四、Git 提交记录（本轮 session）

| Commit | 描述 |
|--------|------|
| `8b67a3a2` | feat: add hot-swap script swap_backend.sh |
| `b465512d` | fix: add enforce-eager and disable multiprocessing in swap |
| `db285924` | fix: skip TriAttention plugin when ENABLE_TRIATTENTION=false |
| `10b3e2cf` | fix: move enforce-eager to common path for fair comparison |
| `2d112c7a` | feat: use 32K max_model_len for triattention mode |
| `89532819` | feat: relax KV cache memory check when TriAttention active |
| `09e5a766` | feat: 32K for baseline + VLLM_RELAXED_KV_CHECK env var |
| `937c6f13` | chore: lower gpu_memory_utilization to 0.90 |
| `555c21b4` | feat: add /v1/completions proxy to demo server |

## 五、下一步建议

1. **重跑对比测试**: baseline 0.90 util vs triattention 0.90 util，30KB 文档集，验证 KV 压缩在显存紧张条件下的优势
2. **修复 `_current_mode` 检测**: 在 `run_vllm_serve.sh` triattention 分支显式 `export ENABLE_TRIATTENTION=true`
3. **清理诊断日志**: 确认 runtime 稳定后移除 `[DIAG]` 日志
4. **考虑让 openclaw 直连 vLLM**: 绕过 demo proxy，减少故障点（新增 SSH 端口转发 local:8002 → remote:8002）
