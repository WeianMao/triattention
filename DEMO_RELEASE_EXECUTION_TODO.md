# Demo Release Execution TODO

Updated: 2026-03-06  
Owner: Codex + Weian  
Status: Active

## 1) Goal

在 `master` 主线（TriAttention runtime 路径）上把 demo 发布链路稳定下来，确保可交接、可复现：

1. Web UI 可用（`demo/vllm`）。
2. OpenClaw -> gateway -> vLLM 路径可跑通（至少流程级验证）。
3. 输出正确、不 OOM，压缩触发行为可观测。
4. 给同事一份明确的运行/排障文档，避免上下文丢失。

## 2) Scope / Non-Goals

### Scope

1. 只基于 `master` 当前 TriAttention runtime 主线推进。
2. 可同步 `linxi-dev` 上与 demo 直接相关、且不破坏主线的文件（优先 docs/scripts/demo assets）。
3. 允许做参数层面的发布稳定性优化（预算、日志、启动参数）。

### Non-Goals (当前阶段)

1. 不回退到 retired 的 V0/V1 TriAttention 接线逻辑。
2. 不做大规模性能重构；先保障 release 可用性。
3. 不在本阶段做重型 benchmark。

## 3) Constraints to Keep

1. 不影响已有主线 TriAttention 算法行为（除非明确授权）。
2. 所有同步改动必须最小化、可回退。
3. 不提交大文件/实验产物/模型权重。
4. 交接文档优先，确保中断后可续接。

## 4) Workstreams

1. A: Branch Sync Hygiene  
把 `linxi-dev` 的 demo 必需资产同步到 `master`，避免旧路径冲突。

2. B: E2E Bring-up  
验证 Web UI 与 OpenClaw 路径的启动、请求、输出、日志。

3. C: Stutter Diagnosis (Streaming pause)  
定位“压缩触发时流式顿挫”是否来自压缩耗时、调度同步、或前端渲染。

4. D: Release Handoff  
产出稳定运行指南 + 已知问题 + 推荐参数。

## 5) Active TODO Checklist

1. [x] A1 - 同步 `demo/vllm` 主目录与 `demo/prompt.txt` 到 `master`。
2. [x] A2 - 产出交接状态文档 `LINXI_DEV_SYNC_STATUS.md` 并推送。
3. [x] A3 - 同步 `linxi-dev` 最新 demo 启动文档并核对路径。
4. [x] A4 - 同步 `demo` 启动依赖脚本 `TriAttention_vLLM/linxi_dev/run_vllm_serve.sh`。
5. [ ] B1 - 本机按文档重跑 demo 启动链路（Web UI）。
6. [ ] B2 - 复现一次 OpenClaw -> gateway -> vLLM 链路（流程级）。
7. [ ] C1 - 做最小复现：baseline vs triattention 的 streaming 卡顿对比。
8. [ ] C2 - 给出可发布参数建议（先稳后快）。
9. [ ] D1 - 产出面向同事的最终 handoff 文档（运行步骤 + 故障排查）。

## 6) Known Risks / Open Questions

1. 同事提到 `DEMO_SETUP.md`，远端实际文件名为 `demo/DEMO_STARTUP.md`。
2. `linxi-dev` 仍存在 `demo/vllm/vllm/` 双副本结构，需避免继续放大分叉。
3. `budget=2048` 在部分问题上可能出现后段重复，不作为当前阻断项但需标注。
4. `run_vllm_serve.sh` 走的是 `vllm serve + plugin` 路径；需与当前 master 的 TriAttention runtime 主线做一次行为确认，避免“可启动但未生效”。

## 7) Progress Log

1. 2026-03-06:
   - 已确认并同步第一批 demo 资产到 `master`（见 `LINXI_DEV_SYNC_STATUS.md`）。
   - 已 push: `5964c08a`。
   - 继续拉取 `linxi-dev` 后发现新提交：
     - `f80bae11 feat: add demo startup and qwen3-32b stats workflow docs`
   - 新提交包含：
     - `demo/DEMO_STARTUP.md`
     - 其他文档/脚本（待筛选同步）。
2. 2026-03-06:
   - 已将 `f80bae11` cherry-pick 到 `master`。
   - 已同步 `demo` 启动依赖脚本：
     - `TriAttention_vLLM/linxi_dev/run_vllm_serve.sh`
     - `TriAttention_vLLM/linxi_dev/run_vllm_triattention.sh`
   - 下一步进入 B 组：按文档进行本机链路验证与卡顿复现。
3. 2026-03-06:
   - 已 push 到 `origin/master`：
     - `ea6f2fe7` (sync `linxi-dev` docs commit `f80bae11`)
     - `7a12f90c` (新增执行看板 + 同步 `linxi_dev` 启动包装脚本)
