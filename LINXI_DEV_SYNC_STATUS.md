# Linxi-Dev Sync Status (2026-03-06)

## 目标
把同事在 `origin/linxi-dev` 上“当前可用、且不会和主线 TriAttention runtime 冲突”的 demo 资料同步到 `master`，并给后续协作一个清晰状态说明。

## 已同步到 master 的内容

来源分支/提交：

- `origin/linxi-dev`
- `ef488169` (`fix: add prompts`)
- `c2032cd5` (`feat: demo for vllm and openclaw`) 中的 demo 主目录文件（按需摘取）

来自 `origin/linxi-dev` 的可用 demo 文件（已同步）：

- `demo/prompt.txt`
- `demo/vllm/__init__.py`
- `demo/vllm/config.py`
- `demo/vllm/server.py`
- `demo/vllm/vllm_client.py`
- `demo/vllm/start_remote_demo.sh`
- `demo/vllm/stop_remote_demo.sh`
- `demo/vllm/static/app.js`
- `demo/vllm/static/index.html`
- `demo/vllm/static/styles.css`

## 为什么不是直接 merge 整个 linxi-dev

`linxi-dev` 与当前 `master` 在 TriAttention 实现路径上分叉较大，直接 merge 会引入大量冲突，尤其集中在：

- 旧的 `triattention.v1_backend` / `triattention.vllm_integration` 路径
- 这两条路径在当前主线已被 retired，不应回退

因此本次采用“只同步 demo 可用文件”的方式，避免把旧集成逻辑带回主线。

## 当前可确认事实（方便交接）

1. 同事新增的 prompt 文件路径是 `demo/prompt.txt`（不是 `demo/vllm/prompts.txt`）。
2. `linxi-dev` 里还有一份重复目录 `demo/vllm/vllm/`；本次未同步该重复副本，只保留主目录版本。
3. 本次同步不改 TriAttention 主线 runtime 行为，只补 demo 资产和提示词文件。

## 后续建议（给同事）

1. 如果要继续推进 demo，请统一只维护 `demo/vllm/` 一套目录，避免 `demo/vllm/vllm/` 双副本分叉。
2. 任何 TriAttention 接线都应基于当前 runtime 主线，不要再依赖 retired 的 V0/V1 集成入口。
3. `demo/prompt.txt` 的两条 prompt 先可用，后续可继续按演示需求微调。
