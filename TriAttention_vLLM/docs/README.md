# TriAttention_vLLM 文档入口（当前默认版本）

> 目标：让后续接手同事快速知道“当前默认入口是什么、代码在哪里、HF 对齐状态如何”。

详细文档维护规范见：`TriAttention_vLLM/docs/DOCS_STANDARDS.md`

## 先读（推荐顺序）

1. `TriAttention_vLLM/docs/interface/PROJECT_GOAL.md`
2. `TriAttention_vLLM/docs/interface/IMPLEMENTATION_OVERVIEW.md`
3. `TriAttention_vLLM/docs/interface/HF_ALIGNMENT_STATUS.md`
4. `TriAttention_vLLM/docs/interface/CURRENT_STATUS.md`
5. `TriAttention_vLLM/docs/interface/GUIDED_TOUR.md`

## 当前默认入口（对外）

1. Dispatch：
   - `TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py`
   - 默认配置：`TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24.yaml`
2. Runner：
   - `TriAttention_vLLM/evaluation/runner/vllm_triattention_runner.py`

说明：
- 当前实现内部目录名为 `triattention_runtime/`。
- `triattention_v2/` 仅作为兼容导入包保留（薄转发层）。
- `vllm_triattention_v2_runner.py`、`triattention_v2_*` 配置文件仍可用，但不再是默认入口。

## 文档取舍（本轮整理后）

1. 历史 debug / 审计流水文档已删除或降级，不再作为接手入口。
2. 当前文档优先引用代码位置，而不是重复维护长篇实现细节。
3. 设计历史仍保留在 `docs/backend/` 和 `docs/archive/`，仅在需要追溯时查看。
