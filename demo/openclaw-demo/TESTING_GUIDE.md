# OpenClaw Demo 测试指南

TriAttention A/B 对比测试。通过 OpenClaw agent 读取 `sample-docs/` 中的全部文档（~30KB）并生成周报，产生足够长的上下文来触发 KV 压缩。

## 1. 启动后端服务

### 1.1 启动 vLLM（远端 GPU 机器）

SSH 到远端 GPU 机器，使用 `swap_backend.sh` 启动。

**TriAttention 模式**：

```bash
cd TriAttention_vLLM

# 默认 budget=2048, protect_prefill=false, port=8002, GPU=1
KV_BUDGET=2048 bash linxi_dev/swap_backend.sh triattention
```

**Baseline 模式**：

```bash
bash linxi_dev/swap_backend.sh baseline
```

**查看状态**：

```bash
bash linxi_dev/swap_backend.sh status
```

可通过环境变量覆盖默认参数：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_PORT` | `8002` | vLLM 监听端口 |
| `KV_BUDGET` | `2048` | TriAttention KV 预算 |
| `CUDA_VISIBLE_DEVICES` | `1` | GPU 编号 |
| `GPU_MEMORY_UTILIZATION` | `0.95` | GPU 显存利用率 |
| `MAX_MODEL_LEN` | `32768` | 最大序列长度 |
| `TRIATTN_RUNTIME_PROTECT_PREFILL` | `false` | 是否保护 prefill token 不被压缩 |

### 1.2 启动 Demo Gateway（远端）

```bash
export PATH="$HOME/.pixi/bin:$PATH"

VLLM_BACKEND_URL=http://127.0.0.1:8002 \
DEMO_HOST=0.0.0.0 \
DEMO_PORT=8010 \
nohup pixi run python -m uvicorn demo.vllm.server:app \
  --host 0.0.0.0 --port 8010 \
  > /tmp/demo_gateway.log 2>&1 &

# 验证
curl -s http://127.0.0.1:8010/healthz | python3 -m json.tool
```

### 1.3 建立 SSH 隧道（本地）

```bash
# 保持终端打开
ssh -L 8010:127.0.0.1:8010 <remote-host>
```

验证：

```bash
curl http://127.0.0.1:8010/healthz
curl http://127.0.0.1:8010/v1/models
```

## 2. 配置 OpenClaw

### 2.1 修改 `~/.openclaw/openclaw.json`

在 `models.providers` 中添加一个 provider 指向本地隧道：

```jsonc
{
  "models": {
    "mode": "merge",
    "providers": {
      // ... 已有 providers ...
      "triattention-demo": {
        "baseUrl": "http://127.0.0.1:8010/v1",
        "apiKey": "local",
        "auth": "api-key",
        "api": "openai-completions",
        "models": [
          {
            "id": "<model-path-to-Qwen3-32B-INT4>",
            "name": "Qwen3-32B-INT4 (TriAttention Demo)",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 32768,
            "maxTokens": 4096
          }
        ]
      }
    }
  }
}
```

> **注意**：`id` 必须与远端 vLLM 加载的模型路径完全一致，否则会 404。可通过 `curl http://127.0.0.1:8010/v1/models` 查看实际 model id。

### 2.2 创建测试 Agent

在 `agents.list` 中添加：

```jsonc
{
  "agents": {
    "list": [
      // ... 已有 agents ...
      {
        "id": "docmind",
        "name": "docmind",
        "workspace": "<repo_root>/demo/openclaw-demo/sample-docs",
        "model": "triattention-demo/<model-path-to-Qwen3-32B-INT4>"
      }
    ]
  }
}
```

将 `<repo_root>` 替换为本机 speckv 仓库的绝对路径。

### 2.3 验证配置

```bash
# 确认 agent 可见
openclaw agents list

# 快速验证连通性
openclaw agent --agent docmind -m "hello"
```

## 3. 清理 Session

每次测试前必须清理历史上下文，否则残留对话会干扰结果。

```bash
# 删除 docmind agent 的所有 session 数据
rm -rf ~/.openclaw/agents/docmind/sessions/*

# 验证清理成功
ls ~/.openclaw/agents/docmind/sessions/
# 应该为空
```

同时确认后端 KV 缓存已空闲：

```bash
curl -s http://127.0.0.1:8010/api/kv-cache | python3 -c "
import sys,json; d=json.load(sys.stdin)
print('KV Usage: %.1f%%' % d.get('usage_percent', 0))"
# 应接近 0%
```

## 4. 发送测试 Prompt

### 4.1 标准测试命令

```bash
openclaw agent --agent docmind -m "请你读取当前目录下的所有文件，然后根据这些文档的内容为 DocMind 项目撰写一份本周周报（W9）。

要求：
1. 先逐个读取所有 6 个文件（会议纪要_0115.md、技术方案_RAG管线.md、评审意见_0207.md、选型报告_向量数据库.md、Bug报告_解析乱码.md、周报_W8.md）
2. 综合所有文档信息，按以下结构撰写周报：
   - 一、本周进展（按后端/算法/前端/测试分组）
   - 二、风险跟踪（更新风险表，标注状态变化）
   - 三、关键决策记录
   - 四、下周计划
   - 五、需要协调事项
3. 对比上周周报（W8），标注哪些事项已完成、哪些延期
4. 周报长度 2000-3000 字"
```

### 4.2 预期行为

1. Agent 调用 `read` 工具逐个读取 6 个文件（~30KB 总文本进入上下文）
2. 全部文件读取完成后，Agent 开始生成周报
3. 生成过程中 KV cache 逐步增长
4. TriAttention 模式下 budget 较小时（如 512-2048），生成过程中会触发压缩

### 4.3 观测指标

测试期间在另一个终端监控，或直接在 http://127.0.0.1:8010 看实时的 streaming 输出：

```bash
# 实时监控 KV 缓存使用率（每 2 秒刷新）
watch -n 2 'curl -s http://127.0.0.1:8010/api/kv-cache | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(\"KV Usage: %.1f%%  Used: %s  Capacity: %s\" % (
    d.get(\"usage_percent\",0),
    int(d.get(\"used_tokens_estimate\",0)),
    int(d.get(\"capacity_tokens_estimate\",0)),
))"'
```

```bash
# 查看压缩日志（远端，TriAttention 模式）
ssh <remote-host> 'tail -f /tmp/vllm_triattention.log | grep -E "compression (applied|skipped)|reclaimed_blocks"'
```

**TriAttention 正常表现**：

- KV usage 在触发压缩后**下降**（而不是单调递增）
- 日志中可见 `compression applied before=XXXX after=YYYY`
- 生成吞吐稳定（~25-30 tok/s）

**Baseline 对照表现**：

- KV usage 单调递增
- 长上下文场景下可能 preempt（KV 满后请求被驱逐，吞吐降为 0）

## 5. A/B 对比测试流程

### 5.1 TriAttention 测试

```bash
# 1. 远端：启动 TriAttention 后端
KV_BUDGET=2048 bash linxi_dev/swap_backend.sh triattention

# 2. 本地：清理 session
rm -rf ~/.openclaw/agents/docmind/sessions/*

# 3. 本地：发送标准 Prompt（第 4.1 节）
openclaw agent --agent docmind -m "..."

# 4. 记录：输出质量、KV 曲线、压缩次数、总耗时
```

### 5.2 Baseline 测试

```bash
# 1. 远端：切换到 Baseline 后端
bash linxi_dev/swap_backend.sh baseline

# 2. 本地：清理 session
rm -rf ~/.openclaw/agents/docmind/sessions/*

# 3. 本地：发送同样的 Prompt
openclaw agent --agent docmind -m "..."

# 4. 记录：输出质量、KV 曲线、是否 preempt、总耗时
```

### 5.3 对比维度

| 指标 | TriAttention | Baseline |
|------|-------------|----------|
| 输出是否完成 | | |
| 输出质量（人工评分 1-5） | | |
| KV 峰值使用率 | | |
| 是否触发 preempt | | |
| 总耗时 | | |
| 生成吞吐 (tok/s) | | |

## 6. 常见问题

### 输出出现乱码/重复字符

**这是已知的最大问题，需要被解决：当触发压缩之后就会出现乱码/重复字符**

### OpenClaw 无法连接

```bash
# 确认隧道存在
lsof -i :8010

# 确认 gateway 健康
curl http://127.0.0.1:8010/healthz

# 确认后端健康
curl http://127.0.0.1:8010/v1/models
```

### Agent 不调用工具读文件

确认 `workspace` 路径指向 `sample-docs/` 目录。如果仍不调用，在 prompt 中明确列出完整文件路径：

```bash
openclaw agent --agent docmind -m "请依次读取以下文件并撰写周报：
- 会议纪要_0115.md
- 技术方案_RAG管线.md
- 评审意见_0207.md
- 选型报告_向量数据库.md
- Bug报告_解析乱码.md
- 周报_W8.md"
```

### 压缩未触发

- `protect_prefill=true`时，prefill 长度必须 < budget 才能压缩
- 30KB 文档 prefill 约 15K tokens，需要 `protect_prefill=false` 或 `budget > 15K`
- `swap_backend.sh` 已默认设置 `protect_prefill=false`

### GPU 显存不足导致启动失败

```bash
# 查看 GPU 占用
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# 找到并清理残留的 wayne 用户 vLLM 进程
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
kill <stale_pid>
```

## 7. Stats 文件说明

`stats/qwen3_32b_int4_speckv_stats.pt` 是 Qwen3-32B-INT4 的 TriAttention 注意力统计文件（6.8MB），由 R-KV 离线评估生成，用于指导运行时 KV 压缩的 token 保留策略。

在远端启动 vLLM 时需要通过 `STATS_PATH` 环境变量指定此文件路径：

```bash
# swap_backend.sh 默认使用以下路径（需确认远端已部署）
STATS_PATH=/path/to/stats/qwen3_32b_int4_speckv_stats.pt

# 如果 stats 文件不在默认位置，可手动指定
STATS_PATH=/path/to/stats.pt KV_BUDGET=2048 bash linxi_dev/swap_backend.sh triattention
```

此文件也包含在本仓库 `demo/openclaw-demo/stats/` 目录中，可按需复制到远端部署。

## 8. Sample Docs 说明

`sample-docs/` 包含 6 个虚构的 DocMind 项目文档，总计约 30KB：

| 文件 | 内容 | 大小 |
|------|------|------|
| 会议纪要_0115.md | 项目启动会，定义 M1-M4 里程碑 | 6.4K |
| 技术方案_RAG管线.md | RAG 管线技术设计 v2.0 | 7.1K |
| 评审意见_0207.md | CTO 技术方案评审反馈 | 4.2K |
| 选型报告_向量数据库.md | Qdrant/Milvus/Weaviate 选型 | 3.9K |
| Bug报告_解析乱码.md | PDF 解析中文乱码 P1 bug | 4.4K |
| 周报_W8.md | 上周周报，作为对比基准 | 4.9K |

这些文档构成一个连贯的项目叙事，Agent 需要综合理解后才能写出有质量的周报。
