# 分布式启动器

## 决策状态：已确认，全部公布

## 核心文件

| 文件 | 大小 | 作用 |
|------|------|------|
| `rkv_sharded_dispatch.py` | 31KB | 主调度器：多GPU分配、断点恢复、自动评估 |
| `rkv_sharded_eval.py` | 33KB | 推理 worker：每GPU一个实例 |
| `rkv_sharded_runner.py` | 689B | 轻量 wrapper（需去掉 PD-L1_binder） |
| `merge_rkv_shards.py` | 2.8KB | 分片结果合并 |
| `process_utils.py` | 1.1KB | 进程命名（需去掉 PD-L1_binder） |
| `rkv_cache_utils.py` | 882B | cache 管理 |

## 功能

- **多GPU分配**：自动检测可用GPU，队列调度
- **断点恢复**：检测已完成的 shard，跳过重复计算
- **分片合并**：按 sample_idx + draw_idx 排序合并
- **自动评估**：合并后自动调用 eval_math_multi.py
- **错误处理**：fail-fast，任一 shard 失败终止全部

## 完整流程

```
用户 shell 脚本 → rkv_sharded_dispatch.py
  → 分配任务到多个 GPU
  → 每个 GPU 运行 rkv_sharded_eval.py（推理）
  → merge_rkv_shards.py（合并分片）
  → eval_math_multi.py（评估）
```

## 清理要求

### 进程伪装代码删除

以下文件中包含 PD-L1_binder 相关代码，release 时必须删除：
- `rkv_sharded_runner.py` — 去掉 PD-L1_binder wrapper
- `process_utils.py` — 去掉 mask_process_command 等进程伪装功能

### 命名清理

release 时启动器文件名中的 `rkv_sharded` 等内部命名需要替换为正式名称。具体方案待确认（见 [../tracking/14_open_items.md](../tracking/14_open_items.md)）。

### 路径清理

- `rkv_sharded_eval.py` 中 PYTHONPATH 添加了内部路径，需改为相对路径
- `weian_development` 路径引用需要重构
- 详见 [../code_cleanup/06_path_cleanup.md](../code_cleanup/06_path_cleanup.md)
