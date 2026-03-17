# Legacy Backup

本目录用于保存 V1 旧版的单文件备份。

当前本地备份文件：

`v1_legacy_source_2026-02-13.tar.gz`

注意：

1. 仓库根 `.gitignore` 默认忽略 `*.tar.gz`，因此该文件是本地备份资产，不会被默认提交。
2. 如需重建备份，可在仓库根目录执行：

```bash
tar -czf TriAttention_vLLM/legacy_backup/v1_legacy_source_2026-02-13.tar.gz \
  -C TriAttention_vLLM \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='.pytest_cache' \
  triattention test benchmarks/reasoning docs/V0 docs/backend/reference docs/archive
```

