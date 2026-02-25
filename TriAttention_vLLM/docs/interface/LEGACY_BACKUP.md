# 旧版备份说明

- 更新时间：2026-02-25
- 状态：Archived（仅用于追溯）

---

## 1. 备份文件

旧版资料已打包为单文件：

`TriAttention_vLLM/legacy_backup/v1_legacy_source_2026-02-13.tar.gz`

重建说明见：

`TriAttention_vLLM/legacy_backup/README.md`

---

## 2. 备份内容范围

1. `triattention/`（旧版实现）
2. `test/`（旧版测试）
3. `benchmarks/reasoning/`（旧版评测脚本）
4. `docs/V0/`（已从主工作树清理；仍包含在备份包中）
5. `docs/backend/reference/`
6. `docs/archive/`

---

## 3. 使用规则

1. 该压缩包仅用于追溯与回归参考。
2. 当前默认入口使用无 `V2` 命名的 runner/config；内部兼容实现目录仍为 `triattention_v2/`。
3. 禁止将新功能继续写入旧版目录。
