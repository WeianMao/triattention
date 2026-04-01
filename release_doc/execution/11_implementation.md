# 实施方案：Worktree + Clean-room 两阶段

## 决策状态：已确认

## 阶段 0：创建 worktree

```bash
git checkout main
git branch release/public
git worktree add ../dc1-release release/public
```

结果：
- `dc1/` -- 主开发目录，不动，正常开发
- `dc1-release/` -- release 分支，独立目录，在这里做所有清理工作

两个目录共享同一个 git，可以同时打开、同时跑测试对比。
`dc1/` 里的 gitignore 文件（如校准文件）完全不受影响。

## 阶段 1：在 dc1-release/ 中整理

- 删除不需要的文件和目录（详见 [../scope/03_scope_exclude.md](../scope/03_scope_exclude.md)）
- 重命名（speckv -> triattention, aime -> benchmark 等）（详见 [../code_cleanup/04_naming.md](../code_cleanup/04_naming.md)）
- 去除敏感信息（内部路径、进程伪装、校准脚本等）（详见 [../code_cleanup/06_path_cleanup.md](../code_cleanup/06_path_cleanup.md)）
- 去除实验性 flag 和对应代码
- 重新组织文件夹结构（详见 [../code_cleanup/05_repo_structure.md](../code_cleanup/05_repo_structure.md)）
- 添加 LICENSE, README, .gitignore
- 如需跑对比测试，从 dc1/ 手动复制或 symlink gitignore 的资源文件

## 阶段 2：对比验证

- 在 dc1/ 和 dc1-release/ 同时跑测试，验证代码行为一致
- 全局扫描敏感信息（aime, weian, /data/rbg, PD-L1 等）— 完整扫描清单见 [../scope/03_scope_exclude.md](../scope/03_scope_exclude.md)
- 单元测试：RKV 和 TriAttention 峰值 KV cache 一致（详见 [10_technical_notes.md](10_technical_notes.md)）

## 阶段 3：Clean-room 发布

```bash
mkdir ~/triattention-public
cp -r ../dc1-release/需要的文件 ~/triattention-public/
cd ~/triattention-public
git init && git add . && git commit -m "Initial release"
# push 到 GitHub public repo
```

最终 public repo 干净无历史。

## 阶段 4：清理

```bash
git worktree remove ../dc1-release  # 删除 worktree，不影响分支
```
