# 实验规范

## 默认配置

### 默认 Head 索引

```
weian_development/spec_sparse_simulator/hybrid_sample_heads_lowret_top10.json
```

### 参考实现（禁止修改）

```
weian_development/spec_sparse_simulator/attention_pruning_case_study_hybrid_rounds_xtrace.py
```

文档未规定的细节，默认遵守此脚本。

### 处理原则

| 情况 | 处理方式 |
|------|----------|
| 文档明确规定 | 按文档执行 |
| 文档未规定 | 遵守参考实现 |
| 用户要求修改 | 按用户要求 |
| **不确定** | **必须询问用户** |

---

## 目录结构

```
experiments/
├── exp_001_sanity_check/
│   ├── README.md          # 必须
│   ├── run.py
│   ├── config.yaml        # 可选
│   └── output/            # 必须
│       ├── logs/
│       ├── checkpoints/
│       ├── results/
│       └── figures/
├── exp_002_xxx/
└── ...
```

### 命名规范

```
exp_{序号}_{描述}/
```
- 序号：3 位数字（001, 002, ...）
- 描述：小写字母 + 下划线

---

## README.md 模板

```markdown
# 实验名称

## 目标
简述实验目的

## 方法
简述实验方法

## 运行方式
\`\`\`bash
python run.py --config config.yaml
\`\`\`

## 结果摘要
- 指标 1: xxx
- 指标 2: xxx

## 结论
实验结论
```

---

## Git 规范

### 必须 Commit
- `*.py` 脚本
- `*.yaml`, `*.json` 配置（小型）
- `README.md`

### 禁止 Commit
- `output/` 目录
- 大型数据文件（>.1MB）
- `__pycache__/`

### .gitignore

```gitignore
experiments/*/output/
*.pt
*.pth
*.pkl
__pycache__/
```

---

## Checklist

新实验前确认：

- [ ] 创建 `exp_{序号}_{描述}/` 文件夹
- [ ] 创建 `output/` 子目录
- [ ] 创建 `README.md`
- [ ] 确认 `output/` 不会被 commit
