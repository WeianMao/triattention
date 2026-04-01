# 阶段 10：端到端测试

## 决策状态：已确认

## 目标

在代码清理、敏感信息扫描全部完成后，启动一个 agent **从零开始**模拟外部用户的完整使用流程，验证公布的代码能正常工作。

## 前置条件

- 阶段 2-9 全部完成（代码清理、重组、文档编写、敏感信息扫描）
- Clean-room 目录已准备好（但还未 push 到 GitHub）

## 测试流程

Agent 在一个干净环境中执行以下步骤，**每一步都必须成功**：

### 第 1 步：创建 conda 环境

```bash
conda create -n triattention_test python=3.10 -y
conda activate triattention_test
```

从零创建环境，不依赖任何已有环境。

### 第 2 步：安装依赖

```bash
cd /path/to/triattention-public
pip install -r requirements.txt
```

验证 requirements.txt 完整，所有依赖都能正常安装。

### 第 3 步：数据集自动下载

运行代码，验证首次运行时自动从 HuggingFace 下载数据集：

```bash
python scripts/run_eval.py --dataset aime24 --help  # 或实际的入口命令
```

验证点：
- 下载链接可用（HuggingFace 源没有失效）
- 下载后自动转换为正确的 JSONL 格式
- 缓存到 `./data/` 目录
- 字段名兼容性正常工作（`problem` → `question` fallback）

### 第 4 步：运行推理

用最小配置运行一次推理，验证代码能正常执行：

```bash
# 用小数据集 + 小 budget 快速验证
python scripts/run_eval.py --dataset aime24 --budget 512 --num-samples 2
```

验证点：
- 模型加载正常（从 HuggingFace hub 自动下载）
- monkeypatch 正常工作
- KV cache 压缩正常执行
- 输出结果格式正确

### 第 5 步：运行评估

```bash
python evaluation/eval_math.py --results-dir ./outputs/
```

验证点：
- 评估脚本能正常读取输出
- 数学等价性判断正常
- 最终分数输出格式正确

### 第 6 步：验证 baseline

运行至少一个 baseline 方法（如 FullKV），确保 baseline 也能正常工作。

## 验证标准

- [ ] conda 环境创建成功
- [ ] 所有依赖安装成功，无缺失包
- [ ] 数据集自动下载成功（3 个数据集都测试）
- [ ] 推理运行成功，无报错
- [ ] 评估运行成功，输出正确格式的结果
- [ ] 至少 1 个 baseline 方法可运行

## 失败处理

如果任何步骤失败：
1. 记录失败原因
2. 回退到对应的代码清理阶段修复
3. 修复后重新运行完整端到端测试

## 注意事项

- 测试需要 GPU（推理步骤），确保环境有可用 GPU
- 模型下载可能需要较长时间，第一次运行预留足够时间
- 如果 HuggingFace 源不可用，需要添加备用下载源
