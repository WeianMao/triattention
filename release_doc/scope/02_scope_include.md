# 第一阶段公布内容

## R-KV/ 中的内容

### 我们的方法（TriAttention）

所有 speckv 相关代码和脚本，统一使用 **TriAttention** 这个名字（不再叫 speckv）。

命名映射详见 [../code_cleanup/04_naming.md](../code_cleanup/04_naming.md)。

### 官方 baseline 方法

- RKV（R1-KV）
- SnapKV
- H2O
- StreamingLLM
- FullKV

### SparsePrefillKeep

这是我们自己方法的一个变种，不是 RKV 的 baseline。

### rkv/ 包（核心压缩实现库）

需要 release，在 clean-room 阶段随目录重组一起完成包名变更。

**已确认的双包策略**：
- baseline 通用方法 → `kv_compress/`（中性通用名）
- 我们的方法实现 → `triattention/`

决策理由：rkv 包不止包含我们的方法，还有所有 baseline（R1KV, SnapKV, H2O 等），用 `triattention` 命名不准确。baseline 部分用中性名 `kv_compress`，我们的方法单独用 `triattention`。

### 多个 Setting 都要公布

- 不同模型（Qwen, LLaMA 等）
- 不同数据集
- 不同配置变体（perhead 等）

所有 setting 以起点脚本为准，详见 [../components/09_reference_script.md](../components/09_reference_script.md)。

## 评估管线

13 个文件 + latex2sympy/，详见 [../components/07_evaluation.md](../components/07_evaluation.md)。

## 分布式启动器

全部公布，详见 [../components/08_launcher.md](../components/08_launcher.md)。

## TriAttention_vLLM/

- Main TriAttention implementation on vLLM
- Status: 有 bug 待修，还需要开发 SGLang 版本，**暂不 release**
- 待完成后再纳入 release
