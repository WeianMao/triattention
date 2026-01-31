# Phase 0 文档修订说明（给同事）

本文档说明我在 Phase 0 相关文档中做了哪些修改、为什么要改、原先错在哪。

---

## 修改 1：隔离开发口径统一（README）

**改动位置**
- `TriAttention_vLLM/docs/phases/phase0_rkv_integration/README.md`
  - 2.2 约束表（隔离开发）
  - 2.3 隔离开发原则（允许/禁止操作表）

**原问题**
- 文档写“不能修改 rkv/ 核心代码”，但当前实际口径是“允许改，但必须参数隔离，默认不影响其他算法”。  
- 口径冲突会导致后续评审/执行标准不一致。

**修正内容**
- 明确允许修改核心文件，但必须通过参数隔离，默认行为不变。  
- 允许“在核心文件中添加 SpeckV 分支（仅参数触发）”。  
- 禁止项改为“无参数隔离的核心修改 / 修改默认参数或默认路径”。

---

## 修改 2：压缩触发条件表述过简（README）

**改动位置**
- `TriAttention_vLLM/docs/phases/phase0_rkv_integration/README.md`
  - 1.2 架构图中“触发条件”一行

**原问题**
- 原文只写 `absolute_position % divide_length == 0`，容易误以为只看取模。  
- 实际逻辑还包括：必须是 decode step、有效缓存长度 >= 阈值、`use_slack_trigger` 分支差异。

**修正内容**
- 改为“decode step 且有效缓存 >= 阈值；若非 slack 还需取模”。

---

## 修改 3：reset_compression_state 描述错误（DESIGN_NOTES）

**改动位置**
- `TriAttention_vLLM/docs/phases/phase0_rkv_integration/DESIGN_NOTES.md`
  - 5.2 / 7.3

**原问题**
- 文档写“评估脚本中需要手动 reset，脚本应已有逻辑”。  
- 实际上 `rkv_sharded_eval.py` 并没有该调用；相反 `speckv_rkv_style.py` 会在空 cache 时自动 reset。

**修正内容**
- 说明默认路径下自动 reset（空 cache 时）。  
- 仅在“复用 past_key_values / 流式拼接多样本”时需要手动 reset。

---

## 修改 4：Stats 元数据校验过于绝对（DESIGN_NOTES）

**改动位置**
- `TriAttention_vLLM/docs/phases/phase0_rkv_integration/DESIGN_NOTES.md`
  - 6.4 / 7.1

**原问题**
- 文档写“初始化时强制完整校验”。  
- 实际上完整校验依赖调用方传入 `metadata_expectations`：  
  - `rkv_sharded_eval.py` 会传，因此在该入口成立  
  - 其他入口若未传，只会做最小默认校验

**修正内容**
- 明确“完整校验由入口保证”，并提示其他入口必须显式传 expectations。

---

如需我继续统一其他文档的口径，或者针对某个入口补充固定模板，请直接说。
