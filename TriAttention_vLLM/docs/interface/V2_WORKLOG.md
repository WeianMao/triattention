# TriAttention_vLLM V2 开发执行日志（精简版）

- 更新时间：2026-02-23
- 状态：Active
- 适用范围：vLLM 0.15.x（V1 Engine）

---

## 1. 用途与边界

本文件用于记录“当前仍影响开发判断/接手效率”的执行事实与阶段性里程碑。  
不再保留过长的逐步施工流水账；已完成且不再影响当前判断的细节通过以下方式追溯：

1. `git log` / 提交记录
2. `interface/CURRENT_STATUS.md`（当前态）
3. `interface/OPEN_ISSUES.md`（当前未解决问题）
4. `backend/DESIGN_DECISIONS.md`（已拍板边界）
5. `interface/V2_SCHEME_ADJUSTMENT_2026-02-23.md`（本轮方案调整与优先级）

---

## 2. 精简里程碑（已完成）

### 2.1 2026-02-16 ～ 2026-02-20（原型闭环与 P0 排查阶段）

已完成并对当前方案仍有影响的事实（精简保留）：

1. 建立了 V2 原型链路：scheduler 信号 -> runner 执行 -> hook 压缩 -> side-channel 回传。
2. 落地了 experimental block reclaim 原型闭环（实验开关保护）。
3. 明确了多个 P0/P1 问题并沉淀到 `OPEN_ISSUES.md`（包括 strict reclaim、runtime 语义冲突、吞吐异常等）。
4. 验证过一轮基础测试与 smoke 路径，形成后续重构前的可运行基线。
5. 明确了“继续在 patch-heavy 路径叠补丁会导致复杂度失控”的事实（为后续方案重置提供依据）。

说明：

1. 该阶段大量逐项修复细节已从本日志移除，避免当前日志过长；
2. 若需要追查具体修复过程，请优先看对应 commit 与 `OPEN_ISSUES.md` 的问题记录。

### 2.2 2026-02-22（方案级复盘与重构定稿启动）

关键产出（保留）：

1. 确认当前主要矛盾已从“局部 bug”转为“方案级复杂度偏航”。
2. 定稿三层分离方向：
   - HF 语义层（selector）
   - 布局/回收层（layout engine）
   - 运行时输入适配层（runtime input adapter）
3. 启动模块拆分与职责收敛（`hook_impl`、`runner`、`gpu_seq_len_patch` 等）。
4. 形成新的架构/执行文档基线（后续以文档 SSOT 为准）。

### 2.3 2026-02-22（重构推进阶段，保留摘要）

本轮重构推进中已完成的“对当前接手仍有价值”的事实：

1. `hook_impl.py` 已显著瘦身，部分职责外移（运行时口径、group 管线、preflight 等）。
2. `runner.py` 已向编排器方向收敛（压缩执行块、生命周期/信号摄取、输出桥接等逻辑拆出）。
3. `gpu_seq_len_patch.py` 已被降级为兼容入口层，底层 patch 能力拆分到 `input_patch_*` 模块。
4. 新增 `selector/layout/input_adapter` 等模块，为后续主线收敛提供落位。
5. 当时的重构代码基线已通过一轮 `tests_v2` 与 `run_smoke.py` 验证（详见 `CURRENT_STATUS.md` 记录）。

说明：

1. 原日志中 5.5～5.20 的逐条拆分记录已压缩为本节摘要；
2. 若后续需要具体到某次模块拆分动作，请查对应提交与代码 diff。

---

## 3. 2026-02-23 方案调整与拍板结果（当前最重要）

本节是当前接手最需要看的执行事实摘要。

### 3.1 本轮复盘结论（为何调整）

基于 `PROJECT_GOAL`、运行时链路复盘与低分异常表现排查，确认当前主要矛盾是：

1. 压缩后有效上下文状态一致性风险（P0 正确性）
2. decode 热路径 patch-heavy 逻辑导致 CPU 拖 GPU（P0 性能/复杂度）

### 3.2 本轮拍板结果（执行边界）

1. 主线目标模式仅：
   - `per_head`
   - `per_layer_per_head`
2. `per_layer` 不是目标，也不是中间收敛态。
3. 布局层主路径为低搬运 fill-hole（不以物理保序为正确性要求）。
4. runtime adapter 目标形态为：
   - 压缩点更新持久状态
   - decode 薄适配
5. decode 热路径新增代码与 metadata 必须最小化（性能优先）。
6. 允许少量 monkey patch / 函数替换，但必须薄、集中、受控（不改 vLLM 源码目录）。
7. “显存未满不压缩、显存压力触发压缩”属于后续功能；当前方案只要求保留清晰接入点。

### 3.3 文档系统调整（已完成）

1. 已将本轮方案与优先级固化到：
   - `interface/V2_SCHEME_ADJUSTMENT_2026-02-23.md`
   - `interface/V2_OVERVIEW.md`
   - `backend/V2_FINAL_ARCHITECTURE.md`
   - `backend/DESIGN_DECISIONS.md`
2. 已清理多份过时/重叠文档，减少重复入口。
3. 已将重叠问题描述收敛到 `OPEN_ISSUES.md` 和方案文档，避免“多个当前结论版本”并存。

---

## 4. 当前开发焦点（供接手者直接行动）

当前不建议继续在旧 patch-heavy 路径上叠修复。主线焦点应为：

1. **状态一致性验证（优先级最高）**
   - 用最小测试验证压缩后 keep 集合、`seq_lens`、`positions`、`slot_mappings` 一致。
2. **Runtime Adapter 收敛**
   - 将 decode 适配收敛为薄路径，减少每步 Python/metadata 开销。
3. **布局层与写入一致性（目标模式）**
   - 围绕 `per_head` / `per_layer_per_head` 验证 fill-hole 与后续写入安全。
4. **之后再做性能回归**
   - 正确性未稳定前，不以 full-run 吞吐结果做主判断。

对应执行计划见：

1. `interface/V2_REFACTOR_EXECUTION_PLAN_2026-02-22.md`
2. `interface/V2_SCHEME_ADJUSTMENT_2026-02-23.md`

---

## 5. 接手提示（避免再次走偏）

1. 先读 `V2_OVERVIEW.md` 和 `V2_SCHEME_ADJUSTMENT_2026-02-23.md`，再看代码。
2. 不要把 `per_layer` 当成主线中间态来规划任务。
3. 不要为了“规范扩展”牺牲前两优先级（HF 对齐、decode 性能）。
4. 也不要为了“快修”回到 patch-heavy 热路径设计。
5. 任何新增 decode 热路径 metadata，都需要说明必要性与性能理由。

---

## 6. 日志维护规则（本文件专用）

后续更新本文件时：

1. 只保留仍影响当前开发判断的事实与里程碑；
2. 已完成且不再影响当前决策的长条目应压缩/删除；
3. 详细施工过程交给 commit 历史与代码 diff，不在本文件长期堆积。
