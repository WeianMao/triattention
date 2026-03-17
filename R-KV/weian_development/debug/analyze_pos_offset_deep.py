#!/usr/bin/env python3
"""
深入分析position offset bug的提升原因
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# 数据路径
BASELINE_EVAL = "R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_budget/eval/aime_sampled8_speckv_aime24_qwen_norm_aligned/aime24/default-default_math_multi_eval.jsonl"
OFFSET_EVAL = "R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_budget_simulated_pos_offset/eval/aime_sampled8_speckv_aime24_qwen_norm_aligned/aime24/default-default_math_multi_eval.jsonl"

BASELINE_MERGED = "R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_budget/merged/merged.jsonl"
OFFSET_MERGED = "R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_budget_simulated_pos_offset/merged/merged.jsonl"

def load_jsonl(path):
    """加载jsonl文件"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_merged_data_by_question(merged_path):
    """按题目ID组织merged数据"""
    data = load_jsonl(merged_path)
    by_question = defaultdict(list)
    for item in data:
        idx = item.get('index', item.get('idx'))
        by_question[idx].append(item)
    return by_question

def main():
    print("=" * 80)
    print("Position Offset Bug 深入分析")
    print("=" * 80)

    # 加载数据
    baseline_merged = get_merged_data_by_question(BASELINE_MERGED)
    offset_merged = get_merged_data_by_question(OFFSET_MERGED)

    baseline_eval = load_jsonl(BASELINE_EVAL)
    offset_eval = load_jsonl(OFFSET_EVAL)

    # 创建问题idx到eval数据的映射
    baseline_eval_map = {item['idx']: item for item in baseline_eval}
    offset_eval_map = {item['idx']: item for item in offset_eval}

    # 分析1: Prefill长度统计
    print("\n" + "=" * 80)
    print("1. Prefill长度统计 (问题长度)")
    print("=" * 80)

    prefill_lengths = {}
    for idx in baseline_merged:
        samples = baseline_merged[idx]
        if samples:
            # prefill长度对所有sample应该一样
            prefill_len = samples[0].get('prefill_tokens', 0)
            prefill_lengths[idx] = prefill_len

    # 按prefill长度排序
    sorted_by_prefill = sorted(prefill_lengths.items(), key=lambda x: x[1])
    print(f"\nPrefill长度范围: {min(prefill_lengths.values())} - {max(prefill_lengths.values())} tokens")
    print(f"平均Prefill长度: {np.mean(list(prefill_lengths.values())):.0f} tokens")

    print("\n按Prefill长度排序的题目:")
    print(f"{'Idx':<5} {'Prefill':<10}")
    print("-" * 20)
    for idx, length in sorted_by_prefill:
        print(f"{idx:<5} {length:<10}")

    # 分析2: 计算每道题的性能变化和prefill关系
    print("\n" + "=" * 80)
    print("2. 性能变化与Prefill长度的关系")
    print("=" * 80)

    perf_changes = []
    for idx in baseline_eval_map:
        base_item = baseline_eval_map[idx]
        off_item = offset_eval_map[idx]

        answer = str(base_item['answer']).strip().lstrip('0') or '0'

        base_correct = sum(1 for p in base_item['pred'] if (str(p).strip().lstrip('0') or '0') == answer)
        off_correct = sum(1 for p in off_item['pred'] if (str(p).strip().lstrip('0') or '0') == answer)

        base_acc = base_correct / len(base_item['pred']) * 100
        off_acc = off_correct / len(off_item['pred']) * 100
        diff = off_acc - base_acc

        # 获取平均output长度
        base_samples = baseline_merged[idx]
        off_samples = offset_merged[idx]
        base_output_len = np.mean([s.get('output_tokens', 0) for s in base_samples])
        off_output_len = np.mean([s.get('output_tokens', 0) for s in off_samples])

        perf_changes.append({
            'idx': idx,
            'prefill_len': prefill_lengths.get(idx, 0),
            'base_acc': base_acc,
            'off_acc': off_acc,
            'diff': diff,
            'base_output_len': base_output_len,
            'off_output_len': off_output_len,
            'avg_output_len': (base_output_len + off_output_len) / 2
        })

    # 计算prefill长度与性能提升的相关性
    prefill_lens = np.array([p['prefill_len'] for p in perf_changes])
    diffs = np.array([p['diff'] for p in perf_changes])
    output_lens = np.array([p['avg_output_len'] for p in perf_changes])

    if len(prefill_lens) > 1:
        prefill_corr = np.corrcoef(prefill_lens, diffs)[0, 1]
        output_corr = np.corrcoef(output_lens, diffs)[0, 1]
        print(f"\nPrefill长度与性能提升的相关系数: {prefill_corr:.3f}")
        print(f"Output长度与性能提升的相关系数: {output_corr:.3f}")

    # 分析3: 按prefill长度分组的性能统计
    print("\n" + "=" * 80)
    print("3. 按Prefill长度分组的性能统计")
    print("=" * 80)

    # 定义分组
    short_prefill = [p for p in perf_changes if p['prefill_len'] < 150]
    medium_prefill = [p for p in perf_changes if 150 <= p['prefill_len'] < 200]
    long_prefill = [p for p in perf_changes if p['prefill_len'] >= 200]

    groups = [
        ("短Prefill (<150 tokens)", short_prefill),
        ("中Prefill (150-200 tokens)", medium_prefill),
        ("长Prefill (>=200 tokens)", long_prefill)
    ]

    for name, group in groups:
        if group:
            avg_diff = np.mean([p['diff'] for p in group])
            avg_output = np.mean([p['avg_output_len'] for p in group])
            print(f"\n{name} ({len(group)}道题):")
            print(f"  平均性能提升: {avg_diff:+.2f}%")
            print(f"  平均Output长度: {avg_output:.0f} tokens")

    # 分析4: 详细看提升最大的题目
    print("\n" + "=" * 80)
    print("4. 提升最大的题目详细分析")
    print("=" * 80)

    sorted_by_diff = sorted(perf_changes, key=lambda x: x['diff'], reverse=True)

    for p in sorted_by_diff[:7]:  # 看前7个提升最大的
        idx = p['idx']
        print(f"\n题目 {idx}:")
        print(f"  Prefill长度: {p['prefill_len']} tokens")
        print(f"  性能提升: {p['diff']:+.1f}% (baseline={p['base_acc']:.1f}%, offset={p['off_acc']:.1f}%)")
        print(f"  Baseline Output长度: {p['base_output_len']:.0f} tokens")
        print(f"  Offset Output长度: {p['off_output_len']:.0f} tokens")
        print(f"  Output长度变化: {p['off_output_len'] - p['base_output_len']:+.0f} tokens")

    # 分析5: 位置偏移效应的理论分析
    print("\n" + "=" * 80)
    print("5. RoPE位置偏移效应的理论分析")
    print("=" * 80)

    print("""
根据文档分析，Bug 896cbca6的效果是：
- 正常情况: Prefill K positions [0, P-1], Decode Q positions [P, P+D-1]
  → 相对位置 = Q - K = P+d - k (距离从P开始)

- Bug情况: Prefill K positions [0, P-1], Decode Q positions [0, D-1]
  → 相对位置 = Q - K = d - k (距离从0开始)

这意味着在Bug情况下，decode token与prefill token的"距离"更近了！

关键洞察：
1. RoPE的attention权重随相对位置距离衰减
2. 在长推理任务中（如AIME），decode可能生成5000-30000 tokens
3. 正常情况下，在decode第10000个token时，问题(prefill)的相对距离是10000+P
4. Bug情况下，相对距离只有0到P-1

这导致：
- 问题内容在整个推理过程中保持"近距离"，attention权重更高
- 模型更容易持续"看到"问题的关键信息
- 对于需要反复参考问题条件的数学推理任务特别有益
""")

    # 分析6: 量化位置偏移的影响
    print("\n" + "=" * 80)
    print("6. 量化位置偏移的影响")
    print("=" * 80)

    for p in sorted_by_diff[:5]:
        idx = p['idx']
        prefill_len = p['prefill_len']
        avg_output = p['avg_output_len']

        # 正常情况下的最大相对位置
        normal_max_dist = prefill_len + avg_output
        # Bug情况下的最大相对位置
        bug_max_dist = avg_output  # 从0开始

        # 相对位置的减少
        dist_reduction = prefill_len

        print(f"\n题目 {idx} (提升 {p['diff']:+.1f}%):")
        print(f"  Prefill长度: {prefill_len} tokens")
        print(f"  平均Output长度: {avg_output:.0f} tokens")
        print(f"  正常最大相对距离: {normal_max_dist:.0f}")
        print(f"  Bug最大相对距离: {bug_max_dist:.0f}")
        print(f"  距离减少: {dist_reduction} tokens ({dist_reduction/normal_max_dist*100:.1f}%)")

    # 分析7: 为什么中等长度提升最大
    print("\n" + "=" * 80)
    print("7. 为什么中等长度(15k-25k)题目提升最大？")
    print("=" * 80)

    print("""
假设分析：

1. 短回答题目 (<15k tokens):
   - 推理过程短，位置衰减问题不严重
   - 即使没有位置偏移，问题仍然"近"
   - 提升空间有限

2. 中等回答题目 (15k-25k tokens):
   - 推理足够长，位置衰减开始显著
   - 但题目本身难度适中，模型有能力正确解答
   - 位置偏移带来的"问题可见性"提升直接转化为正确率提升
   - 这是"sweet spot"

3. 长回答题目 (>25k tokens):
   - 题目本身非常困难
   - 即使能更好地"看到"问题，推理复杂度太高
   - 其他因素（KV压缩质量、推理能力极限）主导性能
   - 位置偏移的收益被其他瓶颈抵消
""")

    # 分析8: 验证假设 - 看中等长度题目的baseline正确率
    print("\n" + "=" * 80)
    print("8. 验证假设 - 不同长度组的baseline正确率")
    print("=" * 80)

    thresholds = [(0, 15000), (15000, 25000), (25000, float('inf'))]
    names = ["短 (<15k)", "中 (15k-25k)", "长 (>25k)"]

    for name, (low, high) in zip(names, thresholds):
        group = [p for p in perf_changes if low <= p['avg_output_len'] < high]
        if group:
            avg_base = np.mean([p['base_acc'] for p in group])
            avg_off = np.mean([p['off_acc'] for p in group])
            avg_diff = np.mean([p['diff'] for p in group])
            print(f"\n{name} ({len(group)}道题):")
            print(f"  Baseline平均正确率: {avg_base:.1f}%")
            print(f"  Offset平均正确率: {avg_off:.1f}%")
            print(f"  平均提升: {avg_diff:+.1f}%")

if __name__ == "__main__":
    main()
