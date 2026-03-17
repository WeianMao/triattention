#!/usr/bin/env python3
"""
分析baseline和模拟位置偏移bug实验的差异
"""
import json
import os
from pathlib import Path
from collections import defaultdict

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

def calculate_accuracy_per_question(eval_data):
    """计算每道题的正确率 (8个sample中正确的比例)"""
    question_acc = {}
    for item in eval_data:
        idx = item['idx']
        answer = str(item['answer']).strip().lstrip('0') or '0'
        preds = item['pred']

        correct_count = 0
        for pred in preds:
            pred_str = str(pred).strip().lstrip('0') or '0'
            if pred_str == answer:
                correct_count += 1

        acc = correct_count / len(preds) * 100
        question_acc[idx] = {
            'acc': acc,
            'correct_count': correct_count,
            'total': len(preds),
            'answer': item['answer']
        }
    return question_acc

def get_response_lengths(merged_path):
    """获取每道题每个sample的回复长度"""
    data = load_jsonl(merged_path)
    lengths = defaultdict(list)
    output_tokens = defaultdict(list)  # 也收集output_tokens

    for item in data:
        idx = item.get('index', item.get('idx', item.get('question_idx')))
        response = item.get('output', item.get('response', ''))
        lengths[idx].append(len(response))
        if 'output_tokens' in item:
            output_tokens[idx].append(item['output_tokens'])

    return lengths, output_tokens

def main():
    print("=" * 80)
    print("Position Offset Bug 实验分析")
    print("=" * 80)

    # 加载评估数据
    baseline_eval = load_jsonl(BASELINE_EVAL)
    offset_eval = load_jsonl(OFFSET_EVAL)

    # 计算每道题的正确率
    baseline_acc = calculate_accuracy_per_question(baseline_eval)
    offset_acc = calculate_accuracy_per_question(offset_eval)

    # 计算差异
    print("\n" + "=" * 80)
    print("每道题的正确率对比 (offset - baseline)")
    print("=" * 80)

    diffs = []
    for idx in sorted(baseline_acc.keys()):
        b_acc = baseline_acc[idx]['acc']
        o_acc = offset_acc[idx]['acc']
        diff = o_acc - b_acc
        diffs.append({
            'idx': idx,
            'baseline_acc': b_acc,
            'offset_acc': o_acc,
            'diff': diff,
            'baseline_correct': baseline_acc[idx]['correct_count'],
            'offset_correct': offset_acc[idx]['correct_count'],
            'answer': baseline_acc[idx]['answer']
        })

    # 按差异排序
    diffs.sort(key=lambda x: x['diff'], reverse=True)

    print(f"\n{'Idx':<5} {'Baseline':<12} {'Offset':<12} {'Diff':<10} {'Base_C':<8} {'Off_C':<8}")
    print("-" * 60)
    for d in diffs:
        print(f"{d['idx']:<5} {d['baseline_acc']:<12.1f} {d['offset_acc']:<12.1f} {d['diff']:<+10.1f} {d['baseline_correct']:<8} {d['offset_correct']:<8}")

    # 总体统计
    total_baseline = sum(d['baseline_acc'] for d in diffs) / len(diffs)
    total_offset = sum(d['offset_acc'] for d in diffs) / len(diffs)

    print("\n" + "=" * 80)
    print("总体统计")
    print("=" * 80)
    print(f"Baseline 平均正确率: {total_baseline:.2f}%")
    print(f"Offset   平均正确率: {total_offset:.2f}%")
    print(f"提升:                 {total_offset - total_baseline:+.2f}%")

    # 分析差异大的题目
    print("\n" + "=" * 80)
    print("差异显著的题目 (|diff| >= 25%)")
    print("=" * 80)

    significant_diffs = [d for d in diffs if abs(d['diff']) >= 25]
    for d in significant_diffs:
        print(f"题目 {d['idx']}: baseline={d['baseline_acc']:.1f}%, offset={d['offset_acc']:.1f}%, diff={d['diff']:+.1f}%")

    # 加载merged数据获取回答长度
    print("\n" + "=" * 80)
    print("分析回答长度")
    print("=" * 80)

    baseline_lengths, baseline_tokens = get_response_lengths(BASELINE_MERGED)
    offset_lengths, offset_tokens = get_response_lengths(OFFSET_MERGED)

    # 计算每道题的平均回答长度
    avg_lengths = {}
    for idx in baseline_lengths:
        base_avg = sum(baseline_lengths[idx]) / len(baseline_lengths[idx]) if baseline_lengths[idx] else 0
        off_avg = sum(offset_lengths[idx]) / len(offset_lengths[idx]) if offset_lengths[idx] else 0
        avg_lengths[idx] = {
            'baseline': base_avg,
            'offset': off_avg,
            'avg': (base_avg + off_avg) / 2
        }

    # 所有题目的平均长度排序
    all_avg_lengths = sorted([(idx, avg_lengths[idx]['avg']) for idx in avg_lengths], key=lambda x: x[1])

    print("\n所有题目按平均回答长度排序:")
    print(f"{'Rank':<6} {'Idx':<5} {'Avg Length':<12} {'Baseline':<12} {'Offset':<12}")
    print("-" * 50)
    for rank, (idx, avg_len) in enumerate(all_avg_lengths, 1):
        print(f"{rank:<6} {idx:<5} {avg_len:<12.0f} {avg_lengths[idx]['baseline']:<12.0f} {avg_lengths[idx]['offset']:<12.0f}")

    # 差异显著题目的长度分析
    print("\n" + "=" * 80)
    print("差异显著题目的长度分析")
    print("=" * 80)

    significant_idxs = [d['idx'] for d in significant_diffs]
    total_questions = len(avg_lengths)

    for d in significant_diffs:
        idx = d['idx']
        avg_len = avg_lengths[idx]['avg']
        # 找到该题目在长度排序中的位置
        rank = next(r for r, (i, _) in enumerate(all_avg_lengths, 1) if i == idx)
        percentile = (rank - 1) / total_questions * 100

        print(f"\n题目 {idx}:")
        print(f"  正确率差异: {d['diff']:+.1f}% (baseline={d['baseline_acc']:.1f}%, offset={d['offset_acc']:.1f}%)")
        print(f"  平均回答长度: {avg_len:.0f} tokens")
        print(f"  长度排名: {rank}/{total_questions} (百分位: {percentile:.1f}%)")

    # 分析提升和下降的题目特征
    print("\n" + "=" * 80)
    print("提升 vs 下降题目的长度特征")
    print("=" * 80)

    improved = [d for d in diffs if d['diff'] > 0]
    declined = [d for d in diffs if d['diff'] < 0]
    unchanged = [d for d in diffs if d['diff'] == 0]

    if improved:
        improved_lengths = [avg_lengths[d['idx']]['avg'] for d in improved]
        print(f"\n提升的题目 ({len(improved)}道):")
        print(f"  平均回答长度: {sum(improved_lengths)/len(improved_lengths):.0f} tokens")
        print(f"  长度范围: {min(improved_lengths):.0f} - {max(improved_lengths):.0f}")

    if declined:
        declined_lengths = [avg_lengths[d['idx']]['avg'] for d in declined]
        print(f"\n下降的题目 ({len(declined)}道):")
        print(f"  平均回答长度: {sum(declined_lengths)/len(declined_lengths):.0f} tokens")
        print(f"  长度范围: {min(declined_lengths):.0f} - {max(declined_lengths):.0f}")

    print(f"\n保持不变的题目: {len(unchanged)}道")

    # 长度与提升的相关性分析
    print("\n" + "=" * 80)
    print("长度与性能变化的相关性")
    print("=" * 80)

    # 按长度分组统计
    short_threshold = 15000  # tokens
    medium_threshold = 25000

    short_questions = [d for d in diffs if avg_lengths[d['idx']]['avg'] < short_threshold]
    medium_questions = [d for d in diffs if short_threshold <= avg_lengths[d['idx']]['avg'] < medium_threshold]
    long_questions = [d for d in diffs if avg_lengths[d['idx']]['avg'] >= medium_threshold]

    if short_questions:
        short_avg_diff = sum(d['diff'] for d in short_questions) / len(short_questions)
        print(f"\n短回答题目 (<{short_threshold} tokens, {len(short_questions)}道):")
        print(f"  平均提升: {short_avg_diff:+.2f}%")

    if medium_questions:
        medium_avg_diff = sum(d['diff'] for d in medium_questions) / len(medium_questions)
        print(f"\n中等回答题目 ({short_threshold}-{medium_threshold} tokens, {len(medium_questions)}道):")
        print(f"  平均提升: {medium_avg_diff:+.2f}%")

    if long_questions:
        long_avg_diff = sum(d['diff'] for d in long_questions) / len(long_questions)
        print(f"\n长回答题目 (>={medium_threshold} tokens, {len(long_questions)}道):")
        print(f"  平均提升: {long_avg_diff:+.2f}%")

if __name__ == "__main__":
    main()
