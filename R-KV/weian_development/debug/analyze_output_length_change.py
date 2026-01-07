#!/usr/bin/env python3
"""
分析位置偏移对output长度的影响，以及与正确率的关系
"""
import json
import numpy as np
from collections import defaultdict

# 数据路径
BASELINE_EVAL = "R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_budget/eval/aime_sampled8_speckv_aime24_qwen_norm_aligned/aime24/default-default_math_multi_eval.jsonl"
OFFSET_EVAL = "R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_budget_simulated_pos_offset/eval/aime_sampled8_speckv_aime24_qwen_norm_aligned/aime24/default-default_math_multi_eval.jsonl"

BASELINE_MERGED = "R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_budget/merged/merged.jsonl"
OFFSET_MERGED = "R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_budget_simulated_pos_offset/merged/merged.jsonl"

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_merged_data_by_question(merged_path):
    data = load_jsonl(merged_path)
    by_question = defaultdict(list)
    for item in data:
        idx = item.get('index', item.get('idx'))
        by_question[idx].append(item)
    return by_question

def main():
    print("=" * 80)
    print("Output长度变化与正确率提升的深入分析")
    print("=" * 80)

    # 加载数据
    baseline_merged = get_merged_data_by_question(BASELINE_MERGED)
    offset_merged = get_merged_data_by_question(OFFSET_MERGED)
    baseline_eval = {item['idx']: item for item in load_jsonl(BASELINE_EVAL)}
    offset_eval = {item['idx']: item for item in load_jsonl(OFFSET_EVAL)}

    # 收集每道题的数据
    results = []
    for idx in baseline_eval:
        base_item = baseline_eval[idx]
        off_item = offset_eval[idx]
        answer = str(base_item['answer']).strip().lstrip('0') or '0'

        # 计算正确率
        base_correct = sum(1 for p in base_item['pred'] if (str(p).strip().lstrip('0') or '0') == answer)
        off_correct = sum(1 for p in off_item['pred'] if (str(p).strip().lstrip('0') or '0') == answer)
        base_acc = base_correct / len(base_item['pred']) * 100
        off_acc = off_correct / len(off_item['pred']) * 100

        # 获取output长度
        base_samples = baseline_merged[idx]
        off_samples = offset_merged[idx]
        base_output_lens = [s.get('output_tokens', 0) for s in base_samples]
        off_output_lens = [s.get('output_tokens', 0) for s in off_samples]

        results.append({
            'idx': idx,
            'base_acc': base_acc,
            'off_acc': off_acc,
            'acc_diff': off_acc - base_acc,
            'base_output_avg': np.mean(base_output_lens),
            'off_output_avg': np.mean(off_output_lens),
            'output_diff': np.mean(off_output_lens) - np.mean(base_output_lens),
            'prefill': base_samples[0].get('prefill_tokens', 0) if base_samples else 0
        })

    # 分析1: Output长度变化与正确率提升的关系
    print("\n" + "=" * 80)
    print("1. Output长度变化与正确率变化")
    print("=" * 80)

    output_diffs = np.array([r['output_diff'] for r in results])
    acc_diffs = np.array([r['acc_diff'] for r in results])

    corr = np.corrcoef(output_diffs, acc_diffs)[0, 1]
    print(f"\nOutput长度变化与正确率变化的相关系数: {corr:.3f}")

    # 分组: output变短 vs output变长
    shorter_output = [r for r in results if r['output_diff'] < -1000]  # 明显变短
    longer_output = [r for r in results if r['output_diff'] > 1000]    # 明显变长
    similar_output = [r for r in results if -1000 <= r['output_diff'] <= 1000]

    print(f"\nOutput明显变短 (<-1000 tokens, {len(shorter_output)}道题):")
    if shorter_output:
        avg_acc_diff = np.mean([r['acc_diff'] for r in shorter_output])
        avg_output_diff = np.mean([r['output_diff'] for r in shorter_output])
        print(f"  平均正确率变化: {avg_acc_diff:+.2f}%")
        print(f"  平均Output变化: {avg_output_diff:+.0f} tokens")

    print(f"\nOutput明显变长 (>+1000 tokens, {len(longer_output)}道题):")
    if longer_output:
        avg_acc_diff = np.mean([r['acc_diff'] for r in longer_output])
        avg_output_diff = np.mean([r['output_diff'] for r in longer_output])
        print(f"  平均正确率变化: {avg_acc_diff:+.2f}%")
        print(f"  平均Output变化: {avg_output_diff:+.0f} tokens")

    print(f"\nOutput基本不变 (-1000~+1000 tokens, {len(similar_output)}道题):")
    if similar_output:
        avg_acc_diff = np.mean([r['acc_diff'] for r in similar_output])
        avg_output_diff = np.mean([r['output_diff'] for r in similar_output])
        print(f"  平均正确率变化: {avg_acc_diff:+.2f}%")
        print(f"  平均Output变化: {avg_output_diff:+.0f} tokens")

    # 分析2: 详细看每道题
    print("\n" + "=" * 80)
    print("2. 每道题的详细数据 (按正确率提升排序)")
    print("=" * 80)

    sorted_results = sorted(results, key=lambda x: x['acc_diff'], reverse=True)

    print(f"\n{'Idx':<5} {'AccDiff':<10} {'BaseAcc':<10} {'OffAcc':<10} {'OutDiff':<12} {'BaseOut':<10} {'OffOut':<10}")
    print("-" * 77)
    for r in sorted_results:
        print(f"{r['idx']:<5} {r['acc_diff']:+8.1f}%  {r['base_acc']:>8.1f}% {r['off_acc']:>8.1f}% {r['output_diff']:>+10.0f} {r['base_output_avg']:>10.0f} {r['off_output_avg']:>10.0f}")

    # 分析3: 关键发现
    print("\n" + "=" * 80)
    print("3. 关键发现总结")
    print("=" * 80)

    # 找出output变短且正确率提升的题目
    improve_shorter = [r for r in results if r['output_diff'] < 0 and r['acc_diff'] > 0]
    improve_longer = [r for r in results if r['output_diff'] > 0 and r['acc_diff'] > 0]
    decline_shorter = [r for r in results if r['output_diff'] < 0 and r['acc_diff'] < 0]
    decline_longer = [r for r in results if r['output_diff'] > 0 and r['acc_diff'] < 0]

    print(f"\nOutput变短 + 正确率提升: {len(improve_shorter)}道题")
    print(f"Output变长 + 正确率提升: {len(improve_longer)}道题")
    print(f"Output变短 + 正确率下降: {len(decline_shorter)}道题")
    print(f"Output变长 + 正确率下降: {len(decline_longer)}道题")

    if improve_shorter:
        print(f"\n详细 - Output变短 + 正确率提升:")
        for r in sorted(improve_shorter, key=lambda x: x['acc_diff'], reverse=True):
            print(f"  题目{r['idx']}: acc {r['acc_diff']:+.1f}%, output {r['output_diff']:+.0f} tokens")

    # 分析4: 假设检验
    print("\n" + "=" * 80)
    print("4. 可能的解释")
    print("=" * 80)

    print("""
观察到的模式：
- 位置偏移后，某些题目的output变短，但正确率提高
- 这看起来反直觉，因为通常更长的推理=更充分的思考

可能的解释：

假说1: "更好的问题可见性 → 更高效的推理"
- 位置偏移使问题内容在整个推理过程中保持高attention权重
- 模型能更好地"记住"问题的关键条件
- 减少了因"忘记问题"而产生的无效推理步骤
- 结果：更短但更有效的推理路径

假说2: "早期正确引导 → 避免错误路径"
- 在推理早期，模型能更好地理解问题
- 避免了走入错误的推理方向
- 不需要后续的"纠正"步骤
- 结果：更短的正确答案

假说3: "RoPE衰减的非线性效应"
- RoPE的attention权重随距离衰减可能是非线性的
- 在某个距离阈值后，衰减变得更剧烈
- 位置偏移使问题保持在"高权重区域"
- 影响可能比预期更大

实验建议：
1. 分析具体推理内容，看是否有"重复查看问题"的模式变化
2. 对比正确和错误sample的推理路径
3. 可视化attention权重的变化
""")

    # 分析5: 回答长度在数据集中的分布
    print("\n" + "=" * 80)
    print("5. 提升显著题目的回答长度在数据集中的位置")
    print("=" * 80)

    # 计算所有题目的平均output长度（baseline和offset的平均）
    all_avg_outputs = [(r['idx'], (r['base_output_avg'] + r['off_output_avg'])/2) for r in results]
    all_avg_outputs.sort(key=lambda x: x[1])

    significant_idxs = [r['idx'] for r in results if r['acc_diff'] >= 25]

    print("\n数据集output长度分布 (按平均值排序):")
    print(f"{'Rank':<6} {'Idx':<5} {'AvgOutput':<12} {'提升?':<8}")
    print("-" * 35)

    for rank, (idx, avg_out) in enumerate(all_avg_outputs, 1):
        is_sig = "★" if idx in significant_idxs else ""
        r = next(x for x in results if x['idx'] == idx)
        print(f"{rank:<6} {idx:<5} {avg_out:<12.0f} {is_sig:<8}")

    # 统计显著提升题目的长度分位数
    sig_ranks = []
    for idx in significant_idxs:
        rank = next(i for i, (id_, _) in enumerate(all_avg_outputs, 1) if id_ == idx)
        sig_ranks.append(rank)

    if sig_ranks:
        print(f"\n显著提升题目 ({len(significant_idxs)}道):")
        print(f"  长度排名: {sig_ranks}")
        print(f"  平均排名: {np.mean(sig_ranks):.1f} / 30")
        print(f"  中位排名: {np.median(sig_ranks):.1f} / 30")
        print(f"  排名范围: {min(sig_ranks)} - {max(sig_ranks)}")

        # 计算百分位
        percentiles = [(r-1)/30*100 for r in sig_ranks]
        print(f"\n  百分位分布:")
        print(f"    前25%位置 (1-8): {sum(1 for r in sig_ranks if r <= 8)}道")
        print(f"    25%-50%位置 (9-15): {sum(1 for r in sig_ranks if 9 <= r <= 15)}道")
        print(f"    50%-75%位置 (16-22): {sum(1 for r in sig_ranks if 16 <= r <= 22)}道")
        print(f"    后25%位置 (23-30): {sum(1 for r in sig_ranks if r >= 23)}道")

if __name__ == "__main__":
    main()
