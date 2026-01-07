"""
分析 Bug 896cbca6 导致的 Δ（相位偏移量）分布

Δ_j = 实际编码位置 - cache_positions[j]

这个脚本分析：
1. 每道题的 prefill 和 decode 长度
2. 模拟 bug 情况下 Δ 的分布
3. Δ 对不同频率分量的影响（相位偏移）
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple
import math


def load_outputs(output_dir: Path) -> List[dict]:
    """加载输出文件，返回按题目分组的结果"""
    results = []
    for shard_file in sorted(output_dir.glob("shard*.jsonl")):
        with open(shard_file, "r") as f:
            for line in f:
                results.append(json.loads(line))
    return results


def simulate_bug_delta(
    questions: List[dict],
    budget: int = 2048,
    divide_length: int = 128,
) -> List[dict]:
    """
    模拟 bug 情况下的状态演变，计算每道题的 Δ 分布

    返回每道题的分析结果
    """
    results = []

    # 初始状态
    absolute_position = 0
    cache_positions = []

    for i, q in enumerate(questions):
        prefill_len = q.get("prefill_tokens", 200)
        decode_len = q.get("output_tokens", 2000)

        if i == 0:
            # 第1题：正常初始化
            cache_positions = list(range(prefill_len))
            absolute_position = prefill_len

            # 模拟 decode 过程（简化版，只考虑最终状态）
            # 假设压缩后 cache 保持在 budget 附近
            for d in range(decode_len):
                cache_positions.append(absolute_position)
                absolute_position += 1

                # 模拟压缩（简化：每 divide_length 压缩一次）
                if len(cache_positions) >= budget + divide_length:
                    # 保留 prefill + 随机保留一些 decode
                    # 简化：均匀采样保留
                    prefix_part = cache_positions[:prefill_len]
                    decode_part = cache_positions[prefill_len:]
                    keep_decode = budget - prefill_len
                    if len(decode_part) > keep_decode:
                        # 均匀采样（模拟 SpeckV 打分后的结果）
                        step = len(decode_part) / keep_decode
                        kept_indices = [int(i * step) for i in range(keep_decode)]
                        decode_part = [decode_part[idx] for idx in kept_indices]
                    cache_positions = prefix_part + decode_part

            results.append({
                "question_idx": i,
                "prefill_len": prefill_len,
                "decode_len": decode_len,
                "delta_min": 0,
                "delta_max": 0,
                "delta_mean": 0,
                "is_first_question": True,
                "cache_positions_sample": cache_positions[:10],
                "absolute_position_end": absolute_position,
            })
        else:
            # 第2题及之后：Bug 触发
            # 新的 prefill 用累积的 absolute_position 编码
            actual_encoding_positions = list(range(
                absolute_position,
                absolute_position + prefill_len
            ))

            # cache_positions 被截断成上一题的最后 prefill_len 个
            if len(cache_positions) > prefill_len:
                cache_positions = cache_positions[-prefill_len:]

            # 计算 Δ = 实际编码位置 - cache_positions
            deltas = [
                actual_encoding_positions[j] - cache_positions[j]
                for j in range(min(len(actual_encoding_positions), len(cache_positions)))
            ]

            # 继续 decode（absolute_position 没更新，会重复！）
            # 实际上 prefill 后 absolute_position 还是之前的值
            # decode 时才开始更新
            for d in range(decode_len):
                cache_positions.append(absolute_position)
                absolute_position += 1

            results.append({
                "question_idx": i,
                "prefill_len": prefill_len,
                "decode_len": decode_len,
                "delta_min": min(deltas) if deltas else 0,
                "delta_max": max(deltas) if deltas else 0,
                "delta_mean": sum(deltas) / len(deltas) if deltas else 0,
                "is_first_question": False,
                "actual_encoding_start": actual_encoding_positions[0] if actual_encoding_positions else 0,
                "cache_positions_sample": cache_positions[:10],
                "absolute_position_end": absolute_position,
                "deltas_sample": deltas[:10] if deltas else [],
            })

    return results


def analyze_phase_offset(delta: float, rope_base: float = 10000.0, head_dim: int = 128):
    """
    分析 Δ 对不同频率分量的相位偏移

    RoPE 频率: ω_i = 1 / (base^(2i/dim))
    相位偏移: Δ × ω_i
    """
    freq_count = head_dim // 2
    results = []

    for i in range(freq_count):
        omega_i = 1.0 / (rope_base ** (2 * i / head_dim))
        phase_offset_rad = delta * omega_i
        phase_offset_deg = math.degrees(phase_offset_rad)
        # 归一化到 [-180, 180]
        phase_offset_deg_normalized = ((phase_offset_deg + 180) % 360) - 180
        # 计算绕了多少圈
        num_rotations = phase_offset_rad / (2 * math.pi)

        results.append({
            "freq_idx": i,
            "omega": omega_i,
            "phase_offset_rad": phase_offset_rad,
            "phase_offset_deg": phase_offset_deg_normalized,
            "num_rotations": num_rotations,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="分析 Bug 896cbca6 的 Δ 分布")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("R-KV/outputs/repository/sample8_rkv_aime24_official/shards"),
        help="输出文件目录"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2048,
        help="KV cache budget"
    )
    parser.add_argument(
        "--divide-length",
        type=int,
        default=128,
        help="压缩触发间隔"
    )
    parser.add_argument(
        "--rope-base",
        type=float,
        default=10000.0,
        help="RoPE base frequency"
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Head dimension"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Bug 896cbca6 Δ 分布分析")
    print("=" * 60)

    # 加载数据
    print(f"\n加载输出文件: {args.output_dir}")
    outputs = load_outputs(args.output_dir)
    print(f"共 {len(outputs)} 条记录")

    # 按题目分组（每道题有 num_samples 个样本）
    questions_by_id = {}
    for o in outputs:
        qid = o.get("id", o.get("index", 0))
        if qid not in questions_by_id:
            questions_by_id[qid] = o

    questions = list(questions_by_id.values())
    print(f"共 {len(questions)} 道题目")

    # 统计 prefill 和 decode 长度
    print("\n" + "-" * 60)
    print("题目统计")
    print("-" * 60)

    prefill_lens = [q.get("prefill_tokens", 200) for q in questions]
    decode_lens = [q.get("output_tokens", 2000) for q in questions]

    print(f"Prefill 长度: min={min(prefill_lens)}, max={max(prefill_lens)}, "
          f"mean={sum(prefill_lens)/len(prefill_lens):.1f}")
    print(f"Decode 长度: min={min(decode_lens)}, max={max(decode_lens)}, "
          f"mean={sum(decode_lens)/len(decode_lens):.1f}")

    # 模拟 bug
    print("\n" + "-" * 60)
    print("模拟 Bug 情况下的 Δ 分布")
    print("-" * 60)

    bug_results = simulate_bug_delta(
        questions,
        budget=args.budget,
        divide_length=args.divide_length
    )

    for r in bug_results:
        if r["is_first_question"]:
            print(f"\n题目 {r['question_idx']}: 第1题，正常执行")
            print(f"  prefill={r['prefill_len']}, decode={r['decode_len']}")
            print(f"  absolute_position 结束时: {r['absolute_position_end']}")
        else:
            print(f"\n题目 {r['question_idx']}: Bug 触发")
            print(f"  prefill={r['prefill_len']}, decode={r['decode_len']}")
            print(f"  Δ: min={r['delta_min']}, max={r['delta_max']}, mean={r['delta_mean']:.1f}")
            print(f"  实际编码起始位置: {r.get('actual_encoding_start', 'N/A')}")
            print(f"  Δ 样本: {r.get('deltas_sample', [])}")

    # 分析典型 Δ 对相位的影响
    print("\n" + "-" * 60)
    print("Δ 对相位偏移的影响分析")
    print("-" * 60)

    # 取第2题的平均 Δ 作为典型值
    if len(bug_results) > 1:
        typical_delta = bug_results[1]["delta_mean"]
    else:
        typical_delta = 400  # 默认值

    print(f"\n典型 Δ = {typical_delta:.1f} tokens")
    print(f"RoPE base = {args.rope_base}, head_dim = {args.head_dim}")

    phase_analysis = analyze_phase_offset(
        typical_delta,
        rope_base=args.rope_base,
        head_dim=args.head_dim
    )

    print(f"\n{'频率索引':<10} {'ω':<15} {'相位偏移(度)':<15} {'绕圈数':<15}")
    print("-" * 55)

    # 打印一些代表性的频率
    indices_to_show = [0, 1, 2, 3, 4, 8, 16, 32, 63]
    for i in indices_to_show:
        if i < len(phase_analysis):
            p = phase_analysis[i]
            print(f"{p['freq_idx']:<10} {p['omega']:<15.6f} "
                  f"{p['phase_offset_deg']:<15.1f} {p['num_rotations']:<15.1f}")

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    high_freq_random = sum(1 for p in phase_analysis if abs(p["num_rotations"]) > 0.5)
    low_freq_stable = sum(1 for p in phase_analysis if abs(p["num_rotations"]) < 0.1)

    print(f"\n对于 Δ = {typical_delta:.1f}:")
    print(f"  - 高频分量（绕圈 > 0.5）: {high_freq_random}/{len(phase_analysis)} 个，相位接近随机")
    print(f"  - 低频分量（绕圈 < 0.1）: {low_freq_stable}/{len(phase_analysis)} 个，相位基本不变")

    print("\n" + "-" * 60)
    print("Δ 的来源分析")
    print("-" * 60)
    print("""
Δ = 实际编码位置 - cache_positions[j]

第1题结束时:
  - absolute_position = P1 + D1 (prefill + decode)
  - cache_positions = [0, 1, ..., P1-1, 一些被保留的decode位置]
                       └─ 前缀保护 ─┘  └─ 经过压缩采样 ─┘

第2题 prefill P2 个 token:
  - 实际编码位置 = [P1+D1, P1+D1+1, ..., P1+D1+P2-1]
  - cache_positions 被截断成最后 P2 个

如果压缩后 cache_positions 末尾是近似连续的:
  cache_positions[-P2:] ≈ [P1+D1-P2, P1+D1-P2+1, ..., P1+D1-1]

则:
  Δ[j] = (P1+D1+j) - (P1+D1-P2+j) = P2

结论: Δ ≈ 当前题目的 prefill 长度 (P2)，而不是前一题的 decode 长度！
""")

    # 验证
    print("-" * 60)
    print("验证: Δ vs prefill_len")
    print("-" * 60)
    for r in bug_results[1:6]:  # 只看前几道
        print(f"  题目 {r['question_idx']}: Δ={r['delta_mean']:.0f}, prefill={r['prefill_len']}")


if __name__ == "__main__":
    main()
