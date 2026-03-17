"""Test to compare compression triggering behavior between aligned and aligned_budget.

This traces the exact sequence of when compression happens in each implementation.

Key differences:
- aligned (speckv_rkv_style.py): triggers when cache >= budget AND (absolute_position % divide_length == 0)
- aligned_budget (rkv_speckv_generate.py): triggers when cache >= budget + divide_length
"""


def simulate_aligned_compression(prefill_len: int, budget: int, divide_length: int, total_steps: int):
    """Simulate speckv_rkv_style.py compression triggering."""
    cache_size = prefill_len
    absolute_position = prefill_len
    compressions = []

    for step in range(total_steps):
        # Simulate adding one token
        cache_size += 1
        absolute_position += 1
        is_decode_step = True

        # Trigger condition from speckv_rkv_style.py:487-489
        should_compress = (absolute_position % divide_length == 0) if is_decode_step else False
        effective_size = cache_size  # with include_prefill_in_budget=True

        if effective_size >= budget and should_compress:
            # Would compress, but compute_keep_indices only prunes if cache > budget
            if cache_size > budget:
                compressions.append({
                    "step": step,
                    "abs_pos": absolute_position,
                    "before": cache_size,
                    "after": budget,
                })
                cache_size = budget

    return compressions


def simulate_aligned_budget_compression(prefill_len: int, budget: int, divide_length: int, total_steps: int):
    """Simulate rkv_speckv_generate.py compression triggering with rkv_aligned_budget=True."""
    cache_size = prefill_len
    absolute_position = prefill_len
    compressions = []

    trigger_threshold = budget + divide_length

    for step in range(total_steps):
        # Simulate adding one token
        cache_size += 1
        absolute_position += 1

        # Trigger condition from rkv_speckv_generate.py:261-265
        if cache_size >= trigger_threshold:
            compressions.append({
                "step": step,
                "abs_pos": absolute_position,
                "before": cache_size,
                "after": budget,
            })
            cache_size = budget

    return compressions


def main():
    prefill_len = 150
    budget = 2048
    divide_length = 128
    total_steps = 5000

    print("=" * 80)
    print("Compression Trigger Difference Test")
    print("=" * 80)
    print(f"\nPrefill: {prefill_len}, Budget: {budget}, Divide length: {divide_length}")

    aligned = simulate_aligned_compression(prefill_len, budget, divide_length, total_steps)
    aligned_budget = simulate_aligned_budget_compression(prefill_len, budget, divide_length, total_steps)

    print(f"\nTotal compressions:")
    print(f"  aligned: {len(aligned)}")
    print(f"  aligned_budget: {len(aligned_budget)}")

    print("\n" + "-" * 40)
    print("First 5 compression events:")
    print("-" * 40)

    print("\naligned:")
    for c in aligned[:5]:
        print(f"  Step {c['step']:4d}: abs_pos={c['abs_pos']:5d}, cache {c['before']:4d} -> {c['after']:4d}")

    print("\naligned_budget:")
    for c in aligned_budget[:5]:
        print(f"  Step {c['step']:4d}: abs_pos={c['abs_pos']:5d}, cache {c['before']:4d} -> {c['after']:4d}")

    # Check if compression happens at the same points
    aligned_positions = set(c['abs_pos'] for c in aligned)
    aligned_budget_positions = set(c['abs_pos'] for c in aligned_budget)

    overlap = aligned_positions & aligned_budget_positions
    only_aligned = aligned_positions - aligned_budget_positions
    only_budget = aligned_budget_positions - aligned_positions

    print("\n" + "-" * 40)
    print("Compression position comparison:")
    print("-" * 40)
    print(f"  Overlap: {len(overlap)}")
    print(f"  Only in aligned: {len(only_aligned)} positions")
    print(f"  Only in aligned_budget: {len(only_budget)} positions")

    if only_aligned:
        print(f"  First 5 aligned-only: {sorted(only_aligned)[:5]}")
    if only_budget:
        print(f"  First 5 budget-only: {sorted(only_budget)[:5]}")

    # Check the critical difference: what tokens are in cache at compression time
    print("\n" + "-" * 40)
    print("Key insight: compression timing affects which tokens are scored")
    print("-" * 40)

    # For first compression in each
    if aligned and aligned_budget:
        a = aligned[0]
        b = aligned_budget[0]
        print(f"\nFirst compression:")
        print(f"  aligned: at step {a['step']}, absolute_pos={a['abs_pos']}, cache_size={a['before']}")
        print(f"  aligned_budget: at step {b['step']}, absolute_pos={b['abs_pos']}, cache_size={b['before']}")
        print(f"  Difference: {abs(a['step'] - b['step'])} steps, {abs(a['before'] - b['before'])} tokens")


if __name__ == "__main__":
    main()
