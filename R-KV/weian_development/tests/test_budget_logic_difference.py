"""
Test to precisely compare budget logic between Original and RKV-style SpeckV.

This test reveals a CRITICAL difference in how the two implementations handle
the budget when include_prefill_in_budget=True.

Scenario: seq_len=100, prefix_length=30, budget=60
"""
import torch

def trace_original_logic():
    """
    Trace the original SpeckV budget logic step by step.

    When include_prefill_in_budget=True:
    - _dynamic_cache_size = len(cache_positions) = 100
    - The pruning is triggered via ensure_capacity or start_next_round

    Looking at ensure_capacity (line 189-197):
        keep_capacity = max(0, self.max_keys - self.round_window)
        if self._dynamic_cache_size <= keep_capacity:
            return past_key_values
        pruned = self._prune_to_size(past_key_values, keep_capacity, dynamic_only=True)

    And _prune_to_size with dynamic_only=True (line 247-265):
        prefix_count = min(self.prefix_length, candidate_count)  # 30
        dynamic_count = max(0, candidate_count - prefix_count)   # 70
        keep_count = max(0, min(keep_count, dynamic_count))      # min(keep_capacity, 70)

    The key question: what is keep_capacity?
    """
    print("="*70)
    print("ORIGINAL SPECKV BUDGET LOGIC TRACE")
    print("="*70)

    # Parameters
    seq_len = 100
    prefix_length = 30
    budget = 60  # max_keys
    round_window = 32  # typical value from aligned config

    # Step 1: _dynamic_cache_size calculation
    # When include_prefill_in_budget=True:
    dynamic_cache_size = seq_len  # = len(cache_positions)
    print(f"\n1. _dynamic_cache_size = {dynamic_cache_size} (include_prefill_in_budget=True)")

    # Step 2: Trigger check in ensure_capacity
    keep_capacity = max(0, budget - round_window)
    print(f"2. keep_capacity = max(0, {budget} - {round_window}) = {keep_capacity}")

    # Step 3: Is pruning needed?
    needs_pruning = dynamic_cache_size > keep_capacity
    print(f"3. Needs pruning? {dynamic_cache_size} > {keep_capacity} = {needs_pruning}")

    # Step 4: _prune_to_size with dynamic_only=True
    candidate_count = seq_len
    prefix_count = min(prefix_length, candidate_count)
    dynamic_count = max(0, candidate_count - prefix_count)
    actual_keep_count = max(0, min(keep_capacity, dynamic_count))

    print(f"\n4. _prune_to_size(keep_count={keep_capacity}, dynamic_only=True):")
    print(f"   prefix_count = min({prefix_length}, {candidate_count}) = {prefix_count}")
    print(f"   dynamic_count = max(0, {candidate_count} - {prefix_count}) = {dynamic_count}")
    print(f"   actual_keep_count = max(0, min({keep_capacity}, {dynamic_count})) = {actual_keep_count}")

    # Step 5: Final cache size
    # prefix + selected_dynamic
    final_size = prefix_count + actual_keep_count
    print(f"\n5. Final cache size = {prefix_count} (prefix) + {actual_keep_count} (selected) = {final_size}")

    return final_size


def trace_rkv_style_logic():
    """
    Trace the RKV-style SpeckV budget logic step by step.
    """
    print("\n" + "="*70)
    print("RKV-STYLE SPECKV BUDGET LOGIC TRACE")
    print("="*70)

    # Parameters
    seq_len = 100
    prefix_length = 30
    budget = 60  # comp.budget

    # Step 1: effective_size calculation
    # When include_prefill_in_budget=True:
    effective_size = seq_len  # No subtraction
    print(f"\n1. effective_size = {effective_size} (include_prefill_in_budget=True)")

    # Step 2: Trigger check
    should_compress = effective_size >= budget
    print(f"2. Should compress? {effective_size} >= {budget} = {should_compress}")

    # Step 3: compute_keep_indices (line 207-228)
    # decode_budget = budget - prefix_length
    decode_budget = budget - prefix_length
    print(f"\n3. compute_keep_indices(budget={budget}, prefix_length={prefix_length}):")
    print(f"   decode_budget = {budget} - {prefix_length} = {decode_budget}")

    # Step 4: Final selection
    # keep_indices = prefix (30) + selected_decode (30)
    final_size = min(budget, prefix_length + decode_budget)
    print(f"\n4. Final cache size = min({budget}, {prefix_length} + {decode_budget}) = {final_size}")

    return final_size


def compare_behaviors():
    """
    Compare the two behaviors and identify the root cause of difference.
    """
    print("\n" + "="*70)
    print("COMPARISON AND ROOT CAUSE ANALYSIS")
    print("="*70)

    orig_size = trace_original_logic()
    rkv_size = trace_rkv_style_logic()

    print("\n" + "-"*70)
    print("COMPARISON RESULT")
    print("-"*70)

    print(f"\n  Original final cache size: {orig_size}")
    print(f"  RKV-style final cache size: {rkv_size}")

    if orig_size == rkv_size:
        print("\n  ✅ IDENTICAL behavior")
    else:
        print(f"\n  ⚠️ DIFFERENT by {abs(orig_size - rkv_size)} tokens")

        print("\n  ROOT CAUSE:")
        print("  " + "="*60)
        print("  Original: keep_capacity = budget - round_window")
        print("            Then: select min(keep_capacity, dynamic_count) from dynamic")
        print("            Result: prefix + min(keep_capacity, dynamic_count)")
        print()
        print("  RKV-style: decode_budget = budget - prefix_length")
        print("             Then: select decode_budget from decode tokens")
        print("             Result: prefix + decode_budget = budget")
        print()
        print("  The DIFFERENCE is in how 'budget' is interpreted:")
        print("  - Original: Uses (budget - round_window) as the dynamic portion budget")
        print("  - RKV-style: Uses (budget - prefix_length) as the decode portion budget")
        print()
        print("  When include_prefill_in_budget=True:")
        print("  - Original preserves prefix + (budget - round_window) tokens")
        print("  - RKV-style preserves exactly 'budget' tokens total")

    return orig_size == rkv_size


def test_with_aligned_parameters():
    """
    Test with actual aligned experiment parameters.
    """
    print("\n" + "="*70)
    print("TEST WITH ALIGNED EXPERIMENT PARAMETERS")
    print("="*70)

    # From aime_sampled8_speckv_aime25_qwen_norm_aligned.yaml:
    # kv_budget: 2048
    # sparse_round_window: 32
    # Typical prefill: ~150 tokens

    budget = 2048
    round_window = 32
    prefix_length = 150

    print(f"\n  Parameters:")
    print(f"    budget (kv_budget): {budget}")
    print(f"    round_window (sparse_round_window): {round_window}")
    print(f"    prefix_length (typical AIME): {prefix_length}")

    # Original: keep_capacity = 2048 - 32 = 2016
    # RKV-style: decode_budget = 2048 - 150 = 1898

    keep_capacity_orig = budget - round_window
    decode_budget_rkv = budget - prefix_length

    print(f"\n  Original implementation:")
    print(f"    keep_capacity = {budget} - {round_window} = {keep_capacity_orig}")
    print(f"    Final size = {prefix_length} + min({keep_capacity_orig}, decode_count) tokens")

    print(f"\n  RKV-style implementation:")
    print(f"    decode_budget = {budget} - {prefix_length} = {decode_budget_rkv}")
    print(f"    Final size = {prefix_length} + {decode_budget_rkv} = {budget} tokens")

    # Assuming decode_count >> keep_capacity in steady state:
    final_orig = prefix_length + keep_capacity_orig
    final_rkv = budget

    print(f"\n  Expected final cache size:")
    print(f"    Original: ~{final_orig} tokens")
    print(f"    RKV-style: {final_rkv} tokens")

    print(f"\n  CONCLUSION:")
    print(f"    Original keeps ~{final_orig - final_rkv} MORE tokens than RKV-style")
    print(f"    This is because round_window ({round_window}) < prefix_length ({prefix_length})")

    return final_orig == final_rkv


if __name__ == "__main__":
    compare_behaviors()
    print("\n")
    test_with_aligned_parameters()
