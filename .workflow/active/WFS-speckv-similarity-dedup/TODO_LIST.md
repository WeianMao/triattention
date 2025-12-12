# Tasks: SpecKV + Similarity Deduplication

## Task Progress

### Phase 1: Algorithm Core
- [x] **IMPL-001**: Add similarity scoring parameters to SparsePruningConfig → [📋](./.task/IMPL-001.json)
- [x] **IMPL-002**: Integrate similarity scoring into _select_keep_indices method → [📋](./.task/IMPL-002.json)

### Phase 2: Parameter Forwarding
- [x] **IMPL-003**: Add CLI arguments to rkv_sharded_eval.py runner → [📋](./.task/IMPL-003.json)
- [x] **IMPL-004**: Add CLI arguments to rkv_sharded_dispatch.py dispatcher → [📋](./.task/IMPL-004.json)

### Phase 3: Experiment Configurations
- [x] **IMPL-005**: Create 5 YAML config files for mix_lambda experiments → [📋](./.task/IMPL-005.json)
  - Parallel steps: lambda01, lambda03, lambda05, lambda07, lambda09
- [x] **IMPL-006**: Create 5 shell scripts for mix_lambda experiments → [📋](./.task/IMPL-006.json)
  - Parallel steps: 5 shell scripts in speckv_similarity/

### Phase 4: Validation
- [x] **IMPL-007**: Verify backward compatibility with existing SpecKV experiments → [📋](./.task/IMPL-007.json)

## Task Dependencies

```
IMPL-001 → IMPL-002 → IMPL-003 → IMPL-004
                         ↓
                      IMPL-005 (parallel config creation)
                         ↓
                      IMPL-006 (parallel script creation)
                         ↓
                      IMPL-007 (compatibility testing)
```

## Critical Files

**Algorithm Modifications**:
- R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py (IMPL-001, IMPL-002)

**Parameter Forwarding**:
- R-KV/weian_development/rkv_sharded_eval.py (IMPL-003)
- R-KV/weian_development/rkv_sharded_dispatch.py (IMPL-004)

**Experiment Artifacts**:
- R-KV/weian_script/configs/sample8_speckv_similarity_lambda*.yaml (IMPL-005)
- R-KV/weian_script/aime24_official_sampled8/speckv_similarity/*.sh (IMPL-006)

**Testing**:
- R-KV/weian_development/test_speckv_backward_compat.py (IMPL-007)

## Status Legend
- `- [ ]` = Pending task
- `- [x]` = Completed task

## Quick Start After Planning

1. **Review IMPL_PLAN.md** for complete implementation strategy
2. **Start with IMPL-001** (config parameters) - foundation for all other tasks
3. **Proceed sequentially** through IMPL-002 → IMPL-003 → IMPL-004
4. **Parallelize IMPL-005/006** (config and script creation can happen concurrently)
5. **Finish with IMPL-007** (verify everything works) before running experiments

## Experiment Execution (After Implementation)

Once all tasks complete, run experiments:
```bash
# Lambda = 0.1 (R-KV default)
bash R-KV/weian_script/aime24_official_sampled8/speckv_similarity/run_speckv_similarity_lambda01.sh

# Lambda = 0.3
bash R-KV/weian_script/aime24_official_sampled8/speckv_similarity/run_speckv_similarity_lambda03.sh

# Lambda = 0.5 (balanced)
bash R-KV/weian_script/aime24_official_sampled8/speckv_similarity/run_speckv_similarity_lambda05.sh

# Lambda = 0.7
bash R-KV/weian_script/aime24_official_sampled8/speckv_similarity/run_speckv_similarity_lambda07.sh

# Lambda = 0.9 (freq-heavy)
bash R-KV/weian_script/aime24_official_sampled8/speckv_similarity/run_speckv_similarity_lambda09.sh
```

Compare results with baseline:
```bash
# Baseline (SpecKV without similarity, for reference)
bash R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8_norm.sh
```
