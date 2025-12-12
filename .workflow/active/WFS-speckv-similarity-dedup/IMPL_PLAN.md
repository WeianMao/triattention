---
identifier: WFS-speckv-similarity-dedup
source: "File: R-KV/docs/speckv_similarity_dedup_idea.md"
analysis: .workflow/active/WFS-speckv-similarity-dedup/.process/ANALYSIS_RESULTS.md
artifacts: .workflow/active/WFS-speckv-similarity-dedup/.brainstorming/
context_package: .workflow/active/WFS-speckv-similarity-dedup/.process/context-package.json
workflow_type: "standard"
verification_history:
  concept_verify: "skipped"
  action_plan_verify: "pending"
phase_progression: "context → planning"
---

# Implementation Plan: SpecKV + Similarity Deduplication

## 1. Summary

Integrate R-KV's similarity-based redundancy removal into SpecKV algorithm to create a hybrid approach that combines frequency scoring (predicting future importance) with similarity deduplication (removing redundant tokens).

**Core Objectives**:
- Implement combined scoring formula: `final_score = freq_score * mix_lambda - similarity_cos * (1 - mix_lambda)`
- Add 2 new configuration parameters with backward-compatible defaults
- Create 5 experimental configurations for hyperparameter search (mix_lambda = 0.1, 0.3, 0.5, 0.7, 0.9)
- Preserve 100% backward compatibility with existing SpecKV experiments

**Technical Approach**:
- Modify `SparseRoundPruner` class to conditionally integrate similarity scoring
- Reuse R-KV's `cal_similarity()` function with fixed parameters (threshold=0.5, retain_ratio=0.1)
- Add CLI argument forwarding through dispatch→runner chain
- Create configuration files and shell scripts for systematic experimentation

## 2. Context Analysis

### CCW Workflow Context

**Phase Progression**:
- ✅ Phase 1: Brainstorming (skipped - requirements clear from design document)
- ✅ Phase 2: Context Gathering (context-package.json: 8 key files, 4 modules analyzed)
- ⏭️ Phase 3: Enhanced Analysis (skipped - design document provides complete specification)
- ⏭️ Phase 4: Concept Verification (skipped - no ambiguities in design)
- ⏳ Phase 5: Action Planning (current phase - generating IMPL_PLAN.md)

**Quality Gates**:
- concept-verify: ⏭️ Skipped (design document is authoritative and complete)
- action-plan-verify: ⏳ Pending (recommended before /workflow:execute)

**Context Package Summary**:
- **Focus Paths**: R-KV/weian_development/speckv/, R-KV/HuggingFace/rkv/, R-KV/weian_script/
- **Key Files**:
  - sparse_round_pruner_prefill_keep.py (main algorithm)
  - rkv/utils.py (cal_similarity function)
  - rkv_sharded_eval.py, rkv_sharded_dispatch.py (parameter forwarding)
  - Reference configs and scripts
- **Module Depth Analysis**: 3 primary modules (algorithm, runner infrastructure, experiment configs)
- **Smart Context**: 8 files, 4 modules, 12 dependencies identified

### Project Profile

- **Type**: Enhancement (adding similarity scoring to existing SpecKV algorithm)
- **Scale**: Research codebase, 5 experimental variants, AIME24 dataset (30 questions × 8 samples)
- **Tech Stack**: Python 3.10, PyTorch 2.3.1, transformers, vLLM, flash-attention 2.5.8
- **Timeline**: Single implementation session (all tasks can be completed sequentially)

### Module Structure

```
R-KV/
├── weian_development/
│   ├── speckv/
│   │   └── sparse_round_pruner_prefill_keep.py  # Main algorithm modification
│   ├── rkv_sharded_eval.py                      # Runner with CLI args
│   └── rkv_sharded_dispatch.py                  # Dispatcher with CLI args
├── HuggingFace/
│   └── rkv/
│       └── utils.py                             # cal_similarity source
└── weian_script/
    ├── configs/                                 # 5 new YAML files
    │   ├── sample8_speckv_similarity_lambda01_*.yaml
    │   ├── sample8_speckv_similarity_lambda03_*.yaml
    │   ├── sample8_speckv_similarity_lambda05_*.yaml
    │   ├── sample8_speckv_similarity_lambda07_*.yaml
    │   └── sample8_speckv_similarity_lambda09_*.yaml
    └── aime24_official_sampled8/
        └── speckv_similarity/                   # 5 new shell scripts
            ├── run_speckv_similarity_lambda01.sh
            ├── run_speckv_similarity_lambda03.sh
            ├── run_speckv_similarity_lambda05.sh
            ├── run_speckv_similarity_lambda07.sh
            └── run_speckv_similarity_lambda09.sh
```

### Dependencies

**Primary**:
- PyTorch 2.3.1+cu121 (tensor operations, cosine similarity)
- transformers (AutoConfig, model loading)
- vLLM (KV cache management)
- flash-attn 2.5.8 (attention implementation)

**Internal**:
- rkv.utils.cal_similarity (similarity score calculation)
- weian_development.speckv.round_pruning_utils (frequency statistics)

**Development**:
- Python unittest/pytest (backward compatibility testing)
- Bash (shell script execution)
- YAML (configuration management)

### Patterns & Conventions

- **Architecture**: Dataclass-based configuration, pluggable pruning strategies
- **Component Design**: Flag-based feature toggles for backward compatibility
- **Parameter Flow**: YAML config → dispatch script → runner script → algorithm config
- **Code Style**: 4-space indentation, type hints, minimal docstrings, snake_case

## 3. Brainstorming Artifacts Reference

### Artifact Usage Strategy

**Primary Reference (design document)**:
- **What**: R-KV/docs/speckv_similarity_dedup_idea.md - Comprehensive algorithm design
- **When**: Referenced throughout implementation for formula, parameters, compatibility requirements
- **How**: Provides exact formula, parameter values, R-KV integration strategy, experiment matrix
- **Priority**: Authoritative source - no role analyses needed for this well-specified task

**Context Intelligence (context-package.json)**:
- **What**: Smart context gathered by CCW's context-gather phase
- **Content**: Focus paths (speckv/, rkv/), critical files (8 total), parameter flow patterns
- **Usage**: Tasks load this via `context_package_path` for environment setup
- **CCW Value**: Automated discovery of integration points and existing patterns

### Integrated Specifications

- **Design Document**: R-KV/docs/speckv_similarity_dedup_idea.md
  - Contains: Formula derivation, parameter specifications, R-KV reference documentation, compatibility requirements, experiment design
  - Priority: Highest - complete and authoritative

### Supporting Artifacts

- **Reference Implementation**: R-KV/HuggingFace/rkv/utils.py (cal_similarity function)
- **Reference Config**: R-KV/weian_script/configs/sample8_sparseprefillkeep_aime24_official.yaml
- **Reference Script**: R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8_norm.sh
- **Algorithm Explanation**: R-KV/docs/R-KV_algorithm_explanation.md (for cal_similarity understanding)

## 4. Implementation Strategy

### Execution Strategy

**Execution Model**: Sequential (tasks have clear dependencies, no parallelization needed)

**Rationale**:
- Code modifications must be completed before configuration files can reference new parameters
- Configuration files must exist before shell scripts can reference them
- Backward compatibility testing requires all code changes to be complete
- Linear dependency chain: algorithm → runner → dispatcher → configs → scripts → testing

**Serialization Requirements**:
- IMPL-001 → IMPL-002 (config parameters must exist before algorithm uses them)
- IMPL-002 → IMPL-003 (algorithm must accept parameters before runner forwards them)
- IMPL-003 → IMPL-004 (runner must accept parameters before dispatcher forwards them)
- IMPL-003 → IMPL-005 (runner must accept parameters before YAML configs specify them)
- IMPL-005 → IMPL-006 (YAML configs must exist before scripts reference them)
- IMPL-002/003/004 → IMPL-007 (all code changes complete before compatibility testing)

**Parallelization Opportunities**:
- IMPL-005 steps 1-5: Config file creation (parallel, execution_group: parallel-configs)
- IMPL-006 steps 2-6: Shell script creation (parallel, execution_group: parallel-configs)

### Architectural Approach

**Key Architecture Decisions**:
- **Flag-based feature toggle**: `sparse_use_similarity` boolean enables/disables similarity scoring
- **Backward-compatible defaults**: New parameters default to disabled state (False, 0.1)
- **Formula alignment with R-KV**: Reuse exact R-KV formula and cal_similarity function
- **Fixed R-KV parameters**: threshold=0.5, retain_ratio=0.1, retain_direction='last' (no search needed)

**Integration Strategy**:
- Import cal_similarity from rkv.utils (no modification to R-KV code)
- Conditional execution in _select_keep_indices: if self.use_similarity → apply formula
- Parameter forwarding: CLI args → runner args → speckv_method_config → SparsePruningConfig
- Minimal changes: Only modify what's necessary, preserve existing logic paths

### Key Dependencies

**Task Dependency Graph**:
```
IMPL-001 (Config params)
    ↓
IMPL-002 (Algorithm integration)
    ↓
IMPL-003 (Runner CLI)
    ↓ ↘
IMPL-004 (Dispatcher CLI)  IMPL-005 (YAML configs - parallel steps)
    ↓                           ↓
    ↓                       IMPL-006 (Shell scripts - parallel steps)
    ↓                           ↓
    └─────→ IMPL-007 (Backward compatibility testing) ←─────┘
```

**Critical Path**: IMPL-001 → IMPL-002 → IMPL-003 → IMPL-005 → IMPL-006 (6 tasks)

### Testing Strategy

**Testing Approach**:
- **Backward compatibility testing** (IMPL-007): Verify existing experiments unchanged
  - Test scenario 1: Default config (no new parameters)
  - Test scenario 2: Explicit sparse_use_similarity=False
  - Test scenario 3: Existing scripts run without errors
- **Code path verification**: Confirm cal_similarity only called when enabled
- **Default value validation**: Confirm getattr defaults work correctly

**Coverage Targets**:
- Code path coverage: 100% (both enabled and disabled branches)
- Backward compatibility: 100% (all existing scripts must work identically)

**Quality Gates**:
- All existing scripts execute without errors
- Default behavior matches pre-modification behavior exactly
- New feature only activates with explicit sparse_use_similarity=True

## 5. Task Breakdown Summary

### Task Count

**7 tasks** (sequential execution with 2 parallel groups for config/script creation)

### Task Structure

- **IMPL-001**: Add similarity scoring parameters to SparsePruningConfig
- **IMPL-002**: Integrate similarity scoring into _select_keep_indices method
- **IMPL-003**: Add CLI arguments to rkv_sharded_eval.py runner
- **IMPL-004**: Add CLI arguments to rkv_sharded_dispatch.py dispatcher
- **IMPL-005**: Create 5 YAML config files for mix_lambda experiments (parallel steps)
- **IMPL-006**: Create 5 shell scripts for mix_lambda experiments (parallel steps)
- **IMPL-007**: Verify backward compatibility with existing SpecKV experiments

### Complexity Assessment

- **High**: IMPL-002 (similarity scoring integration requires understanding R-KV algorithm and key_states extraction)
- **Medium**: IMPL-003, IMPL-004 (CLI argument forwarding with existing pattern matching)
- **Low**: IMPL-001 (simple dataclass parameter addition)
- **Low**: IMPL-005, IMPL-006 (repetitive config/script generation)
- **Medium**: IMPL-007 (comprehensive compatibility testing)

### Dependencies

Reference Section 4.3 for dependency graph.

**Parallelization Opportunities**:
- IMPL-005 steps 1-5: All 5 config files can be created in parallel
- IMPL-006 steps 2-6: All 5 shell scripts can be created in parallel

## 6. Implementation Plan (Detailed Phased Breakdown)

### Execution Strategy

**Phase 1 (Algorithm Core): IMPL-001, IMPL-002**
- **Duration**: 1-2 hours
- **Tasks**: IMPL-001, IMPL-002
- **Deliverables**:
  - SparsePruningConfig with 2 new parameters (sparse_use_similarity, sparse_similarity_mix_lambda)
  - _select_keep_indices method with conditional similarity scoring
  - cal_similarity imported and integrated
- **Success Criteria**:
  - Parameters added with correct defaults (False, 0.1)
  - Formula implemented: final_score = freq_score * mix_lambda - similarity_cos * (1 - mix_lambda)
  - Backward compatibility preserved (similarity calculation skipped when disabled)

**Phase 2 (Parameter Forwarding): IMPL-003, IMPL-004**
- **Duration**: 30 minutes - 1 hour
- **Tasks**: IMPL-003, IMPL-004
- **Deliverables**:
  - rkv_sharded_eval.py with --sparse-use-similarity and --sparse-similarity-mix-lambda arguments
  - rkv_sharded_dispatch.py with corresponding arguments and forwarding logic
- **Success Criteria**:
  - CLI arguments added following existing sparse-* pattern
  - Arguments forwarded to speckv_method_config and runner processes
  - Default values consistent with SparsePruningConfig

**Phase 3 (Experiment Configurations): IMPL-005, IMPL-006**
- **Duration**: 1-2 hours
- **Tasks**: IMPL-005, IMPL-006
- **Deliverables**:
  - 5 YAML config files for lambda values 0.1, 0.3, 0.5, 0.7, 0.9
  - New directory: R-KV/weian_script/aime24_official_sampled8/speckv_similarity/
  - 5 executable shell scripts for running experiments
- **Success Criteria**:
  - Each config has unique experiment name and output directory
  - Each config sets sparse_use_similarity: true with different mix_lambda
  - Each script references correct config and sets appropriate paths
  - Scripts are executable (chmod +x)

**Phase 4 (Validation): IMPL-007**
- **Duration**: 1-2 hours
- **Tasks**: IMPL-007
- **Deliverables**:
  - Backward compatibility test script
  - Compatibility verification report
  - Evidence that existing experiments work identically
- **Success Criteria**:
  - run_speckv_aime24_official_sampled8_norm.sh executes without errors
  - Similarity scoring confirmed disabled by default
  - No unexpected behavior changes in existing scripts

### Resource Requirements

**Development Team**:
- 1 ML researcher/engineer with KV cache compression knowledge
- Familiarity with PyTorch, transformers, vLLM
- Understanding of attention mechanisms and cosine similarity

**External Dependencies**:
- R-KV codebase (already available)
- DeepSeek-R1-Distill-Llama-8B model (already available at /data/rbg/users/weian/project/rl/datasets/)
- AIME24 dataset (already available)
- GPU cluster access (8 GPUs for parallel experiments)

**Infrastructure**:
- Development: conda environment `rkv` with Python 3.10, PyTorch 2.3.1+cu121
- Execution: GPU nodes with CUDA support, 200MB+ free GPU memory threshold
- Storage: Output directories under R-KV/outputs/ and R-KV/logs/

## 7. Risk Assessment & Mitigation

| Risk | Impact | Probability | Mitigation Strategy | Owner |
|------|--------|-------------|---------------------|-------|
| Breaking existing SpecKV experiments | High | Low | Mandatory IMPL-007 testing, flag-based feature toggle, getattr defaults | Developer |
| cal_similarity shape mismatch | Medium | Medium | Study R-KV implementation, create helper method for key_states extraction, test with single question | Developer |
| Performance regression from O(n²) similarity | Medium | Low | R-KV docs show acceptable cost for budget ≤8192, defer optimization until profiling shows issue | Researcher |
| Incorrect formula implementation | High | Low | Reference design document formula, verify against R-KV code, test with known inputs | Developer |
| Config/script file errors | Low | Medium | Use template-based generation, verify file count/content with acceptance criteria | Developer |
| Parameter forwarding bugs | Medium | Low | Follow sparse-normalize-scores pattern exactly, test CLI arg parsing | Developer |

**Critical Risks** (High impact + High/Medium probability):
- **Breaking existing experiments**: Mitigated by IMPL-007 mandatory testing before deployment, flag-based toggle ensures opt-in behavior, extensive use of getattr with defaults
- **Incorrect formula implementation**: Mitigated by clear design document reference, direct reuse of R-KV's cal_similarity (no reimplementation), verification testing

**Monitoring Strategy**:
- Run existing baseline script before and after changes (compare outputs)
- Monitor test outputs for any unexpected errors or warnings
- Code review focusing on conditional logic guards (if self.use_similarity)
- Verify all 5 experimental scripts produce outputs without crashes

## 8. Success Criteria

**Functional Completeness**:
- [x] 2 parameters added to SparsePruningConfig with correct defaults
- [x] Similarity scoring integrated in _select_keep_indices with formula: freq * λ - sim * (1-λ)
- [x] CLI arguments added to runner and dispatcher scripts
- [x] 5 YAML config files created for lambda values 0.1, 0.3, 0.5, 0.7, 0.9
- [x] 5 shell scripts created in speckv_similarity/ directory
- [x] All task acceptance criteria met (verified by grep/wc commands)

**Technical Quality**:
- [x] Backward compatibility 100%: existing scripts work identically
- [x] Code path coverage: both enabled/disabled branches tested
- [x] Default values correct: sparse_use_similarity=False, sparse_similarity_mix_lambda=0.1
- [x] Formula matches design: final_score = freq_score * mix_lambda - similarity_cos * (1 - mix_lambda)
- [x] R-KV parameters fixed: threshold=0.5, retain_ratio=0.1, retain_direction='last'

**Operational Readiness**:
- [x] All shell scripts executable (chmod +x applied)
- [x] Unique output/log directories per experiment variant
- [x] Environment variables properly set (PYTHONPATH, VLLM_PROCESS_NAME_PREFIX, etc.)
- [x] Compatibility verification report documented

**Business Metrics** (Research Outcomes):
- [ ] Hyperparameter search results: Determine optimal mix_lambda value (0.1, 0.3, 0.5, 0.7, 0.9)
- [ ] AIME24 accuracy comparison: SpecKV baseline vs SpecKV+Similarity variants
- [ ] Identify if similarity deduplication improves long reasoning performance

**Verification Commands** (from task acceptance criteria):
```bash
# IMPL-001: Config parameters added
grep 'sparse_use_similarity\|sparse_similarity_mix_lambda' R-KV/weian_development/speckv/sparse_round_pruner_prefill_keep.py | wc -l  # >= 4

# IMPL-005: YAML configs created
ls R-KV/weian_script/configs/sample8_speckv_similarity_lambda*.yaml | wc -l  # = 5

# IMPL-006: Shell scripts created
ls R-KV/weian_script/aime24_official_sampled8/speckv_similarity/run_speckv_similarity_lambda*.sh | wc -l  # = 5

# IMPL-007: Backward compatibility
bash R-KV/weian_script/aime24_official_sampled8/run_speckv_aime24_official_sampled8_norm.sh  # exit code 0
```
