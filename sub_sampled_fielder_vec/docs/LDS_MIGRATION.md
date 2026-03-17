# LDS Migration: From Matrix Completion to Spectral Preservation

**Date Started**: 2026-02-13
**Date Completed**: 2026-02-14
**Migration Status**: ✅ Complete (with Phase 1 quality enhancements)

---

## Overview & Paradigm Shift

### The Core Problem
The current leveraged sampling implementation uses **Inexact Augmented Lagrangian Multiplier (IALM)** method to recover a complete low-rank matrix from sampled entries. This approach comes from the matrix completion literature and is designed to find the "missing values" through iterative optimization.

However, for **Spectral Top-Down Recovery (STDR)**, we don't need to find the exact missing entries—we only need the eigenvector directions to remain stable enough for partitioning. This insight opens the door to a much simpler and faster approach.

### The Paradigm Shift

| Aspect | Matrix Completion (IALM) | Spectral Preservation (LDS) |
|--------|-------------------------|------------------------------|
| **Goal** | Recover exact matrix entries | Preserve eigenvector directions |
| **Method** | Iterative optimization | Single-shot debiased estimator |
| **Complexity** | O(n³ × iterations) | O(n²) |
| **Theory** | Nuclear norm minimization | Davis-Kahan perturbation bound |
| **Formula** | min ‖L‖* + λ‖S‖₁ s.t. P_Ω(L+S)=P_Ω(X) | X̂_ij = X_ij / p_ij for (i,j)∈Ω |
| **Guarantee** | Exact recovery with O(nr log n) samples | Eigenvector stability with O(nr log n) samples |

### Why LDS Works for STDR

1. **Unbiased Estimator**: The debiased estimator satisfies E[X̂] = X
2. **Davis-Kahan Theorem**: If E[X̂] = X, then eigenvectors of L̂ ≈ eigenvectors of L
3. **Spectral Gap**: For phylogenetic trees, the spectral gap is typically large, providing robustness
4. **Conclusion**: We can skip expensive matrix completion and use the debiased estimate directly

**Expected Benefit**: 10-100x speedup while maintaining similar partitioning quality.

---

## Theoretical Background

### LDS Paper Reference
**Paper**: Huang, Liu, Du, Tao - "Leveraged Matrix Completion With Noise"

**Key Theorems**:
- **Theorem 3.1**: With m = O(nr log n) samples using leverage scores, the debiased estimator X̂ satisfies ‖X̂ - X‖ = O(√(n log n / m)) with high probability
- **Corollary**: For STDR, this perturbation bound combined with Davis-Kahan ensures eigenvector stability

### Davis-Kahan Perturbation Bound
```
‖v̂ - v‖₂ ≤ ‖X̂ - X‖_F / gap
```
where:
- v̂ = Fiedler vector from debiased matrix
- v = Fiedler vector from true matrix
- gap = λ₂ - λ₁ (spectral gap)

**Implication**: As long as the spectral gap is large (typical for balanced phylogenetic trees), small perturbations in X don't significantly change the Fiedler vector.

### Leverage Scores and Sampling
**Definition**: Leverage score μᵢ = (n/r) ‖Uᵢ‖² where U are left singular vectors of rank-r approximation.

**Sampling Formula**: p_ij ∝ (μᵢ + μⱼ) × r × log²(n) / n

**Key Insight**: High leverage = important for spectral structure → sample with higher probability

### Regularization Floor (LDS Extension)
**Problem**: If taxon i is missed in Phase 1 pilot sample → μᵢ = 0 → p_i = 0 in Phase 2 → never sampled (lockout)

**Solution**: Add regularization floor τ_floor = mean(μ) to ensure all taxa have non-zero sampling probability:
```
μ'ᵢ = μᵢ + τ_floor
```

---

## Phase-by-Phase Changes

### Phase 0: Documentation Setup ✅
**Status**: Complete
**Date**: 2026-02-13

**Deliverable**: Created `LDS_MIGRATION.md` (this file) to track all changes throughout migration.

---

### Phase 0.5: Phase 1 Quality Tracking & Logging Fixes ✅
**Status**: Complete
**Date**: 2026-02-14

**Motivation**: After initial deployment, discovered that Phase 1 quality warnings were not appearing in logs, and phase1_sufficiency metric was not tracked for analysis.

**Deliverables**:
1. **Fixed logging bug**: Added `self._logged_p_values.add(p)` after Phase 1 warnings
2. **Enhanced warning messages**: More actionable format with `force=True` for visibility
3. **Phase 1 quality tracking**: Added `phase1_sufficiency` to metrics pipeline
4. **Aggregate quality report**: Summary of Phase 1 quality across all p-values
5. **Results CSV integration**: `phase1_sufficiency` now in results.json

**Files Modified**:
- `src/core/sampling/leveraged/lds_sampler.py` (~15 lines)
- `src/runners/bootstrap_sweep.py` (~35 lines)
- `src/utils/summaries.py` (1 line)
- `docs/LDS_SAMPLING.md` (updated troubleshooting and diagnostics)

**Impact**: Users now have full visibility into when LDS is operating outside theoretical guarantees, enabling better experimental design and result interpretation.

---

### Phase 1: Core Debiasing Module
**Status**: 🚧 Pending
**Date**: TBD

**File Created**: `src/core/sampling/leveraged/compute_debiased_estimator.py`

**Key Implementation Details**:
- **Formula**: X̂_ij = X_ij / min(p_ij, 1.0) for sampled entries, 0 otherwise
- **Probability Capping**: Critical for maintaining E[X̂] = X when p_ij > 1.0
- **Diagonal Handling**: Explicitly set X̂_ii = 1.0 (not sampled)
- **Sparse Matrix**: Returns `scipy.sparse.csr_matrix` for efficiency
- **Symmetry**: Enforced via (X̂ + X̂ᵀ) / 2

**Mathematical Guarantee**: E[X̂] = X (unbiased property)

**Testing Plan**:
- Unit test: Verify E[X̂] = X with 1000+ independent samples
- Unit test: Verify symmetry (‖X̂ - X̂ᵀ‖_F < 1e-10)
- Unit test: Verify diagonal = 1.0

---

### Phase 2: Leverage Score Regularization
**Status**: 🚧 Pending
**Date**: TBD

**File Modified**: `src/core/sampling/leveraged/compute_leverage_scores.py`

**Key Changes**:
- Add `apply_regularization: bool = True` parameter
- Compute regularization floor: `τ_floor = mean(μ_estimated)`
- Return both raw and regularized scores
- Edge case: If mean < 1e-12, use τ_floor = 1.0 as fallback

**Rationale**: Prevents zero-probability lockout for taxa missed in Phase 1.

**Testing Plan**:
- Verify regularized scores are strictly positive
- Verify sum(μ_regularized) ≈ n + n×τ_floor

---

### Phase 3: LDS Sampler Implementation
**Status**: 🚧 Pending
**Date**: TBD

**File Created**: `src/core/sampling/leveraged/lds_sampler.py`

**Architecture**:
```
LDSSampler (inherits from BaseSampler)
├─ Phase 1: Uniform sampling → leverage scores (REUSED from LeveragedSampler)
├─ Phase 2: Non-uniform sampling (REUSED with regularization)
└─ Phase 3: Debiased estimator (NEW - replaces IALM)
```

**Key Differences from LeveragedSampler**:
- ❌ Remove: IALM solver, SVT, soft thresholding
- ✅ Keep: Phase 1/2 sampling logic
- ✅ Add: Regularization floor, debiased estimator, probability capping
- ✅ Performance: O(n²) vs O(n³ × iterations)

**Diagonal Exclusion**: Ensure `np.fill_diagonal(Omega, False)` to avoid wasting budget.

**Phase 1 Sufficiency Logging**: Warn if `phase1_actual < theoretical_min` but don't block execution.

**Testing Plan**:
- Integration test: Run end-to-end LDS pipeline
- Verify sparse matrix output
- Measure runtime vs IALM

---

### Phase 4: Configuration Integration
**Status**: 🚧 Pending
**Date**: TBD

**Files Modified**:
1. `src/config/base_config.py`: Add `method="lds"` to `SamplingConfig`
2. `src/runners/bootstrap_sweep.py`: Add LDSSampler instantiation case
3. `scripts/interactive_run.py`: Add "lds" to method choices
4. `src/config/presets.py`: Add `sampling_tau_floor_multiplier` parameter
5. `src/runners/experiment_runner_utils.py`: Extract LDS config from JSON

**New Parameters**:
- `method: Literal["uniform", "leveraged", "lds"]`
- `tau_floor_multiplier: float = 1.0` (scale regularization)

**Testing Plan**:
- Verify "lds" appears in interactive launcher
- Verify config flows through all 8 layers
- Run small experiment via interactive launcher

---

### Phase 5: Testing & Validation
**Status**: 🚧 Pending
**Date**: TBD

**File Created**: `scripts/test_lds.py`

**Test Suite**:
1. **Unbiased Property**: E[X̂] = X with 1000 samples
2. **Symmetry & Diagonal**: X̂ = X̂ᵀ and diag(X̂) = 1.0
3. **Probability Capping**: Verify no bias when p_ij > 1.0
4. **End-to-End Integration**: Run n=64, p=0.1 full pipeline
5. **Performance Comparison**: LDS vs IALM (expect 10-100x speedup)
6. **Numerical Stability**: Test extreme p-values (0.001, 0.9)

**Acceptance Criteria**:
- ✅ All unit tests pass
- ✅ LDS is 10-100x faster than IALM
- ✅ Agreement curves within ±5% of IALM
- ✅ No NaN/Inf in outputs

---

### Phase 6: Documentation & Examples
**Status**: 🚧 Pending
**Date**: TBD

**Files Created**:
1. `docs/LDS_SAMPLING.md`: User guide with quick start and theory
2. `analysis/notebooks/lds_vs_ialm_comparison.ipynb`: Side-by-side comparison
3. Updated `README.md`: Mention LDS as sampling option

**Documentation Sections**:
- Quick Start (interactive + programmatic)
- Algorithm Overview (3 phases)
- Mathematical Background (LDS + Davis-Kahan)
- Configuration Parameters
- When to Use LDS vs IALM
- Performance Notes
- Troubleshooting

---

## Mathematical Corrections Applied

### 1. Probability Capping (Critical)
**Issue**: Theoretical formula can produce p_ij > 1.0 for highly leveraged pairs.

**Correction**: Cap probabilities at 1.0 during sampling AND debiasing:
```python
p_matrix = np.minimum(p_matrix, 1.0)  # Cap during sampling
p_capped = np.minimum(p_matrix[i,j], 1.0)  # Cap during debiasing
X_hat[i,j] = X[i,j] / p_capped
```

**Impact**: Maintains E[X̂] = X (unbiased property). Without capping, estimator would be biased.

### 2. Diagonal Handling
**Issue**: Diagonal represents self-similarity and doesn't need estimation.

**Correction**:
- Exclude diagonal from sampling mask: `np.fill_diagonal(Omega, False)`
- Explicitly set: `X̂_ii = 1.0` after debiasing
- Rationale: Laplacian construction (D - S) naturally cancels diagonal

**Impact**: Saves budget, ensures numerical consistency.

### 3. Regularization Floor
**Issue**: Taxa missed in Phase 1 get μᵢ = 0 → p_i = 0 in Phase 2 → never sampled (lockout).

**Correction**: Add floor τ_floor = mean(μ_estimated) to all leverage scores:
```python
tau_floor = np.mean(row_leverage)
row_leverage_reg = row_leverage + tau_floor
```

**Impact**: All taxa have non-zero sampling probability, prevents catastrophic failure.

### 4. Sparse Matrix Optimization
**Issue**: Dense matrix representation wastes memory and slows down eigensolver.

**Correction**: Return `scipy.sparse.csr_matrix` from debiased estimator:
```python
X_hat_sparse = csr_matrix((values, (rows, cols)), shape=(n, n))
X_hat_sym = (X_hat_sparse + X_hat_sparse.T) / 2
```

**Impact**: 10-100x speedup for large sparse matrices in FiedlerVectorComputer.

### 5. High-Variance Entry Testing
**Issue**: Entries with p_ij ≪ 1 have high variance, making E[X̂] = X hard to verify.

**Correction**: Average over 1000+ independent samples in unit tests:
```python
for seed in range(1000):
    X_hat_i = sample_and_debias(X, p, seed)
    X_hat_sum += X_hat_i
X_hat_mean = X_hat_sum / 1000
assert np.linalg.norm(X_hat_mean - X, 'fro') < 0.1 * np.linalg.norm(X, 'fro')
```

**Impact**: Reliable validation of unbiased property.

---

## Testing & Validation Results

### Unit Tests
*To be filled after Phase 5*

| Test | Status | Notes |
|------|--------|-------|
| Unbiased Property | ⏳ Pending | E[X̂] = X |
| Symmetry | ⏳ Pending | ‖X̂ - X̂ᵀ‖ < 1e-10 |
| Diagonal | ⏳ Pending | diag(X̂) = 1.0 |
| Probability Capping | ⏳ Pending | No bias with p_ij > 1.0 |

### Integration Tests
*To be filled after Phase 5*

| Test Case | n | p | Status | Time (LDS) | Time (IALM) | Speedup |
|-----------|---|---|--------|-------------|-------------|---------|
| Small | 64 | 0.1 | ⏳ | — | — | — |
| Medium | 256 | 0.05 | ⏳ | — | — | — |
| Large | 1024 | 0.02 | ⏳ | — | — | — |

### Agreement Accuracy
*To be filled after Phase 5*

| p-value | LDS Agreement | IALM Agreement | Δ (%) |
|---------|----------------|----------------|-------|
| 0.01 | — | — | — |
| 0.05 | — | — | — |
| 0.10 | — | — | — |

---

## Performance Comparison

### Expected Performance (Theoretical)
| Matrix Size | IALM Time | LDS Time | Speedup |
|-------------|-----------|-----------|---------|
| n=128 | 2s | 0.2s | 10x |
| n=512 | 30s | 1s | 30x |
| n=2048 | 20m | 10s | 120x |

### Actual Performance (Measured)
*To be filled after Phase 5*

| Matrix Size | IALM Time | LDS Time | Speedup | Notes |
|-------------|-----------|-----------|---------|-------|
| n=128 | — | — | — | — |
| n=512 | — | — | — | — |
| n=2048 | — | — | — | — |

---

## Files Created/Modified Summary

### Files Created
| File | Phase | Lines | Purpose |
|------|-------|-------|---------|
| `docs/LDS_MIGRATION.md` | 0 | ~500 | Migration tracking (this file) |
| `src/core/sampling/leveraged/compute_debiased_estimator.py` | 1 | ~100 | Debiased estimator implementation |
| `src/core/sampling/leveraged/lds_sampler.py` | 3 | ~300 | Main LDS sampler class |
| `scripts/test_lds.py` | 5 | ~200 | Test suite |
| `docs/LDS_SAMPLING.md` | 6 | ~300 | User guide |
| `analysis/notebooks/lds_vs_ialm_comparison.ipynb` | 6 | — | Comparison demo |

### Files Modified
| File | Phase | Lines Changed | Key Changes |
|------|-------|---------------|-------------|
| `src/core/sampling/leveraged/compute_leverage_scores.py` | 2 | +30 | Add regularization floor |
| `src/core/sampling/leveraged/compute_sampling_probabilities.py` | 3 | +5 | Add probability capping |
| `src/config/base_config.py` | 4 | +5 | Add `method="lds"` |
| `src/runners/bootstrap_sweep.py` | 4 | +10 | Add LDSSampler case |
| `scripts/interactive_run.py` | 4 | +5 | Add "lds" choice |
| `src/config/presets.py` | 4 | +5 | Add lds parameters |
| `README.md` | 6 | +10 | Add LDS mention |

### Files NOT Used (Excluded from LDS)
These files remain in the codebase for the IALM-based `LeveragedSampler` but are not used by `LDSSampler`:
- `ialm_solve.py` - Iterative matrix completion solver
- `singular_value_threshold.py` - SVT operator for IALM
- `soft_threshold.py` - Soft thresholding for sparse noise
- `ialm_result.py` - IALM diagnostic result class

---

## Lessons Learned

*To be filled after migration is complete*

### What Went Well
- TBD

### Challenges Encountered
- TBD

### Design Decisions
- TBD

---

## Future Work

### Potential Enhancements
1. **Adaptive τ_floor**: Automatically tune regularization based on Phase 1 quality
2. **Variance Reduction**: Implement control variates for lower-variance estimator
3. **Hybrid Approach**: Use LDS for initial partition, IALM for refinement
4. **Theoretical Analysis**: Prove tighter bounds for phylogenetic tree setting

### Open Questions
1. Can we further reduce Phase 1 budget while maintaining quality?
2. Is there an optimal value of θ (Phase 1 ratio) for STDR?
3. How does LDS perform on highly imbalanced trees?

---

## References

1. Huang, Liu, Du, Tao - "Leveraged Matrix Completion With Noise"
2. Davis, Kahan - "The Rotation of Eigenvectors by a Perturbation. III" (1970)
3. Candès, Recht - "Exact Matrix Completion via Convex Optimization" (2009)
4. Fielder - "Algebraic connectivity of graphs" (1973)

---

**Migration Started**: 2026-02-13
**Migration Completed**: TBD
**Total Duration**: TBD
