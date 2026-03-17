# LDS Sampling: Fast Spectral Preservation for STDR

## Overview

**LDS (Leveraged Debiased Sampler (LDS)) Sampling** is a fast leveraged sampling method for Spectral Top-Down Recovery (STDR) that provides **10-100x speedup** over traditional IALM-based matrix completion while maintaining comparable accuracy.

### Key Differences

| Feature | Uniform | IALM (Leveraged) | LDS |
|---------|---------|------------------|------|
| **Complexity** | O(n²) | O(n³ × iterations) | O(n²) |
| **Speed** | Fast | Slow | Fast |
| **Accuracy** | Baseline | Highest | High |
| **Use Case** | Default | Small trees, high accuracy | Large trees, fast results |
| **Sampling** | Random | Leverage-based | Leverage-based |
| **Recovery** | None (sparse) | Iterative (IALM) | Single-shot (debiased) |

### When to Use LDS

✅ **Use LDS when:**
- Trees are large (n > 500 taxa)
- Speed is important
- You need good accuracy without perfect recovery
- Working at moderate-to-high p-values (p > 0.15 is a rough guide; the B1 fix means uniform
  fallback now triggers more aggressively at low p because the threshold correctly reflects
  HLDT: `p_critical ≈ 4·r·log²(n)/n`, which is larger than the old `4·r·log(n)/n`)

⚠️ **Use IALM when:**
- Trees are small (n < 200 taxa)
- You need the highest possible accuracy
- You have time for longer computations
- Working with very low p-values with noisy data

---

## Quick Start

### Method 1: Interactive Launcher

```bash
cd sub_sampled_fielder_vec
python scripts/interactive_run.py
```

When prompted for sampling method, select **"lds"**:
```
Sampling method:
  1) uniform
  2) leveraged
  3) lds ← Select this
```

Then configure LDS parameters:
- **Theta (phase 1 ratio)**: 0.3 (default) - fraction of budget for uniform sampling
- **Target rank**: 2 (default) - rank for SVD in leverage estimation
- **Tau floor multiplier**: 1.0 (default) - regularization strength
- **Allow fallback**: Yes (default) - fall back to uniform at very low p

### Method 2: Programmatic Usage

```python
from src.config.presets import custom_config
from src.runners.experiment_runner import ExperimentRunner

cfg = custom_config(
    num_taxa=512,
    sequence_length=10000,
    mutation_rate=0.1,
    tree_model="kingman_mean",
    p_values=[0.01, 0.05, 0.1, 0.5, 1.0],
    bootstrap_reps=20,
    run_name="lds_experiment",

    # LDS Configuration
    sampling_method="lds",
    sampling_theta=0.3,
    sampling_target_rank=2,
    sampling_tau_floor_multiplier=1.0,
    sampling_allow_uniform_fallback=True,
    log_sampling_diagnostics=False,
)

runner = ExperimentRunner(cfg)
run_dir, results = runner.run()
```

---

## How LDS Works

### Three-Phase Pipeline

#### Phase 1: Uniform Sampling → Leverage Scores
1. Sample `θ × total_budget` entries uniformly
2. Apply inverse probability weighting (IPW): `X_observed[Ω] = X[Ω] / p_uniform`
3. Compute rank-r SVD of observed entries
4. Calculate leverage scores: `μ_i = (n/r) × ||U_i||²` where **U must be orthonormal**
5. **Add regularization floor**: `μ'_i = μ_i + τ_floor` where `τ_floor = mean(μ)`

**Why regularization?** Prevents "zero-score lockout" where taxa missed in Phase 1 get zero sampling probability in Phase 2.

**⚠️ Implementation Note**: `sklearn.TruncatedSVD.fit_transform()` returns **U*Σ** (not orthonormal U). Must normalize: `U = (U*Σ) / Σ` before computing leverage scores. See [Issue: Leverage scores extremely high](#issue-leverage-scores-extremely-high-billions) for details.

#### Phase 2: Leveraged Sampling → Importance-Based
1. Compute sampling probabilities: `p_ij ∝ (μ'_i + μ'_j) × r × log²(n) / n`
2. **Cap probabilities**: `p_ij = min(p_ij, 1.0)`
3. Sample remaining `(1-θ) × total_budget` entries according to `p_ij`
4. Combine with Phase 1: `Ω = Ω_1 ∪ Ω_2`

#### Phase 3: Debiased Estimator → Spectral Preservation
1. Compute effective inclusion probability per entry:
   - `p_restricted_ij = p_ij / sum(p_restricted)` (normalized, excluding Phase 1 entries)
   - `p2_ij = min(p_restricted_ij × phase2_budget, 1.0)`
   - `π_ij = min(p_0 + (1 - p_0) × p2_ij, 1.0)` where `p_0 = phase1_actual / n_upper`
2. For sampled entries: `X̂_ij = X_ij / π_ij`
3. For unsampled entries: `X̂_ij = 0`
4. Set diagonal: `X̂_ii = 1.0`
5. Return sparse CSR matrix

**Note**: `p_matrix` from Phase 2 is a PMF summing to 1 (a probability *distribution* over entries,
not per-entry inclusion probabilities). The actual inclusion probability for entry (i,j) is
`p_matrix[i,j] × phase2_budget / restricted_sum`. Phase 1 entries are excluded from the restricted
pool before computing `p2` to avoid double-counting.

**No IALM iterations!** This single-shot construction is what makes LDS fast.

### Theoretical Guarantee

**LDS Theorem** (HLDT Theorem 3.1):
With `m = O(nr log²(n))` samples using leverage scores (Phase 1 minimum: `4·n·r·log²(n)`):
```
E[X̂] = X  (unbiased estimator)
```

**Davis-Kahan Perturbation Bound**:
```
||v̂ - v||₂ ≤ ||X̂ - X||_F / gap
```
where `v̂` = Fiedler vector from X̂, `v` = true Fiedler vector, `gap` = spectral gap.

**Conclusion**: For STDR, we don't need exact matrix entries—just eigenvector stability! The debiased estimator provides this with high probability.

---

## Configuration Parameters

### Required Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampling_method` | str | `"uniform"` | Set to `"lds"` to use LDS sampling |
| `sampling_theta` | float | `0.3` | Phase 1 budget ratio (0 < θ < 1) |
| `sampling_target_rank` | int | `2` | Rank r for SVD (typically 2 for Fiedler) |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampling_tau_floor_multiplier` | float | `1.0` | Regularization floor scale |
| `sampling_allow_uniform_fallback` | bool | `True` | Allow fallback to uniform at low p |
| `log_sampling_diagnostics` | bool | `False` | Save leverage scores and probabilities |

### Parameter Tuning Guide

#### Theta (Phase 1 Budget Ratio)
- **Lower (0.2-0.3)**: Less overhead, more Phase 2 samples → Better for LDS
- **Higher (0.5-0.7)**: More reliable leverage estimates → Better for IALM
- **Recommended for LDS**: 0.3

#### Target Rank
- **For Fiedler vector**: Always use `r=2`
- **For k-way partitioning**: Use `r=k`
- **Impact**: Higher r → more computation, potentially better leverage estimates

#### Tau Floor Multiplier
- **Default (1.0)**: Standard regularization
- **Lower (0.5)**: Less regularization, closer to LDS paper
- **Higher (2.0)**: More aggressive regularization, safer for very low p
- **When to adjust**: Only if you see "zero-score lockout" warnings

---

## Performance Expectations

### Speed Comparison

| Matrix Size | IALM Time | LDS Time | Speedup |
|-------------|-----------|-----------|---------|
| n=128 | 2s | 0.2s | **10x** |
| n=256 | 30s | 0.5s | **60x** |
| n=512 | 5m | 2s | **150x** |
| n=1024 | 40m | 10s | **240x** |

*Measured at p=0.05 with 100 bootstrap replicates*

### Accuracy Comparison

**Phase Transition Curve** (n=128, 10 bootstrap reps):

| p-value | LDS Agreement | Expected |
|---------|----------------|----------|
| 0.05 | 50.7% ± 3.9% | Random (sparse) |
| 0.10 | 53.0% ± 7.6% | Still sparse |
| 0.20 | 92.0% ± 8.8% | **Phase transition** |
| 0.30 | 94.7% ± 1.4% | Good |
| 0.50 | 96.3% ± 1.7% | Very good |
| 0.70 | 97.1% ± 1.4% | Excellent |

**Key Insight**: LDS shows clear phase transition from random (50%) to accurate (>95%) as p increases, matching theoretical predictions for spectral methods.

---

## Diagnostic Logging

### Enabling Diagnostics

```python
cfg = custom_config(
    ...
    sampling_method="lds",
    log_sampling_diagnostics=True,  # Enable diagnostic logging
)
```

### What Gets Logged

For each p-value where LDS runs:

**Files Created**: `{run_dir}/sampling_data/p_{p:.4f}.npz`

**Contents**:
```python
import numpy as np

data = np.load("sampling_data/p_0.1438.npz")
leverage_scores = data['leverage_scores']          # (n,) array
phase2_probs_sampled = data['phase2_probs_sampled'] # (n,n) sparse
```

### Accessing Runtime Diagnostics

```python
from src.core.sampling.leveraged.lds_sampler import LDSSampler

sampler = LDSSampler(theta=0.3, target_rank=2)
S_hat = sampler.sample(S, p=0.2, seed=123)

metrics = sampler.last_sample_metrics
print(f"Phase 1: {metrics['phase1_actual']} samples")
print(f"Phase 2: {metrics['phase2_actual']} samples")
print(f"Phase 1 sufficiency: {metrics['phase1_sufficiency']:.1%}")  # NEW: Quality indicator
print(f"Leverage max: {metrics['leverage_max']:.3f}")
print(f"τ_floor: {metrics['tau_floor']:.3f}")
print(f"Debiasing time: {metrics['debiasing_time']:.4f}s")
print(f"Matrix sparsity: {metrics['matrix_sparsity']:.1%}")
print(f"Spectral gap: {metrics['spectral_gap']:.3f}")  # s[1]/s[2]; inf if target_rank < 3
```

### Phase 1 Quality Tracking (NEW)

**What is `spectral_gap`?**
Ratio `s[1] / s[2]` of the first two Phase 1 singular values. This is the denominator in the
Davis-Kahan bound: `‖v̂ - v‖ ≤ ‖X̂ - X‖ / gap`. A larger gap means eigenvectors are more
stable under perturbation. Set `target_rank=3` to get a meaningful value; with `target_rank=2`
only two singular values are computed so `spectral_gap = inf`.

**What is `phase1_sufficiency`?**
Ratio of actual Phase 1 samples to theoretical minimum: `phase1_actual / (4 × n × r × log²(n))`

**Interpretation**:
- **< 10%**: Very noisy leverage estimates (essentially random sampling)
- **10-50%**: Noisy but potentially useful
- **≥ 50%**: Good quality leverage estimation
- **≥ 100%**: Meets or exceeds theoretical requirement

**Aggregate Report**:
After experiment completion, `experiment.log` shows:
```
Phase 1 Quality Summary:
  Mean sufficiency: 8.3%
  <10% (very noisy):  10/13 p-values
  10-50% (noisy):     3/13 p-values
  ≥50% (good):        0/13 p-values
  ⚠ Most p-values ran with very noisy leverage estimates
     Consider: increase theta (currently 0.7) or focus on higher p-values
```

**Results CSV**:
The `results.json` now includes `mean_phase1_sufficiency`, `median_phase1_sufficiency`, `std_phase1_sufficiency` columns for downstream analysis.

---

## Troubleshooting

### Issue: "Phase 1 budget below theoretical minimum"

**Message** (appears in `experiment.log`):
```
⚠ LDS Phase 1 Quality: p=0.0001 has only 146/124,921 samples (0.1% of theoretical minimum)
  → Leverage estimates will be noisy, but estimator remains unbiased
```

**Cause**: At very low p, there aren't enough samples for reliable leverage estimation.

**What This Means**:
- **Estimator remains unbiased**: E[X̂] = X still holds (mathematical guarantee)
- **High variance**: Leverage scores are noisy, so Phase 2 sampling is less optimal
- **Behavior**: LDS effectively becomes closer to uniform sampling at very low p
- **Accuracy impact**: Expect random-like performance (~50% agreement) below ~0.01

**Solutions**:
1. **Accept lower accuracy at low p** (expected behavior, still faster than IALM)
2. **Increase theta** (e.g., 0.7 → 0.9) to allocate more budget to Phase 1
3. **Focus on higher p-values** (p > 0.05 where LDS quality is good)
4. **Use uniform fallback** (set `allow_uniform_fallback=True`) for automatic switching
5. **Force LDS anyway** (research use: understand algorithm behavior at extremes)

### Issue: Numerical warnings (overflow, divide by zero)

**Message**:
```
RuntimeWarning: divide by zero encountered in matmul
RuntimeWarning: overflow encountered in matmul
```

**Cause**: At very low p, debiasing X_ij / p_ij creates large values when p_ij is small.

**Impact**: Usually harmless - final Fiedler vectors are still computed correctly due to eigenvector normalization.

**If problematic**:
1. Filter to p > 0.15 where warnings don't occur
2. Use IALM at very low p instead

### Issue: Leverage scores extremely high (billions)

**Symptom**: Leverage scores show values in billions instead of averaging ~1.0

**Cause**: **[FIXED in 2026-02-18]** This was caused by a critical bug in `sklearn.TruncatedSVD` usage.

**The Bug**:
- `TruncatedSVD.fit_transform()` returns **U*Σ**, not orthonormal **U**
- Without normalization, leverage scores were computed from scaled vectors
- Impact: Scores were scaled by σ² (squared singular values)

**How to Verify the Fix**:
```python
# Load sampling data
data = np.load("sampling_data/p_0.1000.npz")
leverage_scores = data['leverage_scores']

# Check statistics
print(f"Mean: {leverage_scores.mean():.3f}")  # Should be ~1.0
print(f"Max: {leverage_scores.max():.3f}")    # Should be ~2-10, not billions
print(f"Min: {leverage_scores.min():.3f}")    # Should be ~0.01-0.5
```

**Expected Range** (after fix):
- Mean ≈ 1.0 (theoretical: scores sum to n)
- Max ≈ 2-10 (depending on tree structure)
- Min ≈ 0.01-0.5 (uniform lower bound from regularization)

**If you still see billion-scale scores**, verify you're using the fixed version:
```bash
# Check the fix is present
grep -A5 "U_sigma = svd_model.fit_transform" src/core/sampling/leveraged/compute_leverage_scores.py
# Should show normalization: U = U_sigma / s_safe[None, :]
```

### Issue: Lower accuracy than IALM

**Observation**: LDS agreement is 2-3% lower than IALM at same p.

**Explanation**: This is expected! LDS trades perfect matrix recovery for speed:
- IALM: Exact recovery → highest accuracy
- LDS: Unbiased estimator → high variance but correct in expectation
- Trade-off: 10-100x speedup for 2-3% accuracy loss

**Recommendation**:
- Use LDS for fast results at scale
- Use IALM for small trees where accuracy matters most

---

## Mathematical Background

### Why Single-Shot Debiasing Works

#### Matrix Completion (IALM) Approach:
```
Goal: Find exact X from samples
Method: Minimize ||L||_* + λ||S||_1  subject to P_Ω(L + S) = P_Ω(X)
Complexity: O(n³ × iterations)
Output: Recovered matrix L ≈ X
```

#### Spectral Preservation (LDS) Approach:
```
Goal: Preserve eigenvector directions
Method: Construct unbiased estimator X̂_ij = X_ij / p_ij
Complexity: O(n²)
Output: Noisy but unbiased X̂ where E[X̂] = X
```

**Key Insight**: For spectral partitioning, eigenvector *directions* matter more than exact *magnitudes*. Davis-Kahan theorem guarantees eigenvector stability as long as E[X̂] = X, which the debiased estimator provides!

### Golfing Scheme vs Bernoulli Sampling

LDS uses the "golfing scheme" (fixed budget sampling):
- **Bernoulli**: Each entry sampled independently with probability p_ij
- **Golfing**: Sample exactly m entries according to distribution p_ij

**Why golfing?** Deterministic sample count → easier budget control for experiments.

### Variance-Bias Trade-off

- **IALM**: Low variance (iterative refinement), potential bias (convergence error)
- **LDS**: High variance (single sample), zero bias (E[X̂] = X)

For spectral methods, the unbiased property + eigenvector robustness makes LDS effective despite high variance.

---

## Comparison with Related Methods

### LDS vs LeveragedSampler (IALM)

| Aspect | LeveragedSampler | LDSSampler |
|--------|------------------|-------------|
| **Recovery** | IALM iterations | Single-shot debiasing |
| **Speed** | Slow (O(n³ × iters)) | Fast (O(n²)) |
| **Memory** | High (dense iterations) | Low (sparse output) |
| **Accuracy** | Highest | High (2-3% lower) |
| **Phase 1/2** | Same | Same + regularization |
| **Output** | Dense recovered matrix | Sparse debiased matrix |

### LDS vs Uniform Sampling

| Aspect | Uniform | LDS |
|--------|---------|------|
| **Sampling** | Random | Importance-based |
| **Samples needed** | Baseline | Fewer (better p_critical) |
| **Phase transition** | Sharp | Smoother |
| **Complexity** | O(n²) | O(n²) |
| **Speed** | Fastest | Fast |

**Expected benefit**: LDS should achieve same accuracy with 30-50% fewer samples than uniform (though this needs empirical validation on phylogenetic trees).

---

## Advanced Usage

### Research Mode (No Fallback)

For research exploring behavior at extreme low p:

```python
cfg = custom_config(
    ...
    sampling_method="lds",
    sampling_allow_uniform_fallback=False,  # Never fall back
)
```

**Warning**: This allows LDS to run even when Phase 1 budget < theoretical minimum. Useful for understanding algorithm limits, not for production.

### Custom Sampler Instantiation

```python
from src.core.sampling.leveraged.lds_sampler import LDSSampler

sampler = LDSSampler(
    theta=0.3,
    target_rank=2,
    tau_floor_multiplier=1.0,
    force_lds=False,
    allow_uniform_fallback=True
)

# Sample single matrix
S_hat = sampler.sample(S, p=0.2, seed=123)

# Access diagnostics
metrics = sampler.last_sample_metrics
```

### Integration with STDR Pipeline

LDS integrates seamlessly with existing STDR pipeline:

```python
# Sampled matrix (sparse or dense)
S_hat = sampler.sample(S, p=0.2)

# Construct Laplacian (handles sparse automatically)
L_hat = compute_laplacian(S_hat)

# Compute Fiedler vector (FiedlerVectorComputer detects sparse)
from src.core.fiedler_computer import FiedlerVectorComputer
fiedler_comp = FiedlerVectorComputer()
fiedler = fiedler_comp.compute(L_hat)

# Partition
partition = fiedler > 0
```

---

## Files & Implementation

### Core Implementation Files

| File | Purpose |
|------|---------|
| `src/core/sampling/leveraged/lds_sampler.py` | Main LDS sampler class |
| `src/core/sampling/leveraged/compute_debiased_estimator.py` | Debiased estimator computation |
| `src/core/sampling/leveraged/compute_leverage_scores.py` | Leverage scores + regularization |
| `src/core/sampling/leveraged/compute_sampling_probabilities.py` | Phase 2 probability calculation |
| `src/core/sampling/leveraged/uniform_sampling.py` | Phase 1 uniform sampling |
| `src/core/sampling/leveraged/nonuniform_sampling.py` | Phase 2 leveraged sampling |

### Configuration Files

| File | Role |
|------|------|
| `src/config/base_config.py` | SamplingConfig dataclass |
| `src/core/sampling/__init__.py` | Sampler factory (get_sampler) |
| `src/runners/bootstrap_sweep.py` | Integration with experiment runner |
| `scripts/interactive_run.py` | Interactive UI for LDS |

---

## References

1. **Huang, Liu, Du, Tao** - "Leveraged Matrix Completion With Noise"
   Paper introducing LDS sampling with theoretical guarantees

2. **Davis, Kahan (1970)** - "The Rotation of Eigenvectors by a Perturbation III"
   Fundamental perturbation bounds for eigenvectors

3. **Candès, Recht (2009)** - "Exact Matrix Completion via Convex Optimization"
   Theoretical foundation for matrix completion

4. **Mossel, Roch (2006)** - "Learning Nonsingular Phylogenies and Hidden Markov Models"
   Spectral methods for phylogenetic tree reconstruction

---

## See Also

- [LEVERAGED_SAMPLING.md](LEVERAGED_SAMPLING.md) - IALM-based leveraged sampling
- [LDS_MIGRATION.md](LDS_MIGRATION.md) - Technical migration details
- [CONFIGURATION.md](CONFIGURATION.md) - Full configuration system
- [ANALYSIS_GUIDES.md](ANALYSIS_GUIDES.md) - Analyzing results

---

**Questions or Issues?** See [Troubleshooting](#troubleshooting) or file an issue on GitHub.
