# Leveraged Sampling Diagnostic Framework

**Purpose:** Analyze leveraged matrix completion sampling performance and identify potential issues.

## Quick Start

### Option 1: Use the Diagnostic Notebook

```bash
cd sub_sampled_fielder_vec/analysis/leveraged_sampling_analysis
jupyter notebook leveraged_diagnostics.ipynb
```

**Edit Cell 1** to point to your leveraged sampling results directory (e.g., `20260117-155215-balanced_binary_leveraged`), then run all cells.

### Option 2: Use Diagnostic Functions Directly

```python
from analysis.leveraged_sampling_analysis.diagnostics import (
    compute_leverage_concentration,
    compute_effective_rank,
    compute_phase1_sample_fraction,
    theoretical_p_star,
    diagnose_phase1_quality,
)

# Example: Check leverage concentration
leverage_max = 50.0
leverage_mean = 1.0
concentration = compute_leverage_concentration(leverage_max, leverage_mean)
print(f"Concentration: {concentration:.2f}")  # Should be >> 1 for good leveraged sampling
```

## Files

### `diagnostics.py`
**Simple metric functions to answer diagnostic questions:**

| Function | Question Answered |
|----------|-------------------|
| `compute_leverage_concentration()` | Are leverage scores concentrated or uniform? |
| `compute_effective_rank()` | Is Phase 1 giving clean rank-2 structure? |
| `compute_spectral_gap_ratio()` | Is there a clear gap after σ₂? |
| `compute_phase1_sample_fraction()` | How much budget goes to Phase 1 overhead? |
| `theoretical_p_star()` | What does matrix completion theory predict? |
| `diagnose_phase1_quality()` | Human-readable verdict on Phase 1 quality |
| `diagnose_sample_allocation()` | Human-readable verdict on budget split |
| `compare_to_theoretical_bound()` | How far are we from optimal? |

### `diagnostic_plots.py`
**Visualization functions:**

| Function | Visualization |
|----------|---------------|
| `plot_leverage_histogram()` | Distribution of leverage scores |
| `plot_phase1_budget()` | Phase 1 vs Phase 2 sample allocation |
| `plot_vs_theory()` | Actual p* vs theoretical predictions |

### `leveraged_diagnostics.ipynb`
**Interactive diagnostic notebook with 5 Q&A sections:**

1. **Are leverage scores concentrated?** → Expect max/mean > 5
2. **Is Phase 1 rank-2 quality good?** → Expect σ₂/σ₃ > 10
3. **How much budget goes to Phase 1?** → Expect < 20%
4. **How does p* compare to theory?** → Expect ratio < 3x
5. **Where does leveraged help?** → Check agreement curves

## Diagnostic Workflow

### Step 1: Run the Diagnostic Notebook

```bash
jupyter notebook leveraged_diagnostics.ipynb
```

### Step 2: Answer the Questions

For each diagnostic section, check:
- ✓ = **GOOD** (as expected)
- ⚠ = **WARNING** (concerning)
- ❌ = **BAD** (root cause found!)

### Step 3: Identify Root Causes

Common failure modes:

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| Concentration ≈ 1 | Leverage scores are uniform | Matrix is already incoherent, no benefit from leveraged sampling |
| σ₂/σ₃ < 3 | Phase 1 gives poor rank-2 | Increase Phase 1 budget or improve SVD method |
| Phase 1 > 50% | Too much overhead | Increase θ parameter to reduce Phase 1 samples |
| Ratio > 10x theory | Fundamentally inefficient | Re-evaluate approach or matrix properties |
| No crossover in curves | Leveraged never helps | Check IALM recovery or leverage computation |

### Step 4: Implement Fixes

Based on diagnosis, common fixes:

1. **Increase θ** (reduce Phase 1 overhead):
   ```python
   # In config
   "leveraged_theta": 0.7  # Was 0.3
   ```

2. **Better leverage estimation**:
   - Use more Phase 1 samples
   - Use different SVD algorithm
   - Compute leverage on FULL matrix first (diagnostic only)

3. **Adjust IALM parameters**:
   - Increase max iterations
   - Adjust convergence tolerance
   - Use different initialization

## Example Output

From the diagnostic notebook, you should see output like:

```
Leverage Score Concentration Analysis
================================================================================

n=512 (at p=1.0000):
  Max leverage score:  50.0869
  Mean leverage score: 1.0000
  Std leverage score:  3.2451
  → Concentration (max/mean): 50.09
  → CV (std/mean): 3.25
  ✓ VERDICT: Well concentrated (leveraged should help)
```

```
Phase 1 SVD Quality Analysis
================================================================================

n=512 (at p=1.0000):
  σ₁ = 15.2341
  σ₂ = 1.8923
  σ₃ = 0.1234
  → Rank ratio (σ₂/σ₁): 0.1242
  → Gap ratio (σ₂/σ₃): 15.34
  ✓ VERDICT: Clean rank-2 structure (σ₂/σ₁=0.1242, σ₂/σ₃=15.34)
```

```
Sample Allocation Analysis
================================================================================

n=512:
          p    Phase1%        Total       Phase1       Phase2  Status
  ----------  ----------  ------------  ------------  ------------  ----------
    0.000100        80.9%        26,214        21,224         4,990  ❌
    0.010000         1.0%     2,621,440        21,224     2,600,216  ✓
    1.000000         0.0%   262,144,000        21,224   262,122,776  ✓
```

## Interpretation Guide

### Good Leveraged Sampling Should Show:

1. **High concentration** (max/mean >> 1)
2. **Clean Phase 1 rank-2** (σ₂/σ₃ > 10)
3. **Low Phase 1 overhead** (< 20% at critical p*)
4. **Near-optimal p*** (within 3x of theory)
5. **Earlier 95% crossing** than uniform

### If Leveraged is Underperforming:

Check diagnostics in order:
1. **Leverage concentration** → If uniform, no point in leveraged sampling
2. **Phase 1 quality** → If bad, leverage scores are unreliable
3. **Sample allocation** → If Phase 1 dominates, just expensive uniform
4. **Theory comparison** → If far off, fundamental issue
5. **Agreement curves** → Visual confirmation of performance

## References

- Candès & Recht, "Exact Matrix Completion via Convex Optimization" (2009)
- Achlioptas & McSherry, "Fast Computation of Low-Rank Approximations" (2007)
- Drineas & Mahoney, "RandNLA: Randomized Numerical Linear Algebra" (2016)
