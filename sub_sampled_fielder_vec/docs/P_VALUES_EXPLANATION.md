# Understanding `p_values` in Uniform vs Leveraged Sampling

## Key Difference

The interpretation of `p` (sampling probability/budget) differs between the two methods:

### **Uniform Sampling** (`method="uniform"`)

- **`p` = Probability per entry**
- Each off-diagonal entry `(i,j)` is sampled independently with probability `p`
- **Stochastic**: The actual number of samples varies between runs
- **Expected samples**: `p × n_upper` where `n_upper = n(n-1)/2` (upper triangle entries)
- **Example**: For `n=1000`, `p=0.1`:
  - Expected samples: `0.1 × 499,500 ≈ 49,950`
  - Actual samples: Random (could be 49,800 or 50,100)

### **Leveraged Sampling** (`method="leveraged"`)

- **`p` = Total budget fraction**
- Determines the **exact total number** of samples to collect
- **Deterministic**: Same number of samples every run (for same `p` and `seed`)
- **Exact samples**: `p × n_upper` (rounded to integer)
- **Example**: For `n=1000`, `p=0.1`:
  - Exact samples: `⌊0.1 × 499,500⌋ = 49,950` (always)
  - These are split into:
    - Phase 1 (uniform): `θ × 49,950 = 14,985` samples
    - Phase 2 (leveraged): `(1-θ) × 49,950 = 34,965` samples

## Detailed Breakdown for Leveraged Method

When you set `p=0.1` with `theta=0.3`:

```
Total budget = p × n_upper = 0.1 × n(n-1)/2

Phase 1 (Uniform):
  Budget = θ × total_budget = 0.3 × total_budget
  Purpose: Estimate leverage scores
  Method: Uniform random sampling

Phase 2 (Leveraged):
  Budget = (1-θ) × total_budget = 0.7 × total_budget  
  Purpose: Sample important entries
  Method: Non-uniform sampling based on leverage scores

Phase 3 (Recovery):
  Input: All samples from Phase 1 + Phase 2
  Purpose: Recover clean low-rank matrix using IALM
  Output: Recovered matrix L
```

## Practical Implications

### For Uniform Method:
```python
p_values = [0.01, 0.05, 0.1, 0.5, 1.0]
# Each p is a probability - actual samples will vary slightly
```

### For Leveraged Method:
```python
p_values = [0.01, 0.05, 0.1, 0.5, 1.0]
# Each p is a budget fraction - exact number of samples
# p=0.01 means "sample 1% of available entries"
# p=0.1 means "sample 10% of available entries"
```

## Why This Matters

1. **Comparability**: When comparing uniform vs leveraged at the same `p`:
   - Uniform: Stochastic number of samples (expected = `p × n_upper`)
   - Leveraged: Exact number of samples (`p × n_upper`)
   - They're approximately comparable, but leveraged is more deterministic

2. **Budget Control**: Leveraged method gives you precise control over total samples, which is important for:
   - Fair comparisons across methods
   - Reproducible experiments
   - Resource planning

3. **Phase Allocation**: In leveraged method, the total budget is split:
   - `θ × p × n_upper` for Phase 1 (exploration)
   - `(1-θ) × p × n_upper` for Phase 2 (exploitation)

## Example Calculation

For a matrix with `n=1000` taxa:

```python
n_upper = 1000 × 999 / 2 = 499,500  # Upper triangle entries

# Uniform method with p=0.1:
expected_samples = 0.1 × 499,500 = 49,950
# Actual samples: ~49,950 (varies by ~100-200)

# Leveraged method with p=0.1, theta=0.3:
total_budget = 0.1 × 499,500 = 49,950  # Exact
phase1_samples = 0.3 × 49,950 = 14,985  # Exact
phase2_samples = 0.7 × 49,950 = 34,965  # Exact
```

## Recommendation

When comparing methods:
- Use the same `p_values` list for both methods
- Understand that uniform is stochastic while leveraged is deterministic
- For fair comparison, you might want to run uniform multiple times and average
- Leveraged method's deterministic nature makes it easier to compare across runs

