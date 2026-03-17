# Metrics and Data Formats

This document describes all metrics computed by the framework and the data formats used for storing results.

## Overview

The framework computes **4 primary metrics** to evaluate Fiedler vector quality under subsampling:

1. **Partition Agreement Metrics** (2 metrics)
   - `partition_agreement_M` - Ideal scenario comparison
   - `partition_agreement_S` - Realistic scenario comparison

2. **Vector Alignment Metrics** (2 metrics)
   - `dot_product` - Continuous vector similarity
   - `sign_agreement` - Legacy sign-based metric

## Partition Agreement Metrics

**Why partition-based?** Simple sign agreement (threshold at zero) doesn't reflect how STDR actually partitions data. STDR uses sophisticated gap-based thresholding that searches for optimal partition points, not just sign changes.

### partition_agreement_M (Ideal Scenario)

- **Question**: "If we had full data, would averaged Fiedler give same partition as full Fiedler?"
- **Method**: Computes 2 partitions using M (full similarity) for both:
  - `partition_taxa(fiedler_full, M)` [reference]
  - `partition_taxa(fiedler_avg, M)` [test]
- **Range**: 0-100%
- **Interpretation**: Tests vector quality in isolation (ignores subsampling effects on matrix)

### partition_agreement_S (Realistic Scenario)

- **Question**: "With subsampled data, how close do we get to ideal partition?"
- **Method**: Computes 2 partitions using different matrices:
  - `partition_taxa(fiedler_full, M)` [reference uses full data]
  - `partition_taxa(fiedler_avg, S_avg)` [test uses averaged subsampled data]
- **Range**: 0-100%
- **Interpretation**: Real-world performance measure (accounts for both vector quality and matrix subsampling)

### How Partitions Work

For each p-value, we compute **3 partitions total**:

```python
# 1. Reference partition (baseline)
partition_ref = partition_taxa(fiedler_full, M, num_gaps=1, min_split=1)

# 2. Test partition vs M (ideal scenario)
partition_avg_vs_M = partition_taxa(fiedler_avg, M, num_gaps=1, min_split=1)

# 3. Test partition vs S_avg (realistic scenario)
partition_avg_vs_S = partition_taxa(fiedler_avg, S_avg, num_gaps=1, min_split=1)
```

Then we compare:
- **partition_agreement_M**: partition_ref vs partition_avg_vs_M
- **partition_agreement_S**: partition_ref vs partition_avg_vs_S

**Note**: Partitions A|B and B|A are treated as equivalent (only orientation differs).

The partition computation uses STDR's `partition_taxa` function from the `spectraltree` library, which:
1. Sorts Fiedler vector entries
2. Searches `num_gaps` largest gaps for optimal partition point
3. Scores each candidate using σ₂(S_AB) (2nd singular value of cross-partition submatrix)
4. Chooses partition with minimum σ₂ (best separation)

## Vector Alignment Metrics

### dot_product

- **Method**: Absolute dot product between normalized Fiedler vectors
- **Formula**: `|v_full · v_avg| / (||v_full|| ||v_avg||)`
- **Range**: 0-1 (higher = better alignment)
- **Use**: Fast continuous measure of vector similarity
- **Computation**: `src/utils/metrics.py::compute_fiedler_dot_product()`

### sign_agreement (Legacy)

- **Method**: Percentage of entries with matching signs at threshold=0
- **Range**: 0-100%
- **Use**: Backward compatibility and simple baseline
- **Note**: Less informative than partition metrics for STDR applications
- **Computation**: `src/utils/metrics.py::compute_sign_agreement()`

## Matrix Metrics

Additional matrix-level metrics are computed for diagnostics. These are aggregated across bootstrap replicates as (mean, median, std).

### Operator Norm Error
- **Formula**: `||M - S_avg||_2` (spectral norm)
- **Interpretation**: Overall matrix reconstruction error

### Empirical Rank
- **Formula**: `||M||_F^2 / ||M||_2^2`
- **Interpretation**: Effective dimensionality of the matrix

### Spectral Gap
- **Formula**: `|λ₃ - λ₂|` (absolute gap between 2nd and 3rd eigenvalues)
- **Interpretation**: Separation between Fiedler eigenvalue and next eigenvalue

### Relative Spectral Gap
- **Formula**: `|λ₃ - λ₂| / λ₂`
- **Interpretation**: Relative separation (normalized by Fiedler eigenvalue)

### Coherence
- **Formula**: `max_i ||u_i||_∞^2` where u_i are top-k singular vectors
- **Interpretation**: Matrix incoherence (lower is better for random sampling)

### Minimum Separation
- **Formula**: Minimum distance between partition clusters
- **Interpretation**: Quality of partition separation

**Computation**: `src/core/metric_computer.py::MetricComputer` class

## Output Data Format

### JSON Results Format

Results are saved as JSON tables with the following structure:

#### Single Experiment

```json
{
  "columns": [
    "p",
    "sign_agreement",
    "partition_agreement_M",
    "partition_agreement_S",
    "dot_product",
    "mean_operator_norm_error",
    "median_operator_norm_error",
    "std_operator_norm_error",
    "mean_empirical_rank",
    "mean_spectral_gap",
    "mean_coherence",
    "..."
  ],
  "rows": [
    {
      "p": 1.0,
      "sign_agreement": 100.0,
      "partition_agreement_M": 100.0,
      "partition_agreement_S": 100.0,
      "dot_product": 1.0,
      "mean_operator_norm_error": 0.0,
      "median_operator_norm_error": 0.0,
      "std_operator_norm_error": 0.0,
      "..."
    },
    {
      "p": 0.1,
      "sign_agreement": 95.2,
      "partition_agreement_M": 96.8,
      "partition_agreement_S": 94.1,
      "dot_product": 0.987,
      "mean_operator_norm_error": 0.234,
      "..."
    }
  ]
}
```

#### Taxa Sweep Results

```json
{
  "columns": [
    "num_taxa",
    "sequence_length",
    "p_values",
    "sign_agreements",
    "partition_agreement_M",
    "partition_agreement_S",
    "dot_products",
    "..."
  ],
  "rows": [
    {
      "num_taxa": 1024,
      "sequence_length": 1000,
      "p_values": [0.01, 0.1, 0.5, 1.0],
      "sign_agreements": [85.2, 95.1, 98.5, 100.0],
      "partition_agreement_M": [87.3, 96.2, 99.1, 100.0],
      "partition_agreement_S": [84.1, 94.8, 98.2, 100.0],
      "dot_products": [0.912, 0.978, 0.995, 1.0],
      "..."
    }
  ]
}
```

**Matrix Metrics**: Each matrix metric includes three columns:
- `mean_{metric}` - Mean across bootstrap replicates
- `median_{metric}` - Median across bootstrap replicates
- `std_{metric}` - Standard deviation across bootstrap replicates

### NumPy Arrays

Additional raw data is saved as NumPy arrays:

- `sign_agreements.npy` - Raw agreement arrays (shape: `[n_p_values, n_bootstrap_reps]`)
- `fiedler_ref*.npy` - Reference Fiedler vectors (one per `(n_taxa, seq_len)` combination)

## Expected Outcomes

### Sanity Checks

- **p=1.0**: All metrics should be perfect
  - `partition_agreement_M = 100%`
  - `partition_agreement_S = 100%`
  - `dot_product = 1.0`
  - `sign_agreement = 100%`

### General Trends

- **High p (> 0.5)**:
  - `partition_agreement_M` ≈ `sign_agreement` ± 1%
  - `partition_agreement_S` ≈ `partition_agreement_M` ± 2%

- **Medium p (0.1-0.5)**:
  - Metrics may diverge by 2-5%
  - Gap-based thresholding effects become visible

- **Low p (< 0.1)**:
  - Significant divergence possible (5-15%)
  - `partition_agreement_S` ≤ `partition_agreement_M` (realistic is harder)

## Implementation Details

### Metric Computation Functions

**Location**: `src/utils/metrics.py`

- `compute_partition_agreement(v_full, v_avg, sim_full, sim_avg, num_gaps, min_split)`
  - Unified function for both partition_agreement_M and partition_agreement_S
  - Computes 2 partitions using `partition_taxa` from spectraltree
  - Compares partitions (handles A|B ≡ B|A equivalence)
  - Returns percentage agreement (0-100)

- `compute_fiedler_dot_product(v_full, v_avg)`
  - Computes absolute dot product between normalized vectors
  - Returns 0-1 (higher = better alignment)

- `compute_sign_agreement(v1, v2)`
  - Legacy metric (backward compatibility)
  - Percentage of entries with matching signs

- `metric_composer(M, S, L_M, L_S, p)`
  - Computes all matrix metrics efficiently
  - Delegates to `MetricComputer` class
  - Returns aggregated statistics (mean, median, std)

### Error Handling

- `partition_taxa` can raise exceptions if partitions violate `min_split`
- Exceptions caught at call site, logged as warnings
- Failed metrics stored as NaN in results

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Codebase structure and entry points
- [CONFIGURATION.md](CONFIGURATION.md) - How to configure metrics computation
