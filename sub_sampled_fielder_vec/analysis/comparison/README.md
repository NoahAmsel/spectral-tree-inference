# Method Comparison: Uniform vs Leveraged Sampling

This directory contains scripts to compare uniform and leveraged sampling methods on **identical matrices** (same tree, same sequences).

## Key Feature: Fair Comparison

Both methods run on the SAME data using a shared random seed, ensuring:
- Identical tree topology
- Identical sequence observations
- Identical full similarity matrix

Only the sampling/recovery method differs.

## Files

- **`compare_methods.py`**: Main comparison script with full configuration
- **`test_compare.py`**: Quick test with minimal parameters (n=500, 3 bootstrap reps)
- **`__init__.py`**: Package marker

## Usage

### Quick Test (Recommended First)
```bash
cd sub_sampled_fielder_vec
python analysis/comparison/test_compare.py
```

**Test parameters:**
- `n_taxa`: [500]
- `bootstrap_reps`: 3
- `p_values`: [0.01, 0.1, 1.0]

**Runtime:** ~2-5 minutes

### Full Comparison
```bash
cd sub_sampled_fielder_vec
python analysis/comparison/compare_methods.py
```

**Full parameters:**
- `n_taxa`: [500, 1000, 3000, 5000, 7000, 10000]
- `bootstrap_reps`: 20
- `p_values`: logspace(-4, 0, 20) — 20 values from 0.0001 to 1.0
- Tree model: Kingman coalescent
- Sequence length: 10,000

**Runtime:** Several hours (depends on hardware and parallelization)

## Configuration

Edit `COMPARISON_CONFIG` in `compare_methods.py`:

```python
COMPARISON_CONFIG = {
    "tree_model": "kingman",
    "taxa_values": [500, 1000, 3000, 5000, 7000, 10000],
    "sequence_length": 10000,
    "mutation_rate": 0.1,
    "bootstrap_reps": 20,
    "num_workers": 8,
    "p_values": list(np.logspace(-4, 0, 20)),
    "seed": 42,  # CRITICAL: shared seed ensures identical data

    # Leveraged-specific parameters
    "leveraged_theta": 0.3,           # Phase 1 budget ratio
    "leveraged_target_rank": 2,        # SVD rank
    "leveraged_ialm_max_iter": 100,    # IALM iterations
    "leveraged_ialm_tol": 1e-6,        # IALM tolerance
}
```

## Output Structure

Results are saved in nested directory structure:

```
results/{timestamp}-method_comparison/
├── comparison_config.json          # Config used for comparison
├── comparison_summary.json         # Summary of all runs
├── uniform/                        # Uniform sampling results
│   ├── n500_L10000/
│   │   ├── results.json
│   │   ├── config.json
│   │   ├── fiedler_vectors.png
│   │   └── tree_plots/
│   ├── n1000_L10000/
│   │   └── ...
│   └── ...
└── leveraged/                      # Leveraged sampling results
    ├── n500_L10000/
    │   ├── results.json
    │   ├── config.json
    │   └── ...
    └── ...
```

## Analyzing Results

### 1. Merge Results (per method)
```bash
# Merge uniform results
python scripts/merge_results.py results/{timestamp}-method_comparison/uniform

# Merge leveraged results
python scripts/merge_results.py results/{timestamp}-method_comparison/leveraged
```

This creates:
- `uniform/results_grid_merged.json`
- `leveraged/results_grid_merged.json`

### 2. Compare Merged Results

Load both JSONs in Python/notebook and compare metrics:

```python
import json
import matplotlib.pyplot as plt

# Load results
with open("uniform/results_grid_merged.json") as f:
    uniform = json.load(f)
with open("leveraged/results_grid_merged.json") as f:
    leveraged = json.load(f)

# Extract partition agreement vs p for each n
for row_u, row_l in zip(uniform["rows"], leveraged["rows"]):
    assert row_u["num_taxa"] == row_l["num_taxa"]
    n = row_u["num_taxa"]

    # Plot agreement vs p
    plt.figure()
    plt.plot(row_u["p_values"], row_u["partition_agreement_M"],
             label="Uniform", marker='o')
    plt.plot(row_l["p_values"], row_l["partition_agreement_M"],
             label="Leveraged", marker='s')
    plt.xlabel("Sampling probability p")
    plt.ylabel("Partition agreement (%)")
    plt.xscale("log")
    plt.title(f"n={n}")
    plt.legend()
    plt.grid(True)
    plt.show()
```

### 3. Key Metrics to Compare

From `results.json`:
- **`partition_agreement_M`**: Main metric (% bootstrap partitions matching ground truth)
- **`partition_agreement_S`**: Strict agreement (smaller partition side)
- **`sign_agreement`**: % Fiedler vector signs matching ground truth
- **`partition_quality`**: σ₂ quality metric

## Research Question

**Hypothesis:** As `n` (taxa) increases, leveraged sampling can achieve similar recovery quality with smaller `p` (fewer samples) compared to uniform sampling.

**What to look for:**
1. For each `n`, find minimum `p` where each method achieves 95% partition agreement
2. Plot `p_min` vs `n` for both methods
3. Expected: leveraged's `p_min` grows slower than uniform's as `n` increases

## Notes

- **Same seed = same data**: Both methods see identical tree and sequences
- **Leveraged is slower**: IALM solver adds computational overhead (~10-100x slower than uniform)
- **IALM convergence**: For very small `p` (<0.01), IALM may not converge in 100 iterations
  - Check logs for "⚠ max_iter" warnings
  - Consider increasing `leveraged_ialm_max_iter` if needed
