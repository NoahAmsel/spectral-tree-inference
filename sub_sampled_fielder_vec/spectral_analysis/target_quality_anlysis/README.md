# Target Quality Analysis

Pre-flight diagnostics for full similarity matrices **before** running subsampling experiments.

## Purpose

This module analyzes the full N×N similarity matrix to assess its quality and suitability for spectral partitioning. It computes key metrics that help you understand:

1. **Matrix structure** (coherence, numerical rank)
2. **Partition quality** (Fiedler vector, σ₂)
3. **Spectral properties** (eigenvalue gaps)

## What It Computes

For each (tree_model, n, L, μ) combination:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Coherence** | max<sub>i</sub> ‖u<sub>i</sub>‖<sub>∞</sub><sup>2</sup> | Matrix incoherence (lower is better for sampling) |
| **Numerical Rank** | ‖M‖<sub>F</sub><sup>2</sup> / ‖M‖<sub>2</sub><sup>2</sup> | Effective dimensionality |
| **Sigma2** | σ₂(M<sub>partition</sub>) | Cross-partition quality (lower is better) |
| **Partition Split** | (n<sub>small</sub>, n<sub>large</sub>) | Partition balance |
| **Spectral Gap** | \|λ₃ - λ₂\| | Absolute eigenvalue gap |
| **Relative Gap** | \|λ₃ - λ₂\| / λ₂ | Relative eigenvalue gap |
| **Lambda2, Lambda3** | λ₂, λ₃ of L<sub>M</sub> | Raw eigenvalues for reference |

Plus: **Eigenvalue scree plots** for both M and L<sub>M</sub>

## Quick Start

### 1. Single-Run Analysis (with plots)

Run diagnostics once per configuration and generate visualizations:

```bash
cd /Users/itaygonnen/Python_Repos/spectral-tree-inference/sub_sampled_fielder_vec
python -m spectral_analysis.target_quality_anlysis.cli.target_analysis_main \
    spectral_analysis/target_quality_anlysis/config_template.json
```

### 1b. Stability Analysis (K trials, statistics only)

Run K independent trials per configuration to assess statistical variability:

```bash
# Run with default K=10 trials
python -m spectral_analysis.target_quality_anlysis.cli.stability_analysis_main \
    spectral_analysis/target_quality_anlysis/config_template.json

# Run with custom number of trials
python -m spectral_analysis.target_quality_anlysis.cli.stability_analysis_main \
    spectral_analysis/target_quality_anlysis/config_template.json --num-trials 30
```

**When to use stability analysis:**
- ✓ When you see "No" for `is_valid_partition` and want to know if it's always like that
- ✓ To quantify randomness in metrics (coherence, σ₂, spectral gap, etc.)
- ✓ To report mean ± std for publication-quality results
- ✓ To determine if partition validity is systematic (always fails) or stochastic (sometimes works)

### 2. Create Your Own Config

Copy and edit `config_template.json`:

```json
{
  "experiment_name": "my_experiment",
  "description": "Description of what I'm testing",

  "models": [
    {
      "name": "balanced_binary",
      "tree": {
        "model": "balanced_binary",
        "params": {"edge_length": 1.0}
      },
      "sequence": {
        "model": "JC69",
        "params": {}
      }
    }
  ],

  "configs": [
    {"n": 512, "L": 1000, "mu": 0.1},
    {"n": 1024, "L": 1000, "mu": 0.1}
  ],

  "analysis_params": {
    "num_gaps": 1,
    "min_split": 2,
    "k_scree": 20,
    "coherence_k": 2
  }
}
```

**Available tree models:**
- `balanced_binary` (params: `edge_length`)
- `kingman_mean` (params: `pop_size`)
- `kingman` (params: `pop_size`)
- `birth_death` (params: `birth_rate`, `death_rate`)
- `lopsided` (params: `edge_length`)

**Available sequence models:**
- `JC69` (simplest, no params needed besides `mutation_rate`)
- `HKY` (params: `kappa`, `stationary_freqs`)
- `GTR`, `TN93`, `T92` (see sequence_models.py for params)

### 3. View Results

**Single-run analysis** results are saved to `analysis_results/{timestamp}-{experiment_name}/`:

```
analysis_results/20251202-212237-quick_test/
├── config.json                    # Copy of input config
├── diagnostics_table.txt          # All metrics in tabular format
├── summary_statistics.txt         # Mean/median/std across configs
├── A_similarity_eigenvalues.png   # Similarity matrix M scree plot
├── B_laplacian_eigenvalues.png    # Laplacian L_M scree plot
└── C_coherence.png                # Coherence comparison (bar chart)
```

**Stability analysis** results are saved to `analysis_results/{timestamp}-{experiment_name}_stability_K{num_trials}/`:

```
analysis_results/20251206-202628-stability_test_stability_K10/
├── config.json                    # Copy of input config (with num_trials added)
├── stability_table.txt            # Main results: mean±std for all metrics
├── stability_summary.txt          # Cross-config summary statistics
└── detailed_metrics.txt           # Detailed per-config metrics with min/max
```

**Example stability_table.txt:**
```
Model               n     L     mu    Trials Coherence      NumRank     Sigma2       Partition      Valid   Gap          ...
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
balanced_binary     1024  10000 0.10  10     0.112±0.003    7.86±0.21   0.143±0.008  512|512±2      10/10   0.142±0.005  ...
kingman_mean        1024  10000 0.10  10     0.098±0.012    8.45±0.67   0.187±0.023  507|517±18     7/10    0.089±0.014  ...
```

**Interpreting stability results:**
- **Valid column** shows success/total (e.g., "7/10" means partition was valid in 7 out of 10 trials)
- **mean±std** shows average value ± standard deviation across trials
- **Partition column** shows typical split ± variability (e.g., "512|512±2" means split varies by ±2 taxa)
- **Low std** indicates stable, reproducible behavior
- **High std** indicates high randomness in that metric

**All plots use side-by-side layout**, showing balanced_binary on the left and kingman_mean on the right, with all (n, L, μ) configurations displayed together. **Each subplot has independent y-axis scaling** to accommodate different value ranges between models.

**Simple alphabetical naming** (A, B, C) ensures plots appear in logical order when sorted.

## Example: Comparing Tree Models

```json
{
  "experiment_name": "kingman_vs_balanced_comparison",
  "models": [
    {
      "name": "kingman_mean",
      "tree": {"model": "kingman_mean", "params": {"pop_size": 1.0}},
      "sequence": {"model": "JC69", "params": {}}
    },
    {
      "name": "balanced_binary",
      "tree": {"model": "balanced_binary", "params": {"edge_length": 1.0}},
      "sequence": {"model": "JC69", "params": {}}
    }
  ],
  "configs": [
    {"n": 512, "L": 500, "mu": 0.1},
    {"n": 512, "L": 1000, "mu": 0.1},
    {"n": 1024, "L": 500, "mu": 0.1},
    {"n": 1024, "L": 1000, "mu": 0.1}
  ]
}
```

This will generate comparison plots showing both models on the same axes.

## Understanding the Output Table

```
Model          n     L     mu    Coherence  NumRank   Sigma2    Partition   Gap       RelGap    Lambda2   Lambda3
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
balanced_binary64    100   0.10  0.112324   7.86      0.1431    32|32       0.142339  0.260533  0.546337  0.688677
```

**Interpretation:**
- **Coherence = 0.112**: Moderate incoherence (good for random sampling)
- **NumRank = 7.86**: Matrix has ~8 effective dimensions (out of 64 possible)
- **Sigma2 = 0.1431**: Cross-partition signal is moderate
- **Partition = 32|32**: Perfect balance
- **Gap = 0.142**: Absolute spectral gap between λ₂ and λ₃
- **RelGap = 0.261**: Gap is 26% of λ₂ (moderate separation)

## Plots

All plots use **side-by-side layout** (balanced_binary | kingman_mean) with:
- **Independent y-axes** for each subplot (better for different scales)
- **Shared legend** across both panels
- Same x-axis structure for alignment

### 1. Scree Plots

**Similarity Matrix (M)**: Shows largest eigenvalues (descending)
   - Indicates rank and signal strength
   - Sharp drop-off = low-rank structure
   - Compare how n, L, μ affect the eigenvalue spectrum

**Laplacian (L_M)**: Shows smallest eigenvalues (ascending)
   - λ₁ ≈ 0 (always, for connected graphs)
   - λ₂ = Fiedler eigenvalue (partition quality)
   - Gap between λ₂ and λ₃ is critical
   - See how parameters affect spectral gap

### 2. Coherence Plot

**Bar chart** showing coherence values for each configuration:
   - **Lower is better** (less coherence = better for random sampling)
   - Bars colored on gradient: green (good) → yellow → red (poor)
   - Values labeled on top of each bar
   - Easy visual comparison across (n, L, μ) combinations

**Example**: With 3 configs per model, the coherence plot shows:
- Left panel: 3 bars for balanced_binary (with its own y-scale)
- Right panel: 3 bars for kingman_mean (with its own y-scale)
- Same x-axis labels (n=512,L=500; n=512,L=1000; etc.)
- Independent y-axes allow optimal visualization of each model's range

## Module Structure

```
target_quality_anlysis/
├── README.md                    # This file
├── __init__.py                  # Package exports
├── config_template.json         # Example configuration
│
├── computation/                 # Individual metric computations
│   ├── __init__.py
│   ├── coherence.py             # compute_coherence()
│   ├── numerical_rank.py        # compute_numerical_rank()
│   ├── partition.py             # compute_partition_diagnostics()
│   ├── spectral_gaps.py         # compute_spectral_gaps()
│   ├── eigenvalues.py           # compute_eigenvalues_for_scree()
│   └── computation_orchestrator.py  # run_full_diagnostics() - orchestrates all
│
├── visualization/               # Plotting functions
│   ├── __init__.py
│   ├── scree_plots.py           # Scree plot generation
│   ├── coherence_plots.py       # Coherence comparison plots
│   └── heatmaps.py              # Metric heatmaps
│
├── output/                      # Output generation
│   ├── __init__.py
│   ├── tables.py                # Text table generation
│   └── partition_validity.py    # Partition validation
│
└── cli/                         # Command-line interface
    ├── __init__.py
    ├── config.py                # Configuration loading
    ├── directory.py             # Output directory creation
    └── target_analysis_main.py  # Main CLI entry point
```

## Design Principles

1. **Simple functions**: Each function computes exactly one metric
2. **Minimal logic**: No complex abstractions
3. **Reuse existing code**: Imports from `src/core/` and `src/utils/`
4. **Single output directory**: All models together for easy comparison
5. **Text tables first**: Get numbers right before plotting

## Advanced Usage

### Custom Analysis Parameters

```json
{
  "analysis_params": {
    "num_gaps": 1,        # Number of gap-based thresholds for partition_taxa
    "min_split": 2,       # Minimum partition size
    "k_scree": 20,        # Number of eigenvalues for scree plots
    "coherence_k": 2      # Number of top singular vectors for coherence
  }
}
```

### Python API

```python
from spectral_analysis.target_quality_anlysis.computation import (
    run_full_diagnostics,
    compute_coherence,
    compute_numerical_rank,
    compute_spectral_gaps
)

# Run full analysis
results = run_full_diagnostics(
    tree_config={'model': 'balanced_binary', 'params': {'num_taxa': 64, 'edge_length': 1.0}},
    seq_config={'model': 'JC69', 'len': 100, 'params': {'mutation_rate': 0.1}}
)

# Access individual metrics
print(f"Coherence: {results['coherence']:.6f}")
print(f"NumRank: {results['numerical_rank']:.2f}")
print(f"Sigma2: {results['sigma2']:.4f}")
```

## When to Use This Tool

**Before running subsampling experiments**, use this tool to:

1. ✅ Verify matrix has good spectral properties
2. ✅ Check partition quality (low σ₂)
3. ✅ Ensure sufficient spectral gap
4. ✅ Compare different tree models
5. ✅ Determine if n or L needs adjustment

**Good indicators:**
- σ₂ < 0.2 (clean partition)
- Relative gap > 0.1 (well-separated eigenvalues)
- NumRank << n (low-rank structure)
- Balanced partition (close to n/2 | n/2)

**Warning signs:**
- σ₂ > 0.3 (weak partition)
- Relative gap < 0.05 (eigenvalues too close)
- NumRank ≈ n (full rank, noisy)
- Unbalanced partition (e.g., 10|90)

## Troubleshooting

**Error: "Unknown tree model"**
- Check `src/models/tree_models.py` for available models
- Ensure `num_taxa` is included in tree params

**Error: "mutation_rate is required"**
- Add `"mutation_rate": 0.1` to sequence params

**Warnings from sklearn**
- These are expected from SVD routines and don't affect results
- They occur during eigenvalue/singular value computations

## Contact

For questions or issues, refer to the main repository documentation.
