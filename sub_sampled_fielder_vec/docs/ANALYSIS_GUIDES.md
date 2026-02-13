# Analysis Guides

This document provides guides for using the various analysis tools and notebooks in the framework.

## Phase Transition Analysis

Phase transition analysis studies how sample complexity scales with problem size, proving that leveraged sampling has better scaling than uniform as n increases.

### Generic Phase Transition Notebook

**File**: `analysis/phase_transition_generic.ipynb`

**Purpose**: Flexible notebook to compare phase transitions across **any set of experiment runs**.

**Use Cases**:
1. Compare tree models: Kingman vs Balanced Binary vs Lopsided
2. Compare sampling methods: Uniform vs Leveraged vs Adaptive
3. Compare parameters: Different θ values, different ranks, etc.
4. Compare anything: Any experiments with multiple n values

**Quick Start**:

1. **Open notebook**:
   ```bash
   cd sub_sampled_fielder_vec/analysis
   jupyter notebook phase_transition_generic.ipynb
   ```

2. **Edit Cell 1** - Configure your runs:
   ```python
   RUNS_CONFIG = [
       {
           "path": "../results/my_run_1",
           "label": "Method A",
           "color": "#2E86AB",
           "marker": "o"
       },
       {
           "path": "../results/my_run_2",
           "label": "Method B",
           "color": "#A23B72",
           "marker": "s"
       },
   ]
   ```

3. **Run all cells** (Kernel → Restart & Run All)

**Configuration Format**:
- **`path`**: Results directory containing `n{X}_L{Y}/results.json` subdirectories
- **`label`**: Display name for plots/tables
- **`color`** (optional): Hex color code (auto-assigned if omitted)
- **`marker`** (optional): Matplotlib marker style (auto-assigned if omitted)

**Output**:
- **Plots**: 
  - Left: Sigmoid-based p* (95% agreement) with power law fits
  - Right: Discrete p* (100% agreement) with connecting lines
  - Saved as: `phase_transition_comparison.png`
- **Tables**: P* values for each run and n, power law equations and exponents

**Requirements**: Each run directory must contain `n{X}_L{Y}/results.json` subdirectories with ≥2 different n values per run.

### Phase Transition Analysis (Method Comparison)

**Files**: 
- `analysis/comparison/phase_transition_utils.py` - All analysis functions
- `analysis/comparison/phase_transition_analysis.ipynb` - Interactive notebook

**Purpose**: Analyze sample complexity scaling to prove leveraged sampling has better scaling than uniform.

**Quick Start**:
```bash
cd sub_sampled_fielder_vec/analysis/comparison
jupyter notebook phase_transition_analysis.ipynb
```

**Edit Cell 2** to set your comparison directory:
```python
COMP_DIR = Path("../../results/YOUR_COMPARISON_DIR")
```

**What It Does**:

1. **Load Data**: Reads both `uniform/` and `leveraged/` results, extracts `(p, partition_agreement_M)` for each n

2. **Compute p* (Critical Sampling Probability)**: Two methods:
   - **Sigmoid fit**: Fit logistic curve, extract p* where sigmoid = 95%
   - **Discrete**: First p where agreement ≥ 100%

3. **Power Law Fit**: Fit: `p* = A · n^α`
   - α < 0: p* decreases as n grows
   - More negative α = better scaling

4. **Generate Plots**: Log-log plot of p* vs n with fitted curves, equations shown on plot

**Expected Result**: Leveraged sampling has MORE NEGATIVE exponent α:
```
α_leverage < α_uniform < 0
```

This means: as n grows, leveraged requires relatively fewer samples to achieve same recovery quality.

**Outputs**: Saved to comparison directory:
- `phase_transition_scaling.png` - Main result (sigmoid-based)
- `phase_transition_discrete.png` - Discrete threshold comparison

**Functions** (from utils.py):
```python
# Load data
data = load_comparison_data(comp_dir)  # {method: {n: [(p, agreement)]}}

# Compute all p* values
transitions = compute_phase_transitions(data)  # DataFrame

# Fit power law
alpha, A, equation = fit_power_law(n_vals, p_star_vals)

# Evaluate fit
p_fit = evaluate_power_law(n_range, alpha, A)
```

## Method Comparison: Uniform vs Leveraged

**Directory**: `analysis/comparison/`

**Purpose**: Compare uniform and leveraged sampling methods on **identical matrices** (same tree, same sequences).

### Key Feature: Fair Comparison

Both methods run on the SAME data using a shared random seed, ensuring:
- Identical tree topology
- Identical sequence observations
- Identical full similarity matrix

Only the sampling/recovery method differs.

### Files

- **`compare_methods.py`**: Main comparison script with full configuration
- **`test_compare.py`**: Quick test with minimal parameters (n=500, 3 bootstrap reps)
- **`plot_comparison.py`**: Generate comparison plots from merged results

### Quick Test (Recommended First)

```bash
cd sub_sampled_fielder_vec
python analysis/comparison/test_compare.py
```

**Test parameters**:
- `n_taxa`: [500]
- `bootstrap_reps`: 3
- `p_values`: [0.01, 0.1, 1.0]

**Runtime**: ~2-5 minutes

### Full Comparison

```bash
cd sub_sampled_fielder_vec
python analysis/comparison/compare_methods.py
```

**Full parameters**:
- `n_taxa`: [500, 1000, 3000, 5000, 7000, 10000]
- `bootstrap_reps`: 20
- `p_values`: logspace(-4, 0, 20) — 20 values from 0.0001 to 1.0
- Tree model: Kingman coalescent
- Sequence length: 10,000

**Runtime**: Several hours (depends on hardware and parallelization)

### Configuration

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

### Output Structure

Results are saved in nested directory structure:

```
results/{timestamp}-method_comparison/
├── comparison_config.json          # Config used for comparison
├── comparison_summary.json         # Summary of all runs
├── uniform/                        # Uniform sampling results
│   ├── n500_L10000/
│   │   ├── results.json
│   │   ├── config.json
│   │   └── ...
│   └── ...
└── leveraged/                      # Leveraged sampling results
    ├── n500_L10000/
    │   ├── results.json
    │   └── ...
    └── ...
```

### Analyzing Results

#### 1. Merge Results (per method)

```bash
# Merge uniform results
python scripts/merge_results.py results/{timestamp}-method_comparison/uniform

# Merge leveraged results
python scripts/merge_results.py results/{timestamp}-method_comparison/leveraged
```

This creates:
- `uniform/results_grid_merged.json`
- `leveraged/results_grid_merged.json`

#### 2. Compare Merged Results

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

#### 3. Key Metrics to Compare

From `results.json`:
- **`partition_agreement_M`**: Main metric (% bootstrap partitions matching ground truth)
- **`partition_agreement_S`**: Strict agreement (smaller partition side)
- **`sign_agreement`**: % Fiedler vector signs matching ground truth
- **`partition_quality`**: σ₂ quality metric

### Research Question

**Hypothesis:** As `n` (taxa) increases, leveraged sampling can achieve similar recovery quality with smaller `p` (fewer samples) compared to uniform sampling.

**What to look for**:
1. For each `n`, find minimum `p` where each method achieves 95% partition agreement
2. Plot `p_min` vs `n` for both methods
3. Expected: leveraged's `p_min` grows slower than uniform's as `n` increases

### Notes

- **Same seed = same data**: Both methods see identical tree and sequences
- **Leveraged is slower**: IALM solver adds computational overhead (~10-100x slower than uniform)
- **IALM convergence**: For very small `p` (<0.01), IALM may not converge in 100 iterations
  - Check logs for "⚠ max_iter" warnings
  - Consider increasing `leveraged_ialm_max_iter` if needed

## Spectral Analysis Framework

**Directory**: `analysis/spectral_analysis/`

### Sweep Parameters Analysis

**Purpose**: Characterizes the **Spectral Concentration Phase Transition** in subsampled Laplacians.

**Three Operating Regimes**:

1. **noise_bulk**: Signal buried in noise (partition_agreement_M ≤ 55%)
   - λ₂(L_S) within Marchenko-Pastur noise bulk
   - Random partitions, high IPR (localized eigenvector)

2. **spectral_emergence**: BBP transition (55% < agreement < 95%)
   - λ₂(L_S) "pops" out of noise
   - Partition agreement: 50% → 100%
   - IPR drops (delocalization)

3. **perturbation_plateau**: Davis-Kahan stability (agreement ≥ 95%)
   - L_S is valid spectral proxy for L_M
   - DK ratio < 0.5
   - Perfect partition agreement

**Usage**:
```bash
# From sub_sampled_fielder_vec/ directory
python -m spectral_analysis.sweep_params_analysis.run_analysis \
  results/20251130-193600-balanced_tree_mu_01
```

**Input Requirements**: The `results_grid_merged.json` file must contain:
- `num_taxa`, `sequence_length`, `p`
- `partition_agreement_M`, `std`
- `lambda_2_S`, `lambda_3_S`, `lambda_2_M`, `lambda_3_M`
- `mean_dk_ratio_S`, `mean_ipr_S`

**Outputs**: Analysis results saved to `results/{run_name}/analysis_{timestamp}/`:

**Files**:
- `regime_classification.csv` - Per-configuration regime labels
- `scaling_law_fit.txt` - Regression: p_crit ~ f(N, L)

**Plots** (5 files total):
- **Graph A**: `A_phase_boundary.png` - Combined scatter plot (p vs L, color by N, size by agreement)
- **Graph B**: `B_eigenvalue_pop.png` - λ₂(p), λ₃(p) trajectories in 5×4 grid
  - Shaded gray "noise ocean" (0 to λ₃)
  - Bold red λ₂ line (signal emerging)
  - Vertical markers: blue (λ₂ > λ₃), green (p_crit)
- **Graph C**: `C_stability_curve.png` - Davis-Kahan ratio vs p in 5×4 grid
- **Graph D**: `D_ipr_delocalization.png` - IPR delocalization curve in 5×4 grid
  - Red dashed baseline at 1/N (perfect delocalization)
  - X-axis synchronized with Graphs B & C
- **Graph E**: `E_scaling_law.png` - p_crit vs L validation
  - One curve per N value
  - Theory overlays: p ∝ L^(-0.5), L^(-1), fitted exponent

**Notation**:
- **Subscripts**: `_S` = subsampled, `_M` = full matrix
- **Variables**: p (sampling rate), N (num_taxa), L (sequence_length)
- **Eigenvalues**: λ₂ (Fiedler), λ₃ (third eigenvalue)

### Target Quality Analysis

**Purpose**: Pre-flight diagnostics for full similarity matrices **before** running subsampling experiments.

**What It Computes**:

For each (tree_model, n, L, μ) combination:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Coherence** | max<sub>i</sub> \|\|u<sub>i</sub>\|\|<sub>∞</sub><sup>2</sup> | Matrix incoherence (lower is better for sampling) |
| **Numerical Rank** | \|\|M\|\|<sub>F</sub><sup>2</sup> / \|\|M\|\|<sub>2</sub><sup>2</sup> | Effective dimensionality |
| **Sigma2** | σ₂(M<sub>partition</sub>) | Cross-partition quality (lower is better) |
| **Partition Split** | (n<sub>small</sub>, n<sub>large</sub>) | Partition balance |
| **Spectral Gap** | \|λ₃ - λ₂\| | Absolute eigenvalue gap |
| **Relative Gap** | \|λ₃ - λ₂\| / λ₂ | Relative eigenvalue gap |
| **Lambda2, Lambda3** | λ₂, λ₃ of L<sub>M</sub> | Raw eigenvalues for reference |

Plus: **Eigenvalue scree plots** for both M and L<sub>M</sub>

**Quick Start**:

#### Single-Run Analysis (with plots)

```bash
cd sub_sampled_fielder_vec
python -m spectral_analysis.target_quality_anlysis.cli.target_analysis_main \
    spectral_analysis/target_quality_anlysis/config_template.json
```

#### Stability Analysis (K trials, statistics only)

```bash
# Run with default K=10 trials
python -m spectral_analysis.target_quality_anlysis.cli.stability_analysis_main \
    spectral_analysis/target_quality_anlysis/config_template.json

# Run with custom number of trials
python -m spectral_analysis.target_quality_anlysis.cli.stability_analysis_main \
    spectral_analysis/target_quality_anlysis/config_template.json --num-trials 30
```

**When to use stability analysis**:
- ✓ When you see "No" for `is_valid_partition` and want to know if it's always like that
- ✓ To quantify randomness in metrics (coherence, σ₂, spectral gap, etc.)
- ✓ To report mean ± std for publication-quality results
- ✓ To determine if partition validity is systematic (always fails) or stochastic (sometimes works)

**Create Your Own Config**:

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

**Available tree models**:
- `balanced_binary` (params: `edge_length`)
- `kingman_mean` (params: `pop_size`)
- `kingman` (params: `pop_size`)
- `birth_death` (params: `birth_rate`, `death_rate`)
- `lopsided` (params: `edge_length`)

**Available sequence models**:
- `JC69` (simplest, no params needed besides `mutation_rate`)
- `HKY` (params: `kappa`, `stationary_freqs`)
- `GTR`, `TN93`, `T92` (see sequence_models.py for params)

**Output Structure**:

**Single-run analysis** results:
```
analysis_results/{timestamp}-{experiment_name}/
├── config.json                    # Copy of input config
├── diagnostics_table.txt          # All metrics in tabular format
├── summary_statistics.txt         # Mean/median/std across configs
├── A_similarity_eigenvalues.png   # Similarity matrix M scree plot
├── B_laplacian_eigenvalues.png    # Laplacian L_M scree plot
└── C_coherence.png                # Coherence comparison (bar chart)
```

**Stability analysis** results:
```
analysis_results/{timestamp}-{experiment_name}_stability_K{num_trials}/
├── config.json                    # Copy of input config (with num_trials added)
├── stability_table.txt            # Main results: mean±std for all metrics
├── stability_summary.txt          # Cross-config summary statistics
└── detailed_metrics.txt           # Detailed per-config metrics with min/max
```

**Interpreting Results**:

**Good indicators**:
- σ₂ < 0.2 (clean partition)
- Relative gap > 0.1 (well-separated eigenvalues)
- NumRank << n (low-rank structure)
- Balanced partition (close to n/2 | n/2)

**Warning signs**:
- σ₂ > 0.3 (weak partition)
- Relative gap < 0.05 (eigenvalues too close)
- NumRank ≈ n (full rank, noisy)
- Unbalanced partition (e.g., 10|90)

## Leverage Sampling Explorer

**File**: `analysis/notebooks/leverage_sampling_explorer.ipynb`

**Purpose**: Validate leverage sampling by comparing estimated leverage scores (from Phase 1) against ground truth computed from the full similarity matrix.

**Research Question**: Does Phase 1 uniform sampling provide reliable leverage score estimates? How does estimation quality depend on the sampling budget p?

### Quick Start

```bash
cd sub_sampled_fielder_vec/analysis/notebooks
jupyter notebook leverage_sampling_explorer.ipynb
```

**Edit Cell 2** - Configuration:
```python
# Path to experiment output directory
EXPERIMENT_PATH = "../../results/20260208-221342-balanced_binary_n1024_mu_0p1_leveraged/n1024_L10000"

# P-value to analyze
P_VALUE = 0.1438  # Must have corresponding sampling_data/p_{p:.4f}.npz file

# Figure size
FIGURE_WIDTH = 18
FIGURE_HEIGHT = 5
```

**Run** all cells (Kernel → Restart & Run All)

### Requirements

**Experiment must have**:
1. Used `sampling_method="leveraged"`
2. Enabled `log_sampling_diagnostics=True`
3. At least one p-value ≥ theoretical minimum (where leveraged sampling runs)

**Check if diagnostics exist**:
```bash
ls results/{your_experiment}/n{taxa}_L{seq_len}/sampling_data/
# Should see: p_0.1438.npz, p_0.2336.npz, etc.
```

### What It Computes

1. **Load Data**:
   - Full similarity matrix M from persistent cache
   - Estimated leverage scores from Phase 1 sampling
   - Phase 2 sampling probabilities (sparse)

2. **Compute Ground Truth**:
   - Run SVD on full matrix M (rank=2)
   - Compute true leverage scores: μᵢ = ||Uᵢ||² (row norms of top-2 left singular vectors)

3. **Compare**:
   - Correlation between true and estimated scores
   - Mean absolute error (MAE)
   - Top-k most important taxa (by true leverage)

### Visualization

**Three-panel figure**:

| Plot | Shows | Interpretation |
|------|-------|----------------|
| **A: Similarity Matrix** | Full M heatmap | Ground truth data |
| **B: Leverage Comparison** | Scatter: true vs estimated scores | Correlation validates Phase 1 quality |
| **C: Sampling Probabilities** | Heatmap of Phase 2 probs (sparse) | Which entries were prioritized? |

**Scatter Plot Features**:
- Red diagonal line = perfect correlation
- Correlation coefficient displayed in title
- Each point = one taxon

### Output

**Summary Statistics**:
```
============================================================
SUMMARY STATISTICS
============================================================
Correlation (true vs estimated): 0.8423
Mean absolute error: 2.3451
Max absolute error: 15.2341

Phase 2 sampling coverage:
  Total entries sampled: 18560
  Sampling rate: 3.54%

Top 5 taxa by true leverage score:
  1. Taxa 277: true=10.9833, estimated=9.1234
  2. Taxa 295: true=10.4828, estimated=8.9012
  ...
```

### Interpreting Results

**High correlation (≥ 0.7)**:
- ✅ Phase 1 budget is sufficient
- ✅ Leverage-based sampling makes sense
- ✅ High-leverage entries are correctly prioritized

**Low correlation (< 0.3)**:
- ⚠️ Phase 1 budget too small
- ⚠️ p-value near theoretical minimum
- ⚠️ Consider higher p or larger θ (Phase 1 ratio)

**Medium correlation (0.3-0.7)**:
- 🔄 Partial signal detected
- 🔄 May benefit from increased Phase 1 budget
- 🔄 Check if p is just above threshold

### Example Findings

**From n=1024, p=0.1438 experiment**:
- Correlation: -0.0728 (nearly zero!)
- Sampling rate: 3.54%
- **Interpretation**: At this threshold p-value, Phase 1 uniform sampling with only 3.54% coverage doesn't reliably estimate leverage scores
- **Validates**: Theoretical minimum budget requirement (4·n·r·log(n))

**Try higher p-values** (e.g., p=0.6158) to see correlation improve with denser sampling.

### Common Issues

**"No cached matrix found"**:
- First run creates cache - subsequent runs load instantly
- Check `src/cache/` directory exists

**"Sampling diagnostics not found"**:
- Ensure `log_sampling_diagnostics=True` in config
- Check p-value is high enough for leveraged sampling to run
- For n=1024: need p ≥ 0.108

**Import errors**:
- Notebook uses `importlib.util` to bypass package `__init__.py` issues
- Should work from `analysis/notebooks/` directory

### Advanced Usage

**Compare multiple p-values**:

Loop over all available diagnostic files:
```python
import glob
sampling_files = glob.glob("../../results/my_experiment/*/sampling_data/p_*.npz")

for file in sampling_files:
    p_val = float(file.split("p_")[1].split(".npz")[0])
    # ... load and analyze ...
```

**Export data for custom analysis**:
```python
# After running notebook cells:
np.savez("leverage_analysis.npz",
    leverage_true=leverage_true,
    leverage_estimated=leverage_estimated,
    correlation=correlation,
    phase2_probs=phase2_probs_sampled
)
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Codebase structure
- [METRICS.md](METRICS.md) - Metrics documentation
- [LEVERAGED_SAMPLING.md](LEVERAGED_SAMPLING.md) - Leveraged sampling details
