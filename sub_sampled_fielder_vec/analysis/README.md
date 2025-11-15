# Spectral Tree Reconstruction Analysis

This directory contains a comprehensive 3-phase analysis framework for understanding the phase transition phenomenon in spectral tree reconstruction with subsampling.

## Overview

The analysis investigates why sign agreement jumps sharply from ~50% (failure) to 100% (success) as sampling probability increases, and why **"bigger matrices need less sampling"**.

## Directory Structure

```
analysis/
├── README.md                           # This file
├── utils.py                            # Shared utilities
├── phase1_diagnostic_metrics.py        # Phase 1: Identify key predictive metrics
├── phase2_scaling_laws.py              # Phase 2: Quantify scaling relationships
├── phase3_theoretical_connection.py    # Phase 3: Theoretical interpretation
└── run_all_phases.py                   # Main runner script

../results/combined_grid_search_results/
├── results_grid.json                   # Input data
└── analysis_outputs/                   # Generated outputs
    ├── phase1/                         # Phase 1 outputs
    │   ├── spectral_gap_ratio_vs_p.png
    │   ├── rank_ratio_L_S_vs_p.png
    │   ├── frobenius_error_vs_p.png
    │   ├── pre_vs_post_transition_n8192_L10000.png
    │   ├── transition_thresholds.csv
    │   └── phase1_summary.md           # Phase 1 report
    ├── phase2/                         # Phase 2 outputs
    │   ├── transition_p_scaling.png
    │   ├── effective_samples_vs_size.png
    │   ├── phase_diagram.png
    │   ├── scaling_coefficients.json
    │   └── phase2_summary.md           # Phase 2 report
    └── phase3/                         # Phase 3 outputs
        ├── eigenvalue_statistics.png
        ├── information_bottleneck.png
        ├── coherence_analysis.png
        ├── theoretical_bounds_comparison.png
        ├── theoretical_bounds.csv
        └── phase3_summary.md           # Phase 3 report
```

## Running the Analysis

### Prerequisites

```bash
# Required packages
pip install numpy pandas matplotlib seaborn scipy
```

### Run All Phases

```bash
cd analysis
python run_all_phases.py
```

### Run Individual Phases

```bash
# Phase 1 only
python run_all_phases.py --phase 1

# Phase 2 only
python run_all_phases.py --phase 2

# Phase 3 only
python run_all_phases.py --phase 3
```

Or run scripts directly:

```bash
python phase1_diagnostic_metrics.py
python phase2_scaling_laws.py
python phase3_theoretical_connection.py
```

## Phase Descriptions

### Phase 1: Diagnostic Metrics Analysis

**Goal**: Identify which linear algebra metrics best predict the phase transition.

**Key Questions**:
- What happens to spectral gap ratio at transition?
- When does rank become full?
- What are the critical threshold values?

**Main Findings**:
- **Spectral gap ratio** (SpGap_LS / SpGap_LM) is the primary indicator
- Transition occurs when ratio drops from >100 to <10
- Full rank is achieved at transition

**Outputs**:
- Faceted plots of metrics vs sampling probability
- Transition threshold table
- Detailed analysis of n=8192, L=10000 case

### Phase 2: Scaling Laws Analysis

**Goal**: Quantify how transition sampling probability scales with matrix size.

**Key Questions**:
- How does p_transition scale with (n × L)?
- What is the power law exponent?
- How do absolute sample requirements scale?

**Main Findings**:
- Power law: `p_transition ∝ (n × L)^b` where b ≈ -0.5 to -0.8
- Larger matrices are more sample-efficient per entry
- Absolute samples still increase sub-quadratically

**Outputs**:
- Scaling law fits with R² values
- Phase diagrams showing success/failure regions
- Effective sample requirements analysis

### Phase 3: Theoretical Interpretation

**Goal**: Connect empirical findings to theoretical frameworks.

**Key Questions**:
- Does failure regime show random matrix behavior?
- What is the information-theoretic threshold?
- Why does coherence increase at transition?

**Main Findings**:
- Undersampled matrices behave like random matrices
- Transition requires ~50-100 samples per tree parameter
- Results align with spectral sparsification theory
- Phase transition explained by perturbation theory (Davis-Kahan)

**Outputs**:
- Eigenvalue behavior across transition
- Information-theoretic analysis
- Comparison to theoretical bounds
- Comprehensive theoretical interpretation

## Key Results Summary

### The Phase Transition Mechanism

1. **Failure regime** (p < p_critical):
   - SpGap_LS >> SpGap_LM (ratio > 100)
   - Sampling noise dominates signal
   - Sign agreement ≈ 50% (random)

2. **Transition** (p ≈ p_critical):
   - SpGap_LS ≈ SpGap_LM (ratio < 10)
   - Spectral structure preserved
   - Sharp jump to 100%

3. **Success regime** (p > p_critical):
   - Full spectral fidelity
   - Maintains 100% agreement

### Scaling Law

```
p_critical ≈ a × (n × L)^b
```

where:
- `a` ≈ 10^-2 to 10^-3
- `b` ≈ -0.5 to -0.8

**Practical interpretation**: As matrix size doubles, required sampling rate drops by ~40-60%.

### Theoretical Foundation

The transition is explained by **spectral perturbation theory**:

```
Eigenvector error ≤ Perturbation magnitude / Spectral gap
```

When sampling provides:
```
p > n² / (spectral gap)²
```

the Fiedler vector is preserved.

## Data Format

### Input: `results_grid.json`

Structure:
```json
{
  "columns": ["num_taxa", "sequence_length", "p", "mean", ...],
  "rows": [
    {
      "num_taxa": 1024,
      "sequence_length": 500,
      "p": 0.0001,
      "mean": 29.01,  // Sign agreement (%)
      "mean_spectral_gap_L_S": 105.2,
      "mean_spectral_gap_L_M": 0.0007,
      ...
    },
    ...
  ]
}
```

### Key Metrics

**Genetic Parameters**:
- `num_taxa` (n): Number of species
- `sequence_length` (L): DNA sequence length
- `p`: Sampling probability (mutation rate analog)

**Performance Metric**:
- `mean`: Sign agreement (%) - reconstruction accuracy

**Linear Algebra Metrics** (all have mean/median/std variants):
- `spectral_gap_M/S`: Eigenvalue gap (λ₂ - λ₁)
- `empirical_rank_M/S`: Effective rank
- `frobenius_error`: ||M - S||_F
- `coherence_M/S`: Max eigenvector coherence
- `min_separation_M/S`: Minimum eigenvalue spacing

**Matrix Types**:
- `M`: Original similarity matrix
- `S`: Sampled version (with probability p)
- `L_M`: Laplacian of M
- `L_S`: Laplacian of S (used for reconstruction)

## Utilities API

### `utils.py`

```python
from utils import load_results, get_transition_point, plot_metric_vs_p_faceted

# Load data
df = load_results()

# Find transition point
trans = get_transition_point(df, num_taxa=8192, sequence_length=10000, threshold=90.0)

# Get all transitions
transitions = get_all_transitions(df, threshold=90.0)

# Create faceted plots
plot_metric_vs_p_faceted(
    df,
    metric='spectral_gap_ratio',
    ylabel='Spectral Gap Ratio',
    output_path='output.png'
)
```

## Extending the Analysis

### Adding New Metrics

1. Add computed column in `utils.py`:
```python
df['my_new_metric'] = df['col1'] / df['col2']
```

2. Use in phase scripts:
```python
plot_metric_vs_p_faceted(df, metric='my_new_metric', ...)
```

### Adding New Analyses

Create new phase script following the template:

```python
from utils import load_results, get_output_dir, save_markdown_report

def my_analysis(df, output_dir):
    # Analysis code
    # Save figures to output_dir
    pass

def main():
    df = load_results()
    output_dir = get_output_dir(4)  # Phase 4
    my_analysis(df, output_dir)
    # Generate report

if __name__ == "__main__":
    main()
```

## Citation

If you use this analysis framework, please cite:

```
[Your paper citation here]
```

## Contact

For questions or issues, please contact [your contact info].

## License

[Your license here]
