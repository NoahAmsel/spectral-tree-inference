# Spectral Tree Reconstruction Analysis

This directory contains a comprehensive 3-phase analysis framework for understanding the phase transition phenomenon in spectral tree reconstruction with subsampling.

## Overview

The analysis investigates why partition agreement jumps sharply from ~50% (failure) to 100% (success) as sampling probability increases, and why **"bigger matrices need less sampling"**.

**Note**: The analysis uses **partition agreement** as the primary metric, which measures whether the actual STDR partitioning algorithm produces the same tree splits, not just whether vector elements have matching signs.

## Directory Structure

```
analysis/
├── README.md                              # This file
├── run_all.py                             # Main entry point - runs all analyses
│
├── shared/                                # Shared utilities (all analyses)
│   ├── __init__.py
│   ├── data_loading.py                    # Load results_grid.json
│   ├── transition_detection.py            # Find transition points
│   ├── plotting_helpers.py                # Reusable plotting functions
│   └── report_writing.py                  # Markdown report generation
│
├── metrics_analysis/                      # Phase 1: What predicts transition?
│   ├── __init__.py
│   ├── run.py                             # Orchestrator for this analysis
│   ├── spectral_gap.py                    # Spectral gap ratio analysis
│   ├── rank_recovery.py                   # Rank preservation analysis
│   ├── frobenius_error.py                 # Frobenius error analysis
│   ├── quality_gap.py                     # Partition M vs S gap
│   ├── transition_table.py                # Create transition threshold table
│   └── write_report.py                    # Generate summary report
│
├── scaling_laws/                          # Phase 2: How do requirements scale?
│   ├── __init__.py
│   ├── run.py                             # Orchestrator for this analysis
│   ├── power_law_fitting.py               # Fit p_crit ~ L^b
│   ├── sample_requirements.py             # Effective sample analysis
│   ├── phase_diagrams.py                  # Create heatmap visualizations
│   └── write_report.py                    # Generate summary report
│
└── theoretical_interpretation/            # Phase 3: Why does it work?
    ├── __init__.py
    ├── run.py                             # Orchestrator for this analysis
    ├── eigenvalue_statistics.py           # Spectral gap evolution
    ├── coherence_analysis.py              # Eigenvector coherence
    ├── theoretical_bounds.py              # Davis-Kahan bounds
    └── write_report.py                    # Generate summary report
```

**Output structure**:
```
../results/combined_grid_search_results/
├── results_grid.json                      # Input data
└── analysis_results/                      # Generated outputs
    ├── metrics_analysis/
    │   ├── spectral_gap_ratio_vs_p.png
    │   ├── rank_ratio_L_S_vs_p.png
    │   ├── frobenius_error_vs_p.png
    │   ├── partition_M_vs_S_gap.png
    │   ├── transition_thresholds.csv
    │   └── metrics_analysis_summary.md
    ├── scaling_laws/
    │   ├── power_law_fit.png
    │   ├── effective_samples_vs_p.png
    │   ├── phase_diagram_heatmap.png
    │   └── scaling_laws_summary.md
    └── theoretical_interpretation/
        ├── eigenvalue_gaps_vs_p.png
        ├── coherence_vs_p.png
        ├── davis_kahan_bound.png
        └── theoretical_interpretation_summary.md
```

## Running the Analysis

### Prerequisites

```bash
# Required packages
pip install numpy pandas matplotlib seaborn scipy
```

### Run All Analyses

```bash
cd analysis
python run_all.py
```

This runs all three phases in sequence and generates comprehensive reports.

### Run Individual Analyses

```bash
# Metrics analysis only
python -m metrics_analysis.run

# Scaling laws only
python -m scaling_laws.run

# Theoretical interpretation only
python -m theoretical_interpretation.run
```

## Analysis Descriptions

### Phase 1: Metrics Analysis

**Question**: What predicts the transition?

**Modules**:
- `spectral_gap.py` - Analyze spectral gap ratio (primary indicator)
- `rank_recovery.py` - Analyze when full rank is achieved
- `frobenius_error.py` - Matrix difference evolution
- `quality_gap.py` - Quality degradation (M vs S_avg)
- `transition_table.py` - Comprehensive transition table

**Key Finding**: **Spectral gap ratio** (SpGap_LS / SpGap_LM) drops from >100 to <10 at transition.

### Phase 2: Scaling Laws

**Question**: How do sample requirements scale with problem size?

**Modules**:
- `power_law_fitting.py` - Fit p_crit ~ L^b relationship
- `sample_requirements.py` - Effective samples needed
- `phase_diagrams.py` - Visualize failure/success regions

**Key Finding**: Power law p_crit ∝ L^b with b < 0 means larger problems are **more sample-efficient**.

### Phase 3: Theoretical Interpretation

**Question**: Why does the phase transition occur?

**Modules**:
- `eigenvalue_statistics.py` - Spectral gap preservation
- `coherence_analysis.py` - Eigenvector incoherence
- `theoretical_bounds.py` - Davis-Kahan theorem validation

**Key Finding**: Transition occurs when Davis-Kahan bound ||E||_F / δ ≈ O(1).

## Code Organization Principles

Each analysis module follows a **simple, focused pattern**:

```python
def compute_stats(df: pd.DataFrame) -> dict:
    """Pure computation. Returns statistics dict."""
    # Compute numerical results
    return {'stat1': value1, 'stat2': value2}

def create_plot(df: pd.DataFrame, output_path: Path):
    """Pure plotting. Saves one focused plot."""
    # Create and save visualization
    pass

def analyze(df: pd.DataFrame, output_dir: Path) -> dict:
    """Orchestrate: compute + plot + print. Returns stats."""
    stats = compute_stats(df)
    create_plot(df, output_dir / "plot.png")
    print(f"  Result: {stats['stat1']}")
    return stats
```

**Benefits**:
- Each file is 50-100 lines
- Clear single responsibility
- Easy to understand and modify
- Simple testing and debugging

## API Reference

### Shared Utilities

```python
from shared import load_results, get_output_dir, find_transition_point, find_all_transitions

# Load data
df = load_results()  # Loads results_grid.json with computed columns

# Get output directory
output_dir = get_output_dir("metrics_analysis")

# Find transitions
transition = find_transition_point(df, num_taxa=8192, sequence_length=10000)
all_transitions = find_all_transitions(df, threshold=90.0)
```

### Plotting Helpers

```python
from shared import plot_metric_faceted, plot_gap_vs_p

# Create faceted plot with partition agreement overlay
plot_metric_faceted(
    df,
    metric='spectral_gap_ratio',
    ylabel='Spectral Gap Ratio',
    title='Spectral Gap Ratio vs p',
    output_path=output_dir / "plot.png"
)

# Plot quality gap
plot_gap_vs_p(df, output_dir / "gap.png")
```

### Report Writing

```python
from shared import save_markdown_report

report = """# My Analysis
## Results
...
"""
save_markdown_report(report, output_dir / "summary.md")
```

## Key Metrics

**Performance Metrics**:
- `partition_agreement_M`: Agreement (%) when both use M (ideal)
- `partition_agreement_S`: Agreement (%) using M vs S_avg (realistic)
- `mean`: Sign agreement (%) - legacy metric

**Spectral Metrics**:
- `spectral_gap_ratio`: SpGap_LS / SpGap_LM (primary indicator)
- `rank_ratio_L_S`: Rank(L_S) / n (full rank = 1.0)
- `mean_frobenius_error`: ||M - S||_F
- `mean_coherence_L_S`: Eigenvector coherence
- `effective_samples`: p × n²

**Matrix Types**:
- `M`: Original similarity matrix
- `S`: Sampled version (averaged over bootstrap samples)
- `L_M`: Laplacian of M
- `L_S`: Laplacian of S (used for reconstruction)

## Extending the Analysis

### Adding a New Module

1. Create new file in appropriate directory:
```python
# metrics_analysis/my_new_analysis.py
def compute_stats(df):
    return {'my_stat': 42}

def create_plot(df, output_path):
    # Create visualization
    pass

def analyze(df, output_dir):
    stats = compute_stats(df)
    create_plot(df, output_dir / "my_plot.png")
    return stats
```

2. Import in `run.py`:
```python
from . import my_new_analysis

# In main():
my_stats = my_new_analysis.analyze(df, output_dir)
```

3. Use stats in report:
```python
write_report.generate_report(..., my_stats, output_dir)
```

### Adding a New Computed Metric

Edit `shared/data_loading.py`:

```python
def load_results(...):
    # ... existing code ...

    # Add your metric
    df['my_metric'] = df['col1'] / df['col2']

    return df
```

## Key Results Summary

### The Phase Transition Mechanism

1. **Failure regime** (p < p_critical):
   - SpGap_LS >> SpGap_LM (ratio > 100)
   - Sampling noise dominates signal
   - Partition agreement ≈ 50% (random)

2. **Transition** (p ≈ p_critical):
   - SpGap_LS ≈ SpGap_LM (ratio < 10)
   - Spectral structure preserved
   - Sharp jump to 100%

3. **Success regime** (p > p_critical):
   - Full spectral fidelity
   - Maintains 100% agreement

### Scaling Law

```
p_critical ≈ a × L^b
```

where b < 0, meaning larger matrices need **less sampling probability** to achieve the same performance.

### Theoretical Foundation

Davis-Kahan theorem:
```
||v - v_perturbed|| ≤ ||E||_F / δ
```

Transition occurs when ||E||_F / δ ≈ O(1).

## Citation

If you use this analysis framework, please cite:

```
[Your paper citation here]
```

## Contact

For questions or issues, please contact [your contact info].
