# Phase Transition Analysis

Analyze sample complexity scaling: prove leveraged sampling has better scaling than uniform as $n$ increases.

## Files

- **`phase_transition_utils.py`**: All analysis functions (sigmoid fitting, power law fitting, data loading)
- **`phase_transition_analysis.ipynb`**: Interactive notebook for visualization and analysis

## Quick Start

```bash
cd sub_sampled_fielder_vec/analysis/comparison
jupyter notebook phase_transition_analysis.ipynb
```

**Edit Cell 2** to set your comparison directory:
```python
COMP_DIR = Path("../../results/YOUR_COMPARISON_DIR")
```

Then run all cells.

## What It Does

### 1. Load Data
- Reads both `uniform/` and `leveraged/` results
- Extracts `(p, partition_agreement_M)` for each $n$

### 2. Compute $p^*$ (Critical Sampling Probability)
Two methods:
- **Sigmoid fit**: Fit logistic curve, extract $p^*$ where sigmoid = 95%
- **Discrete**: First $p$ where agreement ≥ 100%

### 3. Power Law Fit
Fit: $p^* = A \cdot n^\alpha$

- $\alpha < 0$: $p^*$ decreases as $n$ grows
- More negative $\alpha$ = better scaling

### 4. Generate Plots
- **Log-log plot**: $p^*$ vs $n$ with fitted curves
- Equations shown on plot
- Compare slopes: expect leveraged to have steeper (more negative) $\alpha$

## Expected Result

**Hypothesis**: Leveraged sampling has MORE NEGATIVE exponent $\alpha$

$$\alpha_{\text{leverage}} < \alpha_{\text{uniform}} < 0$$

This means: as $n$ grows, leveraged requires relatively fewer samples to achieve same recovery quality.

## Outputs

Saved to comparison directory:
- `phase_transition_scaling.png` - Main result (sigmoid-based)
- `phase_transition_discrete.png` - Discrete threshold comparison

## Functions (from utils.py)

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

## Troubleshooting

**"Fit failed"**: If sigmoid fit fails, check:
- Are there enough p-values? (need at least 5-10)
- Does agreement span from low to high? (need phase transition region)

**"All NaN"**: If no discrete threshold found:
- Agreement never reaches 100% for available p-values
- Use sigmoid-based $p^*$ instead (extrapolates)
