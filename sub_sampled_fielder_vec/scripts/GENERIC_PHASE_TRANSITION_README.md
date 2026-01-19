# Generic Phase Transition Notebook

**Flexible notebook** to compare phase transitions across **any set of experiment runs**.

## File: `phase_transition_generic.ipynb`

## Use Cases

1. **Compare tree models**: Kingman vs Balanced Binary vs Lopsided
2. **Compare sampling methods**: Uniform vs Leveraged vs Adaptive
3. **Compare parameters**: Different θ values, different ranks, etc.
4. **Compare anything**: Any experiments with multiple n values

## Quick Start

1. **Open notebook**:
   ```bash
   cd sub_sampled_fielder_vec/scripts
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

## Configuration Format

Each run needs:
- **`path`**: Results directory containing `n{X}_L{Y}/results.json` subdirectories
- **`label`**: Display name for plots/tables
- **`color`** (optional): Hex color code (auto-assigned if omitted)
- **`marker`** (optional): Matplotlib marker style (auto-assigned if omitted)

## Example Configurations

### Example 1: Compare 3 Tree Models
```python
RUNS_CONFIG = [
    {
        "path": "../results/kingman_sweep",
        "label": "Kingman",
        "color": "#2E86AB",
        "marker": "o"
    },
    {
        "path": "../results/balanced_binary_sweep",
        "label": "Balanced Binary",
        "color": "#A23B72",
        "marker": "s"
    },
    {
        "path": "../results/lopsided_sweep",
        "label": "Lopsided",
        "color": "#F18F01",
        "marker": "^"
    },
]
```

### Example 2: Compare Sampling Methods
```python
RUNS_CONFIG = [
    {
        "path": "../results/20260117.../uniform",
        "label": "Uniform",
    },
    {
        "path": "../results/20260117.../leveraged",
        "label": "Leveraged (θ=0.3)",
    },
]
```

### Example 3: Parameter Sweep
```python
RUNS_CONFIG = [
    {"path": "../results/theta_0.2", "label": "θ=0.2"},
    {"path": "../results/theta_0.5", "label": "θ=0.5"},
    {"path": "../results/theta_0.7", "label": "θ=0.7"},
]
```

## Output

### Plots
- **Left**: Sigmoid-based p* (95% agreement) with power law fits
- **Right**: Discrete p* (100% agreement) with connecting lines
- Saved as: `phase_transition_comparison.png`

### Tables
- P* values for each run and n
- Power law equations and exponents
- Best/worst scaling summary

## Requirements

**Each run directory must contain**:
```
my_run/
├── n512_L10000/
│   └── results.json
├── n1024_L10000/
│   └── results.json
└── n2048_L10000/
    └── results.json
```

**Minimum**: ≥2 different n values per run

## Tips

1. **Auto-colors**: Omit `color` and `marker` - they'll be assigned automatically
2. **Many runs**: Notebook supports any number of runs (limited only by plot clarity)
3. **Missing data**: Runs with missing/invalid paths are skipped with warnings
4. **Re-run**: Just edit Cell 1 and re-run all cells to compare different sets

## Difference from Other Tools

| Tool | Purpose |
|------|---------|
| `phase_transition_generic.ipynb` | **Any N runs** - flexible comparison |
| `phase_transition_analysis.ipynb` | Method comparison (uniform vs leveraged) |
| `plot_phase_transition.py` | Single run analysis (CLI tool) |
