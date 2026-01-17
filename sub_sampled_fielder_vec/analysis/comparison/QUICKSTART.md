# Quick Start: Method Comparison

Compare uniform vs leveraged sampling on identical matrices.

## 1. Quick Test (2-5 min)

```bash
cd sub_sampled_fielder_vec
python analysis/comparison/test_compare.py
```

This runs both methods on n=500 with 3 bootstrap reps.

## 2. View Results

```bash
# Set comparison dir (update timestamp)
COMP_DIR="results/20260117-114348-method_comparison"

# Merge results
python scripts/merge_results.py $COMP_DIR/uniform
python scripts/merge_results.py $COMP_DIR/leveraged

# Generate comparison plots
python analysis/comparison/plot_comparison.py $COMP_DIR

# Open plots
open $COMP_DIR/comparison_plots/partition_agreement_comparison.png
```

## 3. Full Comparison (several hours)

```bash
cd sub_sampled_fielder_vec
python analysis/comparison/compare_methods.py
```

**Parameters:**
- n_taxa: [500, 1k, 3k, 5k, 7k, 10k]
- p_values: logspace(-4, 0, 20)
- bootstrap_reps: 20
- Shared seed: 42 (ensures identical data)

## 4. Customize

Edit `compare_methods.py`:
```python
COMPARISON_CONFIG = {
    "taxa_values": [500, 1000],  # Your n values
    "p_values": [0.01, 0.1, 1.0],  # Your p values
    "bootstrap_reps": 10,
    "seed": 42,  # KEEP SAME for fair comparison
}
```

## Key Files

- **compare_methods.py**: Full comparison (edit config)
- **test_compare.py**: Quick test
- **plot_comparison.py**: Generate plots from merged results
- **README.md**: Detailed documentation

## Output Structure

```
results/{timestamp}-method_comparison/
├── uniform/              # Uniform sampling results
│   ├── n500_L10000/
│   ├── ...
│   └── results_grid_merged.json (after merge)
├── leveraged/            # Leveraged sampling results
│   ├── n500_L10000/
│   ├── ...
│   └── results_grid_merged.json (after merge)
├── comparison_plots/     # Generated plots
│   └── partition_agreement_comparison.png
└── comparison_config.json
```
