# Run Experiment Script

Documentation for `sub_sampled_fielder_vec/scripts/run_experiment.py`

## Overview

The `run_experiment.py` script is the main entry point for running bootstrap sweep experiments that evaluate how well sub-sampled similarity matrices can recover phylogenetic tree structure.

## What It Runs

For each tree size (n) and sequence length (L) combination, the experiment:

1. **Generates a phylogenetic tree** using the Kingman coalescent model
2. **Simulates DNA sequences** along the tree branches
3. **Computes a reference Fiedler vector** from the full similarity matrix
4. **Sweeps over sub-sampling probabilities** — for each p-value (20 log-spaced values from 10⁻⁴ to 1), it:
   - Computes Fiedler vectors from sub-sampled similarity matrices
   - Runs multiple bootstrap replicates
   - Measures how well the sub-sampled Fiedler vectors recover the true tree partition

## Output per p-value

For each sub-sampling probability p, the following metrics are recorded:

| Metric | Description |
|--------|-------------|
| `sign_agreement` | Agreement between sub-sampled and reference Fiedler vector signs |
| `partition_agreement_M` | Partition agreement using method M |
| `partition_agreement_S` | Partition agreement using method S |
| `dot_products` | Dot product between sub-sampled and reference Fiedler vectors |
| `sigma2_avg_M` | Average σ² estimate (method M) |
| `sigma2_avg_S` | Average σ² estimate (method S) |
| `partition_split_M` | Partition split ratio (method M) |
| `partition_split_S` | Partition split ratio (method S) |

## Output Files

Results are saved to a timestamped directory under `results/`:

```
results/<timestamp>-<run_name>/
├── n<taxa>_L<seq_len>/
│   ├── results.json          # All metrics for all p-values
│   ├── plot_single.png       # Partition agreement vs p plot
│   ├── fiedler_vectors.png   # Fiedler vector visualization
│   ├── fiedler_ref.npy       # Reference Fiedler vector
│   └── tree_plots/           # Tree partition visualizations
├── sweep_config.json         # Configuration used for the sweep
├── results_grid_merged.json  # Merged results across all n, L
└── partition_agreement.png   # Combined plot across taxa sizes
```

## Configuration

Edit the `SWEEP_CONFIG` dictionary in the script to customize:

```python
SWEEP_CONFIG = {
    "tree_model": "kingman",
    "taxa_values": [500, 1000, 3000, 5000, 7000, 10000],
    "sequence_length_values": [10000],
    "mutation_rate": 0.1,
    "bootstrap_reps": 5,
    "num_workers": 8,
    "p_values": WIDE_SWEEP_P_VALUES,  # 20 log-spaced values
    "tree_params": {"pop_size": 1.0},
    # ... other options
}
```
