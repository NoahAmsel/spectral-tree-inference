# Partition Quality Analysis

This module analyzes the quality of spectral partitions (σ₂ metric) for various tree and sequence evolution models.

## Overview

The analysis pipeline:
1. Generates a phylogenetic tree using specified model
2. Simulates sequence evolution on the tree
3. Computes similarity matrix M from sequences
4. Computes Laplacian L = D - M
5. Extracts Fiedler vector (2nd eigenvector of L)
6. Partitions taxa using `partition_taxa` algorithm
7. Computes partition quality score σ₂ (lower = better)
8. Generates visualizations

## Two Tools Available

### 1. Single Configuration Analysis (`analyze_quality.py`)
Test one configuration at a time, get detailed output.

### 2. Multi-Configuration Comparison (`compare_quality.py`)  
**NEW!** Compare multiple configurations, generate comparative plots and CSV summaries.

## Usage

### Single Configuration Analysis

```bash
cd sub_sampled_fielder_vec/partition_quality_analysis
python analyze_quality.py --config configs/default.json
```

### Multi-Configuration Comparison

```bash
python compare_quality.py --config configs/comparisons/mutation_rate_sweep.json
```

**See [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md) for detailed comparison tool documentation.**

### Override Parameters

```bash
python analyze_quality.py --config configs/default.json --num_taxa 256 --seq_len 2000
```

### Skip Plots

```bash
python analyze_quality.py --config configs/default.json --no-plots
```

## Configuration

### Configuration File Format

```json
{
  "tree": {
    "model": "balanced_binary",
    "params": {
      "num_taxa": 128,
      "edge_length": 1.0
    }
  },
  "sequence": {
    "model": "JC69",
    "len": 1000,
    "params": {
      "mutation_rate": 0.1,
      "num_classes": 4,
      "seed": 42
    }
  },
  "partition": {
    "num_gaps": 1,
    "min_split": 1
  },
  "output": {
    "dir": "results/my_experiment",
    "save_matrices": false
  }
}
```

## Tree Models

| Model | Description | Required Parameters |
|-------|-------------|---------------------|
| `balanced_binary` | Perfectly balanced binary tree | `num_taxa` (must be power of 2), `edge_length` |
| `lopsided` | Caterpillar/asymmetric tree | `num_taxa`, `edge_length` |
| `kingman_pure` | Coalescent process (random) | `num_taxa`, `pop_size`, `seed` |
| `kingman_mean` | Coalescent (deterministic means) | `num_taxa`, `pop_size` |
| `birth_death` | Birth-death process | `num_taxa`, `birth_rate`, `death_rate` |

## Sequence Evolution Models

| Model | Description | Parameters |
|-------|-------------|------------|
| `JC69` | Jukes-Cantor (all substitutions equal) | `num_classes`, `seed` |
| `HKY` | Different base frequencies + transition/transversion ratio | `kappa`, `stationary_freqs`, `seed` |
| `TN93` | Two transition rate parameters | `stationary_freqs`, `kappa1`, `kappa2`, `seed` |
| `T92` | Tamura 1992 (GC content based) | `theta`, `kappa1`, `kappa2`, `seed` |
| `GTR` | General Time Reversible | `stationary_freqs`, `transition_rates`, `seed` |
| `Gaussian` | Continuous states | `w`, `b`, `seed` |

## Outputs

### Console Output
- Configuration summary
- Similarity matrix statistics
- Fiedler vector statistics
- Partition sizes
- **σ₂ quality score** (main metric)

### Saved Files

In `{output.dir}/`:
- `results.json`: Metrics and configuration
- `similarity_matrix.npy`: Similarity matrix (if `save_matrices: true`)
- `fiedler_vector.npy`: Fiedler vector (if `save_matrices: true`)
- `partition.npy`: Boolean partition array (if `save_matrices: true`)

In `{output.dir}/plots/`:
- `similarity_heatmap.png`: Heatmap of M sorted by Fiedler vector
- `fiedler_vector.png`: Plot of sorted Fiedler values
- `similarity_distributions.png`: Histograms of within/cross-partition similarities

## Interpreting Results

### σ₂ (Partition Quality)
- **σ₂ ≈ 0**: Perfect partition (ideal)
- **σ₂ < 0.1**: Good partition
- **σ₂ > 0.5**: Poor partition
- **σ₂ = ∞**: Partition failed

Lower σ₂ means the cross-partition similarity matrix has lower rank (cleaner separation).

### Visualization Tips
- **Heatmap**: Look for block structure (high similarity within blocks, low across)
- **Fiedler plot**: Sharp transitions indicate clear partitions
- **Distributions**: Separation between within-group vs cross-partition histograms

## Example Configs

See `configs/` directory:
- `default.json`: Balanced binary tree with JC69
- `kingman.json`: Coalescent tree with HKY model
- `lopsided.json`: Asymmetric tree
- `birth_death.json`: Birth-death with TN93

## Notes

- `num_taxa` must be specified in tree parameters
- `seq_len` (sequence length) must be specified at sequence level
- `mutation_rate` controls branch length scaling
- All models use random seed for reproducibility

