# Partition Quality Analysis - Usage Examples

This document provides practical examples for running partition quality analyses.

## Quick Start

```bash
cd sub_sampled_fielder_vec/partition_quality_analysis
python analyze_quality.py --config configs/default.json
```

## Example 1: Basic Balanced Binary Tree with JC69

```bash
python analyze_quality.py --config configs/default.json
```

**What this does:**
- 128 taxa balanced binary tree
- JC69 substitution model
- 1000 bp sequences
- Mutation rate 0.1
- Generates plots and saves results

**Expected σ₂:** ~0.02-0.03 (very good partition)

## Example 2: Override Parameters

```bash
python analyze_quality.py --config configs/default.json --num_taxa 256 --seq_len 2000
```

**What this does:**
- Uses default config but overrides to 256 taxa and 2000bp
- Good for testing scalability

## Example 3: Lopsided (Caterpillar) Tree

```bash
python analyze_quality.py --config configs/lopsided.json
```

**What this does:**
- Asymmetric tree structure
- Uses num_gaps=3 to find better partition
- Tests algorithm on challenging topology

**Expected σ₂:** Higher than balanced tree (worse partition due to asymmetry)

## Example 4: Coalescent Model (Kingman) with HKY

```bash
python analyze_quality.py --config configs/kingman.json
```

**What this does:**
- Realistic coalescent tree
- HKY model with transition/transversion bias
- Saves similarity matrix for inspection

## Example 5: No Plots (Fast Mode)

```bash
python analyze_quality.py --config configs/default.json --no-plots
```

**What this does:**
- Skips plot generation
- Faster for parameter sweeps

## Example 6: Birth-Death Process

```bash
python analyze_quality.py --config configs/birth_death.json
```

**What this does:**
- Birth-death tree model
- TN93 substitution model (two transition rates)

## Creating Custom Configurations

### Template

Create `configs/my_experiment.json`:

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

Then run:
```bash
python analyze_quality.py --config configs/my_experiment.json
```

## Parameter Sweep Example

Test multiple mutation rates:

```bash
for mu in 0.05 0.1 0.15 0.2; do
  python analyze_quality.py --config configs/default.json --mutation_rate $mu --no-plots
  echo "Completed mutation rate: $mu"
done
```

## Interpreting Results

### Console Output

Look for the **PARTITION QUALITY (σ₂)** value:

- **σ₂ < 0.05**: Excellent partition (clear block structure)
- **σ₂ < 0.1**: Good partition (usable for tree inference)
- **σ₂ < 0.3**: Moderate partition (may need more data)
- **σ₂ > 0.5**: Poor partition (insufficient signal)
- **σ₂ = ∞**: Partition failed (unable to split)

### Results Files

Check `results/{experiment_name}/results.json`:

```json
{
  "metrics": {
    "s2": 0.022,              // Main quality metric
    "group_a_size": 64,       // Partition sizes
    "group_b_size": 64,
    "similarity_mean": 0.026, // Average similarity
    "similarity_std": 0.099   // Similarity variance
  }
}
```

### Plots

1. **similarity_heatmap.png**: Look for 2x2 block structure
2. **fiedler_vector.png**: Look for sharp transition (good) vs gradual (poor)
3. **similarity_distributions.png**: Separation between within-group and cross-partition

## Common Configuration Patterns

### High Mutation Rate (Short Branches)

```json
{
  "sequence": {
    "model": "JC69",
    "len": 500,
    "params": {
      "mutation_rate": 0.5  // High rate
    }
  }
}
```

**Expected:** Higher σ₂ (poor partition due to saturation)

### Long Sequences (More Signal)

```json
{
  "sequence": {
    "model": "JC69",
    "len": 5000,  // Long sequences
    "params": {
      "mutation_rate": 0.1
    }
  }
}
```

**Expected:** Lower σ₂ (better partition with more data)

### Many Taxa (Scalability Test)

```json
{
  "tree": {
    "model": "balanced_binary",
    "params": {
      "num_taxa": 512  // Must be power of 2
    }
  }
}
```

**Note:** Computational cost scales as O(n²) for similarity matrix

### HKY Model (Realistic Substitution)

```json
{
  "sequence": {
    "model": "HKY",
    "len": 1000,
    "params": {
      "mutation_rate": 0.15,
      "kappa": 2.5,  // Transition/transversion ratio
      "stationary_freqs": [0.3, 0.2, 0.2, 0.3],  // A, C, G, T
      "seed": 42
    }
  }
}
```

## Troubleshooting

### Partition Failed (σ₂ = ∞)

**Causes:**
- Mutation rate too high (saturation)
- Sequences too short (insufficient signal)
- Tree topology extremely unbalanced

**Solutions:**
- Increase `num_gaps` in partition settings
- Decrease `min_split` (with caution)
- Increase sequence length
- Decrease mutation rate

### Poor Partition Quality

**Causes:**
- Short sequences
- High mutation rate
- Lopsided tree structure

**Solutions:**
- Increase sequence length
- Tune mutation rate to ~0.1-0.2
- Use `num_gaps > 1` for complex topologies

## Advanced: Comparing Configurations

Create a comparison script:

```bash
#!/bin/bash
# compare_models.sh

models=("JC69" "HKY" "TN93")
for model in "${models[@]}"; do
  python analyze_quality.py \
    --config configs/${model,,}.json \
    --no-plots
  grep '"s2"' results/${model,,}_experiment/results.json
done
```

This helps identify which models/parameters give meaningful genetic results.

