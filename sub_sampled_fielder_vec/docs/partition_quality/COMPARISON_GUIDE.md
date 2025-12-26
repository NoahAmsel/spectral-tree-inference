# Comparison Tool Guide

## Overview

The comparison tool (`compare_quality.py`) allows you to analyze partition quality across multiple configurations and generate comparative visualizations.

## Quick Start

```bash
cd sub_sampled_fielder_vec/partition_quality_analysis
python compare_quality.py --config configs/comparisons/mutation_rate_sweep.json
```

## Available Comparisons

### 1. Mutation Rate Sweep
```bash
python compare_quality.py --config configs/comparisons/mutation_rate_sweep.json
```
**Tests:** mutation rates 0.05, 0.1, 0.15, 0.2, 0.3  
**Purpose:** Find optimal mutation rate for clean partitions

### 2. Tree Model Comparison
```bash
python compare_quality.py --config configs/comparisons/tree_model_comparison.json
```
**Tests:** balanced_binary, lopsided, kingman_pure  
**Purpose:** Compare how different tree topologies affect partition quality

### 3. Sequence Length Scaling
```bash
python compare_quality.py --config configs/comparisons/seq_length_scaling.json
```
**Tests:** 500, 1000, 2000, 5000 base pairs  
**Purpose:** Determine minimum sequence length for reliable partitions

### 4. Sequence Model Comparison
```bash
python compare_quality.py --config configs/comparisons/seq_model_comparison.json
```
**Tests:** JC69, HKY, TN93  
**Purpose:** Compare different substitution models

### 5. Taxa Scaling
```bash
python compare_quality.py --config configs/comparisons/taxa_scaling.json
```
**Tests:** 64, 128, 256, 512 taxa  
**Purpose:** Study how partition quality scales with tree size

## Output Files

After running a comparison, you'll find:

```
results/comparisons/{comparison_name}/
├── config_0/              # Individual run results
│   └── results.json
├── config_1/
│   └── results.json
├── ...
└── comparison/            # Comparison outputs
    ├── summary.csv                              # All metrics in table
    ├── sigma2_vs_{parameter}.png                # Main result plot
    ├── fiedler_panels.png                       # Grid of Fiedler vectors
    ├── heatmap_grid.png                         # Side-by-side heatmaps
    ├── statistics_comparison.png                # Bar charts
    └── full_results.json                        # Complete results
```

## Understanding the Plots

### 1. σ₂ vs Parameter Plot
- **Main result:** Shows how partition quality changes with the swept parameter
- **Green/Orange/Red lines:** Quality thresholds
- **Lower is better:** σ₂ < 0.05 is excellent

### 2. Fiedler Panel Plot
- **Single figure with subplots:** Each configuration in its own axis
- **Look for:** Sharp transitions (good) vs gradual (poor)
- **Interpretation:** Steeper = cleaner partition; compare panels directly

### 3. Heatmap Grid
- **Side-by-side heatmaps:** One for each configuration
- **Block structure:** Clear 2x2 blocks = good partition
- **Red lines:** Show where the partition cuts

### 4. Statistics Comparison
- **Three bar charts:**
  1. Mean similarity (how similar sequences are on average)
  2. Similarity std dev (variability in similarity)
  3. Fiedler std dev (spread of Fiedler values)

## CSV Export

The `summary.csv` file contains all metrics in a table format for analysis in R, Excel, etc.

**Columns:**
- `comparison_name`: Name of the comparison
- `config_index`: Index in the sweep
- `swept_parameter`: Parameter that was varied
- `swept_value`: Value for this configuration
- `tree_model`, `num_taxa`, `seq_model`, `seq_len`, `mutation_rate`: Config details
- `s2`: **Main quality metric** (lower = better)
- `group_a_size`, `group_b_size`: Partition sizes
- `similarity_mean`, `similarity_std`: Similarity matrix stats
- `fiedler_mean`, `fiedler_std`: Fiedler vector stats

## Creating Custom Comparisons

### Basic Template

Create `configs/comparisons/my_comparison.json`:

```json
{
  "name": "my_comparison",
  "description": "Description of what you're testing",
  "base_config": {
    "tree": {
      "model": "balanced_binary",
      "params": {"num_taxa": 128, "edge_length": 1.0}
    },
    "sequence": {
      "model": "JC69",
      "len": 1000,
      "params": {"mutation_rate": 0.1, "num_classes": 4, "seed": 42}
    },
    "partition": {
      "num_gaps": 1,
      "min_split": 1
    }
  },
  "sweep": {
    "parameter": "sequence.params.mutation_rate",
    "values": [0.05, 0.1, 0.15, 0.2]
  },
  "output": {
    "dir": "results/comparisons/my_comparison",
    "save_individual_matrices": false
  }
}
```

### Sweep Parameters

You can sweep any nested parameter using dot notation:

**Examples:**
- `"tree.params.num_taxa"` - Number of taxa
- `"sequence.len"` - Sequence length
- `"sequence.params.mutation_rate"` - Mutation rate
- `"tree.model"` - Tree model (categorical)
- `"sequence.model"` - Sequence model (categorical)

### Model-Specific Overrides

For model comparisons (e.g., comparing JC69 vs HKY), you may need different parameters:

```json
{
  "sweep": {
    "parameter": "sequence.model",
    "values": ["JC69", "HKY"],
    "param_overrides": {
      "JC69": {
        "num_classes": 4
      },
      "HKY": {
        "kappa": 2.5,
        "stationary_freqs": [0.3, 0.2, 0.2, 0.3]
      }
    }
  }
}
```

## Command Line Options

```bash
python compare_quality.py --config CONFIG [OPTIONS]
```

**Options:**
- `--no-plots`: Skip generating plots (faster)
- `--no-csv`: Skip CSV export
- `--quiet`: Reduce output verbosity

**Examples:**
```bash
# Fast mode (no plots or CSV)
python compare_quality.py --config configs/comparisons/mutation_rate_sweep.json --no-plots --no-csv

# Quiet mode
python compare_quality.py --config configs/comparisons/tree_model_comparison.json --quiet
```

## Interpreting Results

### Quality Thresholds

| σ₂ Range | Quality | Recommendation |
|----------|---------|----------------|
| < 0.05 | Excellent | ✅ Perfect for tree inference |
| 0.05 - 0.1 | Good | ✅ Proceed with confidence |
| 0.1 - 0.3 | Moderate | ⚠️ May need more data or tuning |
| > 0.3 | Poor | ❌ Adjust configuration |

### Example Analysis

**Mutation Rate Sweep Results:**
```
mutation_rate=0.05 → σ₂=0.116 (Moderate)
mutation_rate=0.10 → σ₂=0.022 (Excellent)
mutation_rate=0.15 → σ₂=0.004 (Excellent)
mutation_rate=0.20 → σ₂=0.001 (Excellent)
```

**Interpretation:**
- Mutation rate 0.05 is too low (not enough signal)
- Rates 0.10-0.20 all give excellent partitions
- **Recommendation:** Use mutation rate ≥ 0.10

### Cross-Partition Histogram Meaning

The three histograms in individual plots show:

1. **Within Group A**: Similarities between taxa in same group (should be HIGH)
2. **Within Group B**: Similarities between taxa in same group (should be HIGH)
3. **Cross-Partition**: Similarities between different groups (should be LOW)

**Good partition:** Clear separation between within-group and cross-partition histograms  
**Poor partition:** All three histograms overlap

## Troubleshooting

### Configuration Errors

**Error:** `KeyError: 'num_taxa'`  
**Fix:** Make sure `num_taxa` is in `tree.params`

**Error:** `ValueError: Unknown tree model`  
**Fix:** Check model name spelling (case-sensitive)

### Runtime Issues

**Slow execution:**  
- Use fewer configurations
- Reduce `num_taxa` or `seq_len`
- Use `--no-plots` for faster runs

**Memory issues:**  
- Don't save matrices (`"save_individual_matrices": false`)
- Reduce number of taxa
- Run one config at a time

## Best Practices

1. **Start small:** Test with a few values first
2. **Check quality:** Look for σ₂ < 0.1 as a minimum
3. **Compare visually:** Use plots to understand differences
4. **Export CSV:** Analyze trends in Excel/R
5. **Document findings:** Note which configs work best for your data

## Integration with Main Analysis

Once you've identified optimal parameters via comparison:

1. Update your main config files
2. Run full experiments with optimal settings
3. Use comparison results to justify parameter choices in papers

## Examples Gallery

See the `results/comparisons/` directory for example outputs from each comparison type.

