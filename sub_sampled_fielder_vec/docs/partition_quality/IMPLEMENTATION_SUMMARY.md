# Implementation Summary: Partition Quality Analysis

## Overview

A comprehensive tool for analyzing the quality of spectral partitions (σ₂ metric) across different tree and sequence evolution models. This helps determine if a configuration has sufficient signal for meaningful phylogenetic inference.

## What Was Implemented

### 1. Directory Structure

```
partition_quality_analysis/
├── __init__.py                  # Module initialization
├── analyze_quality.py           # Main analysis script
├── configs/                     # Configuration files
│   ├── default.json            # Balanced binary + JC69
│   ├── kingman.json            # Coalescent + HKY
│   ├── lopsided.json           # Caterpillar tree
│   └── birth_death.json        # Birth-death + TN93
├── README.md                    # Documentation
├── USAGE_EXAMPLES.md           # Practical examples
└── results/                     # Output directory (created on run)
```

### 2. Core Components

#### A. Configuration System
- **JSON-based configs** with schema validation
- **CLI overrides** for quick parameter changes (`--num_taxa`, `--seq_len`, `--mutation_rate`)
- Easy to create new configurations
- All parameters clearly documented

#### B. Model Factories

**Tree Models Supported:**
- `balanced_binary`: Perfect binary tree (num_taxa must be power of 2)
- `lopsided`: Caterpillar/asymmetric tree
- `kingman_pure`: Pure Kingman coalescent (random)
- `kingman_mean`: Mean Kingman coalescent (deterministic)
- `birth_death`: Birth-death process

**Sequence Models Supported:**
- `JC69`: Jukes-Cantor (simplest)
- `HKY`: Hasegawa-Kishino-Yano (base frequencies + κ)
- `TN93`: Tamura-Nei (two transition rates)
- `T92`: Tamura 1992 (GC-content based)
- `GTR`: General Time Reversible (most general)
- `Gaussian`: Continuous states

#### C. Analysis Pipeline

The script performs:
1. **Tree Generation**: Using specified model and parameters
2. **Sequence Simulation**: Evolution on tree with mutation rate
3. **Similarity Matrix**: JC-corrected similarity (M)
4. **Laplacian**: L = D - M
5. **Fiedler Vector**: 2nd eigenvector of L
6. **Partition**: Using `partition_taxa` algorithm from spectraltree
7. **Quality Score**: σ₂ (second singular value of cross-partition block)

#### D. Visualization (3 plots)

1. **Similarity Heatmap**: Shows block structure when sorted by Fiedler vector
2. **Fiedler Vector Plot**: Shows where the "cut" happens
3. **Similarity Distributions**: Histograms comparing within-group vs cross-partition similarities

#### E. Result Saving

**JSON Results** (`results.json`):
```json
{
  "config": { ... },
  "metrics": {
    "s2": 0.022,
    "group_a_size": 64,
    "group_b_size": 64,
    "similarity_mean": 0.026,
    "similarity_std": 0.099,
    "fiedler_mean": 0.0,
    "fiedler_std": 0.088
  }
}
```

**Optional Matrix Saves** (`.npy` files):
- Similarity matrix
- Fiedler vector
- Partition boolean array

### 3. Usage

**Basic:**
```bash
python analyze_quality.py --config configs/default.json
```

**With Overrides:**
```bash
python analyze_quality.py --config configs/default.json --num_taxa 256 --seq_len 2000
```

**Fast Mode (no plots):**
```bash
python analyze_quality.py --config configs/default.json --no-plots
```

## Key Features

### ✓ Flexible Configuration
- JSON configs for reproducibility
- CLI overrides for experimentation
- Easy to add new configurations

### ✓ Comprehensive Factories
- All tree models from user's spec
- All sequence models from user's spec
- Proper parameter handling (`num_taxa`, `seq_len` injection)

### ✓ Rich Visualizations
- Heatmap shows block structure
- Fiedler plot shows partition quality
- Distributions show separation

### ✓ Robust Error Handling
- Handles partition failures gracefully
- Clear error messages
- Interpretation guidance in output

### ✓ Well Documented
- Main README with all parameters
- Usage examples for common scenarios
- Interpretation guidelines

## Testing

Tested successfully:
- ✓ Default config (128 taxa, JC69, 1000bp)
- ✓ CLI overrides (64 taxa, 500bp)
- ✓ Plot generation
- ✓ Result saving
- ✓ No linter errors

**Results from test run:**
- Configuration: Balanced binary tree, 128 taxa, JC69, 1000bp, μ=0.1
- σ₂ = 0.0224 (excellent partition quality)
- Perfect 64/64 split (as expected for balanced tree)
- Plots generated successfully

## How to Use for Research

### Pre-Experiment Configuration Screening

**Goal:** Determine if a configuration has sufficient signal before running expensive algorithms.

**Workflow:**
1. Create config for your planned experiment
2. Run partition quality analysis
3. Check σ₂ score:
   - **σ₂ < 0.1**: Good configuration, proceed
   - **0.1 < σ₂ < 0.3**: Marginal, may need more data
   - **σ₂ > 0.3**: Poor signal, adjust parameters

**What to adjust if σ₂ is too high:**
- Increase sequence length
- Decrease mutation rate (avoid saturation)
- Use more sophisticated sequence model (JC69 → HKY)
- Increase `num_gaps` for complex trees

### Parameter Exploration

**Example: Find minimum sequence length**
```bash
for len in 500 1000 2000 5000; do
  python analyze_quality.py --config configs/default.json --seq_len $len --no-plots
done
```

Check when σ₂ drops below your threshold.

### Model Comparison

Test different substitution models:
```bash
python analyze_quality.py --config configs/kingman.json  # HKY model
python analyze_quality.py --config configs/default.json  # JC69 model
```

Compare σ₂ values to see which model provides better signal.

## Integration with Existing Codebase

Uses `spectraltree` utilities:
- `spectraltree.balanced_binary`, `lopsided_tree`, etc. for tree generation
- `spectraltree.Jukes_Cantor`, `HKY`, etc. for sequence models
- `spectraltree.simulate_sequences` for evolution
- `spectraltree.similarities.JC_similarity_matrix` for similarity
- `spectraltree.spectral_tree_reconstruction.partition_taxa` for partitioning
- `spectraltree.spectral_tree_reconstruction.svd2` for quality metric

This ensures consistency with the main STDR algorithm.

## Example Configurations Provided

1. **default.json**: Balanced binary + JC69 (baseline)
2. **kingman.json**: Coalescent + HKY (realistic model)
3. **lopsided.json**: Caterpillar tree (worst case topology)
4. **birth_death.json**: Birth-death + TN93 (alternative tree model)

## Future Enhancements (Optional)

Potential additions:
- Batch processing of multiple configs
- Plotting σ₂ vs parameters (sweep visualization)
- Additional quality metrics (spectral gap, etc.)
- Support for real sequence data (not just simulated)
- Comparison across multiple replicates

## Summary

This tool provides a **pre-screening step** before running expensive phylogenetic inference algorithms. It answers the question:

> **"Does this configuration have enough signal to produce meaningful genetic results?"**

By computing the partition quality (σ₂), you can quickly identify parameter regimes where spectral methods will succeed or fail, saving computational resources and guiding experimental design.

