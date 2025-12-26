# Quick Start Guide

## TL;DR

```bash
cd sub_sampled_fielder_vec/partition_quality_analysis
python analyze_quality.py --config configs/default.json
```

Check the **σ₂** value in the output:
- **σ₂ < 0.1**: ✅ Good configuration
- **σ₂ > 0.3**: ❌ Poor configuration, adjust parameters

## Common Use Cases

### 1. Test if your configuration works
```bash
python analyze_quality.py --config configs/default.json
```

### 2. Quickly test different parameters
```bash
python analyze_quality.py --config configs/default.json --num_taxa 256 --seq_len 2000
```

### 3. Fast testing without plots
```bash
python analyze_quality.py --config configs/default.json --no-plots
```

## Create Your Own Config

Copy `configs/default.json` and modify:

```json
{
  "tree": {
    "model": "balanced_binary",
    "params": {
      "num_taxa": 128,           // ← Change this
      "edge_length": 1.0
    }
  },
  "sequence": {
    "model": "JC69",             // ← Or try "HKY", "TN93"
    "len": 1000,                 // ← Sequence length
    "params": {
      "mutation_rate": 0.1       // ← Key parameter!
    }
  },
  "output": {
    "dir": "results/my_test"     // ← Output directory
  }
}
```

## Interpreting σ₂

| σ₂ Range | Quality | Action |
|----------|---------|--------|
| < 0.05 | Excellent | ✅ Perfect, proceed with confidence |
| 0.05 - 0.1 | Good | ✅ Proceed |
| 0.1 - 0.3 | Moderate | ⚠️ Consider more data |
| > 0.3 | Poor | ❌ Adjust parameters |
| ∞ | Failed | ❌ Configuration invalid |

## When to Adjust Parameters

### If σ₂ is too high:

1. **Increase sequence length** (`seq_len`): More data = better signal
2. **Decrease mutation rate**: High rates cause saturation
3. **Try different tree model**: Some topologies are harder
4. **Use more sophisticated sequence model**: HKY instead of JC69

### Example Fix:

```bash
# Original: σ₂ = 0.4 (poor)
python analyze_quality.py --config configs/default.json

# Fix 1: More data
python analyze_quality.py --config configs/default.json --seq_len 5000

# Fix 2: Lower mutation rate  
python analyze_quality.py --config configs/default.json --mutation_rate 0.05
```

## Available Configs

- `configs/default.json` - Balanced tree + JC69 (baseline)
- `configs/kingman.json` - Coalescent tree + HKY (realistic)
- `configs/lopsided.json` - Caterpillar tree (challenging)
- `configs/birth_death.json` - Birth-death + TN93

## Output Files

After running, check:

1. **Console output**: Main σ₂ metric
2. `results/{experiment}/results.json`: All metrics
3. `results/{experiment}/plots/`: Visualizations
   - `similarity_heatmap.png`: Block structure
   - `fiedler_vector.png`: Partition visualization
   - `similarity_distributions.png`: Signal quality

## Need Help?

- See `README.md` for full documentation
- See `USAGE_EXAMPLES.md` for detailed examples
- See `IMPLEMENTATION_SUMMARY.md` for technical details

## Tree Models

- `balanced_binary`: Perfect binary tree (num_taxa = power of 2)
- `lopsided`: Caterpillar tree (asymmetric)
- `kingman_pure`: Coalescent (random)
- `kingman_mean`: Coalescent (deterministic)
- `birth_death`: Birth-death process

## Sequence Models

- `JC69`: Simplest (all substitutions equal)
- `HKY`: Base frequencies + transition/transversion ratio
- `TN93`: Two transition rates
- `T92`: GC-content based
- `GTR`: Most general reversible model

