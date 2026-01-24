# Sub-sampled Fiedler Vector Experiments

A modular framework for analyzing Fiedler vector quality under sub-sampling, using partition-based metrics that reflect how STDR (Spectral Top-Down Recovery) actually partitions phylogenetic trees.

## Scientific Context

**STDR** is a spectral algorithm for recovering latent tree models from observed sequence data. It works by:
1. Computing a similarity matrix M from observed sequences
2. Computing the Fiedler vector (2nd smallest eigenvector) of the Laplacian L(M)
3. Partitioning taxa using **gap-based thresholding** + SVD quality scoring
4. Recursively applying this process to build the tree

**This framework tests**: Does averaging Fiedler vectors from subsampled data preserve STDR's partition quality?

## How Experiments Work

**Process**:
1. Generate sequences for given (n_taxa, seq_len)
2. Compute reference Fiedler vector from full matrix M (p=1.0)
3. For each p-value:
   - **Bootstrap loop** (for K replicates):
     * Subsample similarity matrix: M → S
     * **Update streaming average**: S_avg (using Welford's algorithm - no storage!)
     * Compute Fiedler vector from L(S)
     * Align to reference using dot product
     * Collect aligned vector
   - **After loop**:
     * Average aligned vectors and normalize
     * Compute partition_agreement_M: compare partition_taxa(v_full, M) vs partition_taxa(v_avg, M)
     * Compute partition_agreement_S: compare partition_taxa(v_full, M) vs partition_taxa(v_avg, S_avg)
     * Compute dot_product and sign_agreement
4. Return all metrics and per-p-value diagnostics (σ₂ statistics, partition splits, provenance flags)

The results are plots of agreement % as a function of sampling rate p. We want to have a diagnostic notebook linear algebra wise "what happen here".

## Quick Start

### 1. Configure a Sweep
Edit `scripts/run_experiment.py` and update `SWEEP_CONFIG`:

```python
SWEEP_CONFIG = {
    "tree_model": "balanced_binary",
    "taxa_values": [1024, 2048],
    "sequence_length": 1000,
    "mutation_rate": 0.1,
    "p_values": [0.01, 0.1, 0.5, 1.0],
    "bootstrap_reps": 50,
    "run_name_prefix": "my_experiment",
}
```

### 2. Launch Experiments
```bash
python scripts/run_experiment.py
```

Results are saved to `results/<timestamp>-<run_name>/`

### 3. Inspect Results
- JSON tables: `results/<ts-run_name>/results*.json`
- Plots: `plot_single.png`, `plot_multi_taxa.png`
- Reference vectors: `fiedler_ref*.npy`

## Documentation

For detailed information, see the `docs/` directory:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Codebase structure, entry points, and data flow (start here for AI agents)
- **[METRICS.md](docs/METRICS.md)** - All metrics and data formats
- **[CONFIGURATION.md](docs/CONFIGURATION.md)** - Config system and parameters
- **[LEVERAGED_SAMPLING.md](docs/LEVERAGED_SAMPLING.md)** - Leveraged matrix completion sampling guide
- **[ANALYSIS_GUIDES.md](docs/ANALYSIS_GUIDES.md)** - How to use analysis notebooks and tools

## Citation

If you use this framework, please cite the STDR paper:

**Spectral top-down recovery of latent tree models**  
Roch, S. (2006)
