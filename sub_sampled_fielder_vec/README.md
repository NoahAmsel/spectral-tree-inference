# Sub-sampled Fiedler Vector Experiments

A modular framework for analyzing Fiedler vector quality under sub-sampling, using partition-based metrics that reflect how STDR (Spectral Top-Down Recovery) actually partitions phylogenetic trees.

## Scientific Context

**STDR** is a spectral algorithm for recovering latent tree models from observed sequence data. It works by:
1. Computing a similarity matrix M from observed sequences
2. Computing the Fiedler vector (2nd smallest eigenvector) of the Laplacian L(M)
3. Partitioning taxa using **gap-based thresholding** + SVD quality scoring
4. Recursively applying this process to build the tree

**This framework tests**: Does averaging Fiedler vectors from subsampled data preserve STDR's partition quality?

**Core question**: As we increase n (number of taxa), can we use fewer matrix entries (lower p)? In other words, does the total number of entries needed scale sub-quadratically with n?

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

### Option 1: Interactive Launcher (Recommended)

The interactive launcher provides a menu-driven interface with persistent caching and easy re-runs:

```bash
python scripts/interactive_run.py
```

**Features**:
- 🔄 Re-run last configuration
- 💾 Select from cached matrices (instant loading!)
- 🆕 Create new matrix configuration
- 📊 Automatic plotting and diagnostics

**Example workflow**:
1. Choose "Create new matrix configuration"
2. Enter parameters (or press Enter for defaults)
3. Experiment runs and saves to `results/<timestamp>-<run_name>/`
4. Next time: Select cached matrix for instant loading!

### Option 2: Script-Based Configuration

Edit `scripts/run_experiment.py` and update `SWEEP_CONFIG`:

```python
SWEEP_CONFIG = {
    "tree_model": "balanced_binary",
    "taxa_values": [1024, 2048],
    "sequence_length": 10000,
    "mutation_rate": 0.1,
    "p_values": list(np.logspace(-4, 0, 20)),
    "bootstrap_reps": 10,
    "run_name_prefix": "my_experiment",
    "sampling_method": "leveraged",  # or "uniform"
    "log_sampling_diagnostics": True,  # Enable diagnostic logging
}
```

Launch experiments:
```bash
python scripts/run_experiment.py
```

### Results Structure

```
results/<timestamp>-<run_name>/
├── n{taxa}_L{seq_len}/
│   ├── results.json              # Main metrics
│   ├── config.json               # Configuration used
│   ├── fiedler_vectors.png       # Fiedler vector plots
│   ├── partition_agreement.png   # Agreement curves
│   ├── sampling_data/            # Diagnostic data (if enabled)
│   │   ├── p_0.1438.npz         # Leverage scores + sampling probs
│   │   └── ...
│   └── experiment.log
```

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
