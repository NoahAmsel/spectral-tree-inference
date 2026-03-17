# Codebase Architecture

This document provides a structural overview of the codebase optimized for AI agents and automated workflows.

## Directory Map

```
sub_sampled_fielder_vec/
├── README.md                    # Main entry point - concise overview
├── scripts/                      # CLI entry points and utilities
│   ├── interactive_run.py       # ⭐ Interactive launcher with caching (recommended!)
│   ├── run_experiment.py        # Script-based experiment launcher
│   ├── check_partition_quality.py  # Inspect σ₂-based partition quality
│   └── validation/              # Targeted validation utilities
├── src/                          # Core package code (importable)
│   ├── config/                  # Configuration system (Pydantic models)
│   │   ├── base_config.py        # StructuredConfig, TreeConfig, SequenceConfig, etc.
│   │   ├── presets.py           # Helpers: custom_config(), predefined sweeps
│   │   └── sweeps.py            # Predefined sweep builders
│   ├── models/                   # Tree and sequence model abstractions
│   │   ├── tree_models.py       # Tree generation (balanced_binary, kingman, etc.)
│   │   └── sequence_models.py   # Sequence models (JC69, HKY, GTR, etc.)
│   ├── core/                    # Numerical kernels and matrix operations
│   │   ├── similarity_builder.py  # Build similarity matrices (uniform/leveraged sampling)
│   │   ├── fiedler_computer.py  # Fiedler vector computation (sparse/dense)
│   │   ├── metric_computer.py   # Matrix metrics (operator norm, coherence, etc.)
│   │   └── sampling/            # Sampling methods
│   │       ├── base.py          # BaseSampler interface
│   │       ├── uniform/         # Uniform matrix entry sampling
│   │       └── leveraged/       # Leveraged matrix completion sampling
│   │           ├── sampler.py             # LeveragedSampler (main class)
│   │           ├── compute_leverage_scores.py  # Phase 1: estimate importance
│   │           ├── compute_sampling_probabilities.py  # Phase 2: compute probs
│   │           ├── ialm_solve.py          # Phase 3: matrix recovery
│   │           └── ...                    # Supporting utilities
│   ├── runners/                 # Experiment orchestration
│   │   ├── experiment_runner.py # Main ExperimentRunner class
│   │   ├── bootstrap_sweep.py   # Core bootstrap loop logic
│   │   ├── experiment_runner_utils.py  # Config extraction, directory setup
│   │   └── parallel_bootstrap.py  # Parallel execution utilities
│   ├── utils/                   # Shared utilities
│   │   ├── metrics.py           # Partition agreement, dot product, sign agreement
│   │   ├── summaries.py         # Result saving (JSON tables)
│   │   ├── plotting.py          # Visualization functions
│   │   ├── logging.py           # Logging utilities
│   │   ├── sampling_logger.py   # 🆕 Diagnostic data logging for leverage sampling
│   │   ├── persistent_cache.py  # 🆕 Disk-based caching for matrices
│   │   ├── interactive_ui.py    # 🆕 Menu system for interactive launcher
│   │   └── threshold_utils.py   # Partition thresholding helpers
│   └── cache/                   # 🆕 Persistent cache storage (gitignored)
│       ├── n1024_L10000_mu0.100_balanced_binary_JC69/
│       │   ├── similarity_matrix.npz
│       │   ├── observations.npz
│       │   ├── tree.npz
│       │   └── fiedler_ref.npz
│       └── ...
├── configs/                      # Saved JSON configurations
│   ├── presets/                 # Checked-in preset configs
│   └── custom/                  # User-provided configs (gitignored)
├── last_run.json                # 🆕 Last run configuration (for re-runs)
├── docs/                         # Documentation (this directory)
├── examples/                     # Small runnable code samples
├── tests/                        # Unit and integration tests
├── analysis/                     # Analysis notebooks and tools
│   ├── notebooks/               # 🆕 Interactive exploration notebooks
│   │   └── leverage_sampling_explorer.ipynb  # Visualize leverage diagnostics
│   ├── comparison/              # Method comparison (uniform vs leveraged)
│   ├── leveraged_sampling_analysis/  # Leveraged sampling diagnostics
│   ├── spectral_analysis/       # Spectral analysis frameworks
│   └── generic_analysis/        # Generic analysis utilities
└── results/                      # Auto-generated experiment artifacts (gitignored)
    └── <timestamp>-<run_name>/
        └── n{taxa}_L{seq_len}/
            ├── results.json
            ├── config.json
            ├── experiment.log
            ├── fiedler_vectors.png
            ├── partition_agreement.png
            └── sampling_data/  # 🆕 Diagnostic data (if log_sampling_diagnostics=True)
                ├── p_0.1438.npz  # leverage_scores + phase2_probs_sampled
                ├── p_0.2336.npz
                └── ...
```

## Key Entry Points

### For Running Experiments
- **`scripts/interactive_run.py`** - 🆕 **Recommended**: Interactive menu-driven launcher with persistent caching
  - Re-run last configuration with one keystroke
  - Select from cached matrices for instant loading
  - Create new configurations interactively
- **`scripts/run_experiment.py`** - Script-based entry point. Edit `SWEEP_CONFIG` dictionary to configure experiments.

### For Understanding Configuration
- **`src/config/base_config.py`** - All Pydantic models (`StructuredConfig`, `TreeConfig`, `SequenceConfig`, `ExperimentConfig`, `SamplingConfig`)
- **`src/config/presets.py`** - Helper function `custom_config()` to build validated configs

### For Core Experiment Logic
- **`src/runners/experiment_runner.py`** - `ExperimentRunner` class orchestrates execution
- **`src/runners/bootstrap_sweep.py`** - `sweep_for_params()` function contains the main bootstrap loop

### For Metrics Computation
- **`src/utils/metrics.py`** - `compute_partition_agreement()`, `compute_fiedler_dot_product()`, etc.

### For Analysis
- **`analysis/comparison/compare_methods.py`** - Compare uniform vs leveraged sampling
- **`analysis/leveraged_sampling_analysis/`** - Diagnostic framework for leveraged sampling

## Data Flow

```mermaid
flowchart TD
    Start([scripts/run_experiment.py]) --> Config[Edit SWEEP_CONFIG<br/>custom_config(...)]
    Config --> Runner[ExperimentRunner.__init__]
    
    Runner --> Setup[Setup Phase]
    Setup --> Seed[set_seed]
    Setup --> Dir[make_run_dir]
    Setup --> SaveCfg[StructuredConfig.to_json]
    
    Seed --> Loop[Iterate n_taxa × seq_len]
    Dir --> Loop
    SaveCfg --> Loop
    
    Loop --> Sweep[sweep_for_params]
    
    Sweep --> GenSeq[generate_sequences]
    GenSeq --> FiedlerRef[Compute Reference Fiedler<br/>p=1.0 full matrix]
    FiedlerRef --> Bootstrap[Bootstrap Loop per p-value]
    
    Bootstrap --> SubsampleS[Subsample M → S_k<br/>Streaming average S̄]
    SubsampleS --> EstFiedler[Compute Fiedler from S_k]
    EstFiedler --> Align[Align to reference<br/>(dot-product sign fix)]
    Align --> Collect[Aggregate aligned vectors]
    Collect --> Average[Normalize averaged vector]
    Average --> PartMetrics[Partition Metrics<br/>M vs M and M vs S̄]
    PartMetrics --> VecMetrics[Vector Metrics<br/>dot product, sign agreement]
    
    VecMetrics --> Save[save_single_results / save_taxa_results]
    Save --> Plot[plot_from_json_simple / plot_faceted]
    Plot --> End([results/<timestamp>-<run_name>])
    
    style Start fill:#e1f5ff
    style Runner fill:#fff4e1
    style Sweep fill:#ffe1f5
    style PartMetrics fill:#ffe1e1
    style Save fill:#e1ffe1
    style End fill:#f0f0f0
```

## File Naming Conventions

- **Config files**: `*.json` in `configs/` directory
- **Result files**: `results*.json`, `fiedler_ref*.npy` in `results/<timestamp>-<run_name>/`
- **Plot files**: `plot_*.png` in results directory
- **Test files**: `test_*.py` in `tests/` directory
- **Analysis notebooks**: `*.ipynb` in `analysis/` subdirectories

## Import Patterns

### Standard Imports
```python
# Configuration
from sub_sampled_fielder_vec.src.config.presets import custom_config
from sub_sampled_fielder_vec.src.config.base_config import StructuredConfig

# Running experiments
from sub_sampled_fielder_vec.src.runners.experiment_runner import ExperimentRunner

# Metrics
from sub_sampled_fielder_vec.src.utils.metrics import (
    compute_partition_agreement,
    compute_fiedler_dot_product
)

# Core computation
from sub_sampled_fielder_vec.src.core.fiedler_computer import FiedlerVectorComputer
from sub_sampled_fielder_vec.src.core.similarity_builder import SimilarityMatrixBuilder
```

### Module Structure
- All code is under `sub_sampled_fielder_vec/src/`
- Import paths: `from sub_sampled_fielder_vec.src.<module> import <item>`
- Package is not installed, so use relative imports or add to `PYTHONPATH`

## Key Classes and Functions

### ExperimentRunner
**Location**: `src/runners/experiment_runner.py`

**Main methods**:
- `__init__(cfg: StructuredConfig)` - Initialize with config
- `run()` - Execute experiment (dispatches to single/taxa/grid based on config)

### sweep_for_params
**Location**: `src/runners/bootstrap_sweep.py`

**Signature**: `sweep_for_params(cfg, n_taxa, seq_len, run_dir) -> tuple`

**Returns**: `(fiedler_ref, sign_agreements, partition_agreement_M, partition_agreement_S, dot_products, metrics_dict, ...)`

**Core logic**: Bootstrap loop that subsamples, computes Fiedler vectors, aligns, averages, and computes metrics.

### StructuredConfig
**Location**: `src/config/base_config.py`

**Structure**:
- `tree: TreeConfig` - Tree model parameters
- `sequence: SequenceConfig` - Sequence model parameters
- `experiment: ExperimentConfig` - Experiment parameters (p-values, bootstrap reps, etc.)
- `sampling: SamplingConfig` - Sampling method (uniform/leveraged) with fallback control
- `metrics: MetricsConfig` - Metrics computation parameters
- `guardrails: GuardrailsConfig` - Validation constraints
- `cache: CacheConfig` - Caching settings
- `output: OutputConfig` - Output directory and file settings

### SamplingConfig
**Location**: `src/config/base_config.py:187-198`

**Key Parameters**:
- `method: str` - "uniform" or "leveraged"
- `theta: float` - Phase 1 budget ratio (0.3 = 30% for uniform sampling)
- `target_rank: int` - Rank for SVD in leverage estimation (typically 2)
- `ialm_max_iter: int` - Maximum IALM solver iterations (100)
- `ialm_tol: float` - IALM convergence tolerance (1e-6)
- `ialm_bypass_threshold: float` - Skip IALM when p >= threshold (0.1)
- `force_leveraged: bool` - Force leveraged sampling at very low p (False)
- `allow_uniform_fallback: bool` - Allow fallback to uniform sampling (True = safe mode, False = research mode)
- `log_sampling_diagnostics: bool` - Save leverage scores for analysis (False)

**Modes**:
- **Safe Mode** (default: `allow_uniform_fallback=True`): Falls back to uniform when p too small
- **Research Mode** (`allow_uniform_fallback=False`): Proceeds with leveraged using 90/10 split, no guardrails

## Common Tasks

### Task: Run a Single Experiment
1. Edit `scripts/run_experiment.py` → `SWEEP_CONFIG`
2. Set `taxa_values=[n]` (single value) or omit for single experiment mode
3. Run: `python scripts/run_experiment.py`

### Task: Modify Metrics
1. Edit `src/utils/metrics.py`
2. Add new metric function
3. Call from `src/runners/bootstrap_sweep.py` in the metrics computation section
4. Add to `save_single_results()` in `src/utils/summaries.py`

### Task: Add New Tree Model
1. Add model class to `src/models/tree_models.py`
2. Register in model factory
3. Add to `TreeConfig.model` enum in `src/config/base_config.py`

### Task: Add New Sampling Method
1. Create new subdirectory in `src/core/sampling/`
2. Implement `Sampler` interface (see `src/core/sampling/base.py`)
3. Add to `SamplingConfig.method` enum
4. Integrate in `SimilarityMatrixBuilder` in `src/core/similarity_builder.py`

## Memory and Performance Notes

- **Streaming S average**: Uses Welford's algorithm to avoid storing K matrices (saves ~12.8 GB for n=4000, K=100)
- **Persistent caching**: Matrices cached to disk in `src/cache/` for instant re-runs
- **Sparse SVD**: Automatically used for large sparse matrices in Fiedler computation
- **Parallel execution**: Controlled by `num_workers` in `ExperimentConfig`

## Diagnostic Logging System

### Purpose

The diagnostic logging system captures detailed leverage sampling data for validation and analysis:
- **Estimated leverage scores** from Phase 1 uniform sampling
- **Phase 2 sampling probabilities** for entries actually sampled
- **Sparse representation** to minimize disk space

### Enabling Diagnostics

**Option 1: Interactive launcher**
```bash
python scripts/interactive_run.py
# When prompted: "Log sampling diagnostics (for analysis)?" → Yes
```

**Option 2: Configuration file**
```python
cfg.sampling.log_sampling_diagnostics = True
```

**Option 3: Script SWEEP_CONFIG**
```python
SWEEP_CONFIG = {
    # ... other config ...
    "sampling_method": "leveraged",
    "log_sampling_diagnostics": True,
}
```

### Diagnostic Data Format

**Location**: `{run_dir}/sampling_data/p_{p_value:.4f}.npz`

**Contents** (per p-value):
```python
import numpy as np

data = np.load("sampling_data/p_0.1438.npz")

# Leverage scores estimated from Phase 1
leverage_scores = data['leverage_scores']  # shape: (n_taxa,)

# Phase 2 sampling probabilities (sparse)
# Only non-zero for entries sampled in Phase 2
phase2_probs = data['phase2_probs_sampled']  # shape: (n_taxa, n_taxa)
```

**Data saved for**: Only p-values where leveraged sampling actually runs (p ≥ theoretical minimum)

**Theoretical minimum**: `p_min ≈ (4·n·r·log(n)) / (n(n-1)/2)` where r = target_rank

### Analysis Notebook

**Location**: `analysis/notebooks/leverage_sampling_explorer.ipynb`

**Purpose**: Visualize leverage score quality by comparing estimated vs ground truth

**Usage**:
1. Open notebook in Jupyter
2. Edit configuration cell with experiment path and p-value
3. Run all cells

**Outputs**:
- **Plot 1**: Full similarity matrix heatmap
- **Plot 2**: True vs estimated leverage scores (scatter with correlation)
- **Plot 3**: Phase 2 sampling probabilities heatmap (sparse)
- **Statistics**: Correlation, MAE, sampling coverage, top taxa

**Example findings**: At low p-values (near threshold), correlation may be low, validating the theoretical minimum budget requirement.

### Implementation Details

**Files involved**:
1. `src/core/sampling/leveraged/sampler.py:220-242` - Extends `last_sample_metrics` dict
2. `src/utils/sampling_logger.py` - Saving functions (`save_sampling_diagnostics()`)
3. `src/runners/bootstrap_sweep.py:694-721` - Integration point (saves after bootstrap loop)

**Key design choices**:
- Save **last bootstrap only** (not all K bootstraps) to reduce disk usage
- Use **sparse matrix** representation for Phase 2 probs
- **No ground truth** computed during runs (compute in analysis for efficiency)
- **Symmetric matrices**: Only one set of leverage scores needed

## See Also

- [METRICS.md](METRICS.md) - Detailed metrics documentation
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration system details
- [LEVERAGED_SAMPLING.md](LEVERAGED_SAMPLING.md) - Leveraged sampling implementation
- [ANALYSIS_GUIDES.md](ANALYSIS_GUIDES.md) - Analysis notebooks and tools
