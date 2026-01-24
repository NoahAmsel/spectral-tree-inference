# Codebase Architecture

This document provides a structural overview of the codebase optimized for AI agents and automated workflows.

## Directory Map

```
sub_sampled_fielder_vec/
├── README.md                    # Main entry point - concise overview
├── scripts/                      # CLI entry points and utilities
│   ├── run_experiment.py        # Main experiment launcher (edit SWEEP_CONFIG here)
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
│   │   └── metric_computer.py   # Matrix metrics (operator norm, coherence, etc.)
│   │   └── sampling/            # Sampling methods
│   │       ├── uniform/         # Uniform matrix entry sampling
│   │       └── leveraged/       # Leveraged matrix completion sampling
│   ├── runners/                 # Experiment orchestration
│   │   ├── experiment_runner.py # Main ExperimentRunner class
│   │   ├── bootstrap_sweep.py   # Core bootstrap loop logic
│   │   └── parallel_bootstrap.py  # Parallel execution utilities
│   └── utils/                   # Shared utilities
│       ├── metrics.py           # Partition agreement, dot product, sign agreement
│       ├── summaries.py         # Result saving (JSON tables)
│       ├── plotting.py          # Visualization functions
│       ├── logging.py           # Logging utilities
│       └── threshold_utils.py   # Partition thresholding helpers
├── configs/                      # Saved JSON configurations
│   ├── presets/                 # Checked-in preset configs
│   └── custom/                  # User-provided configs (gitignored)
├── docs/                         # Documentation (this directory)
├── examples/                     # Small runnable code samples
├── tests/                        # Unit and integration tests
├── analysis/                     # Analysis notebooks and tools
│   ├── comparison/              # Method comparison (uniform vs leveraged)
│   ├── leveraged_sampling_analysis/  # Leveraged sampling diagnostics
│   ├── spectral_analysis/       # Spectral analysis frameworks
│   └── generic_analysis/        # Generic analysis utilities
└── results/                      # Auto-generated experiment artifacts (gitignored)
```

## Key Entry Points

### For Running Experiments
- **`scripts/run_experiment.py`** - Main entry point. Edit `SWEEP_CONFIG` dictionary to configure experiments.

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
- `sampling: SamplingConfig` - Sampling method (uniform/leveraged)
- `metrics: MetricsConfig` - Metrics computation parameters
- `guardrails: GuardrailsConfig` - Validation constraints
- `cache: CacheConfig` - Caching settings
- `output: OutputConfig` - Output directory and file settings

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
- **Similarity caching**: M computed once per (n_taxa, seq_len) combination
- **Sparse SVD**: Automatically used for large sparse matrices in Fiedler computation
- **Parallel execution**: Controlled by `num_workers` in `ExperimentConfig`

## See Also

- [METRICS.md](METRICS.md) - Detailed metrics documentation
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration system details
- [LEVERAGED_SAMPLING.md](LEVERAGED_SAMPLING.md) - Leveraged sampling implementation
