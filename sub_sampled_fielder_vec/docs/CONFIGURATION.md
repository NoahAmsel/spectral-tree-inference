# Configuration System

This document describes the configuration system used to define experiments.

## Overview

The framework uses **Pydantic v2** models for type-safe, validated configuration. All configurations are built as `StructuredConfig` objects that group related parameters into logical sections.

## StructuredConfig

**Location**: `src/config/base_config.py`

`StructuredConfig` is the main configuration object that groups:

- `tree: TreeConfig` - Tree model parameters
- `sequence: SequenceConfig` - Sequence model parameters
- `experiment: ExperimentConfig` - Experiment parameters (p-values, bootstrap reps, etc.)
- `sampling: SamplingConfig` - Sampling method (uniform/leveraged) with fallback control (Safe/Research modes)
- `metrics: MetricsConfig` - Metrics computation parameters
- `guardrails: GuardrailsConfig` - Validation constraints
- `cache: CacheConfig` - Caching settings
- `output: OutputConfig` - Output directory and file settings

### Helper Methods

- `to_json_file(path)` - Save config to JSON file
- `summary()` - Human-readable summary string
- `get_num_taxa()` - Extract number of taxa (handles single vs sweep)
- `to_dict()` - Convert to dictionary

## Configuration Sections

### TreeConfig

Tree topology and generation parameters.

**Parameters**:
- `model: str` - Tree model type
  - `"balanced_binary"` - Balanced binary tree
  - `"kingman_mean"` - Kingman coalescent (mean branch lengths)
  - `"kingman"` - Kingman coalescent
  - `"birth_death"` - Birth-death process
  - `"lopsided"` - Lopsided tree
- `num_taxa: int` - Number of taxa (leaves)
- `edge_length: float` - Edge length (for balanced_binary, lopsided)
- `pop_size: float` - Population size (for kingman models)
- `birth_rate: float` - Birth rate (for birth_death)
- `death_rate: float` - Death rate (for birth_death)

**Example**:
```python
tree = TreeConfig(
    model="balanced_binary",
    num_taxa=1024,
    edge_length=1.0
)
```

### SequenceConfig

Sequence model and mutation parameters.

**Parameters**:
- `model: str` - Sequence model type
  - `"JC69"` - Jukes-Cantor (simplest)
  - `"HKY"` - Hasegawa-Kishino-Yano
  - `"GTR"` - General Time Reversible
  - `"TN93"`, `"T92"` - Other models
- `length: int` - Sequence length
- `mutation_rate: float` - Mutation rate (required)
- `kappa: float` - Transition/transversion ratio (for HKY)
- `stationary_freqs: List[float]` - Stationary frequencies (for HKY, GTR)

**Example**:
```python
sequence = SequenceConfig(
    model="JC69",
    length=1000,
    mutation_rate=0.1
)
```

### ExperimentConfig

Experiment execution parameters.

**Parameters**:
- `p_values: Tuple[float, ...]` - Sampling probabilities to test
- `bootstrap_reps: int` - Number of bootstrap replicates per p-value
- `run_name: str` - Name for this experiment run
- `display_mode: str` - Progress display mode
  - `"progress"` - Show progress bars
  - `"quiet"` - Minimal output
  - `"verbose"` - Detailed logging
- `num_workers: int` - Number of parallel workers (0 = sequential)
- `use_middle_out: bool` - Use middle-out optimization
- `seed: Optional[int]` - Random seed (None = random)

**Example**:
```python
experiment = ExperimentConfig(
    p_values=(0.01, 0.1, 0.5, 1.0),
    bootstrap_reps=50,
    run_name="my_experiment",
    display_mode="progress",
    num_workers=4
)
```

### SamplingConfig

Matrix sampling method configuration.

**Parameters**:
- `method: str` - Sampling method
  - `"uniform"` - Uniform random sampling (default)
  - `"leveraged"` - Leveraged matrix completion sampling
- `theta: float` - Phase 1 budget ratio (for leveraged, default: 0.3)
  - Fraction of samples used for uniform Phase 1 leverage estimation
- `target_rank: int` - SVD rank for leverage estimation (for leveraged, default: 2)
  - Typically 2 for Fiedler vector applications
- `ialm_max_iter: int` - IALM solver max iterations (for leveraged, default: 100)
- `ialm_tol: float` - IALM convergence tolerance (for leveraged, default: 1e-6)
- `ialm_bypass_threshold: float` - Skip IALM when p >= threshold (default: 0.1)
  - Saves computation time when sampling is dense enough
- `force_leveraged: bool` - Force leveraged sampling at very low p (default: False)
  - When True, uses leveraged even when Phase 1 budget is theoretically insufficient
- `allow_uniform_fallback: bool` - Allow fallback to uniform sampling (default: True)
  - **Safe Mode** (True): Falls back to uniform when p too small
  - **Research Mode** (False): Proceeds with leveraged using 90/10 split, no guardrails
- `log_sampling_diagnostics: bool` - Save diagnostic data for analysis (default: False)
  - Saves leverage scores and sampling probabilities to disk

**Example (Safe Mode)**:
```python
sampling = SamplingConfig(
    method="leveraged",
    theta=0.3,
    target_rank=2,
    ialm_max_iter=100,
    ialm_tol=1e-6,
    allow_uniform_fallback=True  # Safe mode (default)
)
```

**Example (Research Mode)**:
```python
sampling = SamplingConfig(
    method="leveraged",
    theta=0.3,
    target_rank=2,
    allow_uniform_fallback=False,  # Research mode - no guardrails
    log_sampling_diagnostics=True   # Save diagnostics for analysis
)
```

### MetricsConfig

Metrics computation parameters.

**Parameters**:
- `num_gaps: int` - Number of gaps to search for partition threshold (default: 1)
- `min_split: int` - Minimum partition size (default: 1)
- `coherence_k: int` - Number of top singular vectors for coherence (default: 2)

**Example**:
```python
metrics = MetricsConfig(
    num_gaps=1,
    min_split=1,
    coheren
```

### GuardrailsConfig

Validation constraints to catch errors early.

**Parameters**:
- `max_taxa: Optional[int]` - Maximum allowed taxa (None = no limit)
- `max_sequence_length: Optional[int]` - Maximum sequence length (None = no limit)
- `min_p_value: float` - Minimum allowed p-value (default: 1e-6)
- `max_bootstrap_reps: Optional[int]` - Maximum bootstrap reps (None = no limit)

### CacheConfig

Caching settings for performance.

**Parameters**:
- `use_persistent_cache: bool` - Use disk cache (default: False)
- `cache_dir: Optional[str]` - Cache directory (None = default)
- `clear_cache_between_runs: bool` - Clear cache between parameter combinations (default: True)

### OutputConfig

Output directory and file settings.

**Parameters**:
- `base_dir: str` - Base directory for results (default: "results")
- `save_fiedler_vectors: bool` - Save Fiedler vectors as .npy files (default: True)
- `save_plots: bool` - Generate and save plots (default: True)
- `plot_format: str` - Plot file format (default: "png")

## Building Configurations

### Using custom_config() Helper

**Location**: `src/config/presets.py`

The `custom_config()` function provides a convenient way to build configurations:

```python
from sub_sampled_fielder_vec.src.config.presets import custom_config

cfg = custom_config(
    num_taxa=1024,
    sequence_length=1000,
    mutation_rate=0.1,
    tree_model="balanced_binary",
    p_values=(0.01, 0.1, 0.5, 1.0),
    bootstrap_reps=50,
    run_name="my_experiment",
    display_mode="progress",
    num_workers=4
)
```

**Supported parameters**:
- `num_taxa` or `taxa_values` - Single value or list for sweep
- `sequence_length` or `sequence_length_values` - Single value or list for grid search
- `mutation_rate` - Mutation rate
- `tree_model` - Tree model name
- `p_values` - Sampling probabilities
- `bootstrap_reps` - Bootstrap replicates
- `run_name` - Run name
- `display_mode` - Progress display mode
- `num_workers` - Parallel workers
- `use_middle_out` - Middle-out optimization
- `**tree_kwargs` - Additional tree model parameters

### Direct Pydantic Construction

For full control, construct configs directly:

```python
from sub_sampled_fielder_vec.src.config.base_config import (
    StructuredConfig,
    TreeConfig,
    SequenceConfig,
    ExperimentConfig,
    SamplingConfig
)

cfg = StructuredConfig(
    tree=TreeConfig(
        model="balanced_binary",
        num_taxa=1024,
        edge_length=1.0
    ),
    sequence=SequenceConfig(
        model="JC69",
        length=1000,
        mutation_rate=0.1
    ),
    experiment=ExperimentConfig(
        p_values=(0.01, 0.1, 0.5, 1.0),
        bootstrap_reps=50,
        run_name="my_experiment"
    ),
    sampling=SamplingConfig(method="uniform")
)
```

## Experiment Types

### Single Experiment

One parameter combination (n_taxa, sequence_length):

```python
cfg = custom_config(
    num_taxa=1024,
    sequence_length=1000,
    mutation_rate=0.1,
    p_values=(0.01, 0.1, 0.5, 1.0),
    bootstrap_reps=50,
    run_name="single_experiment"
)
```

### Taxa Sweep

Multiple taxa values, fixed sequence length:

```python
cfg = custom_config(
    taxa_values=[512, 1024, 2048, 4096],
    sequence_length=1000,
    mutation_rate=0.1,
    p_values=tuple(np.logspace(-4, 0, 15)),
    bootstrap_reps=30,
    run_name="taxa_sweep"
)
```

### Grid Search

Multiple taxa values × multiple sequence lengths:

```python
cfg = custom_config(
    taxa_values=[512, 1024, 2048],
    sequence_length_values=[500, 1000, 2000],
    mutation_rate=0.1,
    p_values=tuple(np.logspace(-4, 0, 15)),
    bootstrap_reps=30,
    run_name="grid_search"
)
```

## Using Configurations in Scripts

### Main Entry Point

**File**: `scripts/run_experiment.py`

Edit the `SWEEP_CONFIG` dictionary:

```python
SWEEP_CONFIG: Dict[str, Any] = {
    "tree_model": "balanced_binary",
    "taxa_values": [1024, 2048],
    "sequence_length": 1000,
    "mutation_rate": 0.1,
    "p_values": [0.01, 0.1, 0.5, 1.0],
    "bootstrap_reps": 50,
    "run_name_prefix": "my_experiment",
    "num_workers": 4,
}
```

The script automatically converts this to a `StructuredConfig` using `custom_config()`.

### Programmatic Usage

```python
from sub_sampled_fielder_vec.src.config.presets import custom_config
from sub_sampled_fielder_vec.src.runners.experiment_runner import ExperimentRunner

cfg = custom_config(
    num_taxa=1024,
    sequence_length=1000,
    mutation_rate=0.1,
    p_values=(0.01, 0.1, 0.5, 1.0),
    bootstrap_reps=50,
    run_name="my_experiment"
)

runner = ExperimentRunner(cfg)
run_dir, results = runner.run()
```

## Configuration Validation

Pydantic automatically validates all configurations:

- **Type checking**: Ensures correct types (int, float, str, etc.)
- **Value constraints**: Enforces ranges (e.g., p_values in [0, 1])
- **Required fields**: Ensures all required parameters are provided
- **Enum validation**: Ensures model names match allowed values

**Example error**:
```python
# This will raise a ValidationError
cfg = custom_config(
    num_taxa=-100,  # Negative taxa not allowed
    sequence_length=1000,
    mutation_rate=0.1
)
```

## Saving and Loading Configurations

### Save to JSON

```python
cfg.to_json_file("my_config.json")
```

### Load from JSON

```python
import json
from sub_sampled_fielder_vec.src.config.base_config import StructuredConfig

with open("my_config.json") as f:
    data = json.load(f)
    cfg = StructuredConfig(**data)
```

## Memory Optimization Notes

The configuration system supports memory-efficient execution:

- **Streaming S average**: Uses Welford's algorithm to compute S_avg without storing all K matrices
- For n=4000, K=100: saves ~12.8 GB memory
- Only stores one S_avg matrix (~128 MB) instead of 100 S matrices
- Controlled by `CacheConfig.clear_cache_between_runs`

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Codebase structure
- [LEVERAGED_SAMPLING.md](LEVERAGED_SAMPLING.md) - Leveraged sampling configuration details
