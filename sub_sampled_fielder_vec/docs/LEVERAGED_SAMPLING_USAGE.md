# Leveraged Sampling Usage Guide

This guide explains how to run experiments with the new leveraged matrix completion sampling method.

## Quick Start

### Method 1: Using `run_experiment.py` Script

Edit `scripts/run_experiment.py` and add sampling configuration to `SWEEP_CONFIG`:

```python
SWEEP_CONFIG: Dict[str, Any] = {
    "tree_model": "kingman_mean",
    "taxa_values": [500, 1000],
    "sequence_length_values": [10000],
    "mutation_rate": 0.1,
    "bootstrap_reps": 20,
    "num_workers": 8,
    "use_middle_out": False,
    "run_name_prefix": "leveraged_test",
    "p_values": list(np.logspace(-4, 0, 20)),
    "tree_params": {"pop_size": 1.0},
    "coherence_k": 4,
    "num_gaps": 0,
    "guardrails_enabled": False,
    # Add sampling configuration:
    "sampling_method": "leveraged",  # or "uniform" (default)
    "sampling_theta": 0.3,           # Phase 1 budget ratio
    "sampling_target_rank": 2,        # SVD rank for leverage estimation
    "sampling_ialm_max_iter": 100,    # IALM iterations
    "sampling_ialm_tol": 1e-6,        # IALM tolerance
}
```

Then update the `main()` function to pass sampling config:

```python
cfg = custom_config(
    num_taxa=n_taxa,
    sequence_length=seq_len,
    mutation_rate=mutation_rate,
    tree_model=tree_model,
    p_values=p_values,
    bootstrap_reps=bootstrap_reps,
    run_name=prefix,
    display_mode="progress",
    num_workers=num_workers,
    use_middle_out=use_middle_out,
    **tree_kwargs,
)

# Set sampling method
if "sampling_method" in config:
    from src.config.base_config import SamplingConfig
    cfg.sampling = SamplingConfig(
        method=config.get("sampling_method", "uniform"),
        theta=config.get("sampling_theta", 0.3),
        target_rank=config.get("sampling_target_rank", 2),
        ialm_max_iter=config.get("sampling_ialm_max_iter", 100),
        ialm_tol=config.get("sampling_ialm_tol", 1e-6),
    )
```

Run:
```bash
cd sub_sampled_fielder_vec
python scripts/run_experiment.py
```

### Method 2: Programmatic Usage

#### Example 1: Basic Leveraged Sampling

```python
from src.config.presets import custom_config
from src.config.base_config import SamplingConfig
from src import ExperimentRunner

# Create config with leveraged sampling
cfg = custom_config(
    num_taxa=1024,
    sequence_length=1000,
    mutation_rate=0.1,
    tree_model="balanced_binary",
    p_values=[0.01, 0.1, 0.5, 1.0],
    bootstrap_reps=50,
    run_name="leveraged_experiment"
)

# Override sampling config to use leveraged method
cfg.sampling = SamplingConfig(
    method="leveraged",
    theta=0.3,              # 30% of budget for Phase 1 (uniform sampling)
    target_rank=2,          # Rank for SVD (typically 2 for Fiedler)
    ialm_max_iter=100,      # IALM solver iterations
    ialm_tol=1e-6           # IALM convergence tolerance
)

# Run experiment
runner = ExperimentRunner(cfg, base_dir="results")
run_dir, results = runner.run()
```

#### Example 2: Compare Uniform vs Leveraged

```python
from src.config.presets import custom_config
from src.config.base_config import SamplingConfig
from src import ExperimentRunner

base_cfg = custom_config(
    num_taxa=512,
    sequence_length=1000,
    mutation_rate=0.1,
    p_values=[0.01, 0.05, 0.1, 0.5, 1.0],
    bootstrap_reps=20,
    run_name="comparison"
)

# Run with uniform (default)
cfg_uniform = base_cfg.copy()
cfg_uniform.sampling = SamplingConfig(method="uniform")
runner_uniform = ExperimentRunner(cfg_uniform, base_dir="results/comparison")
run_dir_uniform, results_uniform = runner_uniform.run()

# Run with leveraged
cfg_leveraged = base_cfg.copy()
cfg_leveraged.sampling = SamplingConfig(
    method="leveraged",
    theta=0.3,
    target_rank=2
)
cfg_leveraged.experiment.run_name = "comparison_leveraged"
runner_leveraged = ExperimentRunner(cfg_leveraged, base_dir="results/comparison")
run_dir_leveraged, results_leveraged = runner_leveraged.run()
```

#### Example 3: Direct Sampler Usage

```python
import numpy as np
from src.core.similarity_builder import SimilarityMatrixBuilder
from src.core.sampling.leveraged import LeveragedSampler

# Create similarity matrix builder with leveraged method
builder = SimilarityMatrixBuilder(
    method="leveraged",
    theta=0.3,
    target_rank=2,
    ialm_max_iter=100,
    ialm_tol=1e-6
)

# Build full similarity matrix from observations
observations = ...  # Your sequence data (n_taxa x seq_len)
full_similarity = builder.build_full(observations)

# Subsample/recover using leveraged method
p = 0.1  # 10% sampling budget
recovered_matrix = builder.build_subsampled(observations, p=p, seed=42)

# Use recovered matrix for Fiedler vector computation
from src.core.utils import compute_fielder_vector
fiedler = compute_fielder_vector(recovered_matrix)
```

## Configuration Parameters

### SamplingConfig Parameters

- **`method`**: `"uniform"` or `"leveraged"` (default: `"uniform"`)
- **`theta`**: Phase 1 budget ratio for leveraged sampling (default: `0.7`)
  - Fraction of total samples used for uniform Phase 1
  - Remaining `1-theta` used for Phase 2 (leveraged sampling)
- **`target_rank`**: Rank `r` for SVD in leverage score computation (default: `2`)
  - Typically 2 for Fiedler vector applications
  - Higher ranks may improve leverage estimation but increase computation
- **`ialm_max_iter`**: Maximum IALM solver iterations (default: `100`)
  - More iterations = better recovery but slower
- **`ialm_tol`**: IALM convergence tolerance (default: `1e-6`)
  - Smaller = more accurate but slower convergence

## Algorithm Overview

The leveraged sampling method implements Algorithm 1 from the paper:

1. **Phase 1 (Uniform)**: Sample `θN` entries uniformly to estimate leverage scores
2. **Phase 2 (Leveraged)**: Sample remaining `(1-θ)N` entries based on importance (leverage scores)
3. **Phase 3 (Recovery)**: Use IALM solver to recover clean low-rank matrix from noisy observations

## Performance Notes

- **Leveraged sampling** is slower than uniform but may achieve better recovery with fewer samples
- **IALM solver** adds computational overhead (typically 10-100x slower than uniform)
- For large matrices (n > 5000), consider:
  - Reducing `ialm_max_iter` for faster runs
  - Using `target_rank=2` (sufficient for Fiedler vector)
  - Enabling parallel execution with `num_workers > 1`

## Troubleshooting

### Import Errors
If you see import errors, ensure you're running from the project root:
```bash
cd /path/to/spectral-tree-inference/sub_sampled_fielder_vec
python scripts/run_experiment.py
```

### IALM Convergence Issues
If IALM doesn't converge:
- Increase `ialm_max_iter` (e.g., 200-500)
- Relax `ialm_tol` (e.g., 1e-5)
- Check that sampling budget `p` is sufficient (typically need p > 0.01)

### Memory Issues
For very large matrices:
- Use uniform sampling (`method="uniform"`) which is more memory-efficient
- Reduce `bootstrap_reps` for initial testing
- Enable persistent cache: `cfg.cache.use_persistent_cache = True`

