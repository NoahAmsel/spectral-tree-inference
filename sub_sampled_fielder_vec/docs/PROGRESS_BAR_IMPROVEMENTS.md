# Progress Bar Improvements

## Overview
Improved the logging system to display one progress bar per configuration from the beginning, with p-values as the progress metric within each configuration. Added clean display modes and configuration summary.

## New Format
```
================================================================================
SPECTRAL TREE INFERENCE - Sub-sampled STDR Experiment
================================================================================
Experiment Type:    Grid Search
Configurations:     n=[1024], L=[500, 1000, 5000, 10000]
Mutation Rate:      μ = 0.1
P-values:           21 values from 1.00e-04 to 1.00e+00
Bootstrap Reps:     10
Display Mode:       progress
Run Name:           my_experiment
================================================================================

Conf 1/4 L=  500 n= 1024: p-values  25%|███████              | 5/21
Conf 2/4 L= 1000 n= 1024: p-values   0%|                     | 0/21
Conf 3/4 L= 5000 n= 1024: p-values   0%|                     | 0/21
Conf 4/4 L=10000 n= 1024: p-values   0%|                     | 0/21

Fiedler vectors plot saved to: .../fiedler_vectors_mu=0.1.png
EXPERIMENT_RUNNER | INFO | [done] artifacts written to: ...
```

## Changes Made

### 1. `utils/logging.py`
- Added `create_config_progress_bar()` function for configuration-level progress tracking
- Creates bars with format: "Conf X/Y L=... n=...: p-values"
- All bars are created upfront and stacked vertically

### 2. `experiment/bootstrap_sweep.py`
- Added `progress_callback` parameter to `sweep_for_params()`
- Removed nested p-value progress bar
- Invokes callback after each p-value completes
- Bootstrap progress bars now positioned high (position=100) to avoid interference

### 3. `experiment/experiment_runner.py`
- **Grid Search (`_run_grid_search`)**: Creates all config bars before starting any computations
- **Taxa Sweep (`_run_taxa_sweep`)**: Creates all config bars before starting any computations
- Both methods pass callback functions to update progress bars as p-values complete
- Bootstrap progress hidden by default (`show_progress=False`) to reduce clutter

### 4. Bug Fix
- Fixed `bootstrap_sweep.py:117` to call `cfg.fiedler_method(M)` instead of passing obsolete `observations, p=1.0` parameters

## Testing
Created `test_progress_bars.py` to validate the new system:
- Taxa sweep test: 3 configurations × 4 p-values ✅
- Grid search test: 4 configurations (2×2) × 4 p-values ✅

Both tests show all configuration bars from the start, updating as each p-value completes.

## Benefits
1. **Immediate visibility**: Users see all configurations upfront, not incrementally
2. **Better UX**: Clear progress tracking for long-running experiments
3. **Less clutter**: Bootstrap iterations hidden by default (can be enabled for debugging)
4. **Consistent format**: Uniform display across taxa sweeps and grid searches
