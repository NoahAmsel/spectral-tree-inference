# Display Modes Documentation

## Overview
The experiment system now supports two display modes for better user experience:
- **Progress Mode** (default): Clean progress bars, minimal logging
- **Debug Mode**: Verbose logging for troubleshooting

## Usage

### Setting Display Mode in Config

```python
from utils.experiment_config import Config

# Progress mode (default) - clean progress bars
cfg = Config(
    taxa_values=[1024, 2048],
    display_mode="progress"  # Clean output
)

# Debug mode - verbose logging
cfg = Config(
    taxa_values=[1024, 2048],
    display_mode="debug"  # All logs
)
```

## Progress Mode (Default)

**What you see:**
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
Faceted plot saved to: .../plot_grid_faceted.png
EXPERIMENT_RUNNER | INFO | [done] artifacts written to: ...
```

**Features:**
- ✅ Opening title with experiment configuration summary
- ✅ Clean progress bars showing all configurations
- ✅ No intermediate log messages (guardrails, building tree, etc.)
- ✅ No numpy/sklearn numerical warnings
- ✅ **Plotting/saving messages ARE visible** (important post-processing info)
- ✅ Final completion message shown
- ✅ Perfect for long-running experiments

## Debug Mode

**What you see:**
```
BOOTSTRAP_SWEEP | INFO | n=1024, L=500 building tree and sequences…
BOOTSTRAP_SWEEP | INFO | Computing full similarity + Fiedler…
CACHE_MANAGER | INFO | Computing and caching full similarity matrix...
BOOTSTRAP_SWEEP | INFO | Guardrails triggered: skipping remaining p-values
...
[All detailed logs]
[Final summary message]
```

**Features:**
- ✅ All log messages shown (INFO, WARNING, ERROR)
- ✅ Numpy/sklearn warnings visible
- ✅ Detailed progress information
- ✅ No progress bars (logs are the progress indicator)
- ✅ Perfect for debugging and understanding what's happening

## Implementation Details

### Files Modified
1. **`utils/experiment_config.py`**
   - Added `display_mode: str = "progress"` field to Config

2. **`utils/logging.py`**
   - Added `set_display_mode()`, `get_display_mode()`, `is_progress_mode()`
   - Added `suppress_numerical_warnings()` to filter numpy/sklearn warnings
   - Updated `log_info()` and `log_warning()` to accept `force` parameter
   - In progress mode: suppress all logs unless `force=True`

3. **`experiment/experiment_runner.py`**
   - Added `_print_opening_title()` method to display config summary
   - Sets display mode in `__init__()` via `set_display_mode(cfg.display_mode)`
   - Forces final "[done]" messages with `force=True`
   - Passes `show_progress=not is_progress_mode()` to bootstrap_sweep

4. **`experiment/bootstrap_sweep.py`**
   - No changes needed - existing `log_info()` calls automatically respect mode

5. **`utils/plotting.py`**
   - Already uses `print()` for save messages (always visible in both modes)

### Testing
Run `test_display_modes.py` to verify both modes:
```bash
# Test progress mode
python test_display_modes.py --mode progress

# Test debug mode
python test_display_modes.py --mode debug

# Test both
python test_display_modes.py --mode both
```
