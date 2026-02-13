# Recent Changes: Leveraged Sampling Configuration Flow

**Date**: 2026-02-13
**Impact**: Configuration system, leveraged sampling behavior

## Summary

Added comprehensive configuration control for leveraged sampling behavior, including a "Research Mode" that removes guardrails and allows experimentation at very low p-values without automatic fallback to uniform sampling.

## Key Changes

### 1. New Configuration Parameter: `allow_uniform_fallback`

**Purpose**: Control fallback behavior when Phase 1 budget requirements cannot be met

**Default**: `True` (Safe Mode)

**Modes**:
- **Safe Mode** (`True`): Falls back to uniform sampling with warning when p too small
- **Research Mode** (`False`): Proceeds with leveraged sampling using 90/10 split, no errors

**Location**: `src/config/base_config.py:190`

### 2. Custom Exception: `InsufficientBudgetError`

**Purpose**: Prevent accidental catching by generic exception handlers

**Location**: `src/core/sampling/leveraged/sampler.py:25-27`

**Rationale**: Generic `ValueError` was being caught by menu handlers, hiding the real issue

### 3. Configuration Threading (8 Layers)

The `allow_uniform_fallback` parameter flows through the entire system:

```
User Input → JSON Config → Extraction → Builder → Preset → Dataclass → Runner → Sampler
```

**Files Modified**:
1. `scripts/interactive_run.py:164,244` - UI collection
2. `src/runners/experiment_runner_utils.py:94` - Extraction
3. `src/runners/experiment_runner_utils.py:133` - Passing to builder
4. `src/config/presets.py:159,191,251` - `custom_config()` signature
5. `src/config/base_config.py:190` - `SamplingConfig` dataclass
6. `src/runners/bootstrap_sweep.py:256` - Sampler instantiation
7. `src/core/sampling/leveraged/sampler.py:40-70` - Final usage

### 4. Cache Selection Bug Fix

**Problem**: Menu displayed sorted list but selection used unsorted list, causing index mismatch

**Fix**: Added sorting in selection handler to match menu display order

**Location**: `scripts/interactive_run.py:342`

### 5. Updated Fallback Logic

**Location**: `src/core/sampling/leveraged/sampler.py:136-147`

**New Behavior**:
```python
if phase1_budget >= total_budget:
    if self.force_leveraged or not self.allow_uniform_fallback:
        # Research mode: Proceed with 90/10 split
        phase1_budget = max(1, int(0.9 * total_budget))
        log_info('bootstrap', "Proceeding with leveraged sampling despite insufficient budget...")
    else:
        # Safe mode: Fall back to uniform
        log_info('bootstrap', "Falling back to uniform sampling...")
        return uniform_sample_and_return(...)
```

## Documentation Updates

### Updated Files:
1. **LEVERAGED_SAMPLING.md**
   - Added "Research Mode vs Safe Mode" section
   - Updated Configuration Parameters with new flags
   - Updated Minimum Phase 1 Budget Enforcement section
   - Added Configuration Flow diagram
   - Updated Troubleshooting section
   - Updated Files Modified section

2. **ARCHITECTURE.md**
   - Added detailed SamplingConfig section
   - Documented Safe Mode vs Research Mode

3. **CONFIGURATION.md**
   - Expanded SamplingConfig section
   - Added examples for Safe Mode and Research Mode
   - Updated StructuredConfig description

## Usage Examples

### Interactive Launcher (Research Mode)
```bash
python scripts/interactive_run.py
# When prompted: "Allow fallback to uniform sampling for low p?" → No
```

### Programmatic (Research Mode)
```python
cfg = custom_config(
    num_taxa=512,
    sampling_method="leveraged",
    sampling_allow_uniform_fallback=False,  # Research mode
    # ... other params ...
)
```

### Script-based (Research Mode)
```python
SWEEP_CONFIG = {
    "sampling_method": "leveraged",
    "sampling_allow_uniform_fallback": False,  # Research mode
    # ... other config ...
}
```

## Log Messages

### Safe Mode (Fallback Enabled)
```
p=0.0001: Falling back to uniform sampling (Phase 1 needs 12,543 but budget is 2,621)
```

### Research Mode (Fallback Disabled)
```
p=0.0001: Proceeding with leveraged sampling despite insufficient budget (using 2,358/2,621 for Phase 1, theoretical min: 12,543)
```

## Rationale

**User Request**: "I want the code to allow run whatever I want, we do not need the fallback and the guardrails"

**Solution**: Added research mode that removes automatic fallback and allows full experimentation without restrictions. This is essential for:
- Understanding algorithm behavior at extreme low-p regimes
- Generating diagnostic data for all p-values
- Research and experimentation without guardrails

## Backward Compatibility

✅ **Fully backward compatible**
- Default behavior unchanged (Safe Mode)
- Existing code continues to work without modification
- New parameter is optional with sensible default

## Testing Recommendations

1. **Test Safe Mode** (default):
   - Run experiment with very low p-values
   - Verify fallback to uniform with warning

2. **Test Research Mode**:
   - Set `allow_uniform_fallback=False`
   - Run experiment with very low p-values
   - Verify leveraged sampling proceeds with 90/10 split

3. **Test Cache Selection**:
   - Create multiple cached matrices
   - Select option #3 from menu
   - Verify correct n_taxa is used

## See Also

- [LEVERAGED_SAMPLING.md](LEVERAGED_SAMPLING.md) - Complete leveraged sampling guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration system details
