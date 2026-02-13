# Documentation Update Summary

This document summarizes the documentation updates made to reflect the new leverage sampling diagnostic logging system and interactive launcher.

## Files Updated

### 1. README.md
**Changes**:
- ✅ Added "Option 1: Interactive Launcher (Recommended)" section
- ✅ Updated Quick Start to show both interactive and script-based workflows
- ✅ Added results structure showing `sampling_data/` directory
- ✅ Highlighted persistent caching and diagnostic logging features

**Key additions**:
- Interactive launcher usage: `python scripts/interactive_run.py`
- `log_sampling_diagnostics: True` configuration flag
- Results directory structure with diagnostic data

### 2. docs/ARCHITECTURE.md
**Changes**:
- ✅ Updated directory map with new components (🆕 markers)
- ✅ Added `scripts/interactive_run.py` as recommended entry point
- ✅ Documented new utilities: `sampling_logger.py`, `persistent_cache.py`, `interactive_ui.py`
- ✅ Added `src/cache/` directory structure
- ✅ Added `analysis/notebooks/` with leverage_sampling_explorer.ipynb
- ✅ Added complete "Diagnostic Logging System" section

**New sections**:
- **Persistent caching**: Disk-based matrix storage for instant re-runs
- **Diagnostic Logging System**: Purpose, enabling, data format, analysis notebook
  - How to enable diagnostics (3 methods)
  - Data format (`.npz` files with leverage scores + sampling probs)
  - Analysis notebook usage
  - Implementation details (files, design choices)

### 3. docs/LEVERAGED_SAMPLING.md
**Changes**:
- ✅ Added "Diagnostic Logging (New!)" section before "Diagnostic Framework"
- ✅ Documented what gets logged (leverage scores, Phase 2 probs)
- ✅ Multiple ways to enable logging (interactive, config, script)
- ✅ Data format and location
- ✅ Reference to exploration notebook

**Key points**:
- Diagnostic data saved only for p ≥ theoretical minimum
- Sparse representation for efficiency
- Ground truth computed during analysis (not during runs)
- Link to ANALYSIS_GUIDES.md for full notebook documentation

### 4. docs/ANALYSIS_GUIDES.md
**Changes**:
- ✅ Added comprehensive "Leverage Sampling Explorer" section
- ✅ Documented notebook purpose, usage, and outputs
- ✅ Added interpretation guidelines (high/medium/low correlation)
- ✅ Included example findings from actual experiments
- ✅ Common issues and troubleshooting
- ✅ Advanced usage examples

**New content**:
- **Purpose**: Validate leverage sampling by comparing estimated vs ground truth scores
- **Research question**: Does Phase 1 provide reliable estimates?
- **Quick start**: Configuration and running
- **Requirements**: What experiments need for diagnostics
- **Visualization**: Three-panel figure description
- **Interpreting results**: Correlation thresholds and meanings
- **Example findings**: Real correlation = -0.0728 at p=0.1438
- **Common issues**: Troubleshooting guide
- **Advanced usage**: Multi-p-value comparison, data export

## New Concepts Documented

### 1. Interactive Launcher
- Menu-driven interface for experiment configuration
- Persistent caching for instant matrix loading
- Last run re-execution with one keystroke
- Automatic plotting and diagnostics

### 2. Diagnostic Logging System
- **Purpose**: Validate leverage sampling algorithm
- **Data logged**: Estimated leverage scores + Phase 2 sampling probabilities
- **Storage**: Sparse `.npz` files per p-value
- **Design**: Log last bootstrap only, defer ground truth to analysis

### 3. Leverage Sampling Explorer Notebook
- **Validation tool**: Compare estimated vs ground truth leverage scores
- **Three plots**: Matrix heatmap, scatter comparison, sampling probs heatmap
- **Key metric**: Correlation coefficient validates Phase 1 quality
- **Research finding**: Low correlation near threshold validates theory

### 4. Persistent Caching
- **Location**: `src/cache/{experiment_params}/`
- **Cached data**: Similarity matrix, observations, tree, Fiedler reference
- **Benefit**: Instant loading for re-runs (no recomputation)
- **Integration**: Automatic with interactive launcher

## Experiment Workflow (Updated)

### Before (Script-based only):
1. Edit `SWEEP_CONFIG` in `run_experiment.py`
2. Run script
3. Results saved
4. (No easy re-run or caching)

### After (Interactive + Diagnostics):
1. **Option A**: Run `python scripts/interactive_run.py`
   - Menu shows last run + cached matrices
   - Select cached matrix → instant loading!
   - Or create new configuration interactively
   
2. **Option B**: Script-based (enhanced)
   - Add `log_sampling_diagnostics: True` to config
   - Persistent caching enabled by default
   
3. Results include:
   - Standard metrics (JSON, plots)
   - **NEW**: `sampling_data/` with diagnostic `.npz` files
   
4. Analysis:
   - Open `leverage_sampling_explorer.ipynb`
   - Visualize leverage score quality
   - Validate theoretical predictions

## Key Research Findings (Documented)

From n=1024, p=0.1438 experiment:
- **Correlation**: -0.0728 (nearly zero!)
- **Sampling rate**: 3.54%
- **Interpretation**: Phase 1 uniform sampling with only 3.54% coverage doesn't reliably estimate leverage scores
- **Validates**: Theoretical minimum budget requirement (4·n·r·log(n))
- **Recommendation**: Try higher p-values (p=0.6158) to see correlation improve

## File References Added

Cross-references between documentation files:
- README.md → ARCHITECTURE.md, LEVERAGED_SAMPLING.md, ANALYSIS_GUIDES.md
- ARCHITECTURE.md → ANALYSIS_GUIDES.md (for notebook details)
- LEVERAGED_SAMPLING.md → ANALYSIS_GUIDES.md (for explorer notebook)
- ANALYSIS_GUIDES.md → ARCHITECTURE.md, LEVERAGED_SAMPLING.md

## Implementation Details Documented

### Files involved:
1. `src/core/sampling/leveraged/sampler.py` - Extended metrics tracking
2. `src/utils/sampling_logger.py` - Save functions
3. `src/runners/bootstrap_sweep.py` - Integration point
4. `scripts/interactive_run.py` - Interactive launcher
5. `src/utils/persistent_cache.py` - Caching system
6. `analysis/notebooks/leverage_sampling_explorer.ipynb` - Exploration tool

### Design choices documented:
- Save last bootstrap only (disk efficiency)
- Sparse matrix representation
- Ground truth deferred to analysis
- Symmetric matrices = one set of scores

## Documentation Quality Improvements

1. **Consistency**: All docs use same terminology and structure
2. **Cross-linking**: Clear references between related sections
3. **Examples**: Real experiment data and findings included
4. **Troubleshooting**: Common issues documented with solutions
5. **Progressive disclosure**: Quick start → Details → Advanced usage

## Next Steps for Users

The updated documentation enables users to:

1. ✅ Quickly start with interactive launcher
2. ✅ Enable diagnostic logging for validation
3. ✅ Analyze leverage sampling quality
4. ✅ Understand theoretical predictions
5. ✅ Troubleshoot common issues
6. ✅ Extend analysis for custom research questions

## Summary

The documentation now comprehensively covers:
- ✅ Interactive launcher workflow
- ✅ Persistent caching system
- ✅ Diagnostic logging mechanism
- ✅ Analysis notebook usage
- ✅ Research findings and validation
- ✅ Troubleshooting and best practices

All documentation is synchronized and cross-referenced for easy navigation.
