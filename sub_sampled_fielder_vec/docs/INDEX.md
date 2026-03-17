# Documentation Index

**Navigation guide for all STDR documentation** - Start here to find what you need!

---

## 🚀 Getting Started (Read These First)

If you're new to the project, read these in order:

1. **[../README.md](../README.md)** - Project overview, scientific context, and quick start
2. **[INTERACTIVE_GUIDE.md](INTERACTIVE_GUIDE.md)** - How to use the interactive launcher (easiest way to run experiments)
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Codebase structure and system design *(essential for AI agents and developers)*

---

## 📖 Core User Guides

### Running Experiments
- **[INTERACTIVE_GUIDE.md](INTERACTIVE_GUIDE.md)** - Interactive launcher with caching and menus (recommended)
- **[CONFIGURATION.md](CONFIGURATION.md)** - Complete configuration system reference

### Sampling Methods
Choose your sampling strategy:

| Method | Speed | Accuracy | When to Use | Documentation |
|--------|-------|----------|-------------|---------------|
| **Uniform** | ⚡ Fastest | Baseline | Default, quick tests | Built-in |
| **LDS** | ⚡ Fast | High | Large trees (n>500), production | **[LDS_SAMPLING.md](LDS_SAMPLING.md)** ⭐ |
| **Leveraged (IALM)** | 🐌 Slow | Highest | Small trees (n<200), max accuracy | **[LEVERAGED_SAMPLING.md](LEVERAGED_SAMPLING.md)** |

**Quick decision:** Use **LDS** for most cases - it's 10-100x faster than IALM with comparable accuracy.

### Analysis & Results
- **[ANALYSIS_GUIDES.md](ANALYSIS_GUIDES.md)** - How to analyze results, use notebooks, and create visualizations
- **[METRICS.md](METRICS.md)** - All metrics definitions and data formats

---

## 🔬 Advanced Topics

### Sampling Methods (Deep Dive)

#### LDS Sampling (Recommended for Large-Scale)
- **[LDS_SAMPLING.md](LDS_SAMPLING.md)** - Complete LDS user guide
  - Quick start (interactive + programmatic)
  - Three-phase pipeline explanation
  - Configuration and tuning
  - Performance benchmarks (10-100x speedup)
  - Troubleshooting
  - Mathematical background
- **[LDS_MIGRATION.md](LDS_MIGRATION.md)** - Technical migration details from IALM to LDS *(for developers)*

#### Leveraged Sampling (IALM)
- **[LEVERAGED_SAMPLING.md](LEVERAGED_SAMPLING.md)** - Complete IALM guide
  - Algorithm overview
  - Phase 1/2 sampling
  - Matrix completion with IALM
  - Diagnostic logging
  - Research mode vs safe mode
  - Configuration flow

### Architecture & Development
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture, directory structure, data flow
- **[CONFIGURATION.md](CONFIGURATION.md)** - Configuration system internals

---

## 📊 Analysis Documentation

### Comparison Analysis
Location: `../analysis/comparison/`

- **[../analysis/comparison/README.md](../analysis/comparison/README.md)** - Comparison analysis overview
- **[../analysis/comparison/QUICKSTART.md](../analysis/comparison/QUICKSTART.md)** - Quick start guide
- **[../analysis/comparison/PHASE_TRANSITION_README.md](../analysis/comparison/PHASE_TRANSITION_README.md)** - Phase transition analysis

### Leveraged Sampling Diagnostics
Location: `../analysis/leveraged_sampling_analysis/docs/`

- **[../analysis/leveraged_sampling_analysis/docs/README_DIAGNOSTICS.md](../analysis/leveraged_sampling_analysis/docs/README_DIAGNOSTICS.md)** - Diagnostic analysis
- **[../analysis/leveraged_sampling_analysis/docs/QUICK_FIX.md](../analysis/leveraged_sampling_analysis/docs/QUICK_FIX.md)** - Troubleshooting guide

### Spectral Analysis
Location: `../analysis/spectral_analysis/`

- **[../analysis/spectral_analysis/sweep_params_analysis/README.md](../analysis/spectral_analysis/sweep_params_analysis/README.md)** - Parameter sweep analysis
- **[../analysis/spectral_analysis/target_quality_anlysis/README.md](../analysis/spectral_analysis/target_quality_anlysis/README.md)** - Quality analysis

---

## 📝 Meta Documentation

- **[RECENT_CHANGES.md](RECENT_CHANGES.md)** - Recent configuration system changes
- **[../DOCUMENTATION_UPDATE_SUMMARY.md](../DOCUMENTATION_UPDATE_SUMMARY.md)** - Previous documentation update log

---

## 🎯 Quick Reference by Task

### "I want to run my first experiment"
→ **[INTERACTIVE_GUIDE.md](INTERACTIVE_GUIDE.md)** then `python scripts/interactive_run.py`

### "I need fast sampling for large trees (n>500)"
→ **[LDS_SAMPLING.md](LDS_SAMPLING.md)**

### "I need maximum accuracy for small trees"
→ **[LEVERAGED_SAMPLING.md](LEVERAGED_SAMPLING.md)**

### "How do I analyze my results?"
→ **[ANALYSIS_GUIDES.md](ANALYSIS_GUIDES.md)**

### "What do these metrics mean?"
→ **[METRICS.md](METRICS.md)**

### "How do I configure X?"
→ **[CONFIGURATION.md](CONFIGURATION.md)**

### "I'm an AI agent - where do I start?"
→ **[ARCHITECTURE.md](ARCHITECTURE.md)** (complete system map)

### "How does LDS differ from IALM?"
→ **[LDS_SAMPLING.md](LDS_SAMPLING.md)** section "Comparison with Related Methods"

### "I'm debugging sampling issues"
→ **[LEVERAGED_SAMPLING.md](LEVERAGED_SAMPLING.md)** section "Troubleshooting"

---

## 📚 Documentation Organization

```
docs/
├── INDEX.md ← You are here!
│
├── Getting Started
│   ├── ../README.md
│   ├── INTERACTIVE_GUIDE.md
│   └── ARCHITECTURE.md
│
├── Sampling Methods
│   ├── LDS_SAMPLING.md (recommended)
│   ├── LDS_MIGRATION.md (technical)
│   └── LEVERAGED_SAMPLING.md (IALM)
│
├── Configuration & Metrics
│   ├── CONFIGURATION.md
│   └── METRICS.md
│
├── Analysis
│   └── ANALYSIS_GUIDES.md
│
└── Meta
    ├── RECENT_CHANGES.md
    └── ../DOCUMENTATION_UPDATE_SUMMARY.md
```

---

## 🤖 For AI Agents

**Priority reading order:**
1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system map, entry points, data flow
2. **[CONFIGURATION.md](CONFIGURATION.md)** - All configuration parameters
3. **[LDS_SAMPLING.md](LDS_SAMPLING.md)** or **[LEVERAGED_SAMPLING.md](LEVERAGED_SAMPLING.md)** - Sampling method details
4. **[METRICS.md](METRICS.md)** - Data formats and metrics

**Key files for code navigation:**
- System entry points: `scripts/interactive_run.py`, `scripts/run_experiment.py`
- Core algorithms: `src/core/sampling/`, `src/core/fiedler_computer.py`
- Configuration: `src/config/base_config.py`, `src/config/presets.py`
- Runners: `src/runners/bootstrap_sweep.py`, `src/runners/experiment_runner.py`

---

## 📎 Cross-References

Documents frequently reference each other:
- **README** → **ARCHITECTURE**, **INTERACTIVE_GUIDE**, **LEVERAGED_SAMPLING**, **ANALYSIS_GUIDES**
- **ARCHITECTURE** → **CONFIGURATION**, **ANALYSIS_GUIDES**
- **LDS_SAMPLING** → **LEVERAGED_SAMPLING** (comparison), **CONFIGURATION**
- **LEVERAGED_SAMPLING** → **ANALYSIS_GUIDES** (diagnostics), **CONFIGURATION**
- **INTERACTIVE_GUIDE** → **CONFIGURATION**, **ARCHITECTURE**

---

**Last Updated:** 2026-02-13
**Total Documents:** 19 markdown files
**Maintained by:** STDR Development Team

---

*Can't find what you need? Check the table of contents in individual guides - they're comprehensive!*
