# Interactive STDR Launcher Guide

## Quick Start

Launch the interactive menu system:

```bash
cd sub_sampled_fielder_vec
python scripts/interactive_run.py
```

## Features

### 🎨 Gemini-Style UI
- Colorful gradient logo (blue → cyan → magenta)
- Clean menu system with colored options
- Inline parameter editing with defaults

### 📋 Main Menu Options

When you launch, you'll see:

```
    ███████╗████████╗██████╗ ██████╗
    ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗
    ███████╗   ██║   ██║  ██║██████╔╝
    ╚════██║   ██║   ██║  ██║██╔══██╗
    ███████║   ██║   ██████╔╝██║  ██║
    ╚══════╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝
    Subsampled Spectral Tree Recovery

Main Menu
─────────
Last Run (from last_run.json):
  → n=2048, L=10000, μ=0.1, balanced_binary, 10 bootstraps
  [r] Re-run this config

Cached Matrices:
  [1] n=2048, L=10000, μ=0.1, balanced_binary
  [2] n=4096, L=5000, μ=0.05, kingman

  [n] Create new matrix
  [q] Quit

Choice: _
```

### Option 1: Re-run Last Configuration (`r`)
- Automatically loads your previous experiment config
- Saved in `last_run.json` after each run
- No need to re-enter parameters

### Option 2: Use Cached Matrix (`1`, `2`, ...)
- Select from pre-computed matrices
- Matrix data is loaded from `cache/` directory
- Configure only experiment parameters (p-values, bootstraps, etc.)
- Saves computation time - no need to regenerate tree/sequences/matrix

### Option 3: Create New Matrix (`n`)
- Interactive parameter entry with inline editing
- Press Enter to keep default values
- Creates new tree, sequences, and matrix
- Automatically saved to cache for future use

## Parameter Configuration

When creating a new matrix, you'll be prompted for:

### Matrix Parameters
```
n_taxa [2048]: 4096         ← Type new value or press Enter for default
sequence_length [10000]: ⏎
mutation_rate [0.1]: 0.15
tree_model [balanced_binary]: kingman
```

### Tree-Specific Parameters
- **balanced_binary**: `edge_length` (default: 1.0)
- **kingman/kingman_mean**: `pop_size` (default: 1.0)
- **lopsided**: `edge_length` (default: 1.0)

### Experiment Parameters
```
bootstrap_reps [10]: 20
num_workers [8]: 16
p_values: Use default (20 points logspace)? [Y/n]
sampling_method [uniform]: leveraged
```

### Leveraged Sampling Parameters (if selected)
```
sampling_theta [0.7]: ⏎
sampling_target_rank [2]: ⏎
IALM max_iter [500]: ⏎
IALM tolerance [1e-4]: ⏎
```

## Directory Structure

After using the interactive launcher:

```
sub_sampled_fielder_vec/
├── scripts/
│   ├── interactive_run.py     ← NEW: Interactive launcher (use this!)
│   └── run_experiment.py      ← OLD: Legacy script (reference only)
│
├── cache/                      ← Cached matrices
│   ├── n2048_L10000_mu0.100_balanced_binary_JC69/
│   │   ├── tree.npz
│   │   ├── observations.npz
│   │   ├── similarity_matrix.npz
│   │   ├── fiedler_ref.npz
│   │   └── metadata.json
│   └── n4096_L5000_mu0.050_kingman_JC69/
│       └── ...
│
├── last_run.json              ← Your last experiment config
│
├── results/                    ← Experiment outputs
│   └── 20260124-HHMMSS-prefix/
│       ├── sweep_config.json
│       ├── n2048_L10000/
│       └── partition_agreement.png
│
└── src/utils/
    └── interactive_ui.py      ← UI components (logo, colors, inputs)
```

## Tips

### 💡 First-Time Setup
1. Run `python scripts/interactive_run.py`
2. Choose `[n]` to create your first matrix
3. Configure parameters
4. Matrix gets cached automatically
5. Next time, select from cached matrices!

### ⚡ Speed Up Experiments
- Pre-compute matrices for common configs
- Use cached matrices for parameter sweeps
- Matrix generation (tree + sequences + similarity + Fiedler) happens once
- Experiment sweeps reuse the cached data

### 🔄 Re-running Experiments
- After any run, config is saved to `last_run.json`
- Next launch shows `[r] Re-run` option
- Perfect for debugging or minor tweaks

### 🗑️ Cache Management
View cached experiments:
```python
from src.utils.persistent_cache import list_cached_experiments
cached = list_cached_experiments()
for item in cached:
    print(f"{item['cache_key']}: {item['metadata']}")
```

Clear cache:
```python
from src.utils.persistent_cache import clear_cache
clear_cache()  # Clear all
clear_cache("n2048_L10000_mu0.100_balanced_binary_JC69")  # Clear specific
```

## Advanced: TODO Section

### Custom Parameter Input (TODO in `build_config_from_cache()`)

Currently at **interactive_run.py:103**, there's a section marked for enhancement:

```python
# TODO(human): Implement interactive parameter configuration
# Ask user for: p_values, bootstrap_reps, num_workers, sampling_method, etc.
```

**What to implement:**
- More granular control over p-values (custom ranges, spacing)
- Additional metrics configuration (coherence_k, num_gaps)
- Guardrails configuration
- Output preferences

**Guidance:**
- Use the existing `get_input()` and `get_choice()` helpers from `interactive_ui.py`
- Follow the pattern in `create_new_config()` (lines 169-244)
- Balance between flexibility and usability (too many prompts = bad UX)

## Migration from Old `run_experiment.py`

**Old workflow:**
```python
# Edit SWEEP_CONFIG in run_experiment.py
SWEEP_CONFIG = {
    "tree_model": "balanced_binary",
    "taxa_values": [2048],
    # ... 20 more parameters
}
python scripts/run_experiment.py
```

**New workflow:**
```bash
python scripts/interactive_run.py
# Interactive prompts guide you through configuration
# No file editing required!
```

## Troubleshooting

### Colors not showing?
- Make sure you're using a modern terminal (macOS Terminal, iTerm2, Windows Terminal)
- Older terminals may not support ANSI color codes

### Cache not found?
- Cache directory is created automatically in `sub_sampled_fielder_vec/cache/`
- First run with `use_persistent_cache=True` creates cache

### Last run not showing?
- `last_run.json` is created after first successful experiment
- Located in `sub_sampled_fielder_vec/last_run.json`

---

**Enjoy the new interactive STDR launcher! No more config file editing! 🎉**
