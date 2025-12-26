# Migration Guide: Legacy Config → Structured Config

## Overview

This document describes the changes needed to migrate from the legacy `Config` class to the new `StructuredConfig` system.

## Summary of Changes

### 1. Config Structure
**OLD (Flat):**
```python
Config(
    num_taxa=1024,
    sequence_length=1000,
    mutation_rate=0.1,
    tree_model=lambda n: spectraltree.balanced_binary(n),
    seq_model=lambda: spectraltree.Jukes_Cantor(),
    p_values=(0.1, 0.5, 1.0),
    bootstrap_reps=100,
    ...
)
```

**NEW (Hierarchical):**
```python
StructuredConfig(
    tree=TreeConfig(model="balanced_binary", params={"num_taxa": 1024}),
    sequence=SequenceConfig(model="JC69", len=1000, params={"mutation_rate": 0.1}),
    experiment=ExperimentConfig(p_values=[0.1, 0.5, 1.0], bootstrap_reps=100, ...),
    ...
)
```

### 2. Code Changes Required

#### In `bootstrap_sweep.py`:

**OLD:**
```python
tree = cfg.tree_model(n_taxa)
seq_model = cfg.seq_model()
```

**NEW:**
```python
from utils.models import get_tree_factory, get_sequence_factory

tree_factory = get_tree_factory(cfg.tree.model, cfg.tree.params)
seq_factory = get_sequence_factory(cfg.sequence.model, cfg.sequence.params)

tree = tree_factory()
seq_model = seq_factory()
```

**Accessing Parameters:**
```python
# OLD
cfg.num_taxa
cfg.sequence_length
cfg.mutation_rate
cfg.p_values
cfg.bootstrap_reps

# NEW
cfg.get_num_taxa()  # or cfg.tree.params["num_taxa"]
cfg.get_sequence_length()  # or cfg.sequence.len
cfg.get_mutation_rate()  # or cfg.sequence.params["mutation_rate"]
cfg.experiment.p_values
cfg.experiment.bootstrap_reps
```

####In `experiment_runner.py`:

**Update type hints:**
```python
# OLD
from utils.experiment_config import Config

def __init__(self, cfg: Config):
    ...

# NEW
from utils.config import StructuredConfig

def __init__(self, cfg: StructuredConfig):
    ...
```

**Access nested fields:**
```python
# OLD
cfg.run_name
cfg.display_mode
cfg.num_workers

# NEW
cfg.experiment.run_name
cfg.experiment.display_mode
cfg.experiment.num_workers
```

#### In `presets.py`:

Completely rewrite using `StructuredConfig`. Examples:

```python
from utils.config import StructuredConfig, TreeConfig, SequenceConfig, ExperimentConfig

def create_quick_test_config() -> StructuredConfig:
    return StructuredConfig(
        tree=TreeConfig(model="balanced_binary", params={"num_taxa": 32}),
        sequence=SequenceConfig(model="JC69", len=300, params={"mutation_rate": 0.1}),
        experiment=ExperimentConfig(
            p_values=[0.01, 0.1, 0.5, 1.0],
            bootstrap_reps=5,
            run_name="quick_test",
            display_mode="debug"
        )
    )
```

### 3. Helper Methods in StructuredConfig

The new config provides convenience methods:

```python
cfg.get_num_taxa() -> int
cfg.get_sequence_length() -> int
cfg.get_mutation_rate() -> float
cfg.get_tree_model_name() -> str
cfg.get_seq_model_name() -> str
cfg.summary() -> str  # Human-readable summary
cfg.to_json_file(path)
cfg.from_json_file(path) -> StructuredConfig
```

### 4. Validation Benefits

Pydantic provides automatic validation:

```python
# This will raise ValidationError with clear message:
StructuredConfig(
    tree=TreeConfig(model="invalid_model", params={"num_taxa": 128}),
    ...
)
# Error: model must be one of ['balanced_binary', 'lopsided', 'kingman', 'birth_death']

# This will raise ValidationError:
TreeConfig(model="balanced_binary", params={"num_taxa": 100})
# Error: balanced_binary requires num_taxa to be power of 2
```

### 5. Sweep Configuration Migration

**OLD (Manual logic in experiment_runner.py):**
```python
if cfg.taxa_values is not None:
    for n_taxa in cfg.taxa_values:
        # Run experiment with n_taxa
```

**NEW (Use SweepType utilities):**
```python
from utils.config.sweeps import SweepType

base_cfg = create_standard_config()
sweep_configs = SweepType.taxa_sweep([128, 256, 512], base_cfg)

for cfg in sweep_configs:
    runner = ExperimentRunner(cfg)
    runner.run()
```

## Step-by-Step Migration Plan

1. ✅ Create new config system (DONE)
2. ✅ Create model registries (DONE)
3. ✅ Create sweep utilities (DONE)
4. ⏳ Update `experiment/presets.py`
5. ⏳ Update `experiment/bootstrap_sweep.py`
6. ⏳ Update `experiment/experiment_runner.py`
7. ⏳ Update `main_taxa_sweep.py`
8. ⏳ Test complete system
9. ⏳ Mark `utils/experiment_config.py` as DEPRECATED

## Testing Strategy

After migration:

1. Run `examples/basic_config_examples.py` - should pass ✅
2. Run quick test experiment with new config
3. Compare results with old config (should match exactly)
4. Run full experiment suite

## Backward Compatibility

NO - this is a hard cutover. All existing code using `Config` must be updated to use `StructuredConfig`.

Benefits:
- Cleaner, more maintainable code
- Type safety and validation
- JSON serialization
- Better error messages
- Easier to extend

## Questions?

See `examples/basic_config_examples.py` for working examples of the new system.
