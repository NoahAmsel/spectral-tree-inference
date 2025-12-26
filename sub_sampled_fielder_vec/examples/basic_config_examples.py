"""
Basic examples demonstrating the new structured config system.

This file shows how to create, modify, save, and load configurations for
sub-sampled STDR experiments using the new Pydantic-based system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import (
    StructuredConfig, TreeConfig, SequenceConfig, ExperimentConfig
)
from src.config.sweeps import (
    SweepType, get_default_config, quick_test_config,
    standard_config, large_scale_config
)
import numpy as np


def example_1_create_basic_config():
    """Example 1: Create a basic configuration programmatically."""
    print("\n" + "="*80)
    print("Example 1: Create Basic Configuration")
    print("="*80)

    cfg = StructuredConfig(
        tree=TreeConfig(
            model="balanced_binary",
            params={"num_taxa": 256, "edge_length": 1.0}
        ),
        sequence=SequenceConfig(
            model="JC69",
            len=1000,
            params={"mutation_rate": 0.1}
        ),
        experiment=ExperimentConfig(
            p_values=[0.001, 0.01, 0.1, 0.5, 1.0],
            bootstrap_reps=50,
            seed=42,
            run_name="my_first_experiment"
        )
    )

    print(cfg.summary())
    return cfg


def example_2_use_presets():
    """Example 2: Use predefined preset configurations."""
    print("\n" + "="*80)
    print("Example 2: Use Preset Configurations")
    print("="*80)

    # Quick test config (small, fast)
    cfg_quick = quick_test_config()
    print("Quick test config:")
    print(f"  - Taxa: {cfg_quick.get_num_taxa()}")
    print(f"  - Sequence length: {cfg_quick.get_sequence_length()}")
    print(f"  - Bootstrap reps: {cfg_quick.experiment.bootstrap_reps}")

    # Standard config (medium-sized)
    cfg_standard = standard_config()
    print(f"\nStandard config:")
    print(f"  - Taxa: {cfg_standard.get_num_taxa()}")
    print(f"  - Sequence length: {cfg_standard.get_sequence_length()}")
    print(f"  - Bootstrap reps: {cfg_standard.experiment.bootstrap_reps}")

    # Large scale config (optimized for big experiments)
    cfg_large = large_scale_config()
    print(f"\nLarge scale config:")
    print(f"  - Taxa: {cfg_large.get_num_taxa()}")
    print(f"  - Sequence length: {cfg_large.get_sequence_length()}")
    print(f"  - Num workers: {cfg_large.experiment.num_workers}")
    print(f"  - Middle-out: {cfg_large.experiment.use_middle_out}")

    return cfg_quick, cfg_standard, cfg_large


def example_3_different_models():
    """Example 3: Create configs with different tree and sequence models."""
    print("\n" + "="*80)
    print("Example 3: Different Tree and Sequence Models")
    print("="*80)

    # Kingman tree with HKY model
    cfg_kingman_hky = StructuredConfig(
        tree=TreeConfig(
            model="kingman",
            params={"num_taxa": 512, "pop_size": 1.0}
        ),
        sequence=SequenceConfig(
            model="HKY",
            len=1000,
            params={
                "mutation_rate": 0.1,
                "kappa": 3.0  # Higher transition/transversion ratio
            }
        ),
        experiment=ExperimentConfig(
            p_values=list(np.logspace(-4, 0, 15)),
            bootstrap_reps=100,
            seed=42,
            run_name="kingman_hky"
        )
    )

    print("Kingman tree + HKY model:")
    print(f"  - Tree: {cfg_kingman_hky.tree.model}")
    print(f"  - Pop size: {cfg_kingman_hky.tree.params['pop_size']}")
    print(f"  - Sequence model: {cfg_kingman_hky.sequence.model}")
    print(f"  - Kappa: {cfg_kingman_hky.sequence.params['kappa']}")

    # Birth-death tree with GTR model
    cfg_bd_gtr = StructuredConfig(
        tree=TreeConfig(
            model="birth_death",
            params={
                "num_taxa": 256,
                "birth_rate": 0.7,
                "death_rate": 0.3
            }
        ),
        sequence=SequenceConfig(
            model="GTR",
            len=1000,
            params={
                "mutation_rate": 0.1,
                "transition_rates": [1, 3, 1, 1, 3, 1]  # Favor transitions
            }
        ),
        experiment=ExperimentConfig(
            p_values=list(np.logspace(-4, 0, 15)),
            bootstrap_reps=100,
            seed=42,
            run_name="birth_death_gtr"
        )
    )

    print("\nBirth-death tree + GTR model:")
    print(f"  - Tree: {cfg_bd_gtr.tree.model}")
    print(f"  - Birth rate: {cfg_bd_gtr.tree.params['birth_rate']}")
    print(f"  - Death rate: {cfg_bd_gtr.tree.params['death_rate']}")
    print(f"  - Sequence model: {cfg_bd_gtr.sequence.model}")
    print(f"  - Transition rates: {cfg_bd_gtr.sequence.params['transition_rates']}")

    return cfg_kingman_hky, cfg_bd_gtr


def example_4_sweeps():
    """Example 4: Generate sweep configurations."""
    print("\n" + "="*80)
    print("Example 4: Parameter Sweeps")
    print("="*80)

    # Taxa sweep
    taxa_configs = SweepType.taxa_sweep([128, 256, 512, 1024])
    print(f"Taxa sweep: {len(taxa_configs)} configurations")
    for cfg in taxa_configs:
        print(f"  - {cfg.experiment.run_name}: {cfg.get_num_taxa()} taxa")

    # Tree model sweep
    tree_configs = SweepType.tree_model_sweep()
    print(f"\nTree model sweep: {len(tree_configs)} configurations")
    for cfg in tree_configs:
        print(f"  - {cfg.experiment.run_name}: {cfg.tree.model}")

    # Sequence model sweep
    seq_configs = SweepType.sequence_model_sweep()
    print(f"\nSequence model sweep: {len(seq_configs)} configurations")
    for cfg in seq_configs:
        print(f"  - {cfg.experiment.run_name}: {cfg.sequence.model}")

    return taxa_configs, tree_configs, seq_configs


def example_5_save_and_load():
    """Example 5: Save and load configurations from JSON."""
    print("\n" + "="*80)
    print("Example 5: Save and Load Configurations")
    print("="*80)

    # Create a config
    cfg = StructuredConfig(
        tree=TreeConfig(model="balanced_binary", params={"num_taxa": 512}),
        sequence=SequenceConfig(model="HKY", len=2000, params={"mutation_rate": 0.15, "kappa": 2.5}),
        experiment=ExperimentConfig(
            p_values=[0.01, 0.1, 0.5],
            bootstrap_reps=20,
            run_name="saved_config_example"
        )
    )

    # Save to JSON
    json_path = "/tmp/example_config.json"
    cfg.to_json_file(json_path)
    print(f"Saved config to: {json_path}")

    # Load from JSON
    loaded_cfg = StructuredConfig.from_json_file(json_path)
    print(f"Loaded config:")
    print(f"  - Taxa: {loaded_cfg.get_num_taxa()}")
    print(f"  - Sequence model: {loaded_cfg.sequence.model}")
    print(f"  - Kappa: {loaded_cfg.sequence.params.get('kappa')}")
    print(f"  - Run name: {loaded_cfg.experiment.run_name}")

    # Verify they're equal
    assert cfg.dict() == loaded_cfg.dict(), "Configs should be identical!"
    print("✓ Loaded config matches original")

    return cfg, loaded_cfg


def example_6_modify_config():
    """Example 6: Modify an existing configuration."""
    print("\n" + "="*80)
    print("Example 6: Modify Existing Configuration")
    print("="*80)

    # Start with default
    cfg = get_default_config()
    print(f"Original config: {cfg.get_num_taxa()} taxa, {cfg.sequence.len} sites")

    # Modify via dictionary
    cfg_dict = cfg.dict()
    cfg_dict["tree"]["params"]["num_taxa"] = 2048
    cfg_dict["sequence"]["len"] = 5000
    cfg_dict["experiment"]["bootstrap_reps"] = 200

    modified_cfg = StructuredConfig(**cfg_dict)
    print(f"Modified config: {modified_cfg.get_num_taxa()} taxa, {modified_cfg.sequence.len} sites")
    print(f"  - Bootstrap reps: {modified_cfg.experiment.bootstrap_reps}")

    return modified_cfg


def main():
    """Run all examples."""
    print("\n" + "#"*80)
    print("# STRUCTURED CONFIG SYSTEM - EXAMPLES")
    print("#"*80)

    example_1_create_basic_config()
    example_2_use_presets()
    example_3_different_models()
    example_4_sweeps()
    example_5_save_and_load()
    example_6_modify_config()

    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
