#!/usr/bin/env python3
"""
Run perfox truncation experiment for balanced_binary trees.

Tests truncation threshold of 1e-4 on n=512, 1024, 2048, 4096 with uniform sampling.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path

def main():
    """Run perfox truncation experiment."""

    # Configuration for perfox experiment
    config = {
        "tree_model": "balanced_binary",
        "taxa_values": [512, 1024, 2048, 4096],  # Batch mode
        "sequence_length_values": [10000],
        "mutation_rate": 0.1,
        "tree_params": {"edge_length": 1.0},
        "bootstrap_reps": 10,
        "num_workers": 8,
        "p_values": list(np.logspace(-4, 0, 20)),  # 20 points from 1e-4 to 1.0
        "use_middle_out": False,
        "sampling_method": "uniform",
        "display_mode": "progress",
        "guardrails_enabled": False,
        "run_name_prefix": "perfox",  # Identifiable prefix
        "use_persistent_cache": True,
        "truncation_threshold": 1e-4,  # Key parameter: truncate values below 1e-4
    }

    print("\n" + "="*80)
    print("PERFOX Truncation Experiment")
    print("="*80)
    print(f"Tree model: {config['tree_model']}")
    print(f"n_taxa: {config['taxa_values']}")
    print(f"Sampling: {config['sampling_method']}")
    print(f"Truncation threshold: {config['truncation_threshold']:.0e}")
    print(f"Bootstrap reps: {config['bootstrap_reps']}")
    print("="*80)
    print()

    # Run the experiment directly (bypass confirmation)
    from pathlib import Path
    from src.runners.experiment_runner_utils import (
        extract_config_values,
        generate_run_prefix,
        setup_experiment_directory,
        run_single_experiment,
        auto_generate_plots,
    )

    # Save configuration for future re-runs
    import time
    import json
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR
    LAST_RUN_PATH = PROJECT_ROOT / "last_run.json"

    config_with_timestamp = config.copy()
    config_with_timestamp['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LAST_RUN_PATH, 'w') as f:
        json.dump(config_with_timestamp, f, indent=2)
    print(f"[92m✓[0m Configuration saved to {LAST_RUN_PATH}")

    # Extract and validate config
    WIDE_SWEEP_P_VALUES = list(np.logspace(-4, 0, 20))
    cfg_vals = extract_config_values(config, WIDE_SWEEP_P_VALUES)

    # Generate run prefix
    prefix = generate_run_prefix(config, cfg_vals["mutation_rate"], cfg_vals["taxa_values"])
    print(f"[92m✓[0m Run prefix: {prefix}")

    # Setup experiment directory
    base_dir = setup_experiment_directory(
        config, prefix, config,
        script_dir=str(SCRIPT_DIR / "scripts")
    )
    print(f"[92m✓[0m Output directory: {base_dir}")

    # Run experiments
    multi_run_results = []
    for n_taxa in cfg_vals["taxa_values"]:
        for seq_len in cfg_vals["sequence_length_values"]:
            print(f"\n[1m[96mRunning experiment: n={n_taxa}, L={seq_len}[0m")
            result = run_single_experiment(
                config, base_dir, n_taxa, seq_len,
                cfg_vals["tree_model"], cfg_vals["mutation_rate"],
                cfg_vals["p_values"], cfg_vals["bootstrap_reps"],
                cfg_vals["num_workers"], cfg_vals["use_middle_out"],
                prefix, cfg_vals["tree_kwargs"]
            )
            multi_run_results.append(result)

    # Print summary
    print("\n" + "="*80)
    print(f"[92m✓[0m Completed {cfg_vals['tree_model']} experiments")
    for entry in multi_run_results:
        print(f"  n={entry['num_taxa']:>4}, L={entry['sequence_length']:>5} → {entry['run_dir']}")
    print("="*80)

    # Auto-generate plots
    auto_generate_plots(base_dir)

    print("\n" + "="*80)
    print("Experiment completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
