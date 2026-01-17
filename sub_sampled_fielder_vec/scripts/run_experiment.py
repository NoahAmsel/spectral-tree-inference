"""Main entry point for bootstrap sweep experiments.

Edit the SWEEP_CONFIG dict below to change what gets executed. This keeps the
script dead-simple while still letting us reuse ExperimentRunner.
"""
import os
import sys
from typing import Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.runners.experiment_runner_utils import (
    extract_config_values,
    generate_run_prefix,
    setup_experiment_directory,
    run_single_experiment,
    auto_generate_plots,
)


WIDE_SWEEP_P_VALUES = list(np.logspace(-4, 0, 20))

# -----------------------------------------------------------------------------
# Manual configuration block - edit these values when launching new sweeps.
# -----------------------------------------------------------------------------
SWEEP_CONFIG: Dict[str, Any] = {
    "tree_model": "kingman",
    "taxa_values": [500, 1000, 3000, 5000,7000,10000],
    "sequence_length_values": [10000],
    "mutation_rate": 0.1,
    "bootstrap_reps": 5,
    "num_workers": 8,
    "use_middle_out": False,
    "run_name_prefix": "unrooted_kingman_uniform",
    "p_values": WIDE_SWEEP_P_VALUES,
    "tree_params": {"pop_size": 1.0},
    "coherence_k": 4,
    "num_gaps": 0,
    "guardrails_enabled": False
}


def main():
    """
    Main entry point for running experiments.
    
    Edit SWEEP_CONFIG to control what gets run.
    """
    config = SWEEP_CONFIG.copy()
    
    # Extract config values
    cfg_vals = extract_config_values(config, WIDE_SWEEP_P_VALUES)
    
    # Generate run prefix
    prefix = generate_run_prefix(config, cfg_vals["mutation_rate"], cfg_vals["taxa_values"])
    
    # Setup experiment directory
    base_dir = setup_experiment_directory(
        config, prefix, SWEEP_CONFIG,
        script_dir=os.path.dirname(__file__)
    )
    
    # Run experiments
    multi_run_results = []
    for n_taxa in cfg_vals["taxa_values"]:
        for seq_len in cfg_vals["sequence_length_values"]:
            result = run_single_experiment(
                config, base_dir, n_taxa, seq_len,
                cfg_vals["tree_model"], cfg_vals["mutation_rate"],
                cfg_vals["p_values"], cfg_vals["bootstrap_reps"],
                cfg_vals["num_workers"], cfg_vals["use_middle_out"],
                prefix, cfg_vals["tree_kwargs"]
            )
            multi_run_results.append(result)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Completed {cfg_vals['tree_model']} grid sweep across all n and L combinations.")
    for entry in multi_run_results:
        print(f"n={entry['num_taxa']:>4}, L={entry['sequence_length']:>5} → {entry['run_dir']}")
    print(f"{'='*80}")
    
    # Auto-generate plots
    auto_generate_plots(base_dir)
    
    return multi_run_results


if __name__ == "__main__":
    main()
