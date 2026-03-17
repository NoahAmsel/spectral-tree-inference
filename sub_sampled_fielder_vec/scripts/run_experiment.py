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
LEVERAGED_P_VALUES = list(np.logspace(-2, 0, 20))  # Start from 10^-2 for leveraged sampling

# -----------------------------------------------------------------------------
# Manual configuration block - edit these values when launching new sweeps.
# -----------------------------------------------------------------------------
SWEEP_CONFIG: Dict[str, Any] = {
    "tree_model": "balanced_binary",
    "taxa_values": [2048],
    "sequence_length_values": [10000],
    "mutation_rate": 0.1,
    "bootstrap_reps": 10,
    "num_workers": 8,
    "use_middle_out": False,
    "run_name_prefix": "balanced_binary_leveraged_no_bypass",
    "p_values": LEVERAGED_P_VALUES,
    "tree_params": {"edge_length": 1.0},
    "coherence_k": 4,
    "num_gaps": 0,
    "guardrails_enabled": False,
    "sampling_method": "leveraged",
    "sampling_theta": 0.7,
    "sampling_target_rank": 2,  # Reduced from 10 for lower theoretical minimum
    "sampling_ialm_max_iter": 500,
    "sampling_ialm_tol": 1e-4,
    "sampling_ialm_bypass_threshold": 1.0,  # Never bypass IALM
    "sampling_force_leveraged": True,  # Force leveraged sampling even at low p
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
