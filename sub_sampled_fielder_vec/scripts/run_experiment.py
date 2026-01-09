"""Main entry point for bootstrap sweep experiments.

Edit the SWEEP_CONFIG dict below to change what gets executed. This keeps the
script dead-simple while still letting us reuse ExperimentRunner.
"""
import os
import sys
import time
from typing import Dict, Any, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src import ExperimentRunner
from src.config.presets import custom_config


WIDE_SWEEP_P_VALUES = list(np.logspace(-4, 0, 20))

# -----------------------------------------------------------------------------
# Manual configuration block - edit these values when launching new sweeps.
# -----------------------------------------------------------------------------
SWEEP_CONFIG: Dict[str, Any] = {
    "tree_model": "kingman_mean",
    "taxa_values": [512, 1024],
    "sequence_length_values": [10000],
    "mutation_rate": 0.1,
    "bootstrap_reps": 20,
    "num_workers": 8,
    "use_middle_out": False,
    "run_name_prefix": "leveraged_test",
    "p_values": WIDE_SWEEP_P_VALUES,
    "tree_params": {
        "pop_size": 1.0,
    },
    "coherence_k": 4,
    "num_gaps": 0,
    "guardrails_enabled": False,
    # Sampling method configuration
    "sampling_method": "leveraged",      # "uniform" or "leveraged"
    "sampling_theta": 0.5,               # Phase 1 budget ratio (for leveraged only)
    "sampling_target_rank": 2,            # SVD rank for leverage estimation (for leveraged only)
    "sampling_ialm_max_iter": 100,       # IALM max iterations (for leveraged only)
    "sampling_ialm_tol": 1e-6,           # IALM tolerance (for leveraged only)
}


def main():
    """
    Main entry point for running experiments.
    
    Edit SWEEP_CONFIG to control what gets run.
    """
    config = SWEEP_CONFIG.copy()

    tree_model = config["tree_model"]
    taxa_values: List[int] = config["taxa_values"]
    sequence_length_values: List[int] = config["sequence_length_values"]
    mutation_rate: float = config["mutation_rate"]
    p_values: List[float] = config.get("p_values", WIDE_SWEEP_P_VALUES)
    tree_kwargs = config.get("tree_params", {})
    bootstrap_reps: int = config.get("bootstrap_reps", 10)
    num_workers: int = config.get("num_workers", 8)
    use_middle_out: bool = config.get("use_middle_out", True)

    prefix = config.get("run_name_prefix")
    if not prefix:
        mu_str = str(mutation_rate).replace(".", "p")
        pref_taxa = f"n{taxa_values[0]}" if len(taxa_values) == 1 else f"n{min(taxa_values)}-{max(taxa_values)}"
        sampling_method = config.get("sampling_method", "uniform")
        prefix = f"{tree_model}_{pref_taxa}_mu_{mu_str}_{sampling_method}"
    else:
        # Append sampling method to prefix if not already included
        sampling_method = config.get("sampling_method", "uniform")
        if sampling_method not in prefix and sampling_method != "uniform":
            prefix = f"{prefix}_{sampling_method}"

    ts = time.strftime("%Y%m%d-%H%M%S")
    base_dir_root = config.get(
        "base_dir_root",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results")),
    )
    base_dir = os.path.abspath(os.path.join(base_dir_root, f"{ts}-{prefix}"))
    os.makedirs(base_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Grid Sweep Base Directory: {base_dir}")
    print(f"{'='*80}\n")

    multi_run_results = []

    for n_taxa in taxa_values:
        for seq_len in sequence_length_values:
            # Create subdirectory name for this config
            subdir_name = f"n{n_taxa}_L{seq_len}"

            # Get sampling configuration
            sampling_method = config.get("sampling_method", "uniform")
            sampling_theta = config.get("sampling_theta", 0.3)
            sampling_target_rank = config.get("sampling_target_rank", 2)
            sampling_ialm_max_iter = config.get("sampling_ialm_max_iter", 100)
            sampling_ialm_tol = config.get("sampling_ialm_tol", 1e-6)
            
            cfg = custom_config(
                num_taxa=n_taxa,
                sequence_length=seq_len,
                mutation_rate=mutation_rate,
                tree_model=tree_model,
                p_values=p_values,
                bootstrap_reps=bootstrap_reps,
                run_name=prefix,
                display_mode="progress",
                num_workers=num_workers,
                use_middle_out=use_middle_out,
                # Sampling method parameters
                sampling_method=sampling_method,
                sampling_theta=sampling_theta,
                sampling_target_rank=sampling_target_rank,
                sampling_ialm_max_iter=sampling_ialm_max_iter,
                sampling_ialm_tol=sampling_ialm_tol,
                **tree_kwargs,
            )

            # Apply sweep-level metric/guardrail overrides that custom_config doesn't expose
            cfg.metrics.coherence_k = config.get("coherence_k", cfg.metrics.coherence_k)
            cfg.metrics.num_gaps = config.get("num_gaps", cfg.metrics.num_gaps)
            if "guardrails_enabled" in config:
                cfg.guardrails.enabled = config["guardrails_enabled"]

            print(f"\n{'-'*80}")
            print(f"Launching experiment for {tree_model} tree: n={n_taxa}, L={seq_len}")
            print(f"Sampling method: {sampling_method}")
            if sampling_method == "leveraged":
                print(f"  Phase 1 ratio (theta): {sampling_theta}")
                print(f"  Target rank: {sampling_target_rank}")
                print(f"  IALM max_iter: {sampling_ialm_max_iter}, tol: {sampling_ialm_tol}")
            runner = ExperimentRunner(cfg, base_dir=base_dir, subdir_name=subdir_name)
            run_dir, results = runner.run()
            multi_run_results.append({
                "num_taxa": n_taxa,
                "sequence_length": seq_len,
                "run_dir": run_dir,
                "results": results,
            })
    
    print(f"\n{'='*80}")
    print(f"Completed {tree_model} grid sweep across all n and L combinations.")
    for entry in multi_run_results:
        print(f"n={entry['num_taxa']:>4}, L={entry['sequence_length']:>5} → {entry['run_dir']}")
    print(f"{'='*80}")
    
    return multi_run_results


if __name__ == "__main__":
    main()
