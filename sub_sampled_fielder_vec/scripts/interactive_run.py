#!/usr/bin/env python3
"""
Interactive launcher for STDR experiments.

Provides a user-friendly menu system for:
- Re-running last configuration
- Selecting cached matrices
- Creating new matrix configurations
- Running experiments

No more editing SWEEP_CONFIG - everything is interactive!
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.utils.interactive_ui import (
    print_logo, print_header, print_option, print_cached_matrix,
    print_config_summary, get_input, get_choice, confirm,
    print_error, print_success, print_warning, print_divider
)
from src.utils.persistent_cache import list_cached_experiments
from src.runners.experiment_runner_utils import (
    extract_config_values,
    generate_run_prefix,
    setup_experiment_directory,
    run_single_experiment,
    auto_generate_plots,
)


# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
LAST_RUN_PATH = PROJECT_ROOT / "last_run.json"


def load_last_run() -> Optional[Dict[str, Any]]:
    """Load last run configuration from JSON file."""
    if not LAST_RUN_PATH.exists():
        return None

    try:
        with open(LAST_RUN_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print_warning(f"Could not load last run: {e}")
        return None


def save_last_run(config: Dict[str, Any]):
    """Save configuration to last_run.json for future re-runs."""
    import time

    # Add timestamp
    config_with_timestamp = config.copy()
    config_with_timestamp['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        with open(LAST_RUN_PATH, 'w') as f:
            json.dump(config_with_timestamp, f, indent=2)
        print_success(f"Configuration saved to {LAST_RUN_PATH}")
    except Exception as e:
        print_warning(f"Could not save configuration: {e}")


def list_cached_matrices() -> List[Dict[str, Any]]:
    """Get list of cached matrices."""
    return list_cached_experiments()


def show_main_menu() -> str:
    """Display main menu and get user choice."""
    print_header("Main Menu")

    # Show last run option
    last_run = load_last_run()
    if last_run:
        print_option("r", "Re-run last configuration:", highlight=True)
        print_config_summary(last_run)
        print()

    # Show cached matrices
    cached = list_cached_matrices()
    if cached:
        print_header("Cached Matrices")
        for i, cache_entry in enumerate(cached, 1):
            print_cached_matrix(i, cache_entry['metadata'])
        print()
    else:
        # No cached matrices yet - show helpful message
        from src.utils.interactive_ui import Colors
        print(f"{Colors.YELLOW}💡 No cached matrices yet{Colors.RESET}")
        print(f"{Colors.YELLOW}   Matrices will be cached after your first experiment run{Colors.RESET}")
        print()

    # New matrix option
    print_option("n", "Create new matrix configuration")
    print_option("q", "Quit")
    print()

    # Build valid choices
    valid_choices = ['n', 'q']
    if last_run:
        valid_choices.insert(0, 'r')
    if cached:
        valid_choices.extend([str(i) for i in range(1, len(cached) + 1)])

    choice = get_choice("Choice", valid_choices)
    return choice


def build_config_from_cache(cache_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Build experiment config from cached matrix metadata."""
    # Extract basic params from cache
    config = {
        "tree_model": cache_metadata.get('tree_model', 'balanced_binary'),
        "taxa_values": [cache_metadata.get('n_taxa', 2048)],
        "sequence_length_values": [cache_metadata.get('seq_len', 10000)],
        "mutation_rate": cache_metadata.get('mutation_rate', 0.1),
    }

    print_header("Experiment Configuration")
    print("Using cached matrix. Configure experiment parameters:")
    print()

    # TODO(human): Implement interactive parameter configuration
    # Ask user for: p_values, bootstrap_reps, num_workers, sampling_method, etc.
    # For now, use sensible defaults

    # Prompt for experiment parameters
    config["bootstrap_reps"] = int(get_input("Bootstrap replicates", default="10"))
    config["num_workers"] = int(get_input("Number of workers", default="8"))

    # P-values configuration
    use_default_p = confirm("Use default p-values (20 points logspace)?", default=True)
    if use_default_p:
        config["p_values"] = list(np.logspace(-4, 0, 20))
    else:
        print("Custom p-values not yet implemented - using defaults")
        config["p_values"] = list(np.logspace(-4, 0, 20))

    # Sampling method
    sampling = get_choice("Sampling method [uniform/leveraged]", ['uniform', 'leveraged'])
    config["sampling_method"] = sampling

    if sampling == "leveraged":
        config["sampling_theta"] = float(get_input("Theta (phase 1 ratio)", default="0.7"))
        config["sampling_target_rank"] = int(get_input("Target rank", default="2"))

    # Other defaults
    config["use_middle_out"] = False
    config["guardrails_enabled"] = False
    config["run_name_prefix"] = get_input("Run name prefix (optional)", default="")

    return config


def create_new_config() -> Dict[str, Any]:
    """Create new experiment configuration interactively."""
    print_header("Create New Matrix Configuration")
    print("Press Enter to keep default values")
    print()

    # Basic matrix parameters
    n_taxa = int(get_input("n_taxa (number of taxa)", default="2048"))
    seq_len = int(get_input("sequence_length", default="10000"))
    mutation_rate = float(get_input("mutation_rate", default="0.1"))

    # Tree model
    print("\nAvailable tree models: balanced_binary, lopsided, kingman, kingman_mean, birth_death")
    tree_model = get_input("tree_model", default="balanced_binary")

    # Tree-specific parameters
    tree_params = {}
    if tree_model == "balanced_binary":
        # Check if n_taxa is power of 2
        if n_taxa & (n_taxa - 1) != 0:
            print_error(f"balanced_binary requires n_taxa to be power of 2, got {n_taxa}")
            n_taxa = 2 ** int(np.log2(n_taxa))
            print_warning(f"Adjusting to nearest power of 2: {n_taxa}")
        tree_params["edge_length"] = float(get_input("edge_length", default="1.0"))
    elif tree_model in ["kingman", "kingman_mean"]:
        tree_params["pop_size"] = float(get_input("pop_size (Ne)", default="1.0"))

    # Experiment parameters
    print()
    bootstrap_reps = int(get_input("bootstrap_reps", default="10"))
    num_workers = int(get_input("num_workers", default="8"))

    # P-values
    use_default_p = confirm("Use default p-values (20 points logspace)?", default=True)
    if use_default_p:
        p_values = list(np.logspace(-4, 0, 20))
    else:
        print("Custom p-values not yet implemented - using defaults")
        p_values = list(np.logspace(-4, 0, 20))

    # Sampling method
    print("\nSampling methods: uniform, leveraged")
    sampling_method = get_input("sampling_method", default="uniform")

    config = {
        "tree_model": tree_model,
        "taxa_values": [n_taxa],
        "sequence_length_values": [seq_len],
        "mutation_rate": mutation_rate,
        "tree_params": tree_params,
        "bootstrap_reps": bootstrap_reps,
        "num_workers": num_workers,
        "p_values": p_values,
        "use_middle_out": False,
        "sampling_method": sampling_method,
        "guardrails_enabled": False,
        "run_name_prefix": get_input("run_name_prefix (optional)", default=""),
    }

    # Leveraged sampling parameters
    if sampling_method == "leveraged":
        config["sampling_theta"] = float(get_input("sampling_theta", default="0.7"))
        config["sampling_target_rank"] = int(get_input("sampling_target_rank", default="2"))
        config["sampling_ialm_max_iter"] = int(get_input("IALM max_iter", default="500"))
        config["sampling_ialm_tol"] = float(get_input("IALM tolerance", default="1e-4"))

    # Enable persistent cache for new matrices
    config["use_persistent_cache"] = True

    return config


def run_experiment(config: Dict[str, Any]):
    """Run experiment with given configuration."""
    print_divider()
    print_header("Starting Experiment")
    print_config_summary(config)
    print()

    if not confirm("Confirm and start experiment?"):
        print_warning("Experiment cancelled")
        return

    # Save configuration for future re-runs
    save_last_run(config)

    # Extract and validate config
    WIDE_SWEEP_P_VALUES = list(np.logspace(-4, 0, 20))
    cfg_vals = extract_config_values(config, WIDE_SWEEP_P_VALUES)

    # Generate run prefix
    prefix = generate_run_prefix(config, cfg_vals["mutation_rate"], cfg_vals["taxa_values"])

    # Setup experiment directory
    base_dir = setup_experiment_directory(
        config, prefix, config,
        script_dir=str(SCRIPT_DIR)
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
    print_divider()
    print_success(f"Completed {cfg_vals['tree_model']} experiments")
    for entry in multi_run_results:
        print(f"  n={entry['num_taxa']:>4}, L={entry['sequence_length']:>5} → {entry['run_dir']}")
    print_divider()

    # Auto-generate plots
    auto_generate_plots(base_dir)

    return multi_run_results


def main():
    """Main entry point for interactive launcher."""
    # Print logo
    print_logo()

    while True:
        # Show main menu
        choice = show_main_menu()

        if choice == 'q':
            print_success("Goodbye!")
            sys.exit(0)

        elif choice == 'r':
            # Re-run last configuration
            last_run = load_last_run()
            if last_run:
                # Remove timestamp before running
                config = {k: v for k, v in last_run.items() if k != 'timestamp'}
                run_experiment(config)
            else:
                print_error("No last run found")

        elif choice == 'n':
            # Create new configuration
            config = create_new_config()
            run_experiment(config)

        else:
            # Choice is a number - load from cache
            try:
                cache_idx = int(choice) - 1
                cached = list_cached_matrices()
                if 0 <= cache_idx < len(cached):
                    cache_entry = cached[cache_idx]
                    config = build_config_from_cache(cache_entry['metadata'])
                    run_experiment(config)
                else:
                    print_error("Invalid cache selection")
            except ValueError:
                print_error("Invalid choice")

        # Ask if user wants to run another experiment
        print()
        if not confirm("Run another experiment?", default=False):
            print_success("Goodbye!")
            break


if __name__ == "__main__":
    main()
