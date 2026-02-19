#!/usr/bin/env python3
"""
Interactive launcher for STDR experiments.

Provides a user-friendly menu system for:
- Re-running last configuration
- Selecting cached matrices (single or batch with comma separation)
- Creating new matrix configurations
- Running experiments

Features:
- Single selection: Type "1" to run one cached matrix
- Batch selection: Type "1,3,5" to run multiple matrices in one sweep
- Batch mode requires compatible parameters (same tree_model, seq_len, mutation_rate)

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
    print_config_summary, get_input, get_choice, get_multi_choice, get_menu_choice, confirm,
    print_error, print_success, print_warning, print_divider
)
from src.utils.persistent_cache import list_cached_experiments, clean_incomplete_caches
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


def show_main_menu() -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Display main menu and get user choice(s).

    Returns:
        Tuple of (choices, cached_matrices)
        - choices: List of selected options (single item or multiple for batch)
        - cached_matrices: List of all cached matrices for indexing
    """
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
        # Sort by tree_model first (alphabetically), then by n_taxa (ascending)
        cached = sorted(cached, key=lambda x: (
            x['metadata'].get('tree_model', ''),
            x['metadata'].get('n_taxa', 0)
        ))

        print_header("Cached Matrices")
        print("  💡 Tip: Select multiple with commas (e.g., '1,3,5' for batch run)")
        print()
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

    choices = get_multi_choice("Choice", valid_choices)
    return choices, cached


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

    # Display mode
    display_mode = get_menu_choice("Display mode:", ["progress", "debug"], default_index=0)
    config["display_mode"] = display_mode

    # Sampling method
    sampling = get_menu_choice("Sampling method:", ['uniform', 'leveraged', 'lds'], default_index=2)
    config["sampling_method"] = sampling

    if sampling == "leveraged":
        config["sampling_theta"] = float(get_input("Theta (phase 1 ratio)", default="0.7"))
        config["sampling_target_rank"] = int(get_input("Target rank", default="2"))
        config["sampling_allow_uniform_fallback"] = confirm("Allow fallback to uniform sampling for low p?", default=False)
        config["log_sampling_diagnostics"] = confirm("Log sampling diagnostics (for analysis)?", default=True)
    elif sampling == "lds":
        config["sampling_theta"] = float(get_input("Theta (phase 1 ratio)", default="0.3"))
        config["sampling_target_rank"] = int(get_input("Target rank", default="2"))
        config["sampling_tau_floor_multiplier"] = float(get_input("Tau floor multiplier", default="1.0"))
        config["sampling_allow_uniform_fallback"] = confirm("Allow fallback to uniform sampling for low p?", default=False)
        config["log_sampling_diagnostics"] = confirm("Log sampling diagnostics (for analysis)?", default=True)

    # Other defaults
    config["use_middle_out"] = False
    config["guardrails_enabled"] = False
    config["run_name_prefix"] = get_input("Run name prefix (optional)", default="")

    # Enable persistent cache to use the cached matrix
    config["use_persistent_cache"] = True

    return config


def build_batch_config_from_caches(cache_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build experiment config from multiple cached matrices for batch run.

    All caches must have compatible parameters (same tree_model, seq_len, mutation_rate).
    Collects all unique n_taxa values for batch processing.

    Args:
        cache_entries: List of cache entry dicts with 'metadata' keys

    Returns:
        Config dict with lists of taxa_values for batch processing
    """
    if not cache_entries:
        raise ValueError("No cache entries provided for batch config")

    # Extract metadata from all entries
    metadatas = [entry['metadata'] for entry in cache_entries]

    # Validate compatibility - all must have same tree_model, seq_len, mutation_rate
    first = metadatas[0]
    tree_model = first.get('tree_model')
    seq_len = first.get('seq_len')
    mutation_rate = first.get('mutation_rate')

    for i, meta in enumerate(metadatas[1:], 2):
        if meta.get('tree_model') != tree_model:
            print_error(f"Incompatible tree_model: cache 1 has '{tree_model}', cache {i} has '{meta.get('tree_model')}'")
            raise ValueError("Cannot batch caches with different tree models")
        if meta.get('seq_len') != seq_len:
            print_error(f"Incompatible seq_len: cache 1 has {seq_len}, cache {i} has {meta.get('seq_len')}")
            raise ValueError("Cannot batch caches with different sequence lengths")
        if meta.get('mutation_rate') != mutation_rate:
            print_error(f"Incompatible mutation_rate: cache 1 has {mutation_rate}, cache {i} has {meta.get('mutation_rate')}")
            raise ValueError("Cannot batch caches with different mutation rates")

    # Collect unique n_taxa values
    taxa_values = sorted(set(meta.get('n_taxa') for meta in metadatas))

    # Build config with batch values
    config = {
        "tree_model": tree_model,
        "taxa_values": taxa_values,
        "sequence_length_values": [seq_len],
        "mutation_rate": mutation_rate,
    }

    print_header("Batch Experiment Configuration")
    print(f"Running batch across {len(taxa_values)} cached matrices:")
    for n in taxa_values:
        print(f"  • n={n}")
    print()

    # Prompt for experiment parameters (once for all matrices)
    config["bootstrap_reps"] = int(get_input("Bootstrap replicates", default="10"))
    config["num_workers"] = int(get_input("Number of workers", default="8"))

    # P-values configuration
    use_default_p = confirm("Use default p-values (20 points logspace)?", default=True)
    if use_default_p:
        config["p_values"] = list(np.logspace(-4, 0, 20))
    else:
        print("Custom p-values not yet implemented - using defaults")
        config["p_values"] = list(np.logspace(-4, 0, 20))

    # Display mode
    display_mode = get_menu_choice("Display mode:", ["progress", "debug"], default_index=0)
    config["display_mode"] = display_mode

    # Sampling method
    sampling = get_menu_choice("Sampling method:", ['uniform', 'leveraged', 'lds'], default_index=2)
    config["sampling_method"] = sampling

    if sampling == "leveraged":
        config["sampling_theta"] = float(get_input("Theta (phase 1 ratio)", default="0.7"))
        config["sampling_target_rank"] = int(get_input("Target rank", default="2"))
        config["sampling_allow_uniform_fallback"] = confirm("Allow fallback to uniform sampling for low p?", default=False)
        config["log_sampling_diagnostics"] = confirm("Log sampling diagnostics (for analysis)?", default=True)
    elif sampling == "lds":
        config["sampling_theta"] = float(get_input("Theta (phase 1 ratio)", default="0.3"))
        config["sampling_target_rank"] = int(get_input("Target rank", default="2"))
        config["sampling_tau_floor_multiplier"] = float(get_input("Tau floor multiplier", default="1.0"))
        config["sampling_allow_uniform_fallback"] = confirm("Allow fallback to uniform sampling for low p?", default=False)
        config["log_sampling_diagnostics"] = confirm("Log sampling diagnostics (for analysis)?", default=True)

    # Other defaults
    config["use_middle_out"] = False
    config["guardrails_enabled"] = False
    config["run_name_prefix"] = get_input("Run name prefix (optional)", default="")

    # Enable persistent cache to use cached matrices
    config["use_persistent_cache"] = True

    return config


def create_new_config() -> Dict[str, Any]:
    """Create new experiment configuration interactively."""
    print_header("Create New Matrix Configuration")
    print("Press Enter to keep default values")
    print("💡 Tip: Use commas for batch runs (e.g., '512,1024,2048')")
    print()

    # Basic matrix parameters - accept comma-separated values
    n_taxa_input = get_input("n_taxa (number of taxa, comma-separated for batch)", default="2048")
    taxa_values = [int(x.strip()) for x in n_taxa_input.split(',')]

    seq_len_input = get_input("sequence_length (comma-separated for batch)", default="10000")
    sequence_length_values = [int(x.strip()) for x in seq_len_input.split(',')]

    mutation_rate = float(get_input("mutation_rate", default="0.1"))

    # Tree model
    print("\nAvailable tree models: balanced_binary, lopsided, kingman, kingman_mean, birth_death")
    tree_model = get_input("tree_model", default="balanced_binary")

    # Tree-specific parameters
    tree_params = {}
    if tree_model == "balanced_binary":
        # Check if all n_taxa values are powers of 2
        adjusted = []
        for n in taxa_values:
            if n & (n - 1) != 0:
                adjusted_n = 2 ** int(np.log2(n))
                print_warning(f"balanced_binary requires power of 2: {n} → {adjusted_n}")
                adjusted.append(adjusted_n)
            else:
                adjusted.append(n)
        taxa_values = adjusted
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
    sampling_method = get_menu_choice("Sampling method:", ["uniform", "leveraged", "lds"], default_index=2)

    # Display mode
    display_mode = get_menu_choice("Display mode:", ["progress", "debug"], default_index=0)

    # Show batch summary if multiple values
    if len(taxa_values) > 1 or len(sequence_length_values) > 1:
        print_header("Batch Configuration Summary")
        print(f"  n_taxa: {taxa_values}")
        print(f"  seq_len: {sequence_length_values}")
        print(f"  Total experiments: {len(taxa_values) * len(sequence_length_values)}")
        print()

    config = {
        "tree_model": tree_model,
        "taxa_values": taxa_values,
        "sequence_length_values": sequence_length_values,
        "mutation_rate": mutation_rate,
        "tree_params": tree_params,
        "bootstrap_reps": bootstrap_reps,
        "num_workers": num_workers,
        "p_values": p_values,
        "use_middle_out": False,
        "sampling_method": sampling_method,
        "display_mode": display_mode,
        "guardrails_enabled": False,
        "run_name_prefix": get_input("run_name_prefix (optional)", default=""),
    }

    # Leveraged sampling parameters
    if sampling_method == "leveraged":
        config["sampling_theta"] = float(get_input("sampling_theta", default="0.7"))
        config["sampling_target_rank"] = int(get_input("sampling_target_rank", default="2"))
        config["sampling_allow_uniform_fallback"] = confirm("Allow fallback to uniform sampling for low p?", default=False)
        config["sampling_ialm_max_iter"] = int(get_input("IALM max_iter", default="500"))
        config["sampling_ialm_tol"] = float(get_input("IALM tolerance", default="1e-4"))
        config["log_sampling_diagnostics"] = confirm("Log sampling diagnostics (for analysis)?", default=True)
    elif sampling_method == "lds":
        config["sampling_theta"] = float(get_input("sampling_theta", default="0.3"))
        config["sampling_target_rank"] = int(get_input("sampling_target_rank", default="2"))
        config["sampling_tau_floor_multiplier"] = float(get_input("tau_floor_multiplier", default="1.0"))
        config["sampling_allow_uniform_fallback"] = confirm("Allow fallback to uniform sampling for low p?", default=False)
        config["log_sampling_diagnostics"] = confirm("Log sampling diagnostics (for analysis)?", default=True)

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

    # Clean up incomplete cache entries from interrupted experiments
    clean_incomplete_caches()

    while True:
        # Show main menu
        choices, cached = show_main_menu()

        # Single choice - check for special commands
        if len(choices) == 1 and choices[0] == 'q':
            print_success("Goodbye!")
            sys.exit(0)

        elif len(choices) == 1 and choices[0] == 'r':
            # Re-run last configuration
            last_run = load_last_run()
            if last_run:
                # Remove timestamp before running
                config = {k: v for k, v in last_run.items() if k != 'timestamp'}
                run_experiment(config)
            else:
                print_error("No last run found")

        elif len(choices) == 1 and choices[0] == 'n':
            # Create new configuration
            config = create_new_config()
            run_experiment(config)

        else:
            # Choices are numbers - load from cache (single or batch)
            try:
                # Convert choices to cache indices
                cache_indices = [int(choice) - 1 for choice in choices]

                # Validate all indices
                invalid = [i for i in cache_indices if i < 0 or i >= len(cached)]
                if invalid:
                    print_error(f"Invalid cache selection(s): {[i+1 for i in invalid]}")
                    continue

                # Get selected cache entries
                selected_caches = [cached[i] for i in cache_indices]

                # Single or batch?
                if len(selected_caches) == 1:
                    # Single cache - use existing flow
                    cache_entry = selected_caches[0]
                    config = build_config_from_cache(cache_entry['metadata'])
                    run_experiment(config)
                else:
                    # Batch mode - multiple caches selected
                    try:
                        config = build_batch_config_from_caches(selected_caches)
                        run_experiment(config)
                    except ValueError as e:
                        print_error(str(e))
                        continue
            except ValueError:
                print_error("Invalid choice")

        # Ask if user wants to run another experiment
        print()
        if not confirm("Run another experiment?", default=False):
            print_success("Goodbye!")
            break


if __name__ == "__main__":
    main()
