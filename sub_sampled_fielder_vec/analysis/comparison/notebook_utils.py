"""Utility functions for phase transition analysis notebooks.

This module provides helper functions for loading plotting utilities
and generating merged result plots without triggering package __init__.py
import chains that have external dependencies.

Updated to support new organized directory structure:
    results/{tree_model}/{sampling_method}/{timestamp-experiment_name}/
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Any, Optional, Union

# Package root relative to this file (analysis/comparison/ -> sub_sampled_fielder_vec/)
PKG_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_module_from_file(module_name: str, file_path: Path) -> Any:
    """Load a module directly from file path without triggering package __init__.
    
    Args:
        module_name: Name to register the module under in sys.modules
        file_path: Path to the .py file to load
        
    Returns:
        The loaded module object
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Register so relative imports work
    spec.loader.exec_module(module)
    return module


# Lazy-loaded modules to avoid import overhead
_plotting_module = None
_merge_module = None


def _get_plotting_module():
    """Lazy-load the plotting module."""
    global _plotting_module
    if _plotting_module is None:
        # First load threshold_utils (dependency of plotting)
        _load_module_from_file("threshold_utils", PKG_ROOT / "src/utils/threshold_utils.py")
        # Then load plotting module
        _plotting_module = _load_module_from_file("plotting", PKG_ROOT / "src/utils/plotting.py")
    return _plotting_module


def _get_merge_module():
    """Lazy-load the merge_results module."""
    global _merge_module
    if _merge_module is None:
        _merge_module = _load_module_from_file("merge_results", PKG_ROOT / "scripts/merge_results.py")
    return _merge_module


def get_plot_taxa_sweep() -> Callable:
    """Get the plot_taxa_sweep function from the plotting module."""
    return _get_plotting_module().plot_taxa_sweep


def get_extract_model_name() -> Callable:
    """Get the _extract_model_name function from the plotting module."""
    return _get_plotting_module()._extract_model_name


def get_merge_run_directory() -> Callable:
    """Get the merge_run_directory function from the merge_results module."""
    return _get_merge_module().merge_run_directory


def ensure_merged_plot(run_dir: Path, method_name: str) -> Path:
    """Ensure merged JSON and plot exist, generate if needed.
    
    Args:
        run_dir: Path to the experiment run directory containing n*_L*/ subdirectories
        method_name: Display name for the method (e.g., "Uniform", "Leveraged")
        
    Returns:
        Path to the generated/existing partition agreement plot
    """
    run_dir = Path(run_dir).resolve()
    merged_json = run_dir / "results_grid_merged.json"
    plot_path = run_dir / "partition_agreement.png"
    
    # Get functions
    merge_run_directory = get_merge_run_directory()
    plot_taxa_sweep = get_plot_taxa_sweep()
    _extract_model_name = get_extract_model_name()
    
    # Generate merged JSON if it doesn't exist
    if not merged_json.exists():
        print(f"Generating merged results for {method_name}...")
        merged = merge_run_directory(run_dir)
        with merged_json.open("w") as f:
            json.dump(merged, f, indent=2, allow_nan=True)
        print(f"  Wrote {len(merged['rows'])} rows to {merged_json}")
    else:
        print(f"Merged results already exist: {merged_json}")
        with merged_json.open("r") as f:
            merged = json.load(f)
    
    # Generate plot if it doesn't exist
    if not plot_path.exists():
        print(f"Generating plot for {method_name}...")
        
        # Extract model name and create subtitle
        model_name = _extract_model_name(run_dir.name) + f" ({method_name})"
        
        # Read config for subtitle - try sweep_config.json first
        config_path = run_dir / "sweep_config.json"
        if config_path.exists():
            with config_path.open("r") as f:
                sweep_config = json.load(f)
            seq_len = (
                sweep_config.get("sequence_length_values", [0])[0] 
                if "sequence_length_values" in sweep_config 
                else sweep_config.get("sequence_length", 0)
            )
            mu = sweep_config.get("mutation_rate", "N/A")
            ne = sweep_config.get("tree_params", {}).get("pop_size", "N/A")
            bootstrap_reps = sweep_config.get("bootstrap_reps", "N/A")
            subtitle = f"$L = {seq_len}$, $\\mu = {mu}$, $N_e = {ne}$, {bootstrap_reps} bootstrap reps"
        else:
            # Fallback: extract from merged JSON
            seq_lengths = sorted(set(r.get("sequence_length", 0) for r in merged["rows"]))
            # Try to get mutation rate from one of the config.json files in subdirs
            mu = "N/A"
            bootstrap_reps = "N/A"
            for subdir in run_dir.iterdir():
                if not subdir.is_dir():
                    continue
                config_file = subdir / "config.json"
                if config_file.exists():
                    with config_file.open("r") as f:
                        cfg = json.load(f)
                    if "sequence" in cfg:
                        mu = cfg["sequence"]["params"].get("mutation_rate", mu)
                    if "experiment" in cfg:
                        bootstrap_reps = cfg["experiment"].get("bootstrap_reps", bootstrap_reps)
                    break
            subtitle = f"$L = {seq_lengths[0]}$, $\\mu = {mu}$, {bootstrap_reps} bootstrap reps" if seq_lengths else None
        
        plot_taxa_sweep(
            json_path=str(merged_json),
            output_path=str(plot_path),
            model_name=model_name,
            subtitle=subtitle,
        )
        print(f"  Wrote plot to {plot_path}")
    else:
        print(f"Plot already exists: {plot_path}")
    
    return plot_path


def generate_comparison_plots(
    uniform_dir: Path,
    leveraged_dir: Path,
) -> tuple[Path, Path]:
    """Generate merged plots for both uniform and leveraged sampling.

    Args:
        uniform_dir: Path to uniform sampling results directory
        leveraged_dir: Path to leveraged sampling results directory

    Returns:
        Tuple of (uniform_plot_path, leveraged_plot_path)
    """
    print("=" * 60)
    print("UNIFORM SAMPLING")
    print("=" * 60)
    uniform_plot = ensure_merged_plot(uniform_dir, "Uniform")

    print()
    print("=" * 60)
    print("LEVERAGED SAMPLING")
    print("=" * 60)
    leveraged_plot = ensure_merged_plot(leveraged_dir, "Leveraged")

    return uniform_plot, leveraged_plot


# ============================================================================
# NEW FUNCTIONS: Component-based path construction
# ============================================================================

def construct_experiment_path(
    tree_model: str,
    sampling_method: str,
    experiment_name: Optional[str] = None,
) -> Path:
    """Construct path to experiment from components.

    Args:
        tree_model: Tree model name (e.g., 'kingman_mean', 'balanced_binary')
        sampling_method: Sampling method (e.g., 'uniform', 'leveraged', 'lds')
        experiment_name: Optional experiment directory name
                        If None, returns path to sampling method directory

    Returns:
        Path to experiment directory or sampling method directory

    Examples:
        >>> path = construct_experiment_path('kingman_mean', 'lds', '20260220-120000-...')
        >>> # Returns: .../results/kingman_mean/lds/20260220-120000-.../
    """
    # Import here to avoid circular dependencies
    sys.path.insert(0, str(PKG_ROOT / "analysis" / "leveraged_sampling_analysis" / "io"))
    from path_utils import construct_results_path

    return construct_results_path(tree_model, sampling_method, experiment_name)


def ensure_merged_plot_by_components(
    tree_model: str,
    sampling_method: str,
    experiment_name: Optional[str] = None,
) -> Path:
    """Generate merged plot for experiment specified by components.

    Args:
        tree_model: Tree model name (e.g., 'kingman_mean', 'balanced_binary')
        sampling_method: Sampling method (e.g., 'uniform', 'leveraged', 'lds')
        experiment_name: Optional experiment directory name
                        If None, uses most recent experiment

    Returns:
        Path to generated plot

    Examples:
        >>> # Plot latest kingman_mean LDS experiment
        >>> plot_path = ensure_merged_plot_by_components('kingman_mean', 'lds')

        >>> # Plot specific experiment
        >>> plot_path = ensure_merged_plot_by_components('kingman_mean', 'uniform', '20260220-120000-...')
    """
    # Import here to avoid circular dependencies
    sys.path.insert(0, str(PKG_ROOT / "analysis" / "leveraged_sampling_analysis" / "io"))
    from path_utils import construct_results_path, get_latest_experiment

    if experiment_name is None:
        # Get latest experiment
        run_dir = get_latest_experiment(tree_model, sampling_method)
        if run_dir is None:
            raise ValueError(
                f"No experiments found for tree_model='{tree_model}', "
                f"sampling_method='{sampling_method}'"
            )
    else:
        run_dir = construct_results_path(tree_model, sampling_method, experiment_name)

    if not run_dir.exists():
        raise ValueError(f"Experiment directory not found: {run_dir}")

    # Use existing ensure_merged_plot function
    method_display_name = sampling_method.capitalize()
    return ensure_merged_plot(run_dir, method_display_name)


def generate_comparison_plots_by_components(
    tree_model: str,
    uniform_experiment: Optional[str] = None,
    comparison_method: str = "lds",
    comparison_experiment: Optional[str] = None,
) -> tuple[Path, Path]:
    """Generate comparison plots using component-based paths.

    Args:
        tree_model: Tree model name (e.g., 'kingman_mean')
        uniform_experiment: Optional specific uniform experiment name (uses latest if None)
        comparison_method: Method to compare against ('lds', 'leveraged')
        comparison_experiment: Optional specific comparison experiment name (uses latest if None)

    Returns:
        Tuple of (uniform_plot_path, comparison_plot_path)

    Example:
        >>> # Compare latest uniform vs latest LDS for kingman_mean
        >>> uniform_plot, lds_plot = generate_comparison_plots_by_components('kingman_mean', comparison_method='lds')
    """
    print("=" * 60)
    print(f"UNIFORM SAMPLING ({tree_model})")
    print("=" * 60)
    uniform_plot = ensure_merged_plot_by_components(tree_model, "uniform", uniform_experiment)

    print()
    print("=" * 60)
    print(f"{comparison_method.upper()} SAMPLING ({tree_model})")
    print("=" * 60)
    comparison_plot = ensure_merged_plot_by_components(tree_model, comparison_method, comparison_experiment)

    return uniform_plot, comparison_plot
