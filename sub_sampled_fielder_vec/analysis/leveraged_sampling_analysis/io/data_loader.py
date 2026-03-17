"""Data loading and parsing utilities for leveraged sampling experiments.

This module provides functions to load experiment results from JSON files
and convert them into structured formats for analysis.

Updated to support new organized directory structure:
    results/{tree_model}/{sampling_method}/{timestamp-experiment_name}/

You can now load data by specifying:
    1. Direct path (backward compatible): load_experiment_results(Path("results/..."))
    2. Components: load_experiment_by_components(tree_model, sampling_method, experiment_name)
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

# Try relative import first (when used as package), fall back to direct import
try:
    from .path_utils import construct_results_path, get_latest_experiment, find_experiments
except ImportError:
    from path_utils import construct_results_path, get_latest_experiment, find_experiments


DIR_PATTERN = re.compile(r"^n(?P<num>\d+)_L(?P<len>\d+)$")


def load_experiment_results(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Load single experiment run with multiple n values.

    Args:
        run_dir: Directory containing n{X}_L{Y} subdirectories

    Returns:
        Dict mapping n_taxa -> {sequence_length: int, rows: List[Dict]}
    """
    run_dir = Path(run_dir)
    data = {}

    # Find all n{X}_L{Y} subdirectories
    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("n"):
            continue

        match = DIR_PATTERN.match(subdir.name)
        if not match:
            continue

        n_taxa = int(match.group("num"))
        seq_len = int(match.group("len"))

        # Load results.json
        results_file = subdir / "results.json"
        if not results_file.exists():
            continue

        with results_file.open() as f:
            results = json.load(f)

        data[n_taxa] = {
            "sequence_length": seq_len,
            "columns": results.get("columns", []),
            "rows": results.get("rows", []),
            "reference_partition_quality": results.get("reference_partition_quality"),
        }

    if not data:
        raise ValueError(f"No n*_L* subdirectories found in {run_dir}")

    return data


def load_comparison_results(comp_dir: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load comparison results for both uniform and leveraged methods.

    Args:
        comp_dir: Comparison directory (e.g., results/XXX-method_comparison_YYY)

    Returns:
        Nested dict: {method: {n_taxa: {sequence_length, columns, rows, ...}}}
    """
    comp_dir = Path(comp_dir)
    data = {}

    for method in ["uniform", "leveraged"]:
        method_dir = comp_dir / method
        if not method_dir.exists():
            continue

        data[method] = load_experiment_results(method_dir)

    if not data:
        raise ValueError(f"No method directories found in {comp_dir}")

    return data


def get_available_runs(results_dir: Path) -> List[Dict[str, Any]]:
    """List available experiment directories.

    Args:
        results_dir: Base results directory

    Returns:
        List of dicts with keys: path, name, type (comparison/single)
    """
    results_dir = Path(results_dir)
    runs = []

    if not results_dir.exists():
        return runs

    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue

        run_info = {
            "path": entry,
            "name": entry.name,
        }

        # Check if it's a comparison directory
        if "method_comparison" in entry.name:
            run_info["type"] = "comparison"
            # Check which methods are available
            methods = []
            for method in ["uniform", "leveraged"]:
                if (entry / method).exists():
                    methods.append(method)
            run_info["methods"] = methods
        else:
            run_info["type"] = "single"
            # Check if it has n*_L* subdirectories
            has_subdirs = any(
                DIR_PATTERN.match(d.name) for d in entry.iterdir() if d.is_dir()
            )
            run_info["has_data"] = has_subdirs

        runs.append(run_info)

    return runs


def extract_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Parse config.json for experiment parameters.

    Args:
        run_dir: Directory containing config.json (can be n{X}_L{Y} subdir or parent)

    Returns:
        Config dict or None if not found
    """
    run_dir = Path(run_dir)

    # Try direct path first
    config_file = run_dir / "config.json"
    if config_file.exists():
        with config_file.open() as f:
            return json.load(f)

    # Try comparison_config.json for comparison runs
    config_file = run_dir / "comparison_config.json"
    if config_file.exists():
        with config_file.open() as f:
            return json.load(f)

    # Try sweep_config.json for sweep/solo runs
    config_file = run_dir / "sweep_config.json"
    if config_file.exists():
        with config_file.open() as f:
            return json.load(f)

    # Try parent directory
    parent_config = run_dir.parent / "config.json"
    if parent_config.exists():
        with parent_config.open() as f:
            return json.load(f)

    # Try sweep_config.json in parent directory
    parent_sweep_config = run_dir.parent / "sweep_config.json"
    if parent_sweep_config.exists():
        with parent_sweep_config.open() as f:
            return json.load(f)

    return None


def to_dataframe(
    data: Dict[str, Any], method: Optional[str] = None
) -> pd.DataFrame:
    """Convert nested dict to pandas DataFrame for analysis.

    Args:
        data: Either:
            - Single run: {n_taxa: {sequence_length, columns, rows}}
            - Comparison: {method: {n_taxa: {sequence_length, columns, rows}}}
        method: If data is comparison dict, specify which method to extract

    Returns:
        DataFrame with all rows, including num_taxa and sequence_length columns
    """
    rows = []

    if method is not None:
        # Extract specific method from comparison
        if method not in data:
            raise ValueError(f"Method '{method}' not found in data")
        data = data[method]

    # Check if this is a single run or still nested
    if "uniform" in data or "leveraged" in data:
        raise ValueError(
            "Comparison data detected. Specify 'method' parameter to extract one method."
        )

    # Process single run data
    for n_taxa, run_data in data.items():
        seq_len = run_data["sequence_length"]
        columns = run_data.get("columns", [])
        row_list = run_data.get("rows", [])

        for row in row_list:
            # Use the row dict directly and add metadata
            row["num_taxa"] = n_taxa
            row["sequence_length"] = seq_len
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort by sequence_length, num_taxa, p
    if "p" in df.columns:
        df = df.sort_values(["sequence_length", "num_taxa", "p"])
    else:
        df = df.sort_values(["sequence_length", "num_taxa"])

    return df


def load_single_run_dataframe(run_dir: Path) -> pd.DataFrame:
    """Load single experiment run as a DataFrame.

    Args:
        run_dir: Single run directory containing n{X}_L{Y} subdirectories

    Returns:
        DataFrame with all rows from the experiment
    """
    data = load_experiment_results(run_dir)
    return to_dataframe(data, method=None)


def load_comparison_dataframe(comp_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load comparison results as DataFrames for both methods.

    Args:
        comp_dir: Comparison directory

    Returns:
        Dict mapping method -> DataFrame
    """
    data = load_comparison_results(comp_dir)
    result = {}
    for method in data.keys():
        # Extract the method's data (which is {n_taxa: {...}})
        method_data = data[method]
        # Convert directly to DataFrame
        result[method] = to_dataframe(method_data, method=None)
    return result


# ============================================================================
# NEW FUNCTIONS: Component-based path construction
# ============================================================================

def load_experiment_by_components(
    tree_model: str,
    sampling_method: str,
    experiment_name: Optional[str] = None,
    results_dir: Optional[Union[str, Path]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Load experiment results by specifying tree model, sampling method, and name.

    Args:
        tree_model: Tree model name (e.g., 'kingman_mean', 'balanced_binary')
        sampling_method: Sampling method (e.g., 'uniform', 'leveraged', 'lds')
        experiment_name: Experiment directory name (e.g., '20260220-120000-...')
                       If None, loads the most recent experiment
        results_dir: Optional base results directory

    Returns:
        Dict mapping n_taxa -> {sequence_length, columns, rows, ...}

    Examples:
        >>> # Load specific experiment
        >>> data = load_experiment_by_components('kingman_mean', 'lds', '20260220-120000-kingman_mean_n512_mu_0p1_lds')

        >>> # Load most recent experiment
        >>> data = load_experiment_by_components('kingman_mean', 'lds')

    Raises:
        ValueError: If no experiment found with given components
    """
    if experiment_name is None:
        # Get latest experiment
        run_dir = get_latest_experiment(tree_model, sampling_method, results_dir)
        if run_dir is None:
            raise ValueError(
                f"No experiments found for tree_model='{tree_model}', "
                f"sampling_method='{sampling_method}'"
            )
    else:
        run_dir = construct_results_path(tree_model, sampling_method, experiment_name, results_dir)

    if not run_dir.exists():
        raise ValueError(f"Experiment directory not found: {run_dir}")

    return load_experiment_results(run_dir)


def load_dataframe_by_components(
    tree_model: str,
    sampling_method: str,
    experiment_name: Optional[str] = None,
    results_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load experiment as DataFrame by specifying tree model, sampling method, and name.

    Args:
        tree_model: Tree model name (e.g., 'kingman_mean', 'balanced_binary')
        sampling_method: Sampling method (e.g., 'uniform', 'leveraged', 'lds')
        experiment_name: Experiment directory name (e.g., '20260220-120000-...')
                       If None, loads the most recent experiment
        results_dir: Optional base results directory

    Returns:
        DataFrame with all experiment results

    Examples:
        >>> # Load latest LDS experiment for kingman_mean
        >>> df = load_dataframe_by_components('kingman_mean', 'lds')

        >>> # Load specific experiment
        >>> df = load_dataframe_by_components('kingman_mean', 'uniform', '20260220-120000-...')
    """
    data = load_experiment_by_components(tree_model, sampling_method, experiment_name, results_dir)
    return to_dataframe(data, method=None)


def list_available_experiments(
    tree_model: Optional[str] = None,
    sampling_method: Optional[str] = None,
    results_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """List available experiments, optionally filtered by tree model and/or sampling method.

    Args:
        tree_model: Optional filter for tree model
        sampling_method: Optional filter for sampling method
        results_dir: Optional base results directory

    Returns:
        List of dicts with keys: path, tree_model, sampling_method, experiment_name, timestamp

    Examples:
        >>> # List all experiments
        >>> all_exps = list_available_experiments()

        >>> # List all kingman_mean experiments
        >>> kingman_exps = list_available_experiments(tree_model='kingman_mean')

        >>> # List all LDS experiments
        >>> lds_exps = list_available_experiments(sampling_method='lds')

        >>> # List kingman_mean + LDS experiments
        >>> specific = list_available_experiments(tree_model='kingman_mean', sampling_method='lds')
    """
    try:
        from .path_utils import list_all_experiments
    except ImportError:
        from path_utils import list_all_experiments

    all_experiments = list_all_experiments(results_dir)

    # Apply filters
    if tree_model is not None:
        all_experiments = [e for e in all_experiments if e['tree_model'] == tree_model]

    if sampling_method is not None:
        all_experiments = [e for e in all_experiments if e['sampling_method'] == sampling_method]

    return all_experiments
