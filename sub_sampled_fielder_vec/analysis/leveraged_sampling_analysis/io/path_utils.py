"""Path construction utilities for experiment results.

This module provides functions to construct result paths from experiment components
(tree_model, sampling_method, experiment_name) to work with the new organized
directory structure: results/{tree_model}/{sampling_method}/{timestamp-experiment_name}/
"""
from pathlib import Path
from typing import Optional, Union
import re


# Pattern to match experiment directory names: timestamp-name
EXPERIMENT_DIR_PATTERN = re.compile(r"^\d{8}-\d{6}-.+$")


def construct_results_path(
    tree_model: str,
    sampling_method: str,
    experiment_name: Optional[str] = None,
    results_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Construct path to experiment results from components.

    Args:
        tree_model: Tree model name (e.g., 'kingman_mean', 'balanced_binary')
        sampling_method: Sampling method (e.g., 'uniform', 'leveraged', 'lds')
        experiment_name: Optional experiment directory name (e.g., '20260220-120000-...')
                        If None, returns path to sampling method directory
        results_dir: Optional base results directory. If None, uses default project structure

    Returns:
        Path to the experiment directory or sampling method directory

    Examples:
        >>> # Get path to specific experiment
        >>> path = construct_results_path('kingman_mean', 'lds', '20260220-120000-kingman_mean_n512_mu_0p1_lds')
        >>> # Returns: .../results/kingman_mean/lds/20260220-120000-kingman_mean_n512_mu_0p1_lds/

        >>> # Get path to all lds experiments for kingman_mean
        >>> path = construct_results_path('kingman_mean', 'lds')
        >>> # Returns: .../results/kingman_mean/lds/
    """
    if results_dir is None:
        # Default: go up from this file to sub_sampled_fielder_vec/results/
        results_dir = Path(__file__).resolve().parents[3] / "results"
    else:
        results_dir = Path(results_dir)

    path = results_dir / tree_model / sampling_method

    if experiment_name:
        path = path / experiment_name

    return path


def find_experiments(
    tree_model: str,
    sampling_method: str,
    results_dir: Optional[Union[str, Path]] = None,
) -> list[Path]:
    """Find all experiment directories for given tree model and sampling method.

    Args:
        tree_model: Tree model name
        sampling_method: Sampling method name
        results_dir: Optional base results directory

    Returns:
        List of paths to experiment directories, sorted by timestamp (newest first)

    Example:
        >>> experiments = find_experiments('kingman_mean', 'lds')
        >>> # Returns: [Path('.../20260220-120000-...'), Path('.../20260219-150000-...'), ...]
    """
    method_dir = construct_results_path(tree_model, sampling_method, results_dir=results_dir)

    if not method_dir.exists():
        return []

    experiments = []
    for entry in method_dir.iterdir():
        if entry.is_dir() and EXPERIMENT_DIR_PATTERN.match(entry.name):
            experiments.append(entry)

    # Sort by directory name (timestamp is at the beginning, so this sorts by time)
    experiments.sort(reverse=True)  # Newest first

    return experiments


def get_latest_experiment(
    tree_model: str,
    sampling_method: str,
    results_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Get the most recent experiment for given tree model and sampling method.

    Args:
        tree_model: Tree model name
        sampling_method: Sampling method name
        results_dir: Optional base results directory

    Returns:
        Path to the latest experiment directory, or None if none found

    Example:
        >>> latest = get_latest_experiment('kingman_mean', 'lds')
        >>> # Returns: Path('.../results/kingman_mean/lds/20260220-120000-...')
    """
    experiments = find_experiments(tree_model, sampling_method, results_dir)
    return experiments[0] if experiments else None


def parse_experiment_path(experiment_path: Union[str, Path]) -> dict[str, str]:
    """Parse experiment path to extract components.

    Args:
        experiment_path: Full path to experiment directory

    Returns:
        Dict with keys: tree_model, sampling_method, experiment_name, timestamp

    Example:
        >>> info = parse_experiment_path('.../results/kingman_mean/lds/20260220-120000-name')
        >>> # Returns: {'tree_model': 'kingman_mean', 'sampling_method': 'lds',
        >>>             'experiment_name': '20260220-120000-name', 'timestamp': '20260220-120000'}

    Raises:
        ValueError: If path doesn't match expected structure
    """
    experiment_path = Path(experiment_path).resolve()

    # Extract components from path
    # Expected structure: .../results/{tree_model}/{sampling_method}/{timestamp-name}/
    parts = experiment_path.parts

    # Find 'results' in the path
    try:
        results_idx = parts.index('results')
    except ValueError:
        raise ValueError(f"Path does not contain 'results' directory: {experiment_path}")

    # Check we have enough parts after 'results'
    if len(parts) < results_idx + 4:
        raise ValueError(
            f"Path does not match expected structure "
            f"results/{{tree_model}}/{{sampling_method}}/{{experiment_name}}: {experiment_path}"
        )

    tree_model = parts[results_idx + 1]
    sampling_method = parts[results_idx + 2]
    experiment_name = parts[results_idx + 3]

    # Extract timestamp from experiment_name
    timestamp_match = re.match(r"^(\d{8}-\d{6})-", experiment_name)
    timestamp = timestamp_match.group(1) if timestamp_match else None

    return {
        "tree_model": tree_model,
        "sampling_method": sampling_method,
        "experiment_name": experiment_name,
        "timestamp": timestamp,
    }


def list_all_experiments(results_dir: Optional[Union[str, Path]] = None) -> list[dict]:
    """List all experiments across all tree models and sampling methods.

    Args:
        results_dir: Optional base results directory

    Returns:
        List of dicts with keys: path, tree_model, sampling_method, experiment_name, timestamp

    Example:
        >>> all_experiments = list_all_experiments()
        >>> for exp in all_experiments:
        >>>     print(f"{exp['tree_model']}/{exp['sampling_method']}: {exp['experiment_name']}")
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parents[3] / "results"
    else:
        results_dir = Path(results_dir)

    if not results_dir.exists():
        return []

    experiments = []

    # Iterate through tree models
    for tree_model_dir in sorted(results_dir.iterdir()):
        if not tree_model_dir.is_dir() or tree_model_dir.name.startswith('.'):
            continue

        tree_model = tree_model_dir.name

        # Iterate through sampling methods
        for sampling_dir in sorted(tree_model_dir.iterdir()):
            if not sampling_dir.is_dir() or sampling_dir.name.startswith('.'):
                continue

            sampling_method = sampling_dir.name

            # Find all experiments in this method directory
            for exp_dir in find_experiments(tree_model, sampling_method, results_dir):
                info = {
                    "path": exp_dir,
                    "tree_model": tree_model,
                    "sampling_method": sampling_method,
                    "experiment_name": exp_dir.name,
                }

                # Extract timestamp
                timestamp_match = re.match(r"^(\d{8}-\d{6})-", exp_dir.name)
                if timestamp_match:
                    info["timestamp"] = timestamp_match.group(1)

                experiments.append(info)

    # Sort by timestamp descending (newest first)
    experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return experiments
