"""Data loading and parsing utilities for leveraged sampling experiments.

This module provides functions to load experiment results from JSON files
and convert them into structured formats for analysis.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


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

    # Try parent directory
    parent_config = run_dir.parent / "config.json"
    if parent_config.exists():
        with parent_config.open() as f:
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
            # Create row dict with explicit n_taxa and sequence_length
            df_row = {"num_taxa": n_taxa, "sequence_length": seq_len}
            for col in columns:
                df_row[col] = row.get(col)
            rows.append(df_row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort by sequence_length, num_taxa, p
    if "p" in df.columns:
        df = df.sort_values(["sequence_length", "num_taxa", "p"])
    else:
        df = df.sort_values(["sequence_length", "num_taxa"])

    return df


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
