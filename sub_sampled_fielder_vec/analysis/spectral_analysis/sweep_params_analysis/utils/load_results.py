"""Load experimental results from JSON file."""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load results_grid_merged.json from experiment directory.

    Args:
        results_dir: Path to results directory containing results_grid_merged.json

    Returns:
        List of result rows as dictionaries

    Raises:
        FileNotFoundError: If results_grid_merged.json doesn't exist
        ValueError: If JSON structure is invalid
    """
    json_path = results_dir / "results_grid_merged.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if "rows" not in data:
        raise ValueError("JSON must contain 'rows' key")

    return data["rows"]
