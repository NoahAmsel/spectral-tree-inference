"""Load and prepare experimental data for analysis."""

import json
import pandas as pd
from pathlib import Path
from typing import Optional


def load_results(results_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load results_grid.json and convert to pandas DataFrame.

    Args:
        results_path: Path to results_grid.json. If None, uses default location.

    Returns:
        DataFrame with all results including computed columns
    """
    if results_path is None:
        results_path = Path(__file__).parent.parent.parent / "results" / "combined_grid_search_results" / "results_grid.json"

    with open(results_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data['rows'])

    # Add computed columns for convenience
    df['spectral_gap_ratio'] = df['mean_spectral_gap_L_S'] / df['mean_spectral_gap_L_M']
    df['rank_ratio_L_S'] = df['mean_empirical_rank_L_S'] / df['num_taxa']
    df['rank_ratio_L_M'] = df['mean_empirical_rank_L_M'] / df['num_taxa']
    df['matrix_size'] = df['num_taxa'] * df['sequence_length']
    df['effective_samples'] = df['p'] * df['num_taxa'] ** 2

    return df


def get_output_dir(analysis_name: str) -> Path:
    """
    Get output directory for a specific analysis.

    Args:
        analysis_name: Name of analysis (e.g., 'metrics_analysis', 'scaling_laws')

    Returns:
        Path to output directory (created if doesn't exist)
    """
    base_dir = Path(__file__).parent.parent.parent / "results" / "combined_grid_search_results" / "analysis_outputs"
    analysis_dir = base_dir / analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir
