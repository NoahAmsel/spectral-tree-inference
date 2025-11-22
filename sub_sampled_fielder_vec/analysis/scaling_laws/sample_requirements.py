"""Analyze effective sample requirements at transition."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import find_all_transitions, plot_metric_faceted


def compute_stats(transitions: pd.DataFrame) -> dict:
    """
    Compute sample requirement statistics.

    Args:
        transitions: DataFrame with transition points

    Returns:
        Dictionary of statistics
    """
    if len(transitions) == 0 or 'effective_samples' not in transitions.columns:
        return {}

    eff_samples = transitions['effective_samples']

    stats = {
        'mean_effective_samples': float(eff_samples.mean()),
        'median_effective_samples': float(eff_samples.median()),
        'min_effective_samples': float(eff_samples.min()),
        'max_effective_samples': float(eff_samples.max()),
    }

    # Compute required samples per matrix entry
    if 'num_taxa' in transitions.columns:
        transitions_copy = transitions.copy()
        transitions_copy['matrix_size'] = transitions_copy['num_taxa'] ** 2
        transitions_copy['samples_per_entry'] = (
            transitions_copy['effective_samples'] / transitions_copy['matrix_size']
        )
        stats['mean_samples_per_entry'] = float(transitions_copy['samples_per_entry'].mean())
        stats['median_samples_per_entry'] = float(transitions_copy['samples_per_entry'].median())

    return stats


def create_plot(df: pd.DataFrame, output_path: Path):
    """
    Plot effective samples vs p.

    Args:
        df: Full results DataFrame
        output_path: Where to save the plot
    """
    if 'effective_samples' not in df.columns:
        print("  effective_samples column not found")
        return

    plot_metric_faceted(
        df,
        metric='effective_samples',
        ylabel='Effective Samples',
        output_path=output_path
    )


def analyze(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Run sample requirements analysis.

    Args:
        df: Full results DataFrame
        output_dir: Directory to save outputs

    Returns:
        Sample statistics dictionary
    """
    print("\n=== Sample Requirements Analysis ===")

    # Find transitions
    transitions = find_all_transitions(df, threshold=90.0)

    # Compute stats at transition
    stats = compute_stats(transitions)

    if stats:
        print(f"  Mean effective samples at transition: {stats['mean_effective_samples']:.0f}")
        if 'mean_samples_per_entry' in stats:
            print(f"  Mean samples per matrix entry: {stats['mean_samples_per_entry']:.2f}")

    # Create plot
    create_plot(df, output_dir / "effective_samples_vs_p.png")

    return stats
