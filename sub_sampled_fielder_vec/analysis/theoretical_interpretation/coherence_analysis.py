"""Analyze eigenvector coherence (incoherence property)."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import find_all_transitions, plot_metric_faceted


def compute_stats(transitions: pd.DataFrame) -> dict:
    """
    Compute coherence statistics at transition.

    Args:
        transitions: DataFrame with transition points

    Returns:
        Dictionary of coherence statistics
    """
    if len(transitions) == 0 or 'mean_coherence_L_S' not in transitions.columns:
        return {}

    coherence = transitions['mean_coherence_L_S']

    stats = {
        'mean_coherence': float(coherence.mean()),
        'median_coherence': float(coherence.median()),
        'max_coherence': float(coherence.max()),
        'min_coherence': float(coherence.min()),
    }

    return stats


def create_plot(df: pd.DataFrame, output_path: Path):
    """
    Plot coherence vs p.

    Args:
        df: Full results DataFrame
        output_path: Where to save the plot
    """
    if 'mean_coherence_L_S' not in df.columns:
        print("  Coherence column not found")
        return

    plot_metric_faceted(
        df,
        metric='mean_coherence_L_S',
        ylabel='Coherence',
        output_path=output_path
    )


def analyze(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Run coherence analysis.

    Args:
        df: Full results DataFrame
        output_dir: Directory to save outputs

    Returns:
        Coherence statistics dictionary
    """
    print("\n=== Coherence Analysis ===")

    # Find transitions
    transitions = find_all_transitions(df, threshold=90.0)

    # Compute stats
    stats = compute_stats(transitions)

    if stats:
        print(f"  Mean coherence at transition: {stats['mean_coherence']:.4f}")
        print(f"  Max coherence at transition: {stats['max_coherence']:.4f}")

    # Create plot
    create_plot(df, output_dir / "coherence_vs_p.png")

    return stats
