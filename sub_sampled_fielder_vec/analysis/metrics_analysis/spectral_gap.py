"""Analyze spectral gap ratio behavior at transition."""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import plot_metric_faceted, find_all_transitions


def compute_stats(transitions_df: pd.DataFrame) -> dict:
    """
    Compute spectral gap ratio statistics at transition points.

    Args:
        transitions_df: DataFrame of transition points

    Returns:
        Dictionary of statistics
    """
    if len(transitions_df) == 0:
        return {}

    stats = {
        'mean': float(transitions_df['spectral_gap_ratio'].mean()),
        'median': float(transitions_df['spectral_gap_ratio'].median()),
        'min': float(transitions_df['spectral_gap_ratio'].min()),
        'max': float(transitions_df['spectral_gap_ratio'].max()),
    }

    # Check how many transitions have ratio < 10
    below_10 = (transitions_df['spectral_gap_ratio'] < 10).sum()
    stats['below_10_count'] = int(below_10)
    stats['below_10_percent'] = float(100 * below_10 / len(transitions_df))

    return stats


def create_plot(df: pd.DataFrame, output_path: Path):
    """
    Create faceted plot of spectral gap ratio vs p.

    Args:
        df: Full results DataFrame
        output_path: Where to save the plot
    """
    plot_metric_faceted(
        df,
        metric='spectral_gap_ratio',
        ylabel='Spectral Gap Ratio (L_S / L_M)',
        output_path=output_path,
        log_y=True,
        show_partition_agreement=True
    )


def analyze(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Run complete spectral gap analysis.

    Args:
        df: Full results DataFrame
        output_dir: Directory to save outputs

    Returns:
        Statistics dictionary for report
    """
    print("\n=== Analyzing Spectral Gap Ratio ===")

    # Create plot
    create_plot(df, output_dir / "spectral_gap_ratio_vs_p.png")

    # Compute statistics at transition points
    transitions = find_all_transitions(df, threshold=90.0)
    stats = compute_stats(transitions)

    if stats:
        print(f"  Mean at transition: {stats['mean']:.2f}")
        print(f"  Median at transition: {stats['median']:.2f}")
        print(f"  Transitions with ratio < 10: {stats['below_10_count']}/{len(transitions)} ({stats['below_10_percent']:.1f}%)")

    return stats
