"""Analyze when rank becomes full."""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import plot_metric_faceted, find_all_transitions


def compute_stats(transitions_df: pd.DataFrame) -> dict:
    """
    Compute rank ratio statistics at transition points.

    Args:
        transitions_df: DataFrame of transition points

    Returns:
        Dictionary of statistics
    """
    if len(transitions_df) == 0:
        return {}

    stats = {
        'mean': float(transitions_df['rank_ratio_L_S'].mean()),
        'median': float(transitions_df['rank_ratio_L_S'].median()),
        'min': float(transitions_df['rank_ratio_L_S'].min()),
    }

    # Check how many have full rank (≥99.9%)
    full_rank = (transitions_df['rank_ratio_L_S'] >= 0.999).sum()
    stats['full_rank_count'] = int(full_rank)
    stats['full_rank_percent'] = float(100 * full_rank / len(transitions_df))

    return stats


def create_plot(df: pd.DataFrame, output_path: Path):
    """
    Create faceted plot of rank ratio vs p.

    Args:
        df: Full results DataFrame
        output_path: Where to save the plot
    """
    plot_metric_faceted(
        df,
        metric='rank_ratio_L_S',
        ylabel='Rank Ratio (Rank_L_S / n)',
        output_path=output_path,
        log_y=False,
        show_partition_agreement=True
    )


def analyze(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Run complete rank recovery analysis.

    Args:
        df: Full results DataFrame
        output_dir: Directory to save outputs

    Returns:
        Statistics dictionary for report
    """
    print("\n=== Analyzing Rank Recovery ===")

    # Create plot
    create_plot(df, output_dir / "rank_ratio_L_S_vs_p.png")

    # Compute statistics at transition points
    transitions = find_all_transitions(df, threshold=90.0)
    stats = compute_stats(transitions)

    if stats:
        print(f"  Mean rank ratio at transition: {stats['mean']:.4f}")
        print(f"  Full rank (≥99.9%): {stats['full_rank_count']}/{len(transitions)} ({stats['full_rank_percent']:.1f}%)")

    return stats
