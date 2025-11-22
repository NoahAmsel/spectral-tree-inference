"""Analyze quality gap between partition_agreement_M and partition_agreement_S."""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import plot_gap_vs_p


def compute_stats(df: pd.DataFrame) -> dict:
    """
    Compute gap statistics.

    Args:
        df: DataFrame with partition_agreement_M and partition_agreement_S

    Returns:
        Dictionary of gap statistics
    """
    if 'partition_agreement_M' not in df.columns or 'partition_agreement_S' not in df.columns:
        return {}

    gap = df['partition_agreement_M'] - df['partition_agreement_S']

    stats = {
        'mean_gap': float(gap.mean()),
        'median_gap': float(gap.median()),
        'max_gap': float(gap.max()),
        'min_gap': float(gap.min()),
    }

    # Find where gap becomes small (<5%)
    small_gap = df[gap < 5.0]
    stats['small_gap_count'] = len(small_gap)
    stats['small_gap_percent'] = float(100 * len(small_gap) / len(df))

    return stats


def analyze(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Run quality gap analysis (M vs S_avg).

    Args:
        df: Full results DataFrame
        output_dir: Directory to save outputs

    Returns:
        Gap statistics dictionary
    """
    print("\n=== Analyzing Partition Quality Gap (M vs S) ===")

    if 'partition_agreement_M' not in df.columns or 'partition_agreement_S' not in df.columns:
        print("  Partition agreement columns not found. Skipping.")
        return {}

    # Create gap plot
    plot_gap_vs_p(df, output_dir / "partition_M_vs_S_gap.png")

    # Compute statistics
    stats = compute_stats(df)

    if stats:
        print(f"  Mean gap: {stats['mean_gap']:.2f}%")
        print(f"  Max gap: {stats['max_gap']:.2f}%")
        print(f"  Gap < 5% for {stats['small_gap_percent']:.1f}% of configurations")

    return stats
