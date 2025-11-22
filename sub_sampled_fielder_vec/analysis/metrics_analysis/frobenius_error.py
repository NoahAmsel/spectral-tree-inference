"""Analyze Frobenius error behavior."""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared import plot_metric_faceted


def create_plot(df: pd.DataFrame, output_path: Path):
    """
    Create faceted plot of Frobenius error vs p.

    Args:
        df: Full results DataFrame
        output_path: Where to save the plot
    """
    plot_metric_faceted(
        df,
        metric='mean_frobenius_error',
        ylabel='Frobenius Error (||M - S||_F)',
        output_path=output_path,
        log_y=True,
        show_partition_agreement=True
    )


def analyze(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Run Frobenius error analysis.

    Args:
        df: Full results DataFrame
        output_dir: Directory to save outputs

    Returns:
        Empty dict (no special statistics needed)
    """
    print("\n=== Analyzing Frobenius Error ===")

    # Create plot
    create_plot(df, output_dir / "frobenius_error_vs_p.png")

    return {}
