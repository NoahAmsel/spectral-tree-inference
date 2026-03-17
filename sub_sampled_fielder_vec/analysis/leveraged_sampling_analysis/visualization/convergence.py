"""IALM convergence visualization utilities.

This module provides plotting functions for visualizing IALM solver convergence.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from pathlib import Path


def plot_ialm_convergence(
    df: pd.DataFrame,
    n_taxa_values: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot IALM iterations vs p, colored by n.

    Args:
        df: DataFrame with mean_ialm_iterations column
        n_taxa_values: Optional list of n_taxa to plot
        output_path: Optional path to save figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure
    """
    if n_taxa_values is None:
        n_taxa_values = sorted(df["num_taxa"].unique())

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: IALM iterations vs p
    ax = axes[0]
    for n in n_taxa_values:
        n_df = df[df["num_taxa"] == n].sort_values("p")
        if "mean_ialm_iterations" in n_df.columns:
            ax.plot(
                n_df["p"],
                n_df["mean_ialm_iterations"],
                marker="o",
                label=f"n={n}",
                linewidth=2,
            )
    ax.set_xlabel("Sampling probability p", fontsize=12)
    ax.set_ylabel("IALM Iterations", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("IALM Convergence Iterations", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Operator norm error vs p
    ax = axes[1]
    for n in n_taxa_values:
        n_df = df[df["num_taxa"] == n].sort_values("p")
        if "mean_operator_norm_error" in n_df.columns:
            # Filter out NaN and extreme values for visualization
            valid_mask = ~np.isnan(n_df["mean_operator_norm_error"])
            valid_df = n_df[valid_mask]
            if len(valid_df) > 0:
                ax.plot(
                    valid_df["p"],
                    valid_df["mean_operator_norm_error"],
                    marker="o",
                    label=f"n={n}",
                    linewidth=2,
                )
    ax.set_xlabel("Sampling probability p", fontsize=12)
    ax.set_ylabel("Operator Norm Error", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Matrix Recovery Error", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
