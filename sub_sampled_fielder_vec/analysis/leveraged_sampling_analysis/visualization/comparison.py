"""Method comparison visualization utilities.

This module provides plotting functions for comparing different sampling methods.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from pathlib import Path


def plot_method_comparison_heatmap(
    uniform_df: pd.DataFrame,
    leveraged_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot agreement difference (leveraged - uniform) as heatmap.

    Args:
        uniform_df: DataFrame for uniform method
        leveraged_df: DataFrame for leveraged method
        output_path: Optional path to save figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure
    """
    # Get common n_taxa and p values
    n_values = sorted(set(uniform_df["num_taxa"].unique()) & set(leveraged_df["num_taxa"].unique()))
    p_values = sorted(set(uniform_df["p"].unique()) & set(leveraged_df["p"].unique()))

    # Create difference matrix
    diff_matrix = np.zeros((len(n_values), len(p_values)))

    for i, n in enumerate(n_values):
        for j, p in enumerate(p_values):
            u_agr = uniform_df[
                (uniform_df["num_taxa"] == n) & (uniform_df["p"] == p)
            ]["partition_agreement_M"].values
            l_agr = leveraged_df[
                (leveraged_df["num_taxa"] == n) & (leveraged_df["p"] == p)
            ]["partition_agreement_M"].values

            if len(u_agr) > 0 and len(l_agr) > 0:
                diff_matrix[i, j] = l_agr[0] - u_agr[0]
            else:
                diff_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(diff_matrix, aspect="auto", cmap="RdYlGn", vmin=-20, vmax=20)
    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels([f"{p:.2e}" for p in p_values], rotation=45, ha="right")
    ax.set_yticks(range(len(n_values)))
    ax.set_yticklabels(n_values)
    ax.set_xlabel("Sampling probability p", fontsize=12)
    ax.set_ylabel("Number of Taxa (n)", fontsize=12)
    ax.set_title("Agreement Difference: Leveraged - Uniform (%)", fontsize=14, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Agreement Difference (%)")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
