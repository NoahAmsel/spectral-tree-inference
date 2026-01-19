"""Leverage score visualization utilities.

This module provides plotting functions for visualizing leverage score diagnostics
and Phase 1 SVD quality.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from pathlib import Path


def plot_leverage_diagnostics(
    df: pd.DataFrame,
    n_taxa_values: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Plot 2x2 grid of leverage diagnostics.

    Args:
        df: DataFrame with leveraged metrics (must have leveraged data)
        n_taxa_values: Optional list of n_taxa to plot
        output_path: Optional path to save figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure
    """
    if n_taxa_values is None:
        n_taxa_values = sorted(df["num_taxa"].unique())

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Plot 1: Leverage max vs p
    ax = axes[0]
    for n in n_taxa_values:
        n_df = df[df["num_taxa"] == n].sort_values("p")
        if "mean_leverage_max" in n_df.columns:
            ax.plot(
                n_df["p"],
                n_df["mean_leverage_max"],
                marker="o",
                label=f"n={n}",
                linewidth=2,
            )
    ax.set_xlabel("Sampling probability p", fontsize=12)
    ax.set_ylabel("Max Leverage Score", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("Leverage Score Maximum", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Leverage sum vs p
    ax = axes[1]
    for n in n_taxa_values:
        n_df = df[df["num_taxa"] == n].sort_values("p")
        if "mean_leverage_sum" in n_df.columns:
            ax.plot(
                n_df["p"],
                n_df["mean_leverage_sum"],
                marker="o",
                label=f"n={n}",
                linewidth=2,
            )
    ax.set_xlabel("Sampling probability p", fontsize=12)
    ax.set_ylabel("Sum of Leverage Scores", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("Leverage Score Sum", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Symmetry error vs p
    ax = axes[2]
    for n in n_taxa_values:
        n_df = df[df["num_taxa"] == n].sort_values("p")
        if "mean_leverage_symmetry_error" in n_df.columns:
            ax.plot(
                n_df["p"],
                n_df["mean_leverage_symmetry_error"],
                marker="o",
                label=f"n={n}",
                linewidth=2,
            )
    ax.set_xlabel("Sampling probability p", fontsize=12)
    ax.set_ylabel("Leverage Symmetry Error", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("Leverage Symmetry Error", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: IPR vs p
    ax = axes[3]
    for n in n_taxa_values:
        n_df = df[df["num_taxa"] == n].sort_values("p")
        if "mean_ipr_S" in n_df.columns:
            ax.plot(
                n_df["p"],
                n_df["mean_ipr_S"],
                marker="o",
                label=f"n={n}",
                linewidth=2,
            )
    ax.set_xlabel("Sampling probability p", fontsize=12)
    ax.set_ylabel("Inverse Participation Ratio (IPR)", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("Inverse Participation Ratio", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_phase1_svd(
    df: pd.DataFrame,
    n_taxa_values: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot Phase 1 singular values evolution across p.

    Args:
        df: DataFrame with phase1_s1, phase1_s2, phase1_s3 columns
        n_taxa_values: Optional list of n_taxa to plot
        output_path: Optional path to save figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure
    """
    if n_taxa_values is None:
        n_taxa_values = sorted(df["num_taxa"].unique())

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Singular values vs p
    ax = axes[0]
    for n in n_taxa_values:
        n_df = df[df["num_taxa"] == n].sort_values("p")
        for i, col in enumerate(["mean_phase1_s1", "mean_phase1_s2", "mean_phase1_s3"]):
            if col in n_df.columns:
                ax.plot(
                    n_df["p"],
                    n_df[col],
                    marker="o",
                    label=f"n={n}, s{i+1}",
                    linewidth=2,
                    alpha=0.7,
                )
    ax.set_xlabel("Sampling probability p", fontsize=12)
    ax.set_ylabel("Singular Value", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("Phase 1 Singular Values", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: Rank quality metrics
    ax = axes[1]
    for n in n_taxa_values:
        n_df = df[df["num_taxa"] == n].sort_values("p")
        if "mean_phase1_s1" in n_df.columns and "mean_phase1_s2" in n_df.columns:
            s1 = n_df["mean_phase1_s1"].values
            s2 = n_df["mean_phase1_s2"].values
            rank_quality = np.where(s1 > 0, s2 / s1, np.nan)
            ax.plot(
                n_df["p"],
                rank_quality,
                marker="o",
                label=f"n={n} (s2/s1)",
                linewidth=2,
            )
    ax.set_xlabel("Sampling probability p", fontsize=12)
    ax.set_ylabel("Rank Quality (s2/s1)", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("Phase 1 Rank Quality", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
