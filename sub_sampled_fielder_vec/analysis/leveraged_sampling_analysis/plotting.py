"""Visualization utilities for leveraged sampling analysis.

This module provides plotting functions for visualizing experiment results,
phase transitions, and leveraged-specific diagnostics.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_agreement_vs_p(
    data: Dict[str, pd.DataFrame],
    methods: List[str] = ["uniform", "leveraged"],
    n_taxa_values: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """Plot partition agreement vs p for each method and n.

    Args:
        data: Dict mapping method -> DataFrame with columns: p, partition_agreement_M, num_taxa
        methods: List of methods to plot
        n_taxa_values: Optional list of n_taxa to plot (if None, plots all available)
        output_path: Optional path to save figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure
    """
    # Get all n_taxa values if not specified
    if n_taxa_values is None:
        all_n_values = set()
        for method in methods:
            if method in data:
                all_n_values.update(data[method]["num_taxa"].unique())
        n_taxa_values = sorted(all_n_values)

    n_plots = len(n_taxa_values)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        axes = axes.flatten()

    colors = {"uniform": "#2E86AB", "leveraged": "#A23B72"}
    markers = {"uniform": "o", "leveraged": "s"}

    for idx, n in enumerate(n_taxa_values):
        ax = axes[idx]

        for method in methods:
            if method not in data:
                continue

            method_df = data[method]
            n_df = method_df[method_df["num_taxa"] == n].sort_values("p")

            if len(n_df) == 0:
                continue

            ax.plot(
                n_df["p"],
                n_df["partition_agreement_M"],
                label=method.capitalize(),
                marker=markers.get(method, "o"),
                linewidth=2,
                markersize=6,
                color=colors.get(method, None),
            )

        ax.set_xlabel("Sampling probability p", fontsize=12)
        ax.set_ylabel("Partition agreement (%)", fontsize=12)
        ax.set_xscale("log")
        ax.set_title(f"n = {n} taxa", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        ax.axhline(95, color="red", linestyle="--", alpha=0.5, linewidth=1)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_phase_transition_scaling(
    transitions_df: pd.DataFrame,
    method_col: str = "method",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot p* vs n with power law fits.

    Args:
        transitions_df: DataFrame with columns: n_taxa, p_star_sigmoid_95, method_col
        method_col: Column name for method (default "method")
        output_path: Optional path to save figure
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    methods = transitions_df[method_col].unique() if method_col in transitions_df.columns else [None]
    colors = {"uniform": "#2E86AB", "leveraged": "#A23B72"}
    markers = {"uniform": "o", "leveraged": "s"}

    for method in methods:
        if method_col in transitions_df.columns:
            method_df = transitions_df[transitions_df[method_col] == method]
            label_prefix = method.capitalize()
        else:
            method_df = transitions_df
            label_prefix = "Method"

        n_vals = method_df["n_taxa"].values
        p_sigmoid = method_df["p_star_sigmoid_95"].values
        p_discrete = method_df["p_star_discrete_100"].values

        color = colors.get(method, None) if method else "#2E86AB"
        marker = markers.get(method, "o") if method else "o"

        # LEFT: Sigmoid-based (95% threshold)
        mask_sigmoid = ~np.isnan(p_sigmoid)
        if np.sum(mask_sigmoid) >= 2:
            n_clean = n_vals[mask_sigmoid]
            p_clean = p_sigmoid[mask_sigmoid]

            # Fit power law
            from .metrics import fit_scaling_law

            alpha, A, equation, p_fit = fit_scaling_law(n_clean, p_clean)

            if not np.isnan(alpha):
                # Plot fit
                n_range = np.logspace(
                    np.log10(n_clean.min() * 0.7), np.log10(n_clean.max() * 1.3), 100
                )
                p_fit_range = A * (n_range ** alpha)
                ax1.plot(
                    n_range,
                    p_fit_range,
                    color=color,
                    linestyle="--",
                    linewidth=3,
                    alpha=0.7,
                    label=f"{label_prefix}: {equation}",
                )

                # Add exponent annotation
                ax1.text(
                    0.05,
                    0.95,
                    f"{label_prefix} α = {alpha:.3f}",
                    transform=ax1.transAxes,
                    fontsize=11,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            # Data points
            ax1.scatter(
                n_clean,
                p_clean,
                marker=marker,
                s=250,
                color=color,
                edgecolor="black",
                linewidth=2,
                zorder=3,
            )

        # RIGHT: Discrete (100% threshold)
        mask_discrete = ~np.isnan(p_discrete)
        if np.sum(mask_discrete) >= 2:
            n_clean = n_vals[mask_discrete]
            p_clean = p_discrete[mask_discrete]

            # Fit power law
            from .metrics import fit_scaling_law

            alpha_d, A_d, eq_d, p_fit_d = fit_scaling_law(n_clean, p_clean)

            if not np.isnan(alpha_d):
                n_range_d = np.logspace(
                    np.log10(n_clean.min() * 0.7), np.log10(n_clean.max() * 1.3), 100
                )
                p_fit_range_d = A_d * (n_range_d ** alpha_d)
                ax2.plot(
                    n_range_d,
                    p_fit_range_d,
                    color=color,
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.7,
                    label=f"{label_prefix}: {eq_d}",
                )

                ax2.text(
                    0.05,
                    0.95,
                    f"{label_prefix} α = {alpha_d:.3f}",
                    transform=ax2.transAxes,
                    fontsize=11,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                )

            ax2.scatter(
                n_clean,
                p_clean,
                marker=marker,
                s=250,
                color=color,
                edgecolor="black",
                linewidth=2,
                zorder=3,
            )

            if len(n_clean) > 1:
                ax2.plot(n_clean, p_clean, color=color, linestyle="-", linewidth=2, alpha=0.6)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Taxa (n)", fontsize=15, fontweight="bold")
    ax1.set_ylabel("Critical p* (95% agreement)", fontsize=15, fontweight="bold")
    ax1.set_title("Sigmoid-Based Threshold", fontsize=17, fontweight="bold", pad=15)
    ax1.legend(fontsize=12, loc="best", framealpha=0.95)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.tick_params(labelsize=12)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Taxa (n)", fontsize=15, fontweight="bold")
    ax2.set_ylabel("First p* (100% agreement)", fontsize=15, fontweight="bold")
    ax2.set_title("Discrete Threshold", fontsize=17, fontweight="bold", pad=15)
    ax2.legend(fontsize=12, loc="best", framealpha=0.95)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.tick_params(labelsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


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
                    label=f"n={n}, σ{i+1}",
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
                label=f"n={n} (σ₂/σ₁)",
                linewidth=2,
            )
    ax.set_xlabel("Sampling probability p", fontsize=12)
    ax.set_ylabel("Rank Quality (σ₂/σ₁)", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("Phase 1 Rank Quality", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


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
