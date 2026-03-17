"""Phase transition scaling visualization utilities.

This module provides plotting functions for visualizing phase transition scaling laws.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from pathlib import Path

# Import metrics for scaling law fitting
import sys
from pathlib import Path as PathLib
sys.path.append(str(PathLib(__file__).parent.parent.parent.parent))
from analysis.leveraged_sampling_analysis.metrics.phase_transition import fit_scaling_law


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

    # Track y-position for annotations to avoid overlap
    annotation_y_pos_ax1 = 0.95
    annotation_y_pos_ax2 = 0.95
    annotation_spacing = 0.12  # Vertical spacing between annotations

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

                # Add exponent annotation (stacked vertically for multiple methods)
                ax1.text(
                    0.05,
                    annotation_y_pos_ax1,
                    f"{label_prefix} α = {alpha:.3f}",
                    transform=ax1.transAxes,
                    fontsize=11,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )
                annotation_y_pos_ax1 -= annotation_spacing  # Move down for next method

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
                    annotation_y_pos_ax2,
                    f"{label_prefix} α = {alpha_d:.3f}",
                    transform=ax2.transAxes,
                    fontsize=11,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                )
                annotation_y_pos_ax2 -= annotation_spacing  # Move down for next method

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
