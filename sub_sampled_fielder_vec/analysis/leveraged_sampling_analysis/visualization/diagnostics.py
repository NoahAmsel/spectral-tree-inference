"""Diagnostic visualization utilities for leveraged sampling analysis.

Simple plotting functions to answer specific diagnostic questions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path

import sys
from pathlib import Path as PathLib
sys.path.append(str(PathLib(__file__).parent.parent.parent.parent))
from analysis.leveraged_sampling_analysis.metrics.diagnostics import theoretical_p_star


def plot_leverage_histogram(
    df: pd.DataFrame,
    n_taxa: int,
    p_value: Optional[float] = None,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot histogram of leverage scores for a specific (n, p) configuration.

    Question: Are leverage scores concentrated or nearly uniform?

    Args:
        df: DataFrame with leverage score columns
        n_taxa: Which n value to analyze
        p_value: Optional specific p value (if None, uses highest p)
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Filter to specific n
    n_df = df[df["num_taxa"] == n_taxa].sort_values("p")

    if len(n_df) == 0:
        raise ValueError(f"No data found for n={n_taxa}")

    # Select p value
    if p_value is None:
        # Find highest p that has valid leverage data (not NaN)
        # At high p values, IALM is bypassed and leverage metrics are NaN
        if "mean_leverage_max" in n_df.columns:
            valid_rows = n_df[n_df["mean_leverage_max"].notna()]
            if len(valid_rows) > 0:
                p_value = valid_rows["p"].max()
                row = valid_rows[valid_rows["p"] == p_value].iloc[0]
            else:
                # Fall back to highest p if no valid leverage data
                p_value = n_df["p"].max()
                row = n_df[n_df["p"] == p_value].iloc[0]
        else:
            # No leverage columns at all
            p_value = n_df["p"].max()
            row = n_df[n_df["p"] == p_value].iloc[0]
    else:
        # Find closest matching p value (exact match may not exist due to log-spaced p values)
        p_diff = (n_df["p"] - p_value).abs()
        closest_idx = p_diff.idxmin()
        actual_p = n_df.loc[closest_idx, "p"]
        # Always use closest match, but warn if difference is large
        if abs(actual_p - p_value) > 0.5 * p_value:  # More than 50% difference
            print(f"Note: Using p={actual_p:.6f} (closest to requested p={p_value})")
        p_value = actual_p
        row = n_df.loc[closest_idx]

    # Extract leverage statistics
    # Note: We don't have individual leverage scores, only aggregated stats
    # So we'll visualize the distribution indirectly
    leverage_max = row.get("mean_leverage_max", np.nan)
    leverage_mean = row.get("mean_leverage_sum", n_taxa) / n_taxa  # sum = n by construction
    leverage_std = row.get("mean_leverage_std", np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Summary statistics as bar chart
    stats = {
        "Max": leverage_max,
        "Mean": leverage_mean,
        "Std": leverage_std,
    }

    ax1.bar(stats.keys(), stats.values(), color=["#A23B72", "#2E86AB", "#F18F01"])
    ax1.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax1.set_title(f"Leverage Score Statistics\n(n={n_taxa}, p={p_value:.4f})",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add concentration ratio annotation
    if not np.isnan(leverage_max) and leverage_mean > 0:
        concentration = leverage_max / leverage_mean
        ax1.text(
            0.5, 0.95,
            f"Concentration: max/mean = {concentration:.2f}",
            transform=ax1.transAxes,
            ha="center", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            fontsize=11,
        )

    # Right: Conceptual distribution visualization
    # Since we don't have individual scores, show expected distribution
    if not np.isnan(leverage_std) and leverage_mean > 0:
        # Approximate as log-normal or gamma distribution
        # For visualization: assume ~Gamma distribution
        ax2.text(
            0.5, 0.5,
            "Individual leverage scores\nnot stored in results.\n\n"
            "Statistics show:\n"
            f"• Max/Mean = {leverage_max/leverage_mean:.2f}\n"
            f"• CV (std/mean) = {leverage_std/leverage_mean:.2f}\n\n"
            "Higher ratios → more concentrated",
            transform=ax2.transAxes,
            ha="center", va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
        )
        ax2.axis("off")
    else:
        ax2.text(
            0.5, 0.5,
            "Leverage statistics not available",
            transform=ax2.transAxes,
            ha="center", va="center",
            fontsize=14,
        )
        ax2.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")

    return fig


def plot_phase1_budget(
    df: pd.DataFrame,
    n_taxa_list: Optional[List[int]] = None,
    target_rank: int = 2,
    theta: float = 0.3,
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot fraction of sample budget used for Phase 1 vs Phase 2.

    Question: How much budget is wasted on Phase 1 uniform sampling?

    Args:
        df: DataFrame with leveraged data
        n_taxa_list: List of n values to plot (if None, uses all)
        target_rank: Target rank for Phase 1
        theta: Oversampling parameter
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from analysis.leveraged_sampling_analysis.metrics.diagnostics import compute_phase1_sample_fraction

    if n_taxa_list is None:
        n_taxa_list = sorted(df["num_taxa"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Phase 1 fraction vs p for each n
    for n in n_taxa_list:
        n_df = df[df["num_taxa"] == n].sort_values("p")

        if len(n_df) == 0:
            continue

        p_vals = n_df["p"].values
        fractions = [
            compute_phase1_sample_fraction(p, n, target_rank, theta) for p in p_vals
        ]

        ax1.plot(p_vals, fractions, marker="o", label=f"n={n}", linewidth=2)

    ax1.set_xlabel("Sampling probability p", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Phase 1 sample fraction", fontsize=12, fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_title("Phase 1 Budget Overhead", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add reference lines
    ax1.axhline(0.2, color="orange", linestyle="--", alpha=0.5, linewidth=1.5,
                label="20% (moderate)")
    ax1.axhline(0.5, color="red", linestyle="--", alpha=0.5, linewidth=1.5,
                label="50% (bad)")
    ax1.legend(fontsize=10)

    # Plot 2: Sample counts for a specific (n, p)
    # Choose middle n and middle p for illustration
    if len(n_taxa_list) > 0:
        mid_n = n_taxa_list[len(n_taxa_list) // 2]
        mid_n_df = df[df["num_taxa"] == mid_n].sort_values("p")

        if len(mid_n_df) > 0:
            # Choose middle p value
            mid_idx = len(mid_n_df) // 2
            mid_p = mid_n_df.iloc[mid_idx]["p"]

            # Compute sample counts
            total_samples = mid_p * mid_n * mid_n
            phase1_samples = target_rank * mid_n * np.log(mid_n) / theta
            phase2_samples = total_samples - phase1_samples

            # Bar chart
            categories = ["Phase 1\n(Uniform)", "Phase 2\n(Leveraged)"]
            values = [phase1_samples, max(0, phase2_samples)]
            colors = ["#2E86AB", "#A23B72"]

            bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2)

            ax2.set_ylabel("Number of samples", fontsize=12, fontweight="bold")
            ax2.set_title(f"Sample Allocation\n(n={mid_n}, p={mid_p:.4f})",
                          fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{int(val):,}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

            # Add percentage annotation
            if total_samples > 0:
                phase1_pct = 100 * phase1_samples / total_samples
                ax2.text(
                    0.5, 0.95,
                    f"Phase 1 = {phase1_pct:.1f}% of total budget",
                    transform=ax2.transAxes,
                    ha="center", va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
                    fontsize=11,
                )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")

    return fig


def plot_vs_theory(
    transitions_df: pd.DataFrame,
    method_col: str = "method",
    threshold_col: str = "p_star_discrete_100",
    target_rank: int = 2,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 7),
) -> plt.Figure:
    """Plot actual p* vs theoretical predictions from matrix completion theory.

    Question: Are we in the right ballpark compared to theory?

    Args:
        transitions_df: DataFrame with n_taxa, method, p_star columns
        method_col: Column name for method
        threshold_col: Which p* to use ("p_star_sigmoid_95" or "p_star_discrete_100")
        target_rank: Matrix rank (typically 2)
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    methods = transitions_df[method_col].unique() if method_col in transitions_df.columns else ["method"]
    colors = {"uniform": "#2E86AB", "leveraged": "#A23B72"}
    markers = {"uniform": "o", "leveraged": "s"}

    # Get n range for theory lines
    all_n = transitions_df["n_taxa"].values
    n_range = np.logspace(np.log10(all_n.min() * 0.8), np.log10(all_n.max() * 1.2), 100)

    for method in methods:
        if method_col in transitions_df.columns:
            method_df = transitions_df[transitions_df[method_col] == method]
            label = method.capitalize()
        else:
            method_df = transitions_df
            label = "Method"

        n_vals = method_df["n_taxa"].values
        p_star_vals = method_df[threshold_col].values

        # Remove NaN
        mask = ~np.isnan(p_star_vals)
        n_clean = n_vals[mask]
        p_clean = p_star_vals[mask]

        color = colors.get(method, "#2E86AB")
        marker = markers.get(method, "o")

        if len(n_clean) > 0:
            # Plot actual data
            ax.scatter(
                n_clean,
                p_clean,
                marker=marker,
                s=200,
                color=color,
                edgecolor="black",
                linewidth=2,
                zorder=3,
                label=f"{label} (actual)",
            )

            # Plot theory line
            p_theory_line = np.array([theoretical_p_star(n, target_rank, method) for n in n_range])
            ax.plot(
                n_range,
                p_theory_line,
                linestyle="--",
                linewidth=2.5,
                alpha=0.6,
                color=color,
                label=f"{label} (theory)",
            )

            # Add ratio annotations for each point
            for n, p_actual in zip(n_clean, p_clean):
                p_theory = theoretical_p_star(int(n), target_rank, method)
                if not np.isnan(p_theory) and p_theory > 0:
                    ratio = p_actual / p_theory
                    ax.annotate(
                        f"{ratio:.1f}x",
                        xy=(n, p_actual),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=9,
                        color=color,
                        fontweight="bold",
                    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Taxa (n)", fontsize=14, fontweight="bold")
    ax.set_ylabel(f"Critical p* ({threshold_col.replace('p_star_', '')})",
                  fontsize=14, fontweight="bold")
    ax.set_title("Actual vs Theoretical Sample Complexity", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3, which="both")

    # Add text box explaining theory
    theory_text = (
        "Theory: $p^* \\geq C \\cdot r \\cdot \\log(n) / n$\n"
        f"Uniform: $C \\approx 4$ (conservative)\n"
        f"Leveraged: $C \\approx 2$ (2x better)\n"
        "\n"
        "Ratios show: actual / theory"
    )
    ax.text(
        0.02, 0.98,
        theory_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_path}")

    return fig
