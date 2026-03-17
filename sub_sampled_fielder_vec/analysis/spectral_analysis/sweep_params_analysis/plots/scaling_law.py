"""Plot scaling law validation: p_crit vs L and p_crit vs n."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path


def plot_scaling_law_E1(
    critical_points: List[Dict[str, Any]], output_path: Path
) -> None:
    """Plot p_crit vs L with curves per N value.

    Graph E1: Scaling Law - p_crit as function of L
    - X-axis: Sequence length L (log scale)
    - Y-axis: Critical sampling rate p_crit (log scale)
    - Data: One curve per N value
    - Theory: Three reference lines (p ∝ L^-0.5, L^-1, fitted exponent)

    Args:
        critical_points: List of dicts with keys {N, L, p_crit}
        output_path: Path to save PNG file
    """
    if not critical_points:
        print("  ⚠ No critical points found, skipping Graph E1")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Extract data by N
    N_vals = sorted(set(pt["N"] for pt in critical_points))

    # Use MATLAB-ish color palette
    matlab_colors = [
        (0.0000, 0.4470, 0.7410),  # Blue
        (0.8500, 0.3250, 0.0980),  # Orange
        (0.9290, 0.6940, 0.1250),  # Yellow
        (0.4940, 0.1840, 0.5560),  # Purple
        (0.4660, 0.6740, 0.1880),  # Green
        (0.3010, 0.7450, 0.9330),  # Cyan
        (0.6350, 0.0780, 0.1840),  # Red
    ]

    for i, N in enumerate(N_vals):
        N_points = [pt for pt in critical_points if pt["N"] == N]
        L_vals = np.array([pt["L"] for pt in N_points])
        p_crit_vals = np.array([pt["p_crit"] for pt in N_points])

        # Sort by L for clean lines
        sort_idx = np.argsort(L_vals)
        L_sorted = L_vals[sort_idx]
        p_sorted = p_crit_vals[sort_idx]

        ax.plot(
            L_sorted,
            p_sorted,
            "o-",
            color=matlab_colors[i % len(matlab_colors)],
            linewidth=2,
            markersize=8,
            label=f"N={N}",
        )

    # Theoretical reference lines - extend to cover full data range
    all_L_vals = [pt["L"] for pt in critical_points]
    L_range = np.array([min(all_L_vals), max(all_L_vals)])

    # Fit power law to all data (for reference)
    all_L = np.array([pt["L"] for pt in critical_points])
    all_p_crit = np.array([pt["p_crit"] for pt in critical_points])
    log_L = np.log(all_L)
    log_p = np.log(all_p_crit)
    fit_coef = np.polyfit(log_L, log_p, 1)
    fitted_exp = fit_coef[0]
    fitted_scale = np.exp(fit_coef[1])

    # Plot theory lines
    ax.plot(
        L_range,
        0.5 * L_range**(-0.5),
        "--",
        color="gray",
        linewidth=2,
        alpha=0.7,
        label=r"Theory: $p \propto L^{-0.5}$",
    )
    ax.plot(
        L_range,
        5.0 * L_range**(-1.0),
        "--",
        color="black",
        linewidth=2,
        alpha=0.7,
        label=r"Theory: $p \propto L^{-1}$",
    )
    ax.plot(
        L_range,
        fitted_scale * L_range**fitted_exp,
        ":",
        color="red",
        linewidth=2.5,
        alpha=0.8,
        label=f"Fitted: $p \\propto L^{{{fitted_exp:.2f}}}$",
    )

    ax.set_xlabel("Sequence Length L", fontsize=14)
    ax.set_ylabel("Critical Sampling Rate p_crit", fontsize=14)
    ax.set_title("Graph E1: Scaling Law — p_crit vs L", fontsize=16)
    ax.set_xscale("log")  # Log scale for proper power law visualization
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scaling_law_E2(
    critical_points: List[Dict[str, Any]], output_path: Path
) -> None:
    """Plot p_crit vs n (N) with curves per L value.

    Graph E2: Scaling Law - p_crit as function of n
    - X-axis: Number of leaves n (linear scale)
    - Y-axis: Critical sampling rate p_crit (log scale)
    - Data: One curve per L value
    - Theory: Reference lines with fitted exponent

    Args:
        critical_points: List of dicts with keys {N, L, p_crit}
        output_path: Path to save PNG file
    """
    if not critical_points:
        print("  ⚠ No critical points found, skipping Graph E2")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Extract data by L
    L_vals = sorted(set(pt["L"] for pt in critical_points))

    # Use MATLAB-ish color palette
    matlab_colors = [
        (0.0000, 0.4470, 0.7410),  # Blue
        (0.8500, 0.3250, 0.0980),  # Orange
        (0.9290, 0.6940, 0.1250),  # Yellow
        (0.4940, 0.1840, 0.5560),  # Purple
        (0.4660, 0.6740, 0.1880),  # Green
        (0.3010, 0.7450, 0.9330),  # Cyan
        (0.6350, 0.0780, 0.1840),  # Red
    ]

    for i, L in enumerate(L_vals):
        L_points = [pt for pt in critical_points if pt["L"] == L]
        N_vals = np.array([pt["N"] for pt in L_points])
        p_crit_vals = np.array([pt["p_crit"] for pt in L_points])

        # Sort by N for clean lines
        sort_idx = np.argsort(N_vals)
        N_sorted = N_vals[sort_idx]
        p_sorted = p_crit_vals[sort_idx]

        ax.plot(
            N_sorted,
            p_sorted,
            "o-",
            color=matlab_colors[i % len(matlab_colors)],
            linewidth=2,
            markersize=8,
            label=f"L={L}",
        )

    # Theoretical reference lines - extend to cover full data range
    all_N_vals = [pt["N"] for pt in critical_points]
    N_range = np.array([min(all_N_vals), max(all_N_vals)])

    # Fit power law to all data (for reference)
    all_N = np.array([pt["N"] for pt in critical_points])
    all_p_crit = np.array([pt["p_crit"] for pt in critical_points])
    log_N = np.log(all_N)
    log_p = np.log(all_p_crit)
    fit_coef = np.polyfit(log_N, log_p, 1)
    fitted_exp = fit_coef[0]
    fitted_scale = np.exp(fit_coef[1])

    # Plot theory lines
    ax.plot(
        N_range,
        0.5 * N_range**(0.5),
        "--",
        color="gray",
        linewidth=2,
        alpha=0.7,
        label=r"Theory: $p \propto n^{0.5}$",
    )
    ax.plot(
        N_range,
        0.1 * N_range**(1.0),
        "--",
        color="black",
        linewidth=2,
        alpha=0.7,
        label=r"Theory: $p \propto n^{1}$",
    )
    ax.plot(
        N_range,
        fitted_scale * N_range**fitted_exp,
        ":",
        color="red",
        linewidth=2.5,
        alpha=0.8,
        label=f"Fitted: $p \\propto n^{{{fitted_exp:.2f}}}$",
    )

    ax.set_xlabel("Number of Leaves n", fontsize=14)
    ax.set_ylabel("Critical Sampling Rate p_crit", fontsize=14)
    ax.set_title("Graph E2: Scaling Law — p_crit vs n", fontsize=16)
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# Keep backward compatibility alias
def plot_scaling_law(
    critical_points: List[Dict[str, Any]], output_path: Path
) -> None:
    """Legacy function - redirects to plot_scaling_law_E1."""
    plot_scaling_law_E1(critical_points, output_path)
