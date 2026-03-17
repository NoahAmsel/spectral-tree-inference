"""Plot phase boundary with discrete regime colormap."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from scipy.interpolate import griddata


def plot_phase_boundary(
    rows: List[Dict[str, Any]], output_path: Path
) -> None:
    """Create phase boundary plot with discrete 3-regime colormap.

    Graph A: The Phase Boundary (Enhanced)
    - X-axis: Sampling rate p (log scale)
    - Y-axis: Sequence length L (log scale)
    - Color: Discrete regimes based on partition_agreement_M
      * Gray: 45-55% (noise_bulk - random)
      * Yellow: 55-95% (spectral_emergence - transition)
      * Green: 95-100% (perturbation_plateau - success)
    - Contour: Bold line at 95% agreement (phase boundary)
    - Faceted by N value (columns)

    Args:
        rows: All result rows
        output_path: Path to save PNG file
    """
    # Get unique N values
    N_vals = sorted(set(row["num_taxa"] for row in rows))
    n_panels = len(N_vals)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    # Define discrete colormap for 3 regimes
    cmap = mcolors.ListedColormap(["gray", "gold", "limegreen"])
    boundaries = [0, 55, 95, 100]  # Regime boundaries
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    for idx, N in enumerate(N_vals):
        ax = axes[idx]
        N_rows = [r for r in rows if r["num_taxa"] == N]

        # Extract data
        p_vals = np.array([r["p"] for r in N_rows])
        L_vals = np.array([r["sequence_length"] for r in N_rows])
        agreement = np.array([r["partition_agreement_M"] for r in N_rows])

        # Create meshgrid for interpolation
        p_unique = sorted(set(p_vals))
        L_unique = sorted(set(L_vals))

        # Use log scale for gridding
        p_log = np.log10(p_vals)
        L_log = np.log10(L_vals)

        p_grid_log = np.linspace(min(p_log), max(p_log), 100)
        L_grid_log = np.linspace(min(L_log), max(L_log), 100)
        P_grid_log, L_grid_log_mesh = np.meshgrid(p_grid_log, L_grid_log)

        # Interpolate agreement values
        agreement_grid = griddata(
            (p_log, L_log),
            agreement,
            (P_grid_log, L_grid_log_mesh),
            method="nearest",
        )

        # Convert back to linear scale for plotting
        P_grid = 10**P_grid_log
        L_grid = 10**L_grid_log_mesh

        # Plot filled contours with discrete colormap
        contourf = ax.contourf(
            P_grid, L_grid, agreement_grid, levels=boundaries, cmap=cmap, norm=norm
        )

        # Overlay bold contour line at 95% (phase boundary)
        contour_line = ax.contour(
            P_grid,
            L_grid,
            agreement_grid,
            levels=[95],
            colors="black",
            linewidths=3,
            linestyles="--",
        )
        # Use lambda formatter to avoid format string issues
        ax.clabel(contour_line, inline=True, fontsize=10, fmt=lambda x: "95% boundary")

        ax.set_xlabel("Sampling Rate p", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Sequence Length L", fontsize=12)
        ax.set_title(f"N = {N}", fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")

    # Leave extra space for the shared colorbar/legend beneath the axes
    plt.tight_layout(rect=(0, 0.12, 1, 0.95))
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="horizontal",
    )
    cbar.set_label("Partition Agreement (%)", fontsize=12)
    cbar.set_ticks([27.5, 75, 97.5])
    cbar.ax.set_xticklabels(
        ["Noise\n(45-55%)", "Transition\n(55-95%)", "Success\n(95-100%)"]
    )

    plt.suptitle("Graph A: Phase Transition Boundary", fontsize=16, y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
