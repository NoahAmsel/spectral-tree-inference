"""Plot IPR vs sampling rate showing eigenvector delocalization."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


def plot_ipr_delocalization(
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]],
    output_path: Path,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot Inverse Participation Ratio vs p in 5×4 grid (N rows × L columns).

    Graph D: Structural Delocalization (Enhanced)
    - Grid: 5 rows (N values) × 4 columns (L values)
    - X-axis: Sampling rate p (synchronized with Graphs B & C)
    - Y-axis: mean_ipr_S
    - Baseline: Horizontal line at IPR = 1/N (perfect delocalization)

    Expected: High plateau (localized/noise) → sharp drop → low plateau (global tree).

    Args:
        grouped: Dictionary mapping (N, L) -> list of rows
        output_path: Path to save PNG file
        xlim: Optional x-axis limits to sync with other plots
    """
    # Extract unique N and L values
    N_vals = sorted(set(k[0] for k in grouped.keys()))
    L_vals = sorted(set(k[1] for k in grouped.keys()))

    n_rows = len(N_vals)
    n_cols = len(L_vals)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)

    for i, N in enumerate(N_vals):
        for j, L in enumerate(L_vals):
            ax = axes[i, j] if n_rows > 1 else axes[j]

            if (N, L) not in grouped:
                ax.axis('off')
                continue

            config_data = grouped[(N, L)]
            p_vals = [row["p"] for row in config_data]
            ipr_vals = [row["mean_ipr_S"] for row in config_data]

            # Plot IPR
            ax.plot(p_vals, ipr_vals, "o-", color="green", markersize=3, linewidth=1.5)

            # ENHANCEMENT: Perfect delocalization baseline (1/N)
            perfect_ipr = 1.0 / N
            ax.axhline(
                perfect_ipr,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                label=f"1/N={perfect_ipr:.4f}",
            )

            ax.set_xscale("log")
            ax.grid(alpha=0.2)

            # ENHANCEMENT: Sync x-axis if limits provided
            if xlim is not None:
                ax.set_xlim(xlim)

            # Labels
            if i == 0:
                ax.set_title(f"L={L}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"N={N}\nIPR", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("p", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best", framealpha=0.9)

    plt.suptitle("Graph D: Eigenvector Delocalization (IPR)", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
